"""Low-Rank Coupled DRQN Scheduler for Phase G7-B.

Extends the Decoupled Factorized DRQN with a rank-R band-conditioned interaction term:
  Q(s, b, m) = V(s) + A_b_tilde(s, b) + A_m_tilde(s, m) + I_tilde(s, b, m)

where:
  I_raw(s, b, m) = phi(joint_band(s, b))^T W_m
  phi: R^64 -> R^rank
  W_m: R^rank -> R^5

Crucial Properties:
1. Zero-Step Parity: W_m is initialized to strictly zero (0).
   At step 0, I_tilde(s, b, m) == 0 identically, ensuring exact behavioral equivalence
   with the additive factorized Gate-25k model.
2. Low-Rank Efficiency: Rank R = 8 introduces only (64*8 + 8*5 = 552) parameters.
3. Pure Interaction Centering: I_tilde is doubly-centered across both bands and modes:
   mean_b(I_tilde) == 0 and mean_m(I_tilde) == 0.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("low_rank_coupled_drqn_scheduler")

CANONICAL_OBS_DIM = 360
CANONICAL_N_BANDS = 36
CANONICAL_N_MODES = 5
CANONICAL_N_ACTIONS = 180


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA-256 hash of tensor data bytes."""
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class LowRankCoupledDRQNScheduler(nn.Module):
    """Low-Rank Band-Conditioned Coupled DRQN Scheduler."""

    def __init__(
        self,
        obs_dim: int = CANONICAL_OBS_DIM,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: int = CANONICAL_N_MODES,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_bands = n_bands
        self.n_modes = n_modes
        self.n_actions = n_bands * n_modes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.band_features = obs_dim // n_bands
        self.rank = rank

        # --- Shared Representation (Inherited from Gate-25k) ---
        self.input_norm = nn.LayerNorm(obs_dim)
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.band_encoder = nn.Sequential(
            nn.Linear(self.band_features, 32),
            nn.ReLU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
        )

        # --- Factorized Main Heads ---
        # 1. Band Advantage Head: maps [band_emb_b (32) + ctx (32)] -> 1 scalar advantage per band
        self.band_head = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 2. Mode Advantage Head: maps lstm_out (256) -> 5 dwell mode advantages
        self.mode_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_modes),
        )

        # --- G7-B Low-Rank Interaction Head: maps joint_band (64) -> rank (8) -> modes (5) ---
        self.interaction_proj = nn.Linear(64, rank, bias=False)
        self.interaction_mode = nn.Linear(rank, n_modes, bias=False)

        # Zero-initialize the interaction projection so I(s, b, m) == 0 at step 0
        nn.init.zeros_(self.interaction_mode.weight)
        nn.init.xavier_uniform_(self.interaction_proj.weight)

        # Auxiliary heads for telemetry parity
        self.intercept_prob_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Sigmoid(),
        )
        self.intercept_time_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Softplus(),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return h0, c0

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Returns:
            q_flat: Flat Q-values of shape (B, T, 180) where action a = b*5 + m.
            aux_data: Dict containing decomposition components.
            hidden: Updated LSTM hidden state.
        """
        B, T, _ = obs.shape
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)
        x = self.input_norm(obs)
        lstm_out, next_hidden = self.lstm(x, hidden)

        # Value stream: V(s) shape (B, T, 1)
        v = self.value_stream(lstm_out)

        # Band stream: compute per-band advantage A_b(s, b)
        obs_bands = obs.view(B, T, self.n_bands, self.band_features)
        band_emb = self.band_encoder(obs_bands)  # (B, T, 36, 32)
        ctx = self.ctx_proj(lstm_out).unsqueeze(2).expand(-1, -1, self.n_bands, -1)  # (B, T, 36, 32)
        joint_band = torch.cat([band_emb, ctx], dim=-1)  # (B, T, 36, 64)
        a_band_raw = self.band_head(joint_band).squeeze(-1)  # (B, T, 36)
        a_band_tilde = a_band_raw - a_band_raw.mean(dim=-1, keepdim=True)  # (B, T, 36)

        # Mode stream: compute mode advantage A_m(s, m)
        a_mode_raw = self.mode_head(lstm_out)  # (B, T, 5)
        a_mode_tilde = a_mode_raw - a_mode_raw.mean(dim=-1, keepdim=True)  # (B, T, 5)

        # Low-Rank Interaction stream: I(s, b, m)
        # joint_band: (B, T, 36, 64) -> interaction_proj -> (B, T, 36, rank)
        band_latent = self.interaction_proj(joint_band)
        # interaction_mode -> (B, T, 36, 5)
        i_raw = self.interaction_mode(band_latent)
        # Doubly center across both bands (-2) and modes (-1) to isolate pure interaction:
        mean_b = i_raw.mean(dim=-2, keepdim=True)
        mean_m = i_raw.mean(dim=-1, keepdim=True)
        mean_bm = i_raw.mean(dim=(-2, -1), keepdim=True)
        i_tilde = i_raw - mean_b - mean_m + mean_bm  # (B, T, 36, 5)

        # Coupled joint Q: Q(s, b, m) = V(s) + A_b(s, b) + A_m(s, m) + I(s, b, m)
        q_joint = v.unsqueeze(-1) + a_band_tilde.unsqueeze(-1) + a_mode_tilde.unsqueeze(-2) + i_tilde
        q_flat = q_joint.view(B, T, self.n_actions)  # (B, T, 180)

        aux_data = {
            "v": v,
            "a_band_raw": a_band_raw,
            "a_band_tilde": a_band_tilde,
            "a_mode_raw": a_mode_raw,
            "a_mode_tilde": a_mode_tilde,
            "i_raw": i_raw,
            "i_tilde": i_tilde,
            "q_joint": q_joint,
        }
        return q_flat, aux_data, next_hidden

    @classmethod
    def from_gate25_checkpoint(
        cls,
        ckpt_path: Path,
        rank: int = 8,
        seed: int = 42,
    ) -> Tuple[LowRankCoupledDRQNScheduler, Dict[str, Any]]:
        """Instantiate LowRankCoupledDRQNScheduler inheriting Gate-25k shared representation

        and zero-initializing the interaction head under the G7-B contract.
        """
        torch.manual_seed(seed)
        model = cls(rank=rank)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_sd = payload.get("state_dict", payload.get("online_drqn", payload))

        inherited_params = []
        newly_initialized_params = []
        mapping_records = []

        # 1. Inherit shared representation directly
        shared_prefixes = ["input_norm.", "lstm.", "value_stream.", "band_encoder.", "ctx_proj."]
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in shared_prefixes):
                if name in src_sd:
                    with torch.no_grad():
                        param.copy_(src_sd[name])
                    inherited_params.append(name)
                    mapping_records.append({
                        "dest_name": name,
                        "src_name": name,
                        "shape": list(param.shape),
                        "transformation": "direct_copy",
                        "dest_sha256": compute_tensor_sha256(param),
                    })
                else:
                    raise KeyError(f"Expected shared parameter {name} missing in Gate-25 state dict!")

        # 2. Inherit auxiliary prediction heads
        aux_prefixes = ["intercept_prob_head.", "intercept_time_head."]
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in aux_prefixes):
                if name in src_sd:
                    with torch.no_grad():
                        param.copy_(src_sd[name])
                    inherited_params.append(name)
                    mapping_records.append({
                        "dest_name": name,
                        "src_name": name,
                        "shape": list(param.shape),
                        "transformation": "direct_copy",
                        "dest_sha256": compute_tensor_sha256(param),
                    })

        # 3. Factorized band head (G5 tensor mapping contract)
        with torch.no_grad():
            model.band_head[0].weight.copy_(src_sd["band_advantage_head.0.weight"])
            model.band_head[0].bias.copy_(src_sd["band_advantage_head.0.bias"])
            mean_w = src_sd["band_advantage_head.2.weight"].mean(dim=0, keepdim=True)
            mean_b = src_sd["band_advantage_head.2.bias"].mean(dim=0, keepdim=True)
            model.band_head[2].weight.copy_(mean_w)
            model.band_head[2].bias.copy_(mean_b)

        inherited_params.extend([
            "band_head.0.weight", "band_head.0.bias",
            "band_head.2.weight", "band_head.2.bias",
        ])

        # 4. Mode head initialization (G5/G7-A contract: small gain Xavier)
        # Seed immediately before mode head to guarantee bitwise identical draw with G7-A
        torch.manual_seed(seed)
        with torch.no_grad():
            nn.init.xavier_uniform_(model.mode_head[0].weight)
            nn.init.zeros_(model.mode_head[0].bias)
            nn.init.xavier_uniform_(model.mode_head[2].weight, gain=0.1)
            nn.init.zeros_(model.mode_head[2].bias)

        newly_initialized_params.extend([
            "mode_head.0.weight", "mode_head.0.bias",
            "mode_head.2.weight", "mode_head.2.bias",
        ])

        # 5. G7-B Low-rank interaction head initialization:
        # interaction_mode is zero-initialized to guarantee ZERO-STEP PARITY with G7-A!
        torch.manual_seed(seed + 100)
        with torch.no_grad():
            nn.init.xavier_uniform_(model.interaction_proj.weight)
            nn.init.zeros_(model.interaction_mode.weight)

        newly_initialized_params.extend([
            "interaction_proj.weight",
            "interaction_mode.weight",
        ])

        manifest = {
            "initialization_seed": seed,
            "rank": rank,
            "inherited_params_count": len(inherited_params),
            "newly_initialized_params_count": len(newly_initialized_params),
            "total_params_count": len(list(model.parameters())),
            "inherited_params": inherited_params,
            "newly_initialized_params": newly_initialized_params,
            "mapping_records": mapping_records,
        }
        return model, manifest
