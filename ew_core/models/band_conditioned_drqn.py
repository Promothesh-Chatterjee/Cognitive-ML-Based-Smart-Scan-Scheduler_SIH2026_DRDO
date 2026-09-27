"""Band-Conditioned Factorized DRQN Architecture for Phase G8.3-B.

Replaces the globally shared mode head and sluggish FiLM modulation with an
explicit band-conditioned dwell branch:
    Q(s, b, m) = V(s) + A_band_tilde(s, b) + D_tilde(s, b, m)

where:
    D(s, b, .) = f_D([h_s, e_b]) in R^5
    D_tilde(s, b, m) = D(s, b, m) - 1/5 sum_k D(s, b, k)

Zero-Step Equivalence Contract:
Initializes f_D such that at Step 0:
    D(s, b, m) == A_mode_parent(s, m)  for all b in 0..35
guaranteeing 0.00% zero-step action flips against the Gate-26k parent while
making band-conditioned dwell selection fully learnable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BandConditionedFactorizedDRQN(nn.Module):
    """Band-Conditioned Factorized Deep Recurrent Q-Network (Phase G8.3-B)."""

    def __init__(
        self,
        obs_dim: int = 360,
        n_bands: int = 36,
        n_modes: int = 5,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        band_features: int = 10,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_bands = n_bands
        self.n_modes = n_modes
        self.n_actions = n_bands * n_modes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.band_features = band_features

        # Observation normalization
        self.input_norm = nn.LayerNorm(obs_dim)

        # Recurrent Core
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Per-Band Representation: e_b in R^32
        self.band_encoder = nn.Sequential(
            nn.Linear(self.band_features, 32),
            nn.ReLU(),
        )

        # Recurrent State Context Projection for Band Stream: c_s in R^32
        self.ctx_proj = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
        )

        # Band Advantage Head: maps [e_b (32) + c_s (32)] -> 1 scalar advantage per band
        self.band_head = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # --- Phase G8.3-B Explicit Band-Conditioned Dwell Branch ---
        # Input: [h_s (256) + e_b (32)] = 288 -> 64 -> 5 dwell advantages per band
        self.dwell_head = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 64),
            nn.ReLU(),
            nn.Linear(64, n_modes),
        )

        # Auxiliary prediction heads
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

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = torch.device(device) if isinstance(device, str) else device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        return (h0, c0)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ablate_dwell_branch: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            obs: (B, T, obs_dim) or (B, 1, obs_dim)
            hidden: Optional (h, c) tuple
            ablate_dwell_branch: If True, zeroes out D_tilde to evaluate dwell branch activity.

        Returns:
            q_flat: Flat Q-values of shape (B, T, 180)
            aux: Dict containing decomposition components and auxiliary outputs
            next_hidden: Updated LSTM hidden state
        """
        B, T, _ = obs.shape
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)

        x = self.input_norm(obs)
        lstm_out, next_hidden = self.lstm(x, hidden)

        # 1. State Value: V(s) shape (B, T, 1)
        v = self.value_stream(lstm_out)

        # 2. Band Advantage Stream: A_b(s, b)
        obs_bands = obs.view(B, T, self.n_bands, self.band_features)
        band_emb = self.band_encoder(obs_bands)  # (B, T, 36, 32)
        ctx = self.ctx_proj(lstm_out).unsqueeze(2).expand(-1, -1, self.n_bands, -1)  # (B, T, 36, 32)

        joint_band = torch.cat([band_emb, ctx], dim=-1)  # (B, T, 36, 64)
        a_band_raw = self.band_head(joint_band).squeeze(-1)  # (B, T, 36)
        a_band_tilde = a_band_raw - a_band_raw.mean(dim=-1, keepdim=True)  # (B, T, 36)

        # 3. Band-Conditioned Dwell Branch: D(s, b, m)
        lstm_expanded = lstm_out.unsqueeze(2).expand(-1, -1, self.n_bands, -1)  # (B, T, 36, 256)
        dwell_in = torch.cat([lstm_expanded, band_emb], dim=-1)  # (B, T, 36, 288)

        if ablate_dwell_branch:
            d_raw = torch.zeros(B, T, self.n_bands, self.n_modes, device=obs.device)
            d_tilde = torch.zeros_like(d_raw)
        else:
            d_raw = self.dwell_head(dwell_in)  # (B, T, 36, 5)
            d_tilde = d_raw - d_raw.mean(dim=-1, keepdim=True)  # (B, T, 36, 5) centered per band

        # 4. Joint Q: Q(s, b, m) = V(s) + A_band_tilde(s, b) + D_tilde(s, b, m)
        q_joint = v.unsqueeze(-1) + a_band_tilde.unsqueeze(-1) + d_tilde  # (B, T, 36, 5)
        q_flat = q_joint.view(B, T, self.n_actions)  # (B, T, 180)

        # Auxiliary predictions
        intercept_prob = self.intercept_prob_head(lstm_out)
        intercept_time_us = self.intercept_time_head(lstm_out)

        aux = {
            "v": v,
            "a_band_tilde": a_band_tilde,
            "d_raw": d_raw,
            "d_tilde": d_tilde,
            "q_joint": q_joint,
            "intercept_prob": intercept_prob,
            "intercept_time_us": intercept_time_us,
        }
        return q_flat, aux, next_hidden

    @torch.inference_mode()
    def act(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mode_selection: str = "flat_argmax",
        consecutive_empty: int = 0,
        tau: float = 0.0,
        ablate_dwell_branch: bool = False,
        **kwargs: Any,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-step greedy action selection for training and evaluation."""
        self.eval()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)

        q, _aux, h = self.forward(obs, hidden, ablate_dwell_branch=ablate_dwell_branch)
        raw_q = q[0, -1].cpu().numpy().reshape(-1)
        action = int(np.argmax(raw_q))
        return action, h

    @classmethod
    def from_parent_checkpoint(
        cls,
        ckpt_path: Path | str,
        seed: int = 42,
    ) -> Tuple[BandConditionedFactorizedDRQN, Dict[str, Any]]:
        """Instantiate BandConditionedFactorizedDRQN with exact zero-step equivalence to Gate-26k parent.

        Manifest Rules:
        - INHERITED-EXACT: input_norm, lstm, value_stream, band_encoder, ctx_proj, band_head, aux heads.
        - INHERITED-RESHAPED: dwell_head.0.weight (first 256 cols from parent mode_head.0, last 32 zeroed).
        - INHERITED-EXACT: dwell_head.0.bias, dwell_head.2.weight, dwell_head.2.bias from mode_head.
        - INTENTIONALLY-REPLACED: mode_head.
        - INTENTIONALLY-DROPPED: film_proj.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        ckpt_path = Path(ckpt_path).resolve()
        assert ckpt_path.exists(), f"Parent checkpoint not found: {ckpt_path}"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload.get("state_dict", payload.get("online_drqn", payload))

        model = cls(
            obs_dim=360,
            n_bands=36,
            n_modes=5,
            lstm_hidden=256,
            lstm_layers=2,
            band_features=10,
        )

        manifest = {
            "inherited_exact": [],
            "inherited_reshaped": [],
            "intentionally_replaced": [],
            "intentionally_dropped": [],
        }

        # 1. Exact transfers
        exact_keys = [
            "input_norm.weight", "input_norm.bias",
            "lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
            "lstm.weight_ih_l1", "lstm.weight_hh_l1", "lstm.bias_ih_l1", "lstm.bias_hh_l1",
            "value_stream.0.weight", "value_stream.0.bias", "value_stream.2.weight", "value_stream.2.bias",
            "band_encoder.0.weight", "band_encoder.0.bias",
            "ctx_proj.0.weight", "ctx_proj.0.bias",
            "band_head.0.weight", "band_head.0.bias", "band_head.2.weight", "band_head.2.bias",
            "intercept_prob_head.0.weight", "intercept_prob_head.0.bias", "intercept_prob_head.2.weight", "intercept_prob_head.2.bias",
            "intercept_time_head.0.weight", "intercept_time_head.0.bias", "intercept_time_head.2.weight", "intercept_time_head.2.bias",
        ]

        model_sd = model.state_dict()
        for k in exact_keys:
            if k in sd:
                model_sd[k].copy_(sd[k])
                manifest["inherited_exact"].append(k)
            else:
                logger.warning("Key %s not found in parent checkpoint!", k)

        # 2. Reshaped transfer for dwell_head (zero-step equivalence)
        # parent mode_head[0].weight: (64, 256) -> dwell_head[0].weight: (64, 288)
        if "mode_head.0.weight" in sd:
            parent_w0 = sd["mode_head.0.weight"]
            new_w0 = torch.zeros(64, 288, dtype=parent_w0.dtype)
            new_w0[:, :256].copy_(parent_w0)
            new_w0[:, 256:].zero_()  # Band connections zero-initialized
            model_sd["dwell_head.0.weight"].copy_(new_w0)
            manifest["inherited_reshaped"].append({
                "source": "mode_head.0.weight (64, 256)",
                "target": "dwell_head.0.weight (64, 288)",
                "rule": "Cols 0..255 from parent mode_head.0; Cols 256..287 initialized to 0.0",
            })

        if "mode_head.0.bias" in sd:
            model_sd["dwell_head.0.bias"].copy_(sd["mode_head.0.bias"])
            manifest["inherited_exact"].append("dwell_head.0.bias")

        if "mode_head.2.weight" in sd:
            model_sd["dwell_head.2.weight"].copy_(sd["mode_head.2.weight"])
            manifest["inherited_exact"].append("dwell_head.2.weight")

        if "mode_head.2.bias" in sd:
            model_sd["dwell_head.2.bias"].copy_(sd["mode_head.2.bias"])
            manifest["inherited_exact"].append("dwell_head.2.bias")

        # 3. Track replaced and dropped keys
        for k in sd.keys():
            if k.startswith("mode_head."):
                manifest["intentionally_replaced"].append(k)
            elif k.startswith("film_proj.") or k.startswith("film_"):
                manifest["intentionally_dropped"].append(k)

        model.load_state_dict(model_sd)
        model.eval()

        logger.info(
            "BandConditionedFactorizedDRQN instantiated from parent %s: %d exact, 1 reshaped, %d replaced, %d dropped",
            ckpt_path.name,
            len(manifest["inherited_exact"]),
            len(manifest["intentionally_replaced"]),
            len(manifest["intentionally_dropped"]),
        )
        return model, manifest
