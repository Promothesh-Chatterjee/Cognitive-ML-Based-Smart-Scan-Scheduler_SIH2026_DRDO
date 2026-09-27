"""FiLM-Gated Factorized DRQN Scheduler for Phase G8.

Extends the factorized architecture with band-conditioned Feature-wise Linear Modulation (FiLM):
  Q(s, b, m) = V(s) + A_band_tilde(s, b) + gamma(b) * A_mode_tilde(s, m) + beta(b)

Where:
  - V(s): Scalar value function from LSTM output.
  - A_band_tilde(s, b): Band-specific advantage (centered across bands).
  - A_mode_tilde(s, m): Mode-specific advantage (centered across modes).
  - gamma(b), beta(b): Per-band scale and shift computed from joint band feature + context:
      gamma(b) = 1.0 + tanh(gamma_raw(b)) * 0.5   -> bounded in [0.5, 1.5]
      beta(b)  = beta_raw(b)
  - Identity initialization ensures: gamma(b) == 1.0, beta(b) == 0.0 at step 0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    CANONICAL_N_ACTIONS,
    band_of_action,
    mode_of_action,
)

logger = logging.getLogger(__name__)


class FiLMGatedFactorizedDRQN(nn.Module):
    """FiLM-Gated Factorized DRQN Scheduler with identity initialization."""

    def __init__(
        self,
        obs_dim: int = CANONICAL_OBS_DIM,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: int = CANONICAL_N_MODES,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_bands = n_bands
        self.n_modes = n_modes
        self.n_actions = n_bands * n_modes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.band_features = obs_dim // n_bands

        # --- Shared Representation (Inherited from G7-A / Gate-25k) ---
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

        # --- Factorized Main Heads (Inherited from G7-A) ---
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

        # --- Phase G8 FiLM Gate Head ---
        # Input: joint_band (32 + 32 = 64) -> 32 -> 2 * n_modes (gamma_raw, beta)
        self.film_proj = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * n_modes),
        )

        # Identity initialization: zero-out the final layer so gamma_raw=0 (gamma=1) and beta=0
        nn.init.xavier_uniform_(self.film_proj[0].weight)
        nn.init.zeros_(self.film_proj[0].bias)
        nn.init.zeros_(self.film_proj[2].weight)
        nn.init.zeros_(self.film_proj[2].bias)

        # Auxiliary prediction heads (time-frequency decision system parity)
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

        self.last_decision_telemetry: dict[str, Any] = {}

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = torch.device(device) if isinstance(device, str) else device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        return (h0, c0)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            obs: (B, T, obs_dim) or (B, 1, obs_dim)
            hidden: Optional (h, c) tuple

        Returns:
            q_flat: Flat Q-values of shape (B, T, 180)
            aux: Dict containing "intercept_prob", "intercept_time_us", and decomposition diagnostics
            next_hidden: Updated LSTM hidden state
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

        # FiLM Conditioning: gamma(s, b, m) and beta(s, b, m)
        film_out = self.film_proj(joint_band)  # (B, T, 36, 10)
        gamma_raw = film_out[..., : self.n_modes]  # (B, T, 36, 5)
        beta = film_out[..., self.n_modes :]  # (B, T, 36, 5)
        gamma = 1.0 + torch.tanh(gamma_raw) * 0.5  # Bounded in [0.5, 1.5], centered at 1.0

        # Coupled mode advantage: gamma(b, m) * A_mode_tilde(m) + beta(b, m)
        modulated_mode = gamma * a_mode_tilde.unsqueeze(2) + beta  # (B, T, 36, 5)

        # Joint Q: Q(s, b, m) = V(s) + A_band_tilde(s, b) + modulated_mode(s, b, m)
        q_joint = v.unsqueeze(-1) + a_band_tilde.unsqueeze(-1) + modulated_mode  # (B, T, 36, 5)
        q_flat = q_joint.view(B, T, self.n_actions)  # (B, T, 180)

        # Auxiliary predictions
        intercept_prob = self.intercept_prob_head(lstm_out)  # (B, T, 180)
        intercept_time_us = self.intercept_time_head(lstm_out)  # (B, T, 180)

        aux = {
            "intercept_prob": intercept_prob,
            "intercept_time_us": intercept_time_us,
            "v": v,
            "a_band_tilde": a_band_tilde,
            "a_mode_tilde": a_mode_tilde,
            "film_gamma": gamma,
            "film_beta": beta,
            "q_joint": q_joint,
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
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-step action selection for rollouts and evaluation."""
        self.eval()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)

        q, _aux, h = self.forward(obs, hidden)
        raw_q_np = q[0, -1].detach().cpu().numpy().reshape(-1)
        raw_drqn_action = int(np.argmax(raw_q_np))
        raw_drqn_band = band_of_action(raw_drqn_action, self.n_modes)
        raw_drqn_mode = mode_of_action(raw_drqn_action, self.n_modes)

        action = raw_drqn_action
        final_action = int(action)
        final_band = band_of_action(final_action, self.n_modes)
        final_mode = mode_of_action(final_action, self.n_modes)

        q_selected = float(raw_q_np[final_action]) if 0 <= final_action < len(raw_q_np) else 0.0
        q_max = float(np.max(raw_q_np)) if len(raw_q_np) > 0 else 0.0
        q_mean = float(np.mean(raw_q_np)) if len(raw_q_np) > 0 else 0.0
        q_std = float(np.std(raw_q_np)) if len(raw_q_np) > 0 else 0.0

        self.last_decision_telemetry = {
            "raw_drqn_action": raw_drqn_action,
            "raw_drqn_band": raw_drqn_band,
            "raw_drqn_mode": raw_drqn_mode,
            "final_action": final_action,
            "final_band": final_band,
            "final_mode": final_mode,
            "action_was_overridden": False,
            "override_source": None,
            "exploration_source": "none",
            "decision_source": "ml_exploitation",
            "q_selected": q_selected,
            "q_max": q_max,
            "q_mean": q_mean,
            "q_std": q_std,
        }

        return action, h

    @classmethod
    def from_g7a_checkpoint(
        cls,
        ckpt_path: Path | str,
        seed: int = 42,
    ) -> Tuple[FiLMGatedFactorizedDRQN, Dict[str, Any]]:
        """Instantiate FiLMGatedFactorizedDRQN from G7-A checkpoint.

        Loads shared representation and factorized heads. Drops interaction_proj and
        interaction_mode from G7-B LowRankCoupled architecture with strict=False.
        Identity-initializes FiLM modulation layers.
        """
        torch.manual_seed(seed)
        model = cls()
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_sd = payload.get("state_dict", payload.get("online_drqn", payload))

        shared_prefixes = [
            "input_norm.",
            "lstm.",
            "value_stream.",
            "band_encoder.",
            "ctx_proj.",
            "band_head.",
            "mode_head.",
            "intercept_prob_head.",
            "intercept_time_head.",
        ]

        inherited: list[str] = []
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in shared_prefixes):
                if name not in src_sd:
                    raise KeyError(f"Expected shared parameter '{name}' missing from G7-A state dict!")
                with torch.no_grad():
                    param.copy_(src_sd[name])
                inherited.append(name)

        # Expected keys to be dropped from LowRankCoupledDRQNScheduler
        expected_drops = {"interaction_proj.weight", "interaction_mode.weight"}
        src_keys = set(src_sd.keys())
        inherited_set = set(inherited)
        unexpected = src_keys - inherited_set - expected_drops
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in G7-A state dict not handled by FiLM loader: {unexpected}"
            )

        actually_dropped = expected_drops & src_keys
        logger.info(
            "G7-A checkpoint loaded with strict=False. "
            "Intentionally dropped %d interaction keys (replaced by FiLM gate): %s",
            len(actually_dropped),
            sorted(actually_dropped),
        )

        # Identity-init FiLM modulation layers (gamma=1.0, beta=0.0)
        with torch.no_grad():
            nn.init.xavier_uniform_(model.film_proj[0].weight)
            nn.init.zeros_(model.film_proj[0].bias)
            model.film_proj[2].weight.zero_()
            model.film_proj[2].bias.zero_()

        logger.info("FiLM gate identity-initialized: gamma=1.0, beta=0.0")

        manifest = {
            "initialization_seed": seed,
            "inherited_params_count": len(inherited),
            "dropped_keys": sorted(actually_dropped),
            "newly_initialized_params_count": 4,  # film_proj[0].weight, bias, film_proj[2].weight, bias
            "inherited_params": inherited,
        }
        return model, manifest
