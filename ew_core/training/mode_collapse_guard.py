"""Phase G8 Mode Collapse Circuit Breaker.

Monitors mode selection distribution to detect early signs of mode collapse
(specifically SHORT runaway where SHORT dwell fraction > 88% or relative surge > 8%
coupled with mode entropy < 0.30).

Provides both:
1. Shadow-Greedy Evaluation: Evaluates network greedily on a fixed deterministic probe batch
   independent of online exploration epsilon or trajectory noise.
2. Online Rolling Evaluation: Monitors rolling executed actions (backward compatibility).

When triggered, alerts the training loop to save an emergency checkpoint and fail-closed.
"""

from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModeCollapseTriggered(RuntimeError):
    """Exception raised when mode collapse circuit breaker trips."""
    pass


class ModeCollapseGuard:
    """Circuit breaker monitoring greedy mode selection distribution.

    Dual condition architecture:
    1. Absolute collapse: short_frac > short_ceiling (default 0.88) AND entropy < min_entropy (default 0.30)
    2. Relative degradation: (short_frac - parent_short_frac > degradation_delta) AND entropy < min_entropy (default 0.30)
    """

    def __init__(
        self,
        window: int = 500,
        short_ceiling: float = 0.88,
        min_entropy: float = 0.30,
        parent_short_frac: float = 0.5315,
        degradation_delta: float = 0.08,
        warmup_steps: int = 200,
        min_samples: int = 200,
        checkpoint_dir: Path | str | None = None,
        n_modes: int = 5,
        probe_batch_path: Path | str | None = None,
    ) -> None:
        self.window = int(window)
        self.short_ceiling = float(short_ceiling)
        self.min_entropy = float(min_entropy)
        self.parent_short_frac = float(parent_short_frac)
        self.degradation_delta = float(degradation_delta)
        self.warmup_steps = int(warmup_steps)
        self.min_samples = int(max(min_samples, warmup_steps))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.n_modes = int(n_modes)
        self.history: deque[int] = deque(maxlen=self.window)
        self.total_seen: int = 0

        self.probe_obs: Optional[torch.Tensor] = None
        self.probe_burn_in: int = 8
        if probe_batch_path is not None:
            self.load_probe_batch(probe_batch_path)

    def load_probe_batch(self, probe_batch_path: Path | str) -> None:
        """Load fixed deterministic probe batch from disk."""
        p = Path(probe_batch_path)
        if not p.exists():
            logger.warning("Probe batch file does not exist: %s", p)
            return
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            self.probe_obs = payload["obs"]
            self.probe_burn_in = payload.get("burn_in", 8)
        else:
            self.probe_obs = payload
            self.probe_burn_in = 8
        logger.info(
            "Loaded shadow probe batch from %s (shape %s, burn_in=%d)",
            p,
            list(self.probe_obs.shape),
            self.probe_burn_in,
        )

    def check_shadow_greedy(
        self,
        model: torch.nn.Module,
        global_step: int,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Evaluate policy network greedily on fixed deterministic probe batch."""
        if self.probe_obs is None:
            return {
                "triggered": False,
                "short_frac": 0.0,
                "entropy": 1.609,
                "reason": "no_probe_batch",
            }

        was_training = model.training
        model.eval()
        dev = torch.device(device) if isinstance(device, str) else device
        obs_dev = self.probe_obs.to(dev)

        with torch.no_grad():
            q_flat, _, _ = model(obs_dev)
            active_q = q_flat[:, self.probe_burn_in:, :]
            acts = torch.argmax(active_q, dim=-1).flatten().cpu().numpy()
            modes = acts % self.n_modes

        if was_training:
            model.train()

        counts = np.bincount(modes, minlength=self.n_modes)
        n_samples = len(modes)
        probs = counts.astype(np.float64) / max(1, n_samples)
        short_frac = float(probs[0])

        nonzero_p = probs[probs > 1e-12]
        entropy_val = float(-np.sum(nonzero_p * np.log(nonzero_p))) if len(nonzero_p) > 0 else 0.0

        is_absolute_collapse = (short_frac > self.short_ceiling) and (entropy_val < self.min_entropy)
        is_relative_degradation = ((short_frac - self.parent_short_frac) > self.degradation_delta) and (entropy_val < self.min_entropy)

        triggered = bool(is_absolute_collapse or is_relative_degradation)
        reason = (
            "shadow_absolute_collapse"
            if is_absolute_collapse
            else ("shadow_relative_degradation" if is_relative_degradation else "normal")
        )

        if triggered:
            logger.error(
                "SHADOW-GREEDY ModeCollapseGuard TRIGGERED (%s) at step %d! "
                "Probe SHORT=%.1f%% (parent=%.1f%%, ceiling=%.1f%%), Mode Entropy=%.3f (floor=%.3f)",
                reason,
                global_step,
                short_frac * 100.0,
                self.parent_short_frac * 100.0,
                self.short_ceiling * 100.0,
                entropy_val,
                self.min_entropy,
            )

        return {
            "triggered": triggered,
            "short_frac": short_frac,
            "entropy": entropy_val,
            "mode_counts": counts.tolist(),
            "reason": reason,
            "probe_samples": n_samples,
        }

    def update(self, mode_chosen: int) -> Dict[str, Any]:
        """Record an online mode choice and check circuit breaker conditions.

        Args:
            mode_chosen: Selected dwell mode index (0..4). 0 corresponds to SHORT_DWELL.

        Returns:
            Dict containing triggered, short_frac, entropy, window_size, mode_counts, reason.
        """
        self.history.append(int(mode_chosen))
        self.total_seen += 1
        n_samples = len(self.history)

        if self.total_seen < self.warmup_steps or n_samples < self.min_samples:
            return {
                "triggered": False,
                "short_frac": 0.0,
                "entropy": 1.609,  # ln(5) max entropy
                "window_size": n_samples,
                "mode_counts": [0] * self.n_modes,
                "reason": "warmup",
            }

        counts = np.zeros(self.n_modes, dtype=np.int64)
        for m in self.history:
            if 0 <= m < self.n_modes:
                counts[m] += 1

        probs = counts.astype(np.float64) / n_samples
        short_frac = float(probs[0])

        nonzero_p = probs[probs > 1e-12]
        entropy_val = float(-np.sum(nonzero_p * np.log(nonzero_p))) if len(nonzero_p) > 0 else 0.0

        is_absolute_collapse = (short_frac > self.short_ceiling) and (entropy_val < self.min_entropy)
        is_relative_degradation = (short_frac - self.parent_short_frac > self.degradation_delta) and (entropy_val < self.min_entropy)

        triggered = bool(is_absolute_collapse or is_relative_degradation)
        reason = "absolute_collapse" if is_absolute_collapse else ("relative_degradation" if is_relative_degradation else "normal")

        if triggered:
            logger.error(
                "ModeCollapseGuard TRIGGERED (%s)! Window=%d, SHORT=%.1f%% (parent=%.1f%%, ceiling=%.1f%%), "
                "Mode Entropy=%.3f (floor=%.3f)",
                reason,
                n_samples,
                short_frac * 100.0,
                self.parent_short_frac * 100.0,
                self.short_ceiling * 100.0,
                entropy_val,
                self.min_entropy,
            )

        return {
            "triggered": triggered,
            "short_frac": short_frac,
            "entropy": entropy_val,
            "window_size": n_samples,
            "mode_counts": counts.tolist(),
            "reason": reason,
        }

    def reset(self) -> None:
        self.history.clear()
