"""Phase G8 Relative Dwell Reward Shaper and Cosine Beta Entropy Schedule.

Contains:
1. RelativeDwellShaper: Maintains an EMA of dwell penalties to center dwell adjustments,
   preventing static SHORT bias (+4.5 advantage under fixed G3-D) while maintaining
   gradient sensitivity to relative timing efficiency.
2. cosine_beta_schedule: Cosine-annealed entropy regularization weight schedule with linear warmup.
"""

from __future__ import annotations

import math
import numpy as np


class RelativeDwellShaper:
    """Relative dwell cost reward shaper using an exponential moving average (EMA).

    Under standard G3-D, eff_rew = rew - c_dwell * (tau - 1.0).
    With tau_SHORT = 0.25 and tau_LONG = 2.5:
      delta_r(SHORT) = -2.0 * (0.25 - 1.0) = +1.5
      delta_r(LONG)  = -2.0 * (2.50 - 1.0) = -3.0
    This creates an absolute +4.5 advantage for SHORT regardless of spectrum state.

    RelativeDwellShaper centers the dwell cost around its running EMA:
      cost_raw = c_dwell * (tau - 1.0)
      ema_cost = (1 - alpha) * ema_cost + alpha * cost_raw
      relative_cost = cost_raw - ema_cost
      r_shaped = r_raw - relative_cost
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        c_dwell: float = 2.0,
        base_dwell_us: float = 500.0,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.c_dwell = float(c_dwell)
        self.base_dwell_us = float(base_dwell_us)
        self.ema_dwell_cost: float = 0.0
        self._initialized: bool = False

    def shape(self, raw_reward: float, mode_chosen: int, dwell_us: float) -> float:
        """Apply relative dwell shaping to raw transition reward.

        Args:
            raw_reward: Environment reward.
            mode_chosen: Selected dwell mode index (0..4).
            dwell_us: Actual receiver dwell duration in microseconds.

        Returns:
            Shaped reward with centered dwell cost.
        """
        tau = max(0.1, float(dwell_us) / self.base_dwell_us)
        cost_raw = self.c_dwell * (tau - 1.0)

        if not self._initialized:
            self.ema_dwell_cost = cost_raw
            self._initialized = True
        else:
            self.ema_dwell_cost = (
                (1.0 - self.ema_alpha) * self.ema_dwell_cost + self.ema_alpha * cost_raw
            )

        relative_cost = cost_raw - self.ema_dwell_cost
        shaped_reward = float(raw_reward) - relative_cost
        return shaped_reward

    def reset(self) -> None:
        self.ema_dwell_cost = 0.0
        self._initialized = False


def cosine_beta_schedule(
    step: int,
    total_steps: int,
    start_step: int = 26000,
    beta_max: float = 0.10,
    warmup_frac: float = 0.10,
    beta_min: float = 0.01,
) -> float:
    """Calculate cosine-annealed entropy bonus coefficient beta(t).

    Linearly warms up from beta_min to beta_max over the first warmup_frac of the run,
    then cosine decays back down to beta_min.

    Args:
        step: Current global training step.
        total_steps: Target stop step.
        start_step: Resumed or initial step of this phase (default 26000).
        beta_max: Peak entropy regularization weight (default 0.10).
        warmup_frac: Fraction of steps dedicated to linear warmup (default 0.10).
        beta_min: Minimum baseline entropy weight (default 0.01).

    Returns:
        Effective beta value as float.
    """
    if step < start_step:
        rel_step = max(0, step)
        span = max(1, total_steps)
    else:
        rel_step = max(0, step - start_step)
        span = max(1, total_steps - start_step)

    progress = min(1.0, max(0.0, rel_step / span))

    if progress < warmup_frac:
        warmup_p = progress / max(1e-6, warmup_frac)
        return float(beta_min + (beta_max - beta_min) * warmup_p)
    else:
        decay_p = (progress - warmup_frac) / max(1e-6, 1.0 - warmup_frac)
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * decay_p))
        return float(beta_min + (beta_max - beta_min) * cosine_val)
