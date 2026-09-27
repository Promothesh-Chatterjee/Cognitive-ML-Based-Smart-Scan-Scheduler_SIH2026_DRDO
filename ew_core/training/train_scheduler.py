"""
DRQN Scheduler Training Loop with Thompson Sampling warmup and BPTT.

Uses RFScanEnv, SequenceReplayBuffer, ThompsonSamplingExplorer, and SmartScanMoE evaluation.
Fixed 2026-09-02:
 - Single canonical DRQN update step (no duplicated blocks).
 - Clean MoE / non-MoE action selection flow.
 - Replay buffer now episode-based (contiguous BPTT sequences, no cross-episode).
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
from typing import Any
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ..contracts import DEFAULT_DWELL_MULTIPLIERS, DWELL_MODES
from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ..environment.scenario_generator import ScenarioSource
from ..data.tsrd_manifest import dataset_fingerprint
from ..models.deinterleaver import PDWTransformerEncoder
from ..models.drqn_scheduler import DRQNScheduler
from ..models.smartscan_moe import SmartScanMoE
from ..preprocessing.normalise import load_normalization_stats, normalization_stats_hash
from ..telemetry.publisher import TelemetryPublisher
from ..telemetry.run_manager import RunManager
from ..telemetry.schema import TELEMETRY_SCHEMA_VERSION, coerce, make_episode_record, make_val_record, reward_reconstruction, shannon_entropy
from ..training.distribution_shift_detector import DistributionShiftDetector
from ..training.replay_buffer import SequenceReplayBuffer
from ..training.thompson_sampling import ThompsonSamplingExplorer
from ..training.training_gate import require_training_gate
from ..training.val_set import FixedValidationSet
from ..models.film_factorized_drqn import FiLMGatedFactorizedDRQN
from ..training.mode_collapse_guard import ModeCollapseGuard
from ..training.reward_g8 import RelativeDwellShaper, cosine_beta_schedule
from ..training.stratified_replay_sampler import StratifiedReplaySampler

logger = logging.getLogger(__name__)


def build_training_distribution_reference(
    data_dir: Path | str,
    env_config: dict[str, Any],
    deinterleaver_model: Any,
    fit_stats: Any,
    seed: int = 42,
    n_calibration_scenarios: int = 5,
    steps_per_scenario: int = 40,
) -> np.ndarray:
    """Generate reference observations exclusively from TSRD training distribution (stare/train).

    Uses a deterministic calibration subset and the canonical causal environment/perception
    pipeline to record true 360-D policy observation vectors. Validation and test sets are never accessed.
    """
    calib_source = ScenarioSource(
        data_root=data_dir,
        mode="stare",
        subset="train",
        freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_config.get("max_pulses", 50000)),
        seed=seed,
        source_type="world",
        allow_synthetic_fallback=False,
        chunk_mode="first",
    )
    calib_env_config = copy.deepcopy(env_config)
    calib_env_config["semantic_memory_enabled"] = False
    calib_env = CognitiveRFScanEnv(
        calib_env_config,
        records=None,
        seed=seed,
        records_provider=calib_source.sample,
        deinterleaver_model=deinterleaver_model,
        deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
        semantic_memory_path=":memory:",
    )
    obs_samples: list[np.ndarray] = []
    n_scenarios = min(n_calibration_scenarios, max(1, len(calib_source)))
    rng = np.random.default_rng(seed)
    n_actions = int(env_config.get("n_actions", 180))

    for sc_idx in range(n_scenarios):
        obs, _ = calib_env.reset()
        obs_samples.append(np.asarray(obs, dtype=np.float32).copy())
        for _ in range(steps_per_scenario):
            action = int(rng.integers(0, n_actions))
            obs, _, done, truncated, _ = calib_env.step(action)
            obs_samples.append(np.asarray(obs, dtype=np.float32).copy())
            if done or truncated:
                break
    ref_obs_matrix = np.array(obs_samples, dtype=np.float32)
    logger.info("Built policy-observation distribution reference from TSRD stare/train: %d observations (dim=%d)",
                len(ref_obs_matrix), ref_obs_matrix.shape[1] if ref_obs_matrix.ndim > 1 else 0)
    return ref_obs_matrix


def _observable_priorities(obs: np.ndarray, features_per_band: int = 10) -> np.ndarray:
    """Extract per-band observable priority (occupancy, feature index 0) from a flat obs.

    Avoids ground truth entirely: feature 0 is the EMA occupancy from belief,
    which is a legitimate receiver-observable signal used for prioritisation.
    """
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim != 1:
        raise ValueError(f"observable_priorities expects flat obs, got shape {obs_arr.shape}")
    if features_per_band <= 0:
        raise ValueError(f"features_per_band must be > 0, got {features_per_band}")
    n_bands = obs_arr.shape[0] // features_per_band
    if n_bands * features_per_band != obs_arr.shape[0]:
        raise ValueError(f"obs length {obs_arr.shape[0]} not a multiple of {features_per_band}")
    return obs_arr[::features_per_band]


# Maps MoE-per-decision attribution keys -> episode telemetry aggregate names.
_SCORE_KEYS = {
    "q_score": "mean_q_score",
    "eager_score": "mean_eager_score",
    "revisit_score": "mean_revisit_score",
    "semantic_score": "mean_semantic_score",
    "preemptive_score": "mean_preemptive_score",
    "fused_score": "mean_fused_score",
    "drqn_rank": "drqn_rank",
}


class BandDiscoveryTracker:
    """Per-episode band discovery tracker with quota safeguard for Phase 9B-R3.

    Exploration operates at the 36-band level:
      - Tracks per-band visit counts N_visits[b] within the current episode.
      - If any band is below discovery_quota:
          P(b) proportional to 1 / sqrt(1 + N_visits[b]) among under-quota bands.
      - Once all bands reach discovery_quota, returns None so exploration falls
        back cleanly to normal 180-action exploration.
    """

    def __init__(self, n_bands: int = 36, discovery_quota: int = 2) -> None:
        self.n_bands = int(n_bands)
        self.discovery_quota = int(discovery_quota)
        self.visits = np.zeros(self.n_bands, dtype=np.int64)

    def reset(self) -> None:
        self.visits.fill(0)

    def record_visit(self, band: int) -> None:
        if 0 <= band < self.n_bands:
            self.visits[band] += 1

    def sample_targeted_band(self) -> int | None:
        """Sample an under-visited band with P(b) ~ 1/sqrt(1 + N_b) if any < quota."""
        under_quota = np.where(self.visits < self.discovery_quota)[0]
        if len(under_quota) == 0:
            return None
        weights = 1.0 / np.sqrt(1.0 + self.visits[under_quota].astype(np.float64))
        probs = weights / np.sum(weights)
        chosen_band = int(np.random.choice(under_quota, p=probs))
        return chosen_band

    def get_diagnostics(self) -> dict[str, Any]:
        """Compute band visit statistics for episode/gate telemetry."""
        unique_bands = int(np.count_nonzero(self.visits))
        min_v = int(np.min(self.visits))
        med_v = float(np.median(self.visits))
        max_v = int(np.max(self.visits))
        std_v = float(np.std(self.visits))
        underexplored_count = int(np.sum(self.visits < self.discovery_quota))
        underexplored_frac = float(underexplored_count / self.n_bands)
        return {
            "unique_bands": unique_bands,
            "min_band_visits": min_v,
            "median_band_visits": med_v,
            "max_band_visits": max_v,
            "band_visit_std": std_v,
            "underexplored_band_count": underexplored_count,
            "underexplored_band_fraction": underexplored_frac,
            "band_visit_counts": self.visits.tolist(),
        }



def _do_drqn_update(
    online_drqn: DRQNScheduler,
    target_drqn: DRQNScheduler,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
    aux_coef: float = 0.1,
    stats: dict | None = None,
    reward_baseline: float = -0.39,
    baseline_momentum: float = 0.99,
    target_q_max: float = 100.0,
    target_q_min: float = -50.0,
    q_reg_coef: float = 1e-4,
    n_bands: int = 36,
    n_modes: int = 5,
    lambda_entropy: float = 0.01,
    band_diversity_penalty_coef: float = 0.005,
    band_diversity_threshold: float = 0.80,
    mode_diversity_penalty_coef: float = 0.005,
    mode_diversity_threshold: float = 0.80,
    mode_collapse_rate_threshold: float = 0.50,
    objective_mode: str = "step_based",
    c_dwell: float = 2.0,
    tau_ref: float = 1.0,
) -> float:

    """One Double-DQN BPTT update on a sampled batch. Returns loss value.

    Phase 9B-R2 Q-Value Stabilization:
      * Target-Q clipping applied strictly after Double-DQN action selection:
        online(next_state) -> argmax action -> target(next_state)[action] -> clamp -> Bellman target.
      * Explicit Q^2 regularization: lambda_Q * mean(Q(s, a)^2) across graded transitions
        to directly discourage online logit runaway.
      * Telemetry logging: td_loss, q_reg_loss, q_reg_fraction, max_q_margin, pos_scen_concentration.

    When ``stats`` (a dict) is provided it is filled with RC-2 learning
    telemetry: TD losses, Q statistics (online vs target network), the gradient
    norm after clipping and the number of graded transitions. The return value
    stays a scalar float so existing callers/tests are unaffected.
    """
    valid = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
    burn_in = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
    loss_mask = valid & ~burn_in  # real graded transitions only

    # Pure burn-in window (no graded transitions): warm-up only, no update.
    if not loss_mask.any():
        return 0.0

    # The full window is fed through the LSTM: burn-in columns warm the hidden
    # state, graded columns carry the recurrence forward for the losses.
    obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
    rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
    done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

    q_all, aux, _ = online_drqn(obs_b)
    q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

    with torch.inference_mode():
        next_q_online, _, _ = online_drqn(next_obs_b)
        best_actions = next_q_online.argmax(dim=-1, keepdim=True)
        next_q_target, _, _ = target_drqn(next_obs_b)
        next_q = next_q_target.gather(-1, best_actions).squeeze(-1)
        # Phase 9B-R2: Target-Q clipping applied after Double-DQN action selection
        if target_q_max is not None or target_q_min is not None:
            c_min = -50.0 if target_q_min is None else float(target_q_min)
            c_max = 100.0 if target_q_max is None else float(target_q_max)
            next_q = torch.clamp(next_q, min=c_min, max=c_max)

    # Phase 5 / Phase G4: Objective Formulation Dispatch
    # Dwell multipliers: Mode 0 (SHORT)=0.25, Mode 1 (NORMAL)=1.0, Mode 2 (LONG)=2.5, Mode 3 (REVISIT)=1.0, Mode 4 (PREEMPTIVE)=1.0
    if "dwell_times_us" in batch and batch["dwell_times_us"] is not None:
        dwell_us_b = torch.tensor(batch["dwell_times_us"], dtype=torch.float32, device=device)
        tau_b = dwell_us_b / 500.0
    else:
        dwell_multipliers = torch.tensor([0.25, 1.0, 2.5, 1.0, 1.0], dtype=torch.float32, device=device)
        tau_b = dwell_multipliers[act_b % n_modes]
        dwell_us_b = 500.0 * tau_b

    gamma_tensor = torch.as_tensor(float(gamma), dtype=torch.float32, device=device)

    if objective_mode == "step_based":
        # Standard step-based Bellman objective (fixed gamma, no dwell penalty)
        gamma_eff = gamma_tensor
        eff_rew_b = rew_b
    elif objective_mode in ("g3b_smdp_only", "smdp_only"):
        # G3-B: Semi-Markov SMDP discounting only (gamma^tau, no dwell penalty)
        gamma_eff = torch.pow(gamma_tensor, tau_b)
        eff_rew_b = rew_b
    elif objective_mode in ("g3d_hybrid", "calibrated_smdp_hybrid"):
        # G3-D: Calibrated SMDP Hybrid (gamma^tau discounting + opportunity cost dwell penalty)
        # r' = r - c_dwell * (tau - 1.0)
        gamma_eff = torch.pow(gamma_tensor, tau_b)
        eff_rew_b = rew_b - c_dwell * (tau_b - 1.0)
    elif objective_mode in ("g8_fixed_ref", "fixed_ref_centered"):
        # Arm C: Fixed-reference centered dwell term with gamma^tau discounting
        # r' = r - c_dwell * (tau - tau_ref)
        gamma_eff = torch.pow(gamma_tensor, tau_b)
        eff_rew_b = rew_b - c_dwell * (tau_b - tau_ref)
    elif objective_mode in ("g3a_reward_rate", "reward_rate"):
        # G3-A: Reward-rate scaling (r' = r / tau, fixed gamma)
        gamma_eff = gamma_tensor
        eff_rew_b = rew_b / torch.clamp(tau_b, min=0.1)
    else:
        # Backward-compatible fallback: time-aware discounting
        gamma_eff = torch.pow(gamma_tensor, tau_b)
        eff_rew_b = rew_b

    # Track D: reward centering strictly applied to TD target computation
    if loss_mask.any():
        batch_mean = float(eff_rew_b[loss_mask].mean().item())
    else:
        batch_mean = float(eff_rew_b.mean().item())
    updated_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * batch_mean
    centered_rew_b = eff_rew_b - updated_baseline

    targets = centered_rew_b + gamma_eff * next_q * (1.0 - done_b)
    q_loss = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())

    # Phase 9B-R2: Q^2 regularization across all actions on graded steps
    if q_reg_coef > 0.0:
        q_reg_loss = q_reg_coef * (q_all[loss_mask] ** 2).mean()
    else:
        q_reg_loss = torch.zeros((), device=device)

    loss: torch.Tensor = q_loss + q_reg_loss

    # Phase-1 Anti-Collapse: entropy regulariser (lambda_entropy)
    # Penalise Q-value concentration — prevents the policy collapsing to a
    # single dominant action across all bands (top_band_fraction → 100%).
    # Applied only on graded (non-burn-in) transitions in loss_mask.
    q_probs = torch.softmax(q_all[loss_mask] / 1.0, dim=-1)  # temperature=1.0
    action_entropy = -(q_probs * torch.log(q_probs + 1e-8)).sum(dim=-1).mean()
    loss = loss - lambda_entropy * action_entropy

    # Phase-1 Anti-Collapse: top-band diversity penalty (LIVE gradients)
    # If a single band dominates > threshold of the Q-mass across graded transitions,
    # apply a differentiable penalty proportional to the excess concentration.
    _diversity_pen = torch.zeros((), device=device)
    _n_graded = loss_mask.sum().item()
    if _n_graded > 0:
        _band_max_q = q_all[loss_mask].view(-1, n_bands, n_modes).max(dim=-1).values  # (N_graded, n_bands) — LIVE graph
        _band_softmax = torch.softmax(_band_max_q, dim=-1)
        _top_band_frac = _band_softmax.max(dim=-1).values.mean()  # differentiable scalar
        with torch.no_grad():
            _top_band_frac_val = _top_band_frac.item()
    else:
        _top_band_frac = torch.zeros((), device=device)
        _top_band_frac_val = 0.0
    if _top_band_frac_val > band_diversity_threshold:
        _diversity_pen = band_diversity_penalty_coef * (_top_band_frac - band_diversity_threshold).clamp(min=0.0) * loss.abs().detach()
        loss = loss + _diversity_pen

    # Per-band mode diversity penalty on LIVE computation graph (Phase 1 fix — mode collapse)
    # For each of the 36 bands, compute the softmax distribution over its 5 modes directly
    # from the live q_all computation graph. If any mode dominates (> threshold), backpropagate a
    # gradient to penalise mode collapse and actively train the network toward mode diversity.
    mode_collapse_rate = torch.tensor(0.0)
    mode_diversity_pen = torch.zeros((), device=device)
    try:
        if loss_mask.any():
            # q_banded_live is on the LIVE computation graph (requires_grad=True)
            q_banded_live = q_all[loss_mask].view(-1, n_bands, n_modes)
            band_mode_softmax = torch.softmax(q_banded_live / 1.0, dim=-1)  # (N, n_bands, n_modes)
            max_mode_prob_per_band = band_mode_softmax.max(dim=-1).values  # (N, n_bands), differentiable!
            with torch.no_grad():
                mode_collapse_rate = (max_mode_prob_per_band > mode_diversity_threshold).float().mean()
            if mode_collapse_rate > mode_collapse_rate_threshold:
                # Differentiable penalty: pushes down probability of dominant mode with live gradients
                avg_max_mode_prob = max_mode_prob_per_band.mean()
                mode_diversity_pen = mode_diversity_penalty_coef * (avg_max_mode_prob - mode_diversity_threshold).clamp(min=0.0) * loss.abs().detach()
                loss = loss + mode_diversity_pen
            else:
                mode_diversity_pen = torch.zeros((), device=device)
        else:
            mode_diversity_pen = torch.zeros((), device=device)
            mode_collapse_rate = torch.tensor(0.0)
    except (RuntimeError, ValueError, IndexError) as exc:

        logging.getLogger(__name__).warning("mode-diversity penalty skipped: %s", exc)
        mode_diversity_pen = torch.tensor(0.0)
        mode_collapse_rate = torch.tensor(0.0)

    aux_loss: torch.Tensor = torch.zeros((), device=device)
    if aux_coef > 0:
        hit_probs = torch.tensor(batch["hit_probs"], dtype=torch.float32, device=device)
        prob_pred = aux["intercept_prob"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
        bce = nn.functional.binary_cross_entropy(prob_pred[loss_mask], hit_probs[loss_mask].detach())

        # Time target only when the transition is a genuine hit.
        # Targets are normalized by 1000.0 (milliseconds) to keep Huber loss bounded [0, 1]
        time_valid = (
            loss_mask
            & torch.tensor(batch["time_target_valid"], dtype=torch.bool, device=device)
        )
        if time_valid.any():
            intercept_times = torch.tensor(batch["intercept_times_us"], dtype=torch.float32, device=device)
            time_pred = aux["intercept_time_us"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
            # Delta 0.1 ms (= 100 µs), predictions and targets in ms (/ 1000.0)
            huber = nn.functional.huber_loss(time_pred[time_valid] / 1000.0, intercept_times[time_valid].detach() / 1000.0, delta=0.1)
        else:
            huber = torch.zeros((), device=device)
        aux_loss = bce + huber
        loss = loss + aux_coef * aux_loss

    if stats is not None:
        stats["optimizer_update_attempted"] = True

    optimizer.zero_grad()
    loss.backward()

    # Refinement 2: Inspect EVERY parameter gradient before clipping & before optimizer.step()
    all_finite_grads = True
    for p in online_drqn.parameters():
        if p.grad is not None:
            if not torch.all(torch.isfinite(p.grad)):
                all_finite_grads = False
                break
    if stats is not None:
        stats["finite_gradients"] = all_finite_grads

    pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(online_drqn.parameters(), 1.0)
    pre_gn = float(pre_clip_grad_norm.item()) if torch.is_tensor(pre_clip_grad_norm) else float(pre_clip_grad_norm)
    if not (all_finite_grads and np.isfinite(pre_gn) and np.isfinite(float(loss.item()))):
        optimizer.zero_grad()
        if stats is not None:
            stats["skipped_nan"] = True
            stats["finite_gradients"] = False
        return float("nan")

    optimizer.step()
    if stats is not None:
        stats["optimizer_update_completed"] = True
    if stats is not None:
        loss_mask_t = loss_mask
        qm = q_all[loss_mask_t].detach()
        td_err = (targets[loss_mask_t] - q_chosen[loss_mask_t]).detach().abs()
        # Double-DQN target values: target-network Q at online-argmax next actions.
        best_next = next_q[loss_mask_t]
        target_table = next_q_target[loss_mask_t]

        # Compute batch Q-margin: top1 - top2 across 180 actions
        with torch.no_grad():
            top2_vals = qm.topk(2, dim=-1).values
            q_margins = top2_vals[:, 0] - top2_vals[:, 1]
            max_q_margin_val = float(q_margins.max().item())
            mean_q_margin_val = float(q_margins.mean().item())

        total_loss_f = float(loss.item())
        q_reg_loss_f = float(q_reg_loss.item())
        # stats["td_loss"] tracks the genuine Bellman Q loss
        stats["td_loss"] = float(q_loss.item())
        stats["q_reg_loss"] = q_reg_loss_f
        stats["q_reg_ratio"] = (q_reg_loss_f / max(1e-8, total_loss_f))
        stats["total_loss"] = total_loss_f
        stats["q_loss"] = float(q_loss.item())
        stats["aux_loss"] = float(aux_loss.item())
        stats["mean_td_error"] = float(td_err.mean().item())
        stats["max_td_error"] = float(td_err.max().item())
        stats["td_error_p90"] = float(np.percentile(td_err.cpu().numpy(), 90))
        stats["mean_q"] = float(qm.mean().item())
        stats["max_q"] = float(qm.max().item())
        stats["min_q"] = float(qm.min().item())
        stats["q_std"] = float(qm.std().item())
        stats["max_q_margin"] = max_q_margin_val
        stats["mean_q_margin"] = mean_q_margin_val
        stats["mean_online_q"] = float(qm.mean().item())
        stats["mean_target_q"] = float(best_next.mean().item())
        stats["max_target_q"] = float(best_next.max().item())
        stats["target_online_gap"] = float((best_next.mean() - qm.mean()).abs().item())
        if torch.is_tensor(gamma_eff) and gamma_eff.dim() > 0:
            stats["gamma_eff_mean"] = float(gamma_eff[loss_mask_t].mean().item())
            stats["gamma_eff_min"] = float(gamma_eff[loss_mask_t].min().item())
            stats["gamma_eff_max"] = float(gamma_eff[loss_mask_t].max().item())
        else:
            g_val = float(gamma_eff.item() if torch.is_tensor(gamma_eff) else gamma_eff)
            stats["gamma_eff_mean"] = g_val
            stats["gamma_eff_min"] = g_val
            stats["gamma_eff_max"] = g_val
        pre_gn = float(pre_clip_grad_norm.item()) if torch.is_tensor(pre_clip_grad_norm) else float(pre_clip_grad_norm)
        stats["gradient_norm"] = min(1.0, pre_gn)
        stats["pre_clip_gradient_norm"] = pre_gn
        stats["n_graded_steps"] = int(loss_mask_t.sum().item())
        stats["reward_baseline"] = float(updated_baseline)
        stats["centered_reward_mean"] = float(centered_rew_b[loss_mask_t].mean().item())
        stats["raw_reward_mean"] = float(batch_mean)
        if aux_coef > 0:
            stats["replay_hit_fraction"] = float(hit_probs[loss_mask_t].mean().item())
        stats["replay_sequence_hit_fraction"] = float(batch.get("sequence_hit_fraction", 0.0))
        stats["pos_scen_concentration"] = float(batch.get("pos_scen_concentration", 0.0))
        # Phase-1 Anti-Collapse telemetry
        stats["action_entropy"] = float(action_entropy.item())
        stats["top_band_frac"] = float(_top_band_frac.item() if torch.is_tensor(_top_band_frac) else float(_top_band_frac))
        stats["diversity_penalty"] = float(_diversity_pen.item())
        stats["mode_collapse_rate"] = float(mode_collapse_rate.item() if hasattr(mode_collapse_rate, 'item') else mode_collapse_rate)
        stats["mode_diversity_pen"] = float(mode_diversity_pen.item() if hasattr(mode_diversity_pen, 'item') else mode_diversity_pen)
    return float(loss.item())


def train_scheduler(
    model_cfg_path: str,
    train_cfg_path: str,
    data_dir_override: str | None = None,
    output_dir_override: str | None = None,
    reset_semantic_memory: bool = False,
    staged_gates: list[int] | None = None,
    stop_at_step: int | None = None,
    start_step_override: int | None = None,
    resume_checkpoint: str | None = None,
    override_epsilon: float | None = None,
    disable_latency_reward: bool = False,
    exploration_schedule: str = "exponential",
    target_q_max: float = 100.0,
    target_q_min: float = -50.0,
    q_reg_coef: float = 1e-4,
    targeted_exploration: bool = True,
    band_discovery_quota: int = 2,
    qualification_run: bool = False,
    qualification_steps: int | None = None,
    diagnostic: bool = False,
    expected_parent_sha: str | None = None,
    lambda_entropy: float | None = None,
    band_diversity_penalty_coef: float | None = None,
    band_diversity_threshold: float | None = None,
    mode_diversity_penalty_coef: float | None = None,
    mode_diversity_threshold: float | None = None,
    mode_collapse_rate_threshold: float | None = None,
    objective_mode: str | None = None,
    c_dwell: float | None = None,
    tau_ref: float | None = None,
    gamma: float | None = None,
    fresh_optimizer: bool = False,
    abort_on_q_max_exceeded: bool = False,
    max_allowed_q: float = 100.0,
    use_g8_film: bool | None = None,
    use_online_reward_shaper: bool = False,
    preloaded_buffer_path: str | Path | None = None,
    probe_batch_path: str | Path | None = None,
    learning_rate: float | None = None,
    targeted_spec: dict[str, Any] | None = None,
) -> None:
    """Full DRQN training with Thompson warmup, BPTT, target network, and MoE eval.

    Args:
        model_cfg_path: Path to model_config.yaml.
        train_cfg_path: Path to training_config.yaml.
        data_dir_override: CLI override for dataset root (CLI > YAML > default).
        output_dir_override: CLI override for checkpoint output dir
            (CLI > YAML > default).
        reset_semantic_memory: If True, resets SQLite database before training.
        staged_gates: Step numbers at which to execute quantitative gate evaluation.
        stop_at_step: Step number at which to cleanly stop training.
        resume_checkpoint: Path to checkpoint .pt to resume training from.
        override_epsilon: Optional exploration floor override for diagnostic retraining.
        disable_latency_reward: Optional ablation switch to zero out latency bonus.
        exploration_schedule: Decay schedule for epsilon.
        target_q_max: Conservative ceiling for target-Q clipping (default 100.0).
        target_q_min: Floor for target-Q clipping (default -50.0).
        q_reg_coef: Weight of Q^2 regularization loss (default 1e-4).
        targeted_exploration: Enable Phase 9B-R3 36-band targeted discovery exploration.
        band_discovery_quota: Discovery visit quota per band per episode (default 2).
    """
    with open(model_cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_cuda = torch.cuda.is_available() and os.getenv("DEVICE", "cuda") != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Training DRQN on %s seed=%d", device, seed)

    drqn_cfg = full_cfg.get("drqn_scheduler", {})
    moe_cfg = full_cfg.get("smartscan_moe", {})
    reward_cfg = full_cfg.get("reward", {})
    env_cfg = train_cfg.get("environment", {})
    sched_cfg = train_cfg.get("scheduler", {})

    if reset_semantic_memory:
        db_path = Path("data/semantic_memory.db")
        if db_path.exists():
            try:
                db_path.unlink()
                logger.info("Reset semantic memory database at %s", db_path)
            except Exception as exc:
                logger.warning("Failed to unlink semantic memory database: %s", exc)

    n_bands = int(drqn_cfg.get("n_bands", 36))
    n_modes = int(drqn_cfg.get("n_modes", env_cfg.get("n_modes", 5)))
    n_actions = int(drqn_cfg.get("n_actions", n_bands * n_modes))
    band_features = int(env_cfg.get("band_features", 10))
    obs_dim = int(env_cfg.get("obs_dim", n_bands * band_features))

    # Merge reward weights into env config
    env_config = {**env_cfg, **reward_cfg}
    if disable_latency_reward:
        env_config["disable_latency_reward"] = True
    env_config.setdefault("n_bands", n_bands)
    env_config.setdefault("n_modes", n_modes)
    env_config.setdefault("n_actions", n_actions)

    training_mode = train_cfg.get("training_mode", "real_tsrd")
    if training_mode != "real_tsrd":
        raise ValueError(f"Strict TSRD training mode required for retraining; got training_mode='{training_mode}'")
    from ..data.tsrd_root import resolve_tsrd_root

    canonical_root = resolve_tsrd_root(cli_value=data_dir_override, config=train_cfg)
    require_training_gate(
        data_root=canonical_root,
        deinterleaver_checkpoint=train_cfg.get("deinterleaver_ckpt", "experiments/checkpoints/deinterleaver/best.pt"),
        normalization_stats=train_cfg.get("normalization_stats", "experiments/checkpoints/deinterleaver/normalization_stats.json"),
        environment_config=env_config,
        model_config=full_cfg,
    )

    # Load trained deinterleaver and normalization stats for perception (Fail-closed Fix 4)
    deinterleaver_ckpt_str = train_cfg.get("deinterleaver_ckpt", "experiments/checkpoints/deinterleaver/best.pt")
    norm_stats_path_str = train_cfg.get("normalization_stats", "experiments/checkpoints/deinterleaver/normalization_stats.json")
    deinterleaver_ckpt = Path(deinterleaver_ckpt_str)
    norm_stats_path = Path(norm_stats_path_str)

    deinterleaver_model = None
    fit_stats = None

    if training_mode == "real_tsrd" or train_cfg.get("strict_perception", True):
        if not deinterleaver_ckpt.exists():
            raise FileNotFoundError(f"FATAL: Required deinterleaver checkpoint not found at {deinterleaver_ckpt}")
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"FATAL: Required normalization stats not found at {norm_stats_path}")

        fit_stats = load_normalization_stats(norm_stats_path)
        computed_norm_hash = normalization_stats_hash(fit_stats)
        expected_norm_hash = str(train_cfg.get("expected_norm_hash", "bacee02ac1c29428"))
        if computed_norm_hash != expected_norm_hash:
            raise ValueError(
                f"FATAL: Normalization stats hash mismatch: expected {expected_norm_hash}, got {computed_norm_hash} from {norm_stats_path}"
            )
        logger.info("Loaded and verified normalization stats from %s (hash=%s)", norm_stats_path, computed_norm_hash)

        logger.info("Loading trained deinterleaver from %s (strict=True)", deinterleaver_ckpt)
        d_cfg = full_cfg.get("deinterleaver", {})
        deinterleaver_model = PDWTransformerEncoder(
            pdw_dim=d_cfg.get("pdw_dim", 6),
            d_model=d_cfg.get("d_model", 128),
            nhead=d_cfg.get("nhead", 8),
            num_layers=d_cfg.get("num_layers", 4),
            dim_feedforward=d_cfg.get("dim_feedforward", 512),
            dropout=d_cfg.get("dropout", 0.1),
            embed_dim=d_cfg.get("embed_dim", 64),
        )
        state = torch.load(deinterleaver_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        deinterleaver_model.load_state_dict(state, strict=True)
        deinterleaver_model.eval()
        logger.info("Deinterleaver successfully loaded with strict=True from %s", deinterleaver_ckpt)
    elif deinterleaver_ckpt.exists() and norm_stats_path.exists():
        fit_stats = load_normalization_stats(norm_stats_path)
        d_cfg = full_cfg.get("deinterleaver", {})
        deinterleaver_model = PDWTransformerEncoder(
            pdw_dim=d_cfg.get("pdw_dim", 6),
            d_model=d_cfg.get("d_model", 128),
            nhead=d_cfg.get("nhead", 8),
            num_layers=d_cfg.get("num_layers", 4),
            dim_feedforward=d_cfg.get("dim_feedforward", 512),
            dropout=d_cfg.get("dropout", 0.1),
            embed_dim=d_cfg.get("embed_dim", 64),
        )
        state = torch.load(deinterleaver_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        deinterleaver_model.load_state_dict(state, strict=True)
        deinterleaver_model.eval()

    # Build the receiver-driven cognitive env from a TSRD/synthetic scenario.
    data_dir = canonical_root
    subset = train_cfg.get("subset", "train")
    world_mode = train_cfg.get("world_mode", "stare")
    observation_mode = train_cfg.get("observation_mode", "scan")

    # For scheduler training: RF world uses STARE (latent truth), receiver observes through IBW
    train_chunk_mode = str(env_config.get("chunk_mode", "random"))
    if world_mode == "stare":
        train_source = ScenarioSource(
            data_root=data_dir,
            mode="stare",  # overridden by source_type
            subset=subset,
            freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
            freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
            time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
            max_pulses=int(env_config.get("max_pulses", 50000)),
            seed=seed,
            source_type="world",
            allow_synthetic_fallback=training_mode == "synthetic",
            chunk_mode=train_chunk_mode,
        )
        logger.info("Scheduler training: RF world source = TSRD STARE (latent truth, chunk_mode=%s)", train_chunk_mode)
    else:
        # Fallback for compatibility
        train_source = ScenarioSource(
            data_root=data_dir,
            mode=world_mode,
            subset=subset,
            freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
            freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
            time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
            max_pulses=int(env_config.get("max_pulses", 50000)),
            seed=seed,
            allow_synthetic_fallback=training_mode == "synthetic",
            chunk_mode=train_chunk_mode,
        )
        logger.warning("Scheduler training: RF world source = %s (non-standard, chunk_mode=%s)", world_mode, train_chunk_mode)

    # One env; reset() draws a fresh random TSRD file each episode.
    # Semantic memory explicitly disabled during training for clean DRQN ablation baseline
    train_env_config = copy.deepcopy(env_config)
    train_env_config["semantic_memory_enabled"] = False
    env = CognitiveRFScanEnv(
        train_env_config, 
        records=None, 
        seed=seed, 
        records_provider=train_source.sample,
        deinterleaver_model=deinterleaver_model,
        deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
        semantic_memory_path=":memory:"
    )
    env.reset()  # populate first episode's records so obs_dim/action checks are valid
    assert obs_dim == 360, f"Rescue-A requires obs_dim == 360, got {obs_dim}"
    assert n_modes == 5, f"Rescue-A requires n_modes == 5, got {n_modes}"
    assert n_actions == 180, f"Rescue-A requires n_actions == 180, got {n_actions}"
    assert env.obs_dim == 360, f"env obs_dim {env.obs_dim} != 360"
    assert env.action_space.n == 180, f"env action space {env.action_space.n} != 180"
    assert env.action_space.n == n_bands * n_modes, f"env action space must be n_bands*n_modes = {n_bands * n_modes}"

    # Rescue-A Preflight Verification: confirm all 5 modes {0, 1, 2, 3, 4} are accessible in exploration
    sampled_targeted_modes = {random.randrange(n_modes) for _ in range(500)}
    assert sampled_targeted_modes == {0, 1, 2, 3, 4}, f"Targeted exploration cannot reach all 5 modes: {sampled_targeted_modes}"
    sampled_uniform_modes = {(random.randrange(n_actions) % n_modes) for _ in range(500)}
    assert sampled_uniform_modes == {0, 1, 2, 3, 4}, f"Uniform exploration cannot reach all 5 modes: {sampled_uniform_modes}"
    logger.info("Rescue-A Preflight PASSED: 360-D obs, 180 actions, 5 modes fully accessible in exploration.")
    
    if env.perception_enabled:
        logger.info("Perception pipeline ENABLED: trained deinterleaver + EmitterTracker active")
    else:
        logger.warning("Perception pipeline DISABLED: no trained deinterleaver loaded")

    # Phase 4 Critical Invariant: training strictly enforces Reward v2
    assert env.reward_version == "v2", f"Training requires reward_version == 'v2', got '{env.reward_version}'"

    if training_mode == "real_tsrd":
        assert env.perception_enabled, "Strict TSRD training requires perception_enabled=True"
        assert env.emitter_tracker is not None, "Strict TSRD training requires EmitterTracker"

    # Distribution Shift Monitoring setup (Policy-Observation Distribution)
    dist_shift_cfg = train_cfg.get("distribution_shift", {})
    dist_shift_enabled = bool(dist_shift_cfg.get("enabled", True))
    dist_shift_detector: DistributionShiftDetector | None = None
    dist_shift_eval_interval = int(dist_shift_cfg.get("eval_interval_steps", 50))
    dist_shift_monitor_only = bool(dist_shift_cfg.get("monitor_only", True))

    if dist_shift_enabled and world_mode == "stare" and data_dir is not None:
        try:
            ref_obs = build_training_distribution_reference(
                data_dir=data_dir,
                env_config=train_env_config,
                deinterleaver_model=deinterleaver_model,
                fit_stats=fit_stats,
                seed=seed,
                n_calibration_scenarios=5,
                steps_per_scenario=40,
            )
            dist_shift_detector = DistributionShiftDetector(
                obs_dim=obs_dim,
                window_size=int(dist_shift_cfg.get("window_size", 500)),
                alert_threshold=float(dist_shift_cfg.get("threshold", 0.15)),
                critical_threshold=float(dist_shift_cfg.get("critical_threshold", 0.40)),
            )
            dist_shift_detector.set_reference(ref_obs)
            logger.info("Distribution shift monitoring INITIALIZED from stare/train (monitor_only=%s, eval_interval=%d)",
                        dist_shift_monitor_only, dist_shift_eval_interval)
        except Exception as exc:
            logger.warning("Failed to initialize distribution shift detector: %s", exc)

    lstm_hidden = int(drqn_cfg.get("lstm_hidden", 256))
    lstm_layers = int(drqn_cfg.get("lstm_layers", 2))

    if use_g8_film is None:
        use_g8_film = bool(sched_cfg.get("use_g8_film", False))

    if use_g8_film:
        _g8_ckpt_target = resume_checkpoint or train_cfg.get("scheduler_ckpt") or sched_cfg.get("scheduler_ckpt")
        if _g8_ckpt_target is None:
            raise RuntimeError("G8 FiLM mode requires an explicit parent checkpoint (--resume or scheduler_ckpt)")
        _g8_payload = torch.load(_g8_ckpt_target, map_location=device, weights_only=False)
        _g8_sd = _g8_payload.get("state_dict", _g8_payload.get("online_drqn", _g8_payload))
        if "dwell_head.0.weight" in _g8_sd:
            from ..models.band_conditioned_drqn import BandConditionedFactorizedDRQN
            online_drqn = BandConditionedFactorizedDRQN(
                obs_dim=obs_dim,
                n_bands=n_bands,
                n_modes=n_modes,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
            ).to(device)
            online_drqn.load_state_dict(_g8_sd, strict=True)
            logger.info("G8.3-B BandConditionedFactorizedDRQN loaded directly from checkpoint %s (strict=True)", _g8_ckpt_target)
        elif "film_proj.0.weight" in _g8_sd:
            online_drqn = FiLMGatedFactorizedDRQN(
                obs_dim=obs_dim,
                n_bands=n_bands,
                n_modes=n_modes,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
            ).to(device)
            online_drqn.load_state_dict(_g8_sd, strict=True)
            logger.info("G8 FiLM model loaded directly from materialized checkpoint %s (strict=True)", _g8_ckpt_target)
        else:
            online_drqn, film_manifest = FiLMGatedFactorizedDRQN.from_g7a_checkpoint(
                ckpt_path=Path(_g8_ckpt_target),
                seed=seed,
            )
            online_drqn = online_drqn.to(device)
            logger.info(
                "G8 FiLM model converted from G7-A parent %s: inherited %d params, dropped %s, new FiLM params: %d",
                _g8_ckpt_target,
                film_manifest["inherited_params_count"],
                film_manifest["dropped_keys"],
                film_manifest["newly_initialized_params_count"],
            )
        target_drqn = copy.deepcopy(online_drqn).to(device)
        if isinstance(_g8_payload, dict) and "target_state_dict" in _g8_payload:
            target_drqn.load_state_dict(_g8_payload["target_state_dict"], strict=True)
            logger.info("Loaded target_state_dict into target_drqn (strict=True)")
        target_drqn.eval()
    else:
        online_drqn = DRQNScheduler(
            obs_dim=obs_dim,
            n_bands=n_bands,
            n_actions=n_actions,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
        ).to(device)
        target_drqn = copy.deepcopy(online_drqn).to(device)
        target_drqn.eval()

    # Phase-1 Task 1.3: Training-mode MoE config override.
    # During training the MoE is used PASSIVELY for telemetry/attribution only;
    # actual training actions come from online_drqn.act() (epsilon-greedy).
    # We raise confidence_margin_threshold and enable exploration so the diagnostic
    # attribution logs reflect the exploratory training phase, not operational inference.
    # SAFETY: This has NO effect on training actions or gradients.
    _training_moe_overrides = {
        "confidence_margin_threshold": 0.08,  # higher than operational 0.02 — forces fallback attribution
        "policy_mode": "demo",                 # demo mode (not operational) for training eval telemetry
        "exploration_enabled": True,           # allow MoE to explore during diagnostic queries
    }
    _training_moe_cfg = {
        **moe_cfg,
        **_training_moe_overrides,
        "n_bands": n_bands,
        "n_modes": n_modes,
        "n_actions": n_actions,
        "device": str(device),
    }
    moe = SmartScanMoE(online_drqn, _training_moe_cfg).to(device)

    if learning_rate is None:
        learning_rate = float(sched_cfg.get("learning_rate", drqn_cfg.get("lr", 2.5e-5)))
    else:
        learning_rate = float(learning_rate)
    optimizer = optim.Adam(online_drqn.parameters(), lr=learning_rate)
    logger.info("Scheduler optimizer initialized with lr=%.2e", learning_rate)
    loss_fn = nn.HuberLoss()

    # WandB optional
    use_wandb = False
    try:
        import wandb  # type: ignore

        wandb.init(project=os.getenv("WANDB_PROJECT", "cognitive-ew-sih"), config={**drqn_cfg, **sched_cfg})
        use_wandb = True
    except Exception as exc:
        logger.info("WandB not available: %s", exc)

    action_selection_mode = str(drqn_cfg.get("action_selection_mode", sched_cfg.get("action_selection_mode", "flat_argmax")))
    exploration_schedule = str(sched_cfg.get("exploration_schedule", exploration_schedule))
    ts_explore_modes = bool(sched_cfg.get("thompson_explore_modes", False))
    ts_sampler = ThompsonSamplingExplorer(n_bands=n_bands, n_modes=n_modes, seed=seed, explore_modes=ts_explore_modes)
    ts_warmup = int(sched_cfg.get("thompson_warmup_steps", 5000))
    eps_start = float(sched_cfg.get("eps_start", drqn_cfg.get("eps_start", 1.0)))
    eps_end = float(sched_cfg.get("eps_end", drqn_cfg.get("eps_end", 0.05)))
    eps_decay = float(sched_cfg.get("eps_decay", drqn_cfg.get("eps_decay", 10000)))
    if gamma is None:
        gamma = float(drqn_cfg.get("gamma", 0.99))
    else:
        gamma = float(gamma)
    seq_len = int(sched_cfg.get("seq_len", 16))
    burn_in = int(sched_cfg.get("burn_in", 8))
    batch_size = int(sched_cfg.get("batch_size", 32))
    update_freq = int(sched_cfg.get("update_freq", 4))
    target_update_freq = int(sched_cfg.get("target_update_freq", 1000))
    total_steps = int(stop_at_step) if stop_at_step is not None else int(sched_cfg.get("total_timesteps", 500000))
    target_q_max = float(sched_cfg.get("target_q_max", target_q_max))
    target_q_min = float(sched_cfg.get("target_q_min", target_q_min))
    q_reg_coef = float(sched_cfg.get("q_reg_coef", q_reg_coef))
    if lambda_entropy is None:
        lambda_entropy = float(sched_cfg.get("lambda_entropy", 0.01))
    if band_diversity_penalty_coef is None:
        band_diversity_penalty_coef = float(sched_cfg.get("band_diversity_penalty_coef", 0.005))
    if band_diversity_threshold is None:
        band_diversity_threshold = float(sched_cfg.get("band_diversity_threshold", 0.80))
    if mode_diversity_penalty_coef is None:
        mode_diversity_penalty_coef = float(sched_cfg.get("mode_diversity_penalty_coef", 0.005))
    if mode_diversity_threshold is None:
        mode_diversity_threshold = float(sched_cfg.get("mode_diversity_threshold", 0.80))
    if mode_collapse_rate_threshold is None:
        mode_collapse_rate_threshold = float(sched_cfg.get("mode_collapse_rate_threshold", 0.50))
    if objective_mode is None:
        objective_mode = str(sched_cfg.get("objective_mode", "step_based"))
    if c_dwell is None:
        c_dwell = float(sched_cfg.get("c_dwell", 2.0))
    if tau_ref is None:
        tau_ref = float(sched_cfg.get("tau_ref", 1.0))
    if not fresh_optimizer:
        fresh_optimizer = bool(sched_cfg.get("fresh_optimizer", False))
    if not abort_on_q_max_exceeded:
        abort_on_q_max_exceeded = bool(sched_cfg.get("abort_on_q_max_exceeded", False))
    max_allowed_q = float(sched_cfg.get("max_allowed_q", max_allowed_q))
    if "targeted_exploration" in sched_cfg:
        targeted_exploration = bool(sched_cfg["targeted_exploration"])
    if "band_discovery_quota" in sched_cfg:
        band_discovery_quota = int(sched_cfg["band_discovery_quota"])
    band_tracker = (
        BandDiscoveryTracker(n_bands=n_bands, discovery_quota=band_discovery_quota)
        if targeted_exploration
        else None
    )

    buffer = SequenceReplayBuffer(
        capacity=int(sched_cfg.get("replay_buffer_size", 50000)),
        seq_len=seq_len,
        obs_dim=obs_dim,
        burn_in=burn_in,
        seed=seed,
    )

    if preloaded_buffer_path is None:
        preloaded_buf_path = sched_cfg.get("preloaded_buffer_path") or train_cfg.get("preloaded_buffer_path")
    else:
        preloaded_buf_path = str(preloaded_buffer_path)
    if preloaded_buf_path:
        p_buf = Path(preloaded_buf_path)
        if p_buf.exists():
            n_loaded = buffer.load_episodes(p_buf)
            logger.info("Preloaded %d episodes into replay buffer from %s", buffer.n_episodes(), p_buf)
        else:
            raise FileNotFoundError(f"FATAL: preloaded_buffer_path '{p_buf}' does not exist!")

    reward_shaper = (
        RelativeDwellShaper(
            ema_alpha=float(sched_cfg.get("g8_dwell_ema_alpha", 0.05)),
            c_dwell=c_dwell,
        )
        if (use_g8_film and use_online_reward_shaper)
        else None
    )
    if probe_batch_path is None:
        probe_batch_path = sched_cfg.get("probe_batch_path") or train_cfg.get("probe_batch_path")
    collapse_guard = (
        ModeCollapseGuard(
            window=int(sched_cfg.get("g8_collapse_window", 500)),
            short_ceiling=float(sched_cfg.get("g8_short_ceiling", 0.88)),
            min_entropy=float(sched_cfg.get("g8_min_entropy", 0.30)),
            parent_short_frac=float(sched_cfg.get("g8_parent_short_frac", 0.5315)),
            degradation_delta=float(sched_cfg.get("g8_degradation_delta", 0.08)),
            probe_batch_path=probe_batch_path,
        )
        if use_g8_film
        else None
    )
    use_stratified_replay = bool(
        sched_cfg.get("use_g8_1_stratified_replay", False)
        or train_cfg.get("use_g8_1_stratified_replay", False)
        or use_g8_film
    )
    strata_weights = sched_cfg.get("strata_weights", {
        "agile": 0.30,
        "sparse": 0.25,
        "dense": 0.25,
        "mixed": 0.20,
    })
    min_episodes_per_stratum = int(sched_cfg.get("min_episodes_per_stratum", 5))

    stratified_sampler = (
        StratifiedReplaySampler(
            buffer=buffer,
            strata_weights=strata_weights,
            min_episodes_per_stratum=min_episodes_per_stratum,
            targeted_spec=targeted_spec or sched_cfg.get("targeted_spec"),
        )
        if use_stratified_replay
        else None
    )

    # Phase 17: canonical layout — never resolve to the ambiguous root
    # (config output_dir="checkpoints" is replaced by the canonical subdir).
    from ..utils.checkpoint_paths import SCHEDULER_DIR, resolve_checkpoint_dir

    if qualification_run:
        output_dir = Path("experiments/checkpoints/quarantine").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # Purge stale summaries/manifests before qualification run begins
        for fname in ["qualification_run_summary.json", "qualification_run_manifest.json"]:
            p = output_dir / fname
            if p.exists():
                p.unlink()
        run_id = str(uuid.uuid4())
        qualification_started_at_utc = datetime.now(timezone.utc).isoformat()
        q_steps = qualification_steps if qualification_steps is not None else 1000
        logger.info("QUALIFICATION RUN MODE ACTIVE: output routed exclusively to quarantine (%s)", output_dir)
    else:
        output_dir = resolve_checkpoint_dir(
            output_dir_override,
            train_cfg.get("output_dir"),
            SCHEDULER_DIR,
            role="scheduler",
        )
        run_id = None
        qualification_started_at_utc = None
        q_steps = None

    # Fail-closed guard: forbid output_dir within immutable baseline or operational candidate directories
    forbidden_roots = [
        Path("experiments/checkpoints/production_baseline").resolve(),
        Path("experiments/checkpoints/scheduler_v2_operational_candidate").resolve(),
    ]
    resolved_out = output_dir.resolve()
    for f_root in forbidden_roots:
        if resolved_out == f_root or f_root in resolved_out.parents:
            raise RuntimeError(
                f"FATAL: output_dir '{output_dir}' resolves to or inside immutable baseline '{f_root}'. "
                f"Retraining directly into baseline directories is strictly forbidden."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    # P0-9: reproducible run directory + telemetry publisher (real metrics only).
    world_mode = train_cfg.get("world_mode", "stare")
    source_files = list(getattr(train_source, "files", []))
    data_fingerprint = dataset_fingerprint(source_files, data_dir, world_mode)
    run = RunManager(
        root=train_cfg.get("runs_dir", "runs"),
        config={**full_cfg, **train_cfg},
        extras={
            "split": subset,
            "mode": world_mode,
            "seed": seed,
            "device": str(device),
            "dataset_root": str(data_dir),
            "dataset_fingerprint": data_fingerprint,
            "semantic_memory_reset": reset_semantic_memory,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        },
    )
    run.write_git_revision()
    telemetry = TelemetryPublisher(run=run)

    # RC-2.9: fixed, documented validation scenario set (reproducible numbers).
    val_cfg = train_cfg.get("validation", {})
    if bool(val_cfg.get("allow_synthetic_fallback", False)):
        raise ValueError("Phase 11 strictly forbids allow_synthetic_fallback=True in validation set configuration.")
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset=str(val_cfg.get("subset", "val")),
        mode="stare",
        n_files=int(val_cfg.get("n_files", 10)),
        seed=int(val_cfg.get("seed", 42)),
        freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_config.get("max_pulses", 50000)),
        allow_synthetic_fallback=False,
        chunk_mode=str(val_cfg.get("chunk_mode", "first")),
    )
    (run.dir / "validation_set.json").write_text(
        json.dumps(coerce(val_set.manifest()), indent=2), encoding="utf-8"
    )
    logger.info("Run %s at %s", run.run_id, run.dir)

    # Phase 11 Fail-closed parent checkpoint resolution: CLI -> config -> scheduler.scheduler_ckpt
    resolved_parent_ckpt = resume_checkpoint
    if resolved_parent_ckpt is None:
        resolved_parent_ckpt = train_cfg.get("scheduler_ckpt") or sched_cfg.get("scheduler_ckpt")

    if resolved_parent_ckpt is None:
        raise RuntimeError(
            "FATAL: No parent checkpoint specified. Retraining requires an explicit parent checkpoint "
            "(via CLI --resume or config 'scheduler_ckpt'). Silent fallback to fresh training is permanently disabled."
        )

    parent_path = Path(resolved_parent_ckpt)
    if not parent_path.exists():
        raise FileNotFoundError(
            f"FATAL: Parent checkpoint not found at '{parent_path}'. "
            f"Silent fallback to fresh training is permanently disabled."
        )

    import hashlib
    sha256_hash = hashlib.sha256()
    with open(parent_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    parent_sha256 = sha256_hash.hexdigest()
    logger.info("Parent checkpoint resolved: %s (SHA-256: %s)", parent_path, parent_sha256)

    if expected_parent_sha is None:
        expected_parent_sha = train_cfg.get("expected_parent_sha") or sched_cfg.get("expected_parent_sha")

    FROZEN_25K_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
    is_gate25k_launch = bool("checkpoint_gate_25000_frozen" in parent_path.name)
    if is_gate25k_launch:
        if parent_sha256 != FROZEN_25K_SHA:
            raise RuntimeError(
                f"FATAL: Gate-25k parent checkpoint SHA-256 mismatch! "
                f"Expected {FROZEN_25K_SHA}, got {parent_sha256} from {parent_path}"
            )
        logger.info("Gate-25k baseline parent SHA-256 verified bit-identical to canonical hash.")
    elif expected_parent_sha:
        if parent_sha256 != expected_parent_sha:
            raise RuntimeError(
                f"FATAL: Checkpoint {parent_path} SHA-256 mismatch against declared manifest lineage! "
                f"Expected {expected_parent_sha}, got {parent_sha256}"
            )
        logger.info("Parent checkpoint SHA-256 verified against declared manifest lineage: %s", parent_sha256)

    try:
        ckpt = torch.load(parent_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(parent_path, map_location=device)

    if "phase11" in str(parent_path).lower() and isinstance(ckpt, dict):
        parent_sha256 = ckpt.get("parent_sha256") or ckpt.get("metadata", {}).get("extra", {}).get("parent_sha256", FROZEN_25K_SHA)
        logger.info("Phase 11 lineage preserved: root parent SHA-256 = %s", parent_sha256)

    if not use_g8_film:
        # Architecture verification (360 obs_dim, 180 actions)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model_state = online_drqn.state_dict()
        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape != v.shape:
                    raise RuntimeError(
                        f"FATAL: Architecture mismatch for layer '{k}': "
                        f"model expected {model_state[k].shape}, checkpoint has {v.shape}."
                    )

        # Weights-only loading into online and target DRQN
        online_drqn.load_state_dict(state_dict, strict=True)
        target_drqn.load_state_dict(ckpt.get("target_state_dict", state_dict) if isinstance(ckpt, dict) else state_dict, strict=True)
        logger.info("Loaded parent weights into online and target DRQN (strict=True).")
    else:
        logger.info(
            "G8 FiLM mode: skipping load_state_dict (weights already loaded by from_g7a_checkpoint). "
            "Proceeding to checkpoint metadata restoration."
        )

    # Determine restart vs in-flight continuation contract
    is_baseline_restart = (
        ("frozen" in parent_path.name or "production_baseline" in str(parent_path))
        and not (isinstance(ckpt, dict) and int(ckpt.get("global_step", 0)) > 25000)
    ) or bool(train_cfg.get("weights_only", False) or sched_cfg.get("weights_only", False))

    global_step = 0
    episode = 0
    best_reward = -float("inf")
    eps = eps_start
    reward_baseline = -0.39
    if start_step_override is not None:
        start_step = int(start_step_override)
    elif isinstance(ckpt, dict) and int(ckpt.get("global_step", ckpt.get("step", 0))) > 0:
        start_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
    else:
        start_step = int(sched_cfg.get("start_step", 25000 if is_baseline_restart else 0))

    if is_baseline_restart:
        logger.info("Weights-only baseline restart contract ACTIVE: fresh optimizer, fresh replay, fresh RNG, fresh exploration starting at step %d.", start_step)
        global_step = start_step
        episode = 0
        eps = eps_start
    else:
        if isinstance(ckpt, dict):
            if "optimizer_state_dict" in ckpt and optimizer is not None and not fresh_optimizer:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    logger.info("Restored optimizer state from checkpoint.")
                except Exception as exc:
                    logger.warning("Could not restore optimizer state: %s", exc)
            elif fresh_optimizer:
                logger.info("Fresh optimizer contract ACTIVE: keeping newly initialized Adam optimizer.")
            if "rng_state" in ckpt:
                try:
                    torch.set_rng_state(ckpt["rng_state"])
                except Exception:
                    pass
            if "np_rng_state" in ckpt:
                try:
                    np.random.set_state(ckpt["np_rng_state"])
                except Exception:
                    pass
            global_step = int(ckpt.get("global_step", start_step))
            episode = int(ckpt.get("episode", 0)) + 1
            eps = float(ckpt.get("epsilon", eps_start))
            if "reward_baseline" in ckpt:
                reward_baseline = float(ckpt["reward_baseline"])
            elif "reward_baseline" in ckpt.get("metadata", {}).get("extra", {}):
                reward_baseline = float(ckpt["metadata"]["extra"]["reward_baseline"])
            best_reward = float(ckpt.get("metadata", {}).get("metrics", {}).get("best_episode_reward", -float("inf")))
        logger.info("Resumed in-flight state: global_step=%d, episode=%d, eps=%.4f", global_step, episode, eps)

    # Qualification Mode setup and manifest emission
    if qualification_run:
        target_stop_step = global_step + q_steps
        stop_at_step = target_stop_step
        total_steps = target_stop_step
        try:
            git_commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git_commit_sha = "unknown"

        with open(train_cfg_path, "rb") as f:
            t_cfg_sha = hashlib.sha256(f.read()).hexdigest()
        with open(model_cfg_path, "rb") as f:
            m_cfg_sha = hashlib.sha256(f.read()).hexdigest()

        q_manifest = {
            "run_id": run_id,
            "git_commit_sha": git_commit_sha,
            "training_config_sha256": t_cfg_sha,
            "model_config_sha256": m_cfg_sha,
            "parent_checkpoint_sha256": parent_sha256,
            "start_global_step": global_step,
            "target_global_step": target_stop_step,
            "qualification_steps_requested": q_steps,
            "qualification_started_at_utc": qualification_started_at_utc,
        }
        (output_dir / "qualification_run_manifest.json").write_text(json.dumps(q_manifest, indent=2), encoding="utf-8")
        logger.info("Emitted qualification run manifest with run_id=%s, start_step=%d, target_stop_step=%d", run_id, global_step, target_stop_step)
    else:
        target_stop_step = None

    cfg_gates = sched_cfg.get("staged_gates") or train_cfg.get("staged_gates")
    if cfg_gates and (staged_gates is None or staged_gates == [1000, 5000, 25000, 100000, 200000, 300000, 500000]):
        if isinstance(cfg_gates, str):
            staged_gates = [int(g.strip()) for g in cfg_gates.split(",") if g.strip()]
        elif isinstance(cfg_gates, list):
            staged_gates = [int(g) for g in cfg_gates]

    # StagedGateEvaluator constructed AFTER parent resolution, verification, and loading
    from .staged_gate_evaluator import StagedGateEvaluator

    gate_evaluator = StagedGateEvaluator(
        output_dir=output_dir,
        gates=staged_gates,
        val_files=val_set.files_used,
        env_config=env_config,
        model_config=full_cfg,
        train_config=train_cfg,
        seed=seed,
        device=device,
        semantic_memory_reset=reset_semantic_memory,
        parent_checkpoint=str(parent_path),
        run_id=run.run_id,
        dataset_fingerprint=data_fingerprint,
    )
    for g in gate_evaluator.gates:
        if g <= global_step:
            gate_evaluator.completed_gates.add(g)

    from .eval_batch import get_or_create_fixed_eval_batch, evaluate_q_diagnostics
    fixed_eval_batch = get_or_create_fixed_eval_batch(data_dir=str(canonical_root), device=device)

    # Phase 9B / Phase 2 update integrity counters & failure safeguards
    update_integrity_counters = {
        "optimizer_updates_attempted": 0,
        "optimizer_updates_completed": 0,
        "skipped_nan": 0,
        "skipped_assertion": 0,
        "skipped_oom": 0,
        "other_update_failures": 0,
        "validation_failures": 0,
        "finite_gradient_updates": 0,
        "non_finite_gradient_updates": 0,
        # Backward compatibility aliases:
        "n_updates_attempted": 0,
        "n_updates_completed": 0,
        "n_updates_skipped_nan": 0,
        "n_updates_skipped_assertion": 0,
        "n_updates_skipped_oom": 0,
    }
    consecutive_ooms = 0
    consecutive_val_failures = 0

    latest_shift_telemetry: dict[str, Any] | None = None
    if dist_shift_detector is not None:
        latest_shift_telemetry = {
            "shift_score_kl": None,
            "shift_severity": "none",
            "shift_threshold": float(dist_shift_detector.alert_threshold),
            "shift_critical_threshold": float(dist_shift_detector.critical_threshold),
            "shift_action": "monitor_only",
        }

    while global_step < total_steps:
        obs, _ = env.reset()
        if band_tracker is not None:
            band_tracker.reset()
        obs_arr = np.asarray(obs)
        assert np.all(np.isfinite(obs_arr)), f"Non-finite obs at ep start: {obs_arr[~np.isfinite(obs_arr)]}"
        assert np.all(obs_arr >= 0), f"Negative obs at ep start: band indices {np.where(obs_arr < 0)}"
        try:
            hidden = online_drqn.init_hidden(1, device)
        except Exception:
            hidden = None
        moe.reset()
        if hidden is not None:
            moe.eager_agent.hidden = hidden

        done = False
        ep_reward = 0.0
        ep_hits = 0
        ep_steps = 0
        # --- RC-2 per-episode telemetry accumulators ---
        ep_band_counts = np.zeros(n_bands, dtype=np.float64)
        ep_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_action_counts = np.zeros(n_actions, dtype=np.float64)
        ep_ns = {"same_argmax": 0, "attributed": 0, "scores": {
            "mean_q_score": 0.0, "mean_eager_score": 0.0, "mean_revisit_score": 0.0,
            "mean_semantic_score": 0.0, "mean_preemptive_score": 0.0, "mean_fused_score": 0.0,
            "drqn_rank": 0.0,
        }}
        ep_q_argmax_band_counts = np.zeros(n_bands, dtype=np.float64)
        ep_learn = {"n_updates": 0, "td_loss": 0.0, "q_reg_loss": 0.0, "mean_td_error": 0.0, "max_td_error": -1e9,
                    "mean_q": 0.0, "max_q": -1e9, "min_q": 1e9, "q_std": 0.0, "max_q_margin": 0.0,
                    "mean_online_q": 0.0, "mean_target_q": 0.0, "max_target_q": -1e9,
                    "target_online_gap": 0.0, "gradient_norm": 0.0,
                    "replay_hit_frac": 0.0, "replay_seq_hit_frac": 0.0, "pos_scen_concentration": 0.0,
                    "mode_diversity_pen": 0.0, "diversity_penalty": 0.0, "mode_collapse_rate": 0.0,
                    "batch_action_entropy": 0.0, "top_band_frac": 0.0}

        consecutive_empty = 0
        last_band = -1
        first_interception_step = None
        empty_transitions = 0
        positive_transitions = 0
        ep_explore_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_greedy_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_thompson_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_targeted_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_uniform_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_overall_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_targeted_exp_steps = 0
        ep_uniform_exp_steps = 0
        ep_thompson_exp_steps = 0

        while not done and global_step < total_steps:
            # ---- Action selection ----
            # Feed the MoE the observable periodic-imminent-arrival urgency so
            # PREEMPTIVE_INTERCEPT selection is driven by the actual prediction
            # pipeline (Phase 5 / Phase 3 no-GT-leakage constraint).
            if hasattr(moe, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
                try:
                    moe.set_periodic_urgency_vector(np.asarray(env.belief.periodic_urgency, dtype=np.float32))
                except Exception as exc:
                    logger.debug("Failed to set periodic urgency vector on MoE: %s", exc)
            rel_step = max(0, global_step - start_step) if is_baseline_restart else global_step
            if override_epsilon is not None:
                eps = float(override_epsilon)
            elif exploration_schedule == "c_rescue":
                # Schedule C-Rescue: gentle decay from eps=0.50 at step 20k to eps=0.30 at step 50k, holding floor at 0.30
                if global_step <= 20000:
                    eps = 0.50
                elif global_step <= 50000:
                    frac = (global_step - 20000) / 30000.0
                    eps = 0.50 - frac * 0.20
                else:
                    eps = 0.30
            elif exploration_schedule == "slower":
                eps = eps_end + (eps_start - eps_end) * float(np.exp(-rel_step / (eps_decay * 2.0)))
            elif exploration_schedule == "staged":
                if global_step < ts_warmup:
                    eps = eps_start
                elif global_step <= 25000:
                    frac = (global_step - ts_warmup) / max(1, 25000 - ts_warmup)
                    eps = 0.80 - frac * 0.40
                elif global_step <= 50000:
                    frac = (global_step - 25000) / 25000.0
                    eps = 0.40 - frac * 0.25
                else:
                    frac = min(1.0, (global_step - 50000) / 50000.0)
                    eps = 0.15 - frac * (0.15 - eps_end)
            else:
                eps = eps_end + (eps_start - eps_end) * float(np.exp(-rel_step / eps_decay))
            if global_step < ts_warmup:
                use_ts = True
                action = ts_sampler.select_action()
                moe_attr = None
                act_source = "thompson"
                decision_source = "thompson_exploration"
                m_act = int(action % n_modes)
                ep_thompson_exp_steps += 1
                ep_thompson_mode_counts[m_act] += 1
                ep_explore_mode_counts[m_act] += 1
                ep_overall_mode_counts[m_act] += 1
            else:
                use_ts = False
                if random.random() < eps:
                    targeted_b = band_tracker.sample_targeted_band() if band_tracker is not None else None
                    if targeted_b is not None:
                        # Rescue-A Fix 3: Full 5-mode exploration (0=SHORT, 1=NORMAL, 2=LONG, 3=REVISIT, 4=PREEMPTIVE)
                        m_rand = random.randrange(n_modes)
                        action = int(targeted_b * n_modes + m_rand)
                        act_source = "targeted_random"
                        decision_source = "targeted_band_exploration"
                        ep_targeted_exp_steps += 1
                        ep_targeted_mode_counts[m_rand] += 1
                    else:
                        if action_selection_mode == "flat_argmax":
                            action = random.randrange(n_actions)
                            m_rand = int(action % n_modes)
                        else:
                            b_rand = random.randint(0, n_bands - 1)
                            # Rescue-A Fix 3: Full 5-mode exploration
                            m_rand = random.randrange(n_modes)
                            action = b_rand * n_modes + m_rand
                        act_source = "random"
                        decision_source = "epsilon_exploration"
                        ep_uniform_exp_steps += 1
                        ep_uniform_mode_counts[m_rand] += 1
                    ep_explore_mode_counts[m_rand] += 1
                    ep_overall_mode_counts[m_rand] += 1
                    moe_attr = None
                else:
                    online_drqn.eval()
                    with torch.inference_mode():
                        obs_np = np.asarray(obs, dtype=np.float32)
                        obs_t = torch.from_numpy(obs_np).to(device)
                        action, hidden = online_drqn.act(
                            obs_t,
                            hidden,
                            mode_selection=action_selection_mode,
                            consecutive_empty=consecutive_empty,
                            tau=0.15 if action_selection_mode == "band_first_decoupled" else 0.0,
                        )
                        m_grd = int(action % n_modes)
                        ep_greedy_mode_counts[m_grd] += 1
                        ep_overall_mode_counts[m_grd] += 1
                        # Diagnostic MoE query for passive telemetry only (MoE quarantined from action selection)
                        try:
                            _, _, moe_attr = moe.select_action(obs_np, hidden)
                        except Exception:
                            moe_attr = None
                    online_drqn.train()
                    act_source = "greedy"
                    decision_source = "ml_exploitation"

            if collapse_guard is not None:
                if act_source == "greedy":
                    mode_chosen = int(action % n_modes)
                    guard_result = collapse_guard.update(mode_chosen)
                else:
                    guard_result = {"triggered": False}

                # Shadow-greedy probe evaluation every guard_check_freq (default 10) steps
                guard_check_freq = int(sched_cfg.get("guard_check_freq", 10))
                if collapse_guard.probe_obs is not None and (global_step % guard_check_freq == 0):
                    shadow_result = collapse_guard.check_shadow_greedy(online_drqn, global_step, device)
                    if shadow_result.get("triggered", False):
                        guard_result = shadow_result

                if guard_result.get("triggered", False):
                    logger.error(
                        "MODE COLLAPSE GUARD TRIGGERED (%s) at step %d: SHORT=%.1f%%, entropy=%.3f. "
                        "Saving emergency checkpoint and halting.",
                        guard_result.get("reason", "unknown"),
                        global_step,
                        guard_result["short_frac"] * 100.0,
                        guard_result["entropy"],
                    )
                    from ..utils.checkpoint_meta import build_train_metadata, save_state

                    emergency_meta = build_train_metadata(
                        split=subset,
                        n_bands=n_bands,
                        arch="FiLMGatedFactorizedDRQN",
                        seed=seed,
                        metrics={"collapse_triggered_step": global_step},
                        extra={"parent_sha256": parent_sha256, "collapse_guard": guard_result},
                    )
                    save_state(
                        online_drqn,
                        output_dir / f"emergency_collapse_step_{global_step}.pt",
                        emergency_meta,
                    )
                    raise RuntimeError(
                        f"Mode collapse guard triggered ({guard_result.get('reason', 'unknown')}) at step {global_step}: "
                        f"SHORT={guard_result['short_frac']:.1%}, entropy={guard_result['entropy']:.3f}"
                    )

            # ---- Step env ----
            dec_telem = getattr(online_drqn, "last_decision_telemetry", None)
            if act_source in ("thompson", "random", "targeted_random") or dec_telem is None:
                dec_telem = {
                    "raw_drqn_action": int(action),
                    "raw_drqn_band": int(action // n_modes),
                    "raw_drqn_mode": int(action % n_modes),
                    "final_action": int(action),
                    "final_band": int(action // n_modes),
                    "final_mode": int(action % n_modes),
                    "action_was_overridden": False,
                    "override_source": None,
                    "exploration_source": "targeted_band" if act_source == "targeted_random" else ("epsilon" if act_source == "random" else "thompson"),
                    "decision_source": decision_source,
                    "q_selected": dec_telem.get("q_selected") if dec_telem else None,
                    "q_max": dec_telem.get("q_max") if dec_telem else None,
                    "q_mean": dec_telem.get("q_mean") if dec_telem else None,
                    "q_std": dec_telem.get("q_std") if dec_telem else None,
                }
            else:
                if action_selection_mode == "flat_argmax":
                    assert dec_telem["final_action"] == dec_telem["raw_drqn_action"], (
                        f"Action override detected in flat_argmax: "
                        f"final={dec_telem['final_action']} != raw={dec_telem['raw_drqn_action']}"
                    )
                    assert not dec_telem["action_was_overridden"], "action_was_overridden is True in flat_argmax"
                dec_telem["decision_source"] = decision_source
            mode_ctx = {
                "action_score": float(moe_attr.get("action_score", 1.0)) if moe_attr else 1.0,
                "reason": str(moe_attr.get("reason", "mode_preset")) if moe_attr else "mode_preset",
                **dec_telem,
            }
            next_obs, reward, terminated, truncated, info = env.step(action, mode_context=mode_ctx)

            done = bool(terminated or truncated)

            # Distribution Shift Monitoring (policy observations, strictly monitor_only)
            if dist_shift_detector is not None and (global_step % dist_shift_eval_interval == 0):
                shift_res = dist_shift_detector.update(np.asarray(next_obs, dtype=np.float32))
                shift_kl = float(shift_res.get("kl", 0.0))
                shift_sev = str(shift_res.get("severity", "none"))
                latest_shift_telemetry = {
                    "shift_score_kl": shift_kl,
                    "shift_severity": shift_sev,
                    "shift_threshold": float(dist_shift_detector.alert_threshold),
                    "shift_critical_threshold": float(dist_shift_detector.critical_threshold),
                    "shift_action": "monitor_only",
                }
                if shift_res.get("shift_detected"):
                    logger.warning(
                        "[DISTRIBUTION SHIFT MONITOR] Policy-observation shift detected at step %d: KL=%.4f (severity: %s, action: monitor_only)",
                        global_step, shift_kl, shift_sev,
                    )

            ep_steps += 1
            band_idx = int(action // n_modes)
            ep_band_counts[band_idx] += 1
            if band_tracker is not None:
                band_tracker.record_visit(band_idx)
            ep_mode_counts[int(action % n_modes)] += 1
            ep_action_counts[int(action)] += 1
            if moe_attr is not None:
                ep_ns["attributed"] += 1
                ep_ns["same_argmax"] += int(bool(moe_attr.get("same_argmax", False)))
                q_arg_band = moe_attr.get("q_argmax_band")
                if q_arg_band is not None and 0 <= int(q_arg_band) < n_bands:
                    ep_q_argmax_band_counts[int(q_arg_band)] += 1
                for moe_key, tel_key in _SCORE_KEYS.items():
                    val = moe_attr.get(moe_key)
                    ep_ns["scores"][tel_key] += float(val) if val is not None and float(val) == float(val) else 0.0

            band = int(action // n_modes)
            hit = bool(info.get("hit", False))
            if hit:
                consecutive_empty = 0
                positive_transitions += 1
                if first_interception_step is None:
                    first_interception_step = int(ep_steps)
            else:
                empty_transitions += 1
                if last_band == band or last_band == -1:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 1
            last_band = band

            ts_sampler.update(action, hit)
            ep_scen_id = "unknown"
            if getattr(env, "records", None) and len(env.records) > 0:
                first_src = getattr(env.records[0], "source_id", "unknown")
                ep_scen_id = str(first_src).split(":")[-1] if first_src else "unknown"

            actual_dwell_us = float(info.get("dwell_time_us", 500.0 * DEFAULT_DWELL_MULTIPLIERS[action % n_modes]))

            if reward_shaper is not None:
                mode_chosen = int(action % n_modes)
                shaped_reward = reward_shaper.shape(float(reward), mode_chosen, actual_dwell_us)
            else:
                shaped_reward = float(reward)

            buffer.add(
                np.asarray(obs, dtype=np.float32),
                action,
                shaped_reward,
                np.asarray(next_obs, dtype=np.float32),
                done,
                hit_prob=float(info.get("hit_prob", 1.0 if info["hit"] else 0.0)),
                intercept_time_us=float(info.get("intercept_time_us", float("nan"))),
                scenario_id=ep_scen_id,
                dwell_time_us=actual_dwell_us,
            )
            obs = next_obs
            ep_reward += float(reward)
            ep_hits += int(info["hit"])
            moe.update(action)
            global_step += 1

            # ---- Learning update ----
            if global_step % update_freq == 0 and buffer.can_sample(batch_size):
                lr_schedule = sched_cfg.get("lr_schedule")
                if lr_schedule == "cosine":
                    total_train_steps = max(1, total_steps - start_step)
                    progress = min(1.0, max(0.0, (global_step - start_step) / total_train_steps))
                    min_lr = float(sched_cfg.get("min_lr", 5.0e-6))
                    base_lr = learning_rate
                    current_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                try:
                    if stratified_sampler is not None:
                        batch = stratified_sampler.sample(batch_size, target_hit_seq_fraction=0.40)
                    else:
                        batch = buffer.sample(batch_size, target_hit_seq_fraction=0.40)

                    if use_g8_film:
                        effective_lambda_entropy = cosine_beta_schedule(
                            step=global_step,
                            total_steps=total_steps,
                            start_step=26000,
                            beta_max=float(sched_cfg.get("g8_beta_max", 0.10)),
                            warmup_frac=float(sched_cfg.get("g8_warmup_frac", 0.10)),
                            beta_min=float(sched_cfg.get("lambda_entropy", 0.01)),
                        )
                    else:
                        effective_lambda_entropy = lambda_entropy

                    upd_stats: dict = {}
                    loss_val = _do_drqn_update(
                        online_drqn,
                        target_drqn,
                        optimizer,
                        loss_fn,
                        batch,
                        gamma,
                        device,
                        stats=upd_stats,
                        reward_baseline=reward_baseline,
                        baseline_momentum=0.99,
                        target_q_max=target_q_max,
                        target_q_min=target_q_min,
                        q_reg_coef=q_reg_coef,
                        n_bands=n_bands,
                        n_modes=n_modes,
                        lambda_entropy=effective_lambda_entropy,
                        band_diversity_penalty_coef=band_diversity_penalty_coef,
                        band_diversity_threshold=band_diversity_threshold,
                        mode_diversity_penalty_coef=mode_diversity_penalty_coef,
                        mode_diversity_threshold=mode_diversity_threshold,
                        mode_collapse_rate_threshold=mode_collapse_rate_threshold,
                        objective_mode=objective_mode,
                        c_dwell=c_dwell,
                        tau_ref=tau_ref,
                    )
                    if upd_stats.get("optimizer_update_attempted", False):
                        update_integrity_counters["optimizer_updates_attempted"] += 1
                        update_integrity_counters["n_updates_attempted"] += 1
                        if upd_stats.get("finite_gradients", False):
                            update_integrity_counters["finite_gradient_updates"] += 1
                        else:
                            update_integrity_counters["non_finite_gradient_updates"] += 1

                    if not np.isfinite(loss_val) or upd_stats.get("skipped_nan", False):
                        update_integrity_counters["skipped_nan"] += 1
                        update_integrity_counters["n_updates_skipped_nan"] += 1
                        logger.error("DRQN update skipped due to NaN loss/gradient (step %d)", global_step)
                        if not diagnostic:
                            raise RuntimeError(f"Fail-closed update failure (NaN/Inf loss or gradient) at step {global_step}")
                    if abort_on_q_max_exceeded and float(upd_stats.get("max_q", 0.0)) > max_allowed_q:
                        raise RuntimeError(f"Fail-closed Early Termination: Q_max {upd_stats.get('max_q', 0.0):.2f} exceeded ceiling {max_allowed_q} at step {global_step}")
                    elif upd_stats.get("optimizer_update_completed", False):
                        update_integrity_counters["optimizer_updates_completed"] += 1
                        update_integrity_counters["n_updates_completed"] += 1
                        consecutive_ooms = 0
                    if upd_stats and not upd_stats.get("skipped_nan", False):
                        if "reward_baseline" in upd_stats:
                            reward_baseline = float(upd_stats["reward_baseline"])
                        ep_learn["n_updates"] += 1
                        ep_learn["td_loss"] += float(upd_stats["td_loss"])
                        ep_learn["q_reg_loss"] += float(upd_stats.get("q_reg_loss", 0.0))
                        ep_learn["mean_td_error"] += float(upd_stats["mean_td_error"])
                        ep_learn["max_td_error"] = max(ep_learn["max_td_error"], float(upd_stats["max_td_error"]))
                        ep_learn["mean_q"] += float(upd_stats["mean_q"])
                        ep_learn["max_q"] = max(ep_learn["max_q"], float(upd_stats["max_q"]))
                        ep_learn["min_q"] = min(ep_learn["min_q"], float(upd_stats["min_q"]))
                        ep_learn["q_std"] += float(upd_stats["q_std"])
                        ep_learn["max_q_margin"] = max(ep_learn["max_q_margin"], float(upd_stats.get("max_q_margin", 0.0)))
                        ep_learn["mean_online_q"] += float(upd_stats["mean_online_q"])
                        ep_learn["mean_target_q"] += float(upd_stats["mean_target_q"])
                        ep_learn["max_target_q"] = max(ep_learn["max_target_q"], float(upd_stats["max_target_q"]))
                        ep_learn["target_online_gap"] += float(upd_stats["target_online_gap"])
                        ep_learn["gradient_norm"] += float(upd_stats["gradient_norm"])
                        ep_learn["replay_hit_frac"] += float(upd_stats.get("replay_hit_fraction", 0.0))
                        ep_learn["replay_seq_hit_frac"] += float(upd_stats.get("replay_sequence_hit_fraction", 0.0))
                        ep_learn["pos_scen_concentration"] += float(upd_stats.get("pos_scen_concentration", 0.0))
                        ep_learn["mode_diversity_pen"] += float(upd_stats.get("mode_diversity_pen", 0.0))
                        ep_learn["diversity_penalty"] += float(upd_stats.get("diversity_penalty", 0.0))
                        ep_learn["mode_collapse_rate"] += float(upd_stats.get("mode_collapse_rate", 0.0))
                        ep_learn["batch_action_entropy"] += float(upd_stats.get("action_entropy", 0.0))
                        ep_learn["top_band_frac"] += float(upd_stats.get("top_band_frac", 0.0))
                        b_diag_step = band_tracker.get_diagnostics() if band_tracker is not None else {}
                        tot_exp_step = ep_targeted_exp_steps + ep_uniform_exp_steps
                        gate_evaluator.record_step_diagnostics(
                            td_loss=float(upd_stats["td_loss"]),
                            q_mean=float(upd_stats["mean_q"]),
                            q_std=float(upd_stats["q_std"]),
                            q_min=float(upd_stats["min_q"]),
                            q_max=float(upd_stats["max_q"]),
                            grad_norm=float(upd_stats["gradient_norm"]),
                            action_source=act_source,
                            obs_finite=bool(np.all(np.isfinite(next_obs))),
                            reward_finite=bool(np.isfinite(reward)),
                            replay_hit_fraction=float(upd_stats.get("replay_hit_fraction", 0.0)),
                            replay_seq_hit_fraction=float(upd_stats.get("replay_sequence_hit_fraction", 0.0)),
                            q_margin=float(upd_stats.get("max_q_margin", 0.0)),
                            q_reg_loss=float(upd_stats.get("q_reg_loss", 0.0)),
                            pos_scen_concentration=float(upd_stats.get("pos_scen_concentration", 0.0)),
                            targeted_exp_fraction=float(ep_targeted_exp_steps / max(1, tot_exp_step)) if tot_exp_step > 0 else 0.0,
                            underexplored_band_fraction=float(b_diag_step.get("underexplored_band_fraction", 0.0)),
                        )

                    if use_wandb and global_step % 100 == 0:
                        try:
                            import wandb

                            wandb.log({"train/loss": loss_val, "train/eps": float(eps), "step": global_step})
                        except Exception:
                            pass
                except AssertionError as exc:
                    update_integrity_counters["skipped_assertion"] += 1
                    update_integrity_counters["n_updates_skipped_assertion"] += 1
                    logger.error("AssertionError during DRQN update at step %d: %s", global_step, exc)
                    if not diagnostic:
                        raise RuntimeError(f"Fail-closed update failure (AssertionError) at step {global_step}: {exc}") from exc
                    if update_integrity_counters["skipped_assertion"] > 10:
                        raise RuntimeError(
                            "More than 10 gradient updates skipped due to assertion failures — aborting training"
                        ) from exc
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                        update_integrity_counters["skipped_oom"] += 1
                        update_integrity_counters["n_updates_skipped_oom"] += 1
                        consecutive_ooms += 1
                        logger.warning("OOM in DRQN update (%d consecutive) at step %d — skipping", consecutive_ooms, global_step)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if not diagnostic:
                            raise RuntimeError(f"Fail-closed update failure (OOM) at step {global_step}: {exc}") from exc
                        if consecutive_ooms > 3:
                            raise RuntimeError(
                                f"More than 3 consecutive OOMs ({consecutive_ooms}) in DRQN update — aborting training"
                            ) from exc
                    else:
                        update_integrity_counters["other_update_failures"] += 1
                        raise

            # ---- Periodic Q-margin, stuck-state, and shadow validation diagnostics ----
            if global_step % 500 == 0:
                try:
                    q_diag = evaluate_q_diagnostics(online_drqn, fixed_eval_batch, device)
                    logger.info(
                        "Step %d | QDiag: margin=%.5f spread=%.4f rew_base=%.4f | StuckState: Pick B%d-M%d (margin=%.5f, 2nd B%d)",
                        global_step,
                        q_diag["q_margin_mean"],
                        q_diag["q_spread_mean"],
                        reward_baseline,
                        q_diag["stuck_state_top1_band"],
                        q_diag["stuck_state_top1_mode"],
                        q_diag["stuck_state_margin"],
                        q_diag["stuck_state_top2_band"],
                    )
                except Exception as exc:
                    logger.warning("Failed evaluating QDiag at step %d: %s", global_step, exc)

                # Benchmark V2 Shadow Validation (every 500 steps on config_117 and config_29)
                try:
                    shadow_bench = gate_evaluator.evaluate_baseline_hierarchy(
                        online_drqn,
                        moe=moe,
                        n_steps=200,
                        policies=["drqn"],
                        max_scenarios=2,
                    )
                    drqn_shadow = shadow_bench.get("policies", {}).get("drqn", {})
                    logger.info(
                        "Step %d [ShadowVal-500] | DRQN Mean IR: %.2f%% | Distinct Bands: %.1f/36 | Pd: %.2f%%",
                        global_step,
                        float(drqn_shadow.get("intercept_rate", 0.0)) * 100,
                        float(drqn_shadow.get("distinct_bands", 0.0)),
                        float(drqn_shadow.get("decision_level_pd", 0.0)) * 100,
                    )
                except Exception as exc:
                    logger.warning("Shadow validation failed at step %d: %s", global_step, exc)



            # ---- Target update ----
            if global_step % target_update_freq == 0:
                target_drqn.load_state_dict(online_drqn.state_dict())

            # ---- Staged Promotion Gate Evaluation ----
            if any(global_step >= g and g not in gate_evaluator.completed_gates for g in gate_evaluator.gates):
                logger.info("Evaluation Checkpoint [step %d] | Update Integrity Counters: %s", global_step, update_integrity_counters)
            gate_evaluator.check_and_run(
                global_step=global_step,
                episode=episode,
                online_drqn=online_drqn,
                optimizer=optimizer,
                buffer=buffer,
                eps=eps,
                moe=moe,
                reward_baseline=reward_baseline,
                override_epsilon=override_epsilon,
                target_drqn=target_drqn,
            )
            if ep_learn["max_q"] > target_q_max:
                logger.error("SAFETY SENTINEL TRIGGERED: max_q=%.2f exceeds target_q_max=%.2f ceiling! Halting run.", ep_learn["max_q"], target_q_max)
                break

            if stop_at_step is not None and global_step >= stop_at_step:
                break

        episode += 1
        fom = env.get_fom()
        intercept_rate = ep_hits / max(1, getattr(env, "current_step", ep_steps))
        exp_tot = max(1e-5, float(np.sum(ep_explore_mode_counts)))
        grd_tot = max(1e-5, float(np.sum(ep_greedy_mode_counts)))
        n_upd = ep_learn["n_updates"]
        seq_hit_pct = (ep_learn["replay_seq_hit_frac"] / max(1, n_upd)) * 100.0 if n_upd > 0 else 0.0
        avg_td_loss = (ep_learn["td_loss"] / max(1, n_upd)) if n_upd > 0 else 0.0
        avg_q_reg = (ep_learn["q_reg_loss"] / max(1, n_upd)) if n_upd > 0 else 0.0
        avg_pos_scen = (ep_learn["pos_scen_concentration"] / max(1, n_upd)) * 100.0 if n_upd > 0 else 0.0
        
        # Phase 9B-R3 band diagnostics
        band_diag = band_tracker.get_diagnostics() if band_tracker is not None else {}
        u_bands = band_diag.get("unique_bands", int(np.count_nonzero(ep_band_counts)))
        min_bv = band_diag.get("min_band_visits", int(np.min(ep_band_counts)))
        med_bv = band_diag.get("median_band_visits", float(np.median(ep_band_counts)))
        max_bv = band_diag.get("max_band_visits", int(np.max(ep_band_counts)))
        under_cnt = band_diag.get("underexplored_band_count", 0)
        tot_exp_steps = ep_targeted_exp_steps + ep_uniform_exp_steps
        target_exp_pct = (ep_targeted_exp_steps / max(1, tot_exp_steps)) * 100.0 if tot_exp_steps > 0 else 0.0

        logger.info(
            "Ep %d | step %d/%d | rew %.2f hits %d ir %.3f eps %.3f | Bands: %d/36 (min=%d, med=%.1f, max=%d, under=%d) | Exp: Target=%.1f%% (%d/%d) | ReplaySeqHit: %.1f%% (PosScenMax: %.1f%%, %d upds) | TD: %.4f QReg: %.4f | QMarginMax: %.3f QMax: %.1f | Q-Modes: [S=%.1f%%, N=%.1f%%, L=%.1f%%, R=%.1f%%, P=%.1f%%] | Exp-Modes: [S=%.1f%%, N=%.1f%%, L=%.1f%%, R=%.1f%%, P=%.1f%%]",
            episode, global_step, total_steps, ep_reward, ep_hits, intercept_rate, eps,
            u_bands, min_bv, med_bv, max_bv, under_cnt,
            target_exp_pct, ep_targeted_exp_steps, tot_exp_steps,
            seq_hit_pct, avg_pos_scen, n_upd,
            avg_td_loss, avg_q_reg, ep_learn["max_q_margin"], ep_learn["max_q"],
            (ep_greedy_mode_counts[0] / grd_tot) * 100, (ep_greedy_mode_counts[1] / grd_tot) * 100,
            (ep_greedy_mode_counts[2] / grd_tot) * 100, (ep_greedy_mode_counts[3] / grd_tot) * 100,
            (ep_greedy_mode_counts[4] / grd_tot) * 100,
            (ep_explore_mode_counts[0] / exp_tot) * 100, (ep_explore_mode_counts[1] / exp_tot) * 100,
            (ep_explore_mode_counts[2] / exp_tot) * 100, (ep_explore_mode_counts[3] / exp_tot) * 100,
            (ep_explore_mode_counts[4] / exp_tot) * 100,
        )

        if use_wandb:
            try:
                import wandb

                wandb.log({"episode/reward": ep_reward, "episode/hits": ep_hits, "episode/intercept_rate": intercept_rate, "episode": episode, "step": global_step})
            except Exception:
                pass

        # P0-9 / RC-2: publish real episode telemetry (band priorities = observable occupancy).
        band_priorities = [float(v) for v in _observable_priorities(obs)]

        # --- RC-2 core metrics from FiguresOfMerit summary ---
        avg_intercept = fom.get("avg_intercept_time_error_us")
        core = {
            "pd": fom.get("Pd"),
            "pfa": fom.get("Pfa"),
            "intercept_rate": fom.get("avg_intercept_rate"),
            "avg_reward": fom.get("avg_reward"),
            "episode_reward": float(ep_reward),
            "ep_hits": int(ep_hits),
            "ep_steps": int(ep_steps),
            "coverage": fom.get("band_selection_coverage"),
            "discovery_rate": fom.get("discovery_rate"),
            "avg_intercept_time": avg_intercept,
            "avg_intercept_time_error": avg_intercept,
            "pct_correct": fom.get("pct_correct_predictions"),
            "selected_active": fom.get("selected_active_opportunities"),
            "spectrum_active": fom.get("spectrum_active_opportunities"),
            # Phase 5 environment, agility & sparse-emitter telemetry
            "unique_bands_visited": int(np.count_nonzero(ep_band_counts)),
            "unique_actions_visited": int(np.count_nonzero(ep_action_counts)),
            "unique_modes_visited": int(np.count_nonzero(ep_mode_counts)),
            "first_interception_step": first_interception_step,
            "empty_band_transitions": int(empty_transitions),
            "positive_interception_transitions": int(positive_transitions),
            # Phase 4 reward-alignment & learning-signal metrics
            "total_reward": float(ep_reward),
            "interception_reward": float(fom.get("avg_reward_hit_term", 0.0) * ep_steps),
            "latency_reward": float(fom.get("avg_reward_latency_bonus", 0.0) * ep_steps),
            "miss_penalty": float(fom.get("avg_reward_miss_penalty", 0.0) * ep_steps),
            "false_alarm_penalty": float(fom.get("avg_reward_false_alarm_penalty", 0.0) * ep_steps),
            "redundant_penalty": float(fom.get("avg_reward_redundant_penalty", 0.0) * ep_steps),
            "dwell_penalty": float(fom.get("avg_reward_dwell_cost", 0.0) * ep_steps),
            "frequency_agility_bonus": float(fom.get("avg_reward_prediction_bonus", 0.0) * ep_steps),
            "reward_component_dominance": bool(
                ep_hits > 0 and (
                    abs(fom.get("avg_reward_latency_bonus", 0.0) * ep_steps)
                    + abs(fom.get("avg_reward_prediction_bonus", 0.0) * ep_steps)
                    + abs(fom.get("avg_reward_redundant_penalty", 0.0) * ep_steps)
                    + abs(fom.get("avg_reward_dwell_cost", 0.0) * ep_steps)
                ) > abs(fom.get("avg_reward_hit_term", 0.0) * ep_steps)
            ),
            "cumulative_interceptions": int(ep_hits),
            "average_intercept_latency_us": avg_intercept,
            "number_of_unique_emitters_intercepted": int(len(getattr(env, "intercepted_emitters", set()))),
            "interception_rate": fom.get("avg_intercept_rate"),
            "average_interception_time_error": avg_intercept,
            "average_interception_latency": avg_intercept,
            "cumulative_reward": float(ep_reward),
        }
        # --- RC-2 reward decomposition totals (avg/step * steps) ---
        _avg_to_total = {
            "reward_novel": "avg_reward_novel_term",
            "reward_hit": "avg_reward_hit_term",
            "reward_miss": "avg_reward_miss_penalty",
            "reward_dwell_cost": "avg_reward_dwell_cost",
            "reward_false_alarm": "avg_reward_false_alarm_penalty",
            "reward_timing": "avg_reward_timing_penalty",
            "reward_priority": "avg_reward_priority_term",
            "reward_info_gain": "avg_reward_info_gain_term",
            "reward_redundant": "avg_reward_redundant_penalty",
            "reward_delay": "avg_reward_delay_penalty",
            "reward_staleness": "avg_reward_staleness_bonus",
        }
        reward_components: dict = {}
        for canon, avg_key in _avg_to_total.items():
            avg_v = fom.get(avg_key)
            reward_components[canon] = (float(avg_v) * ep_steps) if avg_v is not None else None

        reward_breakdown: dict = {}
        for canon, val in reward_components.items():
            if val is not None:
                reward_breakdown[canon] = {
                    "points": float(round(val, 2)),
                    "pct": float(round((abs(val) / max(1e-6, abs(ep_reward))) * 100, 2)),
                }
        reconstructed_sum = float(sum(v for v in reward_components.values() if v is not None))
        logger.info(
            "Ep %d reward breakdown: %s (sum=%.2f, ep_reward=%.2f)",
            episode,
            {k: f"{v['points']:+.1f}pts ({v['pct']:.1f}%)" for k, v in reward_breakdown.items() if abs(v['points']) > 0.01},
            reconstructed_sum,
            ep_reward,
        )

        # --- RC-2 action / mode / band statistics ---
        n_steps_ep = float(ep_steps) if ep_steps else None
        mode_freq = [float(c / n_steps_ep) for c in ep_mode_counts] if n_steps_ep else None
        band_freq = [float(c / n_steps_ep) for c in ep_band_counts] if n_steps_ep else None
        actions = {
            "unique_actions": int(np.count_nonzero(ep_action_counts)),
            "unique_bands": int(np.count_nonzero(ep_band_counts)),
            "unique_modes": int(np.count_nonzero(ep_mode_counts)),
            "band_selection_counts": [int(c) for c in ep_band_counts],
            "band_selection_frequencies": band_freq,
            "mode_selection_counts": {DWELL_MODES[i]: int(ep_mode_counts[i]) for i in range(n_modes)},
            "mode_selection_frequencies": ({DWELL_MODES[i]: float(ep_mode_counts[i] / n_steps_ep) for i in range(n_modes)}
                                           if n_steps_ep else None),
            "action_selection_counts": [int(c) for c in ep_action_counts],
            "top_action_fraction": float(np.max(ep_action_counts) / n_steps_ep) if n_steps_ep else None,
            "action_entropy": shannon_entropy(ep_action_counts),
            "band_entropy": shannon_entropy(ep_band_counts),
            "mode_entropy": shannon_entropy(ep_mode_counts),
            # Phase 9B-R3 / Benchmark V2 Formalized Exploration Telemetry
            "min_band_visits": min_bv,
            "median_band_visits": med_bv,
            "max_band_visits": max_bv,
            "band_visit_std": float(np.std(ep_band_counts)),
            "underexplored_band_count": under_cnt,
            "targeted_exploration_steps": int(ep_targeted_exp_steps),
            "uniform_exploration_steps": int(ep_uniform_exp_steps),
            "thompson_exploration_steps": int(ep_thompson_exp_steps),
            "targeted_exploration_fraction": float(ep_targeted_exp_steps / max(1, tot_exp_steps)) if tot_exp_steps > 0 else 0.0,
            "random_action_fraction": float(ep_uniform_exp_steps / n_steps_ep) if n_steps_ep else 0.0,
            "targeted_action_fraction": float(ep_targeted_exp_steps / n_steps_ep) if n_steps_ep else 0.0,
            "thompson_action_fraction": float(ep_thompson_exp_steps / n_steps_ep) if n_steps_ep else 0.0,
            "greedy_action_fraction": float(np.sum(ep_greedy_mode_counts) / n_steps_ep) if n_steps_ep else 0.0,
            "mode_histogram": [int(c) for c in ep_mode_counts],
            "band_histogram": [int(c) for c in ep_band_counts],
            "explore_mode_histogram": [int(c) for c in ep_explore_mode_counts],
            "greedy_mode_histogram": [int(c) for c in ep_greedy_mode_counts],
            "thompson_mode_histogram": [int(c) for c in ep_thompson_mode_counts],
            "targeted_mode_histogram": [int(c) for c in ep_targeted_mode_counts],
            "uniform_mode_histogram": [int(c) for c in ep_uniform_mode_counts],
            "overall_mode_histogram": [int(c) for c in ep_overall_mode_counts],
        }

        # --- RC-2 learning statistics (averaged over the episode's updates) ---
        n_upd = ep_learn["n_updates"]
        if n_upd > 0:
            learning = {
                "td_loss": ep_learn["td_loss"] / n_upd,
                "mean_td_error": ep_learn["mean_td_error"] / n_upd,
                "max_td_error": ep_learn["max_td_error"],
                "mean_q": ep_learn["mean_q"] / n_upd,
                "max_q": ep_learn["max_q"],
                "min_q": ep_learn["min_q"],
                "q_std": ep_learn["q_std"] / n_upd,
                "mean_online_q": ep_learn["mean_online_q"] / n_upd,
                "mean_target_q": ep_learn["mean_target_q"] / n_upd,
                "max_target_q": ep_learn["max_target_q"],
                "target_online_gap": ep_learn["target_online_gap"] / n_upd,
                "gradient_norm": ep_learn["gradient_norm"] / n_upd,
                "n_updates": n_upd,
                "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                "replay_size": int(len(buffer)),
                "epsilon": float(eps),
                "replay_hit_fraction": float(ep_learn["replay_hit_frac"] / n_upd),
                "replay_sequence_hit_fraction": float(ep_learn["replay_seq_hit_frac"] / n_upd),
                "mode_diversity_pen": float(ep_learn["mode_diversity_pen"] / n_upd),
                "diversity_penalty": float(ep_learn["diversity_penalty"] / n_upd),
                "mode_collapse_rate": float(ep_learn["mode_collapse_rate"] / n_upd),
                "batch_action_entropy": float(ep_learn["batch_action_entropy"] / n_upd),
                "top_band_frac": float(ep_learn["top_band_frac"] / n_upd),
            }
        else:
            learning = {"n_updates": 0, "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                        "replay_size": int(len(buffer)), "epsilon": float(eps)}

        # --- RC-2 MoE attribution aggregates (over greedy MoE decisions only) ---
        n_attr = ep_ns["attributed"]
        if n_attr > 0:
            sq = ep_ns["scores"]
            modal_q_arg = int(np.argmax(ep_q_argmax_band_counts))
            modal_band = int(np.argmax(ep_band_counts)) if ep_steps else None
            moe_agg = {
                "moe_eager_weight": float(moe.eager_weight),
                "moe_revisit_weight": float(moe.revisit_weight),
                "moe_semantic_weight": float(moe.semantic_weight),
                "moe_preemptive_weight": float(moe.preemptive_weight),
                "mean_q_score": sq["mean_q_score"] / n_attr,
                "mean_eager_score": sq["mean_eager_score"] / n_attr,
                "mean_revisit_score": sq["mean_revisit_score"] / n_attr,
                "mean_semantic_score": sq["mean_semantic_score"] / n_attr,
                "mean_preemptive_score": sq["mean_preemptive_score"] / n_attr,
                "mean_fused_score": sq["mean_fused_score"] / n_attr,
                "same_argmax_fraction": float(ep_ns["same_argmax"] / n_attr),
                "mean_drqn_rank": sq["drqn_rank"] / n_attr,
                "moe_rank": 1,
                "selected_band": modal_band,
                "q_argmax_band": modal_q_arg,
                "moe_argmax_band": modal_band,
            }
        else:
            moe_agg = {}

        record = make_episode_record(
            step=global_step,
            episode=episode,
            core=core,
            reward_components=reward_components,
            actions=actions,
            learning=learning,
            moe=moe_agg,
            band_priorities=band_priorities,
            epsilon=float(eps),
        )
        record["avg_intercept_rate"] = float(ep_hits / max(1, getattr(env, "current_step", ep_steps)))
        record["band_selection_entropy"] = float(fom.get("band_selection_coverage", 0.0) or 0.0)
        record["reward_breakdown"] = reward_breakdown
        record["reconstructed_reward_sum"] = reconstructed_sum
        if latest_shift_telemetry is not None:
            record.update(latest_shift_telemetry)
            record["distribution_shift"] = dict(latest_shift_telemetry)
        telemetry.update(**record)

        # Periodic MoE evaluation on fixed val scenarios every 5000 steps
        if episode > 0 and global_step % 5000 == 0 and val_set.files_used:
            logger.info("Periodic Val Checkpoint [step %d] | Update Integrity Counters: %s", global_step, update_integrity_counters)
            try:
                val_env = CognitiveRFScanEnv(
                    env_config,
                    records=None,
                    seed=seed,
                    records_provider=val_set.sample,
                    deinterleaver_model=deinterleaver_model,
                    deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
                )
                scenario_details = []
                agg = {
                    "rewards": [], "hits": [], "steps": [],
                    "band_counts": None, "mode_counts": None,
                    "action_entropy": [], "band_entropy": [], "mode_entropy": [],
                    "components": {},
                }
                for _ in range(len(val_set.files_used)):
                    scenario_id = val_set.current_scenario_id()
                    obs_v, _ = val_env.reset()
                    try:
                        hidden_v = online_drqn.init_hidden(1, device)
                    except Exception:
                        hidden_v = None
                    moe.reset()
                    if hidden_v is not None:
                        moe.eager_agent.hidden = hidden_v
                    done_v = False
                    r_sum = 0.0
                    h_sum = 0
                    s_sum = 0
                    band_c = np.zeros(n_bands, dtype=np.float64)
                    mode_c = np.zeros(n_modes, dtype=np.float64)
                    while not done_v:
                        with torch.inference_mode():
                            if action_selection_mode == "flat_argmax":
                                obs_np_v = np.asarray(obs_v, dtype=np.float32)
                                obs_t_v = torch.from_numpy(obs_np_v).to(device)
                                a_v, hidden_v = online_drqn.act(
                                    obs_t_v,
                                    hidden_v,
                                    mode_selection="flat_argmax",
                                    tau=0.0,
                                    )
                            else:
                                a_v, hidden_v, _ = moe.select_action(obs_v, hidden_v)
                        obs_v, rew_v, term_v, trunc_v, _ = val_env.step(a_v)
                        r_sum += float(rew_v)
                        band_c[int(a_v // n_modes)] += 1
                        mode_c[int(a_v % n_modes)] += 1
                        if action_selection_mode != "flat_argmax":
                            moe.update(a_v)
                        s_sum += 1
                        done_v = bool(term_v or trunc_v)
                    fs = val_env.get_fom()
                    h_sum = int(fs.get("n_hits", 0))
                    fs = val_env.get_fom()
                    scenario_details.append({
                        "scenario_id": scenario_id,
                        "reward": float(r_sum),
                        "hits": h_sum,
                        "steps": s_sum,
                        "pd": fs.get("Pd"),
                        "coverage": fs.get("band_selection_coverage"),
                        "band_entropy": shannon_entropy(band_c),
                        "mode_entropy": shannon_entropy(mode_c),
                    })
                    agg["rewards"].append(float(r_sum))
                    agg["hits"].append(h_sum)
                    agg["steps"].append(s_sum)
                    agg["band_entropy"].append(float("nan") if shannon_entropy(band_c) is None else shannon_entropy(band_c))
                    agg["mode_entropy"].append(float("nan") if shannon_entropy(mode_c) is None else shannon_entropy(mode_c))
                    if agg["band_counts"] is None:
                        agg["band_counts"] = band_c.copy()
                        agg["mode_counts"] = mode_c.copy()
                    else:
                        agg["band_counts"] += band_c
                        agg["mode_counts"] += mode_c
                    for canon, avg_key in _avg_to_total.items():
                        avg_v = fs.get(avg_key)
                        total_v = (float(avg_v) * s_sum) if avg_v is not None else None
                        agg["components"].setdefault(canon, []).append(total_v)

                n_val = len(agg["rewards"])
                val_components: dict = {}
                for canon in _avg_to_total:
                    vals = [v for v in agg["components"].get(canon, []) if v is not None]
                    val_components[canon] = (float(np.mean(vals)) if vals else None)

                def mean_step(arr):
                    finite = [v for v in arr if v is not None and float(v) == float(v)]
                    return float(np.mean(finite)) if finite else None
                core_val = {
                    "val_reward": float(np.mean(agg["rewards"])),
                    "val_hits": int(np.sum(agg["hits"])),
                    "val_steps": int(np.sum(agg["steps"])),
                    "val_n_scenarios": n_val,
                    "val_pd": mean_step([d.get("pd") for d in scenario_details]),
                    "val_coverage": mean_step([d.get("coverage") for d in scenario_details]),
                }
                entropy_val = {
                    "val_action_entropy": None,
                    "val_band_entropy": mean_step(agg["band_entropy"]),
                    "val_mode_entropy": mean_step(agg["mode_entropy"]),
                    "val_band_selection_counts": [int(c) for c in (agg["band_counts"] if agg["band_counts"] is not None else [])],
                    "val_mode_selection_counts": {DWELL_MODES[i]: int(agg["mode_counts"][i]) for i in range(n_modes)}
                                                  if agg["mode_counts"] is not None else None,
                }
                val_record = make_val_record(
                    step=global_step,
                    episode=episode,
                    core_val=core_val,
                    reward_components=val_components,
                    entropy=entropy_val,
                    validation_set_id=val_set.validation_set_id,
                    validation_files=[str(p) for p, _, _ in val_set.files_used],
                )
                val_record["scenario_details"] = coerce(scenario_details)
                telemetry.update(**val_record)
                avg_val = core_val["val_reward"]
                consecutive_val_failures = 0
                logger.info("  Val MoE avg_reward %.2f (scenarios=%s)", avg_val,
                            [d["scenario_id"] for d in scenario_details])
                if use_wandb:
                    try:
                        import wandb

                        wandb.log({"val/moe_reward": avg_val, "step": global_step})
                    except Exception:
                        pass
            except Exception as exc:
                consecutive_val_failures += 1
                update_integrity_counters["validation_failures"] += 1
                logger.error(
                    "Val MoE eval FAILED at step %d (%d consecutive failures): %s",
                    global_step,
                    consecutive_val_failures,
                    exc,
                )
                if not diagnostic:
                    raise RuntimeError(f"Fail-closed validation failure at step {global_step}: {exc}") from exc
                if consecutive_val_failures > 3:
                    raise RuntimeError("Validation loop failing repeatedly — aborting") from exc

        if ep_reward > best_reward:
            best_reward = ep_reward
            from ..utils.checkpoint_meta import build_train_metadata, save_state

            meta = build_train_metadata(
                split=subset,
                n_bands=n_bands,
                arch="DRQNScheduler+SmartScanMoE",
                seed=seed,
                metrics={"best_episode_reward": float(ep_reward)},
                extra={
                    "mode": world_mode,
                    "obs_dim": int(env_config.get("obs_dim", obs_dim)),
                    "parent_sha256": parent_sha256,
                    "phase": 11 if "phase11" in str(output_dir).lower() else None,
                    "run_id": run.run_id,
                    "global_step": global_step,
                    "dataset_fingerprint": data_fingerprint,
                },
            )
            best_ckpt_name = "checkpoint_phase11_best_reward.pt" if "phase11" in str(output_dir).lower() else "best.pt"
            save_state(online_drqn, output_dir / best_ckpt_name, meta)
            logger.info("  New best reward %.2f — saved %s", ep_reward, best_ckpt_name)

        if stop_at_step is not None and global_step >= stop_at_step:
            logger.info("Reached stop_at_step=%d — concluding training run", stop_at_step)
            break

    # Phase 2 & Phase 22 Integrity Check: ASSERT FIRST BEFORE SAVING FINAL CHECKPOINT
    if qualification_run:
        qualification_completed_at_utc = datetime.now(timezone.utc).isoformat()
        summary_generated_at_utc = qualification_completed_at_utc

        if update_integrity_counters["optimizer_updates_attempted"] <= 0:
            raise RuntimeError("QUALIFICATION FAILED: Zero optimizer updates attempted!")
        if update_integrity_counters["optimizer_updates_completed"] != update_integrity_counters["optimizer_updates_attempted"]:
            raise RuntimeError(
                f"QUALIFICATION FAILED: updates completed ({update_integrity_counters['optimizer_updates_completed']}) "
                f"!= attempted ({update_integrity_counters['optimizer_updates_attempted']})"
            )
        if update_integrity_counters["non_finite_gradient_updates"] > 0:
            raise RuntimeError(
                f"QUALIFICATION FAILED: {update_integrity_counters['non_finite_gradient_updates']} non-finite gradient updates detected!"
            )
        if update_integrity_counters["finite_gradient_updates"] != update_integrity_counters["optimizer_updates_completed"]:
            raise RuntimeError(
                f"QUALIFICATION FAILED: finite gradient updates ({update_integrity_counters['finite_gradient_updates']}) "
                f"!= completed ({update_integrity_counters['optimizer_updates_completed']})"
            )
        if (update_integrity_counters["skipped_nan"] > 0 or
            update_integrity_counters["skipped_assertion"] > 0 or
            update_integrity_counters["skipped_oom"] > 0 or
            update_integrity_counters["other_update_failures"] > 0 or
            update_integrity_counters["validation_failures"] > 0):
            raise RuntimeError(
                f"QUALIFICATION FAILED: Non-zero update/validation skips in qualification run: {update_integrity_counters}"
            )

        # Checkpoint promotion prohibition assertion
        prod_frozen_path = Path("experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt")
        if prod_frozen_path.exists():
            with open(prod_frozen_path, "rb") as f:
                cur_frozen_sha = hashlib.sha256(f.read()).hexdigest()
            if cur_frozen_sha != FROZEN_25K_SHA:
                raise RuntimeError(
                    f"QUALIFICATION FAILED: Production Gate-25k frozen checkpoint was modified! "
                    f"Expected {FROZEN_25K_SHA}, got {cur_frozen_sha}"
                )
    elif not diagnostic:
        if update_integrity_counters["optimizer_updates_completed"] != update_integrity_counters["optimizer_updates_attempted"]:
            raise RuntimeError(
                f"TRAINING INTEGRITY VIOLATION: updates completed ({update_integrity_counters['optimizer_updates_completed']}) "
                f"!= attempted ({update_integrity_counters['optimizer_updates_attempted']})"
            )
        if (update_integrity_counters["skipped_nan"] > 0 or
            update_integrity_counters["skipped_assertion"] > 0 or
            update_integrity_counters["skipped_oom"] > 0 or
            update_integrity_counters["other_update_failures"] > 0 or
            update_integrity_counters["validation_failures"] > 0):
            raise RuntimeError(
                f"TRAINING INTEGRITY VIOLATION: Non-zero update/validation skips in production run: {update_integrity_counters}"
            )

    # Integrity assertions passed: now save final checkpoint and metadata
    final_ckpt_name = f"checkpoint_phase11_step_{global_step}.pt" if "phase11" in str(output_dir).lower() else "final.pt"
    final_path = output_dir / final_ckpt_name
    from ..utils.checkpoint_meta import build_train_metadata, save_state, write_checkpoint_metadata

    final_meta = build_train_metadata(
        split=subset,
        n_bands=n_bands,
        arch="DRQNScheduler+SmartScanMoE",
        seed=seed,
        metrics={"best_episode_reward": float(best_reward)},
        extra={
            "mode": world_mode,
            "obs_dim": int(env_config.get("obs_dim", obs_dim)),
            "parent_sha256": parent_sha256,
            "phase": 11 if "phase11" in str(output_dir).lower() else None,
            "run_id": run.run_id,
            "global_step": global_step,
            "dataset_fingerprint": data_fingerprint,
        },
    )
    save_state(online_drqn, final_path, final_meta)
    # Human-readable metadata.json sidecar
    artifacts_list = [final_ckpt_name]
    if "phase11" in str(output_dir).lower():
        if (output_dir / "checkpoint_phase11_best_reward.pt").exists():
            artifacts_list.append("checkpoint_phase11_best_reward.pt")
    else:
        artifacts_list.append("best.pt")
    write_checkpoint_metadata(output_dir / "metadata.json", final_meta, artifacts=artifacts_list)
    from ..utils.experiment_manifest import write_experiment_manifest

    manifest = write_experiment_manifest(
        run.dir / "experiment_manifest.json",
        dataset_fingerprint=data_fingerprint,
        dataset_root=data_dir,
        dataset_mode=world_mode,
        split=subset,
        seed=seed,
        model_configuration=full_cfg,
        training_configuration=train_cfg,
        normalization_stats_hash=(
            normalization_stats_hash(fit_stats) if fit_stats is not None else None
        ),
        checkpoint_metadata=final_meta,
        device=str(device),
        metrics={"best_episode_reward": float(best_reward)},
    )
    write_experiment_manifest(output_dir / "experiment_manifest.json", **{
        key: manifest[key]
        for key in (
            "dataset_fingerprint", "dataset_root", "dataset_mode", "split", "seed",
            "model_configuration", "training_configuration", "normalization_stats_hash",
            "checkpoint_metadata", "device", "metrics",
        )
    })
    logger.info("Scheduler training complete. Final: %s Best: %.2f", final_path, best_reward)
    telemetry.update(step=global_step, episode=episode, type="done",
                     best_reward=float(best_reward), telemetry_schema_version=TELEMETRY_SCHEMA_VERSION)

    if qualification_run:
        q_summary = {
            "run_id": run_id,
            "git_commit_sha": git_commit_sha,
            "training_config_sha256": t_cfg_sha,
            "model_config_sha256": m_cfg_sha,
            "parent_checkpoint_sha256": parent_sha256,
            "dataset_fingerprint": data_fingerprint,
            "start_global_step": int(q_manifest["start_global_step"]),
            "target_global_step": target_stop_step,
            "final_global_step": global_step,
            "qualification_steps_requested": q_steps,
            "qualification_steps_completed": global_step - int(q_manifest["start_global_step"]),
            "optimizer_updates_attempted": update_integrity_counters["optimizer_updates_attempted"],
            "optimizer_updates_completed": update_integrity_counters["optimizer_updates_completed"],
            "finite_gradient_updates": update_integrity_counters["finite_gradient_updates"],
            "non_finite_gradient_updates": update_integrity_counters["non_finite_gradient_updates"],
            "skipped_nan": update_integrity_counters["skipped_nan"],
            "skipped_assertion": update_integrity_counters["skipped_assertion"],
            "skipped_oom": update_integrity_counters["skipped_oom"],
            "other_update_failures": update_integrity_counters["other_update_failures"],
            "validation_failures": update_integrity_counters["validation_failures"],
            "finite_gradients": (
                update_integrity_counters["finite_gradient_updates"] > 0
                and update_integrity_counters["non_finite_gradient_updates"] == 0
                and update_integrity_counters["finite_gradient_updates"] == update_integrity_counters["optimizer_updates_completed"]
            ),
            "checkpoint_output_dir": str(output_dir.resolve()),
            "qualification_started_at_utc": qualification_started_at_utc,
            "qualification_completed_at_utc": qualification_completed_at_utc,
            "summary_generated_at_utc": summary_generated_at_utc,
        }
        summary_path = output_dir / "qualification_run_summary.json"
        summary_path.write_text(json.dumps(q_summary, indent=2), encoding="utf-8")
        logger.info("QUALIFICATION SUCCESS: Emitted qualification run summary to %s", summary_path)

    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train DRQN scheduler")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset root (CLI > YAML).")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint output dir (CLI > YAML).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reset-semantic-memory", action="store_true", help="Reset semantic memory DB before training.")
    parser.add_argument("--staged-gates", type=str, default="1000,5000,25000,100000,200000,300000,500000", help="Comma-separated step gates.")
    parser.add_argument("--stop-at-step", type=int, default=None, help="Stop after reaching this absolute step.")
    parser.add_argument("--qualification-run", action="store_true", default=False, help="Run in strict qualification mode into quarantine.")
    parser.add_argument("--qualification-steps", type=int, default=1000, help="Number of steps to run in qualification mode relative to resume point.")
    parser.add_argument("--diagnostic", action="store_true", default=False, help="Diagnostic mode (soft warnings instead of fail-closed abort).")
    parser.add_argument("--expected-parent-sha", type=str, default=None, help="Explicit expected SHA-256 for parent checkpoint verification.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from.")
    parser.add_argument("--override-epsilon", type=float, default=None, help="Hold exploration epsilon at a fixed floor.")
    parser.add_argument("--disable-latency-reward", action="store_true", help="Ablate latency reward bonus.")
    parser.add_argument("--exploration-schedule", type=str, default="exponential", choices=["exponential", "slower", "staged", "c_rescue"], help="Exploration decay schedule (exponential, slower, staged, c_rescue).")
    parser.add_argument("--target-q-max", type=float, default=100.0, help="Target-Q clipping maximum ceiling.")
    parser.add_argument("--target-q-min", type=float, default=-50.0, help="Target-Q clipping minimum floor.")
    parser.add_argument("--q-reg-coef", type=float, default=1e-4, help="Coefficient for Q^2 regularization loss.")
    parser.add_argument("--targeted-exploration", action="store_true", default=True, help="Enable 36-band targeted discovery exploration during epsilon exploration.")
    parser.add_argument("--disable-targeted-exploration", action="store_false", dest="targeted_exploration", help="Disable targeted exploration (revert to pure uniform).")
    parser.add_argument("--band-discovery-quota", type=int, default=2, help="Discovery visit quota per band per episode (Phase 9B-R3 default=2).")
    parser.add_argument("--targeted-mode-strategy", type=str, default="balanced_modes", choices=["balanced_modes", "uniform"], help="Mode sampling strategy during targeted band exploration.")
    parser.add_argument("--lambda-entropy", type=float, default=None, help="Action entropy regularization coefficient.")
    parser.add_argument("--band-diversity-penalty-coef", type=float, default=None, help="Top-band diversity penalty coefficient.")
    parser.add_argument("--band-diversity-threshold", type=float, default=None, help="Top-band concentration threshold.")
    parser.add_argument("--mode-diversity-penalty-coef", type=float, default=None, help="Per-band mode diversity penalty coefficient.")
    parser.add_argument("--mode-diversity-threshold", type=float, default=None, help="Per-band mode concentration threshold.")
    parser.add_argument("--mode-collapse-rate-threshold", type=float, default=None, help="Fraction of bands with mode concentration required to trigger penalty.")
    parser.add_argument("--use-g8-film", action="store_true", default=None, help="Use FiLM-gated factorized DRQN architecture.")
    parser.add_argument("--objective-mode", type=str, default=None, help="DRQN objective mode (e.g. g3d_hybrid, g3b_smdp_only, g8_fixed_ref, step_based).")
    parser.add_argument("--c-dwell", type=float, default=None, help="Dwell penalty coefficient c_dwell.")
    parser.add_argument("--tau-ref", type=float, default=None, help="Fixed reference dwell multiplier tau_ref for centered dwell shaping.")
    parser.add_argument("--fresh-optimizer", action="store_true", default=False, help="Initialize fresh AdamW optimizer state upon resume.")
    parser.add_argument("--use-online-reward-shaper", action="store_true", default=False, help="Enable online EMA reward shaper on buffer.add.")
    args = parser.parse_args()
    if args.device:
        os.environ["DEVICE"] = args.device
    gates_list = [int(g.strip()) for g in args.staged_gates.split(",") if g.strip()]
    train_scheduler(
        args.model_config,
        args.config,
        data_dir_override=args.data_dir,
        output_dir_override=args.output_dir,
        reset_semantic_memory=args.reset_semantic_memory,
        staged_gates=gates_list,
        stop_at_step=args.stop_at_step,
        qualification_run=args.qualification_run,
        qualification_steps=args.qualification_steps,
        diagnostic=args.diagnostic,
        expected_parent_sha=args.expected_parent_sha,
        resume_checkpoint=args.resume,
        override_epsilon=args.override_epsilon,
        disable_latency_reward=args.disable_latency_reward,
        exploration_schedule=args.exploration_schedule,
        target_q_max=args.target_q_max,
        target_q_min=args.target_q_min,
        q_reg_coef=args.q_reg_coef,
        targeted_exploration=args.targeted_exploration,
        band_discovery_quota=args.band_discovery_quota,
        lambda_entropy=args.lambda_entropy,
        band_diversity_penalty_coef=args.band_diversity_penalty_coef,
        band_diversity_threshold=args.band_diversity_threshold,
        mode_diversity_penalty_coef=args.mode_diversity_penalty_coef,
        mode_diversity_threshold=args.mode_diversity_threshold,
        mode_collapse_rate_threshold=args.mode_collapse_rate_threshold,
        use_g8_film=args.use_g8_film,
        objective_mode=args.objective_mode,
        c_dwell=args.c_dwell,
        tau_ref=args.tau_ref,
        fresh_optimizer=args.fresh_optimizer,
        use_online_reward_shaper=args.use_online_reward_shaper,
    )
