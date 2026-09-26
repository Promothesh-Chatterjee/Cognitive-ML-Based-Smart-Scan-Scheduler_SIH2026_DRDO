"""Phase G6: Canonical Factorized G3-D Replication (1,000-step bounded qualification run).

Tests the factorized DRQN scheduler under the canonical G3-D objective:
- Parent: checkpoint_gate_25000_frozen.pt (SHA: 7a99c6...)
- Architecture: Decoupled Factorized (b, m) DRQN under G5 initialization contract (seed 42)
- Discount Factor: gamma = 0.99 (RESTORED CANONICAL)
- Dwell Penalty: c_dwell = 2.0 (CANONICAL)
- Dwell Multipliers: [0.25, 1.0, 2.5, 1.0, 1.0]
- Immediate Reward Shift: [+1.5, 0.0, -3.0, 0.0, 0.0]
- Regularization: Mode-marginal entropy beta_mode = 0.5
- Exploration: Slower schedule (eps: 0.15 -> 0.05, eps_decay = 100,000)
- Horizon: Exactly 1,000 environment steps (Step 25,000 -> 26,000)
- Manual Authorization Boundary: Stop at Gate 26,000
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.optim as optim

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.contracts import (
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    RF_BASE_DWELL_TIME_US,
    band_of_action,
    mode_of_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records, ScenarioSource
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.models.factorized_drqn_scheduler import FactorizedDRQNScheduler
from ew_core.training.replay_buffer import SequenceReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("g6_replication")

CANONICAL_GATE25_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
GATE25_PATH = repo_root / "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g6_canonical_factorized_replication"
REPORTS_DIR = repo_root / "reports"

CANONICAL_SCENARIOS = [
    "config_117",
    "config_119",
    "config_143",
    "config_194",
    "config_195",
    "config_241",
    "config_29",
    "config_42",
    "config_64",
    "config_96",
]
SPARSE_SCENARIOS = {"config_143", "config_119"}
AGILE_SCENARIOS = {"config_119", "config_241", "config_29", "config_195"}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_gate25_invariant() -> str:
    sha = compute_sha256(GATE25_PATH)
    if sha != CANONICAL_GATE25_SHA:
        raise RuntimeError(f"FATAL: Gate-25k frozen root SHA mismatch! Expected {CANONICAL_GATE25_SHA}, got {sha}")
    return sha


def compute_consecutive_runs(sequence: List[int]) -> Tuple[float, int, float]:
    if not sequence:
        return 0.0, 0, 0.0
    runs = []
    current_val = sequence[0]
    current_len = 1
    repeats = 0
    for i in range(1, len(sequence)):
        if sequence[i] == current_val:
            current_len += 1
            repeats += 1
        else:
            runs.append(current_len)
            current_val = sequence[i]
            current_len = 1
    runs.append(current_len)
    mean_run = float(np.mean(runs))
    max_run = int(np.max(runs))
    repeat_frac = float(repeats / float(len(sequence) - 1)) if len(sequence) > 1 else 0.0
    return mean_run, max_run, repeat_frac


def instantiate_g5_factorized_model(parent_checkpoint_path: Path, seed: int = 42) -> FactorizedDRQNScheduler:
    """Instantiates FactorizedDRQNScheduler under the verified G5 tensor mapping contract."""
    parent_ckpt = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)
    parent_state = parent_ckpt["state_dict"]

    model = FactorizedDRQNScheduler(
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        obs_dim=CANONICAL_OBS_DIM,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Direct copy of shared representations
    direct_copy_keys = [
        "input_norm.weight", "input_norm.bias",
        "lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
        "lstm.weight_ih_l1", "lstm.weight_hh_l1", "lstm.bias_ih_l1", "lstm.bias_hh_l1",
        "value_stream.0.weight", "value_stream.0.bias",
        "value_stream.2.weight", "value_stream.2.bias",
        "band_encoder.0.weight", "band_encoder.0.bias",
        "ctx_proj.0.weight", "ctx_proj.0.bias",
        "intercept_prob_head.0.weight", "intercept_prob_head.0.bias",
        "intercept_prob_head.2.weight", "intercept_prob_head.2.bias",
        "intercept_time_head.0.weight", "intercept_time_head.0.bias",
        "intercept_time_head.2.weight", "intercept_time_head.2.bias",
    ]
    with torch.no_grad():
        for k in direct_copy_keys:
            model.state_dict()[k].copy_(parent_state[k])

        # 2. Band advantage head
        model.band_head[0].weight.copy_(parent_state["band_advantage_head.0.weight"])
        model.band_head[0].bias.copy_(parent_state["band_advantage_head.0.bias"])
        # Layer 2: deterministic mean across 5 modes
        mean_w = parent_state["band_advantage_head.2.weight"].mean(dim=0, keepdim=True)
        mean_b = parent_state["band_advantage_head.2.bias"].mean(dim=0, keepdim=True)
        model.band_head[2].weight.copy_(mean_w)
        model.band_head[2].bias.copy_(mean_b)

        # 3. Mode advantage head: Xavier initialization
        nn.init.xavier_uniform_(model.mode_head[0].weight)
        nn.init.zeros_(model.mode_head[0].bias)
        nn.init.xavier_uniform_(model.mode_head[2].weight, gain=0.1)
        nn.init.zeros_(model.mode_head[2].bias)

    return model


def evaluate_gate_checkpoint(
    model: nn.Module,
    val_dir: Path,
    device: torch.device,
    gamma: float = 0.99,
    c_dwell: float = 2.0,
) -> Dict[str, Any]:
    """Deterministically evaluates the model on all 10 canonical scenarios with canonical gamma=0.99."""
    model.eval()
    scen_records = {}
    all_actions = []
    all_bands = []
    all_modes = []
    all_q_values = []
    all_q_margins = []
    all_td_errors = []
    total_hits = 0
    total_novel_hits = 0
    total_dwell_us = 0.0

    for scen_name in CANONICAL_SCENARIOS:
        h5_path = val_dir / f"{scen_name}.h5"
        records = load_h5_records(h5_path, chunk_mode="first")

        env_cfg = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "obs_dim": CANONICAL_OBS_DIM,
            "semantic_memory_path": ":memory:",
            "max_steps_per_episode": 1000,
            "reward": {"version": "v2"},
        }
        env = CognitiveRFScanEnv(env_cfg, records=records, seed=42)
        obs, _ = env.reset(seed=42)
        hidden = model.init_hidden(1, device)

        scen_acts = []
        scen_bnds = []
        scen_mods = []
        scen_hits = 0
        scen_novel_hits = 0
        scen_dwell_us = 0.0
        first_hit_latency = None

        for step in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                q_row = q_flat[0, 0].cpu().numpy()

            act = int(np.argmax(q_row))
            sorted_q = np.sort(q_row)
            q_margin = float(sorted_q[-1] - sorted_q[-2])
            all_q_margins.append(q_margin)
            all_q_values.append(q_row)

            b = band_of_action(act, CANONICAL_N_MODES)
            m = mode_of_action(act, CANONICAL_N_MODES)
            dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m]
            dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

            scen_acts.append(act)
            scen_bnds.append(b)
            scen_mods.append(m)
            scen_dwell_us += dwell_us

            next_obs, reward, term, trunc, info = env.step(act)
            hit = bool(info.get("hit", False))
            is_novel = bool(info.get("novel_emitter", False))

            if hit:
                scen_hits += 1
                if is_novel:
                    scen_novel_hits += 1
                if first_hit_latency is None:
                    first_hit_latency = float(scen_dwell_us)

            # Compute Bellman error under canonical G3-D (gamma = 0.99)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                next_q_flat, _, _ = model(next_obs_t, hidden)
                next_q_max = float(torch.max(next_q_flat[0, 0]).cpu().numpy())

            eff_gamma = gamma ** dwell_mult
            eff_reward = reward - (c_dwell * (dwell_mult - 1.0))
            y_target = eff_reward + eff_gamma * next_q_max
            td_err = abs(y_target - float(q_row[act]))
            all_td_errors.append(td_err)

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        dwell_ms = scen_dwell_us / 1000.0
        mean_run, max_run, repeat_frac = compute_consecutive_runs(scen_bnds)
        m_counts = {DWELL_MODES[i]: int(np.sum(np.array(scen_mods) == i)) for i in range(CANONICAL_N_MODES)}
        short_fraction = float(m_counts["SHORT_DWELL"] / float(len(scen_acts)))

        scen_records[scen_name] = {
            "steps": len(scen_acts),
            "hits": scen_hits,
            "novel_hits": scen_novel_hits,
            "dwell_ms": dwell_ms,
            "gross_hits_per_ms": scen_hits / dwell_ms if dwell_ms > 0 else 0.0,
            "novel_hits_per_ms": scen_novel_hits / dwell_ms if dwell_ms > 0 else 0.0,
            "ir_decision": (scen_hits / float(len(scen_acts))) * 100.0,
            "pd": float(fom.get("Pd", fom.get("pd", 0.0))) * 100.0,
            "pfa": float(fom.get("Pfa", fom.get("pfa", 0.0))),
            "first_hit_latency_ms": (first_hit_latency or scen_dwell_us) / 1000.0,
            "mean_consecutive_band_run": mean_run,
            "max_consecutive_band_run": max_run,
            "repeated_band_fraction": repeat_frac,
            "modes": m_counts,
            "short_fraction_pct": short_fraction * 100.0,
        }
        all_actions.extend(scen_acts)
        all_bands.extend(scen_bnds)
        all_modes.extend(scen_mods)
        total_hits += scen_hits
        total_novel_hits += scen_novel_hits
        total_dwell_us += scen_dwell_us

    n_tot_steps = len(all_actions)
    tot_dwell_ms = total_dwell_us / 1000.0
    all_q_arr = np.array(all_q_values)

    mode_counts = np.bincount(all_modes, minlength=CANONICAL_N_MODES)
    mode_probs = mode_counts / float(n_tot_steps)
    mode_ent = float(entropy(mode_probs + 1e-12, base=np.e))

    mode_transitions = sum(1 for i in range(len(all_modes) - 1) if all_modes[i] != all_modes[i + 1])
    mode_transition_rate = float(mode_transitions / float(len(all_modes) - 1)) if len(all_modes) > 1 else 0.0

    band_counts = np.bincount(all_bands, minlength=CANONICAL_N_BANDS)
    band_probs = band_counts / float(n_tot_steps)
    sorted_band_probs = np.sort(band_probs)
    top_1_band_frac = float(sorted_band_probs[-1])
    top_2_band_frac = float(sorted_band_probs[-1] + sorted_band_probs[-2])
    distinct_bands = int(np.count_nonzero(band_counts))
    band_ent = float(entropy(band_probs + 1e-12, base=np.e))

    mean_run_all, max_run_all, repeat_frac_all = compute_consecutive_runs(all_bands)

    td_arr = np.array(all_td_errors)
    td_mean = float(np.mean(td_arr))
    td_p90 = float(np.percentile(td_arr, 90))
    bellman_loss = float(np.mean(td_arr ** 2))

    sparse_pds = [scen_records[s]["pd"] for s in SPARSE_SCENARIOS]
    agile_pds = [scen_records[s]["pd"] for s in AGILE_SCENARIOS]
    agile_short_ge_1pct = sum(1 for s in AGILE_SCENARIOS if scen_records[s]["short_fraction_pct"] >= 1.0)

    return {
        "mode_entropy": mode_ent,
        "mode_distribution_pct": {
            DWELL_MODES[i]: float(mode_probs[i] * 100.0) for i in range(CANONICAL_N_MODES)
        },
        "mode_transition_rate": mode_transition_rate,
        "short_fraction_pct": float(mode_probs[0] * 100.0),
        "agile_scenarios_short_ge_1pct": agile_short_ge_1pct,
        "q_max": float(np.max(all_q_arr)),
        "q_min": float(np.min(all_q_arr)),
        "q_mean": float(np.mean(all_q_arr)),
        "q_std": float(np.std(all_q_arr)),
        "q_margin_mean": float(np.mean(all_q_margins)),
        "bellman_td_error_mean": td_mean,
        "bellman_td_error_p90": td_p90,
        "bellman_loss": bellman_loss,
        "band_entropy": band_ent,
        "top_1_band_fraction": top_1_band_frac,
        "top_2_band_fraction": top_2_band_frac,
        "distinct_bands": distinct_bands,
        "mean_consecutive_band_run": mean_run_all,
        "max_consecutive_band_run": max_run_all,
        "repeated_band_dwell_fraction": repeat_frac_all,
        "mean_pd": float(np.mean([s["pd"] for s in scen_records.values()])),
        "mean_pfa": float(np.mean([s["pfa"] for s in scen_records.values()])),
        "first_hit_latency_ms_mean": float(np.mean([s["first_hit_latency_ms"] for s in scen_records.values()])),
        "mean_ir_decision": (total_hits / float(n_tot_steps)) * 100.0,
        "gross_hits_per_ms": total_hits / tot_dwell_ms if tot_dwell_ms > 0 else 0.0,
        "novel_hits_per_ms": total_novel_hits / tot_dwell_ms if tot_dwell_ms > 0 else 0.0,
        "sparse_pd": float(np.mean(sparse_pds)),
        "agile_pd": float(np.mean(agile_pds)),
        "config_29_pd": scen_records["config_29"]["pd"],
        "scenarios": scen_records,
    }


def execute_g6_replication(
    model: nn.Module,
    device: torch.device,
    val_dir: Path,
    beta_mode: float = 0.5,
    c_dwell: float = 2.0,
    gamma: float = 0.99,  # RESTORED CANONICAL GAMMA
    max_steps: int = 1000,
    start_step: int = 25000,
) -> Dict[str, Any]:
    """Executes the strictly bounded 1,000-step qualification run."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING PHASE G6: CANONICAL FACTORIZED REPLICATION")
    logger.info("  Start Step: %d | Stop Step: %d (Horizon: %d steps)", start_step, start_step + max_steps, max_steps)
    logger.info("  Objective: Canonical G3-D (c_dwell=%.1f, gamma=%.2f) | beta_mode=%.2f", c_dwell, gamma, beta_mode)
    logger.info("  Dwell Multipliers: [0.25, 1.0, 2.5, 1.0, 1.0]")
    logger.info("  Output Directory: %s", OUTPUT_DIR)
    logger.info("=" * 80)

    # Preflight assertion checks
    assert gamma == 0.99, f"FATAL: Canonical gamma must be exactly 0.99, got {gamma}"
    assert c_dwell == 2.0, f"FATAL: Canonical c_dwell must be exactly 2.0, got {c_dwell}"
    assert beta_mode == 0.5, f"FATAL: Canonical beta_mode must be exactly 0.5, got {beta_mode}"

    verify_gate25_invariant()

    # Deterministic execution seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    target_model = copy.deepcopy(model).to(device)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    loss_fn = nn.HuberLoss()

    replay_buffer = SequenceReplayBuffer(
        capacity=50000,
        seq_len=16,
        obs_dim=CANONICAL_OBS_DIM,
        burn_in=8,
        seed=42,
    )

    orig_sample = replay_buffer.sample
    def only_completed_sample(batch_size, **kwargs):
        if len(replay_buffer._episodes) > 0:
            cur = replay_buffer._current
            cur_len = replay_buffer._current_len
            replay_buffer._current = None
            replay_buffer._current_len = 0
            try:
                res = orig_sample(batch_size, **kwargs)
            finally:
                replay_buffer._current = cur
                replay_buffer._current_len = cur_len
            return res
        return orig_sample(batch_size, **kwargs)

    replay_buffer.sample = only_completed_sample

    train_source = ScenarioSource(
        data_root="D:/TSRD",
        mode="stare",
        subset="train",
        freq_min_mhz=0.0,
        freq_max_mhz=18000.0,
        time_horizon_us=None,
        max_pulses=50000,
        seed=42,
        source_type="world",
        allow_synthetic_fallback=False,
        chunk_mode="first",
    )
    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_OBS_DIM,
        "max_steps_per_episode": 1000,
        "reward": {"version": "v2"},
        "semantic_memory_enabled": False,
    }
    env = CognitiveRFScanEnv(
        env_cfg,
        records=None,
        seed=42,
        records_provider=train_source.sample,
        semantic_memory_path=":memory:",
    )

    eps_start = 0.15
    eps_end = 0.05
    eps_decay = 50000.0

    global_step = start_step
    target_update_freq = 250
    update_freq = 4
    batch_size = 32
    warmup_steps = 100
    reward_baseline = 0.0
    baseline_momentum = 0.99

    pre_clip_grad_norms = []
    td_losses = []
    mode_entropies_marginal = []
    mode_entropies_state = []
    greedy_actions_count = 0
    total_actions_count = 0

    obs, _ = env.reset()
    hidden = model.init_hidden(1, device)

    t0 = time.time()
    evaluation_gates = {}

    for step_idx in range(1, max_steps + 1):
        global_step = start_step + step_idx
        rel_step = step_idx

        # Compute epsilon under slower schedule
        eps = eps_end + (eps_start - eps_end) * float(np.exp(-rel_step / (eps_decay * 2.0)))

        # Action Selection
        total_actions_count += 1
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_flat, aux, hidden = model(obs_t, hidden)
            q_row = q_flat[0, 0].cpu().numpy()

        q_max_now = float(np.max(q_row))
        if q_max_now > 50.0:
            raise RuntimeError(f"FATAL HARD STOP: Online Qmax exceeded 50.0 safety ceiling! Current Qmax: {q_max_now:.2f}")

        if random.random() < eps:
            act = random.randrange(CANONICAL_N_ACTIONS)
        else:
            act = int(np.argmax(q_row))
            greedy_actions_count += 1

        m = mode_of_action(act, CANONICAL_N_MODES)
        dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m]
        dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

        next_obs, reward, term, trunc, info = env.step(act)
        done = bool(term or trunc)

        hit_prob = 1.0 if bool(info.get("hit", False)) else 0.0
        intercept_time = dwell_us if bool(info.get("hit", False)) else 0.0

        replay_buffer.add(
            obs=obs,
            action=act,
            reward=reward,
            next_obs=next_obs,
            done=done,
            hit_prob=hit_prob,
            intercept_time_us=intercept_time,
            dwell_time_us=dwell_us,
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()
            hidden = model.init_hidden(1, device)

        # Optimization Step
        if step_idx > warmup_steps and replay_buffer.can_sample(batch_size) and (step_idx % update_freq == 0):
            batch = replay_buffer.sample(batch_size=batch_size)
            valid = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
            burn_in = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
            loss_mask = valid & ~burn_in

            if loss_mask.any():
                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
                rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

                # Forward pass online
                q_flat_b, aux_b, _ = model(obs_b)
                q_chosen = q_flat_b.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
                mode_logits = aux_b["a_mode_raw"][loss_mask]

                # Target pass (Double DQN)
                with torch.no_grad():
                    next_q_online, _, _ = model(next_obs_b)
                    best_acts = next_q_online.argmax(dim=-1, keepdim=True)
                    next_q_target, _, _ = target_model(next_obs_b)
                    next_q = next_q_target.gather(-1, best_acts).squeeze(-1)

                # Canonical G3-D Bellman Target Formulation with gamma = 0.99
                dwell_multipliers = torch.tensor([0.25, 1.0, 2.5, 1.0, 1.0], dtype=torch.float32, device=device)
                tau_b = dwell_multipliers[act_b % CANONICAL_N_MODES]
                gamma_eff = torch.pow(torch.tensor(gamma, device=device), tau_b)
                eff_rew_b = rew_b - (c_dwell * (tau_b - 1.0))

                # Track D: reward centering
                batch_mean = float(eff_rew_b[loss_mask].mean().item())
                reward_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * batch_mean
                centered_rew_b = eff_rew_b - reward_baseline

                targets = centered_rew_b + gamma_eff * next_q * (1.0 - done_b)
                td_loss = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())

                # Mode-marginal entropy regularization
                p_m_state = torch.softmax(mode_logits / 1.0, dim=-1)
                p_m_marginal = p_m_state.mean(dim=0)
                h_marginal = -(p_m_marginal * torch.log(p_m_marginal + 1e-12)).sum()
                h_state = -(p_m_state * torch.log(p_m_state + 1e-12)).sum(dim=-1).mean()

                loss = 0.20 * (td_loss - beta_mode * h_marginal)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"FATAL HARD STOP: Non-finite loss encountered at step {global_step}!")

                optimizer.zero_grad()
                loss.backward()

                pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                if pre_clip_norm > 50.0:
                    raise RuntimeError(f"FATAL HARD STOP: Pre-clipping gradient norm exceeded threshold at step {step_idx} (global {global_step}): {pre_clip_norm:.2f} > 50.0")

                optimizer.step()

                pre_clip_grad_norms.append(pre_clip_norm)
                td_losses.append(float(td_loss.item()))
                mode_entropies_marginal.append(float(h_marginal.item()))
                mode_entropies_state.append(float(h_state.item()))

        if step_idx % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Milestone logging
        if step_idx % 250 == 0:
            logger.info("Progress: Step %d / %d (Global %d) | Eps: %.3f | Mean TD Loss: %.3f | Mean H_mode: %.3f",
                        step_idx, max_steps, global_step, eps,
                        float(np.mean(td_losses[-50:])) if td_losses else 0.0,
                        float(np.mean(mode_entropies_marginal[-50:])) if mode_entropies_marginal else 0.0)

    # Gate 26,000 (after 1,000 steps)
    logger.info(">>> Reached Gate 26,000 (1,000 steps). Saving checkpoint & running deterministic evaluation...")
    ckpt_26k_path = OUTPUT_DIR / "checkpoint_gate_26000.pt"
    torch.save({
        "global_step": 26000,
        "state_dict": model.state_dict(),
        "arm_id": "g6_canonical_factorized_replication",
        "is_factorized": True,
        "gamma": gamma,
        "c_dwell": c_dwell,
        "beta_mode": beta_mode,
        "parent_sha256": CANONICAL_GATE25_SHA,
    }, ckpt_26k_path)

    eval_26k = evaluate_gate_checkpoint(model, val_dir, device, gamma=gamma, c_dwell=c_dwell)
    elapsed_s = time.time() - t0
    logger.info("Gate 26,000 Evaluation Complete in %.1f seconds: Hmode=%.3f, Pd=%.1f%%, Gross=%.3f hits/ms, Novel=%.4f hits/ms, SHORT=%.1f%%",
                elapsed_s, eval_26k["mode_entropy"], eval_26k["mean_pd"], eval_26k["gross_hits_per_ms"], eval_26k["novel_hits_per_ms"], eval_26k["short_fraction_pct"])

    verify_gate25_invariant()

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase G6: Canonical Factorized G3-D Replication",
        "parent_sha256": CANONICAL_GATE25_SHA,
        "gamma": gamma,
        "c_dwell": c_dwell,
        "beta_mode": beta_mode,
        "elapsed_seconds": elapsed_s,
        "training_telemetry": {
            "mean_pre_clip_grad_norm": float(np.mean(pre_clip_grad_norms)),
            "max_pre_clip_grad_norm": float(np.max(pre_clip_grad_norms)),
            "mean_td_loss": float(np.mean(td_losses)),
            "mean_mode_entropy_marginal": float(np.mean(mode_entropies_marginal)),
            "mean_mode_entropy_state": float(np.mean(mode_entropies_state)),
            "greedy_action_fraction": float(greedy_actions_count / float(total_actions_count)),
            "final_epsilon": eps,
        },
        "gate_26000": {
            "checkpoint_path": str(ckpt_26k_path),
            "checkpoint_sha256": compute_sha256(ckpt_26k_path),
            "evaluation": eval_26k,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phase G6 Canonical Factorized G3-D Replication")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-dir", type=str, default="D:/TSRD/stare/val_stare")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")

    device = torch.device(args.device)

    # 1. Instantiate model under verified G5 contract
    logger.info("Initializing Factorized DRQN under verified contract from Gate-25 frozen root...")
    model = instantiate_g5_factorized_model(GATE25_PATH, seed=42).to(device)

    # 2. Run G6 replication
    manifest_data = execute_g6_replication(
        model=model,
        device=device,
        val_dir=val_dir,
        beta_mode=0.5,
        c_dwell=2.0,
        gamma=0.99,  # TRUE CANONICAL
        max_steps=1000,
        start_step=25000,
    )

    # 3. Save manifest
    manifest_path = REPORTS_DIR / "g6_canonical_factorized_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Saved Phase G6 manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
