"""Phase G7-B Continuation: Bounded Optimization from Gate 26,000 to Gate 27,000.

Tests whether the Low-Rank Coupled DRQN (R=8) survives the critical 1,000-step interval
(26k -> 27k) that triggered catastrophic SHORT collapse in additive factorization.

Invariants:
- Lineage: Resumes from checkpoint_gate_26000.pt (Phase G7-B, SHA: 3cdfa87d...)
- Root Baseline: Immutable frozen Gate-25 root (checkpoint_gate_25000_frozen.pt, SHA: 7a99c6...)
- Architecture: LowRankCoupledDRQNScheduler (Rank R = 8)
- Discount Factor: gamma = 0.99 (CANONICAL)
- Dwell Penalty: c_dwell = 2.0 (CANONICAL)
- Dwell Multipliers: [0.25, 1.0, 2.5, 1.0, 1.0]
- Immediate Reward Shift: [+1.5, 0.0, -3.0, 0.0, 0.0]
- Regularization: Mode-marginal entropy beta_mode = 0.5
- Replay Mix: Exact same 50/50 targeted agile mix (config_29, config_119, config_241 vs general train)
- Horizon: Strictly 1,000 environment steps (Step 26,000 -> 27,000)
- Hard Stop: Gate 27,000 for deterministic evaluation & interaction ablation
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
from ew_core.models.low_rank_coupled_drqn_scheduler import LowRankCoupledDRQNScheduler
from ew_core.training.replay_buffer import SequenceReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("g7b_continuation_27k")

CANONICAL_GATE25_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
GATE25_PATH = repo_root / "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"

CANONICAL_GATE26_SHA = "3cdfa87d88549ffa4982f96458ed316e69347271e51674a5df6638e86d9f8035"
GATE26_PATH = repo_root / "experiments/checkpoints/g7b_low_rank_coupling/checkpoint_gate_26000.pt"

OUTPUT_DIR = repo_root / "experiments/checkpoints/g7b_low_rank_coupling"
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
TARGETED_AGILE_SET = {"config_29", "config_119", "config_241"}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoints() -> None:
    sha25 = compute_sha256(GATE25_PATH)
    if sha25 != CANONICAL_GATE25_SHA:
        raise RuntimeError(f"FATAL: Gate-25k frozen root SHA mismatch! Expected {CANONICAL_GATE25_SHA}, got {sha25}")
    sha26 = compute_sha256(GATE26_PATH)
    if sha26 != CANONICAL_GATE26_SHA:
        raise RuntimeError(f"FATAL: Gate-26k starting checkpoint SHA mismatch! Expected {CANONICAL_GATE26_SHA}, got {sha26}")
    logger.info("Verified Gate-25 root (%s) and Gate-26 starting checkpoint (%s)", sha25[:8], sha26[:8])


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


def load_g7b_model(checkpoint_path: Path, rank: int = 8) -> LowRankCoupledDRQNScheduler:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model = LowRankCoupledDRQNScheduler(rank=rank)
    model.load_state_dict(state)
    return model


def evaluate_gate_checkpoint(
    model: LowRankCoupledDRQNScheduler,
    val_dir: Path,
    device: torch.device,
    gamma: float = 0.99,
    c_dwell: float = 2.0,
    ablate_interaction: bool = False,
) -> Dict[str, Any]:
    """Evaluates the model on all 10 canonical scenarios with or without interaction ablation."""
    model.eval()
    scen_records = {}
    all_actions = []
    all_bands = []
    all_modes = []
    all_q_values = []
    all_q_margins = []
    all_td_errors = []
    all_interaction_norms = []
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
                if ablate_interaction:
                    v = aux["v"]
                    a_band_tilde = aux["a_band_tilde"]
                    a_mode_tilde = aux["a_mode_tilde"]
                    q_ablated = v.unsqueeze(-1) + a_band_tilde.unsqueeze(-1) + a_mode_tilde.unsqueeze(-2)
                    q_row = q_ablated.view(1, 1, CANONICAL_N_ACTIONS)[0, 0].cpu().numpy()
                else:
                    q_row = q_flat[0, 0].cpu().numpy()

                i_tilde_row = aux["i_tilde"][0, 0].cpu().numpy()

            act = int(np.argmax(q_row))
            sorted_q = np.sort(q_row)
            q_margin = float(sorted_q[-1] - sorted_q[-2])
            all_q_margins.append(q_margin)
            all_q_values.append(q_row)
            all_interaction_norms.append(float(np.mean(np.abs(i_tilde_row))))

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
            "actions": scen_acts,
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

    res = {
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
        "mean_interaction_norm": float(np.mean(all_interaction_norms)),
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
        "config_119_pd": scen_records["config_119"]["pd"],
        "config_241_pd": scen_records["config_241"]["pd"],
        "scenarios": scen_records,
        "all_actions": all_actions,
    }
    return res


def execute_continuation(
    model: LowRankCoupledDRQNScheduler,
    device: torch.device,
    val_dir: Path,
    beta_mode: float = 0.5,
    c_dwell: float = 2.0,
    gamma: float = 0.99,
    max_steps: int = 1000,
    start_step: int = 26000,
) -> Dict[str, Any]:
    """Executes the 1,000-step Phase G7-B continuation (Step 26,000 -> 27,000)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING PHASE G7-B CONTINUATION: GATE 26,000 -> 27,000")
    logger.info("  Start Step: %d | Stop Step: %d (Horizon: %d steps)", start_step, start_step + max_steps, max_steps)
    logger.info("  Objective: Canonical G3-D (c_dwell=%.1f, gamma=%.2f) | beta_mode=%.2f", c_dwell, gamma, beta_mode)
    logger.info("  Architecture: LowRankCoupledDRQNScheduler (Rank R=8)")
    logger.info("  Replay Focus: 50/50 Targeted Agile Replay Mix (config_29, config_119, config_241)")
    logger.info("=" * 80)

    verify_checkpoints()

    # Deterministic continuation seed
    random.seed(42 + 703)
    np.random.seed(42 + 703)
    torch.manual_seed(42 + 703)

    target_model = copy.deepcopy(model).to(device)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    loss_fn = nn.HuberLoss()

    replay_buffer = SequenceReplayBuffer(
        capacity=50000,
        seq_len=16,
        obs_dim=CANONICAL_OBS_DIM,
        burn_in=8,
        seed=42 + 703,
    )

    train_source = ScenarioSource(
        data_root="D:/TSRD",
        mode="stare",
        subset="train",
        freq_min_mhz=0.0,
        freq_max_mhz=18000.0,
        time_horizon_us=None,
        max_pulses=50000,
        seed=42 + 703,
        source_type="world",
        allow_synthetic_fallback=False,
        chunk_mode="first",
    )

    train_scan_dir = Path("D:/TSRD/scan/train_scan")
    agile_file_paths = [
        train_scan_dir / "config_29.h5",
        train_scan_dir / "config_119.h5",
        train_scan_dir / "config_241.h5",
    ]
    agile_records_cache = {}
    for p in agile_file_paths:
        if p.exists():
            agile_records_cache[p.stem] = load_h5_records(p, chunk_mode="random", seed=42 + 703)
            logger.info("Loaded targeted agile scenario %s (%d pulses)", p.stem, len(agile_records_cache[p.stem]))
        else:
            logger.warning("Agile file %s not found in train_scan", p)

    episode_idx = 0
    agile_keys = list(agile_records_cache.keys())

    def targeted_scenario_provider():
        nonlocal episode_idx
        episode_idx += 1
        if agile_keys and (episode_idx % 2 == 0):
            chosen_key = random.choice(agile_keys)
            return agile_records_cache[chosen_key]
        else:
            return train_source.sample()

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
        seed=42 + 703,
        records_provider=targeted_scenario_provider,
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
    interaction_grad_norms = []
    greedy_actions_count = 0
    total_actions_count = 0

    obs, _ = env.reset()
    hidden = model.init_hidden(1, device)

    t0 = time.time()

    for step_idx in range(1, max_steps + 1):
        global_step = start_step + step_idx
        rel_step = 1000 + step_idx

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

                dwell_multipliers = torch.tensor([0.25, 1.0, 2.5, 1.0, 1.0], dtype=torch.float32, device=device)
                tau_b = dwell_multipliers[act_b % CANONICAL_N_MODES]
                gamma_eff = torch.pow(torch.tensor(gamma, device=device), tau_b)
                eff_rew_b = rew_b - (c_dwell * (tau_b - 1.0))

                batch_mean = float(eff_rew_b[loss_mask].mean().item())
                reward_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * batch_mean
                centered_rew_b = eff_rew_b - reward_baseline

                targets = centered_rew_b + gamma_eff * next_q * (1.0 - done_b)
                td_loss = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())

                p_m_state = torch.softmax(mode_logits / 1.0, dim=-1)
                p_m_marginal = p_m_state.mean(dim=0)
                h_marginal = -(p_m_marginal * torch.log(p_m_marginal + 1e-12)).sum()
                h_state = -(p_m_state * torch.log(p_m_state + 1e-12)).sum(dim=-1).mean()

                loss = 0.20 * (td_loss - beta_mode * h_marginal)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"FATAL HARD STOP: Non-finite loss encountered at step {global_step}!")

                optimizer.zero_grad()
                loss.backward()

                inter_grads = [p.grad for n, p in model.named_parameters() if "interaction_" in n and p.grad is not None]
                if inter_grads:
                    inter_norm = float(torch.norm(torch.stack([torch.norm(g) for g in inter_grads])).item())
                    interaction_grad_norms.append(inter_norm)

                pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                if pre_clip_norm > 50.0:
                    raise RuntimeError(f"FATAL HARD STOP: Pre-clipping gradient norm exceeded threshold at step {step_idx}: {pre_clip_norm:.2f} > 50.0")

                optimizer.step()

                pre_clip_grad_norms.append(pre_clip_norm)
                td_losses.append(float(td_loss.item()))
                mode_entropies_marginal.append(float(h_marginal.item()))
                mode_entropies_state.append(float(h_state.item()))

        if step_idx % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if step_idx % 250 == 0:
            logger.info("Progress: Step %d / %d (Global %d) | Eps: %.3f | Mean TD Loss: %.3f | Mean H_mode: %.3f | Inter Grad Norm: %.4f",
                        step_idx, max_steps, global_step, eps,
                        float(np.mean(td_losses[-50:])) if td_losses else 0.0,
                        float(np.mean(mode_entropies_marginal[-50:])) if mode_entropies_marginal else 0.0,
                        float(np.mean(interaction_grad_norms[-50:])) if interaction_grad_norms else 0.0)

    # Gate 27,000 Hard Stop
    logger.info(">>> Reached Gate 27,000 HARD STOP (1,000 continuation steps). Saving checkpoint...")
    ckpt_27k_path = OUTPUT_DIR / "checkpoint_gate_27000.pt"
    torch.save({
        "global_step": 27000,
        "state_dict": model.state_dict(),
        "arm_id": "g7b_low_rank_coupled_continuation_27k",
        "is_factorized": True,
        "is_coupled": True,
        "interaction_rank": 8,
        "gamma": gamma,
        "c_dwell": c_dwell,
        "beta_mode": beta_mode,
        "parent_sha256": CANONICAL_GATE26_SHA,
        "root_parent_sha256": CANONICAL_GATE25_SHA,
    }, ckpt_27k_path)

    logger.info("Running Gate 27,000 deterministic evaluation (Full Model)...")
    eval_full = evaluate_gate_checkpoint(model, val_dir, device, gamma=gamma, c_dwell=c_dwell, ablate_interaction=False)

    logger.info("Running Gate 27,000 read-only interaction ablation (I=0)...")
    eval_ablated = evaluate_gate_checkpoint(model, val_dir, device, gamma=gamma, c_dwell=c_dwell, ablate_interaction=True)

    elapsed_s = time.time() - t0

    # Calculate action flips
    actions_full = eval_full["all_actions"]
    actions_abl = eval_ablated["all_actions"]
    action_flips = sum(1 for f, a in zip(actions_full, actions_abl) if f != a)
    action_flips_pct = (action_flips / float(len(actions_full))) * 100.0

    logger.info("Gate 27,000 Evaluation Complete in %.1f seconds:", elapsed_s)
    logger.info("  FULL POLICY: Hmode=%.3f, Pd=%.1f%%, SHORT=%.1f%%, REVISIT=%.1f%%, config_29_Pd=%.1f%%, config_119_Pd=%.1f%%",
                eval_full["mode_entropy"], eval_full["mean_pd"], eval_full["short_fraction_pct"], eval_full["mode_distribution_pct"]["REVISIT"],
                eval_full["config_29_pd"], eval_full["config_119_pd"])
    logger.info("  ABLATION AUDIT: Action Flips = %d / 10000 (%.2f%%) | Ablated Pd = %.1f%% (Delta: %+.2f%%)",
                action_flips, action_flips_pct, eval_ablated["mean_pd"], eval_ablated["mean_pd"] - eval_full["mean_pd"])

    verify_checkpoints()

    # Clean actions array before serializing json
    del eval_full["all_actions"]
    del eval_ablated["all_actions"]
    for s in CANONICAL_SCENARIOS:
        del eval_full["scenarios"][s]["actions"]
        del eval_ablated["scenarios"][s]["actions"]

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase G7-B Continuation: Gate 26,000 -> 27,000",
        "root_parent_sha256": CANONICAL_GATE25_SHA,
        "starting_checkpoint_sha256": CANONICAL_GATE26_SHA,
        "gamma": gamma,
        "c_dwell": c_dwell,
        "beta_mode": beta_mode,
        "rank": 8,
        "elapsed_seconds": elapsed_s,
        "training_telemetry": {
            "mean_pre_clip_grad_norm": float(np.mean(pre_clip_grad_norms)),
            "max_pre_clip_grad_norm": float(np.max(pre_clip_grad_norms)),
            "mean_td_loss": float(np.mean(td_losses)),
            "mean_mode_entropy_marginal": float(np.mean(mode_entropies_marginal)),
            "mean_mode_entropy_state": float(np.mean(mode_entropies_state)),
            "mean_interaction_grad_norm": float(np.mean(interaction_grad_norms)) if interaction_grad_norms else 0.0,
            "greedy_action_fraction": float(greedy_actions_count / float(total_actions_count)),
            "final_epsilon": eps,
        },
        "gate_27000": {
            "checkpoint_path": str(ckpt_27k_path),
            "checkpoint_sha256": compute_sha256(ckpt_27k_path),
            "full_evaluation": eval_full,
            "ablated_evaluation": eval_ablated,
            "ablation_audit": {
                "action_flips": action_flips,
                "action_flips_pct": action_flips_pct,
                "delta_pd": eval_ablated["mean_pd"] - eval_full["mean_pd"],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phase G7-B Continuation (26k -> 27k)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-dir", type=str, default="D:/TSRD/stare/val_stare")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")

    device = torch.device(args.device)

    logger.info("Resuming LowRankCoupledDRQNScheduler from Gate-26 checkpoint...")
    model = load_g7b_model(GATE26_PATH, rank=8).to(device)

    manifest_data = execute_continuation(
        model=model,
        device=device,
        val_dir=val_dir,
        beta_mode=0.5,
        c_dwell=2.0,
        gamma=0.99,
        max_steps=1000,
        start_step=26000,
    )

    manifest_path = REPORTS_DIR / "g7b_continuation_27k_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Saved Phase G7-B Continuation manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
