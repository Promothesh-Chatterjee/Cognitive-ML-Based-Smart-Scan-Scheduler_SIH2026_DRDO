"""Phase G7-B.1: Interaction-Only Learnability Diagnostic (250 steps from Gate-25k Root).

Freezes all base parameters (V, A_b, A_m, LSTM, Encoders) and trains SOLELY the rank-8
interaction head (552 parameters) to isolate whether the interaction pathway can learn
meaningful magnitude and flip greedy actions (H1: Competition Deficit) or remains
dormant even when unhindered (H2: Parameterization Defect).

Invariants:
- Root Lineage: Immutable frozen Gate-25 root (checkpoint_gate_25000_frozen.pt, SHA: 7a99c6...)
- Architecture: LowRankCoupledDRQNScheduler (Rank R = 8, zero-initialized interaction head)
- Trainable: Solely interaction_proj.weight (64x8) and interaction_mode.weight (8x5)
- Frozen: All other parameters (requires_grad = False)
- Discount Factor: gamma = 0.99 (CANONICAL)
- Dwell Penalty: c_dwell = 2.0 (CANONICAL)
- Regularization: Mode-marginal entropy beta_mode = 0.5
- Replay Mix: Exact same 50/50 targeted agile mix (config_29, config_119, config_241 vs general train)
- Horizon: Strictly 250 environment steps (Step 25,000 -> 25,250)
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
logger = logging.getLogger("g7b1_diagnostic")

CANONICAL_GATE25_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
GATE25_PATH = repo_root / "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"

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
DENSE_SCENARIOS = {"config_194", "config_117"}
TARGETED_AGILE_SET = {"config_29", "config_119", "config_241"}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_dual_pass(
    model: LowRankCoupledDRQNScheduler,
    val_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs a paired full-versus-ablated evaluation across all 10 canonical validation scenarios."""
    model.eval()
    scenarios_data = {}

    tot_decisions = 0
    tot_flips = 0
    all_i_abs = []
    all_gap_ratios = []

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

        # 1. Full Policy Pass
        env_full = CognitiveRFScanEnv(env_cfg, records=records, seed=42)
        obs_f, _ = env_full.reset(seed=42)
        hidden_f = model.init_hidden(1, device)

        scen_acts_full = []
        scen_i_abs = []
        scen_gap_ratios = []
        scen_flips = 0

        for step in range(1000):
            obs_t = torch.tensor(obs_f, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden_f = model(obs_t, hidden_f)
                q_row_full = q_flat[0, 0].cpu().numpy()
                i_row = aux["i_tilde"][0, 0].cpu().numpy()

                v = aux["v"]
                a_b = aux["a_band_tilde"]
                a_m = aux["a_mode_tilde"]
                q_ablated = (v.unsqueeze(-1) + a_b.unsqueeze(-1) + a_m.unsqueeze(-2)).view(1, 1, CANONICAL_N_ACTIONS)[0, 0].cpu().numpy()

            act_full = int(np.argmax(q_row_full))
            act_ablated = int(np.argmax(q_ablated))

            if act_full != act_ablated:
                scen_flips += 1

            # Action gap on ablated policy: Q_(1) - Q_(2)
            sorted_q_abl = np.sort(q_ablated)
            q_gap = float(sorted_q_abl[-1] - sorted_q_abl[-2])
            i_chosen_abs = float(np.abs(i_row[act_full // CANONICAL_N_MODES, act_full % CANONICAL_N_MODES]))
            gap_ratio = float(i_chosen_abs / (q_gap + 1e-12))

            scen_acts_full.append(act_full)
            scen_i_abs.append(float(np.mean(np.abs(i_row))))
            scen_gap_ratios.append(gap_ratio)

            obs_f, reward, term, trunc, info = env_full.step(act_full)
            if term or trunc:
                break

        fom_full = env_full.get_fom()
        pd_full = float(fom_full.get("Pd", fom_full.get("pd", 0.0))) * 100.0

        # 2. Pure Ablated Execution Pass (to get true ablated Pd when policy runs closed-loop without I)
        env_abl = CognitiveRFScanEnv(env_cfg, records=records, seed=42)
        obs_a, _ = env_abl.reset(seed=42)
        hidden_a = model.init_hidden(1, device)

        for step in range(1000):
            obs_t = torch.tensor(obs_a, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                _, aux_a, hidden_a = model(obs_t, hidden_a)
                v = aux_a["v"]
                a_b = aux_a["a_band_tilde"]
                a_m = aux_a["a_mode_tilde"]
                q_ablated_only = (v.unsqueeze(-1) + a_b.unsqueeze(-1) + a_m.unsqueeze(-2)).view(1, 1, CANONICAL_N_ACTIONS)[0, 0].cpu().numpy()

            act_a = int(np.argmax(q_ablated_only))
            obs_a, reward, term, trunc, info = env_abl.step(act_a)
            if term or trunc:
                break

        fom_abl = env_abl.get_fom()
        pd_abl = float(fom_abl.get("Pd", fom_abl.get("pd", 0.0))) * 100.0

        scenarios_data[scen_name] = {
            "steps": len(scen_acts_full),
            "action_flips": scen_flips,
            "action_flip_pct": (scen_flips / float(len(scen_acts_full))) * 100.0,
            "pd_full": pd_full,
            "pd_ablated": pd_abl,
            "delta_pd": pd_full - pd_abl,
            "mean_abs_interaction": float(np.mean(scen_i_abs)),
            "max_abs_interaction": float(np.max(scen_i_abs)),
            "mean_gap_ratio": float(np.mean(scen_gap_ratios)),
        }

        tot_decisions += len(scen_acts_full)
        tot_flips += scen_flips
        all_i_abs.extend(scen_i_abs)
        all_gap_ratios.extend(scen_gap_ratios)

    # Aggregates
    macro_mean_pd_full = float(np.mean([s["pd_full"] for s in scenarios_data.values()]))
    macro_mean_pd_abl = float(np.mean([s["pd_ablated"] for s in scenarios_data.values()]))

    agile_flips = sum(scenarios_data[s]["action_flips"] for s in AGILE_SCENARIOS)
    agile_decisions = sum(scenarios_data[s]["steps"] for s in AGILE_SCENARIOS)
    agile_flip_pct = (agile_flips / float(agile_decisions)) * 100.0 if agile_decisions > 0 else 0.0

    dense_flips = sum(scenarios_data[s]["action_flips"] for s in DENSE_SCENARIOS)
    dense_decisions = sum(scenarios_data[s]["steps"] for s in DENSE_SCENARIOS)
    dense_flip_pct = (dense_flips / float(dense_decisions)) * 100.0 if dense_decisions > 0 else 0.0

    agile_mean_i = float(np.mean([scenarios_data[s]["mean_abs_interaction"] for s in AGILE_SCENARIOS]))
    dense_mean_i = float(np.mean([scenarios_data[s]["mean_abs_interaction"] for s in DENSE_SCENARIOS]))

    return {
        "total_decisions": tot_decisions,
        "total_action_flips": tot_flips,
        "total_action_flip_pct": (tot_flips / float(tot_decisions)) * 100.0 if tot_decisions > 0 else 0.0,
        "mean_abs_interaction_overall": float(np.mean(all_i_abs)),
        "max_abs_interaction_overall": float(np.max(all_i_abs)),
        "mean_gap_ratio_overall": float(np.mean(all_gap_ratios)),
        "macro_mean_pd_full": macro_mean_pd_full,
        "macro_mean_pd_ablated": macro_mean_pd_abl,
        "delta_macro_pd": macro_mean_pd_full - macro_mean_pd_abl,
        "agile_flips_pct": agile_flip_pct,
        "dense_flips_pct": dense_flip_pct,
        "agile_mean_interaction": agile_mean_i,
        "dense_mean_interaction": dense_mean_i,
        "scenarios": scenarios_data,
    }


def execute_diagnostic(
    model: LowRankCoupledDRQNScheduler,
    device: torch.device,
    val_dir: Path,
    beta_mode: float = 0.5,
    c_dwell: float = 2.0,
    gamma: float = 0.99,
    max_steps: int = 250,
    start_step: int = 25000,
) -> Dict[str, Any]:
    """Executes the 250-step interaction-only diagnostic run."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING PHASE G7-B.1: INTERACTION-ONLY LEARNABILITY DIAGNOSTIC")
    logger.info("  Start Step: %d | Stop Step: %d (Horizon: %d steps)", start_step, start_step + max_steps, max_steps)
    logger.info("  Trainable: ONLY rank-8 interaction head (interaction_proj + interaction_mode)")
    logger.info("  Frozen: ALL base streams (V, A_b, A_m, LSTM, Encoders)")
    logger.info("  Objective: Canonical G3-D (c_dwell=%.1f, gamma=%.2f) | beta_mode=%.2f", c_dwell, gamma, beta_mode)
    logger.info("  Replay Focus: 50/50 Targeted Agile Replay Mix (config_29, config_119, config_241)")
    logger.info("=" * 80)

    # Verify Gate-25 root invariant
    sha25 = compute_sha256(GATE25_PATH)
    assert sha25 == CANONICAL_GATE25_SHA, f"Gate-25 root SHA mismatch: {sha25}"

    # Freeze ALL parameters except interaction head
    trainable_params = []
    frozen_params_count = 0
    trainable_params_count = 0

    for name, param in model.named_parameters():
        if "interaction_" in name:
            param.requires_grad = True
            trainable_params.append(param)
            trainable_params_count += param.numel()
        else:
            param.requires_grad = False
            frozen_params_count += param.numel()

    logger.info("Parameter Freezing Confirmed:")
    logger.info("  Frozen Parameters: %d", frozen_params_count)
    logger.info("  Trainable Parameters: %d (Solely interaction head)", trainable_params_count)
    assert trainable_params_count == 552, f"Expected 552 trainable parameters, got {trainable_params_count}"

    # Store initial interaction weights W_I(0)
    initial_w_proj = model.interaction_proj.weight.detach().clone()
    initial_w_mode = model.interaction_mode.weight.detach().clone()

    # Deterministic seeds
    random.seed(42 + 704)
    np.random.seed(42 + 704)
    torch.manual_seed(42 + 704)

    target_model = copy.deepcopy(model).to(device)
    target_model.eval()

    optimizer = optim.Adam(trainable_params, lr=5e-6)
    loss_fn = nn.HuberLoss()

    replay_buffer = SequenceReplayBuffer(
        capacity=50000,
        seq_len=16,
        obs_dim=CANONICAL_OBS_DIM,
        burn_in=8,
        seed=42 + 704,
    )

    train_source = ScenarioSource(
        data_root="D:/TSRD",
        mode="stare",
        subset="train",
        freq_min_mhz=0.0,
        freq_max_mhz=18000.0,
        time_horizon_us=None,
        max_pulses=50000,
        seed=42 + 704,
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
            agile_records_cache[p.stem] = load_h5_records(p, chunk_mode="random", seed=42 + 704)
            logger.info("Loaded targeted agile scenario %s (%d pulses)", p.stem, len(agile_records_cache[p.stem]))

    episode_idx = 0
    agile_keys = list(agile_records_cache.keys())

    def targeted_scenario_provider():
        nonlocal episode_idx
        episode_idx += 1
        if agile_keys and (episode_idx % 2 == 0):
            return agile_records_cache[random.choice(agile_keys)]
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
        seed=42 + 704,
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
    warmup_steps = 50
    reward_baseline = 0.0
    baseline_momentum = 0.99

    interaction_grad_norms = []
    td_losses = []

    obs, _ = env.reset()
    hidden = model.init_hidden(1, device)

    t0 = time.time()

    for step_idx in range(1, max_steps + 1):
        global_step = start_step + step_idx
        rel_step = step_idx

        eps = eps_end + (eps_start - eps_end) * float(np.exp(-rel_step / (eps_decay * 2.0)))

        # Action Selection
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_flat, aux, hidden = model(obs_t, hidden)
            q_row = q_flat[0, 0].cpu().numpy()

        if random.random() < eps:
            act = random.randrange(CANONICAL_N_ACTIONS)
        else:
            act = int(np.argmax(q_row))

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

        # Optimization Step (SOLELY for interaction parameters)
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

                q_flat_b, aux_b, _ = model(obs_b)
                q_chosen = q_flat_b.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

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

                loss = td_loss

                optimizer.zero_grad()
                loss.backward()

                # Verify that ONLY interaction params have gradients
                for n, p in model.named_parameters():
                    if "interaction_" not in n:
                        assert p.grad is None, f"Leakage! Frozen parameter {n} received gradient!"

                inter_grads = [p.grad for p in trainable_params if p.grad is not None]
                grad_norm = float(torch.norm(torch.stack([torch.norm(g) for g in inter_grads])).item())
                interaction_grad_norms.append(grad_norm)

                optimizer.step()
                td_losses.append(float(td_loss.item()))

        if step_idx % 50 == 0:
            logger.info("Progress: Step %d / %d | Mean TD Loss: %.4f | Mean ||grad_I||: %.6f",
                        step_idx, max_steps,
                        float(np.mean(td_losses[-20:])) if td_losses else 0.0,
                        float(np.mean(interaction_grad_norms[-20:])) if interaction_grad_norms else 0.0)

    elapsed_s = time.time() - t0

    # Weight displacement ||W_I(250) - W_I(0)||_2
    disp_proj = float(torch.norm(model.interaction_proj.weight.detach() - initial_w_proj).item())
    disp_mode = float(torch.norm(model.interaction_mode.weight.detach() - initial_w_mode).item())
    total_disp = float(np.sqrt(disp_proj**2 + disp_mode**2))

    logger.info(">>> Diagnostic Optimization Completed in %.1f seconds.", elapsed_s)
    logger.info("  Weight Displacement ||W_I(250) - W_I(0)||: %.8f (proj: %.8f, mode: %.8f)", total_disp, disp_proj, disp_mode)

    # Save diagnostic checkpoint
    ckpt_path = OUTPUT_DIR / "checkpoint_gate_25250_diagnostic.pt"
    torch.save({
        "global_step": 25250,
        "state_dict": model.state_dict(),
        "arm_id": "g7b1_interaction_learnability_diagnostic",
        "parent_sha256": CANONICAL_GATE25_SHA,
        "weight_displacement": total_disp,
    }, ckpt_path)

    # Run Dual-Pass Full vs Ablated Evaluation
    logger.info("Running paired dual-pass evaluation across all 10 canonical validation scenarios...")
    dual_eval = evaluate_dual_pass(model, val_dir, device)

    logger.info("=" * 80)
    logger.info("PHASE G7-B.1 DIAGNOSTIC RESULTS:")
    logger.info("  Total Decisions: %d", dual_eval["total_decisions"])
    logger.info("  Action Flips (Full vs Ablated): %d (%.3f%%)", dual_eval["total_action_flips"], dual_eval["total_action_flip_pct"])
    logger.info("  Agile Flips %%: %.3f%% | Dense Flips %%: %.3f%%", dual_eval["agile_flips_pct"], dual_eval["dense_flips_pct"])
    logger.info("  Mean |I| Overall: %.8e | Max |I|: %.8e", dual_eval["mean_abs_interaction_overall"], dual_eval["max_abs_interaction_overall"])
    logger.info("  Mean |I| Agile: %.8e | Mean |I| Dense: %.8e", dual_eval["agile_mean_interaction"], dual_eval["dense_mean_interaction"])
    logger.info("  Mean |I| / Q_gap Ratio: %.8e", dual_eval["mean_gap_ratio_overall"])
    logger.info("  Macro Mean Pd: Full = %.2f%% | Ablated = %.2f%% | Delta = %+.2f%%",
                dual_eval["macro_mean_pd_full"], dual_eval["macro_mean_pd_ablated"], dual_eval["delta_macro_pd"])
    logger.info("=" * 80)

    # Classification logic: H1 vs H2
    if dual_eval["total_action_flip_pct"] > 0.5 and total_disp > 1e-4:
        hypothesis_verdict = "H1_CONFIRMED_INTERACTION_CAPACITY_VIABLE_COMPETITION_WAS_BOTTLENECK"
    else:
        hypothesis_verdict = "H2_CONFIRMED_RANK8_PARAMETERIZATION_INEFFECTIVE_FLAT_GRADIENT"

    logger.info("DIAGNOSTIC HYPOTHESIS VERDICT: %s", hypothesis_verdict)

    result_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase G7-B.1: Interaction-Only Learnability Diagnostic",
        "root_parent_sha256": CANONICAL_GATE25_SHA,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": compute_sha256(ckpt_path),
        "horizon_steps": max_steps,
        "elapsed_seconds": elapsed_s,
        "hypothesis_verdict": hypothesis_verdict,
        "telemetry": {
            "mean_interaction_grad_norm": float(np.mean(interaction_grad_norms)) if interaction_grad_norms else 0.0,
            "max_interaction_grad_norm": float(np.max(interaction_grad_norms)) if interaction_grad_norms else 0.0,
            "weight_displacement_total": total_disp,
            "weight_displacement_proj": disp_proj,
            "weight_displacement_mode": disp_mode,
        },
        "evaluation": dual_eval,
    }

    manifest_path = REPORTS_DIR / "g7b1_learnability_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info("Saved Phase G7-B.1 manifest to %s", manifest_path)

    return result_data


def main():
    parser = argparse.ArgumentParser(description="Phase G7-B.1 Learnability Diagnostic")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-dir", type=str, default="D:/TSRD/stare/val_stare")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")

    device = torch.device(args.device)

    logger.info("Initializing LowRankCoupledDRQNScheduler from Gate-25 frozen root under G7-B.1 contract...")
    model, _ = LowRankCoupledDRQNScheduler.from_gate25_checkpoint(GATE25_PATH, rank=8, seed=42)
    model.to(device)

    execute_diagnostic(
        model=model,
        device=device,
        val_dir=val_dir,
        beta_mode=0.5,
        c_dwell=2.0,
        gamma=0.99,
        max_steps=250,
        start_step=25000,
    )


if __name__ == "__main__":
    main()
