"""Phase G8.5-FINAL Unregularized Longitudinal Stability Trial Runner.

Executes Phase G8.5-FINAL: Steps 27,000 -> 50,000 under canonical unregularized
step-based Bellman objective (gamma=0.99, q_reg=0.0).
Continuation Contract: EXACT-CONTINUATION STATE (single uninterrupted trajectory)
- Checkpoint: Frozen Gate 27,000 (SHA: fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094)
- fresh_optimizer=False (inherits Adam m and v moment states, step count, target net)
- Staged Evaluation Gates: 30,000 -> 35,000 -> 40,000 -> 45,000 -> 50,000
- Multi-Layer Decision Hierarchy:
  1. Primary Operational Acceptance: Mean Pd >= 80%, Agile Pd >= 65%, Dense Pd >= 95%,
     config_29 Pd >= 85%, Worst-Case Pd >= 30%, Blackouts == 0, IR_time >= 0.85, Thrashing < 5.0%.
  2. Value-Function Health Monitoring: Action margins (Delta_Q), TD-error tails,
     online/target divergence, probe policy stability, Q-to-return calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import entropy, linregress, spearmanr
import torch
import torch.nn as nn

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
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.band_conditioned_drqn import BandConditionedFactorizedDRQN
from ew_core.training.replay_buffer import SequenceReplayBuffer
from ew_core.training.train_scheduler import train_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_g8_5_final_trial")

GATE_27000_PATH = repo_root / "experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt"
EXPECTED_GATE_27000_SHA = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g8_5_final_trial"
VAL_DIR = Path("D:/TSRD/stare/val_stare")
REPORTS_DIR = repo_root / "reports"

STAGED_GATES = [30000, 35000, 40000, 45000, 50000]

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
AGILE_SCENARIOS = {"config_29", "config_119", "config_241", "config_42"}
SPARSE_SCENARIOS = {"config_143", "config_119"}
DENSE_SCENARIOS = {"config_117", "config_194", "config_195", "config_64", "config_96"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_htau(mode_counts: np.ndarray) -> Tuple[float, Dict[str, float]]:
    tau_counts = {
        "0.25": float(mode_counts[0]),
        "1.0": float(mode_counts[1] + mode_counts[3] + mode_counts[4]),
        "2.5": float(mode_counts[2]),
    }
    total = sum(tau_counts.values())
    if total == 0:
        return 0.0, {k: 0.0 for k in tau_counts}
    dist = {k: v / total for k, v in tau_counts.items()}
    probs = np.array([dist["0.25"], dist["1.0"], dist["2.5"]])
    h_tau = float(entropy(probs + 1e-12))
    return h_tau, dist


def detect_alternating_cycles(actions: List[int], max_cycle: int = 6) -> Dict[str, Any]:
    if len(actions) < 4:
        return {"max_cycle_length": 0, "cycle_occupancy_fraction": 0.0, "cycle_steps": 0, "total_steps": len(actions)}
    cycle_mask = np.zeros(len(actions), dtype=bool)
    for k in range(2, max_cycle + 1):
        for i in range(len(actions) - 2 * k + 1):
            pattern = actions[i : i + k]
            next_pattern = actions[i + k : i + 2 * k]
            if pattern == next_pattern:
                cycle_mask[i : i + 2 * k] = True
    c_steps = int(np.sum(cycle_mask))
    occ = float(c_steps / len(actions)) if len(actions) > 0 else 0.0
    return {
        "max_cycle_length": max_cycle if occ > 0 else 0,
        "cycle_occupancy_fraction": occ,
        "cycle_steps": c_steps,
        "total_steps": len(actions),
    }


def compute_mc_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def evaluate_checkpoint_g8_5(
    checkpoint_path: Path,
    device: torch.device,
    val_dir: Path = VAL_DIR,
    max_steps_per_scenario: int = 2000,
) -> Dict[str, Any]:
    """Evaluates checkpoint across 10 canonical scenarios + diagnostic value health monitors."""
    logger.info("Evaluating Checkpoint: %s", checkpoint_path)
    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("online_drqn", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    target_model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    target_model.load_state_dict(ckpt.get("target_state_dict", state_dict))
    target_model.eval()

    # Load baseline Gate 27k model for reference probe comparisons
    base_ckpt = torch.load(GATE_27000_PATH, map_location=device, weights_only=False)
    base_model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    base_model.load_state_dict(base_ckpt["state_dict"])
    base_model.eval()

    # 1. Ten Canonical Scenarios Evaluation
    scenario_results: Dict[str, Any] = {}
    scenario_dwell_breakdown: Dict[str, Any] = {}

    all_actions: List[int] = []
    all_bands: List[int] = []
    all_modes: List[int] = []
    all_q_values: List[np.ndarray] = []
    all_delta_q_list: List[float] = []

    c29_delta_q_long_list: List[float] = []
    c29_long_action_count = 0
    c29_pref_long_count = 0
    c29_total_steps = 0

    d_action_flips_global = 0
    total_env_steps = 0
    all_hits = 0
    all_dwell_us = 0.0
    all_band_switches = 0

    for scen_name in CANONICAL_SCENARIOS:
        h5_path = val_dir / f"{scen_name}.h5"
        records = load_h5_records(h5_path, chunk_mode="first")
        env = CognitiveRFScanEnv(records=records)
        obs, _ = env.reset()
        hx = model.init_hidden(batch_size=1, device=device)

        scen_acts: List[int] = []
        scen_bnds: List[int] = []
        scen_mods: List[int] = []
        scen_hits = 0
        scen_dwell_us = 0.0
        scen_band_switches = 0
        first_hit_latency_us = None

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                q_out, aux, hx = model(obs_t, hx)
                q_abl, _, _ = model(obs_t, hx, ablate_dwell_branch=True)

            q_row = q_out.squeeze().cpu().numpy()
            q_row_abl = q_abl.squeeze().cpu().numpy()
            act = int(np.argmax(q_row))
            act_abl = int(np.argmax(q_row_abl))
            if act != act_abl:
                d_action_flips_global += 1

            # Action margin Delta_Q = Q1 - Q2
            q_part = np.partition(q_row, -2)
            delta_q_step = float(q_part[-1] - q_part[-2])
            all_delta_q_list.append(delta_q_step)

            b = band_of_action(act, CANONICAL_N_MODES)
            m = mode_of_action(act, CANONICAL_N_MODES)
            step_dwell_us = RF_BASE_DWELL_TIME_US * DEFAULT_DWELL_MULTIPLIERS[m]

            if len(scen_bnds) > 0 and b != scen_bnds[-1]:
                scen_band_switches += 1

            scen_acts.append(act)
            scen_bnds.append(b)
            scen_mods.append(m)
            scen_dwell_us += step_dwell_us

            if scen_name == "config_29":
                c29_total_steps += 1
                q_grid = q_row.reshape(CANONICAL_N_BANDS, CANONICAL_N_MODES)
                b_star = int(np.argmax(np.max(q_grid, axis=-1)))
                q_band = q_grid[b_star]
                q_long = float(q_band[2])
                q_other_max = float(max(q_band[0], q_band[1], q_band[3], q_band[4]))
                delta_q_long = q_long - q_other_max
                c29_delta_q_long_list.append(delta_q_long)
                if delta_q_long > 0:
                    c29_pref_long_count += 1
                if m == 2:
                    c29_long_action_count += 1

            all_actions.append(act)
            all_bands.append(b)
            all_modes.append(m)
            all_q_values.append(q_row)

            next_obs, reward, term, trunc, info = env.step(act)
            hit = bool(info.get("hit", False))
            if hit:
                scen_hits += 1
                if first_hit_latency_us is None:
                    first_hit_latency_us = float(scen_dwell_us)

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        pd_val = float(fom.get("Pd", float(scen_hits / max(1, len(scen_acts))))) * 100.0
        dwell_ms = scen_dwell_us / 1000.0
        ir_time = float(scen_hits / dwell_ms) if dwell_ms > 0 else 0.0
        ir_step = float(scen_hits / len(scen_acts))

        all_hits += scen_hits
        all_dwell_us += scen_dwell_us
        all_band_switches += scen_band_switches

        scen_thrashing = detect_alternating_cycles(scen_acts)
        scen_b_counts = np.bincount(scen_bnds, minlength=CANONICAL_N_BANDS)
        distinct_bands = int(np.count_nonzero(scen_b_counts))
        top_band_frac = float(np.max(scen_b_counts) / max(1, len(scen_bnds)))

        scenario_results[scen_name] = {
            "pd": pd_val,
            "steps": len(scen_acts),
            "hits": scen_hits,
            "dwell_ms": dwell_ms,
            "gross_hits_per_ms": ir_time,
            "ir_step": ir_step,
            "first_hit_latency_us": first_hit_latency_us or scen_dwell_us,
            "band_switch_rate": float(scen_band_switches / max(1, len(scen_bnds) - 1)),
            "distinct_bands": distinct_bands,
            "top_band_concentration": top_band_frac,
            "thrashing": scen_thrashing,
        }

        scen_m_counts = np.bincount(scen_mods, minlength=CANONICAL_N_MODES)
        scen_m_dist = scen_m_counts / max(1, len(scen_mods))
        scen_htau, scen_tau_dist = compute_htau(scen_m_counts)
        scenario_dwell_breakdown[scen_name] = {
            "p_short": float(scen_m_dist[0]),
            "p_tau_1_0": float(scen_tau_dist["1.0"]),
            "p_long": float(scen_m_dist[2]),
            "h_tau": scen_htau,
            "h_mode": float(entropy(scen_m_dist + 1e-12)),
            "modes": {DWELL_MODES[i]: float(scen_m_dist[i]) for i in range(5)},
        }

    all_pds = [v["pd"] for v in scenario_results.values()]
    mean_pd = float(np.mean(all_pds))
    agile_pds = [scenario_results[s]["pd"] for s in AGILE_SCENARIOS if s in scenario_results]
    agile_pd = float(np.mean(agile_pds)) if agile_pds else 0.0
    sparse_pds = [scenario_results[s]["pd"] for s in SPARSE_SCENARIOS if s in scenario_results]
    sparse_pd = float(np.mean(sparse_pds)) if sparse_pds else 0.0
    dense_pds = [scenario_results[s]["pd"] for s in DENSE_SCENARIOS if s in scenario_results]
    dense_pd = float(np.mean(dense_pds)) if dense_pds else 0.0

    worst_case_scen = min(scenario_results.keys(), key=lambda s: scenario_results[s]["pd"])
    worst_case_pd = float(scenario_results[worst_case_scen]["pd"])
    blackouts = sum(1 for p in all_pds if p == 0.0)

    total_mission_ms = all_dwell_us / 1000.0
    global_ir_time = float(all_hits / total_mission_ms) if total_mission_ms > 0 else 0.0
    global_ir_step = float(all_hits / total_env_steps) if total_env_steps > 0 else 0.0
    global_switch_rate = float(all_band_switches / max(1, total_env_steps - len(CANONICAL_SCENARIOS)))

    all_m_counts = np.bincount(all_modes, minlength=CANONICAL_N_MODES)
    all_m_dist = all_m_counts / max(1, len(all_modes))
    global_htau, global_tau_dist = compute_htau(all_m_counts)

    # Global Thrashing Audit
    global_thrashing = detect_alternating_cycles(all_actions)
    all_b_counts = np.bincount(all_bands, minlength=CANONICAL_N_BANDS)
    global_distinct_bands = int(np.count_nonzero(all_b_counts))
    global_top_band_conc = float(np.max(all_b_counts) / max(1, len(all_bands)))

    thrashing_failed = bool(
        global_thrashing["cycle_occupancy_fraction"] >= 0.05
        or global_distinct_bands < 30
        or global_top_band_conc > 0.40
    )

    # Value distribution
    all_q_matrix = np.array(all_q_values)
    flat_q = all_q_matrix.reshape(-1)
    q_scale_stats = {
        "q_max": float(np.max(flat_q)),
        "q_min": float(np.min(flat_q)),
        "q_mean": float(np.mean(flat_q)),
        "q_std": float(np.std(flat_q)),
        "p95": float(np.percentile(flat_q, 95)),
        "p99": float(np.percentile(flat_q, 99)),
    }

    # Action margin stats
    all_delta_q = np.array(all_delta_q_list)
    action_margins_stats = {
        "mean_delta_q": float(np.mean(all_delta_q)),
        "median_delta_q": float(np.median(all_delta_q)),
        "p10_delta_q": float(np.percentile(all_delta_q, 10)),
        "p1_delta_q": float(np.percentile(all_delta_q, 1)),
        "fraction_near_zero": float(np.mean(all_delta_q < 0.05)),
    }

    # 2. Fixed Deterministic Probe Batch Evaluation
    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]
    burn_in = int(probe_data.get("burn_in", 8))

    with torch.no_grad():
        h_pr = model.init_hidden(probe_obs.size(0), device)
        q_pr, aux_pr, _ = model(probe_obs, h_pr)
        q_pr_abl, _, _ = model(probe_obs, h_pr, ablate_dwell_branch=True)

        h_tg = target_model.init_hidden(probe_obs.size(0), device)
        q_tg, _, _ = target_model(probe_obs, h_tg)

        h_base = base_model.init_hidden(probe_obs.size(0), device)
        q_base_all, _, _ = base_model(probe_obs, h_base)

    q_pr_g = q_pr[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()
    q_pr_abl_g = q_pr_abl[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()
    q_tg_g = q_tg[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()
    q_base_g = q_base_all[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()

    probe_acts = np.argmax(q_pr_g, axis=-1)
    probe_acts_abl = np.argmax(q_pr_abl_g, axis=-1)
    probe_base_acts = np.argmax(q_base_g, axis=-1)

    probe_d_flips = int(np.sum(probe_acts != probe_acts_abl))
    probe_d_flip_rate = float(probe_d_flips / len(probe_acts))

    # Flips vs Gate 27k
    flips_vs_gate27 = int(np.sum(probe_acts != probe_base_acts))
    flip_rate_vs_gate27 = float(flips_vs_gate27 / len(probe_acts))
    rhos = [spearmanr(q_base_g[i], q_pr_g[i]).correlation for i in range(len(probe_acts))]
    mean_rho = float(np.mean(rhos))

    # Mode and band level flips vs Gate 27k
    probe_bands = probe_acts // CANONICAL_N_MODES
    base_bands = probe_base_acts // CANONICAL_N_MODES
    band_flips = int(np.sum(probe_bands != base_bands))

    probe_modes = probe_acts % CANONICAL_N_MODES
    base_modes = probe_base_acts % CANONICAL_N_MODES
    mode_flips = int(np.sum(probe_modes != base_modes))

    # Online vs Target divergence on probe
    ot_gaps = np.abs(q_pr_g - q_tg_g)
    online_target_stats = {
        "mean_gap": float(np.mean(ot_gaps)),
        "p95_gap": float(np.percentile(ot_gaps, 95)),
        "max_gap": float(np.max(ot_gaps)),
    }

    # 3. TD-Error Distribution & Return Calibration on Replay Buffer
    buf = SequenceReplayBuffer(50000, 16, CANONICAL_OBS_DIM, 8, 42)
    buf.load_episodes(PRELOADED_BUFFER_PATH)

    with open(PRELOADED_BUFFER_PATH, "rb") as f:
        import pickle
        buf_data = pickle.load(f)

    episodes = buf_data["episodes"]
    all_pred_q = []
    all_realized_g = []
    all_td_errors = []

    with torch.no_grad():
        for ep in episodes:
            obs_seq = torch.tensor(ep["obs"], dtype=torch.float32, device=device).unsqueeze(0)
            next_obs_seq = torch.tensor(ep["next_obs"], dtype=torch.float32, device=device).unsqueeze(0)
            acts = ep["actions"]
            rews = ep["rewards"]
            dones = ep["dones"]
            g_seq = compute_mc_returns(rews, gamma=0.99)

            h = model.init_hidden(1, device)
            q_out, _, _ = model(obs_seq, h)
            q_ch = q_out.gather(-1, torch.tensor(acts, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1)).squeeze().cpu().numpy()

            h_on_next = model.init_hidden(1, device)
            next_q_on, _, _ = model(next_obs_seq, h_on_next)
            best_a = next_q_on.argmax(dim=-1, keepdim=True)

            h_tg_next = target_model.init_hidden(1, device)
            next_q_tg, _, _ = target_model(next_obs_seq, h_tg_next)
            next_q_val = next_q_tg.gather(-1, best_a).squeeze().cpu().numpy()

            y = rews + 0.99 * next_q_val * (1.0 - dones)
            td_err = np.abs(q_ch - y)

            all_pred_q.extend(q_ch)
            all_realized_g.extend(g_seq)
            all_td_errors.extend(td_err)

    all_pred_q = np.array(all_pred_q)
    all_realized_g = np.array(all_realized_g)
    all_td_errors = np.array(all_td_errors)

    # TD Error Stats
    td_error_stats = {
        "mean_abs_td": float(np.mean(all_td_errors)),
        "std_abs_td": float(np.std(all_td_errors)),
        "p95_abs_td": float(np.percentile(all_td_errors, 95)),
        "p99_abs_td": float(np.percentile(all_td_errors, 99)),
        "max_abs_td": float(np.max(all_td_errors)),
        "fraction_extreme_td": float(np.mean(all_td_errors > 10.0)),
    }

    # Calibration Stats
    reg = linregress(all_pred_q, all_realized_g)
    pred_errors = all_pred_q - all_realized_g
    abs_errors = np.abs(pred_errors)

    return_calibration_stats = {
        "mean_pred_q": float(np.mean(all_pred_q)),
        "mean_realized_g": float(np.mean(all_realized_g)),
        "calibration_bias": float(np.mean(pred_errors)),
        "calibration_slope": float(reg.slope),
        "calibration_intercept": float(reg.intercept),
        "calibration_r_squared": float(reg.rvalue ** 2),
        "mean_absolute_error": float(np.mean(abs_errors)),
        "p95_absolute_error": float(np.percentile(abs_errors, 95)),
    }

    c29_long_rate = float(c29_long_action_count / max(1, c29_total_steps))
    c29_mean_delta_q = float(np.mean(c29_delta_q_long_list)) if c29_delta_q_long_list else 0.0

    return {
        "tier1_primary_operational_metrics": {
            "mean_pd": mean_pd,
            "agile_pd": agile_pd,
            "sparse_pd": sparse_pd,
            "dense_pd": dense_pd,
            "worst_case_scenario": worst_case_scen,
            "worst_case_pd": worst_case_pd,
            "total_blackouts": blackouts,
            "config_29_pd": scenario_results.get("config_29", {}).get("pd", 0.0),
            "global_ir_time_gross_hits_per_ms": global_ir_time,
            "total_mission_time_ms": total_mission_ms,
            "total_hits": all_hits,
            "global_band_switch_rate": global_switch_rate,
            "thrashing_cycle_occupancy": global_thrashing["cycle_occupancy_fraction"],
            "thrashing_failed": thrashing_failed,
        },
        "tier2_value_function_health": {
            "q_scale_distribution": q_scale_stats,
            "action_margins": action_margins_stats,
            "td_error_distribution": td_error_stats,
            "online_target_divergence": online_target_stats,
            "return_calibration": return_calibration_stats,
        },
        "tier3_policy_stability_diagnostics": {
            "probe_spearman_rho_vs_gate27": mean_rho,
            "probe_flips_vs_gate27": flips_vs_gate27,
            "probe_flip_rate_vs_gate27_pct": flip_rate_vs_gate27 * 100.0,
            "probe_band_flips": band_flips,
            "probe_mode_flips": mode_flips,
            "probe_d_flips": probe_d_flips,
            "probe_d_flip_rate": probe_d_flip_rate,
            "short_pct": float(all_m_dist[0] * 100.0),
            "normal_pct": float(all_m_dist[1] * 100.0),
            "long_pct": float(all_m_dist[2] * 100.0),
            "revisit_pct": float(all_m_dist[3] * 100.0),
            "preemptive_pct": float(all_m_dist[4] * 100.0),
            "config_29_long_rate": c29_long_rate,
            "config_29_mean_delta_q_long": c29_mean_delta_q,
        },
        "scenario_results": scenario_results,
        "scenario_dwell_breakdown": scenario_dwell_breakdown,
    }


def check_hard_stopping_rules(eval_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Evaluates the immediate hard operational stopping rules."""
    t1 = eval_data["tier1_primary_operational_metrics"]
    reasons = []

    if t1["mean_pd"] < 80.0:
        reasons.append(f"Mean Pd {t1['mean_pd']:.2f}% < 80.0% floor")
    if t1["agile_pd"] < 65.0:
        reasons.append(f"Agile Pd {t1['agile_pd']:.2f}% < 65.0% floor")
    if t1["dense_pd"] < 95.0:
        reasons.append(f"Dense Pd {t1['dense_pd']:.2f}% < 95.0% floor")
    if t1["config_29_pd"] < 85.0:
        reasons.append(f"config_29 Pd {t1['config_29_pd']:.2f}% < 85.0% floor")
    if t1["worst_case_pd"] < 30.0:
        reasons.append(f"Worst-case Pd {t1['worst_case_pd']:.2f}% < 30.0% floor ({t1['worst_case_scenario']})")
    if t1["total_blackouts"] > 0:
        reasons.append(f"Total blackouts {t1['total_blackouts']} > 0 floor")
    if t1["global_ir_time_gross_hits_per_ms"] < 0.85:
        reasons.append(f"Gross hits/ms {t1['global_ir_time_gross_hits_per_ms']:.4f} < 0.85 floor")
    if t1["thrashing_cycle_occupancy"] >= 0.05:
        reasons.append(f"Thrashing cycle occupancy {t1['thrashing_cycle_occupancy']*100:.2f}% >= 5.0% ceiling")

    should_stop = len(reasons) > 0
    return should_stop, reasons


def main():
    parser = argparse.ArgumentParser(description="Phase G8.5-FINAL Longitudinal Trial Runner")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("   PHASE G8.5-FINAL: UNREGULARIZED LONGITUDINAL STABILITY TRIAL  ")
    logger.info("   STEPS 27,000 -> 50,000 (CANONICAL gamma=0.99, q_reg=0.0)      ")
    logger.info("=================================================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    targeted_spec = {
        "scenario_id": "config_29",
        "mode": 2,
        "target_fraction": 0.15,
        "min_episodes": 2,
    }

    current_checkpoint = GATE_27000_PATH
    current_sha = EXPECTED_GATE_27000_SHA
    current_step = 27000

    trial_history = []

    for gate_step in STAGED_GATES:
        logger.info("=================================================================")
        logger.info(">>> STAGE: Executing Steps %d -> %d (%d updates) <<<",
                    current_step, gate_step, (gate_step - current_step) // 4)
        logger.info("=================================================================")

        gate_ckpt = OUTPUT_DIR / f"checkpoint_gate_{gate_step}.pt"
        if not gate_ckpt.exists():
            gate_ckpt = OUTPUT_DIR / f"checkpoint_step_{gate_step}.pt"

        if not gate_ckpt.exists():
            train_scheduler(
                model_cfg_path="configs/model_config.yaml",
                train_cfg_path="configs/training_config.yaml",
                output_dir_override=str(OUTPUT_DIR),
                resume_checkpoint=str(current_checkpoint),
                expected_parent_sha=current_sha if current_step == 27000 else None,
                stop_at_step=gate_step,
                start_step_override=current_step,
                staged_gates=[gate_step],
                fresh_optimizer=False,
                use_g8_film=True,
                objective_mode="step_based",
                gamma=0.99,  # Canonical discount!
                c_dwell=0.0,
                tau_ref=1.0,
                q_reg_coef=0.0,  # Unregularized!
                use_online_reward_shaper=False,
                preloaded_buffer_path=str(PRELOADED_BUFFER_PATH),
                probe_batch_path=str(PROBE_BATCH_PATH),
                override_epsilon=0.05,
                learning_rate=2.5e-5,
                lambda_entropy=0.05,
                targeted_spec=targeted_spec,
            )
            gate_ckpt = OUTPUT_DIR / f"checkpoint_gate_{gate_step}.pt"
            if not gate_ckpt.exists():
                gate_ckpt = OUTPUT_DIR / f"checkpoint_step_{gate_step}.pt"
            assert gate_ckpt.exists(), f"Checkpoint missing after stage: {gate_ckpt}"
        else:
            logger.info("Checkpoint %s already exists. Skipping training stage.", gate_ckpt)

        # Run Comprehensive Evaluation
        logger.info("Evaluating Gate Checkpoint: %s", gate_ckpt)
        eval_data = evaluate_checkpoint_g8_5(gate_ckpt, device=torch.device(args.device))
        eval_data["gate_step"] = gate_step

        should_stop, stop_reasons = check_hard_stopping_rules(eval_data)
        eval_data["operational_disposition"] = "HALT_TRIGGERED" if should_stop else "PASSED"
        eval_data["operational_stop_reasons"] = stop_reasons

        # Save individual gate report
        gate_report_file = REPORTS_DIR / f"g8_5_eval_step_{gate_step}.json"
        with open(gate_report_file, "w") as f:
            json.dump({"checkpoint": str(gate_ckpt), **eval_data}, f, indent=2)
        logger.info("Saved Gate %d report to: %s", gate_step, gate_report_file)

        t1 = eval_data["tier1_primary_operational_metrics"]
        t2 = eval_data["tier2_value_function_health"]
        t3 = eval_data["tier3_policy_stability_diagnostics"]

        logger.info("-----------------------------------------------------------------")
        logger.info("   GATE %d SUMMARY:", gate_step)
        logger.info("   Operational Disposition: %s", eval_data["operational_disposition"])
        if should_stop:
            for r in stop_reasons:
                logger.info("     * STOP REASON: %s", r)
        logger.info("   Mean Pd: %.2f%% | Agile Pd: %.2f%% | Dense Pd: %.2f%% | c29 Pd: %.2f%%",
                    t1["mean_pd"], t1["agile_pd"], t1["dense_pd"], t1["config_29_pd"])
        logger.info("   Worst-Case: %.2f%% (%s) | Blackouts: %d | Hits/ms: %.4f | Thrash: %.2f%%",
                    t1["worst_case_pd"], t1["worst_case_scenario"], t1["total_blackouts"],
                    t1["global_ir_time_gross_hits_per_ms"], t1["thrashing_cycle_occupancy"] * 100.0)
        logger.info("   Q_Scale: max=%.2f, mean=%.2f, p95=%.2f, p99=%.2f",
                    t2["q_scale_distribution"]["q_max"], t2["q_scale_distribution"]["q_mean"],
                    t2["q_scale_distribution"]["p95"], t2["q_scale_distribution"]["p99"])
        logger.info("   Action Margins (Delta_Q): mean=%.4f, median=%.4f, p10=%.4f, near_zero=%.2f%%",
                    t2["action_margins"]["mean_delta_q"], t2["action_margins"]["median_delta_q"],
                    t2["action_margins"]["p10_delta_q"], t2["action_margins"]["fraction_near_zero"] * 100.0)
        logger.info("   TD-Error Tail: mean=%.4f, p95=%.4f, max=%.4f, extreme(>10)=%.2f%%",
                    t2["td_error_distribution"]["mean_abs_td"], t2["td_error_distribution"]["p95_abs_td"],
                    t2["td_error_distribution"]["max_abs_td"], t2["td_error_distribution"]["fraction_extreme_td"] * 100.0)
        logger.info("   Return Calibration: bias=%.2f, slope=%.4f, R^2=%.4f, MAE=%.2f",
                    t2["return_calibration"]["calibration_bias"], t2["return_calibration"]["calibration_slope"],
                    t2["return_calibration"]["calibration_r_squared"], t2["return_calibration"]["mean_absolute_error"])
        logger.info("   Probe Policy: rho=%.6f, flips=%d/512 (%.1f%%), band_flips=%d, mode_flips=%d",
                    t3["probe_spearman_rho_vs_gate27"], t3["probe_flips_vs_gate27"],
                    t3["probe_flip_rate_vs_gate27_pct"], t3["probe_band_flips"], t3["probe_mode_flips"])
        logger.info("-----------------------------------------------------------------")

        trial_history.append({
            "gate_step": gate_step,
            "checkpoint": str(gate_ckpt),
            "disposition": eval_data["operational_disposition"],
            "stop_reasons": stop_reasons,
            "mean_pd": t1["mean_pd"],
            "agile_pd": t1["agile_pd"],
            "dense_pd": t1["dense_pd"],
            "config_29_pd": t1["config_29_pd"],
            "worst_case_pd": t1["worst_case_pd"],
            "blackouts": t1["total_blackouts"],
            "gross_hits_per_ms": t1["global_ir_time_gross_hits_per_ms"],
            "thrashing_cycle_occupancy": t1["thrashing_cycle_occupancy"],
            "q_max": t2["q_scale_distribution"]["q_max"],
            "q_mean": t2["q_scale_distribution"]["q_mean"],
            "mean_delta_q": t2["action_margins"]["mean_delta_q"],
            "probe_rho": t3["probe_spearman_rho_vs_gate27"],
            "probe_flips": t3["probe_flips_vs_gate27"],
        })

        if should_stop:
            logger.warning("HARD STOP RULE TRIGGERED at Gate %d! Halting longitudinal trial.", gate_step)
            break

        # Advance continuation pointers
        current_checkpoint = gate_ckpt
        current_step = gate_step
        current_sha = None

    # Write summary report
    summary_file = REPORTS_DIR / "g8_5_final_trial_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "status": "COMPLETED" if len(trial_history) == len(STAGED_GATES) and trial_history[-1]["disposition"] == "PASSED" else "HALTED",
            "trial_history": trial_history,
        }, f, indent=2)
    logger.info("Trial summary saved to: %s", summary_file)
    logger.info("=================================================================")
    logger.info("   PHASE G8.5-FINAL TRIAL FINISHED.                              ")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
