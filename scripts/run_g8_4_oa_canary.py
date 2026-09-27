"""Phase G8.4-OA Horizon-Calibrated Objective Candidate Canary Runner.

Executes Phase G8.4-OA: Steps 27,000 -> 28,000 under gamma = 0.95.
Continuation Contract: EXACT-CONTINUATION STATE + OBJECTIVE SWITCH
- Checkpoint: Frozen Gate 27,000 (SHA: fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094)
- fresh_optimizer=False (inherits Adam m and v moment states, step count, target net)
- Objective: gamma = 0.95, objective_mode="step_based", c_dwell=0.0, tau_ref=1.0, q_reg=0.0
- Intermediate Diagnostic Observation at Step 27,500 (read-only, non-halting)
- Hard Acceptance Gate at Step 28,000 (Q_max <= 35.0 ceiling)
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
from scipy.stats import entropy
import torch

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
from ew_core.training.train_scheduler import train_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_g8_4_oa_canary")

GATE_27000_PATH = repo_root / "experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt"
EXPECTED_GATE_27000_SHA = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g8_4_oa_candidate_objective"
VAL_DIR = Path("D:/TSRD/stare/val_stare")
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
    n_total = max(1, int(np.sum(mode_counts)))
    p_025 = float(mode_counts[0]) / n_total
    p_100 = float(mode_counts[1] + mode_counts[3] + mode_counts[4]) / n_total
    p_250 = float(mode_counts[2]) / n_total
    probs = np.array([p_025, p_100, p_250])
    nz = probs[probs > 1e-12]
    h_val = float(-np.sum(nz * np.log(nz))) if len(nz) > 0 else 0.0
    return h_val, {"0.25": p_025, "1.0": p_100, "2.5": p_250}


def detect_alternating_cycles(actions: List[int] | np.ndarray) -> Dict[str, Any]:
    acts = list(actions)
    n = len(acts)
    if n < 4:
        return {"max_cycle_length": 0, "cycle_occupancy_fraction": 0.0, "cycle_steps": 0, "total_steps": n}

    in_cycle = np.zeros(n, dtype=bool)
    max_len = 0
    i = 0
    while i < n - 3:
        if acts[i] != acts[i + 1] and acts[i] == acts[i + 2] and acts[i + 1] == acts[i + 3]:
            a = acts[i]
            b = acts[i + 1]
            j = i + 2
            while j < n:
                expected = a if (j - i) % 2 == 0 else b
                if acts[j] == expected:
                    j += 1
                else:
                    break
            cycle_len = j - i
            if cycle_len >= 4:
                in_cycle[i:j] = True
                if cycle_len > max_len:
                    max_len = cycle_len
            i = j
        else:
            i += 1

    cycle_steps = int(np.sum(in_cycle))
    occupancy = float(cycle_steps / max(1, n))
    return {
        "max_cycle_length": int(max_len),
        "cycle_occupancy_fraction": float(occupancy),
        "cycle_steps": cycle_steps,
        "total_steps": int(n),
    }


def compute_normalized_coupling_cd(
    q_vals: torch.Tensor,
    d_tilde: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    B = q_vals.size(0)
    q_grid = q_vals.view(B, CANONICAL_N_BANDS, CANONICAL_N_MODES)
    top_band_modes = q_grid.max(dim=-1).values
    b_star = top_band_modes.argmax(dim=-1)

    cd_within_band = []
    for i in range(B):
        b = b_star[i].item()
        q_band = q_grid[i, b]
        top2_modes = torch.topk(q_band, 2).indices
        m1, m2 = top2_modes[0].item(), top2_modes[1].item()
        d1 = d_tilde[i, b, m1].item()
        d2 = d_tilde[i, b, m2].item()
        q1 = q_band[m1].item()
        q2 = q_band[m2].item()
        cd = abs(d1 - d2) / (abs(q1 - q2) + eps)
        cd_within_band.append(cd)

    cd_global = []
    for i in range(B):
        top2_acts = torch.topk(q_vals[i], 2).indices
        a1, a2 = top2_acts[0].item(), top2_acts[1].item()
        b1, m1 = a1 // CANONICAL_N_MODES, a1 % CANONICAL_N_MODES
        b2, m2 = a2 // CANONICAL_N_MODES, a2 % CANONICAL_N_MODES
        d1 = d_tilde[i, b1, m1].item()
        d2 = d_tilde[i, b2, m2].item()
        q1 = q_vals[i, a1].item()
        q2 = q_vals[i, a2].item()
        cd = abs(d1 - d2) / (abs(q1 - q2) + eps)
        cd_global.append(cd)

    return {
        "cd_within_top_band_mean": float(np.mean(cd_within_band)),
        "cd_within_top_band_median": float(np.median(cd_within_band)),
        "cd_global_top2_mean": float(np.mean(cd_global)),
        "cd_global_top2_median": float(np.median(cd_global)),
    }


def evaluate_checkpoint_g8_4_oa(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    logger.info("Evaluating Checkpoint: %s", checkpoint_path)
    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("online_drqn", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Deterministic Probe Audit
    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]
    burn_in = int(probe_data.get("burn_in", 8))
    with torch.no_grad():
        h_pr = model.init_hidden(probe_obs.size(0), device)
        q_pr, aux_pr, _ = model(probe_obs, h_pr)
        q_pr_abl, _, _ = model(probe_obs, h_pr, ablate_dwell_branch=True)

    q_pr_g = q_pr[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS)
    q_pr_abl_g = q_pr_abl[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS)
    d_tilde_pr_g = aux_pr["d_tilde"][:, burn_in:, :, :].reshape(-1, CANONICAL_N_BANDS, CANONICAL_N_MODES)

    act_pr_g = q_pr_g.argmax(dim=-1)
    act_pr_abl_g = q_pr_abl_g.argmax(dim=-1)
    probe_d_flips = int((act_pr_g != act_pr_abl_g).sum().item())
    probe_d_flip_rate = float(probe_d_flips / max(1, act_pr_g.numel()))
    probe_cd = compute_normalized_coupling_cd(q_pr_g, d_tilde_pr_g)

    # 10 Scenario Evaluations
    scenario_results: Dict[str, Any] = {}
    scenario_dwell_breakdown: Dict[str, Any] = {}
    all_actions = []
    all_bands = []
    all_modes = []
    all_q_values = []
    all_hits = 0
    all_dwell_us = 0.0
    all_band_switches = 0
    d_action_flips_global = 0

    c29_delta_q_long_list = []
    c29_pref_long_count = 0
    c29_long_action_count = 0
    c29_total_steps = 0

    for scen_name in CANONICAL_SCENARIOS:
        h5_path = VAL_DIR / f"{scen_name}.h5"
        records = load_h5_records(h5_path, chunk_mode="first")
        env = CognitiveRFScanEnv(records=records)
        obs, _ = env.reset()
        hidden = model.init_hidden(1, device)

        scen_acts = []
        scen_bnds = []
        scen_mods = []
        scen_hits = 0
        scen_dwell_us = 0.0
        scen_band_switches = 0
        first_hit_latency_us = None

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_step, aux, hidden = model(obs_tensor, hidden)
                q_abl, _, _ = model(obs_tensor, hidden, ablate_dwell_branch=True)

            q_row = q_step.squeeze().cpu().numpy()
            q_row_abl = q_abl.squeeze().cpu().numpy()
            act = int(np.argmax(q_row))
            act_abl = int(np.argmax(q_row_abl))
            if act != act_abl:
                d_action_flips_global += 1

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
    global_ir_step = float(all_hits / max(1, len(all_actions)))
    global_band_switch_rate = float(all_band_switches / max(1, len(all_actions) - len(CANONICAL_SCENARIOS)))

    global_thrashing = detect_alternating_cycles(all_actions)
    all_b_counts = np.bincount(all_bands, minlength=CANONICAL_N_BANDS)
    global_distinct_bands = int(np.count_nonzero(all_b_counts))
    global_top_band_conc = float(np.max(all_b_counts) / max(1, len(all_bands)))

    thrashing_failed = bool(
        global_thrashing["max_cycle_length"] >= 20
        or global_thrashing["cycle_occupancy_fraction"] >= 0.25
        or global_top_band_conc > 0.35
        or global_distinct_bands < 30
    )

    mode_counts = np.bincount(all_modes, minlength=5)
    mode_dist = mode_counts / max(1, len(all_modes))
    h_mode = float(entropy(mode_dist + 1e-12))
    h_tau, tau_dist = compute_htau(mode_counts)

    all_q_arr = np.array(all_q_values)
    q_max = float(all_q_arr.max())
    q_mean = float(all_q_arr.mean())
    q_std = float(all_q_arr.std())

    c29_pd = scenario_results.get("config_29", {}).get("pd", 0.0)
    c29_long_rate = float(c29_long_action_count / max(1, c29_total_steps))
    c29_delta_q_long_mean = float(np.mean(c29_delta_q_long_list)) if c29_delta_q_long_list else 0.0

    return {
        "tier1_primary_metrics": {
            "mean_pd": mean_pd,
            "agile_pd": agile_pd,
            "sparse_pd": sparse_pd,
            "dense_pd": dense_pd,
            "worst_case_scenario": worst_case_scen,
            "worst_case_pd": worst_case_pd,
            "total_blackouts": blackouts,
            "config_29_pd": c29_pd,
            "global_ir_time_gross_hits_per_ms": global_ir_time,
            "total_mission_time_ms": total_mission_ms,
            "total_hits": all_hits,
        },
        "tier2_secondary_metrics": {
            "global_band_switch_rate": global_band_switch_rate,
        },
        "tier3_diagnostic_metrics": {
            "global_ir_step": global_ir_step,
            "h_mode": h_mode,
            "h_tau": h_tau,
            "tau_distribution": tau_dist,
            "short_pct": float(mode_dist[0] * 100.0),
            "normal_pct": float(mode_dist[1] * 100.0),
            "long_pct": float(mode_dist[2] * 100.0),
            "revisit_pct": float(mode_dist[3] * 100.0),
            "preemptive_pct": float(mode_dist[4] * 100.0),
            "q_max": q_max,
            "q_mean": q_mean,
            "q_std": q_std,
            "probe_d_flips": probe_d_flips,
            "probe_d_flip_rate": probe_d_flip_rate,
            "probe_normalized_coupling_cd": probe_cd,
            "config_29_long_rate": c29_long_rate,
            "config_29_mean_delta_q_long": c29_delta_q_long_mean,
        },
        "thrashing_audit": {
            "thrashing_failed": thrashing_failed,
            "max_cycle_length": global_thrashing["max_cycle_length"],
            "cycle_occupancy_fraction": global_thrashing["cycle_occupancy_fraction"],
            "distinct_bands_scheduled": global_distinct_bands,
            "top_band_concentration": global_top_band_conc,
        },
        "scenario_results": scenario_results,
        "scenario_dwell_breakdown": scenario_dwell_breakdown,
    }


def evaluate_gate_28k_criteria(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    t1 = metrics["tier1_primary_metrics"]
    t3 = metrics["tier3_diagnostic_metrics"]
    thr = metrics["thrashing_audit"]
    reasons = []

    if t1["mean_pd"] < 80.0:
        reasons.append(f"Mean Pd {t1['mean_pd']:.2f}% < 80.0% floor")
    if t1["agile_pd"] < 65.0:
        reasons.append(f"Agile Pd {t1['agile_pd']:.2f}% < 65.0% floor")
    if t1["config_29_pd"] < 85.0:
        reasons.append(f"config_29 Pd {t1['config_29_pd']:.2f}% < 85.0% floor")
    if t1["worst_case_pd"] < 30.0:
        reasons.append(f"Worst-case Pd ({t1['worst_case_scenario']}) {t1['worst_case_pd']:.2f}% < 30.0% floor")
    if t1["total_blackouts"] > 0:
        reasons.append(f"{t1['total_blackouts']} scenario blackouts detected")
    if t1["global_ir_time_gross_hits_per_ms"] < 0.85:
        reasons.append(f"IR_time {t1['global_ir_time_gross_hits_per_ms']:.4f} < 0.85 gross hits/ms floor")
    if t3["q_max"] > 35.0:
        reasons.append(f"Q_max {t3['q_max']:.2f} > 35.0 ceiling")
    if t3["probe_d_flip_rate"] < 0.01:
        reasons.append(f"Probe D-ablation flip rate {t3['probe_d_flip_rate']*100:.2f}% < 1.0% floor")
    if thr["thrashing_failed"]:
        reasons.append("Pathological thrashing failure detected")

    passed = len(reasons) == 0
    return passed, reasons


def main():
    parser = argparse.ArgumentParser(description="Phase G8.4-OA Horizon-Calibrated Canary Runner")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("   PHASE G8.4-OA CANARY: STEPS 27,000 -> 28,000 (gamma = 0.95)  ")
    logger.info("=================================================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targeted_spec = {
        "scenario_id": "config_29",
        "mode": 2,
        "target_fraction": 0.15,
        "min_episodes": 2,
    }

    # -------------------------------------------------------------
    # PART 1: Steps 27,000 -> 27,500
    # -------------------------------------------------------------
    logger.info("--- PART 1: Executing Steps 27,000 -> 27,500 (125 updates) ---")
    train_scheduler(
        model_cfg_path="configs/model_config.yaml",
        train_cfg_path="configs/training_config.yaml",
        output_dir_override=str(OUTPUT_DIR),
        resume_checkpoint=str(GATE_27000_PATH),
        expected_parent_sha=EXPECTED_GATE_27000_SHA,
        stop_at_step=27500,
        start_step_override=27000,
        staged_gates=[27500],
        fresh_optimizer=False,
        use_g8_film=True,
        objective_mode="step_based",
        gamma=0.95,  # Horizon-Calibrated Objective!
        c_dwell=0.0,
        tau_ref=1.0,
        use_online_reward_shaper=False,
        preloaded_buffer_path=str(PRELOADED_BUFFER_PATH),
        probe_batch_path=str(PROBE_BATCH_PATH),
        override_epsilon=0.05,
        learning_rate=2.5e-5,
        lambda_entropy=0.05,
        targeted_spec=targeted_spec,
    )

    ckpt_27500 = OUTPUT_DIR / "checkpoint_gate_27500.pt"
    if not ckpt_27500.exists():
        ckpt_27500 = OUTPUT_DIR / "checkpoint_step_27500.pt"
    assert ckpt_27500.exists(), f"Step 27,500 checkpoint missing: {ckpt_27500}"

    logger.info("Running Intermediate Diagnostic Observation at Step 27,500...")
    eval_27500 = evaluate_checkpoint_g8_4_oa(ckpt_27500, device=torch.device(args.device))
    eval_27500["observation_step"] = 27500
    out_27500_report = REPORTS_DIR / "g8_4_oa_eval_step_27500.json"
    with open(out_27500_report, "w") as f:
        json.dump({"checkpoint": str(ckpt_27500), **eval_27500}, f, indent=2)

    logger.info("=================================================================")
    logger.info("   INTERMEDIATE 27,500 OBSERVATION SUMMARY:")
    logger.info("   Mean Pd: %.2f%% | Agile Pd: %.2f%% | c29 Pd: %.2f%%",
                eval_27500["tier1_primary_metrics"]["mean_pd"],
                eval_27500["tier1_primary_metrics"]["agile_pd"],
                eval_27500["tier1_primary_metrics"]["config_29_pd"])
    logger.info("   Q_max: %.2f | Q_mean: %.2f | Q_std: %.2f",
                eval_27500["tier3_diagnostic_metrics"]["q_max"],
                eval_27500["tier3_diagnostic_metrics"]["q_mean"],
                eval_27500["tier3_diagnostic_metrics"]["q_std"])
    logger.info("   IR_time: %.4f hits/ms | Blackouts: %d",
                eval_27500["tier1_primary_metrics"]["global_ir_time_gross_hits_per_ms"],
                eval_27500["tier1_primary_metrics"]["total_blackouts"])
    logger.info("   Saved intermediate report to %s", out_27500_report)
    logger.info("=================================================================")

    # -------------------------------------------------------------
    # PART 2: Steps 27,500 -> 28,000
    # -------------------------------------------------------------
    logger.info("--- PART 2: Executing Steps 27,500 -> 28,000 (125 updates) ---")
    train_scheduler(
        model_cfg_path="configs/model_config.yaml",
        train_cfg_path="configs/training_config.yaml",
        output_dir_override=str(OUTPUT_DIR),
        resume_checkpoint=str(ckpt_27500),
        expected_parent_sha=None,
        stop_at_step=28000,
        start_step_override=27500,
        staged_gates=[28000],
        fresh_optimizer=False,
        use_g8_film=True,
        objective_mode="step_based",
        gamma=0.95,  # Horizon-Calibrated Objective!
        c_dwell=0.0,
        tau_ref=1.0,
        use_online_reward_shaper=False,
        preloaded_buffer_path=str(PRELOADED_BUFFER_PATH),
        probe_batch_path=str(PROBE_BATCH_PATH),
        override_epsilon=0.05,
        learning_rate=2.5e-5,
        lambda_entropy=0.05,
        targeted_spec=targeted_spec,
    )

    ckpt_28000 = OUTPUT_DIR / "checkpoint_gate_28000.pt"
    if not ckpt_28000.exists():
        ckpt_28000 = OUTPUT_DIR / "checkpoint_step_28000.pt"
    assert ckpt_28000.exists(), f"Gate 28,000 checkpoint missing: {ckpt_28000}"

    logger.info("Running Gate 28,000 Qualification Evaluation...")
    eval_28000 = evaluate_checkpoint_g8_4_oa(ckpt_28000, device=torch.device(args.device))
    passed, reasons = evaluate_gate_28k_criteria(eval_28000)
    eval_28000["gate_evaluation"] = {
        "gate_step": 28000,
        "passed": passed,
        "failure_reasons": reasons,
    }

    out_28000_report = REPORTS_DIR / "g8_4_oa_eval_step_28000.json"
    with open(out_28000_report, "w") as f:
        json.dump({"checkpoint": str(ckpt_28000), **eval_28000}, f, indent=2)

    logger.info("=================================================================")
    logger.info("   GATE 28,000 EVALUATION SUMMARY: %s", "PASSED" if passed else "FAILED / QUARANTINED")
    logger.info("   Mean Pd: %.2f%% | Agile Pd: %.2f%% | Worst-Case (%s): %.2f%%",
                eval_28000["tier1_primary_metrics"]["mean_pd"],
                eval_28000["tier1_primary_metrics"]["agile_pd"],
                eval_28000["tier1_primary_metrics"]["worst_case_scenario"],
                eval_28000["tier1_primary_metrics"]["worst_case_pd"])
    logger.info("   config_29 Pd: %.2f%% | Blackouts: %d",
                eval_28000["tier1_primary_metrics"]["config_29_pd"],
                eval_28000["tier1_primary_metrics"]["total_blackouts"])
    logger.info("   Q_max: %.2f | Q_mean: %.2f | Q_std: %.2f",
                eval_28000["tier3_diagnostic_metrics"]["q_max"],
                eval_28000["tier3_diagnostic_metrics"]["q_mean"],
                eval_28000["tier3_diagnostic_metrics"]["q_std"])
    logger.info("   IR_time (gross hits/ms): %.4f",
                eval_28000["tier1_primary_metrics"]["global_ir_time_gross_hits_per_ms"])
    logger.info("   Thrashing Audit Failed: %s",
                eval_28000["thrashing_audit"]["thrashing_failed"])
    logger.info("   Saved Gate 28k report to %s", out_28000_report)
    logger.info("=================================================================")

    if not passed:
        logger.error("GATE 28,000 CRITERIA FAILED: %s", reasons)
        sys.exit(1)


if __name__ == "__main__":
    main()
