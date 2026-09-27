"""Phase G8.4-REG Value Regularization Diagnostic Canary Runner.

Executes Phase G8.4-REG: Steps 27,000 -> 28,000 under canonical gamma = 0.99 with lambda = 0.005.
Continuation Contract: EXACT-CONTINUATION STATE + OBJECTIVE SWITCH
- Checkpoint: Frozen Gate 27,000 (SHA: fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094)
- fresh_optimizer=False (inherits Adam m and v moment states, step count, target net)
- Objective: gamma = 0.99, objective_mode="step_based", c_dwell=0.0, tau_ref=1.0, q_reg_coef=0.005
- Intermediate Diagnostic Observation at Step 27,500 (read-only, non-halting)
- Hard Acceptance Gate at Step 28,000 (Q_max <= 35.0, Mean Pd >= 80%, Agile Pd >= 65%, c29 Pd >= 85%, Blackouts == 0)
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
from scipy.stats import entropy, spearmanr
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
logger = logging.getLogger("run_g8_4_reg_canary")

GATE_27000_PATH = repo_root / "experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt"
EXPECTED_GATE_27000_SHA = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g8_4_reg_diagnostic"
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


def evaluate_checkpoint_g8_4_reg(
    checkpoint_path: Path,
    device: torch.device,
    val_dir: Path = VAL_DIR,
    max_steps_per_scenario: int = 2000,
) -> Dict[str, Any]:
    """Evaluates checkpoint across 10 canonical scenarios + diagnostic probe batch."""
    logger.info("Evaluating Checkpoint: %s", checkpoint_path)
    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load baseline Gate 27k model to measure probe rank correlation and flips
    base_ckpt = torch.load(GATE_27000_PATH, map_location=device, weights_only=False)
    base_model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    base_model.load_state_dict(base_ckpt["state_dict"])
    base_model.eval()

    scenario_results: Dict[str, Any] = {}
    scenario_dwell_breakdown: Dict[str, Any] = {}

    all_actions: List[int] = []
    all_bands: List[int] = []
    all_modes: List[int] = []
    all_q_values: List[np.ndarray] = []

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
        h5_path = VAL_DIR / f"{scen_name}.h5"
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

        for step in range(max_steps_per_scenario):
            total_env_steps += 1
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

    all_q_matrix = np.array(all_q_values)
    q_max_val = float(np.max(all_q_matrix))
    q_mean_val = float(np.mean(all_q_matrix))
    q_std_val = float(np.std(all_q_matrix))

    # Evaluate probe batch
    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]
    burn_in = int(probe_data.get("burn_in", 8))

    with torch.no_grad():
        h_pr = model.init_hidden(probe_obs.size(0), device)
        q_pr, aux_pr, _ = model(probe_obs, h_pr)
        q_pr_abl, _, _ = model(probe_obs, h_pr, ablate_dwell_branch=True)

        # Baseline Q
        h_base = base_model.init_hidden(probe_obs.size(0), device)
        q_base_all, _, _ = base_model(probe_obs, h_base)

    q_pr_g = q_pr[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()
    q_pr_abl_g = q_pr_abl[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()
    q_base_g = q_base_all[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()

    probe_acts = np.argmax(q_pr_g, axis=-1)
    probe_acts_abl = np.argmax(q_pr_abl_g, axis=-1)
    probe_base_acts = np.argmax(q_base_g, axis=-1)

    probe_d_flips = int(np.sum(probe_acts != probe_acts_abl))
    probe_d_flip_rate = float(probe_d_flips / len(probe_acts))

    # Rank correlation and flips vs Gate 27k
    flips_vs_gate27 = int(np.sum(probe_acts != probe_base_acts))
    flip_rate_vs_gate27 = float(flips_vs_gate27 / len(probe_acts))
    rhos = [spearmanr(q_base_g[i], q_pr_g[i]).correlation for i in range(len(probe_acts))]
    mean_rho = float(np.mean(rhos))

    c_d_within = []
    c_d_top2 = []
    for i in range(q_pr_g.shape[0]):
        row = q_pr_g[i].reshape(CANONICAL_N_BANDS, CANONICAL_N_MODES)
        row_abl = q_pr_abl_g[i].reshape(CANONICAL_N_BANDS, CANONICAL_N_MODES)
        b_top = int(np.argmax(np.max(row, axis=-1)))
        delta_top = float(np.max(np.abs(row[b_top] - row_abl[b_top])))
        span_top = float(np.ptp(row[b_top]))
        c_d_within.append(delta_top / max(1e-4, span_top + delta_top))

        b_top2 = np.argsort(np.max(row, axis=-1))[-2:]
        b_second = int(b_top2[0] if b_top2[1] == b_top else b_top2[1])
        delta_band = float(np.max(np.abs(row[b_top] - row[b_second])))
        c_d_top2.append(delta_top / max(1e-4, delta_band + delta_top))

    c29_long_rate = float(c29_long_action_count / max(1, c29_total_steps))
    c29_mean_delta_q = float(np.mean(c29_delta_q_long_list)) if c29_delta_q_long_list else 0.0

    global_thrashing = detect_alternating_cycles(all_actions)
    all_b_counts = np.bincount(all_bands, minlength=CANONICAL_N_BANDS)
    global_distinct_bands = int(np.count_nonzero(all_b_counts))
    global_top_band_conc = float(np.max(all_b_counts) / max(1, len(all_bands)))

    thrashing_failed = bool(
        global_thrashing["cycle_occupancy_fraction"] > 0.05
        or global_distinct_bands < 30
        or global_top_band_conc > 0.40
    )

    return {
        "tier1_primary_metrics": {
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
        },
        "tier2_secondary_metrics": {
            "global_band_switch_rate": global_switch_rate,
        },
        "tier3_diagnostic_metrics": {
            "global_ir_step": global_ir_step,
            "h_mode": float(entropy(all_m_dist + 1e-12)),
            "h_tau": global_htau,
            "tau_distribution": global_tau_dist,
            "short_pct": float(all_m_dist[0] * 100.0),
            "normal_pct": float(all_m_dist[1] * 100.0),
            "long_pct": float(all_m_dist[2] * 100.0),
            "revisit_pct": float(all_m_dist[3] * 100.0),
            "preemptive_pct": float(all_m_dist[4] * 100.0),
            "q_max": q_max_val,
            "q_mean": q_mean_val,
            "q_std": q_std_val,
            "probe_d_flips": probe_d_flips,
            "probe_d_flip_rate": probe_d_flip_rate,
            "probe_flips_vs_gate27": flips_vs_gate27,
            "probe_flip_rate_vs_gate27_pct": flip_rate_vs_gate27 * 100.0,
            "probe_spearman_rho_vs_gate27": mean_rho,
            "probe_normalized_coupling_cd": {
                "cd_within_top_band_mean": float(np.mean(c_d_within)),
                "cd_within_top_band_median": float(np.median(c_d_within)),
                "cd_global_top2_mean": float(np.mean(c_d_top2)),
                "cd_global_top2_median": float(np.median(c_d_top2)),
            },
            "config_29_long_rate": c29_long_rate,
            "config_29_mean_delta_q_long": c29_mean_delta_q,
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


def verify_gate_28000_acceptance(eval_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Evaluates the 5 mandatory acceptance criteria for Gate 28,000."""
    t1 = eval_data["tier1_primary_metrics"]
    t3 = eval_data["tier3_diagnostic_metrics"]
    thr = eval_data["thrashing_audit"]

    reasons = []
    if t1["mean_pd"] < 80.0:
        reasons.append(f"Mean Pd {t1['mean_pd']:.2f}% < 80.0% floor")
    if t1["agile_pd"] < 65.0:
        reasons.append(f"Agile Pd {t1['agile_pd']:.2f}% < 65.0% floor")
    if t1["config_29_pd"] < 85.0:
        reasons.append(f"config_29 Pd {t1['config_29_pd']:.2f}% < 85.0% floor")
    if t1["total_blackouts"] > 0:
        reasons.append(f"Total blackouts {t1['total_blackouts']} > 0 floor")
    if t3["q_max"] > 35.0:
        reasons.append(f"Q_max {t3['q_max']:.2f} > 35.0 ceiling")
    if thr["thrashing_failed"]:
        reasons.append("Pathological thrashing failure detected")

    passed = len(reasons) == 0
    return passed, reasons


def main():
    parser = argparse.ArgumentParser(description="Phase G8.4-REG Value Regularization Canary Runner")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("   PHASE G8.4-REG CANARY: STEPS 27,000 -> 28,000 (lambda=0.005) ")
    logger.info("=================================================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targeted_spec = {
        "scenario_id": "config_29",
        "mode": 2,
        "target_fraction": 0.15,
        "min_episodes": 2,
    }

    ckpt_27500 = OUTPUT_DIR / "checkpoint_gate_27500.pt"
    if not ckpt_27500.exists():
        ckpt_27500 = OUTPUT_DIR / "checkpoint_step_27500.pt"

    if not ckpt_27500.exists():
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
            gamma=0.99,  # Canonical discount!
            c_dwell=0.0,
            tau_ref=1.0,
            q_reg_coef=0.005,  # Approved Value Regularizer!
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
    else:
        logger.info("Found existing Step 27,500 checkpoint: %s. Skipping Part 1 training.", ckpt_27500)

    logger.info("Running Intermediate Diagnostic Observation at Step 27,500...")
    eval_27500 = evaluate_checkpoint_g8_4_reg(ckpt_27500, device=torch.device(args.device))
    eval_27500["observation_step"] = 27500
    out_27500_report = REPORTS_DIR / "g8_4_reg_eval_step_27500.json"
    with open(out_27500_report, "w") as f:
        json.dump({"checkpoint": str(ckpt_27500), **eval_27500}, f, indent=2)

    logger.info("=================================================================")
    logger.info("   INTERMEDIATE 27,500 OBSERVATION SUMMARY:")
    logger.info("   Mean Pd: %.2f%% | Agile Pd: %.2f%% | c29 Pd: %.2f%%",
                eval_27500["tier1_primary_metrics"]["mean_pd"],
                eval_27500["tier1_primary_metrics"]["agile_pd"],
                eval_27500["tier1_primary_metrics"]["config_29_pd"])
    logger.info("   Q_max: %.2f (target <= 35.0) | Q_mean: %.2f | Q_std: %.2f",
                eval_27500["tier3_diagnostic_metrics"]["q_max"],
                eval_27500["tier3_diagnostic_metrics"]["q_mean"],
                eval_27500["tier3_diagnostic_metrics"]["q_std"])
    logger.info("   IR_time: %.4f hits/ms | Blackouts: %d",
                eval_27500["tier1_primary_metrics"]["global_ir_time_gross_hits_per_ms"],
                eval_27500["tier1_primary_metrics"]["total_blackouts"])
    logger.info("   Probe Rho vs Gate 27k: %.6f | Flips vs Gate 27k: %d/512 (%.1f%%)",
                eval_27500["tier3_diagnostic_metrics"]["probe_spearman_rho_vs_gate27"],
                eval_27500["tier3_diagnostic_metrics"]["probe_flips_vs_gate27"],
                eval_27500["tier3_diagnostic_metrics"]["probe_flip_rate_vs_gate27_pct"])
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
        gamma=0.99,  # Canonical discount!
        c_dwell=0.0,
        tau_ref=1.0,
        q_reg_coef=0.005,  # Approved Value Regularizer!
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

    logger.info("Evaluating Gate 28,000 Checkpoint...")
    eval_28000 = evaluate_checkpoint_g8_4_reg(ckpt_28000, device=torch.device(args.device))
    eval_28000["observation_step"] = 28000

    passed_28k, fail_reasons = verify_gate_28000_acceptance(eval_28000)
    eval_28000["gate_28000_disposition"] = "PASSED" if passed_28k else "FAILED"
    eval_28000["gate_28000_failure_reasons"] = fail_reasons

    out_28000_report = REPORTS_DIR / "g8_4_reg_eval_step_28000.json"
    with open(out_28000_report, "w") as f:
        json.dump({"checkpoint": str(ckpt_28000), **eval_28000}, f, indent=2)

    logger.info("=================================================================")
    logger.info("   GATE 28,000 QUALIFICATION SUMMARY (lambda = 0.005):")
    logger.info("   DISPOSITION: %s", "PASSED" if passed_28k else "FAILED / QUARANTINED")
    if not passed_28k:
        for r in fail_reasons:
            logger.info("     * FAILURE: %s", r)
    logger.info("   Mean Pd: %.2f%% (floor >= 80.0%%)", eval_28000["tier1_primary_metrics"]["mean_pd"])
    logger.info("   Agile Pd: %.2f%% (floor >= 65.0%%)", eval_28000["tier1_primary_metrics"]["agile_pd"])
    logger.info("   config_29 Pd: %.2f%% (floor >= 85.0%%)", eval_28000["tier1_primary_metrics"]["config_29_pd"])
    logger.info("   Worst-Case Pd: %.2f%% (%s)", eval_28000["tier1_primary_metrics"]["worst_case_pd"],
                eval_28000["tier1_primary_metrics"]["worst_case_scenario"])
    logger.info("   Total Blackouts: %d (floor == 0)", eval_28000["tier1_primary_metrics"]["total_blackouts"])
    logger.info("   Gross Hits/ms: %.4f (floor >= 0.85)", eval_28000["tier1_primary_metrics"]["global_ir_time_gross_hits_per_ms"])
    logger.info("   Q_max: %.2f (ceiling <= 35.0)", eval_28000["tier3_diagnostic_metrics"]["q_max"])
    logger.info("   Q_mean: %.2f | Q_std: %.2f",
                eval_28000["tier3_diagnostic_metrics"]["q_mean"],
                eval_28000["tier3_diagnostic_metrics"]["q_std"])
    logger.info("   Probe Rho vs Gate 27k: %.6f | Flips vs Gate 27k: %d/512 (%.1f%%)",
                eval_28000["tier3_diagnostic_metrics"]["probe_spearman_rho_vs_gate27"],
                eval_28000["tier3_diagnostic_metrics"]["probe_flips_vs_gate27"],
                eval_28000["tier3_diagnostic_metrics"]["probe_flip_rate_vs_gate27_pct"])
    logger.info("   Saved Gate 28k evaluation report to %s", out_28000_report)
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
