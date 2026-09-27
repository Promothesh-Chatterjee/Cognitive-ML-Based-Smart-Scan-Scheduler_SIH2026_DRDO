"""Phase G8.4 Step-Based Bellman Diagnostic Objective Arm Runner.

Executes Phase G8.4: Step-Based Bellman Diagnostic Objective:
- Resumes strictly from materialized initial checkpoint:
  experiments/checkpoints/g8_3b_targeted_replay/initial_g8_3b_gate_26000.pt
  (SHA-256: 5bbf15b09caabe3e76331714d68b0ca52774ae88d5d8a6e9dcfc09fa1e7d4ad2).
- Architecture: BandConditionedFactorizedDRQN:
  Q(s, b, m) = V(s) + A_band_tilde(s, b) + D_tilde(s, b, m)
  where D(s, b, .) = f_D([h_s, e_b]) in R^5.
- Objective: Step-based Bellman diagnostic objective:
  y = r + gamma * Q' for every mode (gamma = 0.99, c_dwell = 0.0, tau_ref = 1.0).
- Replay: 24-episode buffer with 15.625% guaranteed minibatch quota (5/32 seqs) from genuine (config_29, LONG),
  and 84.375% (27/32 seqs) 4-strata balanced.
- Controls locked: learning rate = 2.5e-5, override_epsilon = 0.05, sequence_len = 16.
- Shadow-Greedy ModeCollapseGuard active on deterministic probe batch.
- Canary Horizon: Step 26,000 -> 26,500 ONLY.

Mandatory Conditioning-Path Activity Gate & Evaluation at Gate 26,500:
- Delta Q_D = Q_full - Q_{D=0}
- D-ablation greedy action flips globally and on probe batch
- P(LONG | config_29) full vs D-ablated
- Mean |D| on config_29 and globally
- Fraction of config_29 states where dwell branch changes selected mode
- Dwell-head parameter displacement ||theta_D^(26500) - theta_D^(26000)||_2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

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
    DWELL_MODES,
    band_of_action,
    mode_of_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.band_conditioned_drqn import BandConditionedFactorizedDRQN
from ew_core.training.train_scheduler import train_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_g8_4_arm")

MATERIALIZED_PARENT_PATH = repo_root / "experiments/checkpoints/g8_3b_targeted_replay/initial_g8_3b_gate_26000.pt"
EXPECTED_INITIAL_SHA = "5bbf15b09caabe3e76331714d68b0ca52774ae88d5d8a6e9dcfc09fa1e7d4ad2"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g8_4_step_based_objective"
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
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_htau(mode_counts: np.ndarray) -> tuple[float, dict[str, float]]:
    """Compute Shannon entropy over physical dwell duration classes {0.25, 1.0, 2.5}."""
    n_total = max(1, int(np.sum(mode_counts)))
    p_025 = float(mode_counts[0]) / n_total
    p_100 = float(mode_counts[1] + mode_counts[3] + mode_counts[4]) / n_total
    p_250 = float(mode_counts[2]) / n_total
    probs = np.array([p_025, p_100, p_250])
    nz = probs[probs > 1e-12]
    h_val = float(-np.sum(nz * np.log(nz))) if len(nz) > 0 else 0.0
    return h_val, {"0.25": p_025, "1.0": p_100, "2.5": p_250}


def evaluate_checkpoint_g8_4(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
    initial_ckpt_path: Path = MATERIALIZED_PARENT_PATH,
) -> Dict[str, Any]:
    """Evaluates the G8.4 checkpoint across all 10 canonical scenarios and computes the D-activity gate."""
    logger.info("Evaluating G8.4 Checkpoint: %s", checkpoint_path)
    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("online_drqn", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load initial checkpoint for parameter displacement calculation
    init_model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    init_ckpt = torch.load(initial_ckpt_path, map_location=device, weights_only=False)
    init_sd = init_ckpt.get("state_dict", init_ckpt.get("online_drqn", init_ckpt))
    init_model.load_state_dict(init_sd, strict=True)
    init_model.eval()

    # Compute dwell-head parameter displacement ||theta_D^(26500) - theta_D^(26000)||_2
    d_norm_sq = 0.0
    for name, param in model.dwell_head.named_parameters():
        init_param = dict(init_model.dwell_head.named_parameters())[name]
        d_norm_sq += float((param.data - init_param.data).pow(2).sum().item())
    dwell_param_displacement = float(np.sqrt(d_norm_sq))
    logger.info("Dwell-Head Parameter Displacement ||Delta theta_D||: %.6f", dwell_param_displacement)

    # 1. Deterministic Probe Batch Activity Audit
    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]  # (64, 16, 360)
    burn_in = int(probe_data.get("burn_in", 8))
    with torch.no_grad():
        h_pr = model.init_hidden(probe_obs.size(0), device)
        q_pr, aux_pr, _ = model(probe_obs, h_pr)
        q_pr_abl, _, _ = model(probe_obs, h_pr, ablate_dwell_branch=True)

    q_pr_g = q_pr[:, burn_in:, :]
    q_pr_abl_g = q_pr_abl[:, burn_in:, :]
    act_pr_g = q_pr_g.argmax(dim=-1).flatten()
    act_pr_abl_g = q_pr_abl_g.argmax(dim=-1).flatten()
    probe_d_flips = int((act_pr_g != act_pr_abl_g).sum().item())
    probe_d_flip_rate = float(probe_d_flips / max(1, act_pr_g.numel()))
    probe_mean_abs_delta_q = float((q_pr_g - q_pr_abl_g).abs().mean().item())
    logger.info("Probe Batch D-Ablation Flips: %d / %d (%.2f%%)", probe_d_flips, act_pr_g.numel(), probe_d_flip_rate * 100.0)

    scenario_results: Dict[str, Any] = {}
    scenario_dwell_breakdown: Dict[str, Any] = {}
    all_actions = []
    all_modes = []
    all_bands = []
    all_q_values = []
    d_delta_qs = []
    d_action_flips = 0
    total_decision_steps = 0

    c29_delta_q_long_list = []
    c29_pref_long_count = 0
    c29_long_action_count = 0
    c29_long_ablated_count = 0
    c29_mode_changes_by_d = 0
    c29_d_magnitudes = []
    c29_total_steps = 0

    for scen_name in CANONICAL_SCENARIOS:
        h5_path = VAL_DIR / f"{scen_name}.h5"
        records = load_h5_records(h5_path, chunk_mode="first")
        env = CognitiveRFScanEnv(records=records)
        obs, _ = env.reset()
        hidden = model.init_hidden(1, device)

        scen_acts = []
        scen_mods = []
        scen_hits = 0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_step, aux, hidden = model(obs_tensor, hidden)
                q_abl, _, _ = model(obs_tensor, hidden, ablate_dwell_branch=True)

            q_row = q_step.squeeze().cpu().numpy()
            q_row_abl = q_abl.squeeze().cpu().numpy()
            d_tilde_row = aux["d_tilde"].squeeze().cpu().numpy()  # (36, 5)

            act = int(np.argmax(q_row))
            act_abl = int(np.argmax(q_row_abl))
            if act != act_abl:
                d_action_flips += 1

            d_delta_qs.append(float(np.abs(q_row - q_row_abl).mean()))

            b = band_of_action(act, CANONICAL_N_MODES)
            m = mode_of_action(act, CANONICAL_N_MODES)

            # config_29 Mechanistic Q-Margin & D-activity computation
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

                m_abl = mode_of_action(act_abl, CANONICAL_N_MODES)
                if m != m_abl:
                    c29_mode_changes_by_d += 1
                if m_abl == 2:
                    c29_long_ablated_count += 1

                c29_d_magnitudes.append(float(np.abs(d_tilde_row[b_star]).mean()))

            scen_acts.append(act)
            scen_mods.append(m)
            all_actions.append(act)
            all_modes.append(m)
            all_bands.append(b)
            all_q_values.append(q_row)
            total_decision_steps += 1

            next_obs, reward, term, trunc, info = env.step(act)
            if bool(info.get("hit", False)):
                scen_hits += 1

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        pd_val = float(fom.get("Pd", float(scen_hits / max(1, len(scen_acts))))) * 100.0
        scenario_results[scen_name] = {
            "pd": pd_val,
            "steps": len(scen_acts),
            "hits": scen_hits,
            "gross_hits_per_ms": fom.get("gross_hits_per_ms", 0.0),
            "novel_hits_per_ms": fom.get("novel_hits_per_ms", 0.0),
        }

        # Scenario-specific dwell breakdown
        scen_m_counts = np.bincount(scen_mods, minlength=5)
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

    # Summary metrics
    all_pds = [v["pd"] for v in scenario_results.values()]
    mean_pd = float(np.mean(all_pds))
    agile_pds = [scenario_results[s]["pd"] for s in AGILE_SCENARIOS if s in scenario_results]
    agile_pd = float(np.mean(agile_pds)) if agile_pds else 0.0
    sparse_pds = [scenario_results[s]["pd"] for s in SPARSE_SCENARIOS if s in scenario_results]
    sparse_pd = float(np.mean(sparse_pds)) if sparse_pds else 0.0
    dense_pds = [scenario_results[s]["pd"] for s in DENSE_SCENARIOS if s in scenario_results]
    dense_pd = float(np.mean(dense_pds)) if dense_pds else 0.0
    blackouts = sum(1 for p in all_pds if p == 0.0)

    # Mode entropy and Dwell duration entropy H_tau
    mode_counts = np.bincount(all_modes, minlength=5)
    mode_dist = mode_counts / max(1, len(all_modes))
    h_mode = float(entropy(mode_dist + 1e-12))
    h_tau, tau_dist = compute_htau(mode_counts)

    # Value health
    all_q_arr = np.array(all_q_values)
    q_max = float(all_q_arr.max())
    q_mean = float(all_q_arr.mean())
    q_std = float(all_q_arr.std())

    # Conditioning-Path Activity Gate Metrics
    global_d_flip_rate = float(d_action_flips / max(1, total_decision_steps))
    mean_d_delta_q = float(np.mean(d_delta_qs))

    c29_pd = scenario_results.get("config_29", {}).get("pd", 0.0)
    c29_long_rate = float(c29_long_action_count / max(1, c29_total_steps))
    c29_pref_rate = float(c29_pref_long_count / max(1, c29_total_steps))
    c29_delta_q_long_mean = float(np.mean(c29_delta_q_long_list)) if c29_delta_q_long_list else 0.0
    c29_d_magnitude_mean = float(np.mean(c29_d_magnitudes)) if c29_d_magnitudes else 0.0
    c29_d_mode_change_rate = float(c29_mode_changes_by_d / max(1, c29_total_steps))

    activity_gate_passed = bool(
        probe_d_flip_rate >= 0.01
        and dwell_param_displacement > 0.0
    )

    return {
        "mean_pd": mean_pd,
        "agile_pd": agile_pd,
        "sparse_pd": sparse_pd,
        "dense_pd": dense_pd,
        "blackouts": blackouts,
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
        "scenario_results": scenario_results,
        "scenario_dwell_breakdown": scenario_dwell_breakdown,
        "config_29_diagnostics": {
            "pd": c29_pd,
            "p_long": c29_long_rate,
            "p_tau_2_5": c29_long_rate,
            "mean_delta_q_long": c29_delta_q_long_mean,
            "pref_fraction_long": c29_pref_rate,
            "ceiling_ratio": float(c29_pd / 98.9) if c29_pd > 0 else 0.0,
            "forced_long_ceiling_pd": 98.9,
            "d_ablation_mode_change_rate": c29_d_mode_change_rate,
            "mean_abs_d_on_c29": c29_d_magnitude_mean,
        },
        "conditioning_path_activity_gate": {
            "passed": activity_gate_passed,
            "probe_d_ablation_flip_rate": probe_d_flip_rate,
            "probe_d_flips": probe_d_flips,
            "global_d_ablation_flip_rate": global_d_flip_rate,
            "dwell_parameter_displacement": dwell_param_displacement,
            "mean_abs_delta_q_by_d": mean_d_delta_q,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phase G8.4 Step-Based Bellman Diagnostic Objective Arm Runner")
    parser.add_argument("--start-step", type=int, default=26000, help="Initial resume step.")
    parser.add_argument("--stop-at-step", type=int, default=26500, help="Target stop step (Canary=26500 ONLY).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation only.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint to evaluate in eval-only mode.")
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("   PHASE G8.4 STEP-BASED BELLMAN RUNNER: STEPS %d -> %d         ", args.start_step, args.stop_at_step)
    logger.info("=================================================================")

    if args.eval_only:
        if args.checkpoint_path:
            eval_path = Path(args.checkpoint_path)
        else:
            eval_path = OUTPUT_DIR / f"checkpoint_gate_{args.stop_at_step}.pt"
            if not eval_path.exists():
                eval_path = OUTPUT_DIR / f"checkpoint_step_{args.stop_at_step}.pt"
        assert eval_path.exists(), f"Evaluation checkpoint not found: {eval_path}"
        results = evaluate_checkpoint_g8_4(eval_path, device=torch.device(args.device))
        logger.info(
            "Evaluation complete: Mean Pd=%.2f%%, Agile Pd=%.2f%%, Hmode=%.3f, Htau=%.3f, config_29 Pd=%.2f%% (LONG=%.1f%%)",
            results["mean_pd"],
            results["agile_pd"],
            results["h_mode"],
            results["h_tau"],
            results["config_29_diagnostics"]["pd"],
            results["config_29_diagnostics"]["p_long"] * 100.0,
        )
        out_report = REPORTS_DIR / f"g8_4_eval_step_{args.stop_at_step}.json"
        with open(out_report, "w") as f:
            json.dump({"checkpoint": str(eval_path), **results}, f, indent=2)
        logger.info("Saved report to %s", out_report)
        return

    # Determine Resume Checkpoint
    if args.start_step > 26000:
        resume_ckpt = OUTPUT_DIR / f"checkpoint_gate_{args.start_step}.pt"
        if not resume_ckpt.exists():
            resume_ckpt = OUTPUT_DIR / f"checkpoint_step_{args.start_step}.pt"
        expected_sha = None
        assert resume_ckpt.exists(), f"Continuation checkpoint missing: {resume_ckpt}"
    else:
        resume_ckpt = MATERIALIZED_PARENT_PATH
        assert MATERIALIZED_PARENT_PATH.exists(), f"Materialized initial checkpoint missing: {MATERIALIZED_PARENT_PATH}"
        expected_sha = sha256_file(MATERIALIZED_PARENT_PATH)

    assert PRELOADED_BUFFER_PATH.exists(), f"Preloaded buffer missing: {PRELOADED_BUFFER_PATH}"
    assert PROBE_BATCH_PATH.exists(), f"Probe batch missing: {PROBE_BATCH_PATH}"
    assert VAL_DIR.exists(), f"Validation directory missing: {VAL_DIR}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Resume Checkpoint: %s (SHA expected: %s)", resume_ckpt, expected_sha[:16] if expected_sha else "In-Flight")
    logger.info("Architecture: BandConditionedFactorizedDRQN (D(s, b, m) = f_D([h_s, e_b]))")
    logger.info("Objective: Step-Based Bellman Diagnostic Objective (objective_mode=step_based, c_dwell=0.0, tau_ref=1.0, gamma=0.99)")
    logger.info("Replay Buffer: %s", PRELOADED_BUFFER_PATH)
    logger.info("Probe Batch: %s", PROBE_BATCH_PATH)
    logger.info("Replay Quotas: 15.625%% (5/32 seqs) targeted (config_29, LONG); 84.375%% (27/32 seqs) 4-strata balanced")
    logger.info("Controlled Exploration Floor: epsilon = 0.05")
    logger.info("Canonical Learning Rate: lr = 2.5e-5")
    logger.info("Output Directory: %s", OUTPUT_DIR)
    logger.info("Training Steps: %d -> %d", args.start_step, args.stop_at_step)

    targeted_spec = {
        "scenario_id": "config_29",
        "mode": 2,
        "target_fraction": 0.15,
        "min_episodes": 2,
    }

    halted_by_guard = False
    guard_step = None
    try:
        train_scheduler(
            model_cfg_path="configs/model_config.yaml",
            train_cfg_path="configs/training_config.yaml",
            output_dir_override=str(OUTPUT_DIR),
            resume_checkpoint=str(resume_ckpt),
            expected_parent_sha=expected_sha,
            stop_at_step=args.stop_at_step,
            start_step_override=args.start_step,
            staged_gates=[args.stop_at_step],
            fresh_optimizer=True,
            use_g8_film=True,  # triggers DRQN loading branch where BandConditionedFactorizedDRQN is detected
            objective_mode="step_based",
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
    except RuntimeError as e:
        err_msg = str(e)
        if "Mode collapse guard triggered" in err_msg:
            logger.warning("EXECUTION HALTED BY SHADOW-GREEDY MODE COLLAPSE GUARD: %s", err_msg)
            halted_by_guard = True
            import re
            m = re.search(r"step\s+(\d+)", err_msg)
            guard_step = int(m.group(1)) if m else None
        else:
            raise

    # Determine which checkpoint was saved
    eval_ckpt = OUTPUT_DIR / f"checkpoint_gate_{args.stop_at_step}.pt"
    if not eval_ckpt.exists():
        eval_ckpt = OUTPUT_DIR / f"checkpoint_step_{args.stop_at_step}.pt"
    if not eval_ckpt.exists() and halted_by_guard and guard_step:
        eval_ckpt = OUTPUT_DIR / f"checkpoint_guard_step_{guard_step}.pt"

    assert eval_ckpt.exists(), f"No evaluation checkpoint found after training at {eval_ckpt}"
    logger.info("Evaluating Gate Checkpoint: %s", eval_ckpt)

    results = evaluate_checkpoint_g8_4(eval_ckpt, device=torch.device(args.device))
    results["halted_by_guard"] = halted_by_guard
    results["guard_step"] = guard_step
    results["checkpoint_evaluated"] = str(eval_ckpt)
    results["step"] = args.stop_at_step if not halted_by_guard else guard_step

    out_report = REPORTS_DIR / f"g8_4_eval_step_{args.stop_at_step}.json"
    with open(out_report, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved complete evaluation report to %s", out_report)

    # Print summary evaluation card
    logger.info("=================================================================")
    logger.info("   PHASE G8.4 CANARY EVALUATION SUMMARY (STEP %d)               ", args.stop_at_step)
    logger.info("=================================================================")
    logger.info("Mean Pd:          %.2f%%", results["mean_pd"])
    logger.info("Agile Pd:         %.2f%%", results["agile_pd"])
    logger.info("Dense Pd:         %.2f%%", results["dense_pd"])
    logger.info("Sparse Pd:        %.2f%%", results["sparse_pd"])
    logger.info("Blackouts:        %d", results["blackouts"])
    logger.info("H_mode:           %.3f", results["h_mode"])
    logger.info("H_tau:            %.3f", results["h_tau"])
    logger.info("Mode Breakdown:   SHORT=%.1f%%, NORMAL=%.1f%%, LONG=%.1f%%, REVISIT=%.1f%%, PREEMPTIVE=%.1f%%",
                results["short_pct"], results["normal_pct"], results["long_pct"], results["revisit_pct"], results["preemptive_pct"])
    logger.info("config_29 Pd:     %.2f%% (LONG=%.1f%%, Delta Q LONG=%+.4f)",
                results["config_29_diagnostics"]["pd"],
                results["config_29_diagnostics"]["p_long"] * 100.0,
                results["config_29_diagnostics"]["mean_delta_q_long"])
    logger.info("Q_max:            %.2f", results["q_max"])
    logger.info("D-Activity Flips: %.2f%% (Probe), %.2f%% (Global)",
                results["conditioning_path_activity_gate"]["probe_d_ablation_flip_rate"] * 100.0,
                results["conditioning_path_activity_gate"]["global_d_ablation_flip_rate"] * 100.0)
    logger.info("D Displacement:   %.6f", results["conditioning_path_activity_gate"]["dwell_parameter_displacement"])
    logger.info("Activity Gate:    %s", "PASSED" if results["conditioning_path_activity_gate"]["passed"] else "FAILED")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
