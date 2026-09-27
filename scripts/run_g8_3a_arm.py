"""Phase G8.3-A Targeted Long-Dwell Replay Arm Runner.

Executes Phase G8.3-A: Scenario-Conditioned Long-Dwell Recovery:
- Resumes strictly from qualified parent: experiments/checkpoints/g8_2/parent_film_gate_26000.pt (LONG intact).
- Objective: SMDP-only dwell-neutral (c_dwell = 0.0, tau_ref = 1.0, gamma^tau).
- Replay: 24-episode buffer with 15% guaranteed minibatch quota from genuine (config_29, LONG) transitions.
- Controls locked: learning rate = 2.5e-5, override_epsilon = 0.05, sequence_len = 16.
- Shadow-Greedy ModeCollapseGuard active on deterministic probe batch.

Diagnostic Suite:
- Scenario-conditional dwell distributions [P(SHORT), P(tau=1.0), P(LONG)] and H_tau across all 10 scenarios.
- Mechanistic Q-margin on config_29: Delta Q_LONG(s) = Q(s, b*, LONG) - max_{m != LONG} Q(s, b*, m).
- P(LONG | config_29), P(Delta Q_LONG > 0 | config_29), and Ceiling Ratio: Pd(config_29) / 98.90%.
- Both categorical mode entropy H_mode and physical dwell class entropy H_tau.
- FiLM action flip rate, delta Q, and rho_FiLM.
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
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN
from ew_core.training.train_scheduler import train_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_g8_3a_arm")

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
EXPECTED_PARENT_SHA = "a96564846225dc6b56d5392c49012e15e8395ab2f17e724a42290a89f4384a21"

PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g8_3a_targeted_replay"
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


def evaluate_checkpoint_g8_3(ckpt_path: Path, device: torch.device = torch.device("cpu")) -> dict[str, Any]:
    """Execute canonical 10-scenario evaluation with the full Phase G8.3 diagnostic suite."""
    logger.info("Executing Phase G8.3 Canonical Evaluation on %s ...", ckpt_path)
    model = FiLMGatedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = payload.get("state_dict", payload.get("online_drqn", payload))
    model.load_state_dict(sd, strict=True)
    model.eval()

    all_actions = []
    all_modes = []
    all_bands = []
    all_q_values = []
    film_delta_qs = []
    film_action_flips = 0
    total_decision_steps = 0
    scenario_results = {}
    scenario_dwell_breakdown = {}

    # config_29 mechanistic tracking
    c29_delta_q_long_list = []
    c29_pref_long_count = 0
    c29_long_action_count = 0
    c29_total_steps = 0

    for scen_name in CANONICAL_SCENARIOS:
        h5_path = VAL_DIR / f"{scen_name}.h5"
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
        scen_mods = []
        scen_hits = 0

        for _ in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)

                # Compute counterfactual ablated Q (gamma=1, beta=0) for FiLM influence
                v = aux["v"].unsqueeze(-1)
                a_b = aux["a_band_tilde"].unsqueeze(-1)
                a_m = aux["a_mode_tilde"].unsqueeze(2)
                q_ablated = (v + a_b + a_m).view(1, 1, 180)

                q_row = q_flat[0, 0].cpu().numpy()
                q_abl_row = q_ablated[0, 0].cpu().numpy()

            act_film = int(np.argmax(q_row))
            act_abl = int(np.argmax(q_abl_row))
            if act_film != act_abl:
                film_action_flips += 1

            dq = float(np.abs(q_row - q_abl_row).mean())
            film_delta_qs.append(dq)

            act = act_film
            b = band_of_action(act, CANONICAL_N_MODES)
            m = mode_of_action(act, CANONICAL_N_MODES)

            # config_29 Mechanistic Q-Margin computation
            if scen_name == "config_29":
                c29_total_steps += 1
                q_grid = q_row.reshape(CANONICAL_N_BANDS, CANONICAL_N_MODES)
                # Greedy band choice
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
    top2_diffs = np.sort(all_q_arr, axis=-1)[:, -1] - np.sort(all_q_arr, axis=-1)[:, -2]
    greedy_margin = float(top2_diffs.mean())

    # Band distribution
    band_counts = np.bincount(all_bands, minlength=36)
    distinct_bands = int(np.count_nonzero(band_counts))
    top_band_fraction = float(band_counts.max() / max(1, len(all_bands)))

    # FiLM diagnostics
    film_flip_pct = (film_action_flips / max(1, total_decision_steps)) * 100.0
    mean_delta_q = float(np.mean(film_delta_qs)) if film_delta_qs else 0.0
    rho_film = float(mean_delta_q / max(1e-6, q_std))

    # config_29 Mechanistic Metrics
    mean_delta_q_long_c29 = float(np.mean(c29_delta_q_long_list)) if c29_delta_q_long_list else 0.0
    pref_frac_long_c29 = float(c29_pref_long_count / max(1, c29_total_steps))
    p_long_c29 = float(c29_long_action_count / max(1, c29_total_steps))
    ceiling_ratio_c29 = float(scenario_results["config_29"]["pd"] / 98.90)

    return {
        "mean_pd": mean_pd,
        "agile_pd": agile_pd,
        "sparse_pd": sparse_pd,
        "dense_pd": dense_pd,
        "blackout_count": blackouts,
        "h_mode": h_mode,
        "h_tau": h_tau,
        "tau_dist": tau_dist,
        "short_pct": float(mode_dist[0] * 100.0),
        "normal_pct": float(mode_dist[1] * 100.0),
        "long_pct": float(mode_dist[2] * 100.0),
        "revisit_pct": float(mode_dist[3] * 100.0),
        "preemptive_pct": float(mode_dist[4] * 100.0),
        "q_max": q_max,
        "q_mean": q_mean,
        "q_std": q_std,
        "greedy_action_margin": greedy_margin,
        "distinct_bands": distinct_bands,
        "top_band_fraction": top_band_fraction,
        "film_flip_pct": film_flip_pct,
        "film_delta_q": mean_delta_q,
        "film_rho": rho_film,
        "config_29_diagnostics": {
            "pd": scenario_results["config_29"]["pd"],
            "p_long": p_long_c29,
            "p_tau_2_5": p_long_c29,
            "mean_delta_q_long": mean_delta_q_long_c29,
            "pref_fraction_long": pref_frac_long_c29,
            "ceiling_ratio": ceiling_ratio_c29,
            "forced_long_ceiling_pd": 98.90,
        },
        "scenario_dwell_breakdown": scenario_dwell_breakdown,
        "scenarios": scenario_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase G8.3-A Targeted Long-Dwell Replay Arm Runner")
    parser.add_argument("--start-step", type=int, default=26000, help="Initial resume step.")
    parser.add_argument("--stop-at-step", type=int, default=26500, help="Target stop step (Canary=26500, Gate=27000).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation only.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint to evaluate in eval-only mode.")
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("   PHASE G8.3-A TARGETED REPLAY RUNNER: STEPS %d -> %d   ", args.start_step, args.stop_at_step)
    logger.info("=================================================================")

    if args.eval_only:
        if args.checkpoint_path:
            eval_path = Path(args.checkpoint_path)
        else:
            eval_path = OUTPUT_DIR / f"checkpoint_gate_{args.stop_at_step}.pt"
            if not eval_path.exists():
                eval_path = OUTPUT_DIR / f"checkpoint_step_{args.stop_at_step}.pt"
        assert eval_path.exists(), f"Evaluation checkpoint not found: {eval_path}"
        results = evaluate_checkpoint_g8_3(eval_path, device=torch.device(args.device))
        logger.info(
            "Evaluation complete: Mean Pd=%.2f%%, Agile Pd=%.2f%%, Hmode=%.3f, Htau=%.3f, config_29 Pd=%.2f%% (LONG=%.1f%%)",
            results["mean_pd"],
            results["agile_pd"],
            results["h_mode"],
            results["h_tau"],
            results["config_29_diagnostics"]["pd"],
            results["config_29_diagnostics"]["p_long"] * 100.0,
        )
        out_report = REPORTS_DIR / f"g8_3a_eval_step_{args.stop_at_step}.json"
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
        resume_ckpt = PARENT_CKPT_PATH
        expected_sha = EXPECTED_PARENT_SHA
        assert PARENT_CKPT_PATH.exists(), f"Parent checkpoint missing: {PARENT_CKPT_PATH}"
        actual_sha = sha256_file(PARENT_CKPT_PATH)
        assert actual_sha == EXPECTED_PARENT_SHA, f"Parent SHA mismatch: expected {EXPECTED_PARENT_SHA}, got {actual_sha}"

    assert PRELOADED_BUFFER_PATH.exists(), f"Preloaded buffer missing: {PRELOADED_BUFFER_PATH}"
    assert PROBE_BATCH_PATH.exists(), f"Probe batch missing: {PROBE_BATCH_PATH}"
    assert VAL_DIR.exists(), f"Validation directory missing: {VAL_DIR}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Resume Checkpoint: %s (SHA expected: %s)", resume_ckpt, expected_sha[:16] if expected_sha else "In-Flight")
    logger.info("Objective: SMDP-Only Dwell-Neutral (c_dwell=0.0, tau_ref=1.0, gamma^tau)")
    logger.info("Replay Buffer: %s", PRELOADED_BUFFER_PATH)
    logger.info("Probe Batch: %s", PROBE_BATCH_PATH)
    logger.info("Targeted Quota: 15%% (config_29, LONG)")
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
            use_g8_film=True,
            objective_mode="g3b_smdp_only",
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

    # Locate latest checkpoint
    if halted_by_guard and guard_step is not None:
        final_ckpt = OUTPUT_DIR / f"emergency_collapse_step_{guard_step}.pt"
        eval_step = guard_step
    else:
        final_ckpt = OUTPUT_DIR / f"checkpoint_step_{args.stop_at_step}.pt"
        if not final_ckpt.exists():
            final_ckpt = OUTPUT_DIR / f"checkpoint_gate_{args.stop_at_step}.pt"
        if not final_ckpt.exists():
            final_ckpt = OUTPUT_DIR / "final.pt"
        eval_step = args.stop_at_step

    if final_ckpt.exists():
        logger.info("Evaluating final checkpoint: %s (Step %d)", final_ckpt, eval_step)
        eval_metrics = evaluate_checkpoint_g8_3(final_ckpt, device=torch.device(args.device))
        eval_metrics["halted_by_guard"] = halted_by_guard
        eval_metrics["eval_step"] = eval_step
        eval_metrics["arm"] = "G8.3-A"
        eval_metrics["arm_name"] = "Phase G8.3-A Targeted Long-Dwell Replay"
        eval_metrics["checkpoint_path"] = str(final_ckpt)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_file = REPORTS_DIR / f"g8_3a_eval_step_{eval_step}.json"
        with open(report_file, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info("Saved gate report to %s", report_file)

        c29_d = eval_metrics["config_29_diagnostics"]
        print(f"\n=================================================================")
        print(f"   PHASE G8.3-A CANARY RESULTS (Step {eval_step})                ")
        print(f"=================================================================")
        print(f"Halted by Guard:             {halted_by_guard}")
        print(f"Mean Pd:                     {eval_metrics['mean_pd']:.2f}%")
        print(f"Agile Pd:                    {eval_metrics['agile_pd']:.2f}%")
        print(f"Sparse Pd:                   {eval_metrics['sparse_pd']:.2f}%")
        print(f"Dense Pd:                    {eval_metrics['dense_pd']:.2f}%")
        print(f"Hmode:                       {eval_metrics['h_mode']:.3f}")
        print(f"Htau:                        {eval_metrics['h_tau']:.3f} (tau=0.25: {eval_metrics['tau_dist']['0.25']:.1%}, tau=1.0: {eval_metrics['tau_dist']['1.0']:.1%}, tau=2.5: {eval_metrics['tau_dist']['2.5']:.1%})")
        print(f"SHORT:                       {eval_metrics['short_pct']:.1f}%")
        print(f"NORMAL:                      {eval_metrics['normal_pct']:.1f}%")
        print(f"LONG:                        {eval_metrics['long_pct']:.1f}%")
        print(f"REVISIT:                     {eval_metrics['revisit_pct']:.1f}%")
        print(f"PREEMPTIVE:                  {eval_metrics['preemptive_pct']:.1f}%")
        print(f"Blackouts:                   {eval_metrics['blackout_count']}")
        print(f"--- config_29 Mechanistic Diagnostics ---")
        print(f"config_29 Pd:                {c29_d['pd']:.2f}% (Ceiling Ratio: {c29_d['ceiling_ratio']:.1%})")
        print(f"P(LONG | config_29):         {c29_d['p_long']:.1%}")
        print(f"P(Delta Q_LONG > 0 | c29):   {c29_d['pref_fraction_long']:.1%}")
        print(f"Mean Delta Q_LONG on c29:    {c29_d['mean_delta_q_long']:+.4f}")
        print(f"FiLM Action Flip Rate:       {eval_metrics['film_flip_pct']:.2f}%\n")
    else:
        logger.error("No checkpoint found at %s to evaluate!", final_ckpt)


if __name__ == "__main__":
    main()
