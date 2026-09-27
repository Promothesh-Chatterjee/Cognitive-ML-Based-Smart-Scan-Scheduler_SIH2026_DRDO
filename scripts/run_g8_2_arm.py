"""Phase G8.2 Objective Alignment Experiment Runner.

Executes a single matched arm (Arm A, Arm B, or Arm C) from the verified
initial FiLM parent state: experiments/checkpoints/g8_2/parent_film_gate_26000.pt.

Objective Contracts (Identical gamma^tau SMDP discounting):
- Arm A: Exact Canonical G3-D Control
    eff_rew = r - 2.0 * (tau - 1.0)
    gamma_eff = gamma^tau
- Arm B: No Explicit Immediate Dwell Penalty
    eff_rew = r
    gamma_eff = gamma^tau
- Arm C: Fixed-Reference Centered Dwell Term
    eff_rew = r - 1.0 * (tau - 1.0)
    gamma_eff = gamma^tau

Optimizer Contract (Identical across all arms):
- Optimizer: Fresh AdamW (lr=1e-4, weight_decay=1e-4, eps=1e-8)
- Cosine entropy schedule: lambda_entropy = 0.05
- Replay: Stratified 50/50 from g8_1_preloaded_buffer.pkl
- Guard: Hardened Dual-Rule ModeCollapseGuard (warmup 200 steps)
- Buffer storage: Raw physical rewards (use_online_reward_shaper=False)
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
import yaml

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
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN
from ew_core.training.train_scheduler import train_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_g8_2_arm")

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
EXPECTED_PARENT_SHA = "a96564846225dc6b56d5392c49012e15e8395ab2f17e724a42290a89f4384a21"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_1_preloaded_buffer.pkl"
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


ARM_CONFIGS = {
    "A": {
        "name": "Arm A (Exact Canonical G3-D Control)",
        "objective_mode": "g3d_hybrid",
        "c_dwell": 2.0,
        "tau_ref": 1.0,
        "output_dir": repo_root / "experiments/checkpoints/g8_2_arm_a",
    },
    "B": {
        "name": "Arm B (No Explicit Immediate Dwell Penalty)",
        "objective_mode": "g3b_smdp_only",
        "c_dwell": 0.0,
        "tau_ref": 1.0,
        "output_dir": repo_root / "experiments/checkpoints/g8_2_arm_b",
    },
    "C": {
        "name": "Arm C (Fixed-Reference Centered Dwell Term)",
        "objective_mode": "g8_fixed_ref",
        "c_dwell": 1.0,
        "tau_ref": 1.0,
        "output_dir": repo_root / "experiments/checkpoints/g8_2_arm_c",
    },
}


def evaluate_checkpoint_canonical(ckpt_path: Path, device: torch.device = torch.device("cpu")) -> dict[str, Any]:
    """Execute canonical 10-scenario evaluation on val_stare with full value-health & FiLM diagnostics."""
    logger.info("Executing Canonical Evaluation on %s ...", ckpt_path)
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

    # Mode entropy
    mode_counts = np.bincount(all_modes, minlength=5)
    mode_dist = mode_counts / max(1, len(all_modes))
    h_mode = float(entropy(mode_dist + 1e-12))
    short_pct = float(mode_dist[0] * 100.0)
    normal_pct = float(mode_dist[1] * 100.0)
    long_pct = float(mode_dist[2] * 100.0)

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

    return {
        "mean_pd": mean_pd,
        "agile_pd": agile_pd,
        "sparse_pd": sparse_pd,
        "dense_pd": dense_pd,
        "blackout_count": blackouts,
        "h_mode": h_mode,
        "short_pct": short_pct,
        "normal_pct": normal_pct,
        "long_pct": long_pct,
        "q_max": q_max,
        "q_mean": q_mean,
        "q_std": q_std,
        "greedy_action_margin": greedy_margin,
        "distinct_bands": distinct_bands,
        "top_band_fraction": top_band_fraction,
        "film_flip_pct": film_flip_pct,
        "film_delta_q": mean_delta_q,
        "film_rho": rho_film,
        "scenarios": scenario_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase G8.2 Objective Alignment Arm Runner")
    parser.add_argument("--arm", type=str, required=True, choices=["A", "B", "C"], help="Experimental arm (A, B, or C).")
    parser.add_argument("--start-step", type=int, default=26000, help="Initial resume step.")
    parser.add_argument("--stop-at-step", type=int, default=26250, help="Target stop step.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation only.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint to evaluate in eval-only mode.")
    args = parser.parse_args()

    arm_cfg = ARM_CONFIGS[args.arm]
    logger.info("=================================================================")
    logger.info("   PHASE G8.2 OBJECTIVE ALIGNMENT EXPERIMENT: %s   ", arm_cfg["name"])
    logger.info("=================================================================")

    if args.eval_only:
        eval_path = Path(args.checkpoint_path) if args.checkpoint_path else (arm_cfg["output_dir"] / f"checkpoint_step_{args.stop_at_step}.pt")
        assert eval_path.exists(), f"Evaluation checkpoint not found: {eval_path}"
        results = evaluate_checkpoint_canonical(eval_path, device=torch.device(args.device))
        logger.info("Evaluation complete: Mean Pd=%.2f%%, Agile Pd=%.2f%%, Hmode=%.3f, Blackouts=%d",
                    results["mean_pd"], results["agile_pd"], results["h_mode"], results["blackout_count"])
        out_report = REPORTS_DIR / f"g8_2_eval_arm_{args.arm.lower()}_step_{args.stop_at_step}.json"
        with open(out_report, "w") as f:
            json.dump({"arm": args.arm, "checkpoint": str(eval_path), **results}, f, indent=2)
        logger.info("Saved report to %s", out_report)
        return

    # Preflight Verifications
    assert PARENT_CKPT_PATH.exists(), f"Parent checkpoint missing: {PARENT_CKPT_PATH}"
    actual_sha = sha256_file(PARENT_CKPT_PATH)
    assert actual_sha == EXPECTED_PARENT_SHA, f"Parent SHA mismatch: expected {EXPECTED_PARENT_SHA}, got {actual_sha}"
    assert PRELOADED_BUFFER_PATH.exists(), f"Preloaded buffer missing: {PRELOADED_BUFFER_PATH}"
    assert VAL_DIR.exists(), f"Validation directory missing: {VAL_DIR}"

    arm_cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    logger.info("Parent Checkpoint: %s (SHA: %s)", PARENT_CKPT_PATH, actual_sha[:16])
    logger.info("Objective Mode: %s | c_dwell: %.1f | tau_ref: %.1f", arm_cfg["objective_mode"], arm_cfg["c_dwell"], arm_cfg["tau_ref"])
    logger.info("Output Directory: %s", arm_cfg["output_dir"])
    logger.info("Training Steps: %d -> %d", args.start_step, args.stop_at_step)

    # Launch Training
    halted_by_guard = False
    guard_step = None
    try:
        train_scheduler(
            model_cfg_path="configs/model_config.yaml",
            train_cfg_path="configs/training_config.yaml",
            output_dir_override=str(arm_cfg["output_dir"]),
            resume_checkpoint=str(PARENT_CKPT_PATH),
            expected_parent_sha=EXPECTED_PARENT_SHA,
            stop_at_step=args.stop_at_step,
            start_step_override=args.start_step,
            staged_gates=[args.stop_at_step],
            fresh_optimizer=True,
            use_g8_film=True,
            objective_mode=arm_cfg["objective_mode"],
            c_dwell=arm_cfg["c_dwell"],
            tau_ref=arm_cfg["tau_ref"],
            use_online_reward_shaper=False,  # Raw rewards stored in replay buffer
            preloaded_buffer_path=str(PRELOADED_BUFFER_PATH),
            lambda_entropy=0.05,
        )
    except RuntimeError as e:
        err_msg = str(e)
        if "Mode collapse guard triggered" in err_msg:
            logger.warning("EXECUTION HALTED BY MODE COLLAPSE GUARD: %s", err_msg)
            halted_by_guard = True
            # Extract step from message
            import re
            m = re.search(r"step\s+(\d+)", err_msg)
            guard_step = int(m.group(1)) if m else None
        else:
            raise

    # Locate latest checkpoint
    if halted_by_guard and guard_step is not None:
        final_ckpt = arm_cfg["output_dir"] / f"emergency_collapse_step_{guard_step}.pt"
        eval_step = guard_step
    else:
        final_ckpt = arm_cfg["output_dir"] / f"checkpoint_step_{args.stop_at_step}.pt"
        if not final_ckpt.exists():
            final_ckpt = arm_cfg["output_dir"] / f"checkpoint_gate_{args.stop_at_step}.pt"
        if not final_ckpt.exists():
            final_ckpt = arm_cfg["output_dir"] / "final.pt"
        eval_step = args.stop_at_step

    if final_ckpt.exists():
        logger.info("Evaluating final checkpoint: %s (Step %d)", final_ckpt, eval_step)
        eval_metrics = evaluate_checkpoint_canonical(final_ckpt, device=torch.device(args.device))
        eval_metrics["halted_by_guard"] = halted_by_guard
        eval_metrics["eval_step"] = eval_step
        eval_metrics["arm"] = args.arm
        eval_metrics["arm_name"] = arm_cfg["name"]
        eval_metrics["checkpoint_path"] = str(final_ckpt)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_file = REPORTS_DIR / f"g8_2_eval_arm_{args.arm.lower()}_step_{eval_step}.json"
        with open(report_file, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info("Saved gate report to %s", report_file)
        print(f"\n--- G8.2 ARM {args.arm} RESULTS (Step {eval_step}) ---")
        print(f"Halted by Guard: {halted_by_guard}")
        print(f"Mean Pd:         {eval_metrics['mean_pd']:.2f}%")
        print(f"Agile Pd:        {eval_metrics['agile_pd']:.2f}%")
        print(f"Hmode:           {eval_metrics['h_mode']:.3f}")
        print(f"SHORT:           {eval_metrics['short_pct']:.1f}%")
        print(f"Blackouts:       {eval_metrics['blackout_count']}")
        print(f"FiLM Flip Rate:  {eval_metrics['film_flip_pct']:.2f}%\n")
    else:
        logger.error("No checkpoint found at %s to evaluate!", final_ckpt)


if __name__ == "__main__":
    main()
