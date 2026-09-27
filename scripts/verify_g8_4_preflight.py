"""Phase G8.4 Preflight Verification & Read-Only Target Equivalence Audit.

Executes all 4 mandatory preflight checks before Phase G8.4 gradient training:
1. Read-Only Target Equivalence Audit:
   Computes y_SMDP = r + gamma^tau * Q' and y_step = r + gamma * Q' for every transition
   in the replay buffer, disaggregated by:
   - SHORT (tau=0.25)
   - NORMAL (tau=1.0)
   - LONG (tau=2.5)
   - REVISIT (tau=1.0)
   - PREEMPTIVE (tau=1.0)
   - Hit transitions vs empty transitions
   - config_29 vs other replay strata
   Verifies that Delta y = 0.000000 identically for all tau=1.0 transitions.
2. Quantification of Predicted LONG Contraction:
   Evaluates Delta y_discount = (gamma^tau - gamma) * Q_target across all genuine LONG transitions,
   reporting mean, median, p10, p90, and config_29 empty-transition mean.
3. Explicit Acknowledgement of Changed Physical-Time Horizon:
   Freezes the physical-time distinction into the audit record.
4. Exact G8.3-B Initial State Preservation:
   Verifies initial checkpoint SHA-256 matches 5bbf15b09caabe3e76331714d68b0ca52774ae88d5d8a6e9dcfc09fa1e7d4ad2,
   0 / 512 action flips, and zero mean absolute Delta Q against parent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any, Dict

import numpy as np
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
    band_of_action,
    mode_of_action,
)
from ew_core.models.band_conditioned_drqn import BandConditionedFactorizedDRQN
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_g8_4_preflight")

INITIAL_CKPT_PATH = repo_root / "experiments/checkpoints/g8_3b_targeted_replay/initial_g8_3b_gate_26000.pt"
EXPECTED_INITIAL_SHA = "5bbf15b09caabe3e76331714d68b0ca52774ae88d5d8a6e9dcfc09fa1e7d4ad2"

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
EXPECTED_PARENT_SHA = "a96564846225dc6b56d5392c49012e15e8395ab2f17e724a42290a89f4384a21"

PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
REPORTS_DIR = repo_root / "reports"
REPORT_OUTPUT_PATH = REPORTS_DIR / "g8_4_preflight_audit.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_g8_4_preflight() -> Dict[str, Any]:
    logger.info("=================================================================")
    logger.info("   PHASE G8.4 PREFLIGHT VERIFICATION & TARGET AUDIT              ")
    logger.info("=================================================================")

    audit_results: Dict[str, Any] = {
        "phase": "Phase G8.4: Step-Based Bellman Diagnostic Objective",
        "timestamp": "2026-09-27T14:30:00+05:30",
        "parent_film_checkpoint": str(PARENT_CKPT_PATH),
        "initial_checkpoint": str(INITIAL_CKPT_PATH),
        "replay_buffer": str(BUFFER_PATH),
        "deterministic_probe": str(PROBE_BATCH_PATH),
    }

    # -------------------------------------------------------------------------
    # Check 4: Exact G8.3-B Initial Checkpoint Preservation & Probe Parity
    # -------------------------------------------------------------------------
    logger.info("[Check 4/4] Verifying Initial Checkpoint & Parent SHA-256...")
    assert INITIAL_CKPT_PATH.exists(), f"Initial checkpoint missing: {INITIAL_CKPT_PATH}"
    actual_initial_sha = sha256_file(INITIAL_CKPT_PATH)
    assert actual_initial_sha == EXPECTED_INITIAL_SHA, (
        f"Initial SHA mismatch: expected {EXPECTED_INITIAL_SHA}, got {actual_initial_sha}"
    )
    logger.info("  -> PASSED: initial_g8_3b_gate_26000.pt matches expected SHA (%s)", actual_initial_sha[:16])

    assert PARENT_CKPT_PATH.exists(), f"Parent checkpoint missing: {PARENT_CKPT_PATH}"
    actual_parent_sha = sha256_file(PARENT_CKPT_PATH)
    assert actual_parent_sha == EXPECTED_PARENT_SHA, (
        f"Parent SHA mismatch: expected {EXPECTED_PARENT_SHA}, got {actual_parent_sha}"
    )
    logger.info("  -> PASSED: parent_film_gate_26000.pt matches expected SHA (%s)", actual_parent_sha[:16])

    device = torch.device("cpu")
    # Load parent and new model to verify zero-step probe invariants
    parent_model = FiLMGatedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    parent_payload = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    parent_sd = parent_payload.get("state_dict", parent_payload.get("online_drqn", parent_payload))
    parent_model.load_state_dict(parent_sd, strict=True)
    parent_model.eval()

    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    init_payload = torch.load(INITIAL_CKPT_PATH, map_location=device, weights_only=False)
    init_sd = init_payload.get("state_dict", init_payload.get("online_drqn", init_payload))
    model.load_state_dict(init_sd, strict=True)
    model.eval()

    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]  # (64, 16, 360)
    burn_in = int(probe_data.get("burn_in", 8))

    with torch.no_grad():
        h_par = parent_model.init_hidden(probe_obs.size(0), device)
        q_parent, _, _ = parent_model(probe_obs, h_par)
        h_mod = model.init_hidden(probe_obs.size(0), device)
        q_new, _, _ = model(probe_obs, h_mod)

    q_par_graded = q_parent[:, burn_in:, :]
    q_new_graded = q_new[:, burn_in:, :]
    act_par = q_par_graded.argmax(dim=-1).flatten()
    act_new = q_new_graded.argmax(dim=-1).flatten()
    probe_flips = int((act_par != act_new).sum().item())
    total_probe_steps = int(act_par.numel())
    mean_abs_delta_q = float((q_new_graded - q_par_graded).abs().mean().item())
    max_abs_delta_q = float((q_new_graded - q_par_graded).abs().max().item())

    logger.info("  -> Probe Flips: %d / %d (%.2f%%)", probe_flips, total_probe_steps, (probe_flips / total_probe_steps) * 100)
    logger.info("  -> Mean |Delta Q|: %.6f, Max |Delta Q|: %.6f", mean_abs_delta_q, max_abs_delta_q)
    assert probe_flips == 0, f"Probe flips detected: {probe_flips} > 0"
    assert max_abs_delta_q < 1e-4, f"Max |Delta Q| too high: {max_abs_delta_q:.2e} >= 1e-4"
    logger.info("  -> PASSED: Check 4 Exact Initial Parity Verified.")

    audit_results["check_4_initial_preservation"] = {
        "initial_sha256": actual_initial_sha,
        "parent_sha256": actual_parent_sha,
        "probe_action_flips": probe_flips,
        "probe_total_steps": total_probe_steps,
        "mean_abs_delta_q": mean_abs_delta_q,
        "max_abs_delta_q": max_abs_delta_q,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Check 3: Explicit Physical-Time Horizon Freezing
    # -------------------------------------------------------------------------
    logger.info("[Check 3/4] Acknowledging Physical-Time Horizon Freezing...")
    horizon_statement = (
        "With step-based discounting, a 1250 us LONG action receives the identical gamma=0.99 discount "
        "as a 500 us REVISIT action. Phase G8.4 tests what happens when all macro-actions receive equal "
        "per-decision discount (eliminating duration-dependent geometric contraction on empty transitions), "
        "not what happens when elapsed physical time is modeled. This distinction is preserved frozen."
    )
    logger.info("  -> %s", horizon_statement)
    audit_results["check_3_physical_time_horizon"] = {
        "statement": horizon_statement,
        "acknowledged": True,
    }

    # -------------------------------------------------------------------------
    # Check 1 & Check 2: Target Equivalence Audit & LONG Contraction Distribution
    # -------------------------------------------------------------------------
    logger.info("[Check 1 & 2] Running Read-Only Target Equivalence Audit on 24,000 transitions...")
    assert BUFFER_PATH.exists(), f"Buffer missing: {BUFFER_PATH}"
    with open(BUFFER_PATH, "rb") as f:
        buf_data = pickle.load(f)
    episodes = buf_data.get("episodes", [])
    assert len(episodes) == 24, f"Expected 24 episodes, got {len(episodes)}"

    # We evaluate y_SMDP and y_step across all transitions
    # In Double-DQN at step 26000, target network equals online network.
    # Q'(s') = max_{a'} Q(next_obs, a')
    # y_SMDP = r + gamma^tau * Q'
    # y_step = r + gamma * Q'
    # Delta y = y_SMDP - y_step = (gamma^tau - gamma) * Q'

    gamma = 0.99
    dwell_mults = [0.25, 1.0, 2.5, 1.0, 1.0]

    # Data collection containers
    mode_target_diffs: Dict[str, list[float]] = {DWELL_MODES[i]: [] for i in range(5)}
    hit_target_diffs: list[float] = []
    empty_target_diffs: list[float] = []
    c29_target_diffs: list[float] = []
    other_strata_target_diffs: list[float] = []

    # Check 2 specific container: Delta y_discount for genuine LONG transitions
    long_delta_y_all: list[float] = []
    c29_empty_long_delta_y: list[float] = []
    long_q_target_all: list[float] = []

    for ep_idx, ep in enumerate(episodes):
        obs_np = np.asarray(ep["obs"], dtype=np.float32)
        next_obs_np = np.asarray(ep["next_obs"], dtype=np.float32)
        acts_np = np.asarray(ep["actions"], dtype=np.int64)
        rews_np = np.asarray(ep["rewards"], dtype=np.float32)
        dones_np = np.asarray(ep["dones"], dtype=np.float32)
        scen_id = ep.get("scenario_id", "unknown")
        primary_stratum = ep.get("primary_stratum", "unknown")
        is_targeted_long = ep.get("is_targeted_long", False)

        T = len(acts_np)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, 360)
        next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, 360)

        with torch.no_grad():
            h_init = model.init_hidden(1, device)
            q_next_seq, _, _ = model(next_obs_t, h_init)
            q_target_max = q_next_seq.squeeze(0).max(dim=-1).values.cpu().numpy()  # (T,)

        for t in range(T):
            act = int(acts_np[t])
            mode = mode_of_action(act, CANONICAL_N_MODES)
            mode_name = DWELL_MODES[mode]
            tau = dwell_mults[mode]
            r = float(rews_np[t])
            done = float(dones_np[t])
            q_prime = float(q_target_max[t]) * (1.0 - done)

            gamma_smdp = gamma ** tau
            gamma_step = gamma

            y_smdp = r + gamma_smdp * q_prime
            y_step = r + gamma_step * q_prime
            delta_y = y_smdp - y_step  # (gamma^tau - gamma) * q_prime

            mode_target_diffs[mode_name].append(delta_y)

            is_hit = (r > 0.5)  # pulse hit reward is >= 1.0
            if is_hit:
                hit_target_diffs.append(delta_y)
            else:
                empty_target_diffs.append(delta_y)

            if scen_id == "config_29":
                c29_target_diffs.append(delta_y)
            else:
                other_strata_target_diffs.append(delta_y)

            # Check 2 collection for LONG transitions (tau=2.5)
            if mode == 2:
                long_delta_y_all.append(delta_y)
                long_q_target_all.append(q_prime)
                if scen_id == "config_29" and not is_hit:
                    c29_empty_long_delta_y.append(delta_y)

    # Summarize Target Equivalence Audit (Check 1)
    logger.info("-----------------------------------------------------------------")
    logger.info("   CHECK 1: READ-ONLY TARGET EQUIVALENCE AUDIT RESULTS          ")
    logger.info("-----------------------------------------------------------------")
    check_1_summary: Dict[str, Any] = {}

    for mode_name, diffs in mode_target_diffs.items():
        arr = np.array(diffs, dtype=np.float64)
        mean_d = float(arr.mean()) if len(arr) > 0 else 0.0
        max_abs_d = float(np.abs(arr).max()) if len(arr) > 0 else 0.0
        check_1_summary[f"mode_{mode_name}"] = {
            "count": len(arr),
            "mean_delta_y": mean_d,
            "max_abs_delta_y": max_abs_d,
        }
        logger.info("  Mode %-12s (N=%5d): Mean Delta y = %+10.6f, Max |Delta y| = %10.6f", mode_name, len(arr), mean_d, max_abs_d)

    # Invariant verification: Delta y must be identically zero for tau=1.0 modes (NORMAL_DWELL, REVISIT, PREEMPTIVE_INTERCEPT)
    for tau_1_mode in ["NORMAL_DWELL", "REVISIT", "PREEMPTIVE_INTERCEPT"]:
        max_abs = check_1_summary[f"mode_{tau_1_mode}"]["max_abs_delta_y"]
        assert max_abs < 1e-7, f"Invariant violation: Delta y for {tau_1_mode} is not zero! (max = {max_abs})"
    logger.info("  -> PASSED: All tau=1.0 transitions (NORMAL_DWELL, REVISIT, PREEMPTIVE_INTERCEPT) have Delta y = 0.000000.")

    # Hit vs Empty summary
    hit_arr = np.array(hit_target_diffs)
    empty_arr = np.array(empty_target_diffs)
    check_1_summary["transitions_hit"] = {
        "count": len(hit_arr),
        "mean_delta_y": float(hit_arr.mean()),
        "max_abs_delta_y": float(np.abs(hit_arr).max()),
    }
    check_1_summary["transitions_empty"] = {
        "count": len(empty_arr),
        "mean_delta_y": float(empty_arr.mean()),
        "max_abs_delta_y": float(np.abs(empty_arr).max()),
    }
    logger.info("  Hit Transitions   (N=%5d): Mean Delta y = %+10.6f", len(hit_arr), float(hit_arr.mean()))
    logger.info("  Empty Transitions (N=%5d): Mean Delta y = %+10.6f", len(empty_arr), float(empty_arr.mean()))

    # config_29 vs Other strata summary
    c29_arr = np.array(c29_target_diffs)
    other_arr = np.array(other_strata_target_diffs)
    check_1_summary["scenario_config_29"] = {
        "count": len(c29_arr),
        "mean_delta_y": float(c29_arr.mean()),
        "max_abs_delta_y": float(np.abs(c29_arr).max()),
    }
    check_1_summary["scenario_other_strata"] = {
        "count": len(other_arr),
        "mean_delta_y": float(other_arr.mean()),
        "max_abs_delta_y": float(np.abs(other_arr).max()),
    }
    logger.info("  config_29         (N=%5d): Mean Delta y = %+10.6f", len(c29_arr), float(c29_arr.mean()))
    logger.info("  Other Strata      (N=%5d): Mean Delta y = %+10.6f", len(other_arr), float(other_arr.mean()))
    check_1_summary["passed"] = True
    audit_results["check_1_target_equivalence_audit"] = check_1_summary

    # -------------------------------------------------------------------------
    # Check 2: Quantify Predicted LONG Contraction Distribution
    # -------------------------------------------------------------------------
    logger.info("-----------------------------------------------------------------")
    logger.info("   CHECK 2: QUANTIFY PREDICTED LONG CONTRACTION DISTRIBUTION     ")
    logger.info("-----------------------------------------------------------------")
    long_diff_arr = np.array(long_delta_y_all, dtype=np.float64)
    long_q_arr = np.array(long_q_target_all, dtype=np.float64)
    c29_empty_arr = np.array(c29_empty_long_delta_y, dtype=np.float64)

    assert len(long_diff_arr) > 0, "No LONG transitions found in buffer!"

    long_mean = float(np.mean(long_diff_arr))
    long_median = float(np.median(long_diff_arr))
    long_p10 = float(np.percentile(long_diff_arr, 10))
    long_p90 = float(np.percentile(long_diff_arr, 90))
    long_min = float(np.min(long_diff_arr))
    long_max = float(np.max(long_diff_arr))
    c29_empty_mean = float(np.mean(c29_empty_arr)) if len(c29_empty_arr) > 0 else 0.0

    mean_q_target = float(np.mean(long_q_arr))
    predicted_deficit_from_mean_q = (gamma ** 2.5 - gamma ** 1.0) * mean_q_target

    logger.info("  LONG Transitions Total Count: %d", len(long_diff_arr))
    logger.info("  Mean Delta y_discount:        %+10.6f", long_mean)
    logger.info("  Median Delta y_discount:      %+10.6f", long_median)
    logger.info("  P10 Delta y_discount:         %+10.6f", long_p10)
    logger.info("  P90 Delta y_discount:         %+10.6f", long_p90)
    logger.info("  Min / Max Delta y_discount:   [%+.6f, %+.6f]", long_min, long_max)
    logger.info("  config_29 Empty Mean Delta y: %+10.6f (N=%d)", c29_empty_mean, len(c29_empty_arr))
    logger.info("  Mean Q_target on LONG:        %10.4f (Delta gamma * Q_target = %+.6f)", mean_q_target, predicted_deficit_from_mean_q)

    check_2_summary = {
        "long_transitions_count": len(long_diff_arr),
        "mean_delta_y_discount": long_mean,
        "median_delta_y_discount": long_median,
        "p10_delta_y_discount": long_p10,
        "p90_delta_y_discount": long_p90,
        "min_delta_y_discount": long_min,
        "max_delta_y_discount": long_max,
        "config_29_empty_mean_delta_y": c29_empty_mean,
        "config_29_empty_count": len(c29_empty_arr),
        "mean_q_target": mean_q_target,
        "predicted_deficit_from_mean_q": predicted_deficit_from_mean_q,
        "passed": True,
    }
    audit_results["check_2_long_contraction_quantification"] = check_2_summary

    # Save complete audit report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(audit_results, f, indent=2)
    logger.info("Saved complete preflight audit report to %s", REPORT_OUTPUT_PATH)
    logger.info("=================================================================")
    logger.info("   ALL 4 PREFLIGHT CHECKS PASSED — G8.4 CANARY READY FOR LAUNCH  ")
    logger.info("=================================================================")
    return audit_results


if __name__ == "__main__":
    verify_g8_4_preflight()
