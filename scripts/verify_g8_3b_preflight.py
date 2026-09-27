"""Phase G8.3-B Preflight Verification & Zero-Step Behavioral Audit Suite.

Executes all mandatory preflight controls before Phase G8.3-B training:
1. Verifies parent checkpoint (experiments/checkpoints/g8_2/parent_film_gate_26000.pt) and SHA-256.
2. Performs strict tensor transfer from parent to BandConditionedFactorizedDRQN.
3. Produces and saves the Tensor Transfer Manifest.
4. Executes Zero-Step Behavioral Audit on the fixed deterministic probe batch (experiments/checkpoints/g8_3a_probe_batch.pt):
   - Mean absolute Delta Q
   - Max absolute Delta Q
   - Greedy action flip rate (assert <= 5.0%, expected 0.00%)
   - Band flip rate
   - Dwell-class flip rate
5. Executes Zero-Step Behavioral Audit on canonical config_29 (val_stare, 1000 steps):
   - P(LONG | config_29)
   - Pd(config_29)
   - Delta Q_LONG margin
6. Materializes initial G8.3-B model checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
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
    DWELL_MODES,
    band_of_action,
    mode_of_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.band_conditioned_drqn import BandConditionedFactorizedDRQN
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_g8_3b_preflight")

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
EXPECTED_PARENT_SHA = "a96564846225dc6b56d5392c49012e15e8395ab2f17e724a42290a89f4384a21"

PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
VAL_DIR = Path("D:/TSRD/stare/val_stare")
REPORTS_DIR = repo_root / "reports"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g8_3b_targeted_replay"
MATERIALIZED_CKPT_PATH = OUTPUT_DIR / "initial_g8_3b_gate_26000.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_g8_3b_preflight() -> bool:
    logger.info("=================================================================")
    logger.info("   PHASE G8.3-B PREFLIGHT VERIFICATION & ZERO-STEP AUDIT         ")
    logger.info("=================================================================")

    # 1. Parent Checkpoint Verification
    logger.info("[1/6] Verifying Parent Checkpoint SHA-256...")
    assert PARENT_CKPT_PATH.exists(), f"Parent checkpoint missing: {PARENT_CKPT_PATH}"
    actual_sha = sha256_file(PARENT_CKPT_PATH)
    assert actual_sha == EXPECTED_PARENT_SHA, f"SHA mismatch: expected {EXPECTED_PARENT_SHA}, got {actual_sha}"
    logger.info("  -> PASSED: %s matches expected SHA-256 (%s)", PARENT_CKPT_PATH.name, actual_sha[:16])

    # 2. Instantiate and Transfer Weights
    logger.info("[2/6] Executing Tensor Transfer from Parent to BandConditionedFactorizedDRQN...")
    parent_model = FiLMGatedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    )
    parent_payload = torch.load(PARENT_CKPT_PATH, map_location="cpu", weights_only=False)
    parent_sd = parent_payload.get("state_dict", parent_payload.get("online_drqn", parent_payload))
    parent_model.load_state_dict(parent_sd, strict=True)
    parent_model.eval()

    new_model, manifest = BandConditionedFactorizedDRQN.from_parent_checkpoint(PARENT_CKPT_PATH, seed=42)
    new_model.eval()

    # 3. Save Tensor Transfer Manifest
    logger.info("[3/6] Saving Tensor Transfer Manifest...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS_DIR / "g8_3b_tensor_transfer_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("  -> Saved Tensor Transfer Manifest to %s", manifest_path)

    # 4. Zero-Step Behavioral Audit on Fixed Probe Batch
    logger.info("[4/6] Executing Zero-Step Behavioral Audit on Fixed Probe Batch...")
    assert PROBE_BATCH_PATH.exists(), f"Probe batch missing: {PROBE_BATCH_PATH}"
    probe_data = torch.load(PROBE_BATCH_PATH, map_location="cpu", weights_only=False)
    probe_obs = probe_data["obs"]  # (64, 16, 360)
    burn_in = probe_data.get("burn_in", 8)

    with torch.no_grad():
        q_parent, _, _ = parent_model(probe_obs)
        q_new, aux_new, _ = new_model(probe_obs)

    # Post burn-in transitions: (64, 8, 180) -> (512, 180)
    q_p_active = q_parent[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).numpy()
    q_n_active = q_new[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).numpy()

    abs_dq = np.abs(q_n_active - q_p_active)
    mean_abs_dq = float(np.mean(abs_dq))
    max_abs_dq = float(np.max(abs_dq))

    act_p = np.argmax(q_p_active, axis=-1)
    act_n = np.argmax(q_n_active, axis=-1)

    action_flips = int(np.sum(act_p != act_n))
    action_flip_rate = float(action_flips / len(act_p)) * 100.0

    band_p = act_p // CANONICAL_N_MODES
    band_n = act_n // CANONICAL_N_MODES
    band_flips = int(np.sum(band_p != band_n))
    band_flip_rate = float(band_flips / len(band_p)) * 100.0

    mode_p = act_p % CANONICAL_N_MODES
    mode_n = act_n % CANONICAL_N_MODES
    mode_flips = int(np.sum(mode_p != mode_n))
    mode_flip_rate = float(mode_flips / len(mode_p)) * 100.0

    logger.info("  -> Probe Zero-Step Audit Metrics:")
    logger.info("     Mean Absolute Delta Q:    %.6f", mean_abs_dq)
    logger.info("     Max Absolute Delta Q:     %.6f", max_abs_dq)
    logger.info("     Greedy Action Flip Rate:  %.2f%% (%d / %d)", action_flip_rate, action_flips, len(act_p))
    logger.info("     Band Flip Rate:           %.2f%%", band_flip_rate)
    logger.info("     Mode Flip Rate:           %.2f%%", mode_flip_rate)

    # Mandatory Failure Condition: action_flip_rate <= 5.0%
    assert action_flip_rate <= 5.0, (
        f"G8.3-B Zero-Step Preflight Failure: Greedy action flip rate {action_flip_rate:.2f}% "
        f"exceeds the 5.0% quarantine limit!"
    )
    assert mean_abs_dq < 1e-4, f"Unexpected Q discrepancy at zero-step: {mean_abs_dq}"
    logger.info("  -> PASSED: Exact analytical zero-step equivalence demonstrated on probe batch")

    # 5. Zero-Step Behavioral Audit on Canonical config_29
    logger.info("[5/6] Executing Zero-Step Behavioral Audit on Canonical config_29...")
    c29_h5 = VAL_DIR / "config_29.h5"
    assert c29_h5.exists(), f"Val config_29 missing: {c29_h5}"
    records = load_h5_records(c29_h5, chunk_mode="first")
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
    hidden = new_model.init_hidden(1, torch.device("cpu"))

    acts = []
    hits = 0
    delta_q_long_list = []

    for _ in range(1000):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_flat, aux, hidden = new_model(obs_t, hidden)

        q_row = q_flat[0, 0].cpu().numpy()
        q_grid = q_row.reshape(CANONICAL_N_BANDS, CANONICAL_N_MODES)
        b_star = int(np.argmax(np.max(q_grid, axis=-1)))

        q_long = float(q_grid[b_star, 2])
        other_max = float(max(q_grid[b_star, 0], q_grid[b_star, 1], q_grid[b_star, 3], q_grid[b_star, 4]))
        delta_q_long_list.append(q_long - other_max)

        act = int(np.argmax(q_row))
        acts.append(act)

        next_obs, rew, term, trunc, info = env.step(act)
        if bool(info.get("hit", False)):
            hits += 1

        obs = next_obs
        if term or trunc:
            break

    fom = env.get_fom()
    pd_c29 = float(fom.get("Pd", hits / max(1, len(acts)))) * 100.0
    modes = [a % CANONICAL_N_MODES for a in acts]
    p_long_c29 = float(np.mean([m == 2 for m in modes])) * 100.0
    mean_dq_long = float(np.mean(delta_q_long_list))

    logger.info("  -> config_29 Zero-Step Audit Metrics:")
    logger.info("     Evaluated Pd:             %.2f%% (hits=%d)", pd_c29, hits)
    logger.info("     P(LONG | config_29):      %.1f%%", p_long_c29)
    logger.info("     Mean Delta Q_LONG:        %+.4f", mean_dq_long)

    assert abs(pd_c29 - 72.0) < 0.1, f"Expected Pd=72.0%, got {pd_c29:.2f}%"
    assert abs(p_long_c29 - 50.2) < 0.1, f"Expected LONG=50.2%, got {p_long_c29:.1f}%"
    logger.info("  -> PASSED: config_29 behavior identically replicates parent baseline")

    # 6. Materialize Initial Checkpoint
    logger.info("[6/6] Materializing Qualified Initial G8.3-B Checkpoint...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 26000,
            "state_dict": new_model.state_dict(),
            "manifest": manifest,
            "architecture": "BandConditionedFactorizedDRQN",
            "parent_checkpoint": str(PARENT_CKPT_PATH),
            "parent_sha256": EXPECTED_PARENT_SHA,
            "zero_step_metrics": {
                "probe_mean_abs_dq": mean_abs_dq,
                "probe_action_flip_rate": action_flip_rate,
                "config_29_pd": pd_c29,
                "config_29_p_long": p_long_c29,
                "config_29_mean_delta_q_long": mean_dq_long,
            },
        },
        MATERIALIZED_CKPT_PATH,
    )
    mat_sha = sha256_file(MATERIALIZED_CKPT_PATH)
    logger.info("  -> Saved initial G8.3-B checkpoint to %s (SHA-256: %s)", MATERIALIZED_CKPT_PATH, mat_sha[:16])

    # Save complete zero-step audit report
    audit_report = {
        "parent_checkpoint": str(PARENT_CKPT_PATH),
        "parent_sha256": EXPECTED_PARENT_SHA,
        "materialized_checkpoint": str(MATERIALIZED_CKPT_PATH),
        "materialized_sha256": mat_sha,
        "tensor_transfer_manifest": manifest,
        "probe_batch_audit": {
            "mean_abs_dq": mean_abs_dq,
            "max_abs_dq": max_abs_dq,
            "action_flip_rate_pct": action_flip_rate,
            "band_flip_rate_pct": band_flip_rate,
            "mode_flip_rate_pct": mode_flip_rate,
        },
        "config_29_audit": {
            "pd": pd_c29,
            "hits": hits,
            "p_long_pct": p_long_c29,
            "mean_delta_q_long": mean_dq_long,
        },
    }
    audit_report_path = REPORTS_DIR / "g8_3b_zero_step_audit.json"
    with open(audit_report_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    logger.info("  -> Saved Complete Zero-Step Audit Report to %s", audit_report_path)

    logger.info("=================================================================")
    logger.info("   ALL PREFLIGHT INVARIANTS & ZERO-STEP AUDITS PASSED (G8.3-B)   ")
    logger.info("=================================================================")
    return True


if __name__ == "__main__":
    success = verify_g8_3b_preflight()
    if not success:
        sys.exit(1)
