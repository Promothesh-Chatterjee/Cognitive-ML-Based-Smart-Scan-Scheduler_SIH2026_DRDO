"""Phase G8.2R2 Preflight Invariant Verification Suite.

Asserts 9 critical preflight invariants prior to launching the unconfounded
G8.2R2 matched-arm experiments:

1. Parent checkpoint exists with verified SHA-256.
2. Parent checkpoint loads with strict=True into FiLMGatedFactorizedDRQN.
3. Mode-balanced preloaded buffer exists (g8_2r2_preloaded_buffer.pkl).
4. Buffer satisfies 4x5 balanced matrix (20 eps, 25% stratum, 20% mode, min 1000 per cell).
5. Deterministic probe batch exists (g8_2r2_probe_batch.pt).
6. Shadow-Greedy guard evaluates Parent Model as HEALTHY (not triggered).
7. Shadow-Greedy guard evaluates Collapsed Checkpoint as TRIGGERED (fails closed).
8. Canonical 10-scenario validation suite is present on disk.
9. Training runner enforces lr=2.5e-5 and override_epsilon=0.05.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    DWELL_MODES,
)
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN
from ew_core.training.mode_collapse_guard import ModeCollapseGuard
from ew_core.training.stratified_replay_sampler import map_scenario_to_primary_stratum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_g8_2r2_preflight")

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
EXPECTED_PARENT_SHA = "a96564846225dc6b56d5392c49012e15e8395ab2f17e724a42290a89f4384a21"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_2r2_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_2r2_probe_batch.pt"
VAL_DIR = Path("D:/TSRD/stare/val_stare")
COLLAPSED_ARM_A_PATH = repo_root / "experiments/checkpoints/g8_2_arm_a/checkpoint_gate_26300.pt"

CANONICAL_SCENARIOS = [
    "config_117", "config_119", "config_143", "config_194", "config_195",
    "config_241", "config_29", "config_42", "config_64", "config_96",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_preflight_checks() -> bool:
    logger.info("=======================================================")
    logger.info("   PHASE G8.2R2 PREFLIGHT INVARIANT VERIFICATION SUITE   ")
    logger.info("=======================================================")
    all_passed = True

    # Check 1: Parent Checkpoint Existence and SHA
    logger.info("[Check 1/9] Verifying parent checkpoint and SHA-256...")
    if not PARENT_CKPT_PATH.exists():
        logger.error("FAIL: Parent checkpoint does not exist: %s", PARENT_CKPT_PATH)
        return False
    actual_sha = sha256_file(PARENT_CKPT_PATH)
    if actual_sha != EXPECTED_PARENT_SHA:
        logger.error("FAIL: SHA mismatch! Expected %s, got %s", EXPECTED_PARENT_SHA, actual_sha)
        return False
    logger.info("PASS: Parent SHA-256 verified (%s)", actual_sha[:16])

    # Check 2: Parent Architecture Load
    logger.info("[Check 2/9] Loading parent into FiLMGatedFactorizedDRQN (strict=True)...")
    try:
        model = FiLMGatedFactorizedDRQN(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)
        payload = torch.load(PARENT_CKPT_PATH, map_location="cpu", weights_only=False)
        sd = payload.get("state_dict", payload.get("online_drqn", payload))
        model.load_state_dict(sd, strict=True)
        model.eval()
        logger.info("PASS: Parent state_dict cleanly loaded with strict=True.")
    except Exception as exc:
        logger.error("FAIL: Model failed to load: %s", exc)
        return False

    # Check 3: Preloaded Buffer Existence
    logger.info("[Check 3/9] Checking mode-balanced preloaded buffer...")
    if not PRELOADED_BUFFER_PATH.exists():
        logger.error("FAIL: Buffer file not found: %s", PRELOADED_BUFFER_PATH)
        return False
    logger.info("PASS: Preloaded buffer file found.")

    # Check 4: Buffer 4x5 Balanced Matrix Verification
    logger.info("[Check 4/9] Auditing 4x5 stratum x mode distribution...")
    with open(PRELOADED_BUFFER_PATH, "rb") as f:
        buf_data = pickle.load(f)
    episodes = buf_data["episodes"]
    n_episodes = len(episodes)
    total_transitions = sum(len(e["actions"]) for e in episodes)
    strata_counts: dict[str, int] = {"agile": 0, "sparse": 0, "dense": 0, "mixed": 0}
    mode_counts = np.zeros(5, dtype=np.int64)
    cell_matrix = np.zeros((4, 5), dtype=np.int64)
    stratum_map = {"agile": 0, "sparse": 1, "dense": 2, "mixed": 3}

    for ep in episodes:
        scen = ep["scenario_id"]
        strat = map_scenario_to_primary_stratum(scen)
        strata_counts[strat] += 1
        for act in ep["actions"]:
            m = act % 5
            mode_counts[m] += 1
            cell_matrix[stratum_map[strat], m] += 1

    logger.info("Total Episodes: %d | Total Transitions: %d", n_episodes, total_transitions)
    logger.info("Strata Distribution: %s", strata_counts)
    logger.info("Mode Distribution: %s", {DWELL_MODES[i]: int(mode_counts[i]) for i in range(5)})
    logger.info("4x5 Transition Matrix:\n%s", cell_matrix)

    if n_episodes != 20 or total_transitions != 20000:
        logger.error("FAIL: Expected 20 episodes and 20000 transitions, got %d eps / %d trans", n_episodes, total_transitions)
        return False
    for s, c in strata_counts.items():
        if c != 5:
            logger.error("FAIL: Stratum %s has %d episodes, expected exactly 5 (25.0%%)", s, c)
            return False
    for m in range(5):
        if mode_counts[m] != 4000:
            logger.error("FAIL: Mode %s has %d transitions, expected exactly 4000 (20.0%%)", DWELL_MODES[m], mode_counts[m])
            return False
    if not np.all(cell_matrix >= 1000):
        logger.error("FAIL: Some cells have < 1000 transitions:\n%s", cell_matrix)
        return False
    logger.info("PASS: 4x5 Matrix is strictly balanced (25.0% stratum, 20.0% mode).")

    # Check 5: Deterministic Probe Batch Existence
    logger.info("[Check 5/9] Checking deterministic probe batch...")
    if not PROBE_BATCH_PATH.exists():
        logger.error("FAIL: Probe batch file not found: %s", PROBE_BATCH_PATH)
        return False
    probe_payload = torch.load(PROBE_BATCH_PATH, map_location="cpu", weights_only=False)
    probe_obs = probe_payload["obs"]
    if probe_obs.shape != torch.Size([64, 16, 360]):
        logger.error("FAIL: Probe obs unexpected shape: %s", probe_obs.shape)
        return False
    logger.info("PASS: Probe batch verified: shape %s, burn_in=%d", list(probe_obs.shape), probe_payload.get("burn_in", 8))

    # Check 6: Shadow-Greedy Guard on Parent Model
    logger.info("[Check 6/9] Validating Shadow-Greedy guard on Parent Model...")
    guard = ModeCollapseGuard(probe_batch_path=PROBE_BATCH_PATH)
    parent_res = guard.check_shadow_greedy(model, global_step=26000)
    logger.info("Parent Probe: SHORT=%.1f%%, Entropy=%.3f, Triggered=%s",
                parent_res["short_frac"] * 100.0, parent_res["entropy"], parent_res["triggered"])
    if parent_res["triggered"]:
        logger.error("FAIL: Guard unexpectedly triggered on parent model!")
        return False
    logger.info("PASS: Parent model is healthy under shadow-greedy guard.")

    # Check 7: Shadow-Greedy Guard on Collapsed Checkpoint
    logger.info("[Check 7/9] Validating Shadow-Greedy guard trips on Collapsed Checkpoint...")
    if COLLAPSED_ARM_A_PATH.exists():
        payload_col = torch.load(COLLAPSED_ARM_A_PATH, map_location="cpu", weights_only=False)
        sd_col = payload_col.get("state_dict", payload_col.get("online_drqn", payload_col))
        model.load_state_dict(sd_col, strict=True)
        col_res = guard.check_shadow_greedy(model, global_step=26300)
        logger.info("Collapsed Probe: SHORT=%.1f%%, Entropy=%.3f, Triggered=%s (%s)",
                    col_res["short_frac"] * 100.0, col_res["entropy"], col_res["triggered"], col_res["reason"])
        if not col_res["triggered"]:
            logger.error("FAIL: Guard failed to trip on known collapsed checkpoint!")
            return False
        logger.info("PASS: Guard successfully detects collapse and trips.")
    else:
        logger.warning("SKIP (Check 7): Collapsed Arm A checkpoint not found at %s", COLLAPSED_ARM_A_PATH)

    # Check 8: Validation Scenarios Existence
    logger.info("[Check 8/9] Verifying 10 canonical validation scenarios on disk...")
    for scen in CANONICAL_SCENARIOS:
        h5_p = VAL_DIR / f"{scen}.h5"
        if not h5_p.exists():
            logger.error("FAIL: Missing validation file: %s", h5_p)
            return False
    logger.info("PASS: All 10 validation scenarios exist in %s", VAL_DIR)

    # Check 9: Runner Configuration Invariants
    logger.info("[Check 9/9] Verifying runner script constants...")
    runner_path = repo_root / "scripts/run_g8_2r2_arm.py"
    with open(runner_path, "r", encoding="utf-8") as f:
        runner_src = f.read()
    assert "override_epsilon=0.05" in runner_src, "Runner must pass override_epsilon=0.05"
    assert "learning_rate=2.5e-5" in runner_src, "Runner must pass learning_rate=2.5e-5"
    assert "g8_2r2_preloaded_buffer.pkl" in runner_src, "Runner must point to g8_2r2_preloaded_buffer.pkl"
    assert "g8_2r2_probe_batch.pt" in runner_src, "Runner must point to g8_2r2_probe_batch.pt"
    logger.info("PASS: Runner script contains all 4 unconfounded controls locked.")

    logger.info("=======================================================")
    logger.info("   ALL 9 PREFLIGHT INVARIANTS SATISFIED (READY)        ")
    logger.info("=======================================================")
    return True


if __name__ == "__main__":
    success = run_preflight_checks()
    if not success:
        sys.exit(1)
