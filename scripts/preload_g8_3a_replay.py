"""Canonical Environment Replay Preloader for Phase G8.3-A.

Constructs a targeted 24-episode replay buffer (24,000 transitions):
- 20 Base Episodes: from Phase G8.2R2 balanced 4x5 buffer (5 episodes each in Agile, Sparse, Dense, Mixed; 20,000 transitions).
- 4 Targeted Episodes: genuine (config_29, LONG) interactions generated on real TSRD recordings
  (D:/TSRD/scan/train_scan/config_29.h5) using parent model band selection and forced LONG_DWELL (tau=2.5, 1250 us).

Minibatch Sampling Quota Contract:
- 15% quota: 5 out of 32 sequences sampled directly from genuine (config_29, LONG) episodes.
- 85% quota: 27 sequences sampled across the 4 strata (agile, sparse, dense, mixed) with canonical weights (30/25/25/20).

Also validates and copies the deterministic probe batch (experiments/checkpoints/g8_3a_probe_batch.pt)
for the Shadow-Greedy ModeCollapseGuard.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any

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
    RF_BASE_DWELL_TIME_US,
    band_of_action,
    mode_of_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN
from ew_core.training.replay_buffer import SequenceReplayBuffer
from ew_core.training.stratified_replay_sampler import StratifiedReplaySampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preload_g8_3a_replay")

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
EXPECTED_PARENT_SHA = "a96564846225dc6b56d5392c49012e15e8395ab2f17e724a42290a89f4384a21"

BASE_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_2r2_preloaded_buffer.pkl"
BASE_PROBE_PATH = repo_root / "experiments/checkpoints/g8_2r2_probe_batch.pt"

OUTPUT_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
OUTPUT_PROBE_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"

TARGET_H5_PATH = Path("D:/TSRD/scan/train_scan/config_29.h5")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def preload_targeted_g8_3a_buffer() -> None:
    logger.info("=================================================================")
    logger.info("   PHASE G8.3-A TARGETED REPLAY BUFFER PRELOAD GENERATOR        ")
    logger.info("=================================================================")

    # 1. Invariant: Parent Checkpoint Verification
    assert PARENT_CKPT_PATH.exists(), f"Parent checkpoint not found: {PARENT_CKPT_PATH}"
    actual_sha = sha256_file(PARENT_CKPT_PATH)
    assert actual_sha == EXPECTED_PARENT_SHA, f"Parent SHA mismatch: expected {EXPECTED_PARENT_SHA}, got {actual_sha}"
    logger.info("Parent Checkpoint verified: %s (SHA-256: %s)", PARENT_CKPT_PATH, actual_sha)

    # 2. Invariant: Base Buffer Verification
    assert BASE_BUFFER_PATH.exists(), f"Base buffer not found: {BASE_BUFFER_PATH}"
    with open(BASE_BUFFER_PATH, "rb") as f:
        base_data = pickle.load(f)
    base_episodes = base_data.get("episodes", [])
    assert len(base_episodes) == 20, f"Expected 20 base episodes, got {len(base_episodes)}"
    base_total = sum(int(e["length"]) for e in base_episodes)
    assert base_total == 20000, f"Expected 20,000 base transitions, got {base_total}"
    logger.info("Base Buffer verified: 20 episodes, %d transitions from %s", base_total, BASE_BUFFER_PATH)

    # 3. Load Parent Model for Intelligent Band Selection
    model = FiLMGatedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    )
    payload = torch.load(PARENT_CKPT_PATH, map_location="cpu", weights_only=False)
    sd = payload.get("state_dict", payload.get("online_drqn", payload))
    model.load_state_dict(sd, strict=True)
    model.eval()
    device = torch.device("cpu")

    # 4. Generate 4 Genuine (config_29, LONG) Episodes
    assert TARGET_H5_PATH.exists(), f"Target scenario H5 missing: {TARGET_H5_PATH}"
    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_OBS_DIM,
        "semantic_memory_path": ":memory:",
        "max_steps_per_episode": 1000,
        "reward": {"version": "v2"},
    }

    targeted_buffer = SequenceReplayBuffer(
        capacity=10000,
        seq_len=16,
        obs_dim=CANONICAL_OBS_DIM,
        burn_in=8,
        seed=42 + 830,
    )

    logger.info("Generating 4 genuine (config_29, LONG) episodes via CognitiveRFScanEnv...")
    target_mode = 2  # LONG_DWELL (tau=2.5, 1250 us)
    dwell_mult = DEFAULT_DWELL_MULTIPLIERS[target_mode]
    dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

    for ep_idx in range(4):
        seed = 42 + ep_idx * 17
        records = load_h5_records(TARGET_H5_PATH, chunk_mode="random", seed=seed)
        env = CognitiveRFScanEnv(env_cfg, records=records, seed=seed)
        obs, _ = env.reset(seed=seed)
        hidden = model.init_hidden(1, device)

        ep_hits = 0
        ep_actions = []

        for step in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                q_row = q_flat[0, 0].cpu().numpy()

            # Greedy band selection via parent policy
            b_greedy = band_of_action(int(np.argmax(q_row)), CANONICAL_N_MODES)

            # Forced LONG mode
            action = int(b_greedy * CANONICAL_N_MODES + target_mode)
            ep_actions.append(action)

            next_obs, reward, term, trunc, info = env.step(action)
            done = bool(term or trunc or step == 999)

            hit_prob = float(info.get("hit_prob", 1.0 if info.get("hit", False) else 0.0))
            if bool(info.get("hit", False)):
                ep_hits += 1
            intercept_time = float(info.get("intercept_time_us", float("nan")))

            targeted_buffer.add(
                obs=np.asarray(obs, dtype=np.float32),
                action=action,
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=done,
                hit_prob=hit_prob,
                intercept_time_us=intercept_time,
                scenario_id="config_29",
                dwell_time_us=dwell_us,
            )

            obs = next_obs
            if done:
                break

        # Explicitly tag targeted episode
        ep_dict = targeted_buffer._episodes[-1]
        ep_dict["is_targeted_long"] = True
        ep_dict["target_mode"] = target_mode
        ep_dict["primary_stratum"] = "agile"

        fom = env.get_fom()
        pd_val = float(fom.get("Pd", ep_hits / len(ep_actions))) * 100.0
        logger.info(
            "Generated Targeted Ep %d/4: config_29 mode=LONG steps=%d hits=%d Pd=%.2f%%",
            ep_idx + 1,
            len(ep_actions),
            ep_hits,
            pd_val,
        )
        assert pd_val >= 50.0, f"Targeted episode {ep_idx + 1} produced unexpectedly low Pd: {pd_val:.2f}%"

    assert len(targeted_buffer._episodes) == 4, f"Expected 4 targeted episodes, got {len(targeted_buffer._episodes)}"
    assert sum(int(e["length"]) for e in targeted_buffer._episodes) == 4000, "Expected 4000 targeted transitions"

    # 5. Assemble Combined 24-Episode Buffer
    combined_buffer = SequenceReplayBuffer(
        capacity=50000,
        seq_len=16,
        obs_dim=CANONICAL_OBS_DIM,
        burn_in=8,
        seed=42 + 831,
    )

    for ep in base_episodes:
        # Ensure base episodes are marked as non-targeted
        ep["is_targeted_long"] = False
        combined_buffer._episodes.append(ep)
        combined_buffer._total += int(ep["length"])

    for ep in targeted_buffer._episodes:
        combined_buffer._episodes.append(ep)
        combined_buffer._total += int(ep["length"])

    logger.info("=== Combined Buffer Verification ===")
    logger.info("Total Episodes: %d (20 base + 4 targeted)", len(combined_buffer._episodes))
    logger.info("Total Transitions: %d", combined_buffer._total)

    assert len(combined_buffer._episodes) == 24, f"Expected 24 episodes, got {len(combined_buffer._episodes)}"
    assert combined_buffer._total == 24000, f"Expected 24000 transitions, got {combined_buffer._total}"

    # 6. Verify StratifiedReplaySampler with Targeted Spec
    targeted_spec = {
        "scenario_id": "config_29",
        "mode": 2,
        "target_fraction": 0.15,
        "min_episodes": 2,
    }
    sampler = StratifiedReplaySampler(
        buffer=combined_buffer,
        strata_weights={"agile": 0.30, "sparse": 0.25, "dense": 0.25, "mixed": 0.20},
        min_episodes_per_stratum=5,
        targeted_spec=targeted_spec,
    )

    counts = sampler.get_stratum_counts()
    logger.info("Stratified Sampler Partition Counts: %s", counts)
    assert counts.get("targeted") == 4, f"Expected 4 targeted episodes, got {counts.get('targeted')}"
    assert counts.get("agile") == 5, f"Expected 5 agile base episodes, got {counts.get('agile')}"
    assert counts.get("sparse") == 5, f"Expected 5 sparse base episodes, got {counts.get('sparse')}"
    assert counts.get("dense") == 5, f"Expected 5 dense base episodes, got {counts.get('dense')}"
    assert counts.get("mixed") == 5, f"Expected 5 mixed base episodes, got {counts.get('mixed')}"

    # Test sampling a batch of 32
    test_batch = sampler.sample(32, target_hit_seq_fraction=0.40)
    assert test_batch["obs"].shape == (32, 16, 360), f"Unexpected sample obs shape: {test_batch['obs'].shape}"
    assert test_batch["actions"].shape == (32, 16), f"Unexpected sample actions shape: {test_batch['actions'].shape}"
    logger.info("Sampled test batch of 32 sequences successfully: obs shape=%s", test_batch["obs"].shape)

    # 7. Persist Buffer
    OUTPUT_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_buffer.save_episodes(OUTPUT_BUFFER_PATH)
    logger.info("Successfully persisted G8.3-A preloaded buffer to %s", OUTPUT_BUFFER_PATH)

    # 8. Copy/Verify Deterministic Probe Batch
    assert BASE_PROBE_PATH.exists(), f"Base probe batch missing: {BASE_PROBE_PATH}"
    shutil.copyfile(BASE_PROBE_PATH, OUTPUT_PROBE_PATH)
    logger.info("Successfully copied probe batch to %s", OUTPUT_PROBE_PATH)


if __name__ == "__main__":
    preload_targeted_g8_3a_buffer()
