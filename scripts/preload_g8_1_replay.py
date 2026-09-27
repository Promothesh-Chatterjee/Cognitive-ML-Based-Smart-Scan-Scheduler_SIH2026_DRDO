"""Canonical Environment Replay Preloader for Phase G8.1.

Generates real, causally consistent episodes through CognitiveRFScanEnv using the
qualified G7-A Gate-26k parent policy (checkpoint_gate_26000.pt).

Preloads at least 5 complete episodes and at least 2 distinct scenario files per stratum:
- Agile: config_29 (2 eps), config_119 (2 eps), config_241 (2 eps) = 6 episodes
- Sparse: config_143 (3 eps), config_1000 (3 eps) = 6 episodes
- Dense: config_194 (3 eps), config_195 (3 eps) = 6 episodes
- Mixed: config_117 (2 eps), config_64 (2 eps), config_96 (2 eps) = 6 episodes
Total: 24 complete episodes (24,000 transitions).

Enforces mutually exclusive primary_stratum: agile > sparse > dense > mixed.
Saves preloaded buffer to experiments/checkpoints/g8_1_preloaded_buffer.pkl.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

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
from ew_core.training.stratified_replay_sampler import map_scenario_to_primary_stratum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preload_g8_1_replay")

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g7a_targeted_replay/checkpoint_gate_26000.pt"
OUTPUT_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_1_preloaded_buffer.pkl"

PRELOAD_SCENARIO_SPECS = [
    # Agile Stratum (6 episodes, 3 scenarios)
    {"scenario_id": "config_29", "h5_path": Path("D:/TSRD/scan/train_scan/config_29.h5"), "n_episodes": 2},
    {"scenario_id": "config_119", "h5_path": Path("D:/TSRD/scan/train_scan/config_119.h5"), "n_episodes": 2},
    {"scenario_id": "config_241", "h5_path": Path("D:/TSRD/scan/train_scan/config_241.h5"), "n_episodes": 2},
    # Sparse Stratum (6 episodes, 2 scenarios)
    {"scenario_id": "config_143", "h5_path": Path("D:/TSRD/stare/train_stare/config_143.h5"), "n_episodes": 3},
    {"scenario_id": "config_1000", "h5_path": Path("D:/TSRD/stare/train_stare/config_1000.h5"), "n_episodes": 3},
    # Dense Stratum (6 episodes, 2 scenarios)
    {"scenario_id": "config_194", "h5_path": Path("D:/TSRD/stare/train_stare/config_194.h5"), "n_episodes": 3},
    {"scenario_id": "config_195", "h5_path": Path("D:/TSRD/stare/train_stare/config_195.h5"), "n_episodes": 3},
    # Mixed Stratum (6 episodes, 3 scenarios)
    {"scenario_id": "config_117", "h5_path": Path("D:/TSRD/stare/train_stare/config_117.h5"), "n_episodes": 2},
    {"scenario_id": "config_64", "h5_path": Path("D:/TSRD/stare/train_stare/config_64.h5"), "n_episodes": 2},
    {"scenario_id": "config_96", "h5_path": Path("D:/TSRD/stare/train_stare/config_96.h5"), "n_episodes": 2},
]


def preload_canonical_buffer() -> None:
    logger.info("Initializing Preload using G7-A Parent: %s", PARENT_CKPT_PATH)
    assert PARENT_CKPT_PATH.exists(), f"Parent checkpoint not found: {PARENT_CKPT_PATH}"

    model, manifest = FiLMGatedFactorizedDRQN.from_g7a_checkpoint(PARENT_CKPT_PATH, seed=42)
    model.eval()
    device = torch.device("cpu")

    buffer = SequenceReplayBuffer(
        capacity=50000,
        seq_len=16,
        obs_dim=CANONICAL_OBS_DIM,
        burn_in=8,
        seed=42 + 810,
    )

    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_OBS_DIM,
        "semantic_memory_path": ":memory:",
        "max_steps_per_episode": 1000,
        "reward": {"version": "v2"},
    }

    total_episodes = 0
    stratum_counts: dict[str, int] = {"agile": 0, "sparse": 0, "dense": 0, "mixed": 0}
    scenario_counts: dict[str, int] = {}
    mode_counts_total = np.zeros(CANONICAL_N_MODES, dtype=np.int64)

    for spec in PRELOAD_SCENARIO_SPECS:
        scen_id = spec["scenario_id"]
        h5_path = spec["h5_path"]
        n_eps = spec["n_episodes"]

        assert h5_path.exists(), f"Required scenario H5 file not found: {h5_path}"
        primary_stratum = map_scenario_to_primary_stratum(scen_id)

        for ep_i in range(n_eps):
            records = load_h5_records(h5_path, chunk_mode="random", seed=42 + total_episodes * 7)
            env = CognitiveRFScanEnv(env_cfg, records=records, seed=42 + total_episodes)
            obs, _ = env.reset(seed=42 + total_episodes)
            hidden = model.init_hidden(1, device)

            for step in range(1000):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    q_flat, aux, hidden = model(obs_t, hidden)
                    q_row = q_flat[0, 0].cpu().numpy()

                # Action selection with slight epsilon exploration (0.05) to ensure full mode coverage
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, CANONICAL_N_ACTIONS)
                else:
                    action = int(np.argmax(q_row))

                m = mode_of_action(action, CANONICAL_N_MODES)
                mode_counts_total[m] += 1

                dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m]
                dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

                next_obs, reward, term, trunc, info = env.step(action)
                done = bool(term or trunc or step == 999)

                hit_prob = float(info.get("hit_prob", 1.0 if info.get("hit", False) else 0.0))
                intercept_time = float(info.get("intercept_time_us", float("nan")))

                buffer.add(
                    obs=np.asarray(obs, dtype=np.float32),
                    action=action,
                    reward=float(reward),
                    next_obs=np.asarray(next_obs, dtype=np.float32),
                    done=done,
                    hit_prob=hit_prob,
                    intercept_time_us=intercept_time,
                    scenario_id=scen_id,
                    dwell_time_us=dwell_us,
                )

                obs = next_obs
                if done:
                    break

            total_episodes += 1
            stratum_counts[primary_stratum] += 1
            scenario_counts[scen_id] = scenario_counts.get(scen_id, 0) + 1
            logger.info("Generated episode %2d/24: scenario=%s (stratum=%s, transitions=%d)",
                        total_episodes, scen_id, primary_stratum, len(buffer._episodes[-1]["actions"]))

    logger.info("=== Preload Generation Complete ===")
    logger.info("Total Episodes in Buffer: %d", buffer.n_episodes())
    logger.info("Total Transitions: %d", len(buffer))
    logger.info("Stratum Breakdown: %s", stratum_counts)
    logger.info("Scenario Breakdown: %s", scenario_counts)
    logger.info("Total Mode Distribution: %s", {DWELL_MODES[i]: int(mode_counts_total[i]) for i in range(5)})

    # Preflight Assertions on Buffer Integrity
    for stratum, count in stratum_counts.items():
        assert count >= 5, f"Preload failed: Stratum '{stratum}' has only {count} episodes (minimum 5 required)!"

    for m_idx in range(5):
        assert mode_counts_total[m_idx] > 0, f"Mode {DWELL_MODES[m_idx]} has 0 visits in preloaded buffer!"

    # Save to disk
    OUTPUT_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    buffer.save_episodes(OUTPUT_BUFFER_PATH)
    logger.info("Successfully persisted preloaded buffer to %s", OUTPUT_BUFFER_PATH)


if __name__ == "__main__":
    preload_canonical_buffer()
