"""Canonical Environment Replay Preloader for Phase G8.2R2.

Constructs a strictly mode-balanced 4x5 replay buffer (20 episodes, 20,000 transitions):
- 4 Strata: Agile, Sparse, Dense, Mixed (5 episodes each = 25.0%)
- 5 Modes: SHORT (0), NORMAL (1), LONG (2), REVISIT (3), PREEMPTIVE (4) (4 episodes each = 20.0%)
- Matrix coverage: At least 1 episode per mode per stratum (all 20 cells in 4x5 matrix populated).

In each episode:
- The RF receiver interacts with CognitiveRFScanEnv on real TSRD scenario recordings.
- Band selection is driven by the qualified G7-A parent model (checkpoint_gate_26000.pt) RF belief state.
- Mode selection is systematically targeted to guarantee exact 20.0% mode balance.
- All physical transition metadata (rewards, hit_probs, intercept_times_us, dwell_times_us) are causally recorded.

Also exports a deterministic probe batch (experiments/checkpoints/g8_2r2_probe_batch.pt)
for the Shadow-Greedy ModeCollapseGuard.
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
logger = logging.getLogger("preload_g8_2r2_replay")

PARENT_CKPT_PATH = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
OUTPUT_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_2r2_preloaded_buffer.pkl"
OUTPUT_PROBE_PATH = repo_root / "experiments/checkpoints/g8_2r2_probe_batch.pt"

# Authoritative 4x5 Scenario Matrix: 4 strata x 5 modes = 20 episodes
PRELOAD_4X5_SPECS = [
    # --- Agile Stratum (5 episodes: modes 0, 1, 2, 3, 4) ---
    {"stratum": "agile", "scenario_id": "config_29",  "h5_path": Path("D:/TSRD/scan/train_scan/config_29.h5"),  "target_mode": 0},
    {"stratum": "agile", "scenario_id": "config_29",  "h5_path": Path("D:/TSRD/scan/train_scan/config_29.h5"),  "target_mode": 1},
    {"stratum": "agile", "scenario_id": "config_119", "h5_path": Path("D:/TSRD/scan/train_scan/config_119.h5"), "target_mode": 2},
    {"stratum": "agile", "scenario_id": "config_119", "h5_path": Path("D:/TSRD/scan/train_scan/config_119.h5"), "target_mode": 3},
    {"stratum": "agile", "scenario_id": "config_241", "h5_path": Path("D:/TSRD/scan/train_scan/config_241.h5"), "target_mode": 4},

    # --- Sparse Stratum (5 episodes: modes 0, 1, 2, 3, 4) ---
    {"stratum": "sparse", "scenario_id": "config_143",  "h5_path": Path("D:/TSRD/stare/train_stare/config_143.h5"),  "target_mode": 0},
    {"stratum": "sparse", "scenario_id": "config_143",  "h5_path": Path("D:/TSRD/stare/train_stare/config_143.h5"),  "target_mode": 1},
    {"stratum": "sparse", "scenario_id": "config_143",  "h5_path": Path("D:/TSRD/stare/train_stare/config_143.h5"),  "target_mode": 2},
    {"stratum": "sparse", "scenario_id": "config_1000", "h5_path": Path("D:/TSRD/stare/train_stare/config_1000.h5"), "target_mode": 3},
    {"stratum": "sparse", "scenario_id": "config_1000", "h5_path": Path("D:/TSRD/stare/train_stare/config_1000.h5"), "target_mode": 4},

    # --- Dense Stratum (5 episodes: modes 0, 1, 2, 3, 4) ---
    {"stratum": "dense", "scenario_id": "config_194", "h5_path": Path("D:/TSRD/stare/train_stare/config_194.h5"), "target_mode": 0},
    {"stratum": "dense", "scenario_id": "config_194", "h5_path": Path("D:/TSRD/stare/train_stare/config_194.h5"), "target_mode": 1},
    {"stratum": "dense", "scenario_id": "config_194", "h5_path": Path("D:/TSRD/stare/train_stare/config_194.h5"), "target_mode": 2},
    {"stratum": "dense", "scenario_id": "config_195", "h5_path": Path("D:/TSRD/stare/train_stare/config_195.h5"), "target_mode": 3},
    {"stratum": "dense", "scenario_id": "config_195", "h5_path": Path("D:/TSRD/stare/train_stare/config_195.h5"), "target_mode": 4},

    # --- Mixed Stratum (5 episodes: modes 0, 1, 2, 3, 4) ---
    {"stratum": "mixed", "scenario_id": "config_117", "h5_path": Path("D:/TSRD/stare/train_stare/config_117.h5"), "target_mode": 0},
    {"stratum": "mixed", "scenario_id": "config_117", "h5_path": Path("D:/TSRD/stare/train_stare/config_117.h5"), "target_mode": 1},
    {"stratum": "mixed", "scenario_id": "config_64",  "h5_path": Path("D:/TSRD/stare/train_stare/config_64.h5"),  "target_mode": 2},
    {"stratum": "mixed", "scenario_id": "config_64",  "h5_path": Path("D:/TSRD/stare/train_stare/config_64.h5"),  "target_mode": 3},
    {"stratum": "mixed", "scenario_id": "config_96",  "h5_path": Path("D:/TSRD/stare/train_stare/config_96.h5"),  "target_mode": 4},
]


def preload_balanced_4x5_buffer() -> None:
    logger.info("Initializing 4x5 Balanced Preload using Parent: %s", PARENT_CKPT_PATH)
    assert PARENT_CKPT_PATH.exists(), f"Parent checkpoint not found: {PARENT_CKPT_PATH}"

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

    buffer = SequenceReplayBuffer(
        capacity=50000,
        seq_len=16,
        obs_dim=CANONICAL_OBS_DIM,
        burn_in=8,
        seed=42 + 820,
    )

    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_OBS_DIM,
        "semantic_memory_path": ":memory:",
        "max_steps_per_episode": 1000,
        "reward": {"version": "v2"},
    }

    stratum_counts: dict[str, int] = {"agile": 0, "sparse": 0, "dense": 0, "mixed": 0}
    mode_counts_total = np.zeros(CANONICAL_N_MODES, dtype=np.int64)
    cell_matrix = np.zeros((4, 5), dtype=np.int64)
    stratum_map = {"agile": 0, "sparse": 1, "dense": 2, "mixed": 3}

    for ep_idx, spec in enumerate(PRELOAD_4X5_SPECS):
        scen_id = spec["scenario_id"]
        h5_path = spec["h5_path"]
        target_mode = spec["target_mode"]
        target_stratum = spec["stratum"]

        assert h5_path.exists(), f"Required scenario H5 file not found: {h5_path}"
        derived_stratum = map_scenario_to_primary_stratum(scen_id)
        assert derived_stratum == target_stratum, f"Stratum mismatch: spec={target_stratum}, derived={derived_stratum}"

        records = load_h5_records(h5_path, chunk_mode="random", seed=42 + ep_idx * 13)
        env = CognitiveRFScanEnv(env_cfg, records=records, seed=42 + ep_idx)
        obs, _ = env.reset(seed=42 + ep_idx)
        hidden = model.init_hidden(1, device)

        ep_modes = np.zeros(CANONICAL_N_MODES, dtype=np.int64)

        for step in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                q_row = q_flat[0, 0].cpu().numpy()

            # Intelligent band selection via parent network policy
            b_greedy = band_of_action(int(np.argmax(q_row)), CANONICAL_N_MODES)

            # Systematically target mode for this episode to enforce strict 4x5 balance
            action = int(b_greedy * CANONICAL_N_MODES + target_mode)
            m = target_mode

            mode_counts_total[m] += 1
            ep_modes[m] += 1
            cell_matrix[stratum_map[target_stratum], m] += 1

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

        stratum_counts[target_stratum] += 1
        logger.info(
            "Generated ep %2d/20: scenario=%-10s stratum=%-6s target_mode=%-10s transitions=%d (hits=%d)",
            ep_idx + 1,
            scen_id,
            target_stratum,
            DWELL_MODES[target_mode],
            len(buffer._episodes[-1]["actions"]),
            sum(buffer._episodes[-1]["hit_probs"]),
        )

    logger.info("=== Preload Generation Complete ===")
    logger.info("Total Episodes in Buffer: %d", buffer.n_episodes())
    logger.info("Total Transitions: %d", len(buffer))
    logger.info("Stratum Breakdown: %s", stratum_counts)
    logger.info(
        "Mode Distribution: %s",
        {DWELL_MODES[i]: int(mode_counts_total[i]) for i in range(CANONICAL_N_MODES)},
    )
    logger.info("4x5 Stratum x Mode Transition Matrix:\n%s", cell_matrix)

    # Invariant Assertions
    assert buffer.n_episodes() == 20, f"Expected 20 episodes, got {buffer.n_episodes()}"
    assert len(buffer) == 20000, f"Expected 20000 transitions, got {len(buffer)}"

    for s_name, count in stratum_counts.items():
        assert count == 5, f"Stratum '{s_name}' has {count} episodes, expected exactly 5 (25.0%)!"

    for m_idx in range(CANONICAL_N_MODES):
        m_count = mode_counts_total[m_idx]
        assert m_count == 4000, f"Mode {DWELL_MODES[m_idx]} has {m_count} transitions, expected exactly 4000 (20.0%)!"

    # Every cell in the 4x5 matrix must have >= 1 episode (1000 transitions)
    assert np.all(cell_matrix >= 1000), f"Some cells in 4x5 matrix have < 1000 transitions:\n{cell_matrix}"

    # Persist Preloaded Buffer
    OUTPUT_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    buffer.save_episodes(OUTPUT_BUFFER_PATH)
    logger.info("Successfully persisted preloaded buffer to %s", OUTPUT_BUFFER_PATH)

    # Sample and Persist Deterministic Probe Batch (64 sequences of length 16 = 512 active steps)
    logger.info("Sampling deterministic probe batch for Shadow-Greedy ModeCollapseGuard...")
    probe_sample = buffer.sample(64, target_hit_seq_fraction=0.40)
    probe_obs = torch.tensor(probe_sample["obs"], dtype=torch.float32)  # (64, 16, 360)
    torch.save({"obs": probe_obs, "burn_in": 8}, OUTPUT_PROBE_PATH)
    logger.info("Successfully saved deterministic probe batch (shape %s) to %s", list(probe_obs.shape), OUTPUT_PROBE_PATH)

    # Validate parent on probe batch
    with torch.no_grad():
        q_probe, _, _ = model(probe_obs)
        active_q = q_probe[:, 8:, :]  # post burn-in steps
        acts = torch.argmax(active_q, dim=-1).flatten().numpy()
        modes = acts % CANONICAL_N_MODES

    p_counts = np.bincount(modes, minlength=5)
    p_probs = p_counts / len(modes)
    from scipy.stats import entropy
    p_entropy = float(entropy(p_probs + 1e-12))
    logger.info(
        "Probe Validation on Parent Model: SHORT=%.1f%%, Entropy=%.3f (counts: %s)",
        p_probs[0] * 100.0,
        p_entropy,
        p_counts.tolist(),
    )
    assert p_probs[0] < 0.88, f"Parent model unexpectedly tripped short ceiling on probe: {p_probs[0]}"
    assert p_entropy > 0.50, f"Parent model unexpectedly low entropy on probe: {p_entropy}"


if __name__ == "__main__":
    preload_balanced_4x5_buffer()
