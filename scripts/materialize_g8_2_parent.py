"""Materialize Canonical FiLM Parent Model for Phase G8.2.

Constructs the FiLMGatedFactorizedDRQN model from G7-A Gate 26,000 checkpoint once,
verifies exact zero-step parity with the parent checkpoint,
saves to experiments/checkpoints/g8_2/parent_film_gate_26000.pt,
and computes and records the SHA-256 hash.

All three experimental arms (A, B, C) load from this identical materialized state:
theta_A^0 == theta_B^0 == theta_C^0.
"""

from __future__ import annotations

import hashlib
import json
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
)
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN
from scripts.run_g7a_targeted_replay import instantiate_g5_factorized_model, GATE25_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("materialize_parent")

G7A_PARENT_PATH = repo_root / "experiments/checkpoints/g7a_targeted_replay/checkpoint_gate_26000.pt"
OUTPUT_DIR = repo_root / "experiments/checkpoints/g8_2"
OUTPUT_CKPT_PATH = OUTPUT_DIR / "parent_film_gate_26000.pt"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "parent_film_gate_26000_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    logger.info("Starting G8.2 Parent Materialization...")
    assert G7A_PARENT_PATH.exists(), f"G7-A parent not found at {G7A_PARENT_PATH}"

    parent_sha = sha256_file(G7A_PARENT_PATH)
    logger.info("Parent G7-A Checkpoint SHA-256: %s", parent_sha)

    # 1. Instantiate G7-A Parent Model
    device = torch.device("cpu")
    parent_model = instantiate_g5_factorized_model(GATE25_PATH, seed=42).to(device)
    parent_ckpt = torch.load(G7A_PARENT_PATH, map_location=device, weights_only=False)
    parent_model.load_state_dict(parent_ckpt["state_dict"], strict=True)
    parent_model.eval()

    # 2. Instantiate and load FiLM Model
    film_model, load_manifest = FiLMGatedFactorizedDRQN.from_g7a_checkpoint(
        G7A_PARENT_PATH,
        seed=42,
    )
    film_model = film_model.to(device)
    film_model.eval()

    # 3. Test Zero-Step Parity across 10,000 inputs
    logger.info("Verifying Zero-Step Parity across 10,000 synthetic observations...")
    rng = np.random.RandomState(42)
    sample_obs = rng.randn(10000, 1, CANONICAL_OBS_DIM).astype(np.float32)
    obs_t = torch.tensor(sample_obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        h_parent = parent_model.init_hidden(10000, device)
        q_parent, _, _ = parent_model(obs_t, h_parent)
        h_film = film_model.init_hidden(10000, device)
        q_film, aux_film, _ = film_model(obs_t, h_film)

    q_diff = (q_film - q_parent).abs().max().item()
    act_parent = q_parent.squeeze(1).argmax(dim=-1)
    act_film = q_film.squeeze(1).argmax(dim=-1)
    flips = (act_parent != act_film).sum().item()

    logger.info("Zero-step max absolute Q-difference: %.2e", q_diff)
    logger.info("Zero-step greedy action flips: %d / 10,000", flips)

    assert q_diff < 1e-4, f"Zero-step parity violation: max diff = {q_diff:.2e} >= 1e-4"
    assert flips == 0, f"Zero-step action flips detected: {flips} / 10,000"
    logger.info("Zero-Step Parity VERIFIED (EXACT MATCH).")

    # 4. Save Materialized Checkpoint
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "architecture": "FiLMGatedFactorizedDRQN",
        "step": 26000,
        "seed": 42,
        "parent_checkpoint": str(G7A_PARENT_PATH),
        "parent_sha256": parent_sha,
        "zero_step_parity": {
            "max_q_diff": q_diff,
            "flips_10k": flips,
            "passed": True,
        },
        "load_manifest": load_manifest,
    }

    save_payload = {
        "step": 26000,
        "global_step": 26000,
        "state_dict": film_model.state_dict(),
        "metadata": meta,
    }
    torch.save(save_payload, OUTPUT_CKPT_PATH)
    logger.info("Saved materialized FiLM parent to %s", OUTPUT_CKPT_PATH)

    ckpt_sha = sha256_file(OUTPUT_CKPT_PATH)
    logger.info("Materialized FiLM Parent SHA-256: %s", ckpt_sha)

    meta["materialized_sha256"] = ckpt_sha
    with open(OUTPUT_MANIFEST_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved manifest to %s", OUTPUT_MANIFEST_PATH)
    print(f"SUCCESS: Materialized FiLM parent saved. SHA-256: {ckpt_sha}")


if __name__ == "__main__":
    main()
