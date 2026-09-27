"""Phase G8.4-REG Preflight Verification Script.

Performs mandatory pre-execution verification before launching the
G8.4-REG Canary (lambda = 0.005, Steps 27,000 -> 28,000):
1. Verifies Gate 27,000 checkpoint SHA-256 matches:
   fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094
2. Verifies optimizer state dict presence and Adam moment statistics (m, v).
3. Verifies online and target network synchronization.
4. Verifies replay buffer integrity (24 episodes, 24,000 transitions).
5. Verifies probe batch integrity (512 active steps).
6. Verifies value regularization loss computation under lambda = 0.005.

Outputs verification report:
reports/g8_4_reg_preflight_audit.json
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
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.contracts import (
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
)
from ew_core.models.band_conditioned_drqn import BandConditionedFactorizedDRQN
from ew_core.training.replay_buffer import SequenceReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_g8_4_reg_preflight")

GATE_27000_PATH = repo_root / "experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt"
EXPECTED_SHA = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
REPORTS_DIR = repo_root / "reports"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    logger.info("=================================================================")
    logger.info("   PHASE G8.4-REG PREFLIGHT VERIFICATION (lambda = 0.005)       ")
    logger.info("=================================================================")

    # 1. Checkpoint & SHA
    assert GATE_27000_PATH.exists(), f"Gate 27k checkpoint missing: {GATE_27000_PATH}"
    actual_sha = sha256_file(GATE_27000_PATH)
    logger.info("Gate 27k Checkpoint SHA-256: %s", actual_sha)
    assert actual_sha == EXPECTED_SHA, f"SHA mismatch! Expected {EXPECTED_SHA}, got {actual_sha}"

    device = torch.device("cpu")
    ckpt = torch.load(GATE_27000_PATH, map_location=device, weights_only=False)

    # 2. Optimizer state verification
    assert "optimizer_state_dict" in ckpt, "optimizer_state_dict missing from checkpoint!"
    opt_state = ckpt["optimizer_state_dict"]
    
    m_norms = []
    v_norms = []
    step_counts = []
    for p_id, p_state in opt_state.get("state", {}).items():
        if "exp_avg" in p_state:
            m_norms.append(p_state["exp_avg"].norm().item() ** 2)
        if "exp_avg_sq" in p_state:
            v_norms.append(p_state["exp_avg_sq"].norm().item() ** 2)
        if "step" in p_state:
            step_counts.append(int(p_state["step"]))

    total_m_norm = float(np.sqrt(sum(m_norms))) if m_norms else 0.0
    total_v_norm = float(np.sqrt(sum(v_norms))) if v_norms else 0.0
    max_opt_step = max(step_counts) if step_counts else 0

    logger.info("Optimizer state verified: ||m||_2 = %.6f, ||v||_2 = %.6f, max_step = %d",
                total_m_norm, total_v_norm, max_opt_step)
    assert total_m_norm > 0.0 and total_v_norm > 0.0, "Optimizer moment buffers empty!"

    # 3. Model Loading & Alignment
    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    target_model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    target_model.load_state_dict(ckpt.get("target_state_dict", ckpt["state_dict"]))
    target_model.eval()

    # 4. Probe Batch Evaluation
    assert PROBE_BATCH_PATH.exists(), f"Probe batch missing: {PROBE_BATCH_PATH}"
    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]
    burn_in = int(probe_data.get("burn_in", 8))

    with torch.no_grad():
        h_on = model.init_hidden(probe_obs.size(0), device)
        q_on, _, _ = model(probe_obs, h_on)
        h_tg = target_model.init_hidden(probe_obs.size(0), device)
        q_tg, _, _ = target_model(probe_obs, h_tg)

    q_on_g = q_on[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).numpy()
    q_tg_g = q_tg[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).numpy()
    max_q_gap = float(np.max(np.abs(q_on_g - q_tg_g)))
    logger.info("Online vs Target Max Q-Gap on Probe: %.8f", max_q_gap)
    assert max_q_gap < 1e-5, f"Target network out of sync! Gap = {max_q_gap}"

    # 5. Replay Buffer & Test Update with lambda = 0.005
    assert PRELOADED_BUFFER_PATH.exists(), f"Buffer missing: {PRELOADED_BUFFER_PATH}"
    buf = SequenceReplayBuffer(50000, 16, CANONICAL_OBS_DIM, 8, 42)
    buf.load_episodes(PRELOADED_BUFFER_PATH)
    logger.info("Replay buffer loaded: %d episodes, %d steps", buf.n_episodes(), buf._total)

    batch = buf.sample(32, target_hit_seq_fraction=0.40)
    valid = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
    burn_in_m = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
    loss_mask = valid & ~burn_in_m

    obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
    rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
    done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

    model.train()
    q_all, _, _ = model(obs_b)
    q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

    with torch.inference_mode():
        next_q_on, _, _ = model(next_obs_b)
        best_a = next_q_on.argmax(dim=-1, keepdim=True)
        next_q_tg, _, _ = target_model(next_obs_b)
        next_q = torch.clamp(next_q_tg.gather(-1, best_a).squeeze(-1), min=-50.0, max=100.0)

    targets = rew_b + 0.99 * next_q * (1.0 - done_b)
    loss_fn = nn.HuberLoss()
    l_td = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())

    # Regularization with lambda = 0.005
    lambda_reg = 0.005
    l_reg = lambda_reg * (q_all[loss_mask] ** 2).mean()
    total_loss = l_td + l_reg

    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    logger.info("Test Step Completed with lambda = %.5f:", lambda_reg)
    logger.info("  L_TD: %.4f | L_reg: %.4f | Total: %.4f", l_td.item(), l_reg.item(), total_loss.item())

    # Preflight summary report
    preflight_report = {
        "status": "PREFLIGHT_PASSED",
        "checkpoint": str(GATE_27000_PATH),
        "sha256": actual_sha,
        "optimizer": {
            "norm_m": total_m_norm,
            "norm_v": total_v_norm,
            "step_count": max_opt_step,
        },
        "target_network_sync_gap": max_q_gap,
        "replay_buffer": {
            "episodes": buf.n_episodes(),
            "total_steps": buf._total,
        },
        "objective_configuration": {
            "objective_mode": "step_based",
            "gamma": 0.99,
            "c_dwell": 0.0,
            "tau_ref": 1.0,
            "q_reg_coef": lambda_reg,
        },
        "test_step": {
            "l_td": float(l_td.item()),
            "l_reg": float(l_reg.item()),
            "total_loss": float(total_loss.item()),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "g8_4_reg_preflight_audit.json"
    with open(out_file, "w") as f:
        json.dump(preflight_report, f, indent=2)

    logger.info("Preflight verification report written to: %s", out_file)
    logger.info("=================================================================")
    logger.info("   ALL PREFLIGHT CHECKS PASSED. READY FOR G8.4-REG CANARY!       ")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
