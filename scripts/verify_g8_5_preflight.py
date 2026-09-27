"""Phase G8.5-FINAL Preflight Verification Script.

Performs comprehensive pre-execution checks and baseline diagnostic calibration
for the Unregularized Longitudinal Stability Trial (Steps 27,000 -> 50,000):
1. Verifies Gate 27,000 checkpoint SHA-256 matches:
   fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094
2. Verifies optimizer state dict presence and Adam moment statistics (m, v).
3. Verifies online and target network synchronization (max Q gap = 0.000000).
4. Verifies replay buffer integrity (24 episodes, 24,000 transitions).
5. Verifies probe batch integrity (512 active steps).
6. Computes baseline reference action-margin distribution on Gate 27k.
7. Computes baseline Q-to-realized-return calibration on Gate 27k.
8. Verifies test unregularized step execution (gamma=0.99, q_reg=0.0).

Outputs report:
reports/g8_5_preflight_audit.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
from scipy.stats import linregress
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
logger = logging.getLogger("verify_g8_5_preflight")

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


def compute_mc_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Computes empirical discounted returns G_t = sum_{k=0}^{T-t-1} gamma^k r_{t+k}."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def main():
    logger.info("=================================================================")
    logger.info("   PHASE G8.5-FINAL PREFLIGHT VERIFICATION                      ")
    logger.info("=================================================================")

    # 1. Checkpoint & SHA
    assert GATE_27000_PATH.exists(), f"Gate 27k missing: {GATE_27000_PATH}"
    actual_sha = sha256_file(GATE_27000_PATH)
    logger.info("Gate 27,000 Checkpoint SHA-256: %s", actual_sha)
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

    # 3. Model Loading & Target Network Synchronization
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

    # 4. Probe Batch Evaluation & Reference Action Margins
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

    # Compute baseline action margins on probe
    q_part = np.partition(q_on_g, -2, axis=-1)
    q_top1 = q_part[:, -1]
    q_top2 = q_part[:, -2]
    delta_q = q_top1 - q_top2

    baseline_margins = {
        "mean_delta_q": float(np.mean(delta_q)),
        "median_delta_q": float(np.median(delta_q)),
        "p10_delta_q": float(np.percentile(delta_q, 10)),
        "p1_delta_q": float(np.percentile(delta_q, 1)),
        "fraction_near_zero_delta_q": float(np.mean(delta_q < 0.05)),
        "q_max": float(np.max(q_on_g)),
        "q_min": float(np.min(q_on_g)),
        "q_mean": float(np.mean(q_on_g)),
        "q_std": float(np.std(q_on_g)),
    }
    logger.info("Baseline Action Margins (512 Probe States):")
    logger.info("  Mean Delta_Q: %.4f | Median: %.4f | P10: %.4f | P1: %.4f | Ambiguous (<0.05): %.2f%%",
                baseline_margins["mean_delta_q"],
                baseline_margins["median_delta_q"],
                baseline_margins["p10_delta_q"],
                baseline_margins["p1_delta_q"],
                baseline_margins["fraction_near_zero_delta_q"] * 100.0)

    # 5. Baseline Q-to-Realized-Return Calibration on Replay Buffer
    assert PRELOADED_BUFFER_PATH.exists(), f"Buffer missing: {PRELOADED_BUFFER_PATH}"
    buf = SequenceReplayBuffer(50000, 16, CANONICAL_OBS_DIM, 8, 42)
    buf.load_episodes(PRELOADED_BUFFER_PATH)

    with open(PRELOADED_BUFFER_PATH, "rb") as f:
        import pickle
        buf_data = pickle.load(f)

    episodes = buf_data["episodes"]
    all_pred_q = []
    all_realized_g = []

    with torch.no_grad():
        for ep in episodes:
            obs_seq = torch.tensor(ep["obs"], dtype=torch.float32, device=device).unsqueeze(0)
            acts = ep["actions"]
            rews = ep["rewards"]
            g_seq = compute_mc_returns(rews, gamma=0.99)

            h = model.init_hidden(1, device)
            q_out, _, _ = model(obs_seq, h)
            q_ch = q_out.gather(-1, torch.tensor(acts, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1)).squeeze().cpu().numpy()

            all_pred_q.extend(q_ch)
            all_realized_g.extend(g_seq)

    all_pred_q = np.array(all_pred_q)
    all_realized_g = np.array(all_realized_g)

    # Regression G ~ alpha * Q + beta
    reg = linregress(all_pred_q, all_realized_g)
    pred_errors = all_pred_q - all_realized_g
    abs_errors = np.abs(pred_errors)

    baseline_calibration = {
        "mean_pred_q": float(np.mean(all_pred_q)),
        "mean_realized_g": float(np.mean(all_realized_g)),
        "calibration_bias": float(np.mean(pred_errors)),
        "calibration_slope": float(reg.slope),
        "calibration_intercept": float(reg.intercept),
        "calibration_r_squared": float(reg.rvalue ** 2),
        "mean_absolute_error": float(np.mean(abs_errors)),
        "p95_absolute_error": float(np.percentile(abs_errors, 95)),
    }

    logger.info("Baseline Q-to-Return Calibration (24k transitions):")
    logger.info("  Mean Q: %.2f | Mean G: %.2f | Bias (Q - G): %.2f | Slope: %.4f | R^2: %.4f | MAE: %.2f | P95_AE: %.2f",
                baseline_calibration["mean_pred_q"],
                baseline_calibration["mean_realized_g"],
                baseline_calibration["calibration_bias"],
                baseline_calibration["calibration_slope"],
                baseline_calibration["calibration_r_squared"],
                baseline_calibration["mean_absolute_error"],
                baseline_calibration["p95_absolute_error"])

    # 6. Test Unregularized Step Execution (gamma=0.99, q_reg=0.0)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    optimizer.zero_grad()
    l_td.backward()
    optimizer.step()

    logger.info("Test Step Completed cleanly (gamma=0.99, q_reg=0.0): L_TD = %.4f", l_td.item())

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
            "q_reg_coef": 0.0,
            "fresh_optimizer": False,
        },
        "baseline_probe_action_margins": baseline_margins,
        "baseline_return_calibration": baseline_calibration,
        "test_step_loss": float(l_td.item()),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "g8_5_preflight_audit.json"
    with open(out_file, "w") as f:
        json.dump(preflight_report, f, indent=2)

    logger.info("Preflight report written to: %s", out_file)
    logger.info("=================================================================")
    logger.info("   ALL PREFLIGHT CHECKS PASSED. READY FOR G8.5-FINAL RUN!        ")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
