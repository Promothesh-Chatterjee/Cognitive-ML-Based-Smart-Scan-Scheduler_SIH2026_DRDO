"""Phase G8.4-REG: Read-Only Value Regularizer Calibration Audit.

Performs a comprehensive, non-training mathematical and empirical calibration
of the value regularizer loss term:
    L(theta) = L_TD(theta) + lambda * L_reg(theta)
where:
    L_TD = Huber(Q(s, a), y_step)
    L_reg = E_{s, a}[ Q(s, a)^2 ]
    y_step = r + 0.99 * max_a' Q(s', a'; theta^-)

Audit Components:
1. Scalar Fixed-Point Predictions:
   Q^*_lambda = r / (1 - gamma + lambda) across reward regimes (dense r=4.0, mean r=0.72, agile r=0.50).
2. Replay Dataset Loss & Gradient Norms:
   Evaluates L_TD, L_reg, ||grad_TD||_2, ||grad_reg||_2, and gradient cosine similarity.
3. Subgroup Breakdown:
   - Dense transitions
   - Agile transitions
   - Top-Q transitions (top 5%)
   - Q in [10, 20]
   - Q in [30, 35]
   - Q in [50, 60] (evaluated on Gate 28k empirical states)
4. Minibatch Sampling Gradient Distribution:
   Evaluates 100 sampled training minibatches (batch_size=32, seq_len=16, burn_in=8).
5. Deterministic Probe Batch Policy Preservation:
   Evaluates Spearman rank correlation rho_rank(lambda), greedy action flips (out of 512),
   and Q-gap preservation following an Adam update step.

Outputs comprehensive JSON report:
reports/g8_4_reg_calibration_audit.json
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
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
logger = logging.getLogger("audit_g8_4_reg_calibration")

GATE_27000_PATH = repo_root / "experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt"
GATE_28000_PATH = repo_root / "experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_28000.pt"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
REPORTS_DIR = repo_root / "reports"

CANDIDATE_LAMBDAS = [
    0.0,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    0.10,
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_scalar_fixed_points(lambdas: List[float], gamma: float = 0.99) -> Dict[str, Any]:
    """Computes theoretical fixed points Q^*_lambda = r / (1 - gamma + lambda)."""
    reward_cases = {
        "dense_positive_peak": 4.0,
        "dense_nominal": 3.8,
        "single_intercept": 1.0,
        "global_mean": 0.72,
        "agile_nominal": 0.50,
    }
    
    table = {}
    for lam in lambdas:
        row = {}
        for r_name, r_val in reward_cases.items():
            denom = max(1e-6, (1.0 - gamma) + lam)
            q_star = r_val / denom
            row[r_name] = round(q_star, 2)
        table[f"{lam:.1e}" if lam > 0 else "0.0"] = row

    # Calculate exact lambda required to bound Q* <= 35.0
    bounds = {}
    for r_name, r_val in reward_cases.items():
        # r / (0.01 + lam) <= 35  =>  0.01 + lam >= r / 35  =>  lam >= r / 35 - 0.01
        lam_needed = max(0.0, (r_val / 35.0) - (1.0 - gamma))
        bounds[r_name] = {
            "reward": r_val,
            "min_lambda_for_q_35": round(lam_needed, 6),
        }

    return {
        "fixed_points": table,
        "analytical_bounds_for_q_le_35": bounds,
    }


def compute_gradient_norms_and_cos(
    model: nn.Module,
    l_td: torch.Tensor,
    l_reg: torch.Tensor,
) -> Tuple[float, float, float]:
    """Computes ||grad_TD||_2, ||grad_reg||_2, and cosine similarity cos(grad_TD, grad_reg)."""
    g_td = torch.autograd.grad(l_td, model.parameters(), retain_graph=True, allow_unused=True)
    g_reg = torch.autograd.grad(l_reg, model.parameters(), retain_graph=False, allow_unused=True)

    flat_g_td = torch.cat([g.reshape(-1) for g in g_td if g is not None])
    flat_g_reg = torch.cat([g.reshape(-1) for g in g_reg if g is not None])

    norm_td = float(flat_g_td.norm().item())
    norm_reg = float(flat_g_reg.norm().item())

    if norm_td > 1e-9 and norm_reg > 1e-9:
        cos_sim = float((torch.dot(flat_g_td, flat_g_reg) / (norm_td * norm_reg)).item())
    else:
        cos_sim = 0.0

    return norm_td, norm_reg, cos_sim


def main():
    logger.info("=================================================================")
    logger.info("   PHASE G8.4-REG: VALUE REGULARIZER CALIBRATION AUDIT          ")
    logger.info("=================================================================")

    assert GATE_27000_PATH.exists(), f"Gate 27k missing: {GATE_27000_PATH}"
    ckpt_sha = sha256_file(GATE_27000_PATH)
    logger.info("Gate 27,000 Checkpoint SHA-256: %s", ckpt_sha)
    assert ckpt_sha == "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094", "SHA mismatch!"

    device = torch.device("cpu")
    ckpt_27k = torch.load(GATE_27000_PATH, map_location=device, weights_only=False)

    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    model.load_state_dict(ckpt_27k["state_dict"])
    model.eval()

    target_model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    target_model.load_state_dict(ckpt_27k.get("target_state_dict", ckpt_27k["state_dict"]))
    target_model.eval()

    loss_fn = nn.HuberLoss()

    # 1. Analytic Scalar Fixed-Point Analysis
    logger.info("Part 1: Computing Analytic Scalar Fixed-Point Predictions...")
    scalar_analysis = compute_scalar_fixed_points(CANDIDATE_LAMBDAS, gamma=0.99)
    for k, v in scalar_analysis["analytical_bounds_for_q_le_35"].items():
        logger.info("  Regime '%s' (r=%.2f): min lambda for Q*<=35 is %.5f", k, v["reward"], v["min_lambda_for_q_35"])

    # 2. Replay Buffer Extraction & Value Distribution
    logger.info("Part 2: Loading Replay Buffer (24k transitions)...")
    assert PRELOADED_BUFFER_PATH.exists(), f"Buffer missing: {PRELOADED_BUFFER_PATH}"
    buf = SequenceReplayBuffer(50000, 16, CANONICAL_OBS_DIM, 8, 42)
    buf.load_episodes(PRELOADED_BUFFER_PATH)

    with open(PRELOADED_BUFFER_PATH, "rb") as f:
        import pickle
        buf_data = pickle.load(f)

    # Collect transition-level information across all 24 episodes
    episodes = buf_data["episodes"]
    all_q_chosen = []
    all_q_all_max = []
    all_rewards = []
    all_strata = []
    all_scenarios = []

    with torch.no_grad():
        for ep_idx, ep in enumerate(episodes):
            obs_t = torch.tensor(ep["obs"], dtype=torch.float32, device=device).unsqueeze(0)
            acts_t = torch.tensor(ep["actions"], dtype=torch.long, device=device).unsqueeze(0)
            rews_t = ep["rewards"]
            stratum = ep.get("primary_stratum", "unknown")
            scen_id = ep.get("scenario_id", "unknown")

            q_out, _, _ = model(obs_t)  # (1, 1000, 180)
            q_ch = q_out.gather(-1, acts_t.unsqueeze(-1)).squeeze(-1).squeeze(0).numpy()
            q_m = q_out.max(dim=-1).values.squeeze(0).numpy()

            all_q_chosen.extend(q_ch)
            all_q_all_max.extend(q_m)
            all_rewards.extend(rews_t)
            all_strata.extend([stratum] * len(q_ch))
            all_scenarios.extend([scen_id] * len(q_ch))

    all_q_chosen = np.array(all_q_chosen)
    all_q_all_max = np.array(all_q_all_max)
    all_rewards = np.array(all_rewards)
    all_strata = np.array(all_strata)
    all_scenarios = np.array(all_scenarios)

    logger.info("Q_chosen distribution: min=%.2f, mean=%.2f, median=%.2f, max=%.2f, std=%.2f",
                all_q_chosen.min(), all_q_chosen.mean(), np.median(all_q_chosen), all_q_chosen.max(), all_q_chosen.std())

    # Subgroup masks
    dense_mask = (all_strata == "dense")
    agile_mask = (all_strata == "agile")
    top_q_thresh = np.percentile(all_q_chosen, 95)
    top_q_mask = (all_q_chosen >= top_q_thresh)
    q_10_20_mask = (all_q_chosen >= 10.0) & (all_q_chosen < 20.0)
    q_30_35_mask = (all_q_chosen >= 30.0) & (all_q_chosen <= 35.0)
    q_50_60_mask = (all_q_chosen >= 50.0) & (all_q_chosen <= 60.0)

    logger.info("Transition counts by subgroup:")
    logger.info("  Dense: %d (%.1f%%)", np.sum(dense_mask), np.mean(dense_mask) * 100)
    logger.info("  Agile: %d (%.1f%%)", np.sum(agile_mask), np.mean(agile_mask) * 100)
    logger.info("  Top-Q (>=%.2f): %d (%.1f%%)", top_q_thresh, np.sum(top_q_mask), np.mean(top_q_mask) * 100)
    logger.info("  Q in [10, 20]: %d (%.1f%%)", np.sum(q_10_20_mask), np.mean(q_10_20_mask) * 100)
    logger.info("  Q in [30, 35]: %d (%.2f%%)", np.sum(q_30_35_mask), np.mean(q_30_35_mask) * 100)
    logger.info("  Q in [50, 60] at Gate 27k: %d (0.0%%)", np.sum(q_50_60_mask))

    # Evaluate Gate 28k for the Q in [50, 60] empirical regime
    logger.info("Evaluating Gate 28k checkpoint for Q in [50, 60] danger zone states...")
    ckpt_28k = torch.load(GATE_28000_PATH, map_location=device, weights_only=False)
    model_28k = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    model_28k.load_state_dict(ckpt_28k["state_dict"])
    model_28k.eval()

    all_q_28k = []
    with torch.no_grad():
        for ep in episodes:
            obs_t = torch.tensor(ep["obs"], dtype=torch.float32, device=device).unsqueeze(0)
            acts_t = torch.tensor(ep["actions"], dtype=torch.long, device=device).unsqueeze(0)
            q_out, _, _ = model_28k(obs_t)
            q_ch = q_out.gather(-1, acts_t.unsqueeze(-1)).squeeze(-1).squeeze(0).numpy()
            all_q_28k.extend(q_ch)
    all_q_28k = np.array(all_q_28k)
    q_50_60_mask_28k = (all_q_28k >= 50.0) & (all_q_28k <= 60.0)
    logger.info("  Gate 28k Q in [50, 60] count: %d (%.1f%%)", np.sum(q_50_60_mask_28k), np.mean(q_50_60_mask_28k) * 100)

    # 3. Minibatch Sampling Gradient Distribution (100 sampled minibatches)
    logger.info("Part 3: Measuring Minibatch Gradient Norms across 100 sampled training batches...")
    minibatch_norms_td = []
    minibatch_norms_reg = []
    minibatch_cos_sims = []
    minibatch_l_td = []
    minibatch_l_reg = []

    model.train()
    for b_idx in range(100):
        batch = buf.sample(32, target_hit_seq_fraction=0.40)
        valid = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        burn_in = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
        loss_mask = valid & ~burn_in

        obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
        act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
        rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
        next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
        done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

        q_all, _, _ = model(obs_b)
        q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

        with torch.inference_mode():
            next_q_online, _, _ = model(next_obs_b)
            best_actions = next_q_online.argmax(dim=-1, keepdim=True)
            next_q_target, _, _ = target_model(next_obs_b)
            next_q = next_q_target.gather(-1, best_actions).squeeze(-1)
            next_q = torch.clamp(next_q, min=-50.0, max=100.0)

        targets = rew_b + 0.99 * next_q * (1.0 - done_b)
        l_td = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())
        l_reg = (q_all[loss_mask] ** 2).mean()

        norm_td, norm_reg, cos_s = compute_gradient_norms_and_cos(model, l_td, l_reg)
        minibatch_norms_td.append(norm_td)
        minibatch_norms_reg.append(norm_reg)
        minibatch_cos_sims.append(cos_s)
        minibatch_l_td.append(l_td.item())
        minibatch_l_reg.append(l_reg.item())

    mean_norm_td = float(np.mean(minibatch_norms_td))
    mean_norm_reg = float(np.mean(minibatch_norms_reg))
    mean_cos_sim = float(np.mean(minibatch_cos_sims))
    mean_l_td = float(np.mean(minibatch_l_td))
    mean_l_reg = float(np.mean(minibatch_l_reg))

    logger.info("Minibatch Aggregates (100 batches):")
    logger.info("  Mean L_TD: %.4f (std=%.4f)", mean_l_td, np.std(minibatch_l_td))
    logger.info("  Mean L_reg: %.4f (std=%.4f)", mean_l_reg, np.std(minibatch_l_reg))
    logger.info("  Mean Ratio L_reg/L_TD: %.2f", mean_l_reg / mean_l_td)
    logger.info("  Mean ||grad_TD||: %.4f (std=%.4f)", mean_norm_td, np.std(minibatch_norms_td))
    logger.info("  Mean ||grad_reg|| (unscaled, lam=1): %.4f (std=%.4f)", mean_norm_reg, np.std(minibatch_norms_reg))
    logger.info("  Mean Cosine Similarity: %.6f (std=%.6f)", mean_cos_sim, np.std(minibatch_cos_sims))

    # Evaluate gradient norm ratios across candidate lambdas
    lambda_gradient_ratios = {}
    for lam in CANDIDATE_LAMBDAS:
        scaled_reg_norm = lam * mean_norm_reg
        ratio = scaled_reg_norm / mean_norm_td if mean_norm_td > 0 else 0.0
        # Net gradient force along TD axis: (1 + lam * (||grad_reg|| / ||grad_TD||) * cos_sim)
        effective_brake = ratio * abs(mean_cos_sim)
        lambda_gradient_ratios[f"{lam:.1e}" if lam > 0 else "0.0"] = {
            "lambda": lam,
            "mean_norm_grad_reg": round(scaled_reg_norm, 6),
            "ratio_reg_to_td_pct": round(ratio * 100.0, 3),
            "effective_opposing_brake_pct": round(effective_brake * 100.0, 3),
        }
        logger.info("  lambda=%.1e: ||grad_reg||=%.4f, ratio=%.3f%%, opposing_brake=%.3f%%",
                    lam, scaled_reg_norm, ratio * 100.0, effective_brake * 100.0)

    # 4. Subgroup Gradient & Value Audit
    logger.info("Part 4: Measuring Subgroup Gradient Ratios...")
    subgroups = {
        "dense": [10, 11, 12, 13, 14],
        "agile": [0, 1, 2, 3, 4, 20, 21, 22, 23],
        "mixed": [15, 16, 17, 18, 19],
        "sparse": [5, 6, 7, 8, 9],
    }

    subgroup_audit = {}
    for sg_name, ep_indices in subgroups.items():
        sg_obs = torch.tensor(np.stack([episodes[i]["obs"] for i in ep_indices]), dtype=torch.float32, device=device)
        sg_acts = torch.tensor(np.stack([episodes[i]["actions"] for i in ep_indices]), dtype=torch.long, device=device)
        sg_rews = torch.tensor(np.stack([episodes[i]["rewards"] for i in ep_indices]), dtype=torch.float32, device=device)
        sg_next_obs = torch.tensor(np.stack([episodes[i]["next_obs"] for i in ep_indices]), dtype=torch.float32, device=device)
        sg_dones = torch.tensor(np.stack([episodes[i]["dones"] for i in ep_indices]), dtype=torch.float32, device=device)

        model.train()
        q_all, _, _ = model(sg_obs)
        q_ch = q_all.gather(-1, sg_acts.unsqueeze(-1)).squeeze(-1)

        with torch.inference_mode():
            next_q_on, _, _ = model(sg_next_obs)
            best_a = next_q_on.argmax(dim=-1, keepdim=True)
            next_q_tg, _, _ = target_model(sg_next_obs)
            next_q = torch.clamp(next_q_tg.gather(-1, best_a).squeeze(-1), min=-50.0, max=100.0)

        targets = sg_rews + 0.99 * next_q * (1.0 - sg_dones)
        l_td = loss_fn(q_ch, targets.detach())
        l_reg = (q_all ** 2).mean()

        norm_td, norm_reg, cos_s = compute_gradient_norms_and_cos(model, l_td, l_reg)
        
        per_lam = {}
        for lam in [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]:
            per_lam[f"{lam:.1e}"] = {
                "ratio_pct": round((lam * norm_reg / norm_td) * 100.0, 3) if norm_td > 0 else 0.0,
            }

        subgroup_audit[sg_name] = {
            "n_transitions": len(ep_indices) * 1000,
            "mean_reward": float(sg_rews.mean().item()),
            "l_td": float(l_td.item()),
            "l_reg": float(l_reg.item()),
            "norm_grad_td": float(norm_td),
            "norm_grad_reg_unscaled": float(norm_reg),
            "cos_sim": float(cos_s),
            "candidate_ratios": per_lam,
        }
        logger.info("  Subgroup '%s': L_TD=%.4f, L_reg=%.4f, ||grad_TD||=%.2f, ||grad_reg||=%.2f, cos=%.4f",
                    sg_name, l_td.item(), l_reg.item(), norm_td, norm_reg, cos_s)

    # 5. Deterministic Probe Batch Policy Preservation & Action Flips
    logger.info("Part 5: Auditing Probe Batch Action Rankings and Action Flips...")
    assert PROBE_BATCH_PATH.exists(), f"Probe batch missing: {PROBE_BATCH_PATH}"
    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]  # (64, 16, 360)
    burn_in = int(probe_data.get("burn_in", 8))

    model.eval()
    with torch.no_grad():
        h0 = model.init_hidden(probe_obs.size(0), device)
        q0_all, _, _ = model(probe_obs, h0)
        q0 = q0_all[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).numpy()  # (512, 180)
    base_actions = np.argmax(q0, axis=-1)  # (512,)
    base_max_q = np.max(q0, axis=-1)
    base_runnerup_q = np.partition(q0, -2, axis=-1)[:, -2]
    base_gap = base_max_q - base_runnerup_q

    # Use a fixed reference batch to simulate Adam update steps
    ref_batch = buf.sample(32, target_hit_seq_fraction=0.40)
    r_valid = torch.tensor(ref_batch["valid_mask"], dtype=torch.bool, device=device)
    r_burn_in = torch.tensor(ref_batch["burn_in_mask"], dtype=torch.bool, device=device)
    r_loss_mask = r_valid & ~r_burn_in

    r_obs_b = torch.tensor(ref_batch["obs"], dtype=torch.float32, device=device)
    r_act_b = torch.tensor(ref_batch["actions"], dtype=torch.long, device=device)
    r_rew_b = torch.tensor(ref_batch["rewards"], dtype=torch.float32, device=device)
    r_next_obs_b = torch.tensor(ref_batch["next_obs"], dtype=torch.float32, device=device)
    r_done_b = torch.tensor(ref_batch["dones"], dtype=torch.float32, device=device)

    # First compute baseline TD-only updated model (lambda=0.0)
    m_td = BandConditionedFactorizedDRQN(CANONICAL_OBS_DIM, CANONICAL_N_BANDS, CANONICAL_N_MODES).to(device)
    m_td.load_state_dict(ckpt_27k["state_dict"])
    opt_td = torch.optim.Adam(m_td.parameters(), lr=2.5e-5)
    opt_td.load_state_dict(ckpt_27k["optimizer_state_dict"])

    m_td.train()
    q_all, _, _ = m_td(r_obs_b)
    q_ch = q_all.gather(-1, r_act_b.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q_on, _, _ = m_td(r_next_obs_b)
        best_a = next_q_on.argmax(dim=-1, keepdim=True)
        next_q_tg, _, _ = target_model(r_next_obs_b)
        next_q = torch.clamp(next_q_tg.gather(-1, best_a).squeeze(-1), min=-50.0, max=100.0)
    targets = r_rew_b + 0.99 * next_q * (1.0 - r_done_b)
    l_td_base = loss_fn(q_ch[r_loss_mask], targets[r_loss_mask].detach())
    opt_td.zero_grad()
    l_td_base.backward()
    opt_td.step()

    m_td.eval()
    with torch.no_grad():
        h = m_td.init_hidden(probe_obs.size(0), device)
        q_td_all, _, _ = m_td(probe_obs, h)
        q_td_step = q_td_all[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).numpy()
    td_step_actions = np.argmax(q_td_step, axis=-1)

    probe_results = {}
    for lam in CANDIDATE_LAMBDAS:
        m_sim = BandConditionedFactorizedDRQN(CANONICAL_OBS_DIM, CANONICAL_N_BANDS, CANONICAL_N_MODES).to(device)
        m_sim.load_state_dict(ckpt_27k["state_dict"])
        opt_sim = torch.optim.Adam(m_sim.parameters(), lr=2.5e-5)
        opt_sim.load_state_dict(ckpt_27k["optimizer_state_dict"])

        m_sim.train()
        q_all, _, _ = m_sim(r_obs_b)
        q_ch = q_all.gather(-1, r_act_b.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_on, _, _ = m_sim(r_next_obs_b)
            best_a = next_q_on.argmax(dim=-1, keepdim=True)
            next_q_tg, _, _ = target_model(r_next_obs_b)
            next_q = torch.clamp(next_q_tg.gather(-1, best_a).squeeze(-1), min=-50.0, max=100.0)
        targets = r_rew_b + 0.99 * next_q * (1.0 - r_done_b)
        l_td = loss_fn(q_ch[r_loss_mask], targets[r_loss_mask].detach())
        l_reg = (q_all[r_loss_mask] ** 2).mean() if lam > 0 else torch.zeros((), device=device)
        total_loss = l_td + lam * l_reg

        opt_sim.zero_grad()
        total_loss.backward()
        opt_sim.step()

        m_sim.eval()
        with torch.no_grad():
            h = m_sim.init_hidden(probe_obs.size(0), device)
            q_new_all, _, _ = m_sim(probe_obs, h)
            q_new = q_new_all[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).numpy()

        new_actions = np.argmax(q_new, axis=-1)
        flips_vs_base = int(np.sum(new_actions != base_actions))
        flips_vs_td = int(np.sum(new_actions != td_step_actions))
        rhos = [spearmanr(q0[i], q_new[i]).correlation for i in range(512)]
        mean_rho = float(np.mean(rhos))
        max_delta_q = float(np.max(np.abs(q_new - q0)))
        mean_delta_q = float(np.mean(np.abs(q_new - q0)))

        new_max_q = np.max(q_new, axis=-1)
        new_runnerup_q = np.partition(q_new, -2, axis=-1)[:, -2]
        new_gap = new_max_q - new_runnerup_q
        mean_gap_delta = float(np.mean(new_gap - base_gap))

        probe_results[f"{lam:.1e}" if lam > 0 else "0.0"] = {
            "lambda": lam,
            "flips_vs_gate27": flips_vs_base,
            "flip_rate_vs_gate27_pct": round(flips_vs_base / 512.0 * 100.0, 2),
            "flips_attributable_to_reg_vs_td": flips_vs_td,
            "spearman_rho_mean": round(mean_rho, 6),
            "max_abs_delta_q": round(max_delta_q, 4),
            "mean_abs_delta_q": round(mean_delta_q, 4),
            "mean_top1_gap_change": round(mean_gap_delta, 4),
        }
        logger.info("  lambda=%.1e: flips_vs_base=%d/512 (%.1f%%), flips_vs_td=%d, rho=%.6f, max_dQ=%.2f",
                    lam, flips_vs_base, flips_vs_base / 512.0 * 100.0, flips_vs_td, mean_rho, max_delta_q)

    # 6. Synthesize Calibration Verdict and Candidate Ranking
    logger.info("Part 6: Synthesizing Recommendations...")
    candidate_synthesis = []
    for lam in CANDIDATE_LAMBDAS:
        key = f"{lam:.1e}" if lam > 0 else "0.0"
        q_star_dense = scalar_analysis["fixed_points"][key]["dense_positive_peak"]
        grad_ratio = lambda_gradient_ratios[key]["ratio_reg_to_td_pct"]
        brake_pct = lambda_gradient_ratios[key]["effective_opposing_brake_pct"]
        p_res = probe_results[key]
        
        status = "REJECTED_TOO_WEAK"
        if lam == 0.0:
            status = "BASELINE_UNREGULARIZED"
        elif grad_ratio < 2.0:
            status = "REJECTED_TOO_WEAK"
        elif grad_ratio > 150.0:
            status = "REJECTED_TOO_STRONG"
        elif 10.0 <= grad_ratio <= 100.0 and p_res["spearman_rho_mean"] >= 0.96:
            status = "VIABLE_CANDIDATE"
        else:
            status = "MARGINAL_CANDIDATE"

        candidate_synthesis.append({
            "lambda": lam,
            "q_star_dense_theoretical": q_star_dense,
            "gradient_ratio_pct": grad_ratio,
            "opposing_brake_pct": brake_pct,
            "probe_rho": p_res["spearman_rho_mean"],
            "flips_attributable_to_reg": p_res["flips_attributable_to_reg_vs_td"],
            "disposition": status,
        })

    # Assemble comprehensive report
    report = {
        "metadata": {
            "checkpoint_evaluated": str(GATE_27000_PATH),
            "checkpoint_sha256": ckpt_sha,
            "replay_buffer_evaluated": str(PRELOADED_BUFFER_PATH),
            "probe_batch_evaluated": str(PROBE_BATCH_PATH),
            "objective_mode": "step_based",
            "gamma": 0.99,
            "c_dwell": 0.0,
            "tau_ref": 1.0,
        },
        "part1_scalar_fixed_points": scalar_analysis,
        "part2_replay_q_distribution": {
            "q_chosen_min": float(all_q_chosen.min()),
            "q_chosen_mean": float(all_q_chosen.mean()),
            "q_chosen_median": float(np.median(all_q_chosen)),
            "q_chosen_max": float(all_q_chosen.max()),
            "q_chosen_std": float(all_q_chosen.std()),
            "counts": {
                "dense": int(np.sum(dense_mask)),
                "agile": int(np.sum(agile_mask)),
                "top_q_p95": int(np.sum(top_q_mask)),
                "q_10_20": int(np.sum(q_10_20_mask)),
                "q_30_35": int(np.sum(q_30_35_mask)),
                "q_50_60_gate27": int(np.sum(q_50_60_mask)),
                "q_50_60_gate28": int(np.sum(q_50_60_mask_28k)),
            },
        },
        "part3_minibatch_gradient_dynamics": {
            "mean_l_td": mean_l_td,
            "mean_l_reg": mean_l_reg,
            "mean_norm_grad_td": mean_norm_td,
            "mean_norm_grad_reg_unscaled": mean_norm_reg,
            "mean_cosine_similarity": mean_cos_sim,
            "gradient_ratios_by_lambda": lambda_gradient_ratios,
        },
        "part4_subgroup_analysis": subgroup_audit,
        "part5_probe_policy_preservation": probe_results,
        "part6_candidate_synthesis": candidate_synthesis,
    }

    out_path = REPORTS_DIR / "g8_4_reg_calibration_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Calibration audit report written to: %s", out_path)


if __name__ == "__main__":
    main()
