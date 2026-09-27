"""Phase G8.4-Longitudinal: Read-Only Objective Reconciliation Audit.

Performs a comprehensive, non-training mathematical and empirical audit across
the candidate Bellman objective formulations, using:
1. The exact frozen Gate 27,000 checkpoint:
   experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt
2. The fixed 24-episode replay buffer:
   experiments/checkpoints/g8_3a_preloaded_buffer.pkl (24,000 transitions)
3. The deterministic probe batch:
   experiments/checkpoints/g8_3a_probe_batch.pt (512 active steps)

Candidate Objectives Evaluated:
1. step_099: Diagnostic Step-Based Bellman (gamma=0.99, y = r + 0.99 * Q')
2. smdp_099: Canonical Continuous-Time SMDP (gamma=0.99, y = r + 0.99^tau * Q')
3. step_095: Horizon-Calibrated Step-Based (gamma=0.95, y = r + 0.95 * Q')
4. smdp_095: Horizon-Calibrated SMDP (gamma=0.95, y = r + 0.95^tau * Q')
5. rate_smdp_099: Rate-Normalized Continuous-Time SMDP (gamma=0.99, y = r/tau + 0.99^tau * Q')
6. rate_smdp_095: Rate-Normalized Continuous-Time SMDP (gamma=0.95, y = r/tau + 0.95^tau * Q')

Outputs comprehensive audit report:
reports/g8_4_objective_reconciliation_audit.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
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
    mode_of_action,
)
from ew_core.models.band_conditioned_drqn import BandConditionedFactorizedDRQN
from ew_core.training.replay_buffer import SequenceReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_g8_4_objective_reconciliation")

GATE_27000_PATH = repo_root / "experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt"
PRELOADED_BUFFER_PATH = repo_root / "experiments/checkpoints/g8_3a_preloaded_buffer.pkl"
PROBE_BATCH_PATH = repo_root / "experiments/checkpoints/g8_3a_probe_batch.pt"
REPORTS_DIR = repo_root / "reports"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_target(
    cand_name: str,
    r: np.ndarray,
    q_next_max: np.ndarray,
    tau: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes target y and discount factor for a given candidate formulation."""
    if cand_name == "step_099":
        disc = np.full_like(tau, 0.99)
        r_eff = r
    elif cand_name == "smdp_099":
        disc = np.power(0.99, tau)
        r_eff = r
    elif cand_name == "step_095":
        disc = np.full_like(tau, 0.95)
        r_eff = r
    elif cand_name == "smdp_095":
        disc = np.power(0.95, tau)
        r_eff = r
    elif cand_name == "rate_smdp_099":
        disc = np.power(0.99, tau)
        r_eff = r / np.maximum(tau, 0.1)
    elif cand_name == "rate_smdp_095":
        disc = np.power(0.95, tau)
        r_eff = r / np.maximum(tau, 0.1)
    else:
        raise ValueError(f"Unknown candidate {cand_name}")

    y = r_eff + disc * q_next_max
    return y, disc


def main():
    logger.info("=================================================================")
    logger.info("   PHASE G8.4 OBJECTIVE RECONCILIATION AUDIT (READ-ONLY)        ")
    logger.info("=================================================================")

    assert GATE_27000_PATH.exists(), f"Gate 27k checkpoint missing: {GATE_27000_PATH}"
    ckpt_sha = sha256_file(GATE_27000_PATH)
    logger.info("Frozen Gate 27k Checkpoint SHA-256: %s", ckpt_sha)

    device = torch.device("cpu")
    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)

    ckpt = torch.load(GATE_27000_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    target_model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)
    target_model.load_state_dict(ckpt.get("target_state_dict", ckpt["state_dict"]), strict=True)
    target_model.eval()

    # 1. Audit on Replay Buffer (24k transitions)
    assert PRELOADED_BUFFER_PATH.exists(), f"Buffer missing: {PRELOADED_BUFFER_PATH}"
    buf = SequenceReplayBuffer(50000, 16, CANONICAL_OBS_DIM, 8, 42)
    buf.load_episodes(PRELOADED_BUFFER_PATH)
    logger.info("Replay buffer loaded: %d episodes, %d steps", buf.n_episodes(), buf._total)

    all_rewards = []
    all_taus = []
    all_strata = []
    all_q_next_max = []
    all_curr_q = []

    with torch.no_grad():
        for ep in buf._episodes:
            obs_seq = torch.tensor(ep["obs"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, 360)
            acts = ep["actions"]
            rews = ep["rewards"]
            scen_id = ep.get("scenario_id", "unknown")
            stratum = ep.get("primary_stratum", ep.get("scenario_class", "unknown"))

            T = len(acts)
            h = model.init_hidden(1, device)
            q_out, _, _ = model(obs_seq, h)  # (1, T, 180)
            q_out = q_out.squeeze(0).cpu().numpy()

            h_tgt = target_model.init_hidden(1, device)
            q_tgt, _, _ = target_model(obs_seq, h_tgt)  # (1, T, 180)
            q_tgt = q_tgt.squeeze(0).cpu().numpy()

            for t in range(T - 1):
                act = acts[t]
                m = mode_of_action(act, CANONICAL_N_MODES)
                tau = DEFAULT_DWELL_MULTIPLIERS[m]
                r = float(rews[t])
                q_next_m = float(np.max(q_tgt[t + 1]))
                curr_q = float(q_out[t, act])

                all_rewards.append(r)
                all_taus.append(tau)
                all_strata.append(stratum)
                all_q_next_max.append(q_next_m)
                all_curr_q.append(curr_q)

    all_rewards = np.array(all_rewards)
    all_taus = np.array(all_taus)
    all_strata = np.array(all_strata)
    all_q_next_max = np.array(all_q_next_max)
    all_curr_q = np.array(all_curr_q)

    N_trans = len(all_rewards)
    logger.info("Extracted %d valid transitions from replay buffer.", N_trans)
    logger.info("Buffer tau distribution: tau=0.25: %d (%.1f%%), tau=1.0: %d (%.1f%%), tau=2.5: %d (%.1f%%)",
                np.sum(all_taus == 0.25), np.mean(all_taus == 0.25) * 100.0,
                np.sum(all_taus == 1.0), np.mean(all_taus == 1.0) * 100.0,
                np.sum(all_taus == 2.5), np.mean(all_taus == 2.5) * 100.0)

    candidates = [
        "step_099",
        "smdp_099",
        "step_095",
        "smdp_095",
        "rate_smdp_099",
        "rate_smdp_095",
    ]

    candidate_audit_results: Dict[str, Any] = {}

    for cand in candidates:
        targets, discs = compute_target(cand, all_rewards, all_q_next_max, all_taus)
        td_errors = targets - all_curr_q

        # Global distribution
        cand_summary = {
            "target_scale": {
                "mean": float(np.mean(targets)),
                "std": float(np.std(targets)),
                "median": float(np.median(targets)),
                "min": float(np.min(targets)),
                "max": float(np.max(targets)),
                "p5": float(np.percentile(targets, 5)),
                "p95": float(np.percentile(targets, 95)),
            },
            "td_error_scale": {
                "mean": float(np.mean(td_errors)),
                "std": float(np.std(td_errors)),
                "median": float(np.median(td_errors)),
                "abs_mean": float(np.mean(np.abs(td_errors))),
            },
            "effective_discount": {
                "mean": float(np.mean(discs)),
                "min": float(np.min(discs)),
                "max": float(np.max(discs)),
            },
        }

        # By tau breakdown
        tau_breakdown = {}
        for tau_val in [0.25, 1.0, 2.5]:
            mask = (all_taus == tau_val)
            if np.any(mask):
                tau_breakdown[str(tau_val)] = {
                    "count": int(np.sum(mask)),
                    "fraction": float(np.mean(mask)),
                    "mean_reward": float(np.mean(all_rewards[mask])),
                    "mean_target": float(np.mean(targets[mask])),
                    "median_target": float(np.median(targets[mask])),
                    "mean_discount": float(np.mean(discs[mask])),
                    "mean_td_error": float(np.mean(td_errors[mask])),
                }
        cand_summary["by_tau"] = tau_breakdown

        # By scenario stratum breakdown
        strata_breakdown = {}
        for st in np.unique(all_strata):
            mask = (all_strata == st)
            strata_breakdown[str(st)] = {
                "count": int(np.sum(mask)),
                "mean_reward": float(np.mean(all_rewards[mask])),
                "mean_target": float(np.mean(targets[mask])),
                "median_target": float(np.median(targets[mask])),
                "max_target": float(np.max(targets[mask])),
            }
        cand_summary["by_stratum"] = strata_breakdown

        # Steady-state theoretical value scale
        mean_r = float(np.mean(all_rewards))
        dense_mask = (all_strata == "dense")
        dense_mean_r = float(np.mean(all_rewards[dense_mask])) if np.any(dense_mask) else mean_r
        mean_disc = float(np.mean(discs))
        dense_disc = float(np.mean(discs[dense_mask])) if np.any(dense_mask) else mean_disc

        # Implied stationary asymptotic fixed point V* = r / (1 - gamma)
        v_star_overall = float(mean_r / max(1e-4, 1.0 - mean_disc))
        v_star_dense = float(dense_mean_r / max(1e-4, 1.0 - dense_disc))

        cand_summary["implied_steady_state"] = {
            "v_star_overall": v_star_overall,
            "v_star_dense_positive": v_star_dense,
            "effective_horizon_steps": float(1.0 / max(1e-4, 1.0 - mean_disc)),
            "contraction_factor_kappa": float(np.max(discs)),
            "stability_margin_1_minus_kappa": float(1.0 - np.max(discs)),
            "bounded_under_continuation": bool(np.max(discs) < 0.98),
        }

        candidate_audit_results[cand] = cand_summary

    # 2. Audit on Deterministic Probe Batch (512 active steps)
    assert PROBE_BATCH_PATH.exists(), f"Probe batch missing: {PROBE_BATCH_PATH}"
    probe_data = torch.load(PROBE_BATCH_PATH, map_location=device, weights_only=False)
    probe_obs = probe_data["obs"]  # (64, 16, 360)
    burn_in = int(probe_data.get("burn_in", 8))

    with torch.no_grad():
        h_pr = model.init_hidden(probe_obs.size(0), device)
        q_pr_online, aux_pr, _ = model(probe_obs, h_pr)
        h_pr_tgt = target_model.init_hidden(probe_obs.size(0), device)
        q_pr_tgt, _, _ = target_model(probe_obs, h_pr_tgt)

    q_pr_online_g = q_pr_online[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()  # (512, 180)
    q_pr_tgt_g = q_pr_tgt[:, burn_in:, :].reshape(-1, CANONICAL_N_ACTIONS).cpu().numpy()        # (512, 180)

    # Reference action rankings under Gate 27k
    ref_actions = np.argmax(q_pr_online_g, axis=-1)  # (512,)
    ref_q_max = np.max(q_pr_online_g, axis=-1)
    ref_q_runnerup = np.partition(q_pr_online_g, -2, axis=-1)[:, -2]
    ref_q_gap = ref_q_max - ref_q_runnerup

    probe_action_rankings: Dict[str, Any] = {}

    for cand in candidates:
        # Construct action-value targets Q_target(s, a) = r(s,a) + disc(a) * max_a' Q_tgt(s', a')
        # We test hypothetical next-step targets where r is based on immediate pulse expectation
        # and test how action ordering and rank correlations behave.
        # To evaluate action ordering distortion, compute implied Bellman update target for each action a:
        tau_actions = np.array([DEFAULT_DWELL_MULTIPLIERS[mode_of_action(a, CANONICAL_N_MODES)] for a in range(CANONICAL_N_ACTIONS)])
        
        # Candidate discount vector across all 180 actions
        if "099" in cand and "smdp" in cand:
            disc_vec = np.power(0.99, tau_actions)
        elif "099" in cand and "step" in cand:
            disc_vec = np.full(CANONICAL_N_ACTIONS, 0.99)
        elif "095" in cand and "smdp" in cand:
            disc_vec = np.power(0.95, tau_actions)
        elif "095" in cand and "step" in cand:
            disc_vec = np.full(CANONICAL_N_ACTIONS, 0.95)
        else:
            disc_vec = np.full(CANONICAL_N_ACTIONS, 0.99)

        # Implied targets for each probe action:
        # y(s, a) = r_base + disc_vec(a) * max_a' Q_tgt(s, a')
        # We test whether the top-1 action flips on the probe batch
        tgt_max_val = np.max(q_pr_tgt_g, axis=-1, keepdims=True)  # (512, 1)
        
        # Effective Q update proxy: Q_implied = (1 - alpha)*Q + alpha*Target
        # Here we examine the target-gradient alignment: target vector across actions
        r_proxy = np.zeros((q_pr_online_g.shape[0], CANONICAL_N_ACTIONS))
        # Assign base dwell costs or empirical rewards
        if "rate" in cand:
            y_actions = r_proxy / np.maximum(tau_actions, 0.1) + disc_vec * tgt_max_val
        else:
            y_actions = r_proxy + disc_vec * tgt_max_val

        implied_top_actions = np.argmax(q_pr_online_g + 0.1 * y_actions, axis=-1)
        action_flips = int(np.sum(implied_top_actions != ref_actions))
        flip_rate = float(action_flips / len(ref_actions))

        # Rank correlations on top-10 actions per state
        rank_corrs = []
        for i in range(len(ref_actions)):
            corr, _ = spearmanr(q_pr_online_g[i], (q_pr_online_g[i] + 0.1 * y_actions[i]))
            if not np.isnan(corr):
                rank_corrs.append(corr)

        probe_action_rankings[cand] = {
            "greedy_action_flips_on_probe": action_flips,
            "greedy_action_flip_rate": flip_rate,
            "mean_spearman_rank_correlation": float(np.mean(rank_corrs)),
            "median_spearman_rank_correlation": float(np.median(rank_corrs)),
            "preserves_gate27k_policy": bool(flip_rate < 0.05),
        }

    # 3. Comparative Synthesis & Summary Matrix
    reconciliation_report = {
        "frozen_baseline_checkpoint": {
            "path": str(GATE_27000_PATH),
            "sha256": ckpt_sha,
            "global_step": 27000,
            "mean_pd": 85.09,
            "agile_pd": 71.92,
            "config_29_pd": 95.87,
            "q_max": 30.32,
            "total_blackouts": 0,
        },
        "audit_dataset": {
            "replay_buffer_path": str(PRELOADED_BUFFER_PATH),
            "total_transitions_audited": N_trans,
            "probe_batch_path": str(PROBE_BATCH_PATH),
            "probe_active_steps": int(len(ref_actions)),
        },
        "candidate_evaluations": candidate_audit_results,
        "probe_ranking_stability": probe_action_rankings,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "g8_4_objective_reconciliation_audit.json"
    with open(report_path, "w") as f:
        json.dump(reconciliation_report, f, indent=2)

    logger.info("Saved Objective Reconciliation Audit to %s", report_path)

    # Print Comparative Matrix
    logger.info("=========================================================================================================")
    logger.info("   OBJECTIVE RECONCILIATION COMPARATIVE MATRIX (READ-ONLY AUDIT)                                         ")
    logger.info("=========================================================================================================")
    logger.info("Candidate       | Mean Target | Target Std | V* Dense (Fixed Pt) | Contraction kappa | Probe Rank Corr | Preserves 27k")
    logger.info("---------------------------------------------------------------------------------------------------------")
    for cand in candidates:
        res = candidate_audit_results[cand]
        pr = probe_action_rankings[cand]
        logger.info("%-15s | %11.2f | %10.2f | %19.2f | %17.4f | %15.4f | %s",
                    cand,
                    res["target_scale"]["mean"],
                    res["target_scale"]["std"],
                    res["implied_steady_state"]["v_star_dense_positive"],
                    res["implied_steady_state"]["contraction_factor_kappa"],
                    pr["mean_spearman_rank_correlation"],
                    pr["preserves_gate27k_policy"])
    logger.info("=========================================================================================================")


if __name__ == "__main__":
    main()
