"""Read-only Interaction Ablation on Phase G7-B Gate-26,000 Checkpoint.

Compares:
  Q_full = V(s) + A_b(s, b) + A_m(s, m) + I(s, b, m)
versus
  Q_ablation = V(s) + A_b(s, b) + A_m(s, m) (with I == 0)

Measures:
1. Fraction of decisions where greedy action flips (a_full != a_ablation).
2. Per-scenario and macro Pd change under ablation.
3. Mode distribution shift under ablation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
from scipy.stats import entropy
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
from ew_core.models.low_rank_coupled_drqn_scheduler import LowRankCoupledDRQNScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("g7b_ablation")

GATE26_PATH = repo_root / "experiments/checkpoints/g7b_low_rank_coupling/checkpoint_gate_26000.pt"
VAL_DIR = Path("D:/TSRD/stare/val_stare")
REPORTS_DIR = repo_root / "reports"

CANONICAL_SCENARIOS = [
    "config_117",
    "config_119",
    "config_143",
    "config_194",
    "config_195",
    "config_241",
    "config_29",
    "config_42",
    "config_64",
    "config_96",
]


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_policy(model: LowRankCoupledDRQNScheduler, ablate_interaction: bool = False) -> dict:
    device = torch.device("cpu")
    model.eval()

    scen_records = {}
    all_actions = []
    all_bands = []
    all_modes = []
    total_hits = 0
    total_dwell_us = 0.0

    for scen_name in CANONICAL_SCENARIOS:
        h5_path = VAL_DIR / f"{scen_name}.h5"
        records = load_h5_records(h5_path, chunk_mode="first")

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
        hidden = model.init_hidden(1, device)

        scen_acts = []
        scen_bnds = []
        scen_mods = []
        scen_hits = 0
        scen_dwell_us = 0.0

        for step in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                if ablate_interaction:
                    # Ablate interaction: Q = V + A_b + A_m (with I = 0)
                    v = aux["v"]
                    a_band_tilde = aux["a_band_tilde"]
                    a_mode_tilde = aux["a_mode_tilde"]
                    q_ablated = v.unsqueeze(-1) + a_band_tilde.unsqueeze(-1) + a_mode_tilde.unsqueeze(-2)
                    q_row = q_ablated.view(1, 1, CANONICAL_N_ACTIONS)[0, 0].cpu().numpy()
                else:
                    q_row = q_flat[0, 0].cpu().numpy()

            act = int(np.argmax(q_row))
            b = band_of_action(act, CANONICAL_N_MODES)
            m = mode_of_action(act, CANONICAL_N_MODES)
            dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m]
            dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

            scen_acts.append(act)
            scen_bnds.append(b)
            scen_mods.append(m)
            scen_dwell_us += dwell_us

            next_obs, reward, term, trunc, info = env.step(act)
            if bool(info.get("hit", False)):
                scen_hits += 1

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        dwell_ms = scen_dwell_us / 1000.0
        m_counts = {DWELL_MODES[i]: int(np.sum(np.array(scen_mods) == i)) for i in range(CANONICAL_N_MODES)}

        scen_records[scen_name] = {
            "steps": len(scen_acts),
            "hits": scen_hits,
            "dwell_ms": dwell_ms,
            "gross_hits_per_ms": scen_hits / dwell_ms if dwell_ms > 0 else 0.0,
            "pd": float(fom.get("Pd", fom.get("pd", 0.0))) * 100.0,
            "modes": m_counts,
            "actions": scen_acts,
        }
        all_actions.extend(scen_acts)
        all_bands.extend(scen_bnds)
        all_modes.extend(scen_mods)
        total_hits += scen_hits
        total_dwell_us += scen_dwell_us

    n_tot_steps = len(all_actions)
    tot_dwell_ms = total_dwell_us / 1000.0
    mode_counts = np.bincount(all_modes, minlength=CANONICAL_N_MODES)
    mode_probs = mode_counts / float(n_tot_steps)
    mode_ent = float(entropy(mode_probs + 1e-12, base=np.e))

    return {
        "mode_entropy": mode_ent,
        "mode_distribution_pct": {
            DWELL_MODES[i]: float(mode_probs[i] * 100.0) for i in range(CANONICAL_N_MODES)
        },
        "mean_pd": float(np.mean([s["pd"] for s in scen_records.values()])),
        "gross_hits_per_ms": total_hits / tot_dwell_ms if tot_dwell_ms > 0 else 0.0,
        "scenarios": scen_records,
        "all_actions": all_actions,
    }


def main():
    sha = compute_sha256(GATE26_PATH)
    logger.info("Evaluating G7-B Gate-26k Checkpoint (SHA: %s)", sha)

    ckpt = torch.load(GATE26_PATH, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    model = LowRankCoupledDRQNScheduler(rank=8)
    model.load_state_dict(state)

    logger.info("--- Running Evaluation with Full Model (Q = V + Ab + Am + I) ---")
    full_eval = evaluate_policy(model, ablate_interaction=False)

    logger.info("--- Running Evaluation with Ablated Interaction (Q = V + Ab + Am, I=0) ---")
    ablated_eval = evaluate_policy(model, ablate_interaction=True)

    # Compute Action Divergence
    actions_full = full_eval["all_actions"]
    actions_abl = ablated_eval["all_actions"]
    assert len(actions_full) == len(actions_abl)

    diff_actions = sum(1 for f, a in zip(actions_full, actions_abl) if f != a)
    action_divergence_pct = (diff_actions / float(len(actions_full))) * 100.0

    logger.info("=" * 80)
    logger.info("INTERACTION ABLATION RESULTS (Gate 26,000):")
    logger.info("  Total Decisions Evaluated: %d", len(actions_full))
    logger.info("  Action Divergence (Full vs Ablated): %d / %d (%.2f%%)",
                diff_actions, len(actions_full), action_divergence_pct)
    logger.info("  Mean Pd: Full = %.2f%% | Ablated = %.2f%% | Delta = %+.2f%%",
                full_eval["mean_pd"], ablated_eval["mean_pd"],
                ablated_eval["mean_pd"] - full_eval["mean_pd"])
    logger.info("  Mode Entropy: Full = %.3f | Ablated = %.3f",
                full_eval["mode_entropy"], ablated_eval["mode_entropy"])
    logger.info("  SHORT Dwell: Full = %.2f%% | Ablated = %.2f%%",
                full_eval["mode_distribution_pct"]["SHORT_DWELL"],
                ablated_eval["mode_distribution_pct"]["SHORT_DWELL"])
    logger.info("  REVISIT: Full = %.2f%% | Ablated = %.2f%%",
                full_eval["mode_distribution_pct"]["REVISIT"],
                ablated_eval["mode_distribution_pct"]["REVISIT"])
    logger.info("=" * 80)

    scenario_comparison = {}
    for s in CANONICAL_SCENARIOS:
        pd_full = full_eval["scenarios"][s]["pd"]
        pd_abl = ablated_eval["scenarios"][s]["pd"]
        acts_f = full_eval["scenarios"][s]["actions"]
        acts_a = ablated_eval["scenarios"][s]["actions"]
        scen_diff = sum(1 for f, a in zip(acts_f, acts_a) if f != a)
        scenario_comparison[s] = {
            "pd_full": pd_full,
            "pd_ablated": pd_abl,
            "delta_pd": pd_abl - pd_full,
            "action_flips": scen_diff,
            "action_flips_pct": (scen_diff / float(len(acts_f))) * 100.0,
        }
        logger.info("  [%s] Pd: Full=%.1f%% -> Ablated=%.1f%% (Delta: %+.1f%%) | Flips: %d (%.1f%%)",
                    s, pd_full, pd_abl, pd_abl - pd_full, scen_diff, (scen_diff / float(len(acts_f))) * 100.0)

    # Clean actions array before dumping json
    del full_eval["all_actions"]
    del ablated_eval["all_actions"]
    for s in CANONICAL_SCENARIOS:
        del full_eval["scenarios"][s]["actions"]
        del ablated_eval["scenarios"][s]["actions"]

    out_data = {
        "timestamp_utc": "2026-09-26T23:05:00Z",
        "checkpoint_sha256": sha,
        "action_divergence_pct": action_divergence_pct,
        "total_action_flips": diff_actions,
        "full_evaluation": full_eval,
        "ablated_evaluation": ablated_eval,
        "scenario_comparison": scenario_comparison,
    }

    out_path = REPORTS_DIR / "g7b_interaction_ablation_manifest.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    logger.info("Saved ablation manifest to %s", out_path)


if __name__ == "__main__":
    main()
