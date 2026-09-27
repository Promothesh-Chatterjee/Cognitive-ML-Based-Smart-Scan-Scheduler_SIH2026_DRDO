"""Phase G8.3 Diagnostic Q-Component Decomposition Audit on Canonical config_29 States.

Performs a rigorous, read-only mathematical decomposition of pairwise mode margins:
Delta Q_{LONG, j}(s, b*) = Q(s, b*, LONG) - Q(s, b*, j)
                        = [gamma_L A_L - gamma_j A_j] + [beta_L - beta_j]
                        = Delta(gamma A) + Delta(beta)

Evaluates both:
1. Common Parent Checkpoint (Gate 26,000, where LONG was intact)
2. Phase G8.3-A Canary Checkpoint (Gate 26,500, after 15% targeted replay)

Reports separately for LONG vs REVISIT, LONG vs PREEMPTIVE, and LONG vs BEST_OTHER:
- Delta(gamma A) (contribution of the shared mode head under FiLM scaling)
- Delta beta (contribution of the additive band-conditioned FiLM bias)
- Delta A (unweighted centered mode advantage difference: A_L - A_j)
- Total Delta Q
- Fraction of states where each component is positive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

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
    DWELL_MODES,
    band_of_action,
    mode_of_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("diagnose_q_decomposition")

PARENT_CKPT = repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt"
CANARY_CKPT = repo_root / "experiments/checkpoints/g8_3a_targeted_replay/checkpoint_gate_26500.pt"
C29_H5_PATH = Path("D:/TSRD/stare/val_stare/config_29.h5")
OUTPUT_REPORT = repo_root / "reports/g8_3a_q_decomposition_audit.json"


def audit_model_q_decomposition(ckpt_path: Path, device: torch.device = torch.device("cpu")) -> dict[str, Any]:
    logger.info("Auditing Q-decomposition on %s ...", ckpt_path.name)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    assert C29_H5_PATH.exists(), f"config_29 H5 not found: {C29_H5_PATH}"

    model = FiLMGatedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = payload.get("state_dict", payload.get("online_drqn", payload))
    model.load_state_dict(sd, strict=True)
    model.eval()

    records = load_h5_records(C29_H5_PATH, chunk_mode="first")
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

    # Tracking containers
    stats = {
        "vs_revisit": {"total_dq": [], "delta_gamma_a": [], "delta_beta": [], "delta_a": []},
        "vs_preempt": {"total_dq": [], "delta_gamma_a": [], "delta_beta": [], "delta_a": []},
        "vs_best_other": {"total_dq": [], "delta_gamma_a": [], "delta_beta": [], "delta_a": []},
        "raw_components": {
            "a_long": [], "a_revisit": [], "a_preempt": [],
            "gamma_long": [], "gamma_revisit": [], "gamma_preempt": [],
            "beta_long": [], "beta_revisit": [], "beta_preempt": [],
        },
    }

    hits = 0
    actions_taken = []

    for step in range(1000):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_flat, aux, hidden = model(obs_t, hidden)

        q_flat_np = q_flat[0, 0].cpu().numpy()
        q_grid = q_flat_np.reshape(CANONICAL_N_BANDS, CANONICAL_N_MODES)

        # Selected greedy band
        b_star = int(np.argmax(np.max(q_grid, axis=-1)))

        # Mode index definitions
        m_long = 2
        m_rev = 3
        m_pre = 4

        # Extract components at (s, b*)
        # aux["a_mode_tilde"]: (1, 1, 5)
        # aux["film_gamma"]: (1, 1, 36, 5)
        # aux["film_beta"]: (1, 1, 36, 5)
        a_mode = aux["a_mode_tilde"][0, 0].cpu().numpy()
        film_gamma = aux["film_gamma"][0, 0, b_star].cpu().numpy()
        film_beta = aux["film_beta"][0, 0, b_star].cpu().numpy()

        a_L = float(a_mode[m_long])
        a_R = float(a_mode[m_rev])
        a_P = float(a_mode[m_pre])

        g_L = float(film_gamma[m_long])
        g_R = float(film_gamma[m_rev])
        g_P = float(film_gamma[m_pre])

        b_L = float(film_beta[m_long])
        b_R = float(film_beta[m_rev])
        b_P = float(film_beta[m_pre])

        # Record raw components
        stats["raw_components"]["a_long"].append(a_L)
        stats["raw_components"]["a_revisit"].append(a_R)
        stats["raw_components"]["a_preempt"].append(a_P)
        stats["raw_components"]["gamma_long"].append(g_L)
        stats["raw_components"]["gamma_revisit"].append(g_R)
        stats["raw_components"]["gamma_preempt"].append(g_P)
        stats["raw_components"]["beta_long"].append(b_L)
        stats["raw_components"]["beta_revisit"].append(b_R)
        stats["raw_components"]["beta_preempt"].append(b_P)

        # 1. LONG vs REVISIT
        d_gamma_a_rev = (g_L * a_L) - (g_R * a_R)
        d_beta_rev = b_L - b_R
        total_dq_rev = d_gamma_a_rev + d_beta_rev
        d_a_rev = a_L - a_R

        stats["vs_revisit"]["total_dq"].append(total_dq_rev)
        stats["vs_revisit"]["delta_gamma_a"].append(d_gamma_a_rev)
        stats["vs_revisit"]["delta_beta"].append(d_beta_rev)
        stats["vs_revisit"]["delta_a"].append(d_a_rev)

        # 2. LONG vs PREEMPTIVE
        d_gamma_a_pre = (g_L * a_L) - (g_P * a_P)
        d_beta_pre = b_L - b_P
        total_dq_pre = d_gamma_a_pre + d_beta_pre
        d_a_pre = a_L - a_P

        stats["vs_preempt"]["total_dq"].append(total_dq_pre)
        stats["vs_preempt"]["delta_gamma_a"].append(d_gamma_a_pre)
        stats["vs_preempt"]["delta_beta"].append(d_beta_pre)
        stats["vs_preempt"]["delta_a"].append(d_a_pre)

        # 3. LONG vs BEST OTHER MODE
        q_row_band = q_grid[b_star]
        other_modes = [0, 1, 3, 4]
        best_other_m = other_modes[int(np.argmax([q_row_band[m] for m in other_modes]))]

        a_O = float(a_mode[best_other_m])
        g_O = float(film_gamma[best_other_m])
        b_O = float(film_beta[best_other_m])

        d_gamma_a_best = (g_L * a_L) - (g_O * a_O)
        d_beta_best = b_L - b_O
        total_dq_best = d_gamma_a_best + d_beta_best
        d_a_best = a_L - a_O

        stats["vs_best_other"]["total_dq"].append(total_dq_best)
        stats["vs_best_other"]["delta_gamma_a"].append(d_gamma_a_best)
        stats["vs_best_other"]["delta_beta"].append(d_beta_best)
        stats["vs_best_other"]["delta_a"].append(d_a_best)

        # Step environment using model greedy action
        act = int(np.argmax(q_flat_np))
        actions_taken.append(act)
        next_obs, rew, term, trunc, info = env.step(act)
        if bool(info.get("hit", False)):
            hits += 1

        obs = next_obs
        if term or trunc:
            break

    fom = env.get_fom()
    pd_val = float(fom.get("Pd", hits / max(1, len(actions_taken)))) * 100.0
    modes_taken = [a % CANONICAL_N_MODES for a in actions_taken]
    mode_dist = np.bincount(modes_taken, minlength=5) / max(1, len(modes_taken))

    def summarize_vector(vec: list[float]) -> dict[str, float]:
        arr = np.array(vec, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "pos_fraction": float(np.mean(arr > 0.0)),
        }

    summary = {
        "checkpoint": str(ckpt_path),
        "steps_evaluated": len(actions_taken),
        "pd": pd_val,
        "hits": hits,
        "mode_distribution": {DWELL_MODES[i]: float(mode_dist[i]) for i in range(5)},
        "vs_revisit": {k: summarize_vector(v) for k, v in stats["vs_revisit"].items()},
        "vs_preempt": {k: summarize_vector(v) for k, v in stats["vs_preempt"].items()},
        "vs_best_other": {k: summarize_vector(v) for k, v in stats["vs_best_other"].items()},
        "raw_means": {k: float(np.mean(v)) for k, v in stats["raw_components"].items()},
    }
    return summary


def main():
    logger.info("=================================================================")
    logger.info("   PHASE G8.3 Q-COMPONENT DECOMPOSITION AUDIT                   ")
    logger.info("=================================================================")

    parent_results = audit_model_q_decomposition(PARENT_CKPT)
    canary_results = audit_model_q_decomposition(CANARY_CKPT)

    audit_report = {
        "parent_gate_26000": parent_results,
        "canary_gate_26500": canary_results,
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(audit_report, f, indent=2)
    logger.info("Saved audit report to %s", OUTPUT_REPORT)

    print("\n===================================================================================================")
    print("                     Q-COMPONENT DECOMPOSITION AUDIT: config_29 STATES                             ")
    print("===================================================================================================")
    print(f"{'Metric':<32} | {'Parent Gate 26,000':<32} | {'Canary Gate 26,500':<32}")
    print("-" * 101)
    print(f"{'Evaluated Pd':<32} | {parent_results['pd']:>6.2f}% (hits={parent_results['hits']}){' '*13} | {canary_results['pd']:>6.2f}% (hits={canary_results['hits']}){' '*13}")
    print(f"{'LONG Selection Rate':<32} | {parent_results['mode_distribution']['LONG_DWELL']:>6.1%}{' '*25} | {canary_results['mode_distribution']['LONG_DWELL']:>6.1%}{' '*25}")
    print(f"{'REVISIT Selection Rate':<32} | {parent_results['mode_distribution']['REVISIT']:>6.1%}{' '*25} | {canary_results['mode_distribution']['REVISIT']:>6.1%}{' '*25}")
    print(f"{'PREEMPT Selection Rate':<32} | {parent_results['mode_distribution']['PREEMPTIVE_INTERCEPT']:>6.1%}{' '*25} | {canary_results['mode_distribution']['PREEMPTIVE_INTERCEPT']:>6.1%}{' '*25}")
    print("-" * 101)

    p_vs = parent_results["vs_best_other"]
    c_vs = canary_results["vs_best_other"]

    print("--- LONG vs BEST-OTHER MODE ---")
    print(f"{'Total Delta Q (Mean)':<32} | {p_vs['total_dq']['mean']:>+8.4f} (pos={p_vs['total_dq']['pos_fraction']:>5.1%}){' '*10} | {c_vs['total_dq']['mean']:>+8.4f} (pos={c_vs['total_dq']['pos_fraction']:>5.1%}){' '*10}")
    print(f"{'Delta(gamma * A) (Mean)':<32} | {p_vs['delta_gamma_a']['mean']:>+8.4f} (pos={p_vs['delta_gamma_a']['pos_fraction']:>5.1%}){' '*10} | {c_vs['delta_gamma_a']['mean']:>+8.4f} (pos={c_vs['delta_gamma_a']['pos_fraction']:>5.1%}){' '*10}")
    print(f"{'Delta beta (Mean)':<32} | {p_vs['delta_beta']['mean']:>+8.4f} (pos={p_vs['delta_beta']['pos_fraction']:>5.1%}){' '*10} | {c_vs['delta_beta']['mean']:>+8.4f} (pos={c_vs['delta_beta']['pos_fraction']:>5.1%}){' '*10}")
    print(f"{'Delta A (Unweighted)':<32} | {p_vs['delta_a']['mean']:>+8.4f} (pos={p_vs['delta_a']['pos_fraction']:>5.1%}){' '*10} | {c_vs['delta_a']['mean']:>+8.4f} (pos={c_vs['delta_a']['pos_fraction']:>5.1%}){' '*10}")

    print("\n--- LONG vs REVISIT (Dominant Competing Mode) ---")
    p_rev = parent_results["vs_revisit"]
    c_rev = canary_results["vs_revisit"]
    print(f"{'Total Delta Q (Mean)':<32} | {p_rev['total_dq']['mean']:>+8.4f} (pos={p_rev['total_dq']['pos_fraction']:>5.1%}){' '*10} | {c_rev['total_dq']['mean']:>+8.4f} (pos={c_rev['total_dq']['pos_fraction']:>5.1%}){' '*10}")
    print(f"{'Delta(gamma * A) (Mean)':<32} | {p_rev['delta_gamma_a']['mean']:>+8.4f} (pos={p_rev['delta_gamma_a']['pos_fraction']:>5.1%}){' '*10} | {c_rev['delta_gamma_a']['mean']:>+8.4f} (pos={c_rev['delta_gamma_a']['pos_fraction']:>5.1%}){' '*10}")
    print(f"{'Delta beta (Mean)':<32} | {p_rev['delta_beta']['mean']:>+8.4f} (pos={p_rev['delta_beta']['pos_fraction']:>5.1%}){' '*10} | {c_rev['delta_beta']['mean']:>+8.4f} (pos={c_rev['delta_beta']['pos_fraction']:>5.1%}){' '*10}")

    print("\n--- RAW COMPONENT MEANS ---")
    p_raw = parent_results["raw_means"]
    c_raw = canary_results["raw_means"]
    print(f"{'A_mode: LONG / REV / PRE':<32} | {p_raw['a_long']:>+.3f} / {p_raw['a_revisit']:>+.3f} / {p_raw['a_preempt']:>+.3f}{' '*8} | {c_raw['a_long']:>+.3f} / {c_raw['a_revisit']:>+.3f} / {c_raw['a_preempt']:>+.3f}{' '*8}")
    print(f"{'gamma: LONG / REV / PRE':<32} | {p_raw['gamma_long']:>.3f} / {p_raw['gamma_revisit']:>.3f} / {p_raw['gamma_preempt']:>.3f}{' '*8} | {c_raw['gamma_long']:>.3f} / {c_raw['gamma_revisit']:>.3f} / {c_raw['gamma_preempt']:>.3f}{' '*8}")
    print(f"{'beta: LONG / REV / PRE':<32} | {p_raw['beta_long']:>+.4f} / {p_raw['beta_revisit']:>+.4f} / {p_raw['beta_preempt']:>+.4f}{' '*4} | {c_raw['beta_long']:>+.4f} / {c_raw['beta_revisit']:>+.4f} / {c_raw['beta_preempt']:>+.4f}{' '*4}")
    print("===================================================================================================\n")


if __name__ == "__main__":
    main()
