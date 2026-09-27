"""Phase G8-A Post-Hoc FiLM Ablation Diagnostic at Gate 27,000.

Evaluates the exact behavioral contribution of the learned FiLM gate:
Condition A (Full FiLM): Learned gamma(b, m) and beta(b, m)
Condition B (Ablated FiLM): gamma = 1.0, beta = 0.0 (Identity / Pure Factorized)

Measures across all 10 canonical scenarios:
- Greedy action flips (count and %)
- Delta Q statistics: mean |Delta Q|, 95th percentile |Delta Q|, max |Delta Q|
- Pd delta per scenario and overall
- Mode distribution and Hmode delta
- Agile scenario sensitivity (config_29, config_119, config_241)
"""

from __future__ import annotations

import copy
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
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ablate_g8a_film")

GATE27_PATH = repo_root / "experiments/checkpoints/g8a_stabilization/checkpoint_gate_27000.pt"
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
SPARSE_SCENARIOS = {"config_143", "config_119"}
AGILE_SCENARIOS = {"config_119", "config_241", "config_29", "config_195"}


def evaluate_policy(model: FiLMGatedFactorizedDRQN, ablate_film: bool = False, device: torch.device = torch.device("cpu")):
    scen_records = {}
    all_actions = []
    all_modes = []
    all_q_values = []
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
        scen_mods = []
        scen_hits = 0
        scen_dwell_us = 0.0

        for _ in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                if ablate_film:
                    # Counterfactual: V + A_band_tilde + A_mode_tilde (gamma=1, beta=0)
                    v = aux["v"].unsqueeze(-1)
                    a_b = aux["a_band_tilde"].unsqueeze(-1)
                    a_m = aux["a_mode_tilde"].unsqueeze(2)
                    q_flat = (v + a_b + a_m).view(1, 1, 180)

                q_row = q_flat[0, 0].cpu().numpy()

            act = int(np.argmax(q_row))
            m = mode_of_action(act, CANONICAL_N_MODES)
            dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m]
            dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

            scen_acts.append(act)
            scen_mods.append(m)
            scen_dwell_us += dwell_us
            all_q_values.append(q_row)

            next_obs, reward, term, trunc, info = env.step(act)
            if bool(info.get("hit", False)):
                scen_hits += 1

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        pd_val = float(fom.get("Pd", fom.get("pd", 0.0))) * 100.0
        m_counts = {DWELL_MODES[i]: int(np.sum(np.array(scen_mods) == i)) for i in range(CANONICAL_N_MODES)}

        scen_records[scen_name] = {
            "steps": len(scen_acts),
            "hits": scen_hits,
            "pd": pd_val,
            "short_fraction_pct": float(m_counts["SHORT_DWELL"] / float(len(scen_acts))) * 100.0,
            "modes": m_counts,
            "actions": scen_acts,
        }
        all_actions.extend(scen_acts)
        all_modes.extend(scen_mods)
        total_hits += scen_hits
        total_dwell_us += scen_dwell_us

    n_tot_steps = len(all_actions)
    mode_counts = np.bincount(all_modes, minlength=CANONICAL_N_MODES)
    mode_probs = mode_counts / float(n_tot_steps)
    mode_ent = float(entropy(mode_probs + 1e-12, base=np.e))
    all_q_arr = np.array(all_q_values)

    return {
        "mean_pd": float(np.mean([s["pd"] for s in scen_records.values()])),
        "sparse_pd": float(np.mean([scen_records[s]["pd"] for s in SPARSE_SCENARIOS])),
        "agile_pd": float(np.mean([scen_records[s]["pd"] for s in AGILE_SCENARIOS])),
        "mode_entropy": mode_ent,
        "short_pct": float(mode_probs[0] * 100.0),
        "long_pct": float(mode_probs[2] * 100.0),
        "normal_pct": float(mode_probs[1] * 100.0),
        "q_max": float(np.max(all_q_arr)),
        "scenarios": scen_records,
    }


def compute_action_flips_and_delta_q(model: FiLMGatedFactorizedDRQN, device: torch.device = torch.device("cpu")):
    """Run rollout under Condition A, evaluate Condition B Q-values on identical states."""
    delta_qs = []
    action_flips_count = 0
    total_steps = 0
    per_scen_flips = {}

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

        scen_flips = 0

        for _ in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                q_full = q_flat[0, 0].cpu().numpy()

                v = aux["v"].unsqueeze(-1)
                a_b = aux["a_band_tilde"].unsqueeze(-1)
                a_m = aux["a_mode_tilde"].unsqueeze(2)
                q_ablated = (v + a_b + a_m).view(180).cpu().numpy()

            act_full = int(np.argmax(q_full))
            act_ablated = int(np.argmax(q_ablated))

            if act_full != act_ablated:
                action_flips_count += 1
                scen_flips += 1

            diff = np.abs(q_full - q_ablated)
            delta_qs.append(diff)
            total_steps += 1

            next_obs, reward, term, trunc, info = env.step(act_full)
            obs = next_obs
            if term or trunc:
                break

        per_scen_flips[scen_name] = {
            "flips": scen_flips,
            "steps": 1000,
            "flip_pct": (scen_flips / 1000.0) * 100.0,
        }

    all_diffs = np.concatenate(delta_qs)  # (10000 * 180,)
    mean_abs_delta_q = float(np.mean(all_diffs))
    p95_abs_delta_q = float(np.percentile(all_diffs, 95))
    max_abs_delta_q = float(np.max(all_diffs))

    return {
        "total_flips": action_flips_count,
        "total_steps": total_steps,
        "flip_rate_pct": (action_flips_count / float(total_steps)) * 100.0,
        "mean_abs_delta_q": mean_abs_delta_q,
        "p95_abs_delta_q": p95_abs_delta_q,
        "max_abs_delta_q": max_abs_delta_q,
        "per_scenario_flips": per_scen_flips,
    }


def main():
    if not GATE27_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {GATE27_PATH}")

    logger.info("Loading G8-A Gate 27k model from %s", GATE27_PATH)
    ckpt = torch.load(GATE27_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    model = FiLMGatedFactorizedDRQN(
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        obs_dim=CANONICAL_OBS_DIM,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    device = torch.device("cpu")

    logger.info("Running Condition A (Full FiLM)...")
    res_a = evaluate_policy(model, ablate_film=False, device=device)

    logger.info("Running Condition B (FiLM Disabled: gamma=1.0, beta=0.0)...")
    res_b = evaluate_policy(model, ablate_film=True, device=device)

    logger.info("Measuring action flips and Delta Q across state trajectory...")
    flip_stats = compute_action_flips_and_delta_q(model, device=device)

    ablation_manifest = {
        "checkpoint": str(GATE27_PATH),
        "step": 27000,
        "action_flips": flip_stats["total_flips"],
        "total_decisions": flip_stats["total_steps"],
        "action_flip_rate_pct": flip_stats["flip_rate_pct"],
        "delta_q_stats": {
            "mean_abs_delta_q": flip_stats["mean_abs_delta_q"],
            "p95_abs_delta_q": flip_stats["p95_abs_delta_q"],
            "max_abs_delta_q": flip_stats["max_abs_delta_q"],
        },
        "condition_a_full_film": {
            "mean_pd": res_a["mean_pd"],
            "agile_pd": res_a["agile_pd"],
            "sparse_pd": res_a["sparse_pd"],
            "hmode": res_a["mode_entropy"],
            "short_pct": res_a["short_pct"],
            "long_pct": res_a["long_pct"],
            "normal_pct": res_a["normal_pct"],
            "q_max": res_a["q_max"],
        },
        "condition_b_ablated_film": {
            "mean_pd": res_b["mean_pd"],
            "agile_pd": res_b["agile_pd"],
            "sparse_pd": res_b["sparse_pd"],
            "hmode": res_b["mode_entropy"],
            "short_pct": res_b["short_pct"],
            "long_pct": res_b["long_pct"],
            "normal_pct": res_b["normal_pct"],
            "q_max": res_b["q_max"],
        },
        "deltas": {
            "delta_mean_pd": res_a["mean_pd"] - res_b["mean_pd"],
            "delta_agile_pd": res_a["agile_pd"] - res_b["agile_pd"],
            "delta_sparse_pd": res_a["sparse_pd"] - res_b["sparse_pd"],
            "delta_hmode": res_a["mode_entropy"] - res_b["mode_entropy"],
            "delta_short_pct": res_a["short_pct"] - res_b["short_pct"],
        },
        "per_scenario_flips": flip_stats["per_scenario_flips"],
        "scenario_pd_comparison": {
            s: {
                "pd_full": res_a["scenarios"][s]["pd"],
                "pd_ablated": res_b["scenarios"][s]["pd"],
                "delta_pd": res_a["scenarios"][s]["pd"] - res_b["scenarios"][s]["pd"],
                "flips": flip_stats["per_scenario_flips"][s]["flips"],
            }
            for s in CANONICAL_SCENARIOS
        },
    }

    out_path = REPORTS_DIR / "g8a_film_ablation_manifest.json"
    with open(out_path, "w") as f:
        json.dump(ablation_manifest, f, indent=2)

    logger.info("================================================================================")
    logger.info("PHASE G8-A POST-HOC FiLM ABLATION RESULTS (Gate 27,000):")
    logger.info("  Action Flips: %d / %d (%.3f%%)",
                flip_stats["total_flips"], flip_stats["total_steps"], flip_stats["flip_rate_pct"])
    logger.info("  Mean |Delta Q|: %.6f | 95th Percentile |Delta Q|: %.6f | Max |Delta Q|: %.6f",
                flip_stats["mean_abs_delta_q"], flip_stats["p95_abs_delta_q"], flip_stats["max_abs_delta_q"])
    logger.info("  Full FiLM Mean Pd: %.2f%% | Ablated FiLM Mean Pd: %.2f%% (Delta: %+.2f pp)",
                res_a["mean_pd"], res_b["mean_pd"], res_a["mean_pd"] - res_b["mean_pd"])
    logger.info("  Full FiLM Agile Pd: %.2f%% | Ablated FiLM Agile Pd: %.2f%% (Delta: %+.2f pp)",
                res_a["agile_pd"], res_b["agile_pd"], res_a["agile_pd"] - res_b["agile_pd"])
    logger.info("  Full FiLM Hmode: %.3f | Ablated FiLM Hmode: %.3f (Delta: %+.3f)",
                res_a["mode_entropy"], res_b["mode_entropy"], res_a["mode_entropy"] - res_b["mode_entropy"])
    logger.info("  Scenario Flips:")
    for s in CANONICAL_SCENARIOS:
        logger.info("    %s: %d flips (%.1f%%) | Pd: %.1f%% -> %.1f%%",
                    s, flip_stats["per_scenario_flips"][s]["flips"],
                    flip_stats["per_scenario_flips"][s]["flip_pct"],
                    res_a["scenarios"][s]["pd"], res_b["scenarios"][s]["pd"])
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
