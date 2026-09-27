"""Phase G8.1 Multi-Scenario Evaluation and FiLM Diagnostic Suite.

Evaluates any G8.1 checkpoint against the 10 canonical evaluation scenarios:
- Agile: config_29, config_119, config_241, config_42
- Sparse: config_143
- Dense: config_194, config_195, config_64
- Mixed: config_117, config_96

Metrics computed:
1. Mean Pd (Detection probability) overall, agile-only, dense-only
2. Blackout scenario list (Pd = 0.0%)
3. Mode distribution and Shannon entropy H_mode
4. Q-value statistics (mean, max, std)
5. Q-gap normalized FiLM modulation metric:
   rho_FiLM = mean_{s, b} [ |(gamma(s,b,m) - 1.0) * A_mode(m) + beta(s,b,m)| / Delta Q_subopt(s, b) ]
6. FiLM action flip rate (% decisions changed when FiLM gate is zeroed out)
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("eval_g8_1_checkpoint")

CANONICAL_SCENARIOS = [
    {"id": "config_29", "category": "agile", "path": Path("D:/TSRD/scan/train_scan/config_29.h5")},
    {"id": "config_119", "category": "agile", "path": Path("D:/TSRD/scan/train_scan/config_119.h5")},
    {"id": "config_241", "category": "agile", "path": Path("D:/TSRD/scan/train_scan/config_241.h5")},
    {"id": "config_42", "category": "agile", "path": Path("D:/TSRD/stare/val_stare/config_42.h5")},
    {"id": "config_143", "category": "sparse", "path": Path("D:/TSRD/stare/train_stare/config_143.h5")},
    {"id": "config_194", "category": "dense", "path": Path("D:/TSRD/stare/train_stare/config_194.h5")},
    {"id": "config_195", "category": "dense", "path": Path("D:/TSRD/stare/train_stare/config_195.h5")},
    {"id": "config_64", "category": "dense", "path": Path("D:/TSRD/stare/train_stare/config_64.h5")},
    {"id": "config_117", "category": "mixed", "path": Path("D:/TSRD/stare/train_stare/config_117.h5")},
    {"id": "config_96", "category": "mixed", "path": Path("D:/TSRD/stare/train_stare/config_96.h5")},
]


def evaluate_checkpoint(ckpt_path: Path, gate_name: str) -> dict[str, Any]:
    logger.info("Evaluating checkpoint: %s (Gate: %s)", ckpt_path, gate_name)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    device = torch.device("cpu")
    # Load model
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    if "film_proj.0.weight" in state_dict:
        model = FiLMGatedFactorizedDRQN(
            obs_dim=CANONICAL_OBS_DIM,
            n_bands=CANONICAL_N_BANDS,
            n_modes=CANONICAL_N_MODES,
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        model, _ = FiLMGatedFactorizedDRQN.from_g7a_checkpoint(ckpt_path, seed=42)
    model.eval()

    # Also build ablated model (condition B: gamma=1.0, beta=0.0)
    ablated_model = copy.deepcopy(model)
    nn = torch.nn
    nn.init.zeros_(ablated_model.film_proj[0].weight)
    nn.init.zeros_(ablated_model.film_proj[0].bias)
    nn.init.zeros_(ablated_model.film_proj[2].weight)
    nn.init.zeros_(ablated_model.film_proj[2].bias)
    ablated_model.eval()

    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_OBS_DIM,
        "semantic_memory_path": ":memory:",
        "max_steps_per_episode": 1000,
        "reward": {"version": "v2"},
    }

    scenario_results = []
    total_flips = 0
    total_eval_steps = 0
    all_film_rhos = []
    mode_counts_overall = np.zeros(5, dtype=np.int64)

    for spec in CANONICAL_SCENARIOS:
        scen_id = spec["id"]
        category = spec["category"]
        h5_path = spec["path"]

        if not h5_path.exists():
            # Fallback search in val_stare or scan
            cand = Path("D:/TSRD/stare/val_stare") / f"{scen_id}.h5"
            if cand.exists():
                h5_path = cand
            else:
                cand = Path("D:/TSRD/scan/train_scan") / f"{scen_id}.h5"
                if cand.exists():
                    h5_path = cand

        recs = load_h5_records(h5_path)
        env = CognitiveRFScanEnv(env_cfg, records=recs, seed=42)
        obs, _ = env.reset()

        h_full = None
        h_abl = None
        hits = 0
        steps = 0
        scen_flips = 0
        scen_mode_counts = np.zeros(5, dtype=np.int64)
        scen_q_vals = []
        scen_film_rhos = []

        for step in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            with torch.inference_mode():
                q_full, aux_full, h_full = model(obs_t, h_full)
                q_abl, aux_abl, h_abl = ablated_model(obs_t, h_abl)

            q_full_np = q_full[0, 0].cpu().numpy()
            q_abl_np = q_abl[0, 0].cpu().numpy()

            act_full = int(np.argmax(q_full_np))
            act_abl = int(np.argmax(q_abl_np))

            if act_full != act_abl:
                scen_flips += 1
                total_flips += 1

            m = mode_of_action(act_full, CANONICAL_N_MODES)
            b = band_of_action(act_full, CANONICAL_N_MODES)
            scen_mode_counts[m] += 1
            mode_counts_overall[m] += 1
            scen_q_vals.append(float(q_full_np[act_full]))

            # Compute Q-gap normalized FiLM modulation metric rho_FiLM:
            # Band-slice Q values: (5,)
            q_band = q_full_np[b * 5 : (b + 1) * 5]
            sorted_q = np.sort(q_band)[::-1]
            q_gap = max(1e-4, float(sorted_q[0] - sorted_q[1]))

            # Modulation magnitude: |Coupled - Additive|
            # aux_full["film_gamma"] shape (1, 1, 36, 5), beta shape (1, 1, 36, 5)
            gamma_val = aux_full["film_gamma"][0, 0, b, m].item()
            beta_val = aux_full["film_beta"][0, 0, b, m].item()
            a_mode_val = aux_full["a_mode_tilde"][0, 0, m].item()
            delta_film = abs((gamma_val - 1.0) * a_mode_val + beta_val)
            rho_film = delta_film / q_gap
            scen_film_rhos.append(rho_film)
            all_film_rhos.append(rho_film)

            next_obs, reward, term, trunc, info = env.step(act_full)
            if info.get("hit", False):
                hits += 1
            steps += 1
            obs = next_obs
            if term or trunc:
                break

        total_eval_steps += steps
        pd = (hits / max(1, steps)) * 100.0
        mode_probs = scen_mode_counts / max(1, steps)
        h_mode = float(entropy(mode_probs + 1e-12, base=np.e))

        scenario_results.append({
            "id": scen_id,
            "category": category,
            "pd": pd,
            "steps": steps,
            "hits": hits,
            "action_flips": scen_flips,
            "flip_pct": (scen_flips / max(1, steps)) * 100.0,
            "mean_q": float(np.mean(scen_q_vals)),
            "max_q": float(np.max(scen_q_vals)),
            "mean_rho_film": float(np.mean(scen_film_rhos)),
            "h_mode": h_mode,
            "short_pct": float(mode_probs[0] * 100.0),
            "revisit_pct": float(mode_probs[3] * 100.0),
            "long_pct": float(mode_probs[2] * 100.0),
        })

    # Aggregates
    all_pds = [r["pd"] for r in scenario_results]
    agile_pds = [r["pd"] for r in scenario_results if r["category"] == "agile"]
    dense_pds = [r["pd"] for r in scenario_results if r["category"] == "dense"]
    blackouts = [r["id"] for r in scenario_results if r["pd"] == 0.0]

    mean_pd = float(np.mean(all_pds))
    agile_pd = float(np.mean(agile_pds))
    dense_pd = float(np.mean(dense_pds))

    overall_mode_probs = mode_counts_overall / max(1, total_eval_steps)
    overall_h_mode = float(entropy(overall_mode_probs + 1e-12, base=np.e))
    overall_flip_pct = (total_flips / max(1, total_eval_steps)) * 100.0

    verdict = "PASS" if (mean_pd >= 78.0 and agile_pd >= 70.0 and len(blackouts) == 0 and overall_h_mode >= 0.80) else "NON_QUALIFIED"

    summary = {
        "gate": gate_name,
        "checkpoint": str(ckpt_path),
        "mean_pd": mean_pd,
        "agile_mean_pd": agile_pd,
        "dense_mean_pd": dense_pd,
        "blackout_count": len(blackouts),
        "blackouts": blackouts,
        "h_mode": overall_h_mode,
        "short_pct": float(overall_mode_probs[0] * 100.0),
        "normal_pct": float(overall_mode_probs[1] * 100.0),
        "long_pct": float(overall_mode_probs[2] * 100.0),
        "revisit_pct": float(overall_mode_probs[3] * 100.0),
        "preemptive_pct": float(overall_mode_probs[4] * 100.0),
        "film_action_flips": total_flips,
        "film_flip_pct": overall_flip_pct,
        "mean_rho_film": float(np.mean(all_film_rhos)),
        "p95_rho_film": float(np.percentile(all_film_rhos, 95)),
        "verdict": verdict,
        "scenarios": scenario_results,
    }

    report_path = repo_root / f"reports/g8_1_gate_{gate_name}_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Gate %s Evaluation Summary ===", gate_name)
    logger.info("Mean Pd: %.2f%% | Agile Pd: %.2f%% | Dense Pd: %.2f%%", mean_pd, agile_pd, dense_pd)
    logger.info("Blackouts: %d %s", len(blackouts), blackouts)
    logger.info("H_mode: %.3f | SHORT: %.2f%% | REVISIT: %.2f%% | LONG: %.2f%%",
                overall_h_mode, overall_mode_probs[0]*100, overall_mode_probs[3]*100, overall_mode_probs[2]*100)
    logger.info("FiLM Flips: %d / %d (%.2f%%) | rho_FiLM: mean=%.3f, p95=%.3f",
                total_flips, total_eval_steps, overall_flip_pct, np.mean(all_film_rhos), np.percentile(all_film_rhos, 95))
    logger.info("Verdict: %s", verdict)
    logger.info("Report written to %s", report_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gate", type=str, default="26250")
    args = parser.parse_args()

    evaluate_checkpoint(Path(args.checkpoint), args.gate)
