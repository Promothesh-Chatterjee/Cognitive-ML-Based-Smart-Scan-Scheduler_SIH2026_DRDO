"""Task 10: Quantitative Evaluation of Phase G8-A Gate 27,000 Checkpoint.

Evaluates the FiLM-Gated Factorized DRQN model against all 10 canonical validation scenarios
from D:/TSRD/stare/val_stare under deterministic greedy inference.
Evaluates against the 7 project qualification criteria.
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
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_g8a_gate27k")

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


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_consecutive_runs(sequence: list[int]):
    if not sequence:
        return 0.0, 0, 0.0
    runs = []
    current_val = sequence[0]
    current_len = 1
    repeats = 0
    for i in range(1, len(sequence)):
        if sequence[i] == current_val:
            current_len += 1
            repeats += 1
        else:
            runs.append(current_len)
            current_val = sequence[i]
            current_len = 1
    runs.append(current_len)
    mean_run = float(np.mean(runs))
    max_run = int(np.max(runs))
    repeat_frac = float(repeats / float(len(sequence) - 1)) if len(sequence) > 1 else 0.0
    return mean_run, max_run, repeat_frac


def main():
    if not GATE27_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {GATE27_PATH}")

    sha27 = compute_sha256(GATE27_PATH)
    logger.info("Evaluating G8-A Gate 27,000 checkpoint (SHA-256: %s)", sha27)

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

    scen_records = {}
    all_actions = []
    all_bands = []
    all_modes = []
    all_q_values = []
    all_q_margins = []
    all_film_gammas = []
    all_film_betas = []
    total_hits = 0
    total_novel_hits = 0
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
        scen_novel_hits = 0
        scen_dwell_us = 0.0
        first_hit_latency = None

        for step in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                q_row = q_flat[0, 0].cpu().numpy()
                gamma_row = aux["film_gamma"][0, 0].cpu().numpy()
                beta_row = aux["film_beta"][0, 0].cpu().numpy()

            act = int(np.argmax(q_row))
            sorted_q = np.sort(q_row)
            q_margin = float(sorted_q[-1] - sorted_q[-2])
            all_q_margins.append(q_margin)
            all_q_values.append(q_row)
            all_film_gammas.append(gamma_row)
            all_film_betas.append(beta_row)

            b = band_of_action(act, CANONICAL_N_MODES)
            m = mode_of_action(act, CANONICAL_N_MODES)
            dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m]
            dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

            scen_acts.append(act)
            scen_bnds.append(b)
            scen_mods.append(m)
            scen_dwell_us += dwell_us

            next_obs, reward, term, trunc, info = env.step(act)
            hit = bool(info.get("hit", False))
            is_novel = bool(info.get("novel_emitter", False))

            if hit:
                scen_hits += 1
                if is_novel:
                    scen_novel_hits += 1
                if first_hit_latency is None:
                    first_hit_latency = float(scen_dwell_us)

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        dwell_ms = scen_dwell_us / 1000.0
        mean_run, max_run, repeat_frac = compute_consecutive_runs(scen_bnds)
        m_counts = {DWELL_MODES[i]: int(np.sum(np.array(scen_mods) == i)) for i in range(CANONICAL_N_MODES)}
        short_fraction = float(m_counts["SHORT_DWELL"] / float(len(scen_acts)))

        scen_records[scen_name] = {
            "steps": len(scen_acts),
            "hits": scen_hits,
            "novel_hits": scen_novel_hits,
            "dwell_ms": dwell_ms,
            "gross_hits_per_ms": scen_hits / dwell_ms if dwell_ms > 0 else 0.0,
            "novel_hits_per_ms": scen_novel_hits / dwell_ms if dwell_ms > 0 else 0.0,
            "ir_decision": (scen_hits / float(len(scen_acts))) * 100.0,
            "pd": float(fom.get("Pd", fom.get("pd", 0.0))) * 100.0,
            "pfa": float(fom.get("Pfa", fom.get("pfa", 0.0))),
            "first_hit_latency_ms": (first_hit_latency or scen_dwell_us) / 1000.0,
            "mean_consecutive_band_run": mean_run,
            "max_consecutive_band_run": max_run,
            "repeated_band_fraction": repeat_frac,
            "modes": m_counts,
            "short_fraction_pct": short_fraction * 100.0,
        }
        all_actions.extend(scen_acts)
        all_bands.extend(scen_bnds)
        all_modes.extend(scen_mods)
        total_hits += scen_hits
        total_novel_hits += scen_novel_hits
        total_dwell_us += scen_dwell_us

    n_tot_steps = len(all_actions)
    tot_dwell_ms = total_dwell_us / 1000.0
    all_q_arr = np.array(all_q_values)
    all_gammas_arr = np.array(all_film_gammas)
    all_betas_arr = np.array(all_film_betas)

    mode_counts = np.bincount(all_modes, minlength=CANONICAL_N_MODES)
    mode_probs = mode_counts / float(n_tot_steps)
    mode_ent = float(entropy(mode_probs + 1e-12, base=np.e))

    sparse_pds = [scen_records[s]["pd"] for s in SPARSE_SCENARIOS]
    agile_pds = [scen_records[s]["pd"] for s in AGILE_SCENARIOS]

    mean_run_all, max_run_all, repeat_frac_all = compute_consecutive_runs(all_bands)

    mean_pd = float(np.mean([s["pd"] for s in scen_records.values()]))
    agile_pd = float(np.mean(agile_pds))
    sparse_pd = float(np.mean(sparse_pds))
    short_pct = float(mode_probs[0] * 100.0)
    q_max = float(np.max(all_q_arr))

    # Evaluate against 7 criteria
    c1_mean_pd = bool(mean_pd >= 78.0)
    c2_hmode = bool(mode_ent >= 0.40)
    c3_cfg29 = bool(scen_records["config_29"]["pd"] > 0.0)
    c4_cfg119 = bool(scen_records["config_119"]["pd"] > 0.0)
    c5_cfg241 = bool(scen_records["config_241"]["pd"] > 0.0)
    c6_short_range = bool(15.0 <= short_pct <= 80.0)
    c7_qmax = bool(q_max <= 35.0)

    all_criteria_met = all([c1_mean_pd, c2_hmode, c3_cfg29, c4_cfg119, c5_cfg241, c6_short_range, c7_qmax])

    results = {
        "checkpoint_sha256": sha27,
        "phase": "G8-A",
        "step": 27000,
        "mode_entropy": mode_ent,
        "mode_distribution_pct": {
            DWELL_MODES[i]: float(mode_probs[i] * 100.0) for i in range(CANONICAL_N_MODES)
        },
        "short_fraction_pct": short_pct,
        "q_max": q_max,
        "q_min": float(np.min(all_q_arr)),
        "q_mean": float(np.mean(all_q_arr)),
        "q_std": float(np.std(all_q_arr)),
        "mean_pd": mean_pd,
        "sparse_pd": sparse_pd,
        "agile_pd": agile_pd,
        "repeated_band_dwell_fraction": repeat_frac_all,
        "gross_hits_per_ms": total_hits / tot_dwell_ms if tot_dwell_ms > 0 else 0.0,
        "novel_hits_per_ms": total_novel_hits / tot_dwell_ms if tot_dwell_ms > 0 else 0.0,
        "film_gate_stats": {
            "gamma_mean": float(np.mean(all_gammas_arr)),
            "gamma_std": float(np.std(all_gammas_arr)),
            "gamma_min": float(np.min(all_gammas_arr)),
            "gamma_max": float(np.max(all_gammas_arr)),
            "beta_mean": float(np.mean(all_betas_arr)),
            "beta_std": float(np.std(all_betas_arr)),
            "beta_min": float(np.min(all_betas_arr)),
            "beta_max": float(np.max(all_betas_arr)),
        },
        "criteria": {
            "c1_mean_pd_ge_78": {"value": mean_pd, "passed": c1_mean_pd},
            "c2_hmode_ge_040": {"value": mode_ent, "passed": c2_hmode},
            "c3_config29_pd_gt_0": {"value": scen_records["config_29"]["pd"], "passed": c3_cfg29},
            "c4_config119_pd_gt_0": {"value": scen_records["config_119"]["pd"], "passed": c4_cfg119},
            "c5_config241_pd_gt_0": {"value": scen_records["config_241"]["pd"], "passed": c5_cfg241},
            "c6_short_range_15_to_80": {"value": short_pct, "passed": c6_short_range},
            "c7_qmax_le_35": {"value": q_max, "passed": c7_qmax},
        },
        "all_criteria_met": all_criteria_met,
        "verdict": "QUALIFIED" if all_criteria_met else "NON_QUALIFIED",
        "scenarios": scen_records,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "g8a_gate27k_diagnostic_manifest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("================================================================================")
    logger.info("PHASE G8-A GATE 27,000 DIAGNOSTIC RESULTS:")
    logger.info("  Verdict: %s", results["verdict"])
    logger.info("  Mode Entropy (Hmode): %.3f (Target >= 0.40)", mode_ent)
    logger.info("  SHORT Dwell: %.2f%% | NORMAL: %.2f%% | LONG: %.2f%% | REVISIT: %.2f%% | PREEMPTIVE: %.2f%%",
                results["mode_distribution_pct"]["SHORT_DWELL"],
                results["mode_distribution_pct"]["NORMAL_DWELL"],
                results["mode_distribution_pct"]["LONG_DWELL"],
                results["mode_distribution_pct"]["REVISIT"],
                results["mode_distribution_pct"]["PREEMPTIVE_INTERCEPT"])
    logger.info("  Mean Pd: %.2f%% (Target >= 78%%) | Agile Pd: %.2f%% | Sparse Pd: %.2f%%",
                mean_pd, agile_pd, sparse_pd)
    logger.info("  config_29 Pd: %.2f%% | config_119 Pd: %.2f%% | config_241 Pd: %.2f%% | config_42 Pd: %.2f%%",
                scen_records["config_29"]["pd"], scen_records["config_119"]["pd"],
                scen_records["config_241"]["pd"], scen_records["config_42"]["pd"])
    logger.info("  Qmax: %.2f (Target <= 35.0)", q_max)
    logger.info("  FiLM Gamma Mean: %.4f (std: %.4f) | Beta Mean: %.4f (std: %.4f)",
                results["film_gate_stats"]["gamma_mean"], results["film_gate_stats"]["gamma_std"],
                results["film_gate_stats"]["beta_mean"], results["film_gate_stats"]["beta_std"])
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
