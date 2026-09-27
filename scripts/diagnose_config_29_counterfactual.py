"""Phase G8.2R2 config_29 Forensic Diagnostic & Forced-Mode Counterfactual.

Investigates why config_29 remains blacked out (Pd = 0.0%) across all R2 arms
while config_241, config_42, and config_143 recover.

Evaluates:
1. Natural greedy policy forensics (Parent vs Arm A2 vs Arm B2 vs Arm C2):
   - Selected band distribution & top bands
   - Same-band streak & revisit intervals
   - Dwell duration breakdown & H_tau
   - Value statistics (Q_margin, Q_mean, Q_std)
   - FiLM delta Q and flip rate on config_29
2. Forced-mode counterfactual:
   - Evaluates each checkpoint under forced modes m in {SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE}
     while holding learned band selection fixed: action = b_learned * 5 + m.
   - Decisively separates:
     * Case A: All modes give ~0% -> Band selection / timing / representation issue
     * Case B: NORMAL/LONG gives substantial Pd -> Mode selection failure
     * Case C: LONG works, SHORT/NORMAL fail -> Dwell integration / objective failure
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
from scipy.stats import entropy
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.contracts import (
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
logger = logging.getLogger("diagnose_config_29")

CONFIG_29_PATH = Path("D:/TSRD/stare/val_stare/config_29.h5")
if not CONFIG_29_PATH.exists():
    CONFIG_29_PATH = Path("D:/TSRD/scan/train_scan/config_29.h5")

CHECKPOINTS = {
    "Parent (Step 26k)": repo_root / "experiments/checkpoints/g8_2/parent_film_gate_26000.pt",
    "Arm A2 (Step 26.25k)": repo_root / "experiments/checkpoints/g8_2r2_arm_a/checkpoint_gate_26250.pt",
    "Arm B2 (Step 26.25k)": repo_root / "experiments/checkpoints/g8_2r2_arm_b/checkpoint_gate_26250.pt",
    "Arm C2 (Step 26.25k)": repo_root / "experiments/checkpoints/g8_2r2_arm_c/checkpoint_gate_26250.pt",
}

ENV_CFG = {
    "n_bands": CANONICAL_N_BANDS,
    "n_modes": CANONICAL_N_MODES,
    "obs_dim": CANONICAL_OBS_DIM,
    "semantic_memory_path": ":memory:",
    "max_steps_per_episode": 1000,
    "reward": {"version": "v2"},
}


def compute_htau(mode_counts: np.ndarray) -> tuple[float, dict[float, float]]:
    n_total = max(1, int(np.sum(mode_counts)))
    p_025 = float(mode_counts[0]) / n_total
    p_100 = float(mode_counts[1] + mode_counts[3] + mode_counts[4]) / n_total
    p_250 = float(mode_counts[2]) / n_total
    probs = np.array([p_025, p_100, p_250])
    nz = probs[probs > 1e-12]
    h_val = float(-np.sum(nz * np.log(nz))) if len(nz) > 0 else 0.0
    return h_val, {0.25: p_025, 1.0: p_100, 2.5: p_250}


def load_model(ckpt_path: Path, device: torch.device) -> FiLMGatedFactorizedDRQN:
    model = FiLMGatedFactorizedDRQN(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = payload.get("state_dict", payload.get("online_drqn", payload))
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def run_natural_eval(model: FiLMGatedFactorizedDRQN, records: list, device: torch.device) -> dict[str, Any]:
    env = CognitiveRFScanEnv(ENV_CFG, records=records, seed=42)
    obs, _ = env.reset(seed=42)
    hidden = model.init_hidden(1, device)

    actions, bands, modes = [], [], []
    q_values, film_delta_qs = [], []
    film_flips = 0
    hits = 0
    first_hit_step = None
    streak_lens = []
    curr_streak = 1
    last_band = None
    band_last_seen = {}
    revisit_intervals = []

    for step in range(1000):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_flat, aux, hidden = model(obs_t, hidden)
            v = aux["v"].unsqueeze(-1)
            a_b = aux["a_band_tilde"].unsqueeze(-1)
            a_m = aux["a_mode_tilde"].unsqueeze(2)
            q_abl = (v + a_b + a_m).view(1, 1, 180)
            q_row = q_flat[0, 0].cpu().numpy()
            q_abl_row = q_abl[0, 0].cpu().numpy()

        act_film = int(np.argmax(q_row))
        act_abl = int(np.argmax(q_abl_row))
        if act_film != act_abl:
            film_flips += 1
        dq = float(np.abs(q_row - q_abl_row).mean())
        film_delta_qs.append(dq)

        act = act_film
        b = band_of_action(act, CANONICAL_N_MODES)
        m = mode_of_action(act, CANONICAL_N_MODES)

        actions.append(act)
        bands.append(b)
        modes.append(m)
        q_values.append(q_row)

        if last_band is not None:
            if b == last_band:
                curr_streak += 1
            else:
                streak_lens.append(curr_streak)
                curr_streak = 1
        last_band = b

        if b in band_last_seen:
            revisit_intervals.append(step - band_last_seen[b])
        band_last_seen[b] = step

        next_obs, reward, term, trunc, info = env.step(act)
        hit = bool(info.get("hit", False))
        if hit:
            hits += 1
            if first_hit_step is None:
                first_hit_step = step

        obs = next_obs
        if term or trunc:
            break

    if curr_streak > 0:
        streak_lens.append(curr_streak)

    fom = env.get_fom()
    pd_val = float(fom.get("Pd", float(hits / max(1, len(actions))))) * 100.0

    b_counts = np.bincount(bands, minlength=36)
    m_counts = np.bincount(modes, minlength=5)
    m_probs = m_counts / max(1, len(modes))
    h_mode = float(entropy(m_probs + 1e-12))
    h_tau, tau_dist = compute_htau(m_counts)

    q_arr = np.array(q_values)
    top2_margins = np.sort(q_arr, axis=-1)[:, -1] - np.sort(q_arr, axis=-1)[:, -2]

    # Find the bands where emitters actually reside in config_29
    # Analyze records in env
    emitter_bands = set()
    if hasattr(env, "scenario") and env.scenario is not None:
        for em in getattr(env.scenario, "emitters", []):
            if hasattr(em, "band_idx"):
                emitter_bands.add(int(em.band_idx))
            elif hasattr(em, "cf_hz"):
                b_idx = int(min(35, max(0, em.cf_hz / (500.0 * 1e6))))
                emitter_bands.add(b_idx)

    # Calculate steps on emitter bands
    steps_on_emitter_bands = sum(b_counts[eb] for eb in emitter_bands if eb < len(b_counts))

    return {
        "pd": pd_val,
        "hits": hits,
        "first_hit_step": first_hit_step,
        "steps": len(actions),
        "h_mode": h_mode,
        "h_tau": h_tau,
        "tau_dist": tau_dist,
        "mode_counts": m_counts.tolist(),
        "top_5_bands": [int(x) for x in np.argsort(-b_counts)[:5]],
        "top_5_band_counts": [int(b_counts[x]) for x in np.argsort(-b_counts)[:5]],
        "distinct_bands": int(np.count_nonzero(b_counts)),
        "mean_streak": float(np.mean(streak_lens)) if streak_lens else 0.0,
        "max_streak": int(max(streak_lens)) if streak_lens else 0,
        "mean_revisit_interval": float(np.mean(revisit_intervals)) if revisit_intervals else 0.0,
        "q_mean": float(q_arr.mean()),
        "q_std": float(q_arr.std()),
        "greedy_margin": float(top2_margins.mean()),
        "film_flip_pct": (film_flips / max(1, len(actions))) * 100.0,
        "film_delta_q": float(np.mean(film_delta_qs)) if film_delta_qs else 0.0,
        "emitter_bands": sorted(list(emitter_bands)),
        "steps_on_emitter_bands": int(steps_on_emitter_bands),
    }


def run_forced_mode_counterfactual(
    model: FiLMGatedFactorizedDRQN,
    records: list,
    device: torch.device,
) -> dict[str, float]:
    """Force action = b_learned * 5 + m for each mode m in {0, 1, 2, 3, 4}."""
    cf_results = {}

    for m_target in range(5):
        m_name = DWELL_MODES[m_target]
        env = CognitiveRFScanEnv(ENV_CFG, records=records, seed=42)
        obs, _ = env.reset(seed=42)
        hidden = model.init_hidden(1, device)
        hits = 0
        total_steps = 0

        for _ in range(1000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_flat, aux, hidden = model(obs_t, hidden)
                q_row = q_flat[0, 0].cpu().numpy()

            b_learned = band_of_action(int(np.argmax(q_row)), CANONICAL_N_MODES)
            forced_act = int(b_learned * CANONICAL_N_MODES + m_target)

            next_obs, reward, term, trunc, info = env.step(forced_act)
            if bool(info.get("hit", False)):
                hits += 1
            total_steps += 1

            obs = next_obs
            if term or trunc:
                break

        fom = env.get_fom()
        pd_val = float(fom.get("Pd", float(hits / max(1, total_steps)))) * 100.0
        cf_results[m_name] = pd_val

    return cf_results


def main():
    logger.info("==========================================================================")
    logger.info("   PHASE G8.2R2 CONFIG_29 FORENSIC DIAGNOSTIC & COUNTERFACTUAL SUITE   ")
    logger.info("==========================================================================")
    assert CONFIG_29_PATH.exists(), f"config_29 H5 file not found: {CONFIG_29_PATH}"
    logger.info("Using config_29 dataset: %s", CONFIG_29_PATH)

    records = load_h5_records(CONFIG_29_PATH, chunk_mode="first")
    device = torch.device("cpu")

    all_diagnostics = {}

    for name, ckpt_p in CHECKPOINTS.items():
        if not ckpt_p.exists():
            logger.warning("Checkpoint missing: %s", ckpt_p)
            continue
        logger.info("\n>>> Analyzing %s ...", name)
        model = load_model(ckpt_p, device)

        # 1. Natural greedy evaluation
        nat = run_natural_eval(model, records, device)
        logger.info("  [Natural Greedy] Pd=%.2f%% (Hits=%d/%d, FirstHit=%s)", nat["pd"], nat["hits"], nat["steps"], nat["first_hit_step"])
        logger.info("  [Modes] Hmode=%.3f | Htau=%.3f | SHORT=%.1f%%, NORMAL=%.1f%%, LONG=%.1f%%, REVISIT=%.1f%%, PREEMPT=%.1f%%",
                    nat["h_mode"], nat["h_tau"],
                    nat["mode_counts"][0]/10.0, nat["mode_counts"][1]/10.0, nat["mode_counts"][2]/10.0,
                    nat["mode_counts"][3]/10.0, nat["mode_counts"][4]/10.0)
        logger.info("  [Dwell Classes tau] p(0.25)=%.3f, p(1.0)=%.3f, p(2.5)=%.3f",
                    nat["tau_dist"][0.25], nat["tau_dist"][1.0], nat["tau_dist"][2.5])
        logger.info("  [Bands] Distinct=%d, Top Bands=%s (counts=%s), Emitter Bands=%s, Steps on Emitter Bands=%d",
                    nat["distinct_bands"], nat["top_5_bands"], nat["top_5_band_counts"], nat["emitter_bands"], nat["steps_on_emitter_bands"])
        logger.info("  [Dynamics] Mean Streak=%.2f, Max Streak=%d, Mean Revisit Interval=%.2f",
                    nat["mean_streak"], nat["max_streak"], nat["mean_revisit_interval"])
        logger.info("  [FiLM] Flip Rate=%.2f%%, Mean Delta Q=%.5f", nat["film_flip_pct"], nat["film_delta_q"])

        # 2. Forced-mode counterfactual evaluation
        logger.info("  [Running Forced-Mode Counterfactual]...")
        cf = run_forced_mode_counterfactual(model, records, device)
        logger.info("  [Forced Modes Pd] SHORT: %.2f%% | NORMAL: %.2f%% | LONG: %.2f%% | REVISIT: %.2f%% | PREEMPTIVE: %.2f%%",
                    cf["SHORT_DWELL"], cf["NORMAL_DWELL"], cf["LONG_DWELL"], cf["REVISIT"], cf["PREEMPTIVE_INTERCEPT"])

        all_diagnostics[name] = {"natural": nat, "counterfactual": cf}

    out_file = repo_root / "reports/g8_2r2_config_29_diagnostic.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    logger.info("\nSaved complete config_29 diagnostic to %s", out_file)


if __name__ == "__main__":
    main()
