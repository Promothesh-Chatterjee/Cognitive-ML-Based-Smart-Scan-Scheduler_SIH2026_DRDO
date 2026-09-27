"""Phase G8.1 Reward Attribution Diagnostic on G7-A Parent States.

Evaluates reward distributions without gradient updates across all 10 canonical validation scenarios
for the three primary modes: SHORT (0.25x), NORMAL (1.0x), and LONG (2.5x).

Measures:
- Transition class breakdown:
    1. Empty band (no active emitter in selected band)
    2. Active band / miss (active emitter present, but no pulse caught)
    3. Repeat hit (pulse caught from previously seen emitter)
    4. Novel hit (pulse caught from newly discovered emitter)
- E[r_raw | mode, class]
- E[r_shaped | mode, class] (under c_dwell=2.0 and EMA baseline centering)
- Global empirical expectations E[r_raw | mode] and E[r_shaped | mode]

Outputs structured report to reports/g8_reward_attribution_diagnostic.json.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

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
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    RF_BASE_DWELL_TIME_US,
    band_of_action,
    encode_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.training.reward_g8 import RelativeDwellShaper
from scripts.run_g7a_targeted_replay import instantiate_g5_factorized_model, GATE25_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("reward_attribution")

G7A_CKPT_PATH = repo_root / "experiments/checkpoints/g7a_targeted_replay/checkpoint_gate_26000.pt"
VAL_DIR = Path("D:/TSRD/stare/val_stare")
OUTPUT_REPORT_PATH = repo_root / "reports/g8_reward_attribution_diagnostic.json"

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

MODES_TO_EVAL = [0, 1, 2] # SHORT, NORMAL, LONG
MODE_NAMES = {0: "SHORT", 1: "NORMAL", 2: "LONG"}


def run_reward_attribution_diagnostic() -> dict[str, Any]:
    logger.info("Initializing G7-A Parent Model from %s", G7A_CKPT_PATH)
    device = torch.device("cpu")
    model = instantiate_g5_factorized_model(GATE25_PATH, seed=42).to(device)
    ckpt = torch.load(G7A_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Structure to hold transitions per mode and per class
    # Classes: "empty", "active_miss", "repeat_hit", "novel_hit"
    classes = ["empty", "active_miss", "repeat_hit", "novel_hit"]

    results = {
        m_idx: {
            cls_name: {
                "count": 0,
                "raw_rewards": [],
                "shaped_rewards": [],
            }
            for cls_name in classes
        }
        for m_idx in MODES_TO_EVAL
    }

    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_OBS_DIM,
        "semantic_memory_path": ":memory:",
        "max_steps_per_episode": 1000,
        "reward": {"version": "v2"},
    }

    total_scenarios_evaluated = 0

    for m_idx in MODES_TO_EVAL:
        m_name = MODE_NAMES[m_idx]
        shaper = RelativeDwellShaper(ema_alpha=0.05, c_dwell=2.0)
        logger.info("Evaluating forced mode %s (idx=%d, mult=%.2fx)...", m_name, m_idx, DEFAULT_DWELL_MULTIPLIERS[m_idx])

        for scen_name in CANONICAL_SCENARIOS:
            h5_path = VAL_DIR / f"{scen_name}.h5"
            records = load_h5_records(h5_path, chunk_mode="first")
            env = CognitiveRFScanEnv(env_cfg, records=records, seed=42)
            obs, _ = env.reset(seed=42)
            hidden = model.init_hidden(1, device)

            for step in range(1000):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    q_flat, aux, hidden = model(obs_t, hidden)
                    q_row = q_flat[0, 0].cpu().numpy()

                # Policy chooses band b greedily
                act_greedy = int(np.argmax(q_row))
                b_greedy = band_of_action(act_greedy, CANONICAL_N_MODES)

                # Counterfactually execute mode m_idx on the chosen band
                act_forced = encode_action(b_greedy, m_idx, CANONICAL_N_MODES)
                dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m_idx]
                dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult

                next_obs, reward, term, trunc, info = env.step(act_forced)

                hit = bool(info.get("hit", False))
                is_novel = bool(info.get("novel_emitter", False))
                is_band_active = bool(info.get("selected_active", False) or hit)

                if not is_band_active:
                    trans_class = "empty"
                elif not hit:
                    trans_class = "active_miss"
                elif is_novel:
                    trans_class = "novel_hit"
                else:
                    trans_class = "repeat_hit"

                r_raw = float(reward)
                r_shaped = shaper.shape(r_raw, mode_chosen=m_idx, dwell_us=dwell_us)

                bucket = results[m_idx][trans_class]
                bucket["count"] += 1
                bucket["raw_rewards"].append(r_raw)
                bucket["shaped_rewards"].append(r_shaped)

                obs = next_obs
                if term or trunc:
                    break

        total_scenarios_evaluated += 1

    # Aggregate statistics
    summary_table = {}
    for m_idx in MODES_TO_EVAL:
        m_name = MODE_NAMES[m_idx]
        m_data = results[m_idx]
        total_steps_m = sum(m_data[c]["count"] for c in classes)

        all_raw = []
        all_shaped = []
        class_summary = {}

        for c in classes:
            cnt = m_data[c]["count"]
            pct = (cnt / max(1, total_steps_m)) * 100.0
            raw_arr = np.array(m_data[c]["raw_rewards"])
            shaped_arr = np.array(m_data[c]["shaped_rewards"])
            all_raw.extend(m_data[c]["raw_rewards"])
            all_shaped.extend(m_data[c]["shaped_rewards"])

            class_summary[c] = {
                "count": cnt,
                "fraction_pct": pct,
                "e_r_raw": float(np.mean(raw_arr)) if cnt > 0 else 0.0,
                "e_r_shaped": float(np.mean(shaped_arr)) if cnt > 0 else 0.0,
                "std_r_raw": float(np.std(raw_arr)) if cnt > 0 else 0.0,
                "std_r_shaped": float(np.std(shaped_arr)) if cnt > 0 else 0.0,
            }

        summary_table[m_name] = {
            "mode_index": m_idx,
            "multiplier": DEFAULT_DWELL_MULTIPLIERS[m_idx],
            "dwell_us": RF_BASE_DWELL_TIME_US * DEFAULT_DWELL_MULTIPLIERS[m_idx],
            "total_steps": total_steps_m,
            "overall_e_r_raw": float(np.mean(all_raw)),
            "overall_e_r_shaped": float(np.mean(all_shaped)),
            "classes": class_summary,
        }

    # Comparative Deltas (SHORT vs LONG, SHORT vs NORMAL)
    short_classes = summary_table["SHORT"]["classes"]
    long_classes = summary_table["LONG"]["classes"]
    normal_classes = summary_table["NORMAL"]["classes"]

    deltas_short_vs_long = {
        c: {
            "raw_delta": short_classes[c]["e_r_raw"] - long_classes[c]["e_r_raw"],
            "shaped_delta": short_classes[c]["e_r_shaped"] - long_classes[c]["e_r_shaped"],
        }
        for c in classes
    }

    report = {
        "title": "Phase G8.1 Reward Attribution Diagnostic on G7-A Parent States",
        "parent_checkpoint": str(G7A_CKPT_PATH),
        "validation_scenarios": CANONICAL_SCENARIOS,
        "summary": summary_table,
        "deltas_short_vs_long": deltas_short_vs_long,
        "overall_raw_delta_short_vs_long": summary_table["SHORT"]["overall_e_r_raw"] - summary_table["LONG"]["overall_e_r_raw"],
        "overall_shaped_delta_short_vs_long": summary_table["SHORT"]["overall_e_r_shaped"] - summary_table["LONG"]["overall_e_r_shaped"],
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Reward Attribution Diagnostic complete. Report written to %s", OUTPUT_REPORT_PATH)
    return report


if __name__ == "__main__":
    run_reward_attribution_diagnostic()
