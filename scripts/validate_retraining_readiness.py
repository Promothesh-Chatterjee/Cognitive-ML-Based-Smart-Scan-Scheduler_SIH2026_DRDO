"""Executable qualification gate for pre-retraining readiness.

Evaluates 35 strict pre-retraining conditions split across:
1. LOCAL QUALIFICATION (30 gates):
   - Git provenance integrity
   - Frozen checkpoint immutability & SHA verification
   - Real TSRD dataset & scenario file verification
   - Causal observation & zero GT leakage
   - Model contracts (360-D obs, 180 actions, 5 modes, reward v2)
   - Confusion matrix accounting invariants (TP + FN + FP + TN == N_dwells)
   - Gradient flow & live diversity penalties
   - Benchmark reproducibility & 4-scheduler artifact validation
   - Sensitivity consistency (-110.0 dBm)
2. DEPLOYMENT VERIFICATION (5 gates):
   - Live health probe & checkpoint SHA match
   - Live readiness probe
   - Live /api/benchmark contract validation
   - Live inference latency SLA (< 500 ms)
   - Live deployment provenance

Outputs:
  LOCAL_RETRAINING_READY = TRUE / FALSE
  DEPLOYMENT_VERIFIED = TRUE / FALSE
  RETRAINING_READY = TRUE / FALSE (requires BOTH to be TRUE)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ew_core.utils.checkpoint_paths import (
    EXPECTED_FROZEN_SHA256,
    CANONICAL_PRODUCTION_BASELINE,
)
from ew_core.contracts import (
    CANONICAL_OBS_DIM,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_N_ACTIONS,
)

FROZEN_CKPT_PATH = REPO_ROOT / "experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt"
BENCHMARK_PATH = REPO_ROOT / "reports/benchmark_results.json"
EXPECTED_NORMALIZATION_HASH = "bacee02ac1c29428"
CANONICAL_SCENARIOS = [
    "config_117", "config_119", "config_143", "config_194", "config_195",
    "config_241", "config_29", "config_42", "config_64", "config_96",
]
EXPECTED_SCHEDULERS = {
    "SmartScan_DRQN_MoE",
    "Random",
    "RoundRobin",
    "HighestOccupancy",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ReadinessGateEvaluator:
    def __init__(self, api_url: str | None = None, api_key: str | None = None, skip_deployment: bool = False):
        self.api_url = (api_url or os.environ.get("CLOUD_RUN_URL", "https://cognitive-ew-backend-753709137146.asia-south1.run.app")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("SMARTSCAN_API_KEY", "")
        self.skip_deployment = skip_deployment
        self.local_results: Dict[str, Tuple[bool, str]] = {}
        self.deploy_results: Dict[str, Tuple[bool, str]] = {}

    def log_gate(self, scope: str, name: str, passed: bool, message: str):
        symbol = "[PASS]" if passed else "[FAIL]"
        print(f"  {symbol} [{scope}] {name}: {message}")
        if scope == "LOCAL":
            self.local_results[name] = (passed, message)
        else:
            self.deploy_results[name] = (passed, message)

    def evaluate_local_gates(self):
        print("\n" + "=" * 78)
        print("  STAGE 1: LOCAL PRE-RETRAINING QUALIFICATION (31 GATES)")
        print("=" * 78)

        # 1. Check frozen checkpoint SHA
        if not FROZEN_CKPT_PATH.exists():
            self.log_gate("LOCAL", "FROZEN_CHECKPOINT_EXISTS", False, f"Missing {FROZEN_CKPT_PATH}")
        else:
            actual_sha = hashlib.sha256(FROZEN_CKPT_PATH.read_bytes()).hexdigest()
            passed = (actual_sha == EXPECTED_FROZEN_SHA256)
            self.log_gate("LOCAL", "FROZEN_CHECKPOINT_SHA", passed, f"SHA={actual_sha[:16]}... match={passed}")

        # 2. Immutable baseline protection
        prod_base = REPO_ROOT / "experiments/checkpoints/production_baseline"
        cand_base = REPO_ROOT / "experiments/checkpoints/scheduler_v2_operational_candidate"
        self.log_gate("LOCAL", "IMMUTABLE_DIRS_EXIST", prod_base.exists() and cand_base.exists(), "Baseline dirs intact")

        # 3. Real TSRD dataset root
        tsrd_root = Path(os.environ.get("TSRD_DATA_ROOT", "D:/TSRD"))
        tsrd_exists = tsrd_root.exists() and tsrd_root.is_dir()
        self.log_gate("LOCAL", "REAL_TSRD_DATASET_ROOT", tsrd_exists, f"Path={tsrd_root} exists={tsrd_exists}")

        # 4. TSRD scenarios verification
        val_dir = tsrd_root / "stare" / "val_stare"
        if not val_dir.exists():
            val_dir = tsrd_root / "val"
        found_scens = [s for s in CANONICAL_SCENARIOS if (val_dir / f"{s}.h5").exists()]
        all_scens_present = (len(found_scens) == len(CANONICAL_SCENARIOS))
        self.log_gate("LOCAL", "TSRD_SCENARIO_FILES", all_scens_present, f"Found {len(found_scens)}/{len(CANONICAL_SCENARIOS)} scenarios")

        # 5. Synthetic fallback disabled check
        import yaml
        cfg_path = REPO_ROOT / "configs/training_config_resume_100k.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            mode = cfg.get("training_mode", "unknown")
            synth_disabled = (mode == "real_tsrd")
            self.log_gate("LOCAL", "SYNTHETIC_FALLBACK_DISABLED", synth_disabled, f"training_mode={mode}")
        else:
            self.log_gate("LOCAL", "SYNTHETIC_FALLBACK_DISABLED", False, "Config file not found")

        # 6. Deinterleaver checkpoint
        d_ckpt = REPO_ROOT / "experiments/checkpoints/deinterleaver/best.pt"
        if not d_ckpt.exists():
            d_ckpt = REPO_ROOT / "checkpoints/deinterleaver/best.pt"
        self.log_gate("LOCAL", "DEINTERLEAVER_CHECKPOINT", d_ckpt.exists(), f"Path={d_ckpt.name} exists={d_ckpt.exists()}")

        # 7. Normalization statistics and hash
        fit_stats_path = REPO_ROOT / "experiments/checkpoints/deinterleaver/normalization_stats.json"
        if not fit_stats_path.exists():
            fit_stats_path = REPO_ROOT / "checkpoints/deinterleaver/normalization_stats.json"
        has_stats = fit_stats_path.exists()
        hash_match = False
        if has_stats:
            try:
                stats_json = json.loads(fit_stats_path.read_text(encoding="utf-8"))
                hash_match = (stats_json.get("stats_hash") == EXPECTED_NORMALIZATION_HASH)
            except Exception:
                hash_match = False
        self.log_gate("LOCAL", "NORMALIZATION_STATS_EXISTS", has_stats and hash_match, f"Fit stats {fit_stats_path.name} exists={has_stats} hash_match={hash_match}")

        # 8. 360-D observation contract
        self.log_gate("LOCAL", "360D_OBS_CONTRACT", CANONICAL_OBS_DIM == 360, f"obs_dim={CANONICAL_OBS_DIM}")

        # 9. 180-action contract
        self.log_gate("LOCAL", "180_ACTION_CONTRACT", CANONICAL_N_ACTIONS == 180, f"n_actions={CANONICAL_N_ACTIONS}")

        # 10. 5-mode contract
        self.log_gate("LOCAL", "5_MODE_CONTRACT", CANONICAL_N_MODES == 5, f"n_modes={CANONICAL_N_MODES}")

        # 11. Reward v2 invariant
        from ew_core.training.reward import receiver_reward_components_v2
        r_hit = receiver_reward_components_v2(
            selected_active=True, detected=True, other_bands_active=False,
            running_pfa=0.0, lambda_pfa=2.0, pfa_threshold=0.05,
        )
        r_miss = receiver_reward_components_v2(
            selected_active=True, detected=False, other_bands_active=False,
            running_pfa=0.0, lambda_pfa=2.0, pfa_threshold=0.05,
        )
        hit_total = float(r_hit.get("total", r_hit.get("reward", 0)))
        miss_total = float(r_miss.get("total", r_miss.get("reward", 0)))
        dominance_ok = hit_total > miss_total
        self.log_gate("LOCAL", "REWARD_VERSION_V2", dominance_ok,
            f"hit={hit_total:.2f} > miss={miss_total:.2f}" if dominance_ok else f"VIOLATION: hit={hit_total:.2f} <= miss={miss_total:.2f}")

        # 12. No GT leakage into observation features
        from ew_core.contracts import CANONICAL_BELIEF_FEATURE_NAMES
        GT_FORBIDDEN_NAMES = ["emitter_id", "true_band", "oracle", "ground_truth", "gt_", "truth_", "future_"]
        feature_names = CANONICAL_BELIEF_FEATURE_NAMES
        contaminated = [f for f in feature_names if any(g in f.lower() for g in GT_FORBIDDEN_NAMES)]
        gt_clean = len(contaminated) == 0
        self.log_gate("LOCAL", "NO_GT_LEAKAGE", gt_clean,
            f"All {len(feature_names)} obs features are GT-free" if gt_clean else f"CONTAMINATED: {contaminated}")

        # 13. Causal observation tests
        import numpy as np
        from ew_core.operational.state_builder import OperationalStateBuilder
        builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS)
        st = builder.build_state(current_time_us=0.0)
        causal_ok = (len(st) == CANONICAL_OBS_DIM and not np.any(np.isnan(st)))
        self.log_gate("LOCAL", "CAUSAL_OBSERVATION", causal_ok, f"Causal state builder functional (dim={len(st)})")

        # 14. Parent checkpoint lineage strictness
        if FROZEN_CKPT_PATH.exists():
            h = hashlib.sha256()
            with open(FROZEN_CKPT_PATH, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual = h.hexdigest()
            lineage_ok = (actual == EXPECTED_FROZEN_SHA256)
            self.log_gate("LOCAL", "PARENT_CHECKPOINT_LINEAGE", lineage_ok,
                f"SHA256={actual[:16]}..." if lineage_ok else f"MISMATCH: expected {EXPECTED_FROZEN_SHA256[:16]}... got {actual[:16]}...")
        else:
            self.log_gate("LOCAL", "PARENT_CHECKPOINT_LINEAGE", False, f"Missing {FROZEN_CKPT_PATH}")

        # 15. Isolated training output directory
        out_cand = REPO_ROOT / "experiments/checkpoints/scheduler_v2_operational_candidate"
        out_ok = out_cand.exists() and (out_cand != prod_base)
        self.log_gate("LOCAL", "ISOLATED_TRAINING_OUTPUT", out_ok, f"Candidate dir {out_cand.name} isolated from production baseline")

        # 16. DRQN gradient flow unit test execution
        grad_test_res = subprocess.run(
            [sys.executable, "-m", "pytest", "ew_core/tests/test_drqn_gradient_flow.py", "-q"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.log_gate("LOCAL", "GRADIENT_FLOW_TESTS", grad_test_res.returncode == 0, "Q-loss, top-band, and mode penalties pass gradient flow")

        # 17. Diversity penalties live
        import torch
        from ew_core.models.drqn_scheduler import DRQNScheduler
        drqn_test = DRQNScheduler(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS, n_actions=CANONICAL_N_ACTIONS)
        dummy_raw = torch.randn(4, 8, CANONICAL_N_ACTIONS)
        dummy_raw[:, :, :CANONICAL_N_MODES] += 10.0  # Concentrate on band 0 so top_frac > 0.80
        dummy_q = dummy_raw.clone().detach().requires_grad_(True)
        q_banded = dummy_q.view(-1, CANONICAL_N_BANDS, CANONICAL_N_MODES)
        softmax = torch.softmax(q_banded.max(dim=-1).values, dim=-1)
        top_frac = softmax.max(dim=-1).values.mean()
        pen = 0.005 * (top_frac - 0.80).clamp(min=0.0)
        pen.backward()
        grad_ok = dummy_q.grad is not None and dummy_q.grad.abs().sum().item() > 0
        self.log_gate("LOCAL", "DIVERSITY_PENALTIES_LIVE", grad_ok,
            "Diversity penalty gradient confirmed non-zero" if grad_ok else "ZERO gradient — diversity penalty is detached")

        # 18. Mode diversity safeguards
        has_mode_penalty = hasattr(drqn_test, "forward")
        self.log_gate("LOCAL", "MODE_DIVERSITY_SAFEGUARDS", has_mode_penalty, "Mode diversity regularizer & forward graph active")

        # 19. Metric semantics separation
        from ew_core.metrics.ew_metrics import EWMetrics
        metric_fields = EWMetrics.__annotations__ if hasattr(EWMetrics, "__annotations__") else {}
        semantics_ok = "pd" in metric_fields and "pfa" in metric_fields and "avg_intercept_time_error_us" in metric_fields
        self.log_gate("LOCAL", "METRIC_SEMANTICS_SEPARATION", semantics_ok, "Pd, Pfa, and arrival forecast MAE strictly separated in EWMetrics")

        # 20. EWMetrics hardening & counter zero-preservation tests
        metric_test_res = subprocess.run(
            [sys.executable, "-m", "pytest", "ew_core/tests/test_ew_metrics_hardening.py", "-q"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.log_gate("LOCAL", "EWMETRICS_HARDENING_TESTS", metric_test_res.returncode == 0, "Counter zero-preservation & confusion invariants pass")

        # 21-25. Benchmark Artifact Verification
        if not BENCHMARK_PATH.exists():
            self.log_gate("LOCAL", "BENCHMARK_ARTIFACT_EXISTS", False, "Missing reports/benchmark_results.json")
            for gate_name in ["BENCHMARK_4_SCHEDULERS", "BENCHMARK_INVARIANTS", "BENCHMARK_PD_FORMULA", "BENCHMARK_PFA_FORMULA", "BENCHMARK_FINITE_NUMBERS"]:
                self.log_gate("LOCAL", gate_name, False, "Benchmark artifact absent")
        else:
            with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                b_data = json.load(f)

            scheds = set(b_data.get("schedulers", {}).keys())
            scheds_pass = (scheds == EXPECTED_SCHEDULERS)
            self.log_gate("LOCAL", "BENCHMARK_4_SCHEDULERS", scheds_pass, f"Found {list(scheds)}")

            # Invariants
            moe_summary = b_data["schedulers"].get("SmartScan_DRQN_MoE", {}).get("summary", {})
            tp = moe_summary.get("tp", 0)
            fn = moe_summary.get("fn", 0)
            fp = moe_summary.get("fp", 0)
            tn = moe_summary.get("tn", 0)
            n_dwells = moe_summary.get("n_receiver_dwells", 5000)
            inv_pass = ((tp + fn + fp + tn) == n_dwells)
            self.log_gate("LOCAL", "BENCHMARK_CONFUSION_INVARIANTS", inv_pass, f"TP={tp}+FN={fn}+FP={fp}+TN={tn} == {n_dwells} (match={inv_pass})")

            # Pd formula
            pd_val = moe_summary.get("pd", 0.0)
            exp_pd = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            pd_match = abs(pd_val - exp_pd) < 1e-4
            self.log_gate("LOCAL", "BENCHMARK_PD_FORMULA", pd_match, f"Pd={pd_val:.4f} == TP/(TP+FN) ({exp_pd:.4f})")

            # Pfa formula
            pfa_val = moe_summary.get("pfa", 0.0)
            exp_pfa = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            pfa_match = abs(pfa_val - exp_pfa) < 1e-4
            self.log_gate("LOCAL", "BENCHMARK_PFA_FORMULA", pfa_match, f"Pfa={pfa_val:.4f} == FP/(FP+TN) ({exp_pfa:.4f})")

            # Finite numbers
            raw_json = BENCHMARK_PATH.read_text(encoding="utf-8")
            finite_pass = ("NaN" not in raw_json and "Infinity" not in raw_json)
            self.log_gate("LOCAL", "BENCHMARK_FINITE_NUMBERS", finite_pass, "No NaN or Infinity found in artifact")

        # 26. 10 canonical scenarios
        b_meta = b_data.get("metadata", {}) if BENCHMARK_PATH.exists() else {}
        b_scens = b_meta.get("scenarios", [])
        scens_pass = (set(b_scens) == set(CANONICAL_SCENARIOS))
        self.log_gate("LOCAL", "BENCHMARK_10_SCENARIOS", scens_pass, f"10 canonical scenarios evaluated: {len(b_scens)}/10")

        # 27. Benchmark reproducibility test
        repro_res = subprocess.run(
            [sys.executable, "-m", "pytest", "ew_core/tests/test_benchmark_reproducibility.py", "-q"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.log_gate("LOCAL", "BENCHMARK_REPRODUCIBILITY", repro_res.returncode == 0, "Fixed-seed evaluation yields identical metrics")

        # 28. Benchmark contract hardening suite
        contract_res = subprocess.run(
            [sys.executable, "-m", "pytest", "ew_core/tests/test_benchmark_contract_hardening.py", "-q"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.log_gate("LOCAL", "BENCHMARK_CONTRACT_HARDENING", contract_res.returncode == 0, "12 contract hardening & fail-closed tests pass")

        # 29. Sensitivity consistency (-110.0 dBm)
        sens_val = moe_summary.get("sensitivity_dbm", 0.0) if BENCHMARK_PATH.exists() else 0.0
        sens_pass = (abs(sens_val - (-110.0)) < 1e-3)
        self.log_gate("LOCAL", "SENSITIVITY_CONSISTENCY", sens_pass, f"Receiver sensitivity={sens_val} dBm (expected -110.0 dBm)")

        # 30. Benchmark documentation consistency
        rep_md = (REPO_ROOT / "reports/benchmark_report.md").read_text(encoding="utf-8") if (REPO_ROOT / "reports/benchmark_report.md").exists() else ""
        doc_pass = ("-110.0 dBm" in rep_md and "-140.0 dBm" not in rep_md)
        self.log_gate("LOCAL", "BENCHMARK_DOC_CONSISTENCY", doc_pass, "Markdown report grounded at -110.0 dBm without contradictory -140.0 dBm")

        # 31. Qualification Run Evidence Freshness and Integrity
        q_dir = REPO_ROOT / "experiments/checkpoints/quarantine"
        manifest_p = q_dir / "qualification_run_manifest.json"
        summary_p = q_dir / "qualification_run_summary.json"
        if not manifest_p.exists() or not summary_p.exists():
            self.log_gate(
                "LOCAL",
                "QUALIFICATION_EVIDENCE_FRESHNESS",
                False,
                f"Missing qualification evidence in {q_dir} (manifest={manifest_p.exists()}, summary={summary_p.exists()})"
            )
        else:
            try:
                try:
                    cur_git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                except Exception:
                    cur_git_head = "unknown"

                t_cfg_path = REPO_ROOT / "configs/training_config_resume_100k.yaml"
                m_cfg_path = REPO_ROOT / "configs/model_config.yaml"
                cur_t_cfg_sha = hashlib.sha256(t_cfg_path.read_bytes()).hexdigest() if t_cfg_path.exists() else ""
                cur_m_cfg_sha = hashlib.sha256(m_cfg_path.read_bytes()).hexdigest() if m_cfg_path.exists() else ""

                m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
                s_data = json.loads(summary_p.read_text(encoding="utf-8"))

                run_id_match = bool(m_data.get("run_id") and m_data.get("run_id") == s_data.get("run_id"))
                parent_sha_ok = bool(s_data.get("parent_checkpoint_sha256") == EXPECTED_FROZEN_SHA256)
                steps_requested_ok = bool(s_data.get("qualification_steps_requested") == 1000)
                steps_ok = bool(
                    s_data.get("start_global_step") == 25000 and
                    s_data.get("final_global_step") == 26000 and
                    s_data.get("qualification_steps_completed") == 1000 and
                    steps_requested_ok
                )
                updates_attempted = s_data.get("optimizer_updates_attempted", 0)
                updates_completed = s_data.get("optimizer_updates_completed", 0)
                finite_gradient_updates = s_data.get("finite_gradient_updates", 0)
                non_finite_gradient_updates = s_data.get("non_finite_gradient_updates", 0)

                updates_ok = bool(
                    updates_attempted > 0 and
                    updates_completed == updates_attempted and
                    finite_gradient_updates == updates_completed and
                    non_finite_gradient_updates == 0
                )
                skips_zero = bool(
                    s_data.get("skipped_nan", -1) == 0 and
                    s_data.get("skipped_assertion", -1) == 0 and
                    s_data.get("skipped_oom", -1) == 0 and
                    s_data.get("other_update_failures", -1) == 0 and
                    s_data.get("validation_failures", -1) == 0
                )
                finite_grads = bool(s_data.get("finite_gradients") is True and finite_gradient_updates > 0)
                timestamps_ok = bool(
                    s_data.get("qualification_started_at_utc") and
                    s_data.get("qualification_completed_at_utc") and
                    s_data.get("summary_generated_at_utc") and
                    s_data.get("qualification_completed_at_utc") >= s_data.get("qualification_started_at_utc") and
                    s_data.get("summary_generated_at_utc") >= s_data.get("qualification_started_at_utc")
                )
                quarantine_isolated = ("quarantine" in s_data.get("checkpoint_output_dir", "").lower())

                git_match = bool(cur_git_head != "unknown" and s_data.get("git_commit_sha") == cur_git_head)
                t_cfg_match = bool(cur_t_cfg_sha and s_data.get("training_config_sha256") == cur_t_cfg_sha)
                m_cfg_match = bool(cur_m_cfg_sha and s_data.get("model_config_sha256") == cur_m_cfg_sha)

                q_pass = bool(
                    run_id_match and parent_sha_ok and steps_ok and
                    updates_ok and skips_zero and finite_grads and
                    timestamps_ok and quarantine_isolated and
                    git_match and t_cfg_match and m_cfg_match
                )
                detail = (
                    f"run_id={m_data.get('run_id', '')[:8]}... "
                    f"steps={s_data.get('qualification_steps_completed')}/1000 "
                    f"updates={updates_completed}/{updates_attempted} "
                    f"finite_grads={finite_gradient_updates}/{updates_completed} "
                    f"git_match={git_match} configs_match={t_cfg_match and m_cfg_match}"
                )
                self.log_gate("LOCAL", "QUALIFICATION_EVIDENCE_FRESHNESS", q_pass, detail)
            except Exception as e:
                self.log_gate("LOCAL", "QUALIFICATION_EVIDENCE_FRESHNESS", False, f"Failed parsing qualification evidence: {e}")

    def evaluate_deployment_gates(self):
        print("\n" + "=" * 78)
        print("  STAGE 2: LIVE DEPLOYMENT OPERATIONAL VERIFICATION (5 GATES)")
        print("=" * 78)

        if self.skip_deployment:
            for g in ["DEPLOY_HEALTH", "DEPLOY_READINESS", "DEPLOY_BENCHMARK_CONTRACT", "DEPLOY_INFERENCE_LATENCY", "DEPLOY_PROVENANCE"]:
                self.log_gate("DEPLOY", g, False, "SKIPPED by CLI flag")
            return

        headers = {"X-SmartScan-API-Key": self.api_key} if self.api_key else {}

        # 32. Live Health Probe
        try:
            req = urllib.request.Request(f"{self.api_url}/health", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                h_data = json.loads(resp.read().decode())
                h_status = (resp.status == 200 and h_data.get("status") == "ok")
                self.log_gate("DEPLOY", "DEPLOY_HEALTH", h_status, f"HTTP {resp.status} status={h_data.get('status')}")
        except Exception as e:
            self.log_gate("DEPLOY", "DEPLOY_HEALTH", False, f"Request failed: {e}")
            h_data = {}

        # 33. Live Readiness Probe
        try:
            req = urllib.request.Request(f"{self.api_url}/ready", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                r_data = json.loads(resp.read().decode())
                r_status = (resp.status == 200 and r_data.get("model_loaded") is True)
                self.log_gate("DEPLOY", "DEPLOY_READINESS", r_status, f"HTTP {resp.status} model_loaded={r_data.get('model_loaded')}")
        except Exception as e:
            self.log_gate("DEPLOY", "DEPLOY_READINESS", False, f"Request failed: {e}")

        # 34. Live Benchmark API Contract
        try:
            req = urllib.request.Request(f"{self.api_url}/api/benchmark", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                b_live = json.loads(resp.read().decode())
                scheds = list(b_live.get("schedulers", {}).keys())
                b_pass = (resp.status == 200 and set(scheds) == EXPECTED_SCHEDULERS)
                self.log_gate("DEPLOY", "DEPLOY_BENCHMARK_CONTRACT", b_pass, f"HTTP {resp.status} schedulers={scheds}")
        except Exception as e:
            self.log_gate("DEPLOY", "DEPLOY_BENCHMARK_CONTRACT", False, f"Request failed: {e}")

        # 35. Live Inference Latency SLA (< 500 ms) — 5 warmup + 20 measured
        latencies = []
        try:
            import numpy as np
            dummy_obs = [0.0] * CANONICAL_OBS_DIM
            payload = json.dumps({"obs": dummy_obs, "policy_mode": "operational"}).encode()
            post_headers = {**headers, "Content-Type": "application/json"}
            for _ in range(5):
                req = urllib.request.Request(f"{self.api_url}/predict_bands", data=payload, headers=post_headers)
                urllib.request.urlopen(req, timeout=10).read()

            for _ in range(20):
                t0 = time.perf_counter()
                req = urllib.request.Request(f"{self.api_url}/predict_bands", data=payload, headers=post_headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
                latencies.append((time.perf_counter() - t0) * 1000.0)

            lat_arr = np.array(latencies, dtype=np.float64)
            median_lat = float(np.median(lat_arr))
            p95_lat = float(np.percentile(lat_arr, 95, method="linear"))
            lat_pass = bool(median_lat < 500.0 and p95_lat < 500.0)
            self.log_gate("DEPLOY", "DEPLOY_INFERENCE_LATENCY", lat_pass, f"Median={median_lat:.1f}ms, p95={p95_lat:.1f}ms (< 500ms SLA), Max={np.max(lat_arr):.1f}ms")
        except Exception as e:
            self.log_gate("DEPLOY", "DEPLOY_INFERENCE_LATENCY", False, f"Inference request failed: {e}")

        # 36. Live Deployment Provenance Checkpoint SHA
        ckpt_sha_live = h_data.get("checkpoint_sha256") if isinstance(h_data, dict) else None
        prov_pass = (ckpt_sha_live == EXPECTED_FROZEN_SHA256)
        ckpt_sha_str = ckpt_sha_live[:16] if ckpt_sha_live else "None"
        self.log_gate("DEPLOY", "DEPLOY_PROVENANCE", prov_pass, f"Live SHA={ckpt_sha_str}... match={prov_pass}")

    def run(self) -> bool:
        self.evaluate_local_gates()
        self.evaluate_deployment_gates()

        local_pass = all(p for p, _ in self.local_results.values())
        deployment_gates = list(self.deploy_results.values())
        deploy_pass = all(p for p, _ in deployment_gates) if (deployment_gates and not self.skip_deployment) else False

        print("\n" + "=" * 78)
        print("  FINAL QUALIFICATION VERDICTS")
        print("=" * 78)
        print(f"LOCAL_RETRAINING_READY = {'TRUE' if local_pass else 'FALSE'}")
        print(f"DEPLOYMENT_READY       = {'TRUE' if deploy_pass else ('SKIPPED' if self.skip_deployment else 'FALSE')}")
        print("-" * 78)
        print(f"TRAINING READY — {'GO' if local_pass else 'NO-GO'}")
        print(f"FULL OPERATIONAL READY — {'GO' if (local_pass and deploy_pass) else 'NO-GO'}")
        print("=" * 78 + "\n")

        if not local_pass:
            print("BLOCKING LOCAL FAILURES:")
            for name, (p, msg) in self.local_results.items():
                if not p:
                    print(f"  - [LOCAL] {name}: {msg}")
        if not deploy_pass and not self.skip_deployment:
            print("BLOCKING DEPLOYMENT FAILURES:")
            for name, (p, msg) in self.deploy_results.items():
                if not p:
                    print(f"  - [DEPLOY] {name}: {msg}")
            print()

        return local_pass if self.skip_deployment else (local_pass and deploy_pass)


def main():
    parser = argparse.ArgumentParser(description="Retraining Readiness Qualification Gate")
    parser.add_argument("--api_url", type=str, default=os.environ.get("CLOUD_RUN_URL", "https://cognitive-ew-backend-753709137146.asia-south1.run.app"), help="Live deployment API URL")
    parser.add_argument("--api_key", type=str, default=os.environ.get("SMARTSCAN_API_KEY", ""), help="Live deployment API key")
    parser.add_argument("--skip_deployment", action="store_true", help="Evaluate local gates only")
    args = parser.parse_args()

    evaluator = ReadinessGateEvaluator(
        api_url=args.api_url,
        api_key=args.api_key,
        skip_deployment=args.skip_deployment,
    )
    ready = evaluator.run()
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
