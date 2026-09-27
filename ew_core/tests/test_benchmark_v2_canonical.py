"""Unit tests for Benchmark V2 Canonical Contracts, Baseline, and Promotion Sentinel."""

import json
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from ew_core.evaluation.benchmark_contract import (
    BENCHMARK_VERSION,
    CANONICAL_SCENARIOS,
    get_benchmark_contract,
    _file_sha256,
)
from ew_core.evaluation.promotion_sentinel import (
    evaluate_promotion,
    BASELINE_GATE25K,
)
from ew_core.deployment.api import app


class BenchmarkV2CanonicalTests(unittest.TestCase):
    def test_benchmark_contract_constants(self):
        self.assertEqual(BENCHMARK_VERSION, "2026.1-CANONICAL")
        self.assertEqual(len(CANONICAL_SCENARIOS), 10)
        contract = get_benchmark_contract()
        self.assertEqual(len(contract["checkpoint_sha256"]), 64)
        self.assertEqual(contract["checkpoint_sha256"], "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0")

    def test_baseline_gate25k_reproduction_report(self):
        report_path = Path("experiments/reports/benchmark_v2_baseline_gate25k.json")
        if not report_path.exists():
            report_path = Path("reports/benchmark_v2_baseline_gate25k.json")
        self.assertTrue(report_path.exists())

        with open(report_path, "r") as f:
            data = json.load(f)

        self.assertEqual(data["benchmark_version"], "2026.1-CANONICAL")
        self.assertEqual(data["checkpoint_sha256"], "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0")
        self.assertEqual(data["metrics"]["drqn_mean_ir_pct"], BASELINE_GATE25K["mean_ir"])
        self.assertEqual(data["metrics"]["drqn_agile_ir_pct"], BASELINE_GATE25K["agile_ir"])
        self.assertEqual(data["metrics"]["drqn_sparse_ir_pct"], BASELINE_GATE25K["sparse_ir"])
        self.assertEqual(data["metrics"]["drqn_worst_case_ir_pct"], BASELINE_GATE25K["worst_case_ir"])


    def test_multiseed_summary_report(self):
        summary_path = Path("experiments/reports/benchmark_v2_multiseed_summary.json")
        if not summary_path.exists():
            summary_path = Path("reports/benchmark_v2_multiseed_summary.json")
        self.assertTrue(summary_path.exists())

        with open(summary_path, "r") as f:
            data = json.load(f)

        self.assertIn("Gate-25k-Frozen", data)
        self.assertIn("HighestOccupancy heuristic baseline", data)
        self.assertIn("RoundRobin", data)
        self.assertIn("Random", data)

        # Confirm Gate-25k outperforms HighestOccupancy heuristic baseline across seeds
        gate25k_mean = data["Gate-25k-Frozen"]["mean"]
        heuristic_mean = data["HighestOccupancy heuristic baseline"]["mean"]
        self.assertGreater(gate25k_mean, heuristic_mean + 10.0)

    def test_promotion_sentinel_logic(self):
        units = {"mean_ir": "fraction", "agile_ir": "fraction", "sparse_ir": "fraction", "worst_case_ir": "fraction", "pfa": "fraction"}
        # 1. Collapsed candidate should be rejected
        collapsed_mock = {
            "evaluation_type": "metric_only_diagnostic",
            "metric_units": units,
            "training_diagnostics": {"q_max": 20.0},
            "composite_generalization_components_10scen": {
                "overall_ir": 0.55,
                "agile_ir": 0.35,
                "sparse_ir": 0.10,
                "worst_case_ir": 0.05,
            },
            "baseline_hierarchy": {"drqn": {"pfa": 0.0, "distinct_bands": 25.0, "action_entropy": 2.5}},
        }
        promoted, verdict, details = evaluate_promotion(collapsed_mock)
        self.assertFalse(promoted)
        self.assertEqual(details["promotion_status"], "REJECTED")
        self.assertIn("REJECT", verdict)

        # 2. Candidate with Q-explosion should be rejected (Level 0)
        q_exploded_mock = {
            "evaluation_type": "metric_only_diagnostic",
            "metric_units": units,
            "training_diagnostics": {"q_max": 65.0},
            "composite_generalization_components_10scen": {
                "overall_ir": 0.65,
                "agile_ir": 0.50,
                "sparse_ir": 0.20,
                "worst_case_ir": 0.15,
            },
            "baseline_hierarchy": {"drqn": {"pfa": 0.0, "distinct_bands": 30.0, "action_entropy": 2.5}},
        }
        promoted, verdict, details = evaluate_promotion(q_exploded_mock)
        self.assertFalse(promoted)
        self.assertEqual(details["promotion_status"], "REJECTED")
        self.assertIn("Q-explosion", verdict)

        # 3. Superior stable candidate should be promoted (Level 1)
        superior_mock = {
            "evaluation_type": "metric_only_diagnostic",
            "metric_units": units,
            "training_diagnostics": {"q_max": 25.0},
            "composite_generalization_components_10scen": {
                "overall_ir": 0.62,
                "agile_ir": 0.48,
                "sparse_ir": 0.18,
                "worst_case_ir": 0.13,
            },
            "baseline_hierarchy": {"drqn": {"pfa": 0.0, "distinct_bands": 30.0, "action_entropy": 2.5}},
        }
        promoted, verdict, details = evaluate_promotion(superior_mock)
        self.assertTrue(promoted)
        self.assertEqual(details["promotion_status"], "APPROVED")
        self.assertIn("PROMOTED", verdict)

    def test_api_health_endpoint(self):
        with TestClient(app) as client:
            res = client.get("/health")
            self.assertIn(res.status_code, [200, 503], "Health endpoint must return 200 or 503, not crash")
            data = res.json()
            self.assertIn("status", data)
            if res.status_code == 200:
                self.assertEqual(data["status"], "ok")
                self.assertIn(data["active_model"], ["Gate-27 Operational Baseline", "Gate-25k-R4.2-alpha020"])
                self.assertIn(
                    data["checkpoint_sha256"],
                    [
                        "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094",
                        "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
                    ],
                )
                self.assertEqual(data["benchmark_version"], BENCHMARK_VERSION)



if __name__ == "__main__":
    unittest.main()
