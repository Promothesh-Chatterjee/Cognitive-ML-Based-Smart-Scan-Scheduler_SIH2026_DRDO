"""
Phase 8 Dedicated Unit Tests: Operational Readiness Gate Logic.

Covers 25 formal verification tests across:
1. Fail-closed validators (present, finite, non-NaN/Inf, strict typing, bounds, thresholds)
2. OperationalReadinessVerdict schema and serialization
3. CheckpointGuard fail-closed resolution, tamper detection, and unapproved rejection
4. Deployment /health endpoint 10-dimensional prerequisites, HTTP 503 semantics, and readiness_failures
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import Response
from pydantic import ValidationError

from ew_core.deployment.api import (
    STATE,
    HealthResponse,
    health,
)
from ew_core.evaluation.readiness_validator import (
    MissingMetricError,
    NonFiniteMetricError,
    OperationalReadinessVerdict,
    ReadinessError,
    ReadinessStatus,
    check_threshold,
    require_finite_metric,
    require_present_metric,
)
from ew_core.training.safety.checkpoint_guard import (
    CheckpointGuard,
    CheckpointSecurityError,
    CheckpointTamperedError,
    ExplicitPromotionRequiredError,
    sha256_file,
)
from ew_core.utils.checkpoint_paths import EXPECTED_FROZEN_SHA256


# ── Group 1: Fail-Closed Validator Tests (14 Tests) ──────────────────────────

def test_require_present_metric_success():
    """Verify require_present_metric successfully extracts existing key."""
    data = {"pd": 0.45, "latency": 15.0}
    assert require_present_metric(data, "pd", "metric_pd") == 0.45
    assert require_present_metric(data, "latency", "metric_lat") == 15.0


def test_require_present_metric_missing_key():
    """Verify require_present_metric raises MissingMetricError when key is absent."""
    data = {"pd": 0.45}
    with pytest.raises(MissingMetricError, match="Missing required metric: 'metric_lat'"):
        require_present_metric(data, "latency", "metric_lat")


def test_require_present_metric_none_value():
    """Verify require_present_metric raises MissingMetricError when value is None."""
    data = {"pd": None}
    with pytest.raises(MissingMetricError, match="absent or None"):
        require_present_metric(data, "pd", "metric_pd")


def test_require_present_metric_non_dict():
    """Verify require_present_metric raises MissingMetricError when container is not a dict."""
    with pytest.raises(MissingMetricError, match="not a dict"):
        require_present_metric([1, 2, 3], "pd", "metric_pd")  # type: ignore


def test_require_finite_metric_valid_float():
    """Verify require_finite_metric accepts valid positive/negative floats."""
    assert require_finite_metric(42.5, "test_float") == 42.5
    assert require_finite_metric(-0.001, "neg_float") == -0.001
    assert require_finite_metric(0.0, "zero") == 0.0


def test_require_finite_metric_valid_int():
    """Verify require_finite_metric coerces integer to valid float."""
    assert require_finite_metric(10, "test_int") == 10.0


def test_require_finite_metric_nan():
    """Verify require_finite_metric raises NonFiniteMetricError on NaN."""
    with pytest.raises(NonFiniteMetricError, match="non-finite or NaN"):
        require_finite_metric(float("nan"), "metric_nan")


def test_require_finite_metric_inf():
    """Verify require_finite_metric raises NonFiniteMetricError on positive/negative Infinity."""
    with pytest.raises(NonFiniteMetricError, match="non-finite or NaN"):
        require_finite_metric(float("inf"), "metric_pos_inf")
    with pytest.raises(NonFiniteMetricError, match="non-finite or NaN"):
        require_finite_metric(float("-inf"), "metric_neg_inf")


def test_require_finite_metric_bool():
    """Verify require_finite_metric rejects boolean values (fail-closed typing)."""
    with pytest.raises(NonFiniteMetricError, match="must be numeric"):
        require_finite_metric(True, "metric_bool")


def test_require_finite_metric_string():
    """Verify require_finite_metric rejects string values."""
    with pytest.raises(NonFiniteMetricError, match="must be numeric"):
        require_finite_metric("42.5", "metric_str")


def test_require_finite_metric_min_bound_violation():
    """Verify require_finite_metric rejects values below min_val."""
    with pytest.raises(ValueError, match="below minimum allowed"):
        require_finite_metric(-0.01, "metric_pfa", min_val=0.0)


def test_require_finite_metric_max_bound_violation():
    """Verify require_finite_metric rejects values above max_val."""
    with pytest.raises(ValueError, match="above maximum allowed"):
        require_finite_metric(1.05, "metric_pd", max_val=1.0)


def test_check_threshold_operators():
    """Verify check_threshold evaluates all supported operators correctly."""
    # >=
    p, msg = check_threshold(0.4063, 0.4063, ">=", "Pd")
    assert p is True and "PASS" in msg
    p, msg = check_threshold(0.4000, 0.4063, ">=", "Pd")
    assert p is False and "FAIL" in msg

    # <=
    p, msg = check_threshold(26.5, 27.0, "<=", "Latency")
    assert p is True and "PASS" in msg
    p, msg = check_threshold(27.5, 27.0, "<=", "Latency")
    assert p is False and "FAIL" in msg

    # >
    p, msg = check_threshold(2.05, 2.0, ">", "PreferenceRatio")
    assert p is True and "PASS" in msg
    p, msg = check_threshold(2.0, 2.0, ">", "PreferenceRatio")
    assert p is False and "FAIL" in msg

    # <
    p, msg = check_threshold(4.2, 5.0, "<", "CycleTime")
    assert p is True and "PASS" in msg
    p, msg = check_threshold(5.0, 5.0, "<", "CycleTime")
    assert p is False and "FAIL" in msg

    # ==
    p, msg = check_threshold(100.0, 100.0, "==", "Escape")
    assert p is True and "PASS" in msg


def test_check_threshold_unsupported_op():
    """Verify check_threshold raises ValueError on invalid operator."""
    with pytest.raises(ValueError, match="Unsupported comparison operator"):
        check_threshold(1.0, 1.0, "!=", "Test")


# ── Group 2: Operational Readiness Verdict & Status (3 Tests) ────────────────

def test_readiness_status_enum_values():
    """Verify allowed readiness status values under Phase 8 contract."""
    assert ReadinessStatus.READY.value == "READY"
    assert ReadinessStatus.NOT_READY.value == "NOT_READY"
    assert ReadinessStatus.INTEGRITY_FAILURE.value == "INTEGRITY_FAILURE"
    assert ReadinessStatus.EVALUATION_ERROR.value == "EVALUATION_ERROR"


def test_operational_readiness_verdict_serialization():
    """Verify OperationalReadinessVerdict serializes to dictionary with expected keys."""
    v = OperationalReadinessVerdict(
        schema_version="phase8",
        readiness_status=ReadinessStatus.READY.value,
        all_gates_passed=True,
        checkpoint={"sha256": EXPECTED_FROZEN_SHA256},
        gates={"gate_a": {"passed": True}},
        failures=[],
        metadata={"seed": 42},
    )
    d = v.to_dict()
    assert d["schema_version"] == "phase8"
    assert d["readiness_status"] == "READY"
    assert d["all_gates_passed"] is True
    assert d["checkpoint"]["sha256"] == EXPECTED_FROZEN_SHA256
    assert d["gates"]["gate_a"]["passed"] is True


def test_operational_readiness_verdict_fail_closed_defaults():
    """Verify default OperationalReadinessVerdict is fail-closed (NOT_READY)."""
    v = OperationalReadinessVerdict()
    assert v.readiness_status == ReadinessStatus.NOT_READY.value
    assert v.all_gates_passed is False
    assert len(v.failures) == 0


# ── Group 3: CheckpointGuard Resolution & Governance (3 Tests) ───────────────

def test_checkpoint_guard_resolves_active_approved():
    """Verify CheckpointGuard correctly returns active approved checkpoint."""
    guard = CheckpointGuard("experiments/checkpoints/scheduler_v2_operational_candidate")
    active_ckpt = guard.get_active_checkpoint()
    assert active_ckpt.is_file()
    assert active_ckpt.name in ("checkpoint_gate_27000_operational.pt", "checkpoint_gate_25000_frozen.pt")
    assert sha256_file(active_ckpt) in (
        "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094",
        EXPECTED_FROZEN_SHA256,
    )


def test_checkpoint_guard_detects_tampering(tmp_path):
    """Verify CheckpointGuard raises CheckpointTamperedError if on-disk SHA differs."""
    guard = CheckpointGuard(tmp_path)
    fake_ckpt = tmp_path / "fake_model.pt"
    fake_ckpt.write_bytes(b"content_a")
    manifest = {
        "status": "APPROVED",
        "promotion_status": "APPROVED",
        "checkpoint_path": str(fake_ckpt),
        "checkpoint_sha256": "0" * 64,  # mismatched SHA
    }
    with open(tmp_path / "ACTIVE_CHECKPOINT.json", "w") as f:
        import json
        json.dump(manifest, f)

    with pytest.raises(CheckpointTamperedError, match="CRITICAL INTEGRITY FAILURE"):
        guard.get_active_checkpoint()


def test_checkpoint_guard_fails_closed_without_manifest(tmp_path):
    """Verify CheckpointGuard raises ExplicitPromotionRequiredError if manifest is absent."""
    guard = CheckpointGuard(tmp_path)
    (tmp_path / "candidate_model.pt").write_bytes(b"dummy")
    with pytest.raises(ExplicitPromotionRequiredError, match="No explicit active promotion manifest"):
        guard.get_active_checkpoint()


# ── Group 4: Deployment /health Endpoint Contract Tests (5 Tests) ───────────

def test_health_reports_degraded_when_models_unloaded():
    """Verify /health returns HTTP 503 and degraded status when models are missing."""
    resp = Response()
    # Mock clear STATE
    with patch.dict(STATE, {}, clear=True):
        health_resp = health(response=resp)
        assert resp.status_code == 503
        assert health_resp.status == "degraded"
        assert health_resp.operational_mode_ready is False
        assert len(health_resp.readiness_failures) > 0
        assert any("Scheduler neural policy not loaded" in f for f in health_resp.readiness_failures)
        assert any("PDW deinterleaver transformer not loaded" in f for f in health_resp.readiness_failures)


def test_health_reports_degraded_on_dimension_failure():
    """Verify /health reports failure when dimension check is False."""
    resp = Response()
    mock_state = {
        "scheduler": object(),
        "deinterleaver": object(),
        "controller": object(),
        "dimension_check_passed": False,  # FAIL
        "normalization_hash_match": True,
        "hidden_state_ready": True,
        "active_model": "Gate-25k-R4.2-alpha020",
        "scheduler_ckpt_sha256": EXPECTED_FROZEN_SHA256,
        "policy_mode": "operational",
    }
    with patch.dict(STATE, mock_state, clear=True):
        health_resp = health(response=resp)
        assert resp.status_code == 503
        assert health_resp.operational_mode_ready is False
        assert any("Canonical 360-D observation dimension verification failed" in f for f in health_resp.readiness_failures)


def test_health_reports_degraded_when_exploration_enabled():
    """Verify /health enforces fail-closed deterministic policy (exploration=False)."""
    resp = Response()
    mock_moe = MagicMock()
    mock_moe.policy_mode = "operational"
    mock_moe.exploration_enabled = True  # VIOLATION: exploration enabled in operational mode

    mock_state = {
        "scheduler": object(),
        "deinterleaver": object(),
        "controller": object(),
        "dimension_check_passed": True,
        "normalization_hash_match": True,
        "hidden_state_ready": True,
        "active_model": "Gate-25k-R4.2-alpha020",
        "scheduler_ckpt_sha256": EXPECTED_FROZEN_SHA256,
        "moe": mock_moe,
    }
    with patch.dict(STATE, mock_state, clear=True):
        health_resp = health(response=resp)
        assert resp.status_code == 503
        assert health_resp.operational_mode_ready is False
        assert any("Exploration enabled (True)" in f for f in health_resp.readiness_failures)


def test_health_reports_ok_when_all_prerequisites_met():
    """Verify /health returns HTTP 200 OK and operational_mode_ready=True when all pass."""
    resp = Response()
    mock_moe = MagicMock()
    mock_moe.policy_mode = "operational"
    mock_moe.exploration_enabled = False

    mock_state = {
        "scheduler": object(),
        "deinterleaver": object(),
        "controller": object(),
        "dimension_check_passed": True,
        "normalization_hash_match": True,
        "hidden_state_ready": True,
        "active_model": "Gate-25k-R4.2-alpha020",
        "scheduler_ckpt_sha256": EXPECTED_FROZEN_SHA256,
        "moe": mock_moe,
    }
    with patch.dict(STATE, mock_state, clear=True):
        health_resp = health(response=resp)
        assert resp.status_code == 200
        assert health_resp.status == "ok"
        assert health_resp.operational_mode_ready is True
        assert len(health_resp.readiness_failures) == 0


def test_health_operational_ready_decoupled_from_mission_active():
    """Verify operational_mode_ready does not require a mission to be actively running."""
    resp = Response()
    mock_moe = MagicMock()
    mock_moe.policy_mode = "operational"
    mock_moe.exploration_enabled = False

    mock_state = {
        "scheduler": object(),
        "deinterleaver": object(),
        "controller": object(),
        "dimension_check_passed": True,
        "normalization_hash_match": True,
        "hidden_state_ready": True,
        "active_model": "Gate-25k-R4.2-alpha020",
        "scheduler_ckpt_sha256": EXPECTED_FROZEN_SHA256,
        "moe": mock_moe,
        "is_mission_active": False,  # Mission is idle!
    }
    with patch.dict(STATE, mock_state, clear=True):
        health_resp = health(response=resp)
        assert resp.status_code == 200
        assert health_resp.operational_mode_ready is True
