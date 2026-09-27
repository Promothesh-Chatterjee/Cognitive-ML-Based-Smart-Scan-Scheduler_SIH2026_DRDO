"""Verification test for Gate-27 Qualified Operational Baseline.

Validates:
  1. Gate-27 operational checkpoint exists and matches SHA-256 fac05774...
  2. Historical Gate-25 baseline remains untouched and matches 7a99c659...
  3. CheckpointGuard resolves Gate-27 as the active approved checkpoint.
  4. BandConditionedFactorizedDRQN architecture instantiates with canonical dimensions.
  5. Manifests (ACTIVE_CHECKPOINT.json and GATE27_OPERATIONAL_MANIFEST.json) are valid.
  6. Backend /health identifies Gate-27 Operational Baseline.
  7. /predict_bands executes real non-fabricated inference.
  8. /mission lifecycle executes closed-loop with Gate-27.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from ew_core.contracts import (
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    DEFAULT_DWELL_MULTIPLIERS,
    RF_BASE_DWELL_TIME_US,
)
from ew_core.deployment.api import STATE, app
from ew_core.models.band_conditioned_drqn import BandConditionedFactorizedDRQN
from ew_core.training.safety.checkpoint_guard import CheckpointGuard, sha256_file

EXPECTED_GATE27_SHA = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"
EXPECTED_GATE25_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"

GATE27_OP_PATH = Path("experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_27000_operational.pt")
GATE27_SRC_PATH = Path("experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt")
GATE25_HIST_PATH = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")


def test_gate27_checkpoint_integrity():
    """Verify Gate-27 files exist and match exact required bitwise hash."""
    assert GATE27_SRC_PATH.is_file(), f"Source Gate-27 checkpoint missing: {GATE27_SRC_PATH}"
    assert GATE27_OP_PATH.is_file(), f"Operational candidate Gate-27 checkpoint missing: {GATE27_OP_PATH}"
    
    src_sha = sha256_file(GATE27_SRC_PATH)
    op_sha = sha256_file(GATE27_OP_PATH)
    
    assert src_sha == EXPECTED_GATE27_SHA, f"Source SHA mismatch: {src_sha}"
    assert op_sha == EXPECTED_GATE27_SHA, f"Operational candidate SHA mismatch: {op_sha}"


def test_gate25_historical_baseline_untouched():
    """Verify historical Gate-25 baseline remains completely untouched."""
    assert GATE25_HIST_PATH.is_file(), f"Historical Gate-25 checkpoint missing: {GATE25_HIST_PATH}"
    hist_sha = sha256_file(GATE25_HIST_PATH)
    assert hist_sha == EXPECTED_GATE25_SHA, f"Gate-25 historical baseline hash changed! Got {hist_sha}"


def test_checkpoint_guard_resolves_gate27():
    """Verify CheckpointGuard resolves Gate-27 operational candidate."""
    cand_dir = Path("experiments/checkpoints/scheduler_v2_operational_candidate")
    guard = CheckpointGuard(cand_dir)
    active_ckpt = guard.get_active_checkpoint()
    
    assert active_ckpt.name == "checkpoint_gate_27000_operational.pt"
    resolved_sha = sha256_file(active_ckpt)
    assert resolved_sha == EXPECTED_GATE27_SHA


def test_gate27_model_instantiation():
    """Verify checkpoint loads into BandConditionedFactorizedDRQN with finite weights."""
    ckpt = torch.load(str(GATE27_OP_PATH), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), f"Non-finite weights found in {k}"
            
    model = BandConditionedFactorizedDRQN(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0
    assert model.obs_dim == CANONICAL_OBS_DIM
    assert model.n_bands == CANONICAL_N_BANDS
    assert model.n_modes == CANONICAL_N_MODES
    assert model.n_actions == CANONICAL_N_ACTIONS


def test_manifest_provenance_and_schema():
    """Verify ACTIVE_CHECKPOINT.json and GATE27_OPERATIONAL_MANIFEST.json contents."""
    active_manifest_path = Path("experiments/checkpoints/scheduler_v2_operational_candidate/ACTIVE_CHECKPOINT.json")
    assert active_manifest_path.is_file()
    
    with open(active_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["status"] == "APPROVED"
    assert manifest["checkpoint_filename"] == "checkpoint_gate_27000_operational.pt"
    assert manifest["checkpoint_sha256"] == EXPECTED_GATE27_SHA
    assert manifest["baseline_checkpoint_sha256"] == EXPECTED_GATE25_SHA
    assert manifest["training_step"] == 27000

    op_manifest_path = Path("experiments/checkpoints/scheduler_v2_operational_candidate/GATE27_OPERATIONAL_MANIFEST.json")
    assert op_manifest_path.is_file()
    with open(op_manifest_path, "r", encoding="utf-8") as f:
        op_manifest = json.load(f)
    assert op_manifest["cloud_artifact_sha256"] == EXPECTED_GATE27_SHA
    assert op_manifest["cloud_status"] == "NOT_DEPLOYED"
    assert op_manifest["research_status"] == "CLOSED"
    assert op_manifest["training_status"] == "FROZEN"


def test_fastapi_health_identifies_gate27():
    """Verify FastAPI /health identifies active Gate-27 baseline."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["active_model"] == "Gate-27 Operational Baseline"
        assert data["checkpoint_sha256"] == EXPECTED_GATE27_SHA
        assert data["operational_mode_ready"] is True
        assert data["exploration_enabled"] is False
        assert len(data["readiness_failures"]) == 0


def test_fastapi_real_inference():
    """Verify /predict_bands executes real non-fabricated inference."""
    with TestClient(app) as client:
        obs = np.random.randn(360).tolist()
        resp = client.post("/predict_bands", json={"obs": obs, "policy_mode": "operational"})
        assert resp.status_code == 200
        data = resp.json()
        
        action = data["action"]
        band = data.get("selected_band", data.get("band"))
        mode = data.get("selected_mode", data.get("mode"))
        dwell_us = data["dwell_time_us"]
        prob = data["intercept_probability"]
        pred_time_us = data["predicted_intercept_time_us"]
        latency_ms = data["latency_ms"]
        
        assert 0 <= action < 180
        assert 0 <= band < 36
        assert 0 <= mode < 5
        expected_dwell = RF_BASE_DWELL_TIME_US * DEFAULT_DWELL_MULTIPLIERS[mode]
        assert abs(dwell_us - expected_dwell) < 1e-3
        assert 0.0 <= prob <= 1.0
        assert pred_time_us >= 0.0
        assert latency_ms > 0.0


def test_fastapi_mission_lifecycle():
    """Verify full closed-loop mission lifecycle with Gate-27."""
    with TestClient(app) as client:
        # 1. Reset
        r_reset = client.post("/reset")
        assert r_reset.status_code == 200
        
        # 2. Start mission
        r_start = client.post("/mission/start", json={"initial_time_us": 0.0, "auto_stream": False})
        assert r_start.status_code == 200
        assert r_start.json()["mission_active"] is True
        
        # 3. Mission steps
        for _ in range(3):
            r_step = client.post("/mission/step", json={})
            assert r_step.status_code == 200
            step_body = r_step.json()
            assert step_body["status"] == "ok"
            frame = step_body["frame"]
            assert 0 <= frame["selected_band"] < 36
            assert 0 <= frame["selected_mode"] < 5
            
        # 4. Status
        r_status = client.get("/mission/status")
        assert r_status.status_code == 200
        assert r_status.json()["is_mission_active"] is True
        
        # 5. Stop
        r_stop = client.post("/mission/stop")
        assert r_stop.status_code == 200
        assert r_stop.json()["status"] == "mission_stopped"
        assert r_stop.json()["total_dwells"] == 3
