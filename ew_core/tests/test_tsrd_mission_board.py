"""Test suite for TSRD Mission Board and Dynamic Scenario Streaming.

Verifies:
  1. GET /mission/scenarios returns comprehensive scenario catalog.
  2. Canonical benchmark scenarios are prioritized and classified deterministically.
  3. POST /mission/start resolves and accepts real TSRD H5 scenario (e.g. config_119).
  4. POST /mission/start fails closed on missing TSRD scenario (no synthetic fallback).
  5. OperationalReceiverController and frozen Gate-27 scheduler execute mission steps.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ew_core.deployment.api import app
from ew_core.deployment.dataset_service import (
    CANONICAL_BENCHMARK_CONFIGS,
    list_tsrd_scenarios,
    resolve_scenario_path,
)


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized FastAPI lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_tsrd_scenarios_catalog_endpoint(client: TestClient):
    """Verify GET /mission/scenarios returns catalog with benchmark configs."""
    resp = client.get("/mission/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 10

    # Ensure all 10 canonical benchmark configs are present and marked benchmark=True
    bench_ids = {s["id"] for s in data if s.get("benchmark")}
    expected_benchmarks = {b["id"] for b in CANONICAL_BENCHMARK_CONFIGS}
    assert expected_benchmarks.issubset(bench_ids)

    # Verify config_119 has expected metadata
    cfg119 = next((s for s in data if s["id"] == "config_119"), None)
    assert cfg119 is not None
    assert cfg119["class"] == "sparse"
    assert cfg119["display_class"] == "Sparse Agile"
    assert cfg119["benchmark"] is True
    assert cfg119["source"] == "tsrd"


def test_tsrd_mission_start_config119(client: TestClient):
    """Verify POST /mission/start successfully initializes with config_119."""
    # Check if config_119.h5 is resolvable on this environment
    resolved = resolve_scenario_path("config_119")
    if resolved is None:
        pytest.skip("config_119.h5 not found in local TSRD environment")

    resp = client.post(
        "/mission/start",
        json={
            "scenario": "config_119",
            "speed_hz": 15.0,
            "max_dwells": 100,
            "auto_stream": False,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "mission_started"
    assert payload["mission_active"] is True
    assert payload["scenario"] == "config_119"
    assert "config_119" in payload["scenario_path"]


def test_tsrd_mission_fail_closed_on_missing_config(client: TestClient):
    """Verify POST /mission/start fails closed (HTTP 404) for non-existent TSRD scenario."""
    resp = client.post(
        "/mission/start",
        json={
            "scenario": "config_99999_nonexistent_threat",
            "speed_hz": 15.0,
            "max_dwells": 100,
            "auto_stream": False,
        },
    )
    assert resp.status_code == 404
    detail = resp.json().get("detail", "")
    assert "not found" in detail.lower()
    assert "fail-closed" in detail.lower()


def test_tsrd_mission_step_execution(client: TestClient):
    """Verify POST /mission/step runs a real operational dwell cycle with Gate-27."""
    # Start mission first
    client.post("/mission/start", json={"scenario": "config_119", "auto_stream": False})

    # Step with physical PDWs
    sample_pdws = [
        {
            "toa_us": 50.0,
            "time_us": 50.0,
            "frequency_mhz": 3500.0,
            "pulse_width_us": 2.5,
            "amplitude_db": -50.0,
            "aoa_deg": 12.0,
        },
        {
            "toa_us": 120.0,
            "time_us": 120.0,
            "frequency_mhz": 3510.0,
            "pulse_width_us": 2.5,
            "amplitude_db": -52.0,
            "aoa_deg": 12.0,
        },
    ]

    resp = client.post("/mission/step", json={"pdws": sample_pdws})
    assert resp.status_code == 200
    step_data = resp.json()
    assert step_data.get("status") == "ok"
    frame = step_data.get("frame", {})
    assert "selected_band" in frame
    assert "mode_name" in frame
    assert "center_frequency_mhz" in frame
