import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import json
from fastapi.testclient import TestClient
from ew_core.deployment.api import app, STATE
from ew_core.contracts import DEFAULT_DWELL_MULTIPLIERS, RF_BASE_DWELL_TIME_US

with TestClient(app) as client:
    # 1. Test /predict_bands
    obs = np.random.randn(360).tolist()
    resp = client.post('/predict_bands', json={'obs': obs, 'policy_mode': 'operational'})
    print('/predict_bands status code:', resp.status_code)
    data = resp.json()
    print('/predict_bands response:')
    print(json.dumps(data, indent=2))
    
    action = data['action']
    band = data.get('selected_band', data.get('band'))
    mode = data.get('selected_mode', data.get('mode'))
    dwell_us = data['dwell_time_us']
    prob = data['intercept_probability']
    pred_time_us = data['predicted_intercept_time_us']
    latency_ms = data['latency_ms']
    
    assert 0 <= action < 180, f'Action {action} out of range [0, 179]'
    assert 0 <= band < 36, f'Band {band} out of range [0, 35]'
    assert 0 <= mode < 5, f'Mode {mode} out of range [0, 4]'
    expected_dwell = RF_BASE_DWELL_TIME_US * DEFAULT_DWELL_MULTIPLIERS[mode]
    assert abs(dwell_us - expected_dwell) < 1e-3, f'Dwell time mismatch: {dwell_us} vs {expected_dwell}'
    assert 0.0 <= prob <= 1.0, f'Prob {prob} out of bounds'
    assert pred_time_us >= 0.0, f'pred_time_us {pred_time_us} negative'
    assert latency_ms > 0.0, f'latency {latency_ms} not positive'
    print('REAL INFERENCE VERIFICATION PASSED!')
    
    # 2. Test Mission Lifecycle: reset -> start -> step -> status -> stop
    print('--- Testing Mission Lifecycle ---')
    r_reset = client.post('/reset')
    print('/reset status:', r_reset.status_code)
    
    r_start = client.post('/mission/start', json={'initial_time_us': 0.0, 'auto_stream': False})
    print('/mission/start status:', r_start.status_code, r_start.json())
    
    # Execute several steps
    for step_i in range(3):
        r_step = client.post('/mission/step', json={})
        step_data = r_step.json()
        print(f"Step {step_i+1} status: {r_step.status_code}, action: {step_data.get('action')}, band: {step_data.get('selected_band')}, mode: {step_data.get('selected_mode')}, hit: {step_data.get('hit')}")
    
    r_status = client.get('/mission/status')
    print('/mission/status:', r_status.status_code, r_status.json().get('status'))
    
    r_metrics = client.get('/metrics')
    fc = r_metrics.json().get('frozen_candidate', {})
    print('/metrics frozen_candidate designation:', fc.get('designation'), 'mean_pd_pct:', fc.get('mean_pd_pct'))
    
    r_stop = client.post('/mission/stop')
    print('/mission/stop status:', r_stop.status_code, r_stop.json())
    print('MISSION LIFECYCLE VERIFICATION PASSED!')
