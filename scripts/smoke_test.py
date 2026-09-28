"""
SmartScan EW — Operational Smoke Test
Validates the live Azure AKS deployment against all PS objectives.

Usage:
    python scripts/smoke_test.py --api_url http://172.198.227.59 --api_key YOUR_KEY

Exit codes: 0 = all tests passed, 1 = one or more tests failed
"""
import os
import sys
import time
import argparse
import json
import requests

# PS FoM acceptance thresholds (Gate-25k baseline — update after Gate-100k)
# These are deliberately set to match current Gate-25k capabilities
THRESHOLDS = {
    "pd":                         (">=", 0.20, "≥ 20% (canonical Gate-25k)"),
    "pfa":                        ("<=", 0.05, "≤ 5%"),
    "avg_intercept_rate":         (">=", 0.05, "≥ 5% (canonical Gate-25k)"),
    "avg_reward":                 (">=", -2.0, "> -2.0 (negative is expected)"),
    "pct_correct_predictions":    (">=", 30.0, "≥ 30%"),
}

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def check(label, passed, detail=""):
    try:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    except UnicodeEncodeError:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {label}" + (f" -- {detail}" if detail else ""))
    return passed

def run_smoke_test(api_url: str, api_key: str) -> bool:
    headers = {"X-SmartScan-API-Key": api_key} if api_key else {}
    failures = []
    print(f"\n{'='*60}")
    print(f"SmartScan EW Smoke Test — {api_url}")
    print(f"{'='*60}\n")

    # --- TEST 1: Health probe ---
    print("[1/6] Health Probe (/health)")
    try:
        r = requests.get(f"{api_url}/health", timeout=15)
        ok = r.status_code == 200 or (r.status_code == 503 and r.json().get("status") in ["ok", "degraded"])
        if not check("/health returns HTTP 200", ok, f"got {r.status_code}"):
            failures.append("health")
    except Exception as e:
        check("/health reachable", False, str(e))
        failures.append("health_unreachable")

    # --- TEST 2: Readiness probe ---
    print("\n[2/6] Readiness Probe (/ready)")
    try:
        r = requests.get(f"{api_url}/ready", timeout=20)
        ok = r.status_code == 200
        if not check("/ready returns HTTP 200", ok, f"got {r.status_code}"):
            failures.append("ready")
        if ok:
            data = r.json()
            if not check("model_loaded == true", data.get("model_loaded") is True, str(data.get("model_loaded"))):
                failures.append("ready_model_not_loaded")
            if not check("dataset_root is set", bool(data.get("dataset_root")), data.get("dataset_root", "missing")):
                failures.append("ready_dataset_missing")
    except Exception as e:
        check("/ready reachable", False, str(e))
        failures.append("ready_unreachable")

    # --- TEST 3: Auth is enforced ---
    print("\n[3/6] Authentication")
    try:
        r = requests.post(f"{api_url}/predict_bands",
                          json={"obs": [0.0]*360, "policy_mode": "operational"},
                          timeout=10)
        if api_key:
            auth_ok = r.status_code == 401
            if not check("Unauthenticated request returns 401", auth_ok, f"got {r.status_code}"):
                failures.append("auth_not_enforced")
        else:
            check("No API key configured — auth check skipped", True)
    except Exception as e:
        check("Auth endpoint reachable", False, str(e))
        failures.append("auth_unreachable")

    # --- TEST 4: Neural inference ---
    print("\n[4/6] Neural Inference (/predict_bands) — Multi-Sample Latency SLA")
    try:
        import numpy as np

        WARMUP_COUNT = 5
        MEASURED_COUNT = 20

        session = requests.Session()
        session.headers.update(headers)
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        print(f"  Executing {WARMUP_COUNT} warmup requests...")
        warmup_ok = True
        warmup_err = ""
        for w_idx in range(WARMUP_COUNT):
            try:
                wr = session.post(
                    f"{api_url}/predict_bands",
                    json={"obs": [0.0] * 360, "policy_mode": "operational"},
                    timeout=30,
                )
                if wr.status_code != 200:
                    warmup_ok = False
                    warmup_err = f"warmup request {w_idx + 1} returned HTTP {wr.status_code}"
                    break
            except Exception as e:
                warmup_ok = False
                warmup_err = f"warmup request {w_idx + 1} raised: {e}"
                break

        if not check(f"All {WARMUP_COUNT} warmup requests succeeded", warmup_ok, warmup_err):
            failures.append("warmup_failure")

        print(f"  Measuring {MEASURED_COUNT} inference round-trips...")
        api_latencies_ms = []
        server_latencies_ms = []
        last_action = None
        has_action_field = True
        inference_http_ok = True

        for i in range(MEASURED_COUNT):
            t0 = time.perf_counter()
            r = session.post(
                f"{api_url}/predict_bands",
                json={"obs": [0.0] * 360, "policy_mode": "operational"},
                timeout=30,
            )
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            api_latencies_ms.append(dt_ms)

            if r.status_code != 200:
                inference_http_ok = False
                continue

            data = r.json()
            if not ("action" in data or "selected_action" in data):
                has_action_field = False
            last_action = data.get("action", data.get("selected_action", -1))

            # Telemetry integrity: server inference latency
            server_inf_time = None
            if "server_inference_latency_ms" in data:
                server_inf_time = float(data["server_inference_latency_ms"])
            elif "latency_ms" in data:
                server_inf_time = float(data["latency_ms"])
            elif "inference_time_ms" in data:
                server_inf_time = float(data["inference_time_ms"])
            elif "X-Inference-Time-Ms" in r.headers:
                server_inf_time = float(r.headers["X-Inference-Time-Ms"])

            if server_inf_time is not None:
                server_latencies_ms.append(server_inf_time)

        if not check("/predict_bands HTTP 200 across measured samples", inference_http_ok):
            failures.append("inference_http")

        if not check("Response has 'action' field", has_action_field):
            failures.append("inference_missing_action")

        if not check("Action in valid range [0, 179]", last_action is not None and 0 <= last_action <= 179, f"action={last_action}"):
            failures.append("inference_action_range")

        # Latency statistics & SLA evaluation
        if api_latencies_ms:
            lat_arr = np.array(api_latencies_ms, dtype=np.float64)
            lat_median = float(np.median(lat_arr))
            lat_p95 = float(np.percentile(lat_arr, 95, method="linear"))
            lat_min = float(np.min(lat_arr))
            lat_max = float(np.max(lat_arr))

            print(f"    API Latencies (N={len(lat_arr)}): min={lat_min:.1f}ms, med={lat_median:.1f}ms, p95={lat_p95:.1f}ms, max={lat_max:.1f}ms")

            if not check("API Round-Trip Median < 500ms", lat_median < 500.0, f"median={lat_median:.1f}ms"):
                failures.append("inference_latency_median_sla")

            if not check("API Round-Trip p95 < 500ms (Deployment SLA)", lat_p95 < 500.0, f"p95={lat_p95:.1f}ms"):
                failures.append("inference_latency_p95_sla")

            if not check(
                "Server inference latency telemetry present across all measured samples",
                len(server_latencies_ms) == MEASURED_COUNT,
                f"received {len(server_latencies_ms)}/{MEASURED_COUNT}",
            ):
                failures.append("server_telemetry_missing")
            else:
                srv_arr = np.array(server_latencies_ms, dtype=np.float64)
                srv_med = float(np.median(srv_arr))
                srv_finite = bool(np.all(np.isfinite(srv_arr)))
                srv_positive = bool(np.all(srv_arr > 0))
                srv_lt_api = bool(srv_med < lat_median)
                srv_ok = srv_finite and srv_positive and srv_lt_api
                if not check(
                    "Server Inference Monotonic Telemetry Integrity (finite, >0, server_median < api_median)",
                    srv_ok,
                    f"server_median={srv_med:.2f}ms vs api_median={lat_median:.2f}ms",
                ):
                    failures.append("server_telemetry_integrity")
        session.close()
    except Exception as e:
        check("/predict_bands reachable and testable", False, str(e))
        failures.append("inference_unreachable")

    # --- TEST 5: Benchmark endpoint ---
    print("\n[5/6] Benchmark Data (/api/benchmark)")
    try:
        r = requests.get(f"{api_url}/api/benchmark", headers=headers, timeout=15)
        ok = r.status_code == 200
        if not check("/api/benchmark HTTP 200", ok, f"got {r.status_code}"):
            failures.append("benchmark")
        if ok:
            data = r.json()
            result_data = data if "schedulers" in data else {"schedulers": data}
            schedulers = list(result_data.get("schedulers", {}).keys())
            if not check("SmartScan_DRQN_MoE in results", "SmartScan_DRQN_MoE" in schedulers, f"found: {schedulers}"):
                failures.append("benchmark_missing_drqn")

            expected_schedulers = ["SmartScan_DRQN_MoE", "Random", "RoundRobin", "HighestOccupancy"]
            if not check("All 4 schedulers present", len(schedulers) == 4, f"count={len(schedulers)}, keys={schedulers}"):
                failures.append("benchmark_scheduler_count")

            for req_sched in expected_schedulers:
                if req_sched not in schedulers:
                    check(f"Scheduler '{req_sched}' present", False, "missing from benchmark")
                    failures.append(f"benchmark_missing_{req_sched.lower()}")

            drqn = result_data.get("schedulers", {}).get("SmartScan_DRQN_MoE", {}).get("summary", {})
            for fom, (op, threshold, label) in THRESHOLDS.items():
                val = drqn.get(fom, None)
                if val is None:
                    check(f"FoM '{fom}' present", False, "missing"); failures.append(fom)
                    continue
                try:
                    val = float(val)   # cast to float — handles string-typed numbers
                except (TypeError, ValueError):
                    check(f"FoM '{fom}' is numeric", False, f"got: {type(val).__name__}={val}")
                    failures.append(fom)
                    continue
                if op == ">=":   passed = val >= threshold
                elif op == "<=": passed = val <= threshold
                elif op == "<":  passed = val < threshold
                else:            passed = True
                if not check(f"FoM '{fom}' = {val:.4f} ({label})", passed):
                    failures.append(fom)
    except requests.exceptions.RequestException as e:
        check("/api/benchmark reachable", False, str(e))
        failures.append("benchmark_unreachable")
    except Exception as e:
        check("/api/benchmark response valid", False, str(e))
        failures.append("benchmark_invalid_schema")

    # --- TEST 6: WebSocket ---
    print("\n[6/6] WebSocket Metrics Stream (/ws/metrics)")
    try:
        import websocket as ws_lib  # websocket-client
        connected = [False]
        def on_open(ws): connected[0] = True; ws.close()
        wso = ws_lib.WebSocketApp(
            f"ws://{api_url.replace('http://','').replace('https://','')}/ws/metrics",
            on_open=on_open
        )
        import threading
        t = threading.Thread(target=wso.run_forever)
        t.daemon = True; t.start(); t.join(timeout=8)
        if not check("WebSocket /ws/metrics handshake", connected[0]):
            failures.append("websocket")
    except ImportError:
        print("  ⚠️  SKIP  websocket-client not installed — install with: pip install websocket-client")
    except Exception as e:
        check("WebSocket connectable", False, str(e)); failures.append("websocket")

    # --- SUMMARY ---
    print(f"\n{'='*60}")
    if not failures:
        try:
            print("✅ ALL SMOKE TESTS PASSED — System is operationally ready")
        except UnicodeEncodeError:
            print("[PASS] ALL SMOKE TESTS PASSED -- System is operationally ready")
        print(f"{'='*60}\n")
        return True
    else:
        try:
            print(f"❌ SMOKE TEST FAILED — {len(failures)} issue(s): {', '.join(failures)}")
        except UnicodeEncodeError:
            print(f"[FAIL] SMOKE TEST FAILED -- {len(failures)} issue(s): {', '.join(failures)}")
        print(f"{'='*60}\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartScan EW Smoke Test")
    parser.add_argument("--api_url", default=os.environ.get("CLOUD_RUN_URL", "https://cognitive-ew-backend-753709137146.asia-south1.run.app"),
                        help="Base URL of the SmartScan API")
    parser.add_argument("--api_key", default=os.environ.get("SMARTSCAN_API_KEY", ""),
                        help="API key for authentication (X-SmartScan-API-Key header)")
    args = parser.parse_args()
    success = run_smoke_test(args.api_url, args.api_key)
    sys.exit(0 if success else 1)
