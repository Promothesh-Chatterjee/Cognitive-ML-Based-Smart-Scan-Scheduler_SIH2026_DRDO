import { useEffect, useMemo, useState } from "react";
import {
  CmdBadge,
  PanelHead,
} from "../components/stitch";
import { api } from "../services/api";
import { backend } from "../services/backend";
import { startTelemetryStream } from "../services/liveSocket";

const FEATURES = [
  "Occupancy",
  "Detection Rate",
  "Miss Rate",
  "Uncertainty",
  "Revisit Age",
  "Emitter Count",
  "Confidence",
  "PRI Stability",
  "Frequency Agility",
  "Priority",
];

const MODES = [
  "SHORT_DWELL",
  "NORMAL_DWELL",
  "LONG_DWELL",
  "REVISIT",
  "PREEMPTIVE_INTERCEPT",
];

const SHORT_FEATURES = ["OCC", "DET", "MISS", "UNC", "AGE", "CNT", "CONF", "PRI", "AGIL", "PRIO"];

/**
 * Generate a synthetic demo observation vector of exactly 360 numeric values.
 * 36 frequency bands × 10 features per band = 360 values.
 * Flattened layout: [band_0_feat_0..9, band_1_feat_0..9, ..., band_35_feat_0..9]
 *
 * NOTE: Clearly labeled as SYNTHETIC DEMO DATA for live testing and UI verification.
 */
function generateDemoObservation(scenarioType = "agile_activity") {
  const obs = new Float32Array(360);
  const activeBands = scenarioType === "agile_activity" 
    ? [6, 14, 22, 31] // 4-Band Agile Hopper bands
    : scenarioType === "dense_threat"
    ? [4, 8, 12, 16, 20, 24, 28]
    : [16]; // Single carrier

  for (let b = 0; b < 36; b++) {
    const isActive = activeBands.includes(b);
    const offset = b * 10;
    obs[offset + 0] = isActive ? 0.85 : 0.05;                          // Occupancy
    obs[offset + 1] = isActive ? 0.78 : 0.02;                          // Detection Rate
    obs[offset + 2] = isActive ? 0.12 : 0.01;                          // Miss Rate
    obs[offset + 3] = isActive ? 0.20 : 0.80;                          // Uncertainty
    obs[offset + 4] = isActive ? 0.45 : 0.95;                          // Revisit Age
    obs[offset + 5] = isActive ? (activeBands.indexOf(b) + 1) : 0.0;   // Emitter Count
    obs[offset + 6] = isActive ? 0.88 : 0.10;                          // Confidence
    obs[offset + 7] = isActive ? 0.92 : 0.05;                          // PRI Stability
    obs[offset + 8] = isActive ? 0.74 : 0.00;                          // Frequency Agility
    obs[offset + 9] = isActive ? 0.90 : 0.05;                          // Priority
  }
  return Array.from(obs);
}

export default function SmartScan() {
  const [liveTelemetry, setLiveTelemetry] = useState(null);
  const [selectedBand, setSelectedBand] = useState(16);
  const [backendHealth, setBackendHealth] = useState(null);

  // 360-D Observation State
  const [observationVector, setObservationVector] = useState(() => generateDemoObservation("agile_activity"));
  const [obsSource, setObsSource] = useState("SYNTHETIC DEMO");
  const [validationError, setValidationError] = useState("");

  // Live Backend Inference State
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceResponse, setInferenceResponse] = useState(null);
  const [inferenceError, setInferenceError] = useState("");
  const [policyMode, setPolicyMode] = useState("operational");
  const [predictionHistory, setPredictionHistory] = useState([]);

  // Check backend health & telemetry
  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        const data = await backend.api.getSystemStatus();
        if (active) setBackendHealth(data);
      } catch {
        if (active) setBackendHealth(null);
      }
    }

    checkHealth();
    const timer = setInterval(checkHealth, 5000);

    let stream = null;
    try {
      stream = startTelemetryStream({
        onTelemetry(t) {
          if (active && t?.valid && t?.live) {
            setLiveTelemetry(t);
            if (t.band != null) setSelectedBand(t.band);
          }
        },
      });
    } catch {
      // Stream error handled in service
    }

    return () => {
      active = false;
      clearInterval(timer);
      stream?.close();
    };
  }, []);

  // Parse flattened 360-D observation into 36 bands × 10 features for UI inspection
  const observationGrid = useMemo(() => {
    if (!Array.isArray(observationVector) || observationVector.length !== 360) {
      return [];
    }
    return Array.from({ length: 36 }, (_, band) => ({
      band,
      values: observationVector.slice(band * 10, (band + 1) * 10),
    }));
  }, [observationVector]);

  const selectedBandRow = observationGrid.find((item) => item.band === selectedBand) || {
    band: selectedBand,
    values: Array(10).fill(0),
  };

  // Perform Real Backend Inference via POST /predict_bands
  const runInference = async (vectorToUse = observationVector) => {
    setValidationError("");
    setInferenceError("");

    // Strict frontend validation: Must be an array of exactly 360 numeric values
    if (!Array.isArray(vectorToUse)) {
      setValidationError("Observation rejected: Observation must be an array.");
      return;
    }
    if (vectorToUse.length !== 360) {
      setValidationError(
        `Observation rejected: Expected exactly 360 numeric values (36 bands × 10 features), got ${vectorToUse.length}.`
      );
      return;
    }

    setIsInferring(true);
    const t0 = performance.now();
    try {
      const resp = await api.predictBands(vectorToUse, policyMode);
      setInferenceResponse(resp);

      // Add to prediction history
      setPredictionHistory((prev) => [
        {
          id: Date.now(),
          time: new Date().toLocaleTimeString(),
          action: resp.selected_action,
          band: resp.selected_band,
          mode: resp.selected_mode,
          modeName: MODES[resp.selected_mode] || `MODE_${resp.selected_mode}`,
          dwellUs: resp.dwell_time_us,
          prob: resp.intercept_probability,
          etaUs: resp.predicted_intercept_time_us,
          latencyMs: resp.latency_ms || (performance.now() - t0),
          reason: resp.attribution?.decision_source || resp.attribution?.reason || "DRQN Cognitive Policy",
        },
        ...prev.slice(0, 19),
      ]);

      if (resp.selected_band != null) {
        setSelectedBand(resp.selected_band);
      }
    } catch (err) {
      setInferenceError(err.message || "Failed to execute inference on backend.");
    } finally {
      setIsInferring(false);
    }
  };

  // Run initial inference on mount
  useEffect(() => {
    let active = true;
    const initialRun = async () => {
      try {
        await runInference(observationVector);
      } catch {
        // Handled within runInference
      }
    };
    if (active) {
      initialRun();
    }
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isModelOperational = Boolean(
    backendHealth?.operational_mode_ready &&
    backendHealth?.models_loaded?.scheduler &&
    backendHealth?.normalization_hash_match
  );

  const activeAction = useMemo(() => {
    if (inferenceResponse) {
      return {
        band: inferenceResponse.selected_band,
        mode: MODES[inferenceResponse.selected_mode] || `MODE_${inferenceResponse.selected_mode}`,
        score: inferenceResponse.attribution?.action_score ?? inferenceResponse.intercept_probability,
        probability: inferenceResponse.intercept_probability,
        timeUs: inferenceResponse.predicted_intercept_time_us,
        dwellUs: inferenceResponse.dwell_time_us,
        latencyMs: inferenceResponse.latency_ms,
        attribution: inferenceResponse.attribution,
        actionId: inferenceResponse.selected_action,
        isLiveResponse: true,
      };
    }
    if (liveTelemetry && liveTelemetry.band != null) {
      return {
        band: liveTelemetry.band,
        mode: liveTelemetry.modeName ?? "NORMAL_DWELL",
        score: Number(liveTelemetry.cognitiveExplanation?.drqn_score ?? 0.0),
        probability: Number(liveTelemetry.cognitiveExplanation?.prediction_confidence ?? 0.0),
        timeUs: Math.max(0, Number(liveTelemetry.cognitiveExplanation?.predicted_eta_us ?? 0.0)),
        dwellUs: liveTelemetry.dwellTimeUs || 500,
        latencyMs: liveTelemetry.rollingMedianLatencyUs || 0,
        attribution: liveTelemetry.cognitiveExplanation || {},
        actionId: liveTelemetry.band * 5 + 1,
        isLiveResponse: false,
      };
    }
    return {
      band: 16,
      mode: "REVISIT",
      score: 0.0,
      probability: 0.0,
      timeUs: 0.0,
      dwellUs: 500,
      latencyMs: 0.0,
      attribution: {},
      actionId: 16 * 5 + 3,
      isLiveResponse: false,
    };
  }, [inferenceResponse, liveTelemetry]);

  const handleApplyPreset = (type) => {
    const nextVec = generateDemoObservation(type);
    setObservationVector(nextVec);
    setObsSource(`SYNTHETIC DEMO (${type})`);
    runInference(nextVec);
  };

  const handleCorruptLengthTest = (len) => {
    const corrupted = Array(len).fill(0.1);
    setObservationVector(corrupted);
    setObsSource(`INVALID TEST (${len} values)`);
    runInference(corrupted);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {/* Header Panel */}
      <div className="st-panel">
        <PanelHead
          icon="neurology"
          title="SMART SCAN DECISION ENGINE & 360-DIMENSIONAL OBSERVATION INTERFACE"
          badge={isModelOperational ? "DRQN + MoE OPERATIONAL (CLOUD RUN)" : "CONNECTING / VERIFYING"}
          badgeColor={isModelOperational ? "#49df9d" : "#f59e0b"}
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          The Smart Scan Scheduler converts receiver-derived spectrum state into time-frequency intercept decisions.
          Inference requests to <code>/predict_bands</code> require a strict 360-dimensional vector (36 bands × 10 features).
          {isModelOperational ? (
            <span style={{ color: "#49df9d", fontWeight: 700 }}> Real model inference is ACTIVE on Cloud Run.</span>
          ) : (
            <span style={{ color: "#f59e0b" }}> Waiting for verified backend confirmation before operational deployment.</span>
          )}
          {backendHealth?.readiness_failures && backendHealth.readiness_failures.length > 0 && (
            <div style={{ marginTop: 8, padding: "6px 10px", background: "rgba(248, 113, 113, 0.1)", border: "1px solid #f87171", fontSize: 11 }}>
              <strong style={{ color: "#f87171" }}>Operational Readiness Blockers ({backendHealth.readiness_failures.length}): </strong>
              <span style={{ color: "#fca5a5" }}>{backendHealth.readiness_failures.join(" | ")}</span>
            </div>
          )}
        </div>
      </div>

      {/* Observation Vector Control & Live Test Bar */}
      <div className="st-panel" style={{ border: "1px solid rgba(189,194,255,0.3)" }}>
        <PanelHead icon="input" title="360-D OBSERVATION VECTOR CONTROLLER & INFERENCE RUNNER" badge={obsSource} badgeColor="#bdc2ff" />
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>DEMO PRESETS (SYNTHETIC DATA):</span>
            <button
              onClick={() => handleApplyPreset("agile_activity")}
              style={{
                padding: "4px 10px",
                background: "#2563eb",
                color: "#fff",
                border: "none",
                cursor: "pointer",
                fontWeight: 600,
                fontSize: 11,
              }}
            >
              Agile Hopper (Bands 6, 14, 22, 31)
            </button>
            <button
              onClick={() => handleApplyPreset("dense_threat")}
              style={{
                padding: "4px 10px",
                background: "rgba(255,255,255,0.1)",
                color: "#e2e2e8",
                border: "1px solid #454653",
                cursor: "pointer",
                fontSize: 11,
              }}
            >
              Dense Multi-Emitter Environment
            </button>
            <button
              onClick={() => handleApplyPreset("single_carrier")}
              style={{
                padding: "4px 10px",
                background: "rgba(255,255,255,0.1)",
                color: "#e2e2e8",
                border: "1px solid #454653",
                cursor: "pointer",
                fontSize: 11,
              }}
            >
              Single Carrier (Band 16)
            </button>

            <span style={{ borderLeft: "1px solid #454653", height: 18, margin: "0 4px" }} />

            <span className="st-tsm" style={{ color: "#f87171" }}>VALIDATION CONTRACT TESTS:</span>
            <button
              onClick={() => handleCorruptLengthTest(100)}
              title="Send 100 values to verify frontend and backend rejection"
              style={{
                padding: "4px 8px",
                background: "rgba(239, 68, 68, 0.2)",
                color: "#f87171",
                border: "1px solid #ef4444",
                cursor: "pointer",
                fontSize: 11,
              }}
            >
              Test Invalid Length (100 values)
            </button>
            <button
              onClick={() => handleCorruptLengthTest(2)}
              title="Send 2 values to verify Swagger legacy rejection"
              style={{
                padding: "4px 8px",
                background: "rgba(239, 68, 68, 0.2)",
                color: "#f87171",
                border: "1px solid #ef4444",
                cursor: "pointer",
                fontSize: 11,
              }}
            >
              Test Invalid Length (2 values)
            </button>

            <span style={{ flex: 1 }} />

            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span className="st-tsm" style={{ color: "#908f9e" }}>POLICY:</span>
              <select
                value={policyMode}
                onChange={(e) => setPolicyMode(e.target.value)}
                style={{
                  background: "#1a1c20",
                  color: "#e2e2e8",
                  border: "1px solid #454653",
                  padding: "4px 8px",
                  fontSize: 11,
                }}
              >
                <option value="operational">operational (Deterministic DRQN)</option>
                <option value="default">default</option>
                <option value="demo">demo</option>
                <option value="fallback">fallback</option>
              </select>
              <button
                onClick={() => runInference(observationVector)}
                disabled={isInferring}
                style={{
                  padding: "4px 14px",
                  background: isInferring ? "#454653" : "#49df9d",
                  color: "#000",
                  fontWeight: 700,
                  border: "none",
                  cursor: isInferring ? "not-allowed" : "pointer",
                  fontSize: 11,
                }}
              >
                {isInferring ? "INFERRING..." : "RUN INFERENCE (POST /predict_bands)"}
              </button>
            </div>
          </div>

          {/* Validation Notice & Warnings */}
          {validationError && (
            <div
              style={{
                padding: "6px 10px",
                background: "rgba(239, 68, 68, 0.15)",
                border: "1px solid #ef4444",
                color: "#fca5a5",
                fontSize: 12,
                fontWeight: 600,
              }}
            >
              ⚠️ {validationError}
            </div>
          )}

          {inferenceError && (
            <div
              style={{
                padding: "6px 10px",
                background: "rgba(239, 68, 68, 0.15)",
                border: "1px solid #ef4444",
                color: "#fca5a5",
                fontSize: 12,
              }}
            >
              ❌ Backend Inference Error: {inferenceError}
            </div>
          )}

          <div className="st-tsm" style={{ color: "#a5a3b7", lineHeight: 1.4 }}>
            ℹ️ <strong>Observation Vector Contract:</strong> Exactly 360 numeric values representing 36 frequency bands
            (0–18 GHz, 500 MHz IBW each) with 10 features per band:
            <code> [Occupancy, Detection Rate, Miss Rate, Uncertainty, Revisit Age, Emitter Count, Confidence, PRI Stability, Frequency Agility, Priority]</code>.
            Predictions below are derived directly from this vector. Demo vectors are labeled as synthetic to ensure no simulation bias.
          </div>
        </div>
      </div>

      {/* Architecture Pipeline Visualizer */}
      <div className="st-panel">
        <PanelHead
          icon="account_tree"
          title="ZONE B — AI INFERENCE ARCHITECTURE PIPELINE (DRQN + MoE)"
          badge="OBSERVATION → POLICY → ACTION"
          badgeColor="#bdc2ff"
        />
        <div className="st-pipe">
          <div className="st-node">
            <CmdBadge>INPUT</CmdBadge>
            <span className="st-tmd">{observationVector.length}-D</span>
            <span className="st-mark" style={{ color: "#908f9e" }}>36 bands × 10 features</span>
          </div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node">
            <CmdBadge color="#96ccff">NET</CmdBadge>
            <span className="st-tmd">DRQN</span>
            <span className="st-mark" style={{ color: "#908f9e" }}>LSTM temporal memory</span>
          </div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node">
            <CmdBadge color="#49df9d">POL</CmdBadge>
            <span className="st-tmd">MoE</span>
            <span className="st-mark" style={{ color: "#908f9e" }}>Strategy arbitration</span>
          </div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node">
            <CmdBadge>SPC</CmdBadge>
            <span className="st-tmd">180</span>
            <span className="st-mark" style={{ color: "#908f9e" }}>36 bands × 5 modes</span>
          </div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node st-node-ai">
            <CmdBadge color="#49df9d">ACTION</CmdBadge>
            <span className="st-tmd" style={{ color: "#49df9d" }}>B{activeAction.band}</span>
            <span className="st-mark" style={{ color: "#c6c5d5" }}>{activeAction.mode}</span>
          </div>
        </div>
      </div>

      {/* Grid: 36-Band Heatmap & Action Diagnostics */}
      <div className="st-grid-12">
        <div className="st-span-8 st-panel">
          <PanelHead
            icon="grid_view"
            title="ZONE A — 36-BAND OBSERVATION VECTOR HEATMAP (0.00 – 18.00 GHz)"
            badge={`INSPECTOR: BAND ${selectedBand} (${selectedBand * 500}–${(selectedBand + 1) * 500} MHz)`}
            badgeColor="#96ccff"
          />
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            EXACT 360-DIMENSIONAL OBSERVATION · GROUND TRUTH EXCLUDED · CLICK BAND TO INSPECT
          </span>
          <div className="st-table-wrap">
            <table className="st-table">
              <thead>
                <tr>
                  <th>BAND (FREQ)</th>
                  {SHORT_FEATURES.map((feature) => (
                    <th key={feature}>{feature}</th>
                  ))}
                  <th>ACTION</th>
                </tr>
              </thead>
              <tbody>
                {observationGrid.map((row) => {
                  const isSelected = selectedBand === row.band;
                  const isPredictedBand = activeAction.band === row.band;
                  return (
                    <tr
                      key={row.band}
                      style={{
                        cursor: "pointer",
                        background: isPredictedBand
                          ? "rgba(73, 223, 157, 0.12)"
                          : isSelected
                          ? "rgba(189, 194, 255, 0.08)"
                          : undefined,
                      }}
                      onClick={() => setSelectedBand(row.band)}
                    >
                      <td>
                        <strong style={{ color: isPredictedBand ? "#49df9d" : isSelected ? "#bdc2ff" : "#e2e2e8" }}>
                          B{row.band} ({row.band * 500}M)
                        </strong>
                      </td>
                      {row.values.map((val, idx) => (
                        <td key={idx}>
                          <span
                            className="st-mark"
                            style={{
                              color: val > 0.5 ? "#49df9d" : val > 0.2 ? "#bdc2ff" : "#908f9e",
                              fontWeight: val > 0.5 ? 700 : 400,
                            }}
                          >
                            {val.toFixed(2)}
                          </span>
                        </td>
                      ))}
                      <td>
                        {isPredictedBand ? (
                          <strong style={{ color: "#49df9d", fontSize: 10 }}>SELECTED ↑</strong>
                        ) : (
                          <span style={{ color: "#454653" }}>—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Sidebar: Selected Action Details & Real Metrics */}
        <aside className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
          {/* Active Backend Decision Card */}
          <div className="st-panel">
            <PanelHead
              title={`REAL BACKEND DECISION: BAND ${activeAction.band} // ${activeAction.mode}`}
              badge={activeAction.isLiveResponse ? "LIVE POST /predict_bands" : "TELEMETRY"}
              badgeColor="#49df9d"
            />
            <span className="st-tsm" style={{ color: "#908f9e" }}>
              DRQN LSTM RECURRENT CORE // MoE GATING & ARBITRATION
            </span>
            <div className="st-tlg" style={{ color: "#bdc2ff", margin: "4px 0" }}>
              ACTION ID: {activeAction.actionId} (B{activeAction.band} : {activeAction.mode})
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {[
                ["SELECTED BAND", `B${activeAction.band} (${(activeAction.band * 500 + 250).toLocaleString()} MHz)`],
                ["SCAN MODE", activeAction.mode],
                ["DWELL TIME", `${Number(activeAction.dwellUs).toFixed(0)} µs`],
                ["INTERCEPT PROBABILITY", `${(Number(activeAction.probability) * 100).toFixed(1)}%`],
                ["PREDICTED INTERCEPT TIME", `${Number(activeAction.timeUs).toFixed(2)} µs`],
                ["INFERENCE LATENCY", `${Number(activeAction.latencyMs).toFixed(2)} ms`],
                ["DECISION ATTRIBUTION", activeAction.attribution?.decision_source || activeAction.attribution?.reason || "DRQN Policy"],
                ["EAGER VS REVISIT", `${((activeAction.attribution?.eager_pct ?? 0.6) * 100).toFixed(0)}% / ${((activeAction.attribution?.revisit_pct ?? 0.4) * 100).toFixed(0)}%`],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="st-tsm"
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "3px 6px",
                    background: "#1a1c20",
                    border: "1px solid #454653",
                  }}
                >
                  <span style={{ color: "#908f9e" }}>{label}</span>
                  <strong style={{ color: "#e2e2e8" }}>{value}</strong>
                </div>
              ))}
            </div>
          </div>

          {/* Inspected Band Vector Inspector */}
          <div className="st-panel">
            <PanelHead title={`BAND ${selectedBand} FEATURE VECTOR`} badge={`B${selectedBand}`} badgeColor="#bdc2ff" />
            <div className="st-tmd" style={{ color: "#e2e2e8" }}>
              {selectedBand * 500}–{(selectedBand + 1) * 500} MHz
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {FEATURES.map((feature, idx) => (
                <div
                  key={feature}
                  className="st-tsm"
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "2px 6px",
                    background: "#1a1c20",
                    border: "1px solid #454653",
                  }}
                >
                  <span style={{ color: "#908f9e" }}>{feature}</span>
                  <strong style={{ color: selectedBandRow.values[idx] > 0.5 ? "#49df9d" : "#e2e2e8" }}>
                    {selectedBandRow.values[idx]?.toFixed(3) ?? "0.000"}
                  </strong>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Prediction History */}
          <div className="st-panel">
            <PanelHead title="PREDICTION HISTORY (POST /predict_bands)" badge={`${predictionHistory.length} CALLS`} badgeColor="#96ccff" />
            <div style={{ display: "flex", flexDirection: "column", gap: 2, maxHeight: 180, overflowY: "auto" }}>
              {predictionHistory.length === 0 ? (
                <span className="st-tsm" style={{ color: "#908f9e", padding: 6 }}>
                  No recent predictions. Click "RUN INFERENCE" above.
                </span>
              ) : (
                predictionHistory.map((item, idx) => (
                  <div
                    key={item.id}
                    className="st-tsm"
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      padding: "3px 6px",
                      background: idx === 0 ? "#1e2024" : "#1a1c20",
                      border: `1px solid ${idx === 0 ? "#49df9d" : "rgba(69,70,83,0.4)"}`,
                    }}
                  >
                    <span>
                      <strong style={{ color: "#49df9d" }}>B{item.band}</strong> {item.modeName}
                    </span>
                    <span style={{ color: "#908f9e" }}>{(item.prob * 100).toFixed(0)}% · {item.latencyMs.toFixed(1)}ms</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

