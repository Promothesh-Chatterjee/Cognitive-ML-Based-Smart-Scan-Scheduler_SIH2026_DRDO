import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
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
const POLICY_MODE = "operational";

export default function SmartScan() {
  const [selectedBand, setSelectedBand] = useState(16);
  const [backendHealth, setBackendHealth] = useState(null);

  // 360-D Observation State: null until a valid real observation arrives from live telemetry
  const [observationVector, setObservationVector] = useState(null);
  const [obsSource, setObsSource] = useState("AWAITING LIVE 360-D OBSERVATION");
  const [validationError, setValidationError] = useState("");

  // Live Backend Inference State
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceResponse, setInferenceResponse] = useState(null);
  const [inferenceError, setInferenceError] = useState("");
  const [predictionHistory, setPredictionHistory] = useState([]);

  // Deduplication, in-flight lock, and trailing frame buffer
  const lastInferredFrameKeyRef = useRef(null);
  const isInferringRef = useRef(false);
  const pendingFrameRef = useRef(null);

  // Parse flattened 360-D observation into 36 bands × 10 features for UI inspection
  const observationGrid = useMemo(() => {
    if (!Array.isArray(observationVector) || observationVector.length !== 360) {
      return Array.from({ length: 36 }, (_, band) => ({
        band,
        values: Array(10).fill(0),
      }));
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

  // Perform Real Backend Inference via POST /predict_bands (100% stable reference: [] deps)
  const runInference = useCallback(async (initialVector) => {
    setValidationError("");
    setInferenceError("");

    let currentVector = initialVector;
    setIsInferring(true);
    isInferringRef.current = true;

    try {
      while (currentVector) {
        if (!Array.isArray(currentVector) || currentVector.length !== 360) {
          setValidationError(
            `Observation rejected: Expected exactly 360 numeric values (36 bands × 10 features), got ${currentVector?.length}.`
          );
          break;
        }

        const t0 = performance.now();
        const resp = await api.predictBands(currentVector, POLICY_MODE);
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

        // Drain trailing pending frame if one arrived during this inference
        if (pendingFrameRef.current) {
          const next = pendingFrameRef.current;
          pendingFrameRef.current = null;
          if (next.frameKey !== lastInferredFrameKeyRef.current) {
            lastInferredFrameKeyRef.current = next.frameKey;
            currentVector = next.obs;
            continue;
          }
        }
        currentVector = null;
      }
    } catch (err) {
      setInferenceError(err.message || "Failed to execute inference on backend.");
    } finally {
      setIsInferring(false);
      isInferringRef.current = false;
    }
  }, []);

  // Check backend health & listen to telemetry stream with authoritative frame deduplication
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
          if (!active || !t?.valid || !t?.live) return;

          if (t.band != null) setSelectedBand(t.band);

          // Extract observation vector from live telemetry
          const obs = t.observation || t.raw?.observation || t.metrics?.observation;
          const isValid360 =
            Array.isArray(obs) &&
            obs.length === 360 &&
            obs.every((v) => Number.isFinite(Number(v)));

          if (isValid360) {
            setObservationVector(obs);
            setObsSource("LIVE TELEMETRY (CLOUD RUN)");

            // Authoritative frame key: step, clockUs, or timestamp
            const frameKey =
              t.step != null
                ? `step-${t.step}`
                : t.clockUs != null && t.clockUs > 0
                ? `clock-${t.clockUs}`
                : t.raw?.timestamp
                ? `ts-${t.raw.timestamp}`
                : `obs-${obs[0]}-${obs[180]}-${obs[359]}`;

            // If identical to last inferred frame, ignore duplicate poll
            if (frameKey === lastInferredFrameKeyRef.current) {
              return;
            }

            // If an inference is currently in flight, record this newest frame as pending
            if (isInferringRef.current) {
              pendingFrameRef.current = { frameKey, obs };
            } else {
              // Otherwise, fire inference immediately
              pendingFrameRef.current = null;
              lastInferredFrameKeyRef.current = frameKey;
              runInference(obs);
            }
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
  }, [runInference]);

  const isModelOperational = Boolean(
    backendHealth?.operational_mode_ready &&
    backendHealth?.models_loaded?.scheduler &&
    backendHealth?.normalization_hash_match
  );

  // Authoritative action: derived strictly from real inference response, null if awaiting data
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
    return null;
  }, [inferenceResponse]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {/* Header Panel with Operational Inference Controls */}
      <div className="st-panel">
        <PanelHead
          icon="neurology"
          title="SMART SCAN DECISION ENGINE & 360-DIMENSIONAL OBSERVATION INTERFACE"
          badge={isModelOperational ? "DRQN + MoE OPERATIONAL (CLOUD RUN)" : "CONNECTING / VERIFYING"}
          badgeColor={isModelOperational ? "var(--success, #49df9d)" : "var(--warning, #f59e0b)"}
        />
        <div className="st-body" style={{ color: "var(--text, #c6c5d5)" }}>
          The Smart Scan Scheduler converts receiver-derived spectrum state into time-frequency intercept decisions.
          Inference requests to <code>/predict_bands</code> require a strict 360-dimensional vector (36 bands × 10 features).
          {isModelOperational ? (
            <span style={{ color: "var(--success, #49df9d)", fontWeight: 700 }}> Real model inference is ACTIVE on Cloud Run.</span>
          ) : (
            <span style={{ color: "var(--warning, #f59e0b)" }}> Waiting for verified backend confirmation before operational deployment.</span>
          )}
          {backendHealth?.readiness_failures && backendHealth.readiness_failures.length > 0 && (
            <div style={{ marginTop: 8, padding: "6px 10px", background: "rgba(248, 113, 113, 0.1)", border: "1px solid var(--danger, #f87171)", fontSize: 11 }}>
              <strong style={{ color: "var(--danger, #f87171)" }}>Operational Readiness Blockers ({backendHealth.readiness_failures.length}): </strong>
              <span style={{ color: "var(--danger, #fca5a5)" }}>{backendHealth.readiness_failures.join(" | ")}</span>
            </div>
          )}

          {/* Operational Action Bar: Input Source & Inference Status */}
          <div style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--border-subtle, #333539)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>INPUT SOURCE:</span>
              <span className="st-badge" style={{ color: "var(--accent, #bdc2ff)", background: "var(--panel-3, #282a2e)", border: "1px solid var(--border, #454653)" }}>
                {obsSource} (360-D)
              </span>
            </div>
            {isInferring && (
              <span className="st-badge" style={{ color: "var(--accent, #bdc2ff)", background: "rgba(189, 194, 255, 0.15)", border: "1px solid var(--accent, #bdc2ff)" }}>
                INFERRING (POST /predict_bands)...
              </span>
            )}
          </div>

          {/* Validation Notice & Warnings */}
          {validationError && (
            <div
              style={{
                marginTop: 8,
                padding: "6px 10px",
                background: "rgba(239, 68, 68, 0.15)",
                border: "1px solid var(--danger, #ef4444)",
                color: "var(--danger, #fca5a5)",
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
                marginTop: 8,
                padding: "6px 10px",
                background: "rgba(239, 68, 68, 0.15)",
                border: "1px solid var(--danger, #ef4444)",
                color: "var(--danger, #fca5a5)",
                fontSize: 12,
              }}
            >
              ❌ Backend Inference Error: {inferenceError}
            </div>
          )}
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
                  const isPredictedBand = activeAction?.band != null && activeAction.band === row.band;
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
              title={activeAction ? `REAL BACKEND DECISION: BAND ${activeAction.band} // ${activeAction.mode}` : "AWAITING LIVE 360-D OBSERVATION"}
              badge={activeAction ? "LIVE POST /predict_bands" : "AWAITING INFERENCE"}
              badgeColor={activeAction ? "var(--success, #49df9d)" : "var(--warning, #f59e0b)"}
            />
            <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
              DRQN LSTM RECURRENT CORE // MoE GATING & ARBITRATION
            </span>
            <div className="st-tlg" style={{ color: activeAction ? "var(--accent, #bdc2ff)" : "var(--muted, #908f9e)", margin: "4px 0" }}>
              {activeAction ? `ACTION ID: ${activeAction.actionId} (B${activeAction.band} : ${activeAction.mode})` : "ACTION ID: —"}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {[
                ["SELECTED BAND", activeAction?.band != null ? `B${activeAction.band} (${(activeAction.band * 500 + 250).toLocaleString()} MHz)` : "—"],
                ["SCAN MODE", activeAction?.mode ?? "—"],
                ["DWELL TIME", activeAction?.dwellUs != null ? `${Number(activeAction.dwellUs).toFixed(0)} µs` : "—"],
                ["INTERCEPT PROBABILITY", activeAction?.probability != null ? `${(Number(activeAction.probability) * 100).toFixed(1)}%` : "—"],
                ["PREDICTED INTERCEPT TIME", activeAction?.timeUs != null ? `${Number(activeAction.timeUs).toFixed(2)} µs` : "—"],
                ["INFERENCE LATENCY", activeAction?.latencyMs != null ? `${Number(activeAction.latencyMs).toFixed(2)} ms` : "—"],
                ["DECISION ATTRIBUTION", activeAction?.attribution?.decision_source || activeAction?.attribution?.reason || (activeAction ? "DRQN Policy" : "—")],
                ["EAGER VS REVISIT", activeAction ? `${((activeAction.attribution?.eager_pct ?? 0.6) * 100).toFixed(0)}% / ${((activeAction.attribution?.revisit_pct ?? 0.4) * 100).toFixed(0)}%` : "—"],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="st-tsm"
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "3px 6px",
                    background: "var(--panel-2, #1a1c20)",
                    border: "1px solid var(--border, #454653)",
                  }}
                >
                  <span style={{ color: "var(--muted, #908f9e)" }}>{label}</span>
                  <strong style={{ color: "var(--text, #e2e2e8)" }}>{value}</strong>
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
                <span className="st-tsm" style={{ color: "var(--muted, #908f9e)", padding: 6 }}>
                  Awaiting live 360-D observation for automated inference.
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

