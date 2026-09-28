import { PanelHead, StitchTable } from "../components/stitch";
import { backend } from "../services/backend";
import { useEffect, useState } from "react";

import { getApiBaseUrl, setApiBaseUrl } from "../services/api";

const SUBSYSTEMS = [
  "RF SIMULATOR",
  "PDW DETECTOR",
  "RX TUNER (PLL)",
  "360-D OBS PIPELINE",
  "SCHEDULER",
  "DATASET & MEM",
];

export default function SystemConfig() {
  const [status, setStatus] = useState("CHECKING");
  const [apiUrl, setApiUrl] = useState(getApiBaseUrl());
  const [savedNotice, setSavedNotice] = useState("");

  const testConnection = (targetUrl) => {
    if (targetUrl) {
      setApiBaseUrl(targetUrl);
    }
    setStatus("TESTING...");
    backend.api
      .getSystemStatus()
      .then((data) => {
        setStatus("CONNECTED");
        setSavedNotice(`Connected: model ${data?.active_model ? data.active_model.split(/[\\\\/]/).pop() : "active"}`);
      })
      .catch((err) => {
        setStatus("OFFLINE — SYNTHETIC FALLBACK");
        setSavedNotice(`Connection failed: ${err.message || "Failed to reach endpoint"}`);
      });
  };

  useEffect(() => {
    let active = true;
    const initialCheck = async () => {
      try {
        const data = await backend.api.getSystemStatus();
        if (active) {
          setStatus("CONNECTED");
          setSavedNotice(`Connected: model ${data?.active_model ? data.active_model.split(/[\\\\/]/).pop() : "active"}`);
        }
      } catch (err) {
        if (active) {
          setStatus("OFFLINE — SYNTHETIC FALLBACK");
          setSavedNotice(`Connection failed: ${err.message || "Failed to reach endpoint"}`);
        }
      }
    };
    initialCheck();
    return () => {
      active = false;
    };
  }, []);

  const handleSaveApiUrl = () => {
    const trimmed = apiUrl.trim().replace(/\/+$/, "");
    setApiBaseUrl(trimmed);
    setApiUrl(trimmed || getApiBaseUrl());
    testConnection(trimmed);
  };

  const handleResetApiUrl = () => {
    setApiBaseUrl("");
    const def = getApiBaseUrl();
    setApiUrl(def);
    testConnection(def);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead icon="tune" title="System Architecture, Parameters & Hardware Configuration" badge="SYS" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Subsystem control, interface parameters, and read-only hardware
          state. Backend: <strong style={{ color: status === "CONNECTED" ? "#34d399" : "#f87171" }}>{status}</strong>.
        </div>
        <div className="st-tsm" style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {SUBSYSTEMS.map((s) => (
            <span key={s} className="st-badge">
              {s}
            </span>
          ))}
        </div>
      </div>

      <div className="st-panel" style={{ border: "1px solid rgba(59, 130, 246, 0.4)" }}>
        <PanelHead icon="cloud_sync" title="00 // LIVE BACKEND CLOUD CONNECTION (CLOUD RUN / VERCEL)" badge="API" />
        <div className="st-body" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontSize: "0.85rem", color: "#a5a3b7" }}>
            Active Backend URL for REST endpoints and secure WebSockets (<code>wss://</code>). Connects this Vercel deployment directly to your Cloud Run backend service:
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="https://cognitive-ew-backend-753709137146.asia-south1.run.app"
              style={{
                flex: "1 1 320px",
                padding: "8px 12px",
                backgroundColor: "rgba(0,0,0,0.3)",
                border: "1px solid rgba(255,255,255,0.18)",
                color: "#fff",
                borderRadius: "4px",
                fontFamily: "monospace",
                fontSize: "0.85rem",
              }}
            />
            <button
              onClick={handleSaveApiUrl}
              style={{
                padding: "8px 16px",
                backgroundColor: "#2563eb",
                color: "#fff",
                border: "none",
                borderRadius: "4px",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Connect & Save
            </button>
            <button
              onClick={handleResetApiUrl}
              style={{
                padding: "8px 16px",
                backgroundColor: "rgba(255,255,255,0.1)",
                color: "#ddd",
                border: "none",
                borderRadius: "4px",
                cursor: "pointer",
              }}
            >
              Reset Default
            </button>
          </div>
          {savedNotice && (
            <div style={{ fontSize: "0.8rem", color: status === "CONNECTED" ? "#34d399" : "#fbbf24" }}>
              {savedNotice}
            </div>
          )}
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-6 st-panel">
          <PanelHead title="01 // RF ENVIRONMENT SPECIFICATIONS" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["Total spectrum", "0–18,000 MHz", "NOMINAL"],
              ["Bands", "36 × 500 MHz", "NOMINAL"],
            ]}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="02 // RECEIVER HARDWARE CONFIGURATION" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["IBW", "1 GHz locked", "LOCKED"],
              ["Frequency step", "500 MHz", "NOMINAL"],
              ["Detection floor", "SNR ≥ 15 dB", "NOMINAL"],
              ["Dwell", "120 µs revisit", "ARMED"],
            ]}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="03 // SMART SCHEDULER (DRQN + MoE)" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["Observation", "360-D (36×10)", "READY"],
              ["Actions", "180 (36×5)", "READY"],
              ["Core", "DRQN LSTM-256 ×2 + MoE", "READY"],
            ]}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="04 // DATASET & BUFFER SPECIFICATIONS" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["Dwell buffer", "Last 500 ms", "READY"],
              ["GT isolation", "Strict separation", "ENFORCED"],
            ]}
          />
        </div>
      </div>
    </div>
  );
}
