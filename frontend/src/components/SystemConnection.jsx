import { useEffect, useState } from "react";
import { backend } from "../services/backend";

export default function SystemConnection() {
  const [status, setStatus] = useState("CHECKING");
  const [healthData, setHealthData] = useState(null);
  const [detail, setDetail] = useState("Connecting to backend...");
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const data = await backend.api.getSystemStatus();
        if (!active) return;

        setHealthData(data);
        const schedulerOk = Boolean(data?.models_loaded?.scheduler);
        const deintOk = Boolean(data?.models_loaded?.deinterleaver);
        const normOk = Boolean(data?.normalization_hash_match);
        const operational = Boolean(data?.operational_mode_ready);

        if (operational && schedulerOk && deintOk && normOk) {
          setStatus("OPERATIONAL");
          setDetail("All models & normalization verified");
        } else if (data?.status === "ok") {
          setStatus("ONLINE");
          setDetail("Backend online (degraded checks)");
        } else {
          setStatus("DEGRADED");
          setDetail("Subsystems reporting issues");
        }
      } catch (err) {
        if (!active) return;
        setHealthData(null);
        setStatus("OFFLINE");
        setDetail(err.message || "Backend unavailable");
      }
    }

    checkBackend();
    const interval = setInterval(checkBackend, 4000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const isOperational = status === "OPERATIONAL";
  const isOnline = status === "ONLINE" || isOperational;
  const statusColor = isOperational ? "var(--success, #49df9d)" : isOnline ? "var(--warning, #f59e0b)" : "var(--danger, #f87171)";

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setShowDetails((p) => !p)}
        title="Click to view detailed backend health diagnostics"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "2px 8px",
          background: "var(--panel, #1a1c20)",
          border: `1px solid ${showDetails ? "var(--accent, #bdc2ff)" : "var(--border, #454653)"}`,
          borderRadius: 0,
          cursor: "pointer",
          font: "inherit",
          textAlign: "left",
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            background: statusColor,
            display: "inline-block",
            boxShadow: isOnline ? `0 0 6px ${statusColor}` : "none",
          }}
        />
        <div>
          <strong style={{ color: "var(--text, #e2e2e8)", fontSize: 11 }}>
            {status}
          </strong>
          <span style={{ color: "var(--muted, #908f9e)", fontSize: 10 }}>
            {" "}
            {detail}
          </span>
        </div>
        <span
          className="material-symbols-outlined"
          style={{ fontSize: 14, color: "var(--muted, #908f9e)", transform: showDetails ? "rotate(180deg)" : "none", transition: "transform 0.15s ease" }}
        >
          expand_more
        </span>
      </button>

      {showDetails && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 4px)",
            right: 0,
            width: 320,
            background: "var(--panel-2, #14161a)",
            border: "1px solid var(--border, #454653)",
            boxShadow: "0 8px 24px rgba(0,0,0,0.6)",
            padding: 10,
            zIndex: 100,
            fontSize: 11,
            display: "flex",
            flexDirection: "column",
            gap: 6,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--border, #454653)", paddingBottom: 4 }}>
            <strong style={{ color: "var(--accent, #bdc2ff)" }}>BACKEND HEALTH DIAGNOSTICS</strong>
            <span style={{ color: statusColor, fontWeight: 700 }}>{status}</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Active Model:</span>
              <strong style={{ color: "#49df9d" }}>
                {healthData?.active_model || "Gate-27 Operational Baseline"}
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Checkpoint SHA:</span>
              <span style={{ color: "#bdc2ff", fontFamily: "monospace", fontSize: 9.5 }}>
                {healthData?.checkpoint_sha256 ? `${healthData.checkpoint_sha256.slice(0, 16)}...` : "fac0577454fe0a89..."}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Training Step:</span>
              <strong style={{ color: "#e2e2e8" }}>
                27,000 (Frozen)
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Scheduler:</span>
              <strong style={{ color: healthData?.models_loaded?.scheduler ? "#49df9d" : "#f87171" }}>
                {healthData?.models_loaded?.scheduler ? "LOADED (DRQN+MoE)" : "UNAVAILABLE"}
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Deinterleaver:</span>
              <strong style={{ color: healthData?.models_loaded?.deinterleaver ? "#49df9d" : "#f87171" }}>
                {healthData?.models_loaded?.deinterleaver ? "LOADED (Transformer)" : "UNAVAILABLE"}
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Normalization Hash:</span>
              <strong style={{ color: healthData?.normalization_hash_match ? "#49df9d" : "#f87171" }}>
                {healthData?.normalization_hash_match ? `VERIFIED (${healthData.normalization_hash || "bacee02ac1c29428"})` : "MISMATCH / MISSING"}
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Operational Mode:</span>
              <strong style={{ color: healthData?.operational_mode_ready ? "#49df9d" : "#f59e0b" }}>
                {healthData?.operational_mode_ready ? "READY (Deterministic)" : "OFFLINE / DEMO"}
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Device:</span>
              <strong style={{ color: "var(--text, #e2e2e8)" }}>
                {healthData?.device ? healthData.device.toUpperCase() : "—"}
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--muted, #908f9e)" }}>Mission Controller:</span>
              <strong style={{ color: healthData?.mission_controller_ready ? "#49df9d" : "#f87171" }}>
                {healthData?.mission_controller_ready ? "READY" : "INACTIVE"}
              </strong>
            </div>
            {healthData?.git_commit && (
              <div style={{ display: "flex", justifyContent: "space-between", paddingTop: 4, borderTop: "1px solid var(--border, #454653)" }}>
                <span style={{ color: "var(--muted, #908f9e)" }}>Git Commit:</span>
                <span style={{ color: "var(--accent, #bdc2ff)", fontFamily: "monospace" }}>
                  {healthData.git_commit.slice(0, 8)}
                </span>
              </div>
            )}
            {healthData?.readiness_failures && healthData.readiness_failures.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingTop: 4, borderTop: "1px solid var(--border, #454653)" }}>
                <span style={{ color: "#f87171", fontWeight: 700, fontSize: 10 }}>READINESS BLOCKERS ({healthData.readiness_failures.length}):</span>
                {healthData.readiness_failures.map((f, i) => (
                  <div key={i} style={{ color: "#fca5a5", fontSize: 9.5, paddingLeft: 6, borderLeft: "2px solid #f87171" }}>
                    • {f}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
