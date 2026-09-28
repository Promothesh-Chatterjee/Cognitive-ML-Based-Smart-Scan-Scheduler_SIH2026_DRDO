import { useEffect, useMemo, useState } from "react";
import {
  BandMatrix,
  CmdBadge,
  DataSourceBadge,
  KpiCard,
  PanelHead,
  PipelineFlow,
} from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";
import { useMetricsWebSocket } from "../hooks/useMetricsWebSocket";
import { api } from "../services/api";
import LiveMetricsDashboard from "../components/LiveMetricsDashboard";

// ── Fallback Scenario Catalog ────────────────────────────────────────────────
const FALLBACK_BENCHMARKS = [
  { id: "config_119", name: "Config 119", display_name: "Config 119 — Sparse Agile", class: "sparse", display_class: "Sparse Agile", benchmark: true, source: "tsrd" },
  { id: "config_29", name: "Config 29", display_name: "Config 29 — Slow Agile Hopper", class: "slow_agile", display_class: "Slow Agile Hopper", benchmark: true, source: "tsrd" },
  { id: "config_241", name: "Config 241", display_name: "Config 241 — Fast Agile Hopper", class: "fast_agile", display_class: "Fast Agile Hopper", benchmark: true, source: "tsrd" },
  { id: "config_195", name: "Config 195", display_name: "Config 195 — Dense Battlefield", class: "dense", display_class: "Dense Battlefield", benchmark: true, source: "tsrd" },
  { id: "config_64", name: "Config 64", display_name: "Config 64 — Dense Agile", class: "dense", display_class: "Dense Agile", benchmark: true, source: "tsrd" },
  { id: "config_42", name: "Config 42", display_name: "Config 42 — Dense Radar", class: "dense", display_class: "Dense Radar", benchmark: true, source: "tsrd" },
  { id: "config_96", name: "Config 96", display_name: "Config 96 — Mixed Multi-Emitter", class: "mixed", display_class: "Mixed Multi-Emitter", benchmark: true, source: "tsrd" },
  { id: "config_117", name: "Config 117", display_name: "Config 117 — Mixed Multi-Emitter", class: "mixed", display_class: "Mixed Multi-Emitter", benchmark: true, source: "tsrd" },
  { id: "config_143", name: "Config 143", display_name: "Config 143 — Sparse Agile", class: "sparse", display_class: "Sparse Agile", benchmark: true, source: "tsrd" },
  { id: "config_194", name: "Config 194", display_name: "Config 194 — Sparse Agile", class: "sparse", display_class: "Sparse Agile", benchmark: true, source: "tsrd" },
  { id: "final_grc", name: "final.grc", display_name: "final.grc — GNU Radio 5-Emitter Agile FHSS", class: "fast_agile", display_class: "GNU Radio Agile FHSS", benchmark: false, source: "gnu_radio" },
  { id: "saa_grc", name: "saa.grc", display_name: "saa.grc — GNU Radio Sample & Hold / Chirp", class: "mixed", display_class: "GNU Radio S&H Chirp", benchmark: false, source: "gnu_radio" },
];

function getScenarioBadgeStyle(scenarioClass) {
  switch (scenarioClass) {
    case "sparse":
      return { bg: "rgba(73, 223, 157, 0.12)", border: "#49df9d", color: "#49df9d" };
    case "fast_agile":
      return { bg: "rgba(56, 189, 248, 0.12)", border: "#38bdf8", color: "#38bdf8" };
    case "slow_agile":
      return { bg: "rgba(189, 194, 255, 0.12)", border: "#bdc2ff", color: "#bdc2ff" };
    case "dense":
      return { bg: "rgba(245, 158, 11, 0.12)", border: "#f59e0b", color: "#f59e0b" };
    case "mixed":
      return { bg: "rgba(236, 72, 153, 0.12)", border: "#ec4899", color: "#ec4899" };
    case "periodic":
      return { bg: "rgba(168, 85, 247, 0.12)", border: "#a855f7", color: "#a855f7" };
    default:
      return { bg: "rgba(144, 143, 158, 0.12)", border: "#908f9e", color: "#908f9e" };
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────

function pct(v) {
  if (v == null || isNaN(Number(v))) return "—";
  const n = Number(v);
  return isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—";
}

function fmtUs(us) {
  const n = Number(us);
  return !isNaN(n) && isFinite(n) ? `${n.toFixed(0)} µs` : "0.0 µs";
}

function fmtScore(v) {
  const n = Number(v);
  return !isNaN(n) && isFinite(n) ? n.toFixed(3) : "0.000";
}

// ── Connection State Badge ───────────────────────────────────────────────────

function ConnectionStateBadge({ state, pollingIntervalMs }) {
  let color;
  let label;
  let isPulsing = false;

  switch (state) {
    case "POLLING_LIVE":
      color = "#49df9d";
      label = `Polling live telemetry (${pollingIntervalMs}ms)`;
      isPulsing = true;
      break;
    case "BACKEND_CONNECTED":
      color = "#38bdf8";
      label = "Backend connected";
      break;
    case "MISSION_INACTIVE":
      color = "#f59e0b";
      label = "Mission inactive";
      break;
    case "STREAM_INACTIVE":
      color = "#f59e0b";
      label = "Stream inactive";
      break;
    case "BACKEND_UNAVAILABLE":
      color = "#ef4444";
      label = "Backend unavailable";
      break;
    default:
      color = "#908f9e";
      label = state || "Connecting...";
  }

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        fontSize: 11,
        color,
        fontWeight: 700,
        letterSpacing: "0.04em",
        background: "var(--panel-3, rgba(0,0,0,0.3))",
        border: `1px solid ${color}40`,
        padding: "2px 8px",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          background: color,
          display: "inline-block",
          borderRadius: 2,
          boxShadow: isPulsing ? `0 0 6px ${color}` : "none",
          animation: isPulsing ? "st-pulse 1.5s ease-in-out infinite" : "none",
        }}
      />
      {label.toUpperCase()}
    </span>
  );
}

// ── Mission Control Toolbar ──────────────────────────────────────────────────

function MissionControls({
  t,
  isOperating,
  setIsOperating,
  controlError,
  setControlError,
}) {
  const [scenarios, setScenarios] = useState(FALLBACK_BENCHMARKS);
  const [selectedScenario, setSelectedScenario] = useState("config_119");
  const [selectedSpeed, setSelectedSpeed] = useState(15.0);

  const {
    streamRunning,
    missionActive,
    connectionState,
    pollingIntervalMs,
    setPollingInterval,
    startStream,
    stopStream,
    startMission,
    stepMission,
    stopMission,
    resetMission,
    totalDwells,
    lastError,
  } = t;

  useEffect(() => {
    let unmounted = false;
    api
      .getMissionScenarios()
      .then((data) => {
        if (!unmounted && Array.isArray(data) && data.length > 0) {
          setScenarios(data);
          setSelectedScenario((prev) => {
            if (data.some((s) => s.id === prev)) return prev;
            const firstBench = data.find((s) => s.benchmark) || data[0];
            return firstBench ? firstBench.id : prev;
          });
        }
      })
      .catch((err) => {
        console.warn("Failed fetching scenarios from backend, using fallback list:", err);
      });
    return () => {
      unmounted = true;
    };
  }, []);

  const currentScenarioObj = useMemo(() => {
    return scenarios.find((s) => s.id === selectedScenario) || FALLBACK_BENCHMARKS[0];
  }, [scenarios, selectedScenario]);

  const benchmarkScenarios = useMemo(() => scenarios.filter((s) => s.benchmark), [scenarios]);
  const otherTsrdScenarios = useMemo(() => scenarios.filter((s) => s.source === "tsrd" && !s.benchmark), [scenarios]);
  const gnuScenarios = useMemo(() => scenarios.filter((s) => s.source === "gnu_radio"), [scenarios]);
  const badgeStyle = getScenarioBadgeStyle(currentScenarioObj?.class);

  const isLiveActive = streamRunning || missionActive;

  const handleAction = async (fn, desc) => {
    setIsOperating(true);
    setControlError("");
    try {
      await fn();
    } catch (err) {
      setControlError(`Failed to ${desc}: ${err.message || String(err)}`);
    } finally {
      setIsOperating(false);
    }
  };

  return (
    <div
      className="st-panel"
      style={{
        padding: "10px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.06em",
              color: "var(--accent, #bdc2ff)",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
              play_circle
            </span>
            MISSION CONTROLLER:
          </span>

          {/* Scenario & Speed Selectors */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
              disabled={isOperating || isLiveActive}
              style={{
                background: "var(--panel-2, #1c1d22)",
                color: "var(--text, #e2e2e8)",
                border: "1px solid var(--border, #454653)",
                fontSize: 11,
                padding: "4px 8px",
                cursor: isLiveActive ? "not-allowed" : "pointer",
                fontWeight: 600,
                maxWidth: 380,
              }}
              title="Select threat scenario dataset"
            >
              {benchmarkScenarios.length > 0 && (
                <optgroup label="CANONICAL BENCHMARK CONFIGURATIONS (TSRD H5)">
                  {benchmarkScenarios.map((s) => (
                    <option key={s.id} value={s.id}>
                      ★ {s.display_name} {s.available === false ? "(missing)" : ""}
                    </option>
                  ))}
                </optgroup>
              )}
              {otherTsrdScenarios.length > 0 && (
                <optgroup label={`ADDITIONAL TSRD VALIDATION RECORDINGS (${otherTsrdScenarios.length} files)`}>
                  {otherTsrdScenarios.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.display_name}
                    </option>
                  ))}
                </optgroup>
              )}
              {gnuScenarios.length > 0 && (
                <optgroup label="GNU RADIO PHYSICAL EMULATION">
                  {gnuScenarios.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.display_name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>

            {/* Dynamic Metadata Badges */}
            <span
              style={{
                background: badgeStyle.bg,
                border: `1px solid ${badgeStyle.border}60`,
                color: badgeStyle.color,
                fontSize: 10,
                fontWeight: 700,
                padding: "3px 8px",
                letterSpacing: "0.04em",
              }}
              title={`Scenario Class: ${currentScenarioObj?.display_class}`}
            >
              {currentScenarioObj?.display_class?.toUpperCase() || "TSRD SCENARIO"}
            </span>

            <span
              style={{
                background: "var(--panel-3, rgba(0, 0, 0, 0.35))",
                border: "1px solid #49df9d40",
                color: "#49df9d",
                fontSize: 10,
                fontWeight: 700,
                padding: "3px 8px",
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
              }}
              title="Dataset verification: real HDF5 flight recording"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>
                verified
              </span>
              {currentScenarioObj?.source === "tsrd" ? "TSRD REAL H5" : "GNU RADIO"}
            </span>

            <select
              value={selectedSpeed}
              onChange={(e) => setSelectedSpeed(Number(e.target.value))}
              disabled={isOperating || isLiveActive}
              style={{
                background: "var(--panel-2, #1c1d22)",
                color: "var(--text, #e2e2e8)",
                border: "1px solid var(--border, #454653)",
                fontSize: 11,
                padding: "4px 8px",
                cursor: isLiveActive ? "not-allowed" : "pointer",
              }}
              title="Simulation speed in Hz"
            >
              <option value={10.0}>10 Hz</option>
              <option value={15.0}>15 Hz (Standard)</option>
              <option value={25.0}>25 Hz</option>
              <option value={50.0}>50 Hz (Fast)</option>
            </select>

            {/* Primary Action: START / STOP MISSION */}
            {!isLiveActive ? (
              <button
                type="button"
                onClick={() =>
                  handleAction(
                    async () => {
                      await startMission(0.0, {
                        scenario: selectedScenario,
                        speed_hz: selectedSpeed,
                        max_dwells: 4000,
                      });
                      try {
                        await startStream({
                          scenario: selectedScenario,
                          speed_hz: selectedSpeed,
                          max_dwells: 4000,
                        });
                      } catch {
                        // Stream already started via auto_stream
                      }
                    },
                    "start mission",
                  )
                }
                disabled={isOperating}
                style={{
                  background: "#0a2a18",
                  border: "1px solid #49df9d",
                  color: "#49df9d",
                  fontWeight: 700,
                  fontSize: 11,
                  padding: "4px 12px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  boxShadow: "0 0 8px rgba(73, 223, 157, 0.2)",
                }}
                title="Start continuous operational mission streaming real TSRD pulses"
              >
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                  play_arrow
                </span>
                START MISSION
              </button>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  style={{
                    color: "#49df9d",
                    fontSize: 11,
                    fontWeight: 700,
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    animation: "st-pulse 1.5s ease-in-out infinite",
                  }}
                >
                  ● STREAMING {currentScenarioObj?.name || selectedScenario} ({totalDwells} Dwells)
                </span>
                <button
                  type="button"
                  onClick={() =>
                    handleAction(
                      async () => {
                        try {
                          await stopStream();
                        } catch {
                          // Ignore
                        }
                        await stopMission();
                      },
                      "stop mission",
                    )
                  }
                  disabled={isOperating}
                  style={{
                    background: "#2a0a0a",
                    border: "1px solid #ef4444",
                    color: "#ef4444",
                    fontWeight: 700,
                    fontSize: 11,
                    padding: "4px 12px",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                  title="Stop continuous operational mission"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                    stop
                  </span>
                  STOP MISSION
                </button>
              </div>
            )}

            <button
              type="button"
              onClick={() => handleAction(() => stepMission(), "step mission")}
              disabled={isOperating}
              style={{
                background: "var(--panel-2, #1a1c22)",
                border: "1px solid var(--border, #8e9099)",
                color: "var(--text, #e2e2e8)",
                fontWeight: 600,
                fontSize: 11,
                padding: "4px 8px",
                cursor: "pointer",
              }}
              title="Step exactly 1 dwell via POST /mission/step"
            >
              STEP (1 DWELL)
            </button>

            <button
              type="button"
              onClick={() => handleAction(() => resetMission(), "reset mission")}
              disabled={isOperating}
              style={{
                background: "var(--panel-2, #1a1c22)",
                border: "1px solid var(--border, #8e9099)",
                color: "var(--muted, #908f9e)",
                fontWeight: 600,
                fontSize: 11,
                padding: "4px 8px",
                cursor: "pointer",
              }}
              title="Reset mission clock, metrics, and memory via POST /reset"
            >
              RESET
            </button>
          </div>
        </div>

        {/* Polling Interval Config */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: "var(--muted, #908f9e)" }}>POLL RATE:</span>
          {[500, 1000, 2000].map((rate) => (
            <button
              key={rate}
              type="button"
              onClick={() => setPollingInterval(rate)}
              style={{
                background: pollingIntervalMs === rate ? "var(--accent-glow, #2b3040)" : "var(--panel-2, #16171b)",
                border: `1px solid ${pollingIntervalMs === rate ? "var(--accent, #bdc2ff)" : "var(--border, #3b3d48)"}`,
                color: pollingIntervalMs === rate ? "var(--accent, #bdc2ff)" : "var(--muted, #908f9e)",
                fontWeight: pollingIntervalMs === rate ? 700 : 500,
                fontSize: 10,
                padding: "2px 6px",
                cursor: "pointer",
              }}
            >
              {rate}ms{rate === 1000 ? " (DEF)" : ""}
            </button>
          ))}
        </div>
      </div>

      {/* Error & Cold Start Warnings */}
      {(controlError || (connectionState === "BACKEND_UNAVAILABLE" && lastError)) && (
        <div
          style={{
            background: "rgba(239, 68, 68, 0.15)",
            border: "1px solid var(--danger, #ef4444)",
            padding: "6px 10px",
            color: "var(--danger, #ffb4ab)",
            fontSize: 11,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16, color: "#ef4444" }}>
            warning
          </span>
          <span>
            {controlError ||
              `Backend connection notice: ${lastError}. (Backend instances may take a moment to respond; retrying with exponential backoff).`}
          </span>
        </div>
      )}
    </div>
  );
}

// ── Scheduler Decision Panel ──────────────────────────────────────────────────

function SchedulerPanel({ scheduler, live }) {
  const rows = [
    [
      "CHOSEN TARGET",
      live ? `BAND ${Number(scheduler.chosenBand) + 1} (${scheduler.chosenFreqMHz.toLocaleString()} MHz)` : "BAND 0.0",
      "#bdc2ff",
    ],
    [
      "SCAN MODE",
      live ? `${scheduler.scanMode} (${Number(scheduler.dwellUs).toFixed(0)} µs)` : "0.0 (0.0 µs)",
      "#96ccff",
      true,
    ],
    ["ACTION SCORE", fmtScore(scheduler.drqnScore), "#e2e2e8"],
    ["INTERCEPT PROBABILITY", pct(scheduler.interceptProbability), "#e2e2e8"],
    [
      "PREDICTED ETA",
      scheduler.predictedEtaUs > 0 ? fmtUs(scheduler.predictedEtaUs) : "0.0 µs",
      "#e2e2e8",
    ],
    ["ACTION SPACE", String(scheduler.actionSpace ?? 180), "#e2e2e8"],
  ];

  return (
    <div className="st-panel">
      <PanelHead
        icon="psychology"
        title="CURRENT SCHEDULER DECISION"
        badge={live ? "DRQN + MoE ACTIVE" : "AWAITING DATA"}
      />
      <div className="st-body">
        {rows.map(([label, val, color, isBold], i) => (
          <div key={i} className="st-row">
            <span className="st-tsm" style={{ color: "var(--muted)" }}>
              {label}
            </span>
            <span
              className="st-tmd"
              style={{
                color: color ?? "var(--text)",
                fontWeight: isBold ? 700 : 500,
                letterSpacing: "0.02em",
              }}
            >
              {val}
            </span>
          </div>
        ))}
        <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 6 }}>
          <div className="st-row">
            <span className="st-tsm" style={{ color: "var(--muted)" }}>
              POLICY MODE
            </span>
            <span className="st-tmd" style={{ color: "var(--accent, #bdc2ff)", fontWeight: 700 }}>
              OPERATIONAL CANDIDATE
            </span>
          </div>
          <div className="st-row">
            <span className="st-tsm" style={{ color: "var(--muted)" }}>
              DECISION REASON
            </span>
            <span className="st-tsm" style={{ color: "var(--accent, #bdc2ff)", fontStyle: "italic", textAlign: "right" }}>
              {scheduler.decisionReason}
            </span>
          </div>
          <div className="st-row">
            <span className="st-tsm" style={{ color: "var(--muted)" }}>
              EXPLORATION PRESSURE
            </span>
            <span className="st-tmd" style={{ color: "#f59e0b" }}>
              {pct(scheduler.explorationPressure)}
            </span>
          </div>
          <div className="st-row">
            <span className="st-tsm" style={{ color: "var(--muted)" }}>
              Q-MARGIN
            </span>
            <span className="st-tmd" style={{ color: "#6afcb8" }}>
              {fmtScore(scheduler.qMargin)}
            </span>
          </div>
          <div className="st-row">
            <span className="st-tsm" style={{ color: "var(--muted)" }}>
              MoE GATING
            </span>
            <span className="st-tmd" style={{ color: "#96ccff" }}>
              {pct(scheduler.moeGating)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Environment Spectrum Status ───────────────────────────────────────────────

function EnvironmentSpectrum({ activeBands, quietBands, currentBand, currentFreqMHz, currentDwellUs }) {
  const rows = [
    ["TOTAL SPECTRUM", "18.00 GHz (36 × 500 MHz)", "var(--text)"],
    ["INSTANTANEOUS BW", "1,000 MHz (IBW)", "#96ccff"],
    ["CURRENT BAND", `B${String(Number(currentBand) + 1).padStart(2, "0")} (${currentFreqMHz.toLocaleString()} MHz)`, "var(--accent, #bdc2ff)"],
    ["ACTIVE BANDS", String(activeBands), "#49df9d"],
    ["QUIET BANDS", String(quietBands), "var(--muted)"],
    ["RECEIVER DWELL", fmtUs(currentDwellUs), "var(--accent, #bdc2ff)"],
  ];

  return (
    <div className="st-panel">
      <PanelHead icon="analytics" title="ENVIRONMENT SPECTRUM" badge="36 BANDS" />
      <div className="st-body">
        {rows.map(([label, val, color], i) => (
          <div key={i} className="st-row">
            <span className="st-tsm" style={{ color: "var(--muted)" }}>
              {label}
            </span>
            <span className="st-tmd" style={{ color: color ?? "var(--text)" }}>
              {val}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page component ───────────────────────────────────────────────────────

export default function MissionOverview() {
  const t = useOverviewTelemetry();
  const { metrics: wsMetrics } = useMetricsWebSocket();
  const [isOperating, setIsOperating] = useState(false);
  const [controlError, setControlError] = useState("");

  const {
    connectionState,
    pollingIntervalMs,
    live,
    activeBands,
    quietBands,
    currentBand,
    currentFreqMHz,
    currentMode,
    currentDwellUs,
    totalHits,
    totalDwells,
    rollingPd,
    rollingMedianLatencyUs,
    missionClockUs,
    missionActive,
    streamRunning,
    bandHeights,
    bandStates,
    scheduler,
    instantaneousPd,
    sessionAvgPd: tSessionAvgPd,
    liveMetrics,
  } = t;

  const freqLabel = `${currentFreqMHz.toLocaleString()} MHz`;
  const bandLabel = `B${String(Number(currentBand) + 1).padStart(2, "0")}`;
  const ibwRange = `${(currentFreqMHz - 500).toLocaleString()}–${(currentFreqMHz + 500).toLocaleString()} MHz`;

  // Session average Pd = cumulative hits / cumulative dwells
  const sessionAvgPd = totalDwells > 0 ? totalHits / totalDwells : (tSessionAvgPd || 0);
  const activeInstantaneousPd = instantaneousPd || rollingPd || 0;
  // Session ended = we have data but mission/stream is no longer active
  const sessionEnded = !missionActive && !streamRunning && totalDwells > 0 && live;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {/* Header */}
      <div className="st-panel">
        <PanelHead icon="grid_view" title="SMART SCAN MISSION OVERVIEW" badge="OPERATIONAL" />
        <div className="st-body" style={{ color: "var(--text-muted, #c6c5d5)", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <span>Intelligent frequency and dwell selection across wideband RF environment (36 bands × 500 MHz).</span>
          <DataSourceBadge connected={live} />
          <ConnectionStateBadge state={connectionState} pollingIntervalMs={pollingIntervalMs} />
          {(missionActive || streamRunning) && (
            <span style={{ color: "var(--success, #49df9d)", fontWeight: 700 }}>
              ● {streamRunning ? "STREAM ACTIVE" : "MISSION ACTIVE"} · {totalDwells} DWELLS · T={Number(missionClockUs).toFixed(0)} µs
            </span>
          )}
        </div>
      </div>

      {/* Mission Controls Toolbar */}
      <MissionControls
        t={t}
        isOperating={isOperating}
        setIsOperating={setIsOperating}
        controlError={controlError}
        setControlError={setControlError}
      />

      {/* Post-session summary banner — appears only when mission has stopped */}
      {sessionEnded && (
        <div
          style={{
            background: "var(--panel-2, #1e2024)",
            border: "1px solid var(--success, #49df9d)",
            borderLeft: "4px solid var(--success, #49df9d)",
            padding: "8px 12px",
            display: "flex",
            alignItems: "center",
            gap: 20,
            flexWrap: "wrap",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--success, #49df9d)", fontWeight: 700, letterSpacing: "0.06em", fontSize: 11 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
              flag
            </span>
            SESSION COMPLETE
          </span>
          <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
            TOTAL DWELLS: <strong style={{ color: "var(--text, #e2e2e8)" }}>{totalDwells}</strong>
          </span>
          <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
            TOTAL HITS: <strong style={{ color: "var(--success, #49df9d)" }}>{totalHits}</strong>
          </span>
          <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
            INSTANTANEOUS Pd: <strong style={{ color: "var(--accent, #6afcb8)" }}>{pct(activeInstantaneousPd)}</strong>
          </span>
          <span
            style={{
              background: "var(--panel-3, #282a2e)",
              border: "1px solid var(--success, #49df9d)",
              padding: "2px 10px",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
              SESSION AVG Pd:
            </span>
            <strong style={{ color: "var(--success, #49df9d)", fontSize: 15, letterSpacing: "0.04em" }}>
              {pct(sessionAvgPd)}
            </strong>
          </span>
          <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
            CLOCK: <strong style={{ color: "var(--accent, #bdc2ff)" }}>{Number(missionClockUs).toFixed(0)} µs</strong>
          </span>
        </div>
      )}



      {/* KPI Strip */}
      <section className="st-kpi-grid" aria-label="Mission KPI strip">
        <KpiCard
          label="TOTAL SPECTRUM"
          icon="tune"
          value="18.00"
          unit="GHz"
          footLeft="0.00 MHz"
          footRight="18,000 MHz"
          valueColor="#bdc2ff"
        />
        <KpiCard
          label="INSTANTANEOUS BW"
          icon="cell_tower"
          value="1,000"
          unit="MHz"
          footLeft="CANONICAL IBW"
          footRight={ibwRange}
          valueColor="#96ccff"
        />
        <KpiCard
          label="INTERCEPTION Pd"
          icon="radar"
          value={activeInstantaneousPd != null && !isNaN(activeInstantaneousPd) ? (activeInstantaneousPd * 100).toFixed(1) : "—"}
          unit={activeInstantaneousPd != null && !isNaN(activeInstantaneousPd) ? "%" : ""}
          footLeft="INSTANTANEOUS (WINDOW)"
          footRight={`SESSION AVG: ${sessionAvgPd != null && !isNaN(sessionAvgPd) ? `${(sessionAvgPd * 100).toFixed(1)}%` : "—"}`}
          valueColor="var(--accent-bright, #6afcb8)"
        />
        <KpiCard
          label="REVISIT LATENCY"
          icon="timer"
          value={rollingMedianLatencyUs > 0 ? rollingMedianLatencyUs.toFixed(0) : "0.0"}
          unit="µs"
          footLeft="P50 MEDIAN"
          footRight={rollingMedianLatencyUs > 0 ? `${(rollingMedianLatencyUs / 1000).toFixed(2)} ms` : "0.0 ms"}
          valueColor="var(--warning, #ffd700)"
        />
        <KpiCard
          label="SCHEDULER CONFIDENCE"
          icon="speed"
          value={scheduler.interceptProbability > 0 ? (scheduler.interceptProbability * 100).toFixed(1) : "0.0"}
          unit="%"
          footLeft={`ETA: ${fmtUs(scheduler.predictedEtaUs)}`}
          footRight={scheduler.scanMode}
          valueColor="var(--accent, #bdc2ff)"
        />
        <KpiCard
          label="MISSION CLOCK"
          icon="schedule"
          value={missionClockUs > 0 ? (missionClockUs / 1000).toFixed(1) : "0.0"}
          unit="ms"
          footLeft={`T=${Number(missionClockUs).toFixed(0)} µs`}
          footRight={live ? "LIVE CLOCK" : "INACTIVE"}
          valueColor="var(--accent, #bdc2ff)"
        />
      </section>

      {/* Live Cognitive EW Metrics (Phase 4) */}
      <section style={{ display: "flex", flexDirection: "column", gap: 8, margin: "8px 0" }}>
        <LiveMetricsDashboard metrics={liveMetrics || wsMetrics} />
      </section>

      {/* Main 2-column layout */}
      <div className="st-main-cols">
        {/* Left column: 36-Band Matrix */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {/* 36-Band Spectrum Allocation Matrix */}
          <div className="st-panel">
            <PanelHead
              icon="grid_4x4"
              title="36-BAND SPECTRUM MATRIX"
              badge={`${activeBands} ACTIVE · ${quietBands} QUIET`}
            />
            <div className="st-body">
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 8,
                  flexWrap: "wrap",
                  gap: 8,
                }}
              >
                <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
                  CURRENT DWELL:{" "}
                  <strong style={{ color: "var(--accent, #bdc2ff)" }}>{bandLabel}</strong>
                  {" · "}
                  <strong style={{ color: "var(--text, #e2e2e8)" }}>{freqLabel}</strong>
                  {" · "}
                  <strong style={{ color: "var(--secondary, #96ccff)" }}>{currentMode}</strong>
                  {" · "}
                  <strong style={{ color: "var(--accent, #bdc2ff)" }}>{fmtUs(currentDwellUs)}</strong>
                </span>
                <span style={{ display: "flex", gap: 6 }}>
                  <CmdBadge label="IBW 1 GHz" active={live} />
                  <CmdBadge label="HOPPER ACTIVE" active={live && activeBands > 0} />
                </span>
              </div>
              <BandMatrix
                bandHeights={bandHeights}
                bandStates={bandStates}
                currentBand={currentBand}
                freqMHz={currentFreqMHz}
                dwellUs={currentDwellUs}
              />
            </div>
          </div>

          {/* Pipeline Flow */}
          <div className="st-panel">
            <PanelHead icon="account_tree" title="PROCESSING PIPELINE" badge="5-STAGE ARCHITECTURE" />
            <div className="st-body">
              <PipelineFlow currentStage={live ? 2 : 0} />
            </div>
          </div>
        </div>

        {/* Right column: Decision Panels */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {/* Current Decision */}
          <SchedulerPanel
            scheduler={scheduler}
            live={live}
          />

          {/* Environment Status */}
          <EnvironmentSpectrum
            activeBands={activeBands}
            quietBands={quietBands}
            currentBand={currentBand}
            currentFreqMHz={currentFreqMHz}
            currentDwellUs={currentDwellUs}
          />
        </div>
      </div>
    </div>
  );
}
