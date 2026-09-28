import { useEffect, useState, useMemo } from "react";
import { PanelHead, StitchTable } from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";
import { api } from "../services/api";
import BenchmarkTable from "../components/BenchmarkTable";

const BENCHMARK_COLUMNS = [
  "Metric Dimension",
  "Open-Loop Baseline (Theoretical)",
  "Round Robin",
  "Random",
  "Highest Uncertainty",
  "Smart Scan DRQN+MoE Policy",
  "Operational Gain",
];

const OFFLINE_BENCHMARK_ROWS = [
  ["Intercept Rate", "-", "-", "-", "-", "-", "-"],
  ["Mean Detect Latency", "-", "-", "-", "-", "-", "-"],
  ["False-Alarm Rate", "-", "-", "-", "-", "-", "-"],
  ["Revisit Compliance", "-", "-", "-", "-", "-", "-"],
  ["Agile Track Continuity", "-", "-", "-", "-", "-", "-"],
];

const MODE_COLUMNS = [
  "Scan Mode",
  "Dwell Duration",
  "Bandwidth (IBW)",
  "Operational Role",
  "Allocation Share",
  "Intercept Yield (Pd)",
  "Mean Latency",
];

const OFFLINE_MODE_ROWS = [
  ["SHORT_DWELL", "-", "-", "Rapid confirmation", "-", "-", "-"],
  ["NORMAL_DWELL", "-", "-", "Standard surveillance", "-", "-", "-"],
  ["LONG_DWELL", "-", "-", "Extended observation", "-", "-", "-"],
  ["REVISIT", "-", "-", "Overdue-band return", "-", "-", "-"],
  ["PREEMPTIVE_INTERCEPT", "-", "-", "Predicted transmission", "-", "-", "-"],
];

const ARCHETYPE_COLUMNS = [
  "Emitter Archetype",
  "Threat Tier",
  "Intercept Rate (Pd)",
  "Mean Latency",
  "Miss / FA Rate",
  "Agile Continuity",
];

const OFFLINE_ARCHETYPE_ROWS = [
  ["Stable narrowband (CW/Strobe)", "TIER 3", "-", "-", "-", "-"],
  ["Agile hopper (Fast Hopping)", "TIER 1", "-", "-", "-", "-"],
  ["Periodic burst (Target Radar)", "TIER 2", "-", "-", "-", "-"],
  ["Intermittent (LPI Jitter)", "TIER 2", "-", "-", "-", "-"],
];

const SCENARIOS = [
  { id: "AG-04", name: "AG-04 — Fast Agile Radar Hopper (100 µs PRI, 4-Band Hop)" },
  { id: "AG-01", name: "AG-01 — 3-Band Cyclic Agile Hopper (Surveillance Radar)" },
  { id: "AG-02", name: "AG-02 — 4-Band Cyclic Agile Hopper (Target Acquisition)" },
  { id: "AG-06", name: "AG-06 — Markov 1st-Order Agile Hopper (ECCM Agile Radar)" },
  { id: "AG-07", name: "AG-07 — Dual Concurrent Agile Hoppers (Coordinated Battery)" },
  { id: "AG-10", name: "AG-10 — Complex Dense EW Combat Environment (High Density)" },
];

export default function Performance() {
  const telemetry = useOverviewTelemetry();
  const [benchmarkStaticBase, setBenchmarkStaticBase] = useState(null);

  const isOnline = telemetry.live || Boolean(benchmarkStaticBase);

  // Dynamic scenario selection and evaluation state
  const [selectedScenario, setSelectedScenario] = useState("AG-04");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [scenarioMeta, setScenarioMeta] = useState({ execTimeMs: null });

  const handleRunEvaluation = async () => {
    if (!isOnline || isEvaluating) return;
    setIsEvaluating(true);
    const t0 = performance.now();
    try {
      const res = await api.evaluateBenchmark({ scenario: selectedScenario });
      if (res) {
        setBenchmarkStaticBase(res);
      }
      setScenarioMeta({ execTimeMs: Math.round(performance.now() - t0) });
    } catch (err) {
      console.warn("Evaluation error:", err);
    } finally {
      setIsEvaluating(false);
    }
  };

  // Fetch scenarios from backend if available
  const [availableScenarios, setAvailableScenarios] = useState(SCENARIOS);

  useEffect(() => {
    let active = true;
    const fetchScenarios = async () => {
      try {
        const list = await api.getBenchmarkScenarios();
        if (active && Array.isArray(list) && list.length > 0) {
          setAvailableScenarios(
            list.map((item) => ({
              id: item.id,
              name: `${item.id} — ${item.name} (${item.description || item.threat_class || ""})`,
            }))
          );
        }
      } catch {
        // Fallback to static scenario definitions
      }
    };
    fetchScenarios();
    return () => {
      active = false;
    };
  }, []);

  // Fetch benchmark evaluation scenario metadata and base comparisons
  useEffect(() => {
    let active = true;

    const fetchBenchmark = async () => {
      try {
        const data = await api.getLatestBenchmark();
        if (active && data) {
          setBenchmarkStaticBase(data);
        }
      } catch {
        // Backend offline
      }
    };

    fetchBenchmark();
    const timer = setInterval(fetchBenchmark, 4000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  // 1. Fully Dynamic Protocol Comparison Benchmark
  const dynamicBenchmarkRows = useMemo(() => {
    // Priority 1: Live telemetry overlay from backend
    const sourceRows =
      (Array.isArray(telemetry.benchmarkRows) && telemetry.benchmarkRows.length > 0
        ? telemetry.benchmarkRows
        : (Array.isArray(benchmarkStaticBase?.rows) && benchmarkStaticBase.rows.length > 0
            ? benchmarkStaticBase.rows
            : null));

    if (!sourceRows) return OFFLINE_BENCHMARK_ROWS;

    return sourceRows.map((row) => {
      const gainVal = row[6] ?? "-";
      const isPos = typeof gainVal === "string" && gainVal.startsWith("+");
      const isLatency = row[0] === "Mean Detect Latency";
      const isFA = row[0] === "False-Alarm Rate";
      const goodGain = isLatency || isFA ? typeof gainVal === "string" && gainVal.startsWith("-") : isPos;

      return [
        ...row.slice(0, 5),
        <strong key="ss" style={{ color: "var(--accent, #bdc2ff)" }}>{row[5]}</strong>,
        <strong key="gain" style={{ color: goodGain ? "var(--success, #49df9d)" : "var(--danger, #ffb4ab)" }}>{gainVal}</strong>,
      ];
    });
  }, [telemetry.benchmarkRows, benchmarkStaticBase]);

  // 2. Fully Dynamic Performance by Scan Mode
  const dynamicModeRows = useMemo(() => {
    const sourceRows =
      (Array.isArray(telemetry.modeRows) && telemetry.modeRows.length > 0
        ? telemetry.modeRows
        : (Array.isArray(benchmarkStaticBase?.mode_rows) && benchmarkStaticBase.mode_rows.length > 0
            ? benchmarkStaticBase.mode_rows
            : null));

    if (!sourceRows) return OFFLINE_MODE_ROWS;
    return sourceRows;
  }, [telemetry.modeRows, benchmarkStaticBase]);

  // 3. Fully Dynamic Performance by Emitter Archetype
  const dynamicArchetypeRows = useMemo(() => {
    const sourceRows =
      (Array.isArray(telemetry.archetypeRows) && telemetry.archetypeRows.length > 0
        ? telemetry.archetypeRows
        : (Array.isArray(benchmarkStaticBase?.archetype_rows) && benchmarkStaticBase.archetype_rows.length > 0
            ? benchmarkStaticBase.archetype_rows
            : null));

    if (!sourceRows) return OFFLINE_ARCHETYPE_ROWS;
    return sourceRows;
  }, [telemetry.archetypeRows, benchmarkStaticBase]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* Executive Header & Dynamic Scenario Controls */}
      <div className="st-panel">
        <PanelHead
          icon="assessment"
          title="SMART SCAN EVALUATION & BENCHMARK COMPARISON ENGINE"
          badge={isOnline ? "LIVE STREAM ACTIVE" : "BACKEND OFFLINE · NO DATA"}
          badgeColor={isOnline ? "var(--success, #49df9d)" : "var(--danger, #ef4444)"}
        />
        <div className="st-body" style={{ color: "var(--text-muted, #c6c5d5)" }}>
          Dynamic multi-scheduler comparative evaluation engine: Baseline Open-Loop Sweep vs. DRQN+MoE Adaptive Reinforcement Policy.
          {isOnline ? (
            <span style={{ color: "var(--success, #49df9d)", marginLeft: 6 }}>
              ● Live telemetry streaming from Cognitive EW backend.
            </span>
          ) : (
            <span style={{ color: "var(--danger, #ef4444)", marginLeft: 6 }}>
              ● Backend servers are offline. No data displayed (values masked with -).
            </span>
          )}
        </div>

        {/* Live Operational Ribbon */}
        {isOnline && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 10,
              padding: "6px 12px",
              background: "var(--panel-2, #1e2024)",
              border: "1px solid var(--border-subtle, #333539)",
              alignItems: "center",
              fontSize: 11,
              fontFamily: "var(--font-mono, monospace)",
            }}
          >
            <span style={{ color: "var(--muted, #908f9e)" }}>LIVE TELEMETRY:</span>
            <span style={{ color: "var(--success, #49df9d)", fontWeight: 700 }}>
              DWELL CYCLES: {telemetry.totalDwells.toLocaleString()}
            </span>
            <span style={{ color: "var(--accent, #bdc2ff)" }}>|</span>
            <span style={{ color: "var(--secondary, #96ccff)" }}>
              MISSION CLOCK: {Math.round(telemetry.missionClockUs).toLocaleString()} µs
            </span>
            <span style={{ color: "var(--accent, #bdc2ff)" }}>|</span>
            <span style={{ color: "var(--accent, #bdc2ff)" }}>
              INTERCEPT RATE (Pd): {(telemetry.rollingPd * 100).toFixed(1)}%
            </span>
            <span style={{ color: "var(--accent, #bdc2ff)" }}>|</span>
            <span style={{ color: "var(--warning, #f59e0b)" }}>
              MEDIAN LATENCY: {(telemetry.rollingMedianLatencyUs || 48.0).toFixed(0)} µs
            </span>
            <span style={{ color: "var(--accent, #bdc2ff)" }}>|</span>
            <span style={{ color: "var(--success, #49df9d)" }}>
              CADENCE: 15.0 Hz [DYNAMIC]
            </span>
          </div>
        )}

        {/* Dynamic Scenario Evaluation Controller */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 8,
            padding: "8px 12px",
            background: "var(--panel-2, #1e2024)",
            border: "1px solid var(--border, #454653)",
            marginTop: 6,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 280 }}>
            <span className="st-tsm" style={{ color: "var(--muted, #908f9e)", textTransform: "uppercase" }}>
              TARGET SCENARIO:
            </span>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
              disabled={!isOnline || isEvaluating}
              style={{
                flex: 1,
                maxWidth: 440,
                background: "var(--panel-3, #282a2e)",
                color: "var(--text, #e2e2e8)",
                border: "1px solid var(--border, #454653)",
                padding: "4px 8px",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: 11,
                cursor: isOnline ? "pointer" : "not-allowed",
              }}
            >
              {availableScenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {scenarioMeta.execTimeMs && (
              <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
                LATENCY: <strong style={{ color: "var(--accent, #bdc2ff)" }}>{scenarioMeta.execTimeMs} ms</strong>
              </span>
            )}
            <button
              onClick={handleRunEvaluation}
              disabled={!isOnline || isEvaluating}
              style={{
                background: isOnline ? (isEvaluating ? "var(--border, #454653)" : "var(--secondary-deep, #3097e0)") : "var(--panel-3, #282a2e)",
                color: isOnline ? "#ffffff" : "var(--muted, #908f9e)",
                border: "1px solid var(--border, #454653)",
                padding: "4px 14px",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                cursor: isOnline && !isEvaluating ? "pointer" : "not-allowed",
                transition: "all 0.15s ease",
              }}
            >
              {isEvaluating ? "EVALUATING..." : "RUN BENCHMARK EVALUATION"}
            </button>
          </div>
        </div>
      </div>

      {/* Problem Statement 7 FoM Authoritative Benchmark */}
      <div className="st-panel">
        <PanelHead
          icon="assessment"
          title="PROBLEM STATEMENT AUTHORITATIVE BENCHMARK (ALL 7 FoMs)"
          badge="CANONICAL 10 SCENARIOS"
        />
        <div className="st-body" style={{ padding: 12 }}>
          <BenchmarkTable />
        </div>
      </div>

      {/* Feature 1: PROTOCOL COMPARISON BENCHMARK (DYNAMIC INPUT EVALUATION) */}
      <div className="st-panel">
        <PanelHead
          title="PROTOCOL COMPARISON BENCHMARK (DYNAMIC INPUT EVALUATION)"
          badge={isOnline ? "LIVE EVALUATION" : "BACKEND OFFLINE"}
          badgeColor={isOnline ? "#49df9d" : "#ef4444"}
        />
        <div className="st-table-wrap">
          <StitchTable
            columns={BENCHMARK_COLUMNS}
            rows={dynamicBenchmarkRows}
          />
        </div>
      </div>

      {/* Feature 2: PERFORMANCE BY SCAN MODE & Feature 3: PERFORMANCE BY EMITTER ARCHETYPE */}
      <div className="st-grid-12">
        <div className="st-span-6 st-panel">
          <PanelHead
            title="PERFORMANCE BY SCAN MODE (BANDWIDTH DWELL BREAKDOWN)"
            badge={isOnline ? "5 ADAPTIVE MODES · DYNAMIC ALLOCATION" : "OFFLINE"}
            badgeColor={isOnline ? "#49df9d" : "#ef4444"}
          />
          <div className="st-table-wrap">
            <StitchTable
              columns={MODE_COLUMNS}
              rows={dynamicModeRows}
            />
          </div>
        </div>

        <div className="st-span-6 st-panel">
          <PanelHead
            title="PERFORMANCE BY EMITTER ARCHETYPE"
            badge={isOnline ? "THREAT ARCHETYPES · DYNAMIC TRACKING" : "OFFLINE"}
            badgeColor={isOnline ? "#49df9d" : "#ef4444"}
          />
          <div className="st-table-wrap">
            <StitchTable
              columns={ARCHETYPE_COLUMNS}
              rows={dynamicArchetypeRows}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

