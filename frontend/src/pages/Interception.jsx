import { useMemo, useState } from "react";
import { PanelHead, CmdBadge } from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

const TYPE_LABELS = {
  HIT: "HIT",
  INTERCEPTION: "INTERCEPTION",
  MISS: "MISS",
  FALSE_ALARM: "FALSE ALARM",
};

const TYPE_COLORS = {
  HIT: "#49df9d",
  INTERCEPTION: "#3097e0",
  MISS: "#ffb4ab",
  FALSE_ALARM: "#f59e0b",
};

const TYPE_BG = {
  HIT: "rgba(73, 223, 157, 0.12)",
  INTERCEPTION: "rgba(48, 151, 224, 0.14)",
  MISS: "rgba(255, 180, 171, 0.12)",
  FALSE_ALARM: "rgba(245, 158, 11, 0.12)",
};

export default function Interception() {
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [filterMode, setFilterMode] = useState("ALL"); // "ALL" | "HITS" | "MISSES"
  const [viewLayout, setViewLayout] = useState("SPLIT"); // "SPLIT" | "STREAM_ONLY" | "TELEMETRY_ONLY"

  const t = useOverviewTelemetry();
  const {
    live,
    totalHits,
    totalDwells,
    rollingPd,
    rollingMedianLatencyUs,
    currentBand,
    currentMode,
    missionClockUs,
    emitters,
    recentDwells,
  } = t;

  // Map real dwell events from backend telemetry
  const displayEvents = useMemo(() => {
    const rawList = Array.isArray(recentDwells) && recentDwells.length > 0
      ? recentDwells
      : [];

    return rawList.map((d, idx) => {
      const isHit = d.type === "HIT" || d.type === "INTERCEPTION" || Boolean(d.hit);
      const category = d.type || (isHit ? "HIT" : "MISS");
      const band = Number(d.band ?? 0);
      const timeUs = Math.round(Number(d.time_us ?? d.clock_us ?? 0));
      const freq = Number(d.frequency_mhz ?? (band * 500 + 250));
      const dwellDur = Math.round(Number(d.dwell_us ?? d.dwell_time_us ?? 100));
      const numPulses = d.num_pulses != null ? Number(d.num_pulses) : (isHit ? 1 : 0);
      const ampDb = d.amplitude_db != null ? Number(d.amplitude_db).toFixed(1) : null;
      const snrDb = d.snr_db != null ? Number(d.snr_db).toFixed(1) : null;
      const pwUs = d.pulse_width_us != null ? Number(d.pulse_width_us).toFixed(2) : null;
      const aoaDeg = d.aoa_deg != null ? Number(d.aoa_deg).toFixed(1) : null;
      const trk = d.track_id ? String(d.track_id) : null;
      const emit = d.emitter_id ? String(d.emitter_id) : null;

      return {
        id: d.id || `${timeUs}-${band}-${idx}`,
        step: d.step ?? idx,
        timeUs,
        band,
        frequencyMHz: freq,
        mode: d.mode ?? d.mode_name ?? "NORMAL_DWELL",
        dwellUs: dwellDur,
        type: category,
        expectedUs: d.expected_us != null ? Math.round(Number(d.expected_us)) : null,
        actualUs: d.actual_us != null ? Math.round(Number(d.actual_us)) : (isHit ? timeUs : null),
        errorUs: d.error_us != null ? Math.round(Number(d.error_us) * 10) / 10 : null,
        trackId: trk,
        emitterId: emit,
        amplitudeDb: ampDb,
        snrDb: snrDb,
        pulseWidthUs: pwUs,
        aoaDeg: aoaDeg,
        numPulses,
        decisionReason: d.decision_reason || "DRQN Cognitive Temporal Revisit Policy",
      };
    });
  }, [recentDwells]);

  // Filtered subset for the table
  const filteredEvents = useMemo(() => {
    if (filterMode === "HITS") {
      return displayEvents.filter((e) => e.type === "HIT" || e.type === "INTERCEPTION");
    }
    if (filterMode === "MISSES") {
      return displayEvents.filter((e) => e.type === "MISS" || e.type === "FALSE_ALARM");
    }
    return displayEvents;
  }, [displayEvents, filterMode]);

  const hitsCount = useMemo(
    () => displayEvents.filter((e) => e.type === "HIT" || e.type === "INTERCEPTION").length,
    [displayEvents]
  );
  const missesCount = useMemo(
    () => displayEvents.filter((e) => e.type === "MISS" || e.type === "FALSE_ALARM").length,
    [displayEvents]
  );

  const gridColumns = 30;
  const windowDurationUs = 500;

  // Build grid of executed dwells and RF coincidence for T-F Coincidence Plane
  const gridSignals = useMemo(() => {
    const grid = Array.from({ length: 36 }, () => Array(gridColumns).fill(null));
    if (displayEvents.length === 0) return grid;

    const nowUs = missionClockUs > 0 ? missionClockUs : (displayEvents[0]?.timeUs ?? 0);
    const startUs = Math.max(0, nowUs - windowDurationUs);
    const colWidthUs = windowDurationUs / gridColumns;

    // 1. Map physical RF emitter pulse arrivals (subtle pulses)
    if (Array.isArray(emitters) && emitters.length > 0) {
      for (const emit of emitters) {
        const band = emit.band ?? 0;
        if (band < 0 || band >= 36) continue;
        const pri = emit.pri_us && emit.pri_us > 0 ? emit.pri_us : 100;
        const firstPulse = Math.ceil(startUs / pri) * pri;
        for (let pTime = firstPulse; pTime <= nowUs; pTime += pri) {
          const col = Math.min(gridColumns - 1, Math.max(0, Math.floor((pTime - startUs) / colWidthUs)));
          grid[band][col] = {
            id: `rf-pulse-${band}-${Math.round(pTime)}`,
            timeUs: Math.round(pTime),
            band,
            frequencyMHz: emit.frequency_mhz || (band * 500 + 250),
            mode: "RADAR_EMISSION",
            type: "PULSE_ACTIVITY",
            dwellUs: 2,
            trackId: emit.tag || (emit.track_id ? `TRK-${emit.track_id}` : null),
            emitterId: emit.emitter_id || `EMIT-${band + 1}`,
            numPulses: 1,
            decisionReason: "Active radar pulse in propagation medium",
          };
        }
      }
    }

    // 2. Overlay real executed receiver dwells (HIT, INTERCEPTION, MISS take priority)
    for (const evt of displayEvents) {
      const band = evt.band ?? 0;
      if (band < 0 || band >= 36) continue;
      if (evt.timeUs >= startUs - colWidthUs && evt.timeUs <= nowUs + colWidthUs) {
        const col = Math.min(gridColumns - 1, Math.max(0, Math.floor((evt.timeUs - startUs) / colWidthUs)));
        grid[band][col] = evt;
      }
    }

    return grid;
  }, [displayEvents, missionClockUs, emitters]);

  // Selected event resolution
  const selectedEvent = useMemo(() => {
    if (selectedEventId != null) {
      const found = displayEvents.find((event) => event.id === selectedEventId);
      if (found) return found;
      for (let b = 0; b < 36; b++) {
        for (let c = 0; c < gridColumns; c++) {
          if (gridSignals[b]?.[c]?.id === selectedEventId) {
            return gridSignals[b][c];
          }
        }
      }
    }
    // Default to first event in filtered list, or first in display list, or fallback
    return (
      filteredEvents[0] ||
      displayEvents[0] || {
        id: "INIT-00",
        step: 0,
        timeUs: 14200,
        band: 16,
        frequencyMHz: 8250,
        mode: "REVISIT",
        dwellUs: 120,
        type: "HIT",
        expectedUs: 14195,
        actualUs: 14200,
        errorUs: 5.0,
        trackId: "TRK-01",
        emitterId: "EMIT-01",
        amplitudeDb: "-48.5",
        snrDb: "46.5",
        pulseWidthUs: "1.50",
        aoaDeg: "315.0",
        numPulses: 2,
        decisionReason: "High-priority threat revisit deadline met",
      }
    );
  }, [selectedEventId, filteredEvents, displayEvents, gridSignals]);

  // Handle previous / next navigation in selected event panel
  const currentIndex = useMemo(() => {
    return filteredEvents.findIndex((e) => e.id === selectedEvent.id);
  }, [filteredEvents, selectedEvent.id]);

  const handlePrevEvent = () => {
    if (filteredEvents.length === 0) return;
    const prevIdx = currentIndex > 0 ? currentIndex - 1 : filteredEvents.length - 1;
    setSelectedEventId(filteredEvents[prevIdx].id);
  };

  const handleNextEvent = () => {
    if (filteredEvents.length === 0) return;
    const nextIdx = currentIndex >= 0 && currentIndex < filteredEvents.length - 1 ? currentIndex + 1 : 0;
    setSelectedEventId(filteredEvents[nextIdx].id);
  };

  // KPIs
  const hitCount = live ? totalHits : hitsCount;
  const missCount = live ? Math.max(0, totalDwells - totalHits) : missesCount;
  const pdPct = live ? (rollingPd * 100).toFixed(1) + "%" : (displayEvents.length > 0 ? ((hitsCount / displayEvents.length) * 100).toFixed(1) + "%" : "52.4%");
  const latencyUs = live ? rollingMedianLatencyUs.toFixed(1) + " µs" : "4.8 µs";
  const totalCount = live ? totalDwells : displayEvents.length;
  const currentAperture = live ? `B${currentBand} · ${currentMode}` : "B16 · REVISIT";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Header Panel */}
      <div className="st-panel">
        <PanelHead
          icon="grid_on"
          title="TIME – FREQUENCY INTERCEPTION MATRIX & SEARCH COINCIDENCE"
          badge={live ? "LIVE MISSION TELEMETRY" : "STANDBY / STREAM READY"}
          badgeColor={live ? "#49df9d" : "#96ccff"}
        />
        <div className="st-body" style={{ color: "#c6c5d5", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <div>
            Joint 2-D time × frequency coincidence plane tracking real receiver dwells, pulse arrivals, and cognitive dwell scheduling outcomes.
            {selectedEvent.type === "HIT" || selectedEvent.type === "INTERCEPTION" ? (
              <span style={{ color: "#49df9d", marginLeft: 6 }}>
                ● Selected event captured {selectedEvent.numPulses || 1} radar pulse(s) with {selectedEvent.snrDb ?? "45.0"} dB SNR.
              </span>
            ) : (
              <span style={{ color: "#ffb4ab", marginLeft: 6 }}>
                ○ Selected event missed pulse arrival window (Tuning: B{selectedEvent.band}, Mode: {selectedEvent.mode}).
              </span>
            )}
          </div>
          {live && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, color: "#49df9d", fontSize: 12, fontWeight: 600 }}>
              <span className="pulse-indicator" style={{ width: 8, height: 8, borderRadius: "50%", background: "#49df9d" }} />
              RECEIVER ACTIVE (Clock: {Math.round(missionClockUs).toLocaleString()} µs)
            </div>
          )}
        </div>
      </div>

      {/* Interception KPI Strip */}
      <section className="st-kpi-grid" aria-label="Interception KPI strip">
        {[
          ["HITS", `${hitCount}`, "Intercepted emissions", "#49df9d"],
          ["MISSES", `${missCount}`, "Missed dwell windows", "#ffb4ab"],
          ["INTERCEPT RATE (Pd)", `${pdPct}`, "Cognitive scheduler efficiency", "#49df9d"],
          ["MEDIAN LATENCY", `${latencyUs}`, "Lead time to intercept", "#96ccff"],
          ["TOTAL DWELLS", `${totalCount}`, "Operational cycles executed", "#e2e2e8"],
          ["SEARCH DIM", "2-D", "Time × Frequency coincidence", "#bdc2ff"],
          ["ACTIVE APERTURE", currentAperture, "Live tuned receiver window", "#bdc2ff"],
        ].map(([label, value, foot, valColor]) => (
          <div className="st-kpi" key={label}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>{label}</span>
            <span className="st-tlg" style={{ color: valColor }}>
              {value}
            </span>
            <span className="st-kpi-foot"><span>{foot}</span></span>
          </div>
        ))}
      </section>

      {/* Feature 8: T-F APERTURE COINCIDENCE PLANE */}
      <div className="st-panel">
        <PanelHead
          icon="apps"
          title="T-F APERTURE COINCIDENCE PLANE (TIME × FREQUENCY GRID)"
          badge="36 RF BANDS (0.5 – 18.0 GHz)"
          badgeColor="#bdc2ff"
        />
        <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e", marginBottom: 4 }}>
          {["T-500 µs", "T-400 µs", "T-300 µs", "T-200 µs", "T-100 µs", "NOW (T-0)"].map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
        <div className="st-spec" style={{ padding: 0 }}>
          <div style={{ display: "flex", gap: 2 }}>
            <div style={{ display: "flex", flexDirection: "column", width: 32, minWidth: 32 }}>
              {Array.from({ length: 36 }, (_, band) => (
                <div
                  key={band}
                  className="st-mark"
                  style={{
                    height: 14,
                    lineHeight: "14px",
                    color: band === selectedEvent?.band ? "#bdc2ff" : "#908f9e",
                    fontWeight: band === selectedEvent?.band ? "bold" : "normal",
                    textAlign: "right",
                    paddingRight: 4,
                    fontSize: "10px",
                  }}
                >
                  B{band}
                </div>
              ))}
            </div>
            <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
              {Array.from({ length: 36 }, (_, band) => (
                <div className="st-timeline" key={band} style={{ height: 14, gap: 1, marginBottom: 1 }}>
                  {Array.from({ length: gridColumns }, (_, column) => {
                    const event = gridSignals[band]?.[column];
                    const isSelected = selectedEvent?.id != null && selectedEvent.id === event?.id;
                    let bg = "transparent";
                    if (event?.type === "HIT") bg = TYPE_COLORS.HIT;
                    else if (event?.type === "INTERCEPTION") bg = TYPE_COLORS.INTERCEPTION;
                    else if (event?.type === "MISS") bg = TYPE_COLORS.MISS;
                    else if (event?.type === "FALSE_ALARM") bg = TYPE_COLORS.FALSE_ALARM;
                    else if (event?.type === "PULSE_ACTIVITY") bg = "rgba(189, 194, 255, 0.35)";

                    return (
                      <button
                        key={column}
                        onClick={() => {
                          if (event) setSelectedEventId(event.id);
                        }}
                        title={
                          event
                            ? `Band ${band} (${event.frequencyMHz} MHz) - ${TYPE_LABELS[event.type] ?? event.type} at ${event.timeUs} µs`
                            : `Band ${band} (${band * 500 + 250} MHz) - Idle`
                        }
                        aria-label={`Band ${band}, ${event ? event.type : "Idle"}`}
                        style={{
                          flex: 1,
                          margin: 0,
                          padding: 0,
                          minWidth: 0,
                          background: bg,
                          border: isSelected
                            ? "1px solid #ffffff"
                            : event
                            ? "1px solid rgba(255,255,255,0.4)"
                            : "1px solid rgba(69,70,83,0.25)",
                          boxShadow: isSelected ? "0 0 8px #ffffff" : undefined,
                          cursor: event ? "pointer" : "default",
                          transition: "background 0.15s ease",
                        }}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div
          className="st-tsm"
          style={{
            display: "flex",
            gap: 16,
            color: "#908f9e",
            marginTop: 8,
            alignItems: "center",
            flexWrap: "wrap",
            borderTop: "1px solid rgba(69,70,83,0.3)",
            paddingTop: 8,
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <i style={{ display: "inline-block", width: 10, height: 10, background: TYPE_COLORS.HIT, borderRadius: 2 }} />
            <strong style={{ color: "#e2e2e8" }}>HIT</strong> (Matched Revisit / Search)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <i style={{ display: "inline-block", width: 10, height: 10, background: TYPE_COLORS.INTERCEPTION, borderRadius: 2 }} />
            <strong style={{ color: "#e2e2e8" }}>INTERCEPTION</strong> (Preemptive Intercept)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <i style={{ display: "inline-block", width: 10, height: 10, background: TYPE_COLORS.MISS, borderRadius: 2 }} />
            <strong style={{ color: "#e2e2e8" }}>MISS</strong> (Zero Pulse Coincidence)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <i style={{ display: "inline-block", width: 10, height: 10, background: TYPE_COLORS.FALSE_ALARM, borderRadius: 2 }} />
            <strong style={{ color: "#e2e2e8" }}>FALSE ALARM</strong> (Spurious Detection)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <i style={{ display: "inline-block", width: 10, height: 10, background: "rgba(189, 194, 255, 0.4)", borderRadius: 2 }} />
            <strong style={{ color: "#e2e2e8" }}>RF ACTIVITY</strong> (Emitter Pulse Stream)
          </span>
        </div>
      </div>

      {/* Feature 9 & 10: View Mode Selector and Console Layout */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 6,
          padding: "6px 10px",
          background: "var(--panel-2, #1a1c20)",
          border: "1px solid var(--border, #454653)",
          borderRadius: 2,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="material-symbols-outlined" style={{ fontSize: 16, color: "var(--accent, #bdc2ff)" }}>
            tune
          </span>
          <span className="st-headline" style={{ fontSize: 11, color: "var(--accent, #bdc2ff)" }}>
            INTERCEPTION CONSOLE VIEW:
          </span>
        </div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {[
            { id: "SPLIT", label: "📊 SPLIT (STREAM + TELEMETRY)" },
            { id: "STREAM_ONLY", label: "📋 STREAM ONLY (EXPANDED)" },
            { id: "TELEMETRY_ONLY", label: "🎯 TELEMETRY ONLY (EXPANDED)" },
          ].map((v) => {
            const isAct = viewLayout === v.id;
            return (
              <button
                key={v.id}
                type="button"
                onClick={() => setViewLayout(v.id)}
                style={{
                  padding: "4px 10px",
                  fontSize: 10,
                  fontWeight: 700,
                  cursor: "pointer",
                  borderRadius: 3,
                  border: isAct ? "1px solid var(--accent, #bdc2ff)" : "1px solid rgba(69, 70, 83, 0.5)",
                  background: isAct ? "rgba(189, 194, 255, 0.2)" : "transparent",
                  color: isAct ? "var(--accent, #bdc2ff)" : "#908f9e",
                  transition: "all 0.15s ease",
                  whiteSpace: "nowrap",
                }}
              >
                {v.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Stream and Selected Event Inspector */}
      <div className="st-grid-12" style={{ alignItems: "stretch", minWidth: 0 }}>
        {/* Stream Table */}
        {(viewLayout === "SPLIT" || viewLayout === "STREAM_ONLY") && (
          <div
            className={`${viewLayout === "SPLIT" ? "st-span-8" : "st-span-12"} st-panel`}
            style={{ display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}
          >
            <PanelHead
              icon="view_timeline"
              title="DWELL INTERCEPTION STREAM"
              badge={`${displayEvents.length} DWELLS`}
              badgeColor="#bdc2ff"
            />

            {/* Tactical Filter Segmented Tabs — CSS Grid 1fr 1fr 1fr (ZERO OVERLAPPING) */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 4,
                background: "#121418",
                padding: 3,
                borderRadius: 4,
                border: "1px solid var(--border, #454653)",
                width: "100%",
                boxSizing: "border-box",
              }}
            >
              {[
                { id: "ALL", label: "ALL EVENTS", count: displayEvents.length, color: "#3097e0" },
                { id: "HITS", label: "HITS", count: hitsCount, color: "#49df9d" },
                { id: "MISSES", label: "MISSES", count: missesCount, color: "#ffb4ab" },
              ].map((tab) => {
                const isActive = filterMode === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setFilterMode(tab.id)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 6,
                      background: isActive ? `${tab.color}28` : "transparent",
                      color: isActive ? tab.color : "#908f9e",
                      border: isActive ? `1px solid ${tab.color}` : "1px solid transparent",
                      borderRadius: 3,
                      padding: "6px 4px",
                      fontSize: 11,
                      fontWeight: 700,
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                      width: "100%",
                      boxSizing: "border-box",
                      minWidth: 0,
                    }}
                  >
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: isActive ? tab.color : "#666", flexShrink: 0 }} />
                    <span style={{ whiteSpace: "nowrap" }}>{tab.label}</span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 800,
                        background: isActive ? tab.color : "#282a30",
                        color: isActive ? "#0d1117" : "#c6c5d5",
                        padding: "1px 6px",
                        borderRadius: 8,
                        flexShrink: 0,
                      }}
                    >
                      {tab.count}
                    </span>
                  </button>
                );
              })}
            </div>

            <div
              className="st-table-wrap st-table-wrap-compact"
              style={{
                maxHeight: "440px",
                overflowY: "auto",
                overflowX: "auto",
                position: "relative",
                border: "1px solid var(--border, #454653)",
                flex: 1,
                minWidth: 0,
              }}
            >
              <table className="st-table" style={{ width: "100%", minWidth: "550px" }}>
                <thead style={{ position: "sticky", top: 0, zIndex: 2, background: "var(--panel-3, #282a2e)" }}>
                  <tr>
                    {["TIME (µs)", "BAND / FREQ", "DWELL MODE", "OUTCOME", "EMITTER / TRK", "TIMING DELTA", "SNR / POWER"].map((c) => (
                      <th key={c} style={{ background: "var(--panel-3, #282a2e)", whiteSpace: "nowrap" }}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredEvents.length === 0 ? (
                    <tr>
                      <td colSpan={7} style={{ textAlign: "center", color: "#908f9e", padding: "36px 8px" }}>
                        NO EVENTS MATCHING FILTER "{filterMode}" — RECEPTOR ACTIVE
                      </td>
                    </tr>
                  ) : (
                    filteredEvents.map((event) => {
                      const isSelected = selectedEvent?.id === event.id;
                      const isHit = event.type === "HIT" || event.type === "INTERCEPTION";
                      return (
                        <tr
                          key={event.id}
                          onClick={() => setSelectedEventId(event.id)}
                          style={{
                            cursor: "pointer",
                            background: isSelected ? "rgba(189, 194, 255, 0.16)" : undefined,
                            borderLeft: isSelected ? "3px solid #bdc2ff" : `3px solid ${TYPE_COLORS[event.type] ?? "transparent"}`,
                          }}
                        >
                          <td style={{ fontWeight: 500, fontFamily: "monospace" }}>{event.timeUs.toLocaleString()} µs</td>
                          <td>
                            <strong>B{event.band}</strong>{" "}
                            <span style={{ color: "#908f9e", fontSize: 11 }}>({event.frequencyMHz.toLocaleString()} MHz)</span>
                          </td>
                          <td>
                            {event.mode}{" "}
                            <span style={{ color: "#908f9e", fontSize: 11 }}>({event.dwellUs} µs)</span>
                          </td>
                          <td>
                            <span
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: 4,
                                padding: "2px 6px",
                                borderRadius: 3,
                                fontSize: 11,
                                fontWeight: 700,
                                background: TYPE_BG[event.type] ?? "rgba(255,255,255,0.08)",
                                color: TYPE_COLORS[event.type] ?? "#e2e2e8",
                                border: `1px solid ${TYPE_COLORS[event.type] ?? "#454653"}40`,
                              }}
                            >
                              <span style={{ width: 6, height: 6, borderRadius: "50%", background: TYPE_COLORS[event.type] }} />
                              {TYPE_LABELS[event.type] ?? event.type}
                            </span>
                          </td>
                          <td>
                            {event.trackId ? (
                              <span style={{ color: isHit ? "#bdc2ff" : "#908f9e", fontWeight: isHit ? 600 : 400 }}>
                                {event.trackId}
                              </span>
                            ) : (
                              <span style={{ color: "#666" }}>Surveillance</span>
                            )}
                          </td>
                          <td>
                            {event.errorUs != null ? (
                              <span style={{ color: Math.abs(event.errorUs) <= 10 ? "#49df9d" : "#ffb4ab", fontFamily: "monospace" }}>
                                {event.errorUs > 0 ? `+${event.errorUs}` : `${event.errorUs}`} µs
                              </span>
                            ) : (
                              <span style={{ color: "#666" }}>—</span>
                            )}
                          </td>
                          <td>
                            {isHit ? (
                              <span style={{ color: "#49df9d", fontSize: 11, fontFamily: "monospace" }}>
                                {event.numPulses} pls · {event.snrDb} dB
                              </span>
                            ) : (
                              <span style={{ color: "#908f9e", fontSize: 11 }}>0 pls · &lt; Floor</span>
                            )}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Feature 10: Rich Tactical Event Telemetry Panel */}
        {(viewLayout === "SPLIT" || viewLayout === "TELEMETRY_ONLY") && (
          <aside
            className={`${viewLayout === "SPLIT" ? "st-span-4" : "st-span-12"} st-panel`}
            style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0, overflow: "hidden" }}
          >
            <PanelHead
              icon="radar"
              title="EVENT TELEMETRY"
              badge={`DWELL #${selectedEvent.step ?? "00"}`}
              badgeColor={TYPE_COLORS[selectedEvent.type] ?? "#bdc2ff"}
            />

            {/* Stepper Sub-Bar with Zero Overlap */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "5px 8px",
                background: "rgba(26, 28, 32, 0.7)",
                border: "1px solid var(--border, #454653)",
                borderRadius: 4,
                gap: 6,
              }}
            >
              <div style={{ fontSize: 11, color: "#908f9e", display: "flex", alignItems: "center", gap: 6, overflow: "hidden" }}>
                <span style={{ whiteSpace: "nowrap" }}>EVENT:</span>
                <strong style={{ color: "#bdc2ff", fontFamily: "monospace", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                  #{selectedEvent.id}
                </strong>
                <span style={{ color: "#666", fontSize: 10, whiteSpace: "nowrap" }}>
                  ({filteredEvents.length > 0 ? (currentIndex >= 0 ? currentIndex + 1 : 1) : 0}/{filteredEvents.length})
                </span>
              </div>
              <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
                <button
                  type="button"
                  onClick={handlePrevEvent}
                  disabled={filteredEvents.length <= 1}
                  title="Select Previous Event"
                  style={{
                    background: "var(--panel-3, #282a2e)",
                    border: "1px solid var(--border, #454653)",
                    color: "var(--accent, #bdc2ff)",
                    borderRadius: 3,
                    padding: "3px 8px",
                    cursor: filteredEvents.length <= 1 ? "not-allowed" : "pointer",
                    fontSize: 10,
                    fontWeight: 700,
                    opacity: filteredEvents.length <= 1 ? 0.5 : 1,
                    whiteSpace: "nowrap",
                  }}
                >
                  ◀ PREV
                </button>
                <button
                  type="button"
                  onClick={handleNextEvent}
                  disabled={filteredEvents.length <= 1}
                  title="Select Next Event"
                  style={{
                    background: "var(--panel-3, #282a2e)",
                    border: "1px solid var(--border, #454653)",
                    color: "var(--accent, #bdc2ff)",
                    borderRadius: 3,
                    padding: "3px 8px",
                    cursor: filteredEvents.length <= 1 ? "not-allowed" : "pointer",
                    fontSize: 10,
                    fontWeight: 700,
                    opacity: filteredEvents.length <= 1 ? 0.5 : 1,
                    whiteSpace: "nowrap",
                  }}
                >
                  NEXT ▶
                </button>
              </div>
            </div>

          {/* Big Status Banner */}
          <div
            style={{
              padding: "10px 14px",
              borderRadius: 4,
              background: TYPE_BG[selectedEvent.type] ?? "rgba(255,255,255,0.05)",
              border: `1px solid ${TYPE_COLORS[selectedEvent.type] ?? "#454653"}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div>
              <div style={{ fontSize: 10, color: "#908f9e", textTransform: "uppercase", letterSpacing: 1 }}>
                INTERCEPTION VERDICT
              </div>
              <div style={{ fontSize: 18, fontWeight: 700, color: TYPE_COLORS[selectedEvent.type] ?? "#e2e2e8" }}>
                {selectedEvent.type === "HIT"
                  ? "● HIT — INTERCEPT CONFIRMED"
                  : selectedEvent.type === "INTERCEPTION"
                  ? "● INTERCEPTION — PREEMPTIVE COINCIDENCE"
                  : selectedEvent.type === "MISS"
                  ? "○ MISS — NO COINCIDENCE DETECTED"
                  : "▲ FALSE ALARM — SPURIOUS ENERGY"}
              </div>
            </div>
            <CmdBadge color={TYPE_COLORS[selectedEvent.type]}>
              {selectedEvent.type === "HIT" || selectedEvent.type === "INTERCEPTION"
                ? `${selectedEvent.numPulses || 1} PULSES`
                : "0 PULSES"}
            </CmdBadge>
          </div>

          {/* Tactical Context Explanation */}
          <div
            style={{
              fontSize: 12,
              color: "#c6c5d5",
              background: "#16181c",
              padding: "8px 10px",
              borderRadius: 4,
              borderLeft: `3px solid ${TYPE_COLORS[selectedEvent.type]}`,
            }}
          >
            {selectedEvent.type === "HIT" || selectedEvent.type === "INTERCEPTION" ? (
              <>
                Receiver dwell synchronized with radar pulse arrival. Successful detection & PDW parameter extraction recorded in Band B{selectedEvent.band} ({selectedEvent.frequencyMHz.toLocaleString()} MHz).
              </>
            ) : (
              <>
                Receiver dwell window executed on Band B{selectedEvent.band} ({selectedEvent.frequencyMHz.toLocaleString()} MHz) for {selectedEvent.dwellUs} µs, but no emitter pulses arrived during this aperture. Cognitive scheduler adapting next dwell interval.
              </>
            )}
          </div>

          {/* Detailed Metric Key-Value Grid */}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {[
              ["EVENT IDENTIFIER", selectedEvent.id],
              ["BAND & FREQUENCY", `Band B${selectedEvent.band} · ${selectedEvent.frequencyMHz.toLocaleString()} MHz`],
              ["DWELL MODE & APERTURE", `${selectedEvent.mode} (${selectedEvent.dwellUs} µs duration)`],
              ["EXECUTION TIME (CLOCK)", `${selectedEvent.timeUs.toLocaleString()} µs`],
              [
                "INTERCEPT TOA",
                selectedEvent.actualUs != null
                  ? `${selectedEvent.actualUs.toLocaleString()} µs`
                  : "None (No Coincidence)",
              ],
              [
                "PREDICTED ETA / DEADLINE",
                selectedEvent.expectedUs != null
                  ? `${selectedEvent.expectedUs.toLocaleString()} µs`
                  : "Unscheduled / Wideband Surveillance",
              ],
              [
                "TIMING ERROR (Δt)",
                selectedEvent.errorUs != null
                  ? `${selectedEvent.errorUs > 0 ? "+" : ""}${selectedEvent.errorUs} µs`
                  : "N/A",
              ],
              [
                "DETECTED PULSES",
                selectedEvent.numPulses > 0
                  ? `${selectedEvent.numPulses} pulse(s) captured`
                  : "0 pulses (Below Threshold)",
              ],
              [
                "SIGNAL POWER & SNR",
                selectedEvent.snrDb != null
                  ? `${selectedEvent.snrDb} dB SNR (${selectedEvent.amplitudeDb} dBm)`
                  : "Noise floor (< -95 dBm)",
              ],
              [
                "PULSE WIDTH (PW)",
                selectedEvent.pulseWidthUs != null
                  ? `${selectedEvent.pulseWidthUs} µs`
                  : "—",
              ],
              [
                "ANGLE OF ARRIVAL (AOA)",
                selectedEvent.aoaDeg != null
                  ? `${selectedEvent.aoaDeg}° Azimuth`
                  : "—",
              ],
              [
                "ASSOCIATED TRACK & EMITTER",
                `${selectedEvent.trackId || "Unassociated"} · ${selectedEvent.emitterId || "—"}`,
              ],
              ["COGNITIVE REASON", selectedEvent.decisionReason],
            ].map(([label, value]) => (
              <div
                key={label}
                className="st-tsm"
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "5px 8px",
                  background: "var(--panel-2, #1a1c20)",
                  border: "1px solid var(--border, #454653)",
                  borderRadius: 3,
                }}
              >
                <span style={{ color: "var(--muted, #908f9e)", fontSize: 11 }}>{label}</span>
                <strong
                  style={{
                    color:
                      label === "INTERCEPTION VERDICT" || label === "SIGNAL POWER & SNR"
                        ? selectedEvent.type === "HIT" || selectedEvent.type === "INTERCEPTION"
                          ? "var(--success, #49df9d)"
                          : "var(--danger, #ffb4ab)"
                        : "var(--text, #e2e2e8)",
                    fontFamily:
                      label.includes("TIME") || label.includes("FREQ") || label.includes("TOA") || label.includes("ERROR")
                        ? "monospace"
                        : "inherit",
                    fontSize: 12,
                    textAlign: "right",
                    maxWidth: "60%",
                    wordBreak: "break-word",
                  }}
                >
                  {value}
                </strong>
              </div>
            ))}
          </div>
        </aside>
      )}
    </div>
  </div>
);
}