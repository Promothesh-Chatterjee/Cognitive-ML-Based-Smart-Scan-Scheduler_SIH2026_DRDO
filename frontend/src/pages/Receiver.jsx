import { useMemo, useState, useEffect, useRef } from "react";
import {
  DataSourceBadge,
  KpiCard,
  PanelHead,
  StitchTable,
} from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

const FREQUENCY_RANGE = { min: 0, max: 18000 };
const NUM_BANDS = 36;

const FALLBACK_RECEIVER_PDWS = [
  { id: 3561, toaUs: 356100.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3556, toaUs: 355600.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -54.0, snrDb: 41.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3553, toaUs: 355300.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3552, toaUs: 355200.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -56.0, snrDb: 39.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3549, toaUs: 354900.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3544, toaUs: 354400.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3539, toaUs: 353900.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -54.0, snrDb: 41.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3532, toaUs: 353200.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3525, toaUs: 352500.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3518, toaUs: 351800.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -56.0, snrDb: 39.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3511, toaUs: 351100.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, status: "DETECTED" },
  { id: 3504, toaUs: 350400.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -54.0, snrDb: 41.0, aoaDeg: 12.0, status: "DETECTED" },
];

// Normalise a raw backend PDW record into a display object
function normReceiverPdw(p, idx, currentBand = 0) {
  const pId = p.pulse_id != null ? p.pulse_id : (p.id != null ? p.id : `P-${idx + 1}`);
  const tUs = Number(p.time_us != null ? p.time_us : (p.toa_us != null ? p.toa_us : (p.toaUs != null ? p.toaUs : 0.0)));
  const fMhz = Number(p.frequency_mhz != null ? p.frequency_mhz : (p.freq_mhz != null ? p.freq_mhz : (p.frequencyMHz != null ? p.frequencyMHz : (currentBand * 500 + 250))));
  const pwUs = Number(p.pulse_width_us != null ? p.pulse_width_us : (p.pw_us != null ? p.pw_us : (p.pulseWidthUs != null ? p.pulseWidthUs : 1.0)));
  const ampDb = Number(p.amplitude_db != null ? p.amplitude_db : (p.amp_db != null ? p.amp_db : (p.amplitudeDb != null ? p.amplitudeDb : -65.0)));
  const snrVal = p.snr_db != null ? Number(p.snr_db) : (p.snrDb != null ? Number(p.snrDb) : Number((ampDb + 95.0).toFixed(1)));
  const aoaVal = Number(p.aoa_deg != null ? p.aoa_deg : (p.aoa != null ? p.aoa : (p.aoaDeg != null ? p.aoaDeg : 0.0)));
  return {
    id: pId,
    toaUs: tUs,
    frequencyMHz: fMhz,
    pulseWidthUs: pwUs,
    amplitudeDb: ampDb,
    snrDb: snrVal,
    aoaDeg: aoaVal,
    status: "DETECTED",
  };
}

export default function Receiver() {
  const t = useOverviewTelemetry();
  const {
    wsStatus,
    live,
    currentBand,
    currentDwellUs,
    bandHeights,
    pdws,
    allIncidentPdws,
  } = t;

  const [manualBand, setManualBand] = useState(null);
  const activeBand = manualBand !== null ? manualBand : (currentBand != null ? currentBand : 0);
  const centerFrequencyMHz = activeBand * 500 + 250;
  const dwellTimeUs = live ? currentDwellUs : (currentDwellUs || 500.0);
  const thresholdDb = -140.0;

  const centerPct = ((activeBand + 0.5) / NUM_BANDS) * 100;
  const boxWidthPct = (2 / NUM_BANDS) * 100; // 1 GHz = 2 x 500 MHz channels = ~5.56% aperture frame
  const leftPct = Math.max(0, Math.min(100 - boxWidthPct, centerPct - boxWidthPct / 2));

  const windowStart = useMemo(
    () => Math.max(FREQUENCY_RANGE.min, centerFrequencyMHz - 500),
    [centerFrequencyMHz]
  );
  const windowEnd = useMemo(
    () => Math.min(FREQUENCY_RANGE.max, centerFrequencyMHz + 500),
    [centerFrequencyMHz]
  );

  // Rolling last 30 hit records from the receiver dwell detections
  const [pdwHistory, setPdwHistory] = useState([]);
  const seenPulseIds = useRef(new Set());

  useEffect(() => {
    if (!Array.isArray(pdws) || pdws.length === 0) return;
    setPdwHistory((prev) => {
      const incoming = [];
      for (let i = 0; i < pdws.length; i++) {
        const p = pdws[i];
        const norm = normReceiverPdw(p, i, currentBand);
        const uid = `${norm.id}-${norm.toaUs.toFixed(1)}`;
        if (!seenPulseIds.current.has(uid)) {
          seenPulseIds.current.add(uid);
          incoming.push(norm);
        }
      }
      if (incoming.length === 0) return prev;
      return [...incoming, ...prev].slice(0, 30);
    });
  }, [pdws, currentBand]);

  // PDW stream: detected (HIT) pulses only, newest-first — used in PDW stream table
  const pdwStream = useMemo(() => {
    if (pdwHistory.length > 0) return pdwHistory.slice(0, 30);
    if (Array.isArray(pdws) && pdws.length > 0) {
      return pdws.map((p, idx) => normReceiverPdw(p, idx, currentBand)).slice(0, 30);
    }
    return FALLBACK_RECEIVER_PDWS;
  }, [pdwHistory, pdws, currentBand]);

  // Scroll ref — auto-scroll the PDW stream table container to top when new pulses arrive
  const pdwTableRef = useRef(null);
  useEffect(() => {
    if (pdwTableRef.current) {
      pdwTableRef.current.scrollTop = 0;
    }
  }, [pdwHistory]);

  // Set of recently observed bands from detected pulse history
  const observedBands = useMemo(() => {
    const s = new Set();
    pdwStream.forEach((p) => {
      const b = Math.floor(p.frequencyMHz / 500);
      if (b >= 0 && b < NUM_BANDS) s.add(b);
    });
    return s;
  }, [pdwStream]);

  const hitCount  = pdwStream.length;
  const missCount = Array.isArray(allIncidentPdws) ? Math.max(0, allIncidentPdws.length - hitCount) : 0;

  const kpis = [
    ["TOTAL BANDWIDTH", "18.00", "GHz", "0–18,000 MHz", "CANONICAL"],
    ["INSTANTANEOUS BW", live ? "1.00" : "0.0", "GHz", "Current receiver IBW", "1,000 MHz"],
    ["FREQUENCY STEP", live ? "500" : "0.0", "MHz", "Tunable increment", "CANONICAL"],
    ["DETECTION THRESHOLD", live ? "-140" : "0.0", "dBm", "Production sensitivity", "HARDWARE GATE"],
    ["CURRENT DWELL", live ? `${Number(dwellTimeUs).toFixed(0)}` : "0.0", "µs", "Active receiver dwell", "CLOSED-LOOP"],
    ["PDWS DETECTED", live ? `${hitCount}` : "0.0", "", live ? `Last ${pdwStream.length} intercepted` : "Awaiting stream", "HITS ONLY"],
    ["LIVE STREAM", live ? "ACTIVE" : (wsStatus === "ONLINE" ? "ONLINE" : "0.0"), "", "Pulse descriptor words", "REAL RF FEED"],
  ];


  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* 1. RECEIVER TELEMETRY & PULSE DESCRIPTOR WORD (PDW) PIPELINE */}
      <div className="st-panel">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <PanelHead
            icon="settings_input_antenna"
            title="RECEIVER TELEMETRY & PULSE DESCRIPTOR WORD (PDW) PIPELINE"
            badge={live ? "RECEIVER ONLINE" : (wsStatus === "ONLINE" ? "STANDBY" : "0.0")}
            badgeColor={live ? "#49df9d" : "#96ccff"}
          />
          <DataSourceBadge connected={live} />
        </div>
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Narrow-IBW receiver state, causal pulse detection, and instantaneous RF observation window.
        </div>
      </div>

      <section className="st-kpi-grid" aria-label="Receiver KPI strip">
        {kpis.map(([label, value, unit, footLeft, footRight]) => (
          <KpiCard
            key={label}
            label={label}
            value={value}
            unit={unit}
            footLeft={footLeft}
            footRight={footRight}
            valueColor={label === "LIVE STREAM" && live ? "#49df9d" : label === "CURRENT DWELL" ? "#bdc2ff" : "#e2e2e8"}
          />
        ))}
      </section>

      {/* 2 & 3. 18.0 GHz SURVEILLANCE APERTURE ENVELOPE + RECEIVER STATE */}
      <div className="st-grid-12">
        {/* 2. Chart of 18.0 GHz SURVEILLANCE APERTURE ENVELOPE */}
        <div className="st-span-8 st-panel">
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
            <PanelHead
              icon="show_chart"
              title="18.0 GHz SURVEILLANCE APERTURE ENVELOPE"
              badge={live ? `${centerFrequencyMHz.toLocaleString()} MHz` : "0.0 MHz"}
              badgeColor="#96ccff"
            />
            {manualBand !== null && (
              <button
                type="button"
                style={{
                  background: "transparent",
                  border: "1px solid var(--accent, #bdc2ff)",
                  color: "var(--accent, #bdc2ff)",
                  fontSize: 10,
                  padding: "2px 8px",
                  cursor: "pointer",
                  borderRadius: 2,
                }}
                onClick={() => setManualBand(null)}
              >
                FOLLOW LIVE RECEIVER
              </button>
            )}
          </div>
          <div className="st-spec">
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", color: "#908f9e" }}>
              <span>0 GHz</span><span>4 GHz</span><span>8 GHz</span><span>12 GHz</span><span>16 GHz</span><span>18 GHz</span>
            </div>
            <div style={{ position: "relative", height: "auto" }}>
              <div className="st-bars" style={{ height: 180 }}>
                {Array.from({ length: NUM_BANDS }, (_, i) => {
                  const isTuned = i === activeBand;
                  const isThreat = observedBands.has(i) || (live && bandHeights && bandHeights[i] > 0.10);
                  let h = 3;
                  if (live) {
                    if (isTuned) h = 94;
                    else if (isThreat) h = Math.max(20, Math.min(85, Math.round((bandHeights?.[i] ?? 0.05) * 100)));
                    else h = Math.max(5, Math.round((bandHeights?.[i] ?? 0.02) * 40));
                  }

                  return (
                    <div
                      key={i}
                      className="st-bar"
                      title={`BAND ${String(i + 1).padStart(2, "0")} (${i * 500}–${(i + 1) * 500} MHz) [${isTuned ? "TUNED APERTURE" : isThreat ? "ACTIVE THREAT" : "QUIET"}]`}
                      style={{
                        height: `${h}%`,
                        background: isTuned
                          ? "var(--accent, #bdc2ff)"
                          : isThreat
                          ? "var(--secondary-deep, #3097e0)"
                          : "var(--panel-3, #282a2e)",
                        boxShadow: isTuned ? "0 0 12px rgba(189,194,255,0.85)" : "none",
                        opacity: isTuned ? 1 : isThreat ? 0.8 : 0.35,
                        cursor: "pointer",
                        transition: "height 0.25s ease, background 0.25s ease, opacity 0.25s ease",
                      }}
                      onClick={() => setManualBand(i)}
                    />
                  );
                })}
              </div>
              <div
                className="st-ibw"
                style={{
                  position: "absolute",
                  left: `${leftPct}%`,
                  width: `${boxWidthPct}%`,
                  top: 0,
                  bottom: 0,
                  transition: "left 0.25s ease, width 0.25s ease",
                }}
              >
                <span className="st-badge" style={{ color: "var(--secondary, #96ccff)" }}>1 GHz RECEIVER WINDOW</span>
                <span className="st-mark" style={{ color: "var(--secondary, #96ccff)", textAlign: "center" }}>
                  {centerFrequencyMHz.toLocaleString()} MHz
                </span>
              </div>
            </div>
          </div>

          <div className="st-tsm" style={{ display: "flex", gap: 8, color: "#908f9e", marginTop: 4 }}>
            {[0, 4, 8, 12, 16, 18].map((g) => (
              <span key={g} style={{ flex: 1 }}>{g} GHz</span>
            ))}
          </div>
        </div>

        {/* 3. RECEIVER STATE */}
        <aside className="st-span-4 st-panel">
          <PanelHead
            title="RECEIVER STATE"
            badge={live ? "ACTIVE" : (wsStatus === "ONLINE" ? "STANDBY" : "0.0")}
            badgeColor={live ? "#49df9d" : "#96ccff"}
          />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="st-tmd" style={{ color: "#bdc2ff" }}>
              {live ? `${centerFrequencyMHz.toLocaleString()} MHz` : "0.0 MHz"}
            </div>
            <div className="st-tsm" style={{ color: "#908f9e" }}>
              Window: {live ? `${windowStart.toLocaleString()}–${windowEnd.toLocaleString()} MHz` : "0.0–0.0 MHz"}
            </div>
            {[
              ["STATUS", live ? "ACTIVE" : (wsStatus === "ONLINE" ? "STANDBY" : "0.0")],
              ["IBW", live ? "1 GHz (1,000 MHz)" : "0.0 MHz"],
              ["STEP", live ? "500 MHz" : "0.0 MHz"],
              ["HITS", live ? `${hitCount} DETECTED` : "0.0"],
              ["MISSES", live ? `${missCount} INCIDENT` : "0.0"],
              ["THRESHOLD", live ? `${thresholdDb.toFixed(0)} dBm` : "0.0 dBm"],
            ].map(([label, value]) => (
              <div
                key={label}
                className="st-tsm"
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "4px 6px",
                  background: "var(--panel-2, #1a1c20)",
                  border: "1px solid var(--border, #454653)",
                }}
              >
                <span style={{ color: "var(--muted, #908f9e)" }}>{label}</span>
                <strong style={{
                  color: label === "STATUS" && live ? "var(--success, #49df9d)"
                    : label === "HITS" ? "var(--success, #49df9d)"
                    : label === "MISSES" ? "var(--danger, #ff6b6b)"
                    : "var(--text, #e2e2e8)"
                }}>
                  {value}
                </strong>
              </div>
            ))}
          </div>
          <div className="st-truth" style={{ marginTop: 6 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 14, color: "#49df9d" }}>
              visibility
            </span>
            <span className="st-body" style={{ color: "#c6c5d5" }}>
              Causal receiver telemetry observable from the live RF stream. Ground-truth emitter identity is excluded.
            </span>
          </div>
        </aside>
      </div>

      {/* 4. LIVE PULSE DESCRIPTOR WORD (PDW) STREAM */}
      <div className="st-panel">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
          <PanelHead
            icon="stream"
            title="LIVE PULSE DESCRIPTOR WORD (PDW) STREAM"
            badge={
              pdwStream.length > 0
                ? `${pdwStream.length} RECORDS`
                : (live ? "AWAITING INTERCEPTIONS" : "0.0")
            }
            badgeColor={pdwStream.length > 0 ? "#49df9d" : "#96ccff"}
          />
          {live && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 10, color: "#908f9e", flexShrink: 0 }}>
              <span className="st-badge" style={{ color: "#49df9d", border: "1px solid #49df9d22", background: "rgba(73, 223, 157, 0.08)", padding: "2px 6px" }}>
                ● STREAMING REAL-TIME INTERCEPTIONS
              </span>
            </div>
          )}
        </div>
        <div ref={pdwTableRef} style={{ maxHeight: 340, overflowY: "auto", transition: "all 0.2s ease" }}>
          <StitchTable
            columns={[
              "PULSE ID",
              "TIME OF ARRIVAL (TOA UTC)",
              "FREQUENCY (MHZ)",
              "PW (µS)",
              "AMP (DBM)",
              "SNR (DB)",
              "AOA (DEG)",
              "STATUS",
            ]}
            rows={
              pdwStream.length > 0
                ? pdwStream.map((pdw, idx) => {
                    const isNewest = idx === 0;
                    return [
                      <span key="id" style={{ fontWeight: isNewest ? 700 : 500, color: isNewest ? "#bdc2ff" : "#e2e2e8" }}>
                        {pdw.id}
                      </span>,
                      `${pdw.toaUs.toFixed(1)} µs`,
                      pdw.frequencyMHz.toFixed(1),
                      pdw.pulseWidthUs.toFixed(2),
                      pdw.amplitudeDb.toFixed(1),
                      pdw.snrDb.toFixed(1),
                      `${pdw.aoaDeg.toFixed(1)}°`,
                      <span
                        key={`${pdw.id}-${pdw.toaUs}`}
                        style={{
                          color: "#49df9d",
                          background: "rgba(73, 223, 157, 0.1)",
                          fontWeight: 700,
                          fontSize: 10,
                          padding: "2px 6px",
                          border: "1px solid #49df9d",
                          borderRadius: 2,
                          letterSpacing: 1,
                          display: "inline-block",
                        }}
                      >
                        DETECTED
                      </span>,
                    ];
                  })
                : [["—", "—", "—", "—", "—", "—", "—",
                    <span key="empty" style={{ color: "var(--muted, #908f9e)" }}>—</span>]]
            }
          />
        </div>
        <div className="st-tsm" style={{ color: "#908f9e", marginTop: 4, display: "flex", justifyContent: "space-between" }}>
          <span>
            Real-time causal intercepted Pulse Descriptor Words (PDWs) output by wideband front-end channelizers and demodulators.
          </span>
          <span style={{ color: "#49df9d" }}>
            AUTO-STREAMING • NEWEST FIRST
          </span>
        </div>
      </div>
    </div>
  );
}