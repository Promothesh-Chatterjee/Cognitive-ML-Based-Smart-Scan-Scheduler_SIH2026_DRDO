/* Shared Stitch tactical primitives. Appearance must not change per-page. */

export function CmdBadge({ children, color = "var(--text, #e2e2e8)" }) {
  return (
    <span className="st-badge" style={{ color }}>
      {children}
    </span>
  );
}

export function PanelHead({ icon, title, badge, badgeColor = "var(--secondary, #96ccff)" }) {
  return (
    <div className="st-panel-head">
      <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {icon && (
          <span className="material-symbols-outlined" style={{ fontSize: 16, color: "var(--secondary, #96ccff)" }}>
            {icon}
          </span>
        )}
        <span className="st-headline" style={{ color: "var(--accent, #bdc2ff)" }}>
          {title}
        </span>
      </span>
      {badge && <CmdBadge color={badgeColor}>{badge}</CmdBadge>}
    </div>
  );
}

export function KpiCard({ label, icon, value, unit, footLeft, footRight, valueColor = "var(--text, #e2e2e8)" }) {
  return (
    <div className="st-kpi">
      <div className="st-kpi-top">
        <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>
          {label}
        </span>
        {icon && (
          <span className="material-symbols-outlined" style={{ fontSize: 14, color: "var(--muted, #908f9e)" }}>
            {icon}
          </span>
        )}
      </div>
      <div className="st-tlg" style={{ color: valueColor }}>
        {value} {unit && <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>{unit}</span>}
      </div>
      {(footLeft || footRight) && (
        <div className="st-kpi-foot">
          <span>{footLeft}</span>
          <span>{footRight}</span>
        </div>
      )}
    </div>
  );
}

const PIPE_NODES = [
  ["01", "18 GHz RF", "Spectrum Base", "#bdc2ff"],
  ["02", "1 GHz Rx", "Instantaneous IBW", "#96ccff"],
  ["03", "PDW Detector", "SNR \u2265 15 dB Floor", "#e2e2e8"],
  ["04", "360-D State", "Matrix (36\u00d710)", "#dfe0ff"],
  ["AI", "DRQN + MoE", "Recurrent Core", "#49df9d", true],
  ["06", "180 Actions", "36 Bands \u00d7 5 Modes", "#e2e2e8"],
  ["07", "Band Decision", "", "#96ccff"],
  ["08", "T-F Intercept", "Coincidence Match", "#49df9d"],
  ["09", "Reward Opt", "Temporal Update", "#e2e2e8"],
];

export function PipelineFlow() {
  return (
    <section className="st-pipe" aria-label="Architectural pipeline flow">
      {PIPE_NODES.map(([num, title, sub, color, ai], i) => (
        <span key={num + i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span className={ai ? "st-node st-node-ai" : "st-node"}>
            <CmdBadge color={color}>{num}</CmdBadge>
            <span style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color, lineHeight: 1 }}>
                {title}
              </span>
              <span className="st-mark" style={{ color: "#908f9e" }}>
                {sub}
              </span>
            </span>
          </span>
          {i < PIPE_NODES.length - 1 && (
            <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          )}
        </span>
      ))}
    </section>
  );
}


export function BandMatrix({
  currentBand,
  tuneBand = 16,
  dwellUs = 120,
  freqMHz = null,
  bandHeights = null,
  bandStates = null,
}) {
  const N = 36;
  const safeTune = Math.max(0, Math.min(N - 1, Number(currentBand != null ? currentBand : tuneBand) || 0));
  const activeFreqMHz = freqMHz != null ? freqMHz : (safeTune * 500 + 250);

  // Display real channel occupancy activity from telemetry; tuned receiver aperture is separately indicated by the IBW frame
  const heights = Array.isArray(bandHeights) && bandHeights.length === N
    ? bandHeights.map((h) => Math.max(3, Math.round(Number(h) * 100)))
    : Array.from({ length: N }, () => 3);
  const states = Array.isArray(bandStates) && bandStates.length === N
    ? bandStates.map((s, i) => (i === safeTune ? "active" : s))
    : Array.from({ length: N }, (_, i) => (i === safeTune ? "active" : "quiet"));

  // 1 GHz IBW aperture covers 2 adjacent 500 MHz channels around the tune window
  const ibwLo = Math.max(0, safeTune - 1);
  const ibwHi = Math.min(N - 1, safeTune + 1);
  const ibwLoGHz = (ibwLo * 0.5).toFixed(1);
  const ibwHiGHz = (ibwHi * 0.5).toFixed(1);

  const tuneGHz = (activeFreqMHz / 1000).toFixed(3);

  // Position aperture overlay centered around tuned band (1 GHz IBW = 2 bands)
  const centerPct = ((safeTune + 0.5) / N) * 100;
  const boxWidthPct = (2 / N) * 100; // 1 GHz / 18 GHz = ~5.56% aperture frame
  const leftPct = Math.max(0, Math.min(100 - boxWidthPct, centerPct - boxWidthPct / 2));

  return (
    <div className="st-spec">
      <div
        style={{ position: "relative", height: 18, padding: "2px 4px" }}
        className="st-tsm"
      >
        <div style={{ display: "flex", justifyContent: "space-between", color: "var(--muted, #908f9e)" }}>
          {["0 GHz", "2.0 GHz", "4.0 GHz", "6.0 GHz", "8.0 GHz", "10.0 GHz", "12.0 GHz", "14.0 GHz", "16.0 GHz", "18.0 GHz"].map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
        <span
          style={{
            position: "absolute",
            top: 1,
            left: `${centerPct}%`,
            transform: "translateX(-50%)",
            color: "var(--accent, #bdc2ff)",
            fontWeight: 700,
            whiteSpace: "nowrap",
            background: "var(--panel-2, #1e2024)",
            padding: "0 4px",
            border: "1px solid var(--accent, #bdc2ff)",
            boxShadow: "0 0 8px rgba(0,0,0,0.5)",
            transition: "left 0.25s ease",
            zIndex: 3,
          }}
        >
          {tuneGHz} GHz ↑
        </span>
      </div>
      <div style={{ position: "relative" }}>
        <div className="st-bars">
          {Array.from({ length: N }, (_, i) => {
            const state = states[i];
            const h = heights[i];
            const isActive = i === safeTune;
            return (
              <div
                key={i}
                className={`st-bar st-bar-${state}`}
                title={`B${String(i + 1).padStart(2, "0")} ${i * 500}–${(i + 1) * 500} MHz [${state.toUpperCase()}]`}
                style={{
                  height: `${h}%`,
                  boxShadow: isActive
                    ? "0 0 12px rgba(189,194,255,0.85)"
                    : "none",
                  opacity: state === "quiet" ? 0.2 : 1,
                  transition: "height 0.25s ease, background 0.25s ease, opacity 0.25s ease",
                }}
              />
            );
          })}
        </div>
        <div
          className="st-ibw"
          style={{
            left: `${leftPct}%`,
            width: `${boxWidthPct}%`,
            transition: "left 0.25s ease, width 0.25s ease",
          }}
        >
          <span className="st-badge" style={{ color: "var(--accent, #bdc2ff)" }}>
            IBW RX: {ibwLoGHz}–{ibwHiGHz} GHz
          </span>
          <span className="st-mark" style={{ color: "var(--accent, #bdc2ff)", textAlign: "center" }}>
            1 GHz IBW LOCKED
          </span>
          <span className="st-badge" style={{ color: "var(--accent, #bdc2ff)", justifyContent: "center" }}>
            DWELL: {Number(dwellUs).toFixed(0)}µs
          </span>
        </div>
      </div>
      <div
        className="st-mark"
        style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", color: "var(--muted, #908f9e)" }}
      >
        {["B01", "B04", "B08", "B12", "B16", "B20", "B24", "B28", "B32", "B36"].map((t) => {
          const bNum = parseInt(t.replace("B", ""), 10) - 1;
          const isNear = Math.abs(bNum - safeTune) <= 1;
          return (
            <span key={t} style={isNear ? { color: "var(--accent, #bdc2ff)", fontWeight: 700 } : undefined}>
              {t}{bNum === safeTune ? " ↑" : ""}
            </span>
          );
        })}
      </div>
    </div>
  );
}


const DWELL_KIND_COLORS = {
  hit:  "#49df9d",
  miss: "#908f9e",
  now:  "#0b1c93",
};

const STATIC_DWELLS = [
  { id: "1", band: 4,  freqMHz: 2250,  mode: "PREEMPTIVE_INTERCEPT", hit: true,  dwellUs: 80,  now: false, clockUs: -500000 },
  { id: "2", band: 12, freqMHz: 6250,  mode: "NORMAL_DWELL",         hit: false, dwellUs: 110, now: false, clockUs: -400000 },
  { id: "3", band: 24, freqMHz: 12250, mode: "SHORT_DWELL",          hit: false, dwellUs: 60,  now: false, clockUs: -300000 },
  { id: "4", band: 6,  freqMHz: 3250,  mode: "SEARCH",               hit: true,  dwellUs: 140, now: false, clockUs: -200000 },
  { id: "5", band: 28, freqMHz: 14250, mode: "LONG_DWELL",           hit: true,  dwellUs: 100, now: false, clockUs: -100000 },
  { id: "6", band: 16, freqMHz: 8250,  mode: "REVISIT",              hit: false, dwellUs: 120, now: true,  clockUs: 0 },
];

/**
 * DwellTimeline accepts live `entries` array from useOverviewTelemetry.
 * Each entry: { id, band, freqMHz, mode, hit, dwellUs, clockUs, now }
 * Falls back to static placeholder when no live data.
 */
export function DwellTimeline({ entries = null }) {
  const data = entries && entries.length > 0 ? entries : STATIC_DWELLS;
  // Pad to always show 6 slots
  const slots = [...data];
  while (slots.length < 6) {
    slots.unshift({ id: `pad-${slots.length}`, band: "—", mode: "—", hit: null, dwellUs: 0, now: false, freqMHz: 0, clockUs: 0 });
  }

  // Compute relative timestamps from the newest entry
  const newestClock = slots[slots.length - 1]?.clockUs ?? 0;

  return (
    <div className="st-panel">
      <PanelHead icon="timeline" title="DWELL & INTERCEPT EVENT TIMELINE" badge="LAST 6 DWELLS" />
      <div className="st-timeline">
        {slots.map((entry, idx) => {
          const kind = entry.now ? "now" : entry.hit ? "hit" : "miss";
          const label = entry.band !== "—"
            ? `B${String(Number(entry.band) + 1).padStart(2, "0")}: ${entry.mode}`
            : "—";
          const sub = entry.now
            ? `DWELL: ${Number(entry.dwellUs).toFixed(0)}µs · ARMED [NOW]`
            : entry.band !== "—"
            ? `${Number(entry.dwellUs).toFixed(0)}µs · ${entry.hit ? "HIT ✓" : "MISS"}`
            : "—";

          return (
            <div
              key={entry.id ?? idx}
              className={kind === "now" ? "st-dwell st-dwell-now" : "st-dwell"}
              style={{ flex: 1 }}
            >
              <span className="st-badge" style={{ color: kind === "now" ? "#0b1c93" : DWELL_KIND_COLORS[kind] }}>
                {label}
              </span>
              <span className="st-mark" style={{ color: kind === "now" ? "#0b1c93" : "#908f9e" }}>
                {sub}
              </span>
            </div>
          );
        })}
      </div>
      <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}>
        {slots.map((entry, idx) => {
          const deltUs = entry.clockUs ? entry.clockUs - newestClock : null;
          if (entry.now) return <span key={idx}>NOW [T-0]</span>;
          if (deltUs === null || entry.band === "—") return <span key={idx}>—</span>;
          const ms = Math.round(deltUs / 1000);
          return <span key={idx}>T{ms} ms</span>;
        })}
      </div>
    </div>
  );
}


export function CandidateActions() {
  const rows = [
    ["01", "B16 // REVISIT", "Q: 0.941", "+18.4 U", true],
    ["02", "B06 // SEARCH", "Q: 0.722", "+11.2 U", false],
    ["03", "B28 // NORMAL", "Q: 0.615", "+8.9 U", false],
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span className="st-headline" style={{ color: "var(--text, #e2e2e8)" }}>
        ACTION USAGE
      </span>
      {rows.map(([n, act, q, u, sel]) => (
        <div
          key={n}
          className="st-tsm"
          style={{
            display: "flex",
            justifyContent: "space-between",
            background: sel ? "var(--panel-2, #1e2024)" : "var(--panel, #1a1c20)",
            border: "1px solid var(--border, #454653)",
            padding: "4px 6px",
          }}
        >
          <span style={{ display: "flex", gap: 6 }}>
            <CmdBadge color={sel ? "var(--success, #49df9d)" : "var(--text, #e2e2e8)"}>{n}</CmdBadge>
            <strong style={{ color: "var(--text, #e2e2e8)" }}>{act}</strong>
          </span>
          <span style={{ display: "flex", gap: 8 }}>
            <span style={{ color: "var(--text, #e2e2e8)" }}>{q}</span>
            <strong style={{ color: "var(--text, #e2e2e8)" }}>{u}</strong>
          </span>
        </div>
      ))}
    </div>
  );
}

export function StitchTable({ columns, rows, caption }) {
  return (
    <div className="st-table-wrap">
      {caption && (
        <div className="st-tsm" style={{ padding: "4px 8px", color: "#908f9e" }}>
          {caption}
        </div>
      )}
      <table className="st-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TruthBanner() {
  return (
    <div className="st-truth" role="note" aria-label="Simulation truth banner">
      <span className="material-symbols-outlined" style={{ fontSize: 16, color: "#f59e0b" }}>
        warning
      </span>
      <span className="st-body-bold">SIMULATION TRUTH // OPERATOR VIEW ONLY</span>
      <span className="st-body" style={{ color: "#c6c5d5" }}>
        Isolated ground-truth store. Never fed to scheduler observations.
      </span>
    </div>
  );
}

export function DataSourceBadge({ connected }) {
  return (
    <span className="st-badge" style={{ color: connected ? "#49df9d" : "#f59e0b" }}>
      <span
        style={{
          width: 6,
          height: 6,
          background: connected ? "#49df9d" : "#f59e0b",
          display: "inline-block",
        }}
      />
      {connected ? "LIVE BACKEND (CLOUD RUN)" : "SYNTHETIC DEMO / TSRD REPLAY"}
    </span>
  );
}
