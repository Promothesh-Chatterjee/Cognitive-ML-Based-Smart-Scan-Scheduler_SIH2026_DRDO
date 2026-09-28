export function SpectrumWaterfall({ history = [] }) {
  const N_BANDS = 36;
  const N_STEPS = Math.min(history.length, 50);
  const recent = history.slice(-N_STEPS);

  if (!history || history.length === 0) {
    return (
      <div className="waterfall-wrap">
        <div className="waterfall-title" style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8, fontSize: 12, fontWeight: 700 }}>
          <span>36-CHANNEL ACTIVITY WATERFALL</span>
        </div>
        <div
          style={{
            padding: '24px 12px',
            textAlign: 'center',
            color: 'var(--muted, #a8a7b8)',
            background: 'var(--panel-2, #1e2024)',
            border: '1px solid var(--border-subtle, #333539)',
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: 11,
          }}
        >
          NO CHANNEL ACTIVITY FRAMES RECORDED · AWAITING ACTIVE MISSION TELEMETRY
        </div>
      </div>
    );
  }

  const getCellColor = (step, band) => {
    const s = recent[step];
    if (!s) return 'var(--panel-lowest, #0a1628)';
    const isTuned = (s.tuned_band ?? s.last_band) === band;
    const isThreat = s.active_bands?.includes(band);
    const occupancy = Number(s.channel_occupancy?.[band] ?? s.band_priorities?.[band] ?? s.band_powers?.[band] ?? 0);

    if (isTuned) return 'var(--accent, #bdc2ff)'; // receiver aperture tuned here
    if (isThreat) return 'var(--danger, #ef4444)'; // active incident emitter signal
    if (occupancy > 0.5) return 'rgba(150, 204, 255, 0.75)'; // high channel occupancy / priority
    if (occupancy > 0.15) return 'rgba(100, 150, 220, 0.35)'; // moderate channel occupancy / priority
    return 'var(--panel-3, #0f2040)'; // quiescent
  };

  return (
    <div className="waterfall-wrap">
      <div className="waterfall-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 6, fontSize: 12, fontWeight: 700 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span>REAL-TIME 36-CHANNEL ACTIVITY WATERFALL (last {N_STEPS} steps)</span>
          <span style={{ fontSize: 10, fontWeight: 400, color: 'var(--muted, #908f9e)' }}>
            [Occupancy / Activity Priority · 36 Bands × 500 MHz]
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11 }}>
          <span style={{ color: 'var(--accent, #bdc2ff)' }}>■ Receiver Tuned</span>
          <span style={{ color: 'var(--danger, #ef4444)' }}>■ Incident Emitter</span>
          <span style={{ color: 'rgba(150, 204, 255, 0.75)' }}>■ High Occupancy</span>
        </div>
      </div>
      <div
        className="waterfall-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${N_BANDS}, 1fr)`,
          gap: '1px',
          width: '100%',
          background: 'rgba(0,0,0,0.2)',
          padding: 2,
          border: '1px solid var(--border-dim, #1e293b)',
        }}
      >
        {Array.from({ length: N_STEPS }).map((_, t) =>
          Array.from({ length: N_BANDS }).map((_, b) => {
            const frame = recent[t];
            const occupancy = Number(frame?.channel_occupancy?.[b] ?? frame?.band_priorities?.[b] ?? frame?.band_powers?.[b] ?? 0);
            const timeUs = frame?.timestamp != null ? Number(frame.timestamp) : null;
            const timeLabel = timeUs != null && timeUs > 0 ? `T+${(timeUs / 1000).toFixed(1)}ms` : `Step ${frame?.step ?? t}`;
            return (
              <div
                key={`${t}-${b}`}
                style={{
                  height: '8px',
                  background: getCellColor(t, b),
                  borderRadius: '1px',
                }}
                title={`${timeLabel} · Band B${String(b + 1).padStart(2, '0')} (${b * 500}–${(b + 1) * 500} MHz) · Occupancy Priority: ${(occupancy * 100).toFixed(1)}%`}
              />
            );
          })
        )}
      </div>
      <div
        className="waterfall-axis"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 4,
          fontSize: 10,
          color: 'var(--muted, #a8a7b8)',
          fontFamily: 'var(--font-mono, monospace)',
        }}
      >
        {[0, 9, 17, 26, 35].map(b => (
          <span key={b}>{b * 500 + 250} MHz</span>
        ))}
      </div>
      <div style={{ fontSize: 9, color: 'var(--muted, #908f9e)', marginTop: 4 }}>
        * Intensity represents backend channel occupancy / activity priority vector (36 bands × 500 MHz), not measured RF power in dBm.
      </div>
    </div>
  );
}

export default SpectrumWaterfall;
