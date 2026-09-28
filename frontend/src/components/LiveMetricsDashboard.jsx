import { useMetricsWebSocket } from '../hooks/useMetricsWebSocket';

const FOM_CONFIG = [
  {
    key: 'pd',
    altKeys: ['Pd', 'rolling_pd', 'sensitivity'],
    label: 'Probability of Detection',
    fmt: v => v != null && !isNaN(v) ? `${(Number(v) * 100).toFixed(2)}%` : '—',
    target: '≥ 99%',
    good: v => v != null && !isNaN(v) ? Number(v) >= 0.99 : null,
  },
  {
    key: 'pfa',
    altKeys: ['Pfa'],
    label: 'Prob. False Alarm',
    fmt: v => v != null && !isNaN(v) ? `${(Number(v) * 100).toFixed(3)}%` : '—',
    target: '≤ 0.05%',
    good: v => v != null && !isNaN(v) ? Number(v) <= 0.0005 : null,
  },
  {
    key: 'avg_intercept_rate',
    altKeys: ['intercept_rate', 'mean_ir'],
    label: 'Avg Intercept Rate',
    fmt: v => v != null && !isNaN(v) ? `${(Number(v) * 100).toFixed(1)}%` : '—',
    target: '≥ 60%',
    good: v => v != null && !isNaN(v) ? Number(v) >= 0.60 : null,
  },
  {
    key: 'avg_reward',
    altKeys: ['reward_per_dwell', 'reward'],
    label: 'Avg Reward / Step',
    fmt: v => v != null && !isNaN(v) ? Number(v).toFixed(3) : '—',
    target: '> 0',
    good: v => v != null && !isNaN(v) ? Number(v) > 0 : null,
  },
  {
    key: 'pct_correct_predictions',
    altKeys: ['correct_pct'],
    label: '% Correct Predictions',
    fmt: v => v != null && !isNaN(v) ? `${Number(v).toFixed(1)}%` : '—',
    target: '≥ 80%',
    good: v => v != null && !isNaN(v) ? Number(v) >= 80 : null,
  },
  {
    key: 'avg_intercept_time_error_us',
    altKeys: ['avg_time_error'],
    label: 'Avg Intercept Time Error',
    fmt: v => v != null && !isNaN(v) ? `${Number(v).toFixed(0)} µs` : '—',
    target: '≤ 300µs',
    good: v => v != null && !isNaN(v) ? Number(v) <= 300 : null,
  },
];

export function LiveMetricsDashboard({ metrics: externalMetrics }) {
  const ws = useMetricsWebSocket();
  const rawMetrics = externalMetrics || ws.metrics || {};
  const isConnected = Boolean(rawMetrics.connected || ws.metrics.connected);
  const step = rawMetrics.step ?? ws.metrics.step ?? 0;

  const getMetricVal = (cfg) => {
    if (rawMetrics[cfg.key] !== undefined && rawMetrics[cfg.key] !== null) return rawMetrics[cfg.key];
    if (cfg.altKeys) {
      for (const k of cfg.altKeys) {
        if (rawMetrics[k] !== undefined && rawMetrics[k] !== null) return rawMetrics[k];
      }
    }
    if (ws.metrics[cfg.key] !== undefined && ws.metrics[cfg.key] !== null) return ws.metrics[cfg.key];
    if (cfg.altKeys) {
      for (const k of cfg.altKeys) {
        if (ws.metrics[k] !== undefined && ws.metrics[k] !== null) return ws.metrics[k];
      }
    }
    return null;
  };

  return (
    <div className="metrics-dashboard">
      <div className="conn-status" style={{ color: isConnected ? "var(--success, #49df9d)" : "var(--danger, #ef4444)" }}>
        <span>{isConnected ? "● LIVE TELEMETRY" : "○ Awaiting Stream"}</span>
        <span> | Step {step}</span>
      </div>
      
      <div className="fom-grid">
        {FOM_CONFIG.map(f => {
          const val = getMetricVal(f);
          const isGood = f.good(val);
          const statusClass = isGood === true ? "good" : (isGood === false ? "warn" : "neutral");
          return (
            <div key={f.key} className={`fom-card ${statusClass}`}>
              <div className="fom-label">{f.label}</div>
              <div className="fom-value">{f.fmt(val)}</div>
              <div className="fom-target">Target: {f.target}</div>
            </div>
          );
        })}
      </div>
      
      <div className="action-attribution">
        <div className="attr-title">Last Decision</div>
        <div>Band: <b>{rawMetrics.last_band ?? ws.metrics.last_band ?? "—"}</b></div>
        <div>Mode: <b>{rawMetrics.last_mode ?? ws.metrics.last_mode ?? "—"}</b></div>
        <div>Reason: <b>{rawMetrics.decision_reason ?? ws.metrics.decision_reason ?? "—"}</b></div>
      </div>
    </div>
  );
}

export default LiveMetricsDashboard;
