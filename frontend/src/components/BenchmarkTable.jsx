import { useEffect, useState, useMemo } from 'react';
import { api } from '../services/api';

/**
 * BenchmarkTable
 *
 * Canonical problem statement benchmark comparison table:
 * Evaluates SmartScan DRQN+MoE policy against standard scanning strategies
 * (Open-Loop Baseline, Round Robin, Random, Highest Uncertainty).
 *
 * Consumes the authoritative backend contract from GET /benchmark/latest.
 */
export function BenchmarkTable({ data: initialData, onDataLoaded }) {
  const [fetchedData, setFetchedData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(!initialData);

  useEffect(() => {
    if (initialData) return;

    let active = true;

    api
      .getLatestBenchmark()
      .then((res) => {
        if (active) {
          setFetchedData(res);
          setLoading(false);
          onDataLoaded?.(res);
        }
      })
      .catch((err) => {
        if (active) {
          setError(err.message || 'Failed to load benchmark data.');
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [initialData, onDataLoaded]);

  const data = initialData || fetchedData;

  // Normalized columns and rows adapter
  const tableContent = useMemo(() => {
    if (!data) return null;

    // Case 1: Backend canonical contract (columns + rows)
    if (Array.isArray(data.columns) && Array.isArray(data.rows) && data.rows.length > 0) {
      return {
        columns: data.columns,
        rows: data.rows,
        scenario: data.scenario || 'AG-04',
        scenarioName: data.scenario_name || 'Fast Agile Radar Hopper',
        threatClass: data.threat_class || 'Pulsed Agile Fire-Control Radar',
        description: data.description,
        executionTimeMs: data.execution_time_ms,
      };
    }

    // Case 2: Legacy results/schedulers format fallback adapter
    const results =
      data.results ||
      (data.schedulers
        ? Object.fromEntries(
            Object.entries(data.schedulers).map(([k, v]) => [k, v.summary || v])
          )
        : null);

    if (results && typeof results === 'object') {
      const schedulers = Object.keys(results);
      const foms = [
        'avg_intercept_rate',
        'pd',
        'pfa',
        'avg_reward',
        'pct_correct_predictions',
        'avg_intercept_time_error_us',
      ];
      const fomLabels = [
        'Intercept Rate',
        'Pd',
        'Pfa',
        'Avg Reward',
        'Prediction Accuracy',
        'Latency Error (µs)',
      ];

      const columns = ['Metric Dimension', ...schedulers];
      const rows = fomLabels.map((label, fIdx) => {
        const fomKey = foms[fIdx];
        const rowVals = [label];
        for (const s of schedulers) {
          const val = results[s]?.[fomKey];
          if (val === undefined || val === null) {
            rowVals.push('—');
          } else if (fomKey === 'avg_intercept_rate' || fomKey === 'pd' || fomKey === 'pfa') {
            rowVals.push(`${(Number(val) * 100).toFixed(1)}%`);
          } else if (fomKey === 'pct_correct_predictions') {
            rowVals.push(`${Number(val).toFixed(1)}%`);
          } else if (fomKey === 'avg_intercept_time_error_us') {
            rowVals.push(`${Number(val).toFixed(1)} µs`);
          } else {
            rowVals.push(Number(val).toFixed(3));
          }
        }
        return rowVals;
      });

      return {
        columns,
        rows,
        scenario: data.scenario || 'Canonical Suite',
        scenarioName: data.scenario_name || 'Multi-Scheduler Benchmark',
        threatClass: 'Multi-Threat Aggregate',
      };
    }

    return null;
  }, [data]);

  if (error) {
    return (
      <div
        className="benchmark-error"
        style={{
          color: 'var(--danger, #ef4444)',
          background: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid var(--danger, #ef4444)',
          padding: '10px 14px',
          fontSize: 12,
          fontFamily: 'var(--font-mono, monospace)',
        }}
      >
        Unable to load benchmark: {error}
      </div>
    );
  }

  if (loading) {
    return (
      <div
        style={{
          padding: '16px 12px',
          color: 'var(--muted, #a8a7b8)',
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 12,
        }}
      >
        Loading authoritative benchmark results from backend (/benchmark/latest)...
      </div>
    );
  }

  if (!tableContent || tableContent.rows.length === 0) {
    return (
      <div
        style={{
          padding: '16px 12px',
          color: 'var(--muted, #a8a7b8)',
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 12,
        }}
      >
        No benchmark results available from backend.
      </div>
    );
  }

  const { columns, rows, scenario, scenarioName, threatClass, executionTimeMs } = tableContent;

  return (
    <div className="benchmark-table-wrap">
      {/* Benchmark Metadata Header */}
      <div
        style={{
          marginBottom: 10,
          padding: '6px 10px',
          background: 'var(--panel-2, #1e2024)',
          border: '1px solid var(--border-subtle, #333539)',
          fontSize: 11,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: 'var(--accent, #bdc2ff)', fontWeight: 700 }}>
            {scenario}:
          </span>
          <span style={{ color: 'var(--text, #e2e2e8)' }}>
            {scenarioName}
          </span>
          <span
            style={{
              color: 'var(--muted, #a8a7b8)',
              background: 'var(--panel-3, #282a2e)',
              padding: '1px 6px',
              borderRadius: 2,
              fontSize: 10,
            }}
          >
            {threatClass}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontFamily: 'var(--font-mono, monospace)', fontSize: 10 }}>
          {executionTimeMs != null && (
            <span style={{ color: 'var(--muted, #a8a7b8)' }}>
              EXEC: <strong style={{ color: 'var(--secondary, #96ccff)' }}>{Math.round(executionTimeMs)} ms</strong>
            </span>
          )}
          <span style={{ color: 'var(--success, #49df9d)', fontWeight: 700 }}>
            AUTHORITATIVE CONTRACT
          </span>
        </div>
      </div>

      {/* Benchmark Table Grid */}
      <div style={{ overflowX: 'auto' }}>
        <table
          className="benchmark-table"
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left',
            fontSize: 12,
          }}
        >
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border, #454653)', background: 'var(--panel-2, #1e2024)' }}>
              {columns.map((colName, cIdx) => {
                const isSmartScan = colName.includes('Smart Scan') || colName.includes('DRQN');
                const isGain = colName.includes('Gain');
                return (
                  <th
                    key={colName}
                    style={{
                      padding: '8px 12px',
                      color: isSmartScan
                        ? 'var(--accent, #bdc2ff)'
                        : isGain
                        ? 'var(--success, #49df9d)'
                        : 'var(--muted, #a8a7b8)',
                      fontWeight: isSmartScan || isGain ? 700 : 600,
                      fontFamily: cIdx === 0 ? 'inherit' : 'var(--font-mono, monospace)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {colName}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((rowVals, rIdx) => {
              const metricName = rowVals[0];
              const isLatency = String(metricName).toLowerCase().includes('latency');
              const isFA = String(metricName).toLowerCase().includes('false-alarm');

              return (
                <tr
                  key={metricName || rIdx}
                  style={{
                    borderBottom: '1px solid var(--border-dim, #1e293b)',
                    background: rIdx % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.02)',
                  }}
                >
                  {rowVals.map((cellVal, cIdx) => {
                    const colName = columns[cIdx] || '';
                    const isMetricDim = cIdx === 0;
                    const isSmartScan = colName.includes('Smart Scan') || colName.includes('DRQN');
                    const isGain = colName.includes('Gain');

                    let cellColor = 'var(--text, #e2e2e8)';
                    let cellWeight = 400;

                    if (isMetricDim) {
                      cellColor = 'var(--text-bright, #ffffff)';
                      cellWeight = 600;
                    } else if (isSmartScan) {
                      cellColor = 'var(--accent, #bdc2ff)';
                      cellWeight = 700;
                    } else if (isGain) {
                      const gainStr = String(cellVal).trim();
                      const isPos = gainStr.startsWith('+');
                      const isNeg = gainStr.startsWith('-');
                      const goodGain = isLatency || isFA ? isNeg : isPos;
                      cellColor = goodGain ? 'var(--success, #49df9d)' : 'var(--danger, #ffb4ab)';
                      cellWeight = 700;
                    }

                    return (
                      <td
                        key={cIdx}
                        style={{
                          padding: '8px 12px',
                          color: cellColor,
                          fontWeight: cellWeight,
                          fontFamily: isMetricDim ? 'inherit' : 'var(--font-mono, monospace)',
                          whiteSpace: 'nowrap',
                          background: isSmartScan ? 'rgba(189, 194, 255, 0.05)' : 'transparent',
                        }}
                      >
                        {cellVal !== undefined && cellVal !== null && cellVal !== "" ? cellVal : '—'}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default BenchmarkTable;
