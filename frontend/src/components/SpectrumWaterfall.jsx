import { useEffect, useRef, useState, useCallback } from "react";

const TOTAL_COLS = 180; // 180 high-resolution frequency bins (100 MHz resolution) across 0–18 GHz
const TOTAL_BANDS = 36;
const FREQ_START_MHZ = 0;
const FREQ_END_MHZ = 18000;

function getSdrColor(val) {
  const v = Math.min(100, Math.max(0, val));
  const t = v / 100;
  let r, g, b;

  if (t < 0.12) {
    // Deep navy space noise floor
    r = 3; g = 8; b = Math.round(20 + (t / 0.12) * 35);
  } else if (t < 0.35) {
    // Deep blue to cyan
    const s = (t - 0.12) / 0.23;
    r = 0; g = Math.round(s * 160); b = Math.round(55 + s * 180);
  } else if (t < 0.58) {
    // Cyan to Emerald mint
    const s = (t - 0.35) / 0.23;
    r = Math.round(s * 60); g = Math.round(160 + s * 80); b = Math.round(235 - s * 140);
  } else if (t < 0.80) {
    // Mint to radiant amber
    const s = (t - 0.58) / 0.22;
    r = Math.round(60 + s * 190); g = Math.round(240 - s * 30); b = Math.round(95 - s * 80);
  } else {
    // Amber to incandescent red/white
    const s = (t - 0.80) / 0.20;
    r = Math.round(250 + s * 5); g = Math.round(210 - s * 150); b = Math.round(15 + s * 230);
  }
  return `rgb(${r},${g},${b})`;
}

export function SpectrumWaterfall({ history = [] }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hoverInfo, setHoverInfo] = useState(null);

  const N_STEPS = Math.min(history.length, 60);
  const recent = history.slice(-N_STEPS);

  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas || recent.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const marginLeft = 48;
    const marginRight = 16;
    const marginTop = 6;
    const marginBottom = 20;
    const plotW = rect.width - marginLeft - marginRight;
    const plotH = rect.height - marginTop - marginBottom;

    if (x >= marginLeft && x <= marginLeft + plotW && y >= marginTop && y <= marginTop + plotH) {
      const normX = (x - marginLeft) / plotW;
      const freqMHz = Math.round(FREQ_START_MHZ + normX * (FREQ_END_MHZ - FREQ_START_MHZ));
      const band = Math.min(TOTAL_BANDS - 1, Math.max(0, Math.floor(normX * TOTAL_BANDS)));
      const stepIdx = Math.min(recent.length - 1, Math.max(0, Math.floor(((y - marginTop) / plotH) * recent.length)));
      const frame = recent[stepIdx];
      const timeLabel = frame?.timestamp ? `T+${(frame.timestamp / 1000).toFixed(1)}ms` : `Step ${frame?.step ?? stepIdx}`;
      const tooltipX = Math.min(x + 10, Math.max(10, rect.width - 180));
      setHoverInfo({ x, y, tooltipX, freqMHz, band, timeLabel });
    } else {
      setHoverInfo(null);
    }
  }, [recent]);

  const handleMouseLeave = useCallback(() => {
    setHoverInfo(null);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const W = Math.max(200, Math.round(rect.width));
    const H = 220;

    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, W, H);

    const marginLeft = 48;
    const marginRight = 16;
    const marginTop = 6;
    const marginBottom = 20;
    const plotW = Math.max(10, W - marginLeft - marginRight);
    const plotH = Math.max(10, H - marginTop - marginBottom);

    // Background fill
    ctx.fillStyle = "#060913";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#080c18";
    ctx.fillRect(marginLeft, marginTop, plotW, plotH);

    if (recent.length === 0) {
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      ctx.fillStyle = "#8e919d";
      ctx.fillText("NO CHANNEL ACTIVITY RECORDED · AWAITING ACTIVE MISSION TELEMETRY", marginLeft + plotW / 2, marginTop + plotH / 2);
      return;
    }

    const rows = recent.length;
    const rowH = plotH / rows;
    const cellW = plotW / TOTAL_COLS;

    // Build synthesized high-resolution spectral buffer across rows
    const spectrogram = [];
    let prevRow = null;

    for (let r = 0; r < rows; r++) {
      const frame = recent[r];
      const rowBins = new Float32Array(TOTAL_COLS);

      // 1. Thermal noise floor
      for (let c = 0; c < TOTAL_COLS; c++) {
        rowBins[c] = 5.0 + 2.0 * Math.sin(c * 0.15) + (Math.random() - 0.5) * 2.0;
      }

      // 2. Persistence decay
      if (prevRow) {
        for (let c = 0; c < TOTAL_COLS; c++) {
          if (prevRow[c] > 12) {
            rowBins[c] = Math.max(rowBins[c], prevRow[c] * 0.78);
          }
        }
      }

      // 3. Channel occupancy and active bands
      const rawPriors = frame.channel_occupancy || frame.band_priorities || frame.band_powers || [];
      const activeBands = frame.active_bands || [];

      if (Array.isArray(rawPriors) && rawPriors.length === TOTAL_BANDS) {
        for (let b = 0; b < TOTAL_BANDS; b++) {
          const occ = Number(rawPriors[b]) || 0;
          if (occ > 0.05) {
            const startCol = b * 5;
            for (let sub = 0; sub < 5; sub++) {
              const c = startCol + sub;
              if (c < TOTAL_COLS) {
                rowBins[c] = Math.max(rowBins[c], occ * 40.0 + 10.0);
              }
            }
          }
        }
      }

      // Active threat emitter carriers: Gaussian profile
      activeBands.forEach((b) => {
        const centerCol = b * 5 + 2;
        const amp = 82.0;
        for (let offset = -4; offset <= 4; offset++) {
          const c = centerCol + offset;
          if (c >= 0 && c < TOTAL_COLS) {
            const g = amp * Math.exp(-(offset * offset) / (2 * 1.5 * 1.5));
            rowBins[c] = Math.max(rowBins[c], g);
          }
        }
      });

      // Tuned aperture window
      const tunedBand = frame.tuned_band ?? frame.last_band;
      if (tunedBand != null && tunedBand >= 0 && tunedBand < TOTAL_BANDS) {
        const centerCol = tunedBand * 5 + 2;
        const amp = frame.is_hit ? 95.0 : 45.0;
        for (let offset = -3; offset <= 3; offset++) {
          const c = centerCol + offset;
          if (c >= 0 && c < TOTAL_COLS) {
            const g = amp * Math.exp(-(offset * offset) / (2 * 1.4 * 1.4));
            rowBins[c] = Math.max(rowBins[c], g);
          }
        }
      }

      spectrogram.push(rowBins);
      prevRow = rowBins;
    }

    // Draw high-resolution continuous spectrogram
    for (let r = 0; r < rows; r++) {
      const y = marginTop + r * rowH;
      const bins = spectrogram[r];
      for (let c = 0; c < TOTAL_COLS; c++) {
        const val = bins[c];
        ctx.fillStyle = getSdrColor(val);
        ctx.fillRect(marginLeft + c * cellW, y, Math.ceil(cellW) + 0.5, Math.ceil(rowH) + 0.5);
      }
    }

    // Gridlines at 500 MHz band boundaries
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    for (let b = 1; b < TOTAL_BANDS; b++) {
      const x = marginLeft + (b / TOTAL_BANDS) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();
    }

    // Major Frequency markers
    const freqMarkers = [3000, 6000, 9000, 12000, 15000, 18000];
    ctx.strokeStyle = "rgba(69,70,83,0.3)";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 3]);
    freqMarkers.forEach((f) => {
      const x = marginLeft + (f / FREQ_END_MHZ) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Frequency labels along bottom axis
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#8e919d";
    [0, 3000, 6000, 9000, 12000, 15000, 18000].forEach((f) => {
      const x = marginLeft + (f / FREQ_END_MHZ) * plotW;
      ctx.fillText(`${f / 1000}G`, x, marginTop + plotH + 4);
    });

    // Time axis labels along left axis
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("NOW", marginLeft - 6, marginTop + 6);
    ctx.fillText(`-${(rows * 0.1).toFixed(1)}s`, marginLeft - 6, marginTop + plotH - 6);
  }, [recent]);

  if (!history || history.length === 0) {
    return (
      <div className="waterfall-wrap" ref={containerRef}>
        <div className="waterfall-title" style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8, fontSize: 12, fontWeight: 700 }}>
          <span>REAL-TIME RF SPECTRUM WATERFALL (0.00 – 18.00 GHz)</span>
        </div>
        <div
          style={{
            padding: "24px 12px",
            textAlign: "center",
            color: "var(--muted, #a8a7b8)",
            background: "var(--panel-2, #1e2024)",
            border: "1px solid var(--border-subtle, #333539)",
            fontFamily: "var(--font-mono, monospace)",
            fontSize: 11,
          }}
        >
          NO CHANNEL ACTIVITY FRAMES RECORDED · AWAITING ACTIVE MISSION TELEMETRY
        </div>
      </div>
    );
  }

  return (
    <div className="waterfall-wrap" ref={containerRef} style={{ position: "relative" }}>
      <div className="waterfall-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8, marginBottom: 6, fontSize: 12, fontWeight: 700 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span>REAL-TIME 180-CHANNEL SDR SPECTROGRAM (0.00 – 18.00 GHz)</span>
          <span style={{ fontSize: 10, fontWeight: 400, color: "var(--muted, #908f9e)" }}>
            [100 MHz Resolution · {N_STEPS} Dwell Slices History]
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 11 }}>
          <span style={{ color: "#38bdf8" }}>■ Receiver Tuned</span>
          <span style={{ color: "#ef4444" }}>■ Emitter Burst</span>
          <span style={{ color: "#49df9d" }}>■ Intercept Hit</span>
        </div>
      </div>

      <div style={{ position: "relative", width: "100%", background: "#060913", border: "1px solid var(--border-dim, #1e293b)", borderRadius: 2 }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ display: "block", cursor: "crosshair" }}
        />
        {hoverInfo && (
          <div
            style={{
              position: "absolute",
              left: hoverInfo.tooltipX,
              top: Math.max(8, hoverInfo.y - 36),
              background: "rgba(10, 14, 23, 0.92)",
              border: "1px solid #38bdf8",
              padding: "4px 8px",
              borderRadius: 3,
              fontSize: 10,
              fontFamily: "var(--font-mono, monospace)",
              color: "#e2e2e8",
              pointerEvents: "none",
              zIndex: 10,
              boxShadow: "0 2px 8px rgba(0,0,0,0.5)",
            }}
          >
            <div><strong>{hoverInfo.freqMHz.toLocaleString()} MHz</strong> (Band B{hoverInfo.band + 1})</div>
            <div style={{ color: "#908f9e" }}>{hoverInfo.timeLabel}</div>
          </div>
        )}
      </div>

      <div style={{ fontSize: 9, color: "var(--muted, #908f9e)", marginTop: 4 }}>
        * Continuous tactical spectrogram with RF noise floor texture, Gaussian spectral carrier profiles, and decaying hop burst persistence.
      </div>
    </div>
  );
}

export default SpectrumWaterfall;
