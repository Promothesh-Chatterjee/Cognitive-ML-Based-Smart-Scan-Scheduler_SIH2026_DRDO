import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { loadLiveTelemetry } from "../services/liveService";
import { startTelemetryStream } from "../services/liveSocket";
import { startSyntheticStream } from "../services/syntheticStream";
import { syntheticSystem } from "../data/mockSystem";
import { PanelHead, DataSourceBadge } from "../components/stitch";
import { useTheme } from "../context/ThemeContext";

const NUM_BANDS = 36;
const FREQ_START_MHZ = 0;
const FREQ_END_MHZ = 18000;
const BAND_WIDTH_MHZ = 500;
const WATERFALL_BINS = 360; // 50 MHz per bin across 0–18 GHz
const WATERFALL_ROWS = 140; // 140 time slices history depth

const SYNTHETIC_EVENTS = [
  { time: "NOW", band: 16, frequencyMHz: 8250, type: "ARMED", mode: "PREEMPTIVE_INTERCEPT" },
  { time: "T-60 ms", band: 16, frequencyMHz: 8250, type: "HIT", mode: "REVISIT" },
  { time: "T-120 ms", band: 28, frequencyMHz: 14250, type: "SEARCH", mode: "LONG_DWELL" },
  { time: "T-180 ms", band: 22, frequencyMHz: 11250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-240 ms", band: 16, frequencyMHz: 8250, type: "HIT", mode: "REVISIT" },
  { time: "T-300 ms", band: 14, frequencyMHz: 7250, type: "SEARCH", mode: "SHORT_DWELL" },
  { time: "T-360 ms", band: 10, frequencyMHz: 5250, type: "DETECTION", mode: "SHORT_DWELL" },
  { time: "T-420 ms", band: 8, frequencyMHz: 4250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-480 ms", band: 6, frequencyMHz: 3250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-540 ms", band: 4, frequencyMHz: 2250, type: "SEARCH", mode: "NORMAL_DWELL" },
];

/**
 * Multi-palette SDR Spectrogram Colormapping.
 * Supports Tactical SDR (default), CRT Phosphor Night-Vision, and Infrared Magma.
 */
function getWaterfallColor(value, palette = "tactical", isDark = true, gain = 1.0) {
  const v = Math.min(100, Math.max(0, value * gain));
  const t = v / 100;
  let r, g, b;

  if (palette === "phosphor") {
    if (isDark) {
      // CRT Night-vision Phosphor Green
      if (t < 0.10) {
        r = 1; g = Math.round(14 + (t / 0.10) * 20); b = 3;
      } else if (t < 0.45) {
        const s = (t - 0.10) / 0.35;
        r = Math.round(5 + s * 25);
        g = Math.round(34 + s * 140);
        b = Math.round(10 + s * 30);
      } else if (t < 0.80) {
        const s = (t - 0.45) / 0.35;
        r = Math.round(30 + s * 120);
        g = Math.round(174 + s * 70);
        b = Math.round(40 + s * 60);
      } else {
        const s = (t - 0.80) / 0.20;
        r = Math.round(150 + s * 105);
        g = 255;
        b = Math.round(100 + s * 155);
      }
    } else {
      if (t < 0.10) {
        const s = t / 0.10;
        r = Math.round(245 - s * 15); g = Math.round(250 - s * 10); b = Math.round(245 - s * 15);
      } else if (t < 0.45) {
        const s = (t - 0.10) / 0.35;
        r = Math.round(230 - s * 150); g = Math.round(240 - s * 50); b = Math.round(230 - s * 160);
      } else {
        const s = (t - 0.45) / 0.55;
        r = Math.round(80 - s * 60); g = Math.round(190 - s * 50); b = Math.round(70 - s * 50);
      }
    }
  } else if (palette === "magma") {
    if (isDark) {
      // Infrared Magma Heatmap
      if (t < 0.12) {
        r = Math.round(6 + t * 40); g = 2; b = Math.round(14 + t * 80);
      } else if (t < 0.38) {
        const s = (t - 0.12) / 0.26;
        r = Math.round(11 + s * 90); g = Math.round(5 + s * 25); b = Math.round(24 + s * 110);
      } else if (t < 0.65) {
        const s = (t - 0.38) / 0.27;
        r = Math.round(101 + s * 125); g = Math.round(30 + s * 50); b = Math.round(134 - s * 70);
      } else if (t < 0.88) {
        const s = (t - 0.65) / 0.23;
        r = Math.round(226 + s * 25); g = Math.round(80 + s * 120); b = Math.round(64 - s * 40);
      } else {
        const s = (t - 0.88) / 0.12;
        r = 255; g = Math.round(200 + s * 55); b = Math.round(24 + s * 231);
      }
    } else {
      if (t < 0.12) {
        const s = t / 0.12;
        r = Math.round(253 - s * 10); g = Math.round(248 - s * 15); b = Math.round(250 - s * 15);
      } else if (t < 0.45) {
        const s = (t - 0.12) / 0.33;
        r = Math.round(243 - s * 60); g = Math.round(233 - s * 150); b = Math.round(235 - s * 90);
      } else {
        const s = (t - 0.45) / 0.55;
        r = Math.round(183 + s * 50); g = Math.round(83 - s * 60); b = Math.round(145 - s * 120);
      }
    }
  } else {
    // Tactical SDR (Default High-Contrast Gradient)
    if (isDark) {
      if (t < 0.12) {
        // Deep space floor
        r = 2; g = 5; b = Math.round(16 + (t / 0.12) * 35);
      } else if (t < 0.32) {
        // Blue to Cyan
        const s = (t - 0.12) / 0.20;
        r = 0; g = Math.round(s * 150); b = Math.round(51 + s * 180);
      } else if (t < 0.55) {
        // Cyan to Emerald Mint
        const s = (t - 0.32) / 0.23;
        r = Math.round(s * 50); g = Math.round(150 + s * 90); b = Math.round(231 - s * 120);
      } else if (t < 0.78) {
        // Mint to Bright Amber
        const s = (t - 0.55) / 0.23;
        r = Math.round(50 + s * 195); g = Math.round(240 - s * 20); b = Math.round(111 - s * 95);
      } else {
        // Amber to Radiant Red/White
        const s = (t - 0.78) / 0.22;
        r = Math.round(245 + s * 10); g = Math.round(220 - s * 140); b = Math.round(16 + s * 230);
      }
    } else {
      if (t < 0.12) {
        const s = t / 0.12;
        r = Math.round(248 - s * 12); g = Math.round(250 - s * 12); b = Math.round(252 - s * 10);
      } else if (t < 0.35) {
        const s = (t - 0.12) / 0.23;
        r = Math.round(236 - s * 180); g = Math.round(238 - s * 45); b = Math.round(242 + s * 10);
      } else if (t < 0.60) {
        const s = (t - 0.35) / 0.25;
        r = Math.round(56 + s * 10); g = Math.round(193 + s * 40); b = Math.round(252 - s * 150);
      } else if (t < 0.82) {
        const s = (t - 0.60) / 0.22;
        r = Math.round(66 + s * 180); g = Math.round(233 - s * 80); b = Math.round(102 - s * 90);
      } else {
        const s = (t - 0.82) / 0.18;
        r = Math.round(246 + s * 9); g = Math.round(153 - s * 120); b = Math.round(12 - s * 5);
      }
    }
  }
  return `rgb(${r},${g},${b})`;
}

function useCanvasResize(canvasRef, containerRef) {
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.scale(dpr, dpr);
    }

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    return () => ro.disconnect();
  }, [canvasRef, containerRef]);
}

/**
 * Generate a 360-bin channel activity slice for waterfall history.
 * Intensity directly reflects backend channel occupancy/activity priority vector (36 bands × 10 bins).
 */
function createWaterfallRow({
  liveBand,
  isHit,
  telemetry,
  priorRow,
  step = 0,
  clockUs = 0,
  modeName = "NORMAL_DWELL",
}) {
  const bins = new Float32Array(WATERFALL_BINS);
  const priorBins = priorRow?.bins || (Array.isArray(priorRow) ? priorRow : null);

  // 1. Decay of prior row signals
  for (let b = 0; b < WATERFALL_BINS; b++) {
    bins[b] = priorBins && priorBins[b] !== undefined ? priorBins[b] * 0.86 : 0.0;
  }

  // 2. Real channel occupancy / activity priority from backend telemetry (band_priorities)
  const bandPriors = telemetry?.bandPriorities || telemetry?.metrics?.band_priorities || [];
  if (Array.isArray(bandPriors) && bandPriors.length > 0) {
    for (let bIdx = 0; bIdx < NUM_BANDS; bIdx++) {
      const prio = Math.max(0.0, Math.min(1.0, Number(bandPriors[bIdx]) || 0));
      if (prio > 0) {
        const startBin = bIdx * 10;
        for (let offset = 0; offset < 10; offset++) {
          bins[startBin + offset] = Math.max(bins[startBin + offset], prio * 100);
        }
      }
    }
  }

  // 3. Tuned receiver aperture window (500 MHz span = 10 bins)
  if (liveBand !== undefined && liveBand !== null && liveBand >= 0 && liveBand < NUM_BANDS) {
    const startBin = liveBand * 10;
    const apertureLevel = isHit ? 75 : 35;
    for (let offset = 0; offset < 10; offset++) {
      bins[startBin + offset] = Math.max(bins[startBin + offset], apertureLevel);
    }
  }

  return {
    bins,
    timeLabel: clockUs > 0 ? `T+${(clockUs / 1000).toFixed(1)}ms` : "NOW",
    timestamp: Date.now(),
    step,
    clockUs,
    band: liveBand,
    isHit,
    modeName,
  };
}

/**
 * Enhanced Real-Time Spectrum Trace Canvas.
 * Calibrated -110 to -20 dBm scale, instantaneous trace + Max-Hold persistence decay,
 * realistic RF noise grass, exact pulse Gaussian peaks, and tuned 500 MHz aperture reticle.
 */
function SpectrumCanvas({
  bandHeights,
  currentBand,
  activeBands,
  liveTelemetry,
  traceMode = "LIVE_PEAK", // "LIVE_PEAK", "LIVE_ONLY", "PEAK_ONLY"
}) {
  const { isDark } = useTheme();
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const peakHoldRef = useRef(null);
  const lastFrameIdRef = useRef(null);
  const [hoverInfo, setHoverInfo] = useState(null);

  useCanvasResize(canvasRef, containerRef);

  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const marginLeft = 52;
    const marginRight = 16;
    const marginTop = 14;
    const marginBottom = 30;
    const plotW = rect.width - marginLeft - marginRight;
    const plotH = rect.height - marginTop - marginBottom;

    if (x >= marginLeft && x <= marginLeft + plotW && y >= marginTop && y <= marginTop + plotH) {
      const normX = (x - marginLeft) / plotW;
      const freqMHz = Math.round(FREQ_START_MHZ + normX * (FREQ_END_MHZ - FREQ_START_MHZ));
      const normY = (y - marginTop) / plotH;
      const levelDBm = -20 - normY * 90; // -20 dBm to -110 dBm
      const band = Math.min(35, Math.floor(freqMHz / BAND_WIDTH_MHZ));
      setHoverInfo({ x, y, freqMHz, levelDBm, band });
    } else {
      setHoverInfo(null);
    }
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoverInfo(null);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;

    ctx.clearRect(0, 0, W, H);

    const marginLeft = 52;
    const marginRight = 16;
    const marginTop = 14;
    const marginBottom = 30;
    const plotW = Math.max(10, W - marginLeft - marginRight);
    const plotH = Math.max(10, H - marginTop - marginBottom);

    // Background fill
    ctx.fillStyle = isDark ? "#060913" : "#ffffff";
    ctx.fillRect(0, 0, W, H);

    // Plot area background
    ctx.fillStyle = isDark ? "#080c18" : "#f8fafc";
    ctx.fillRect(marginLeft, marginTop, plotW, plotH);

    // Calibrated dBm Y-scale: -110 dBm to -20 dBm (Span: 90 dB)
    const yFromDBm = (db) => {
      const clamped = Math.min(-20, Math.max(-110, db));
      return marginTop + ((-20 - clamped) / 90) * plotH;
    };

    // Horizontal dBm grid lines & labels
    const dbTicks = [-30, -50, -70, -90, -110];
    const minorTicks = [-40, -60, -80, -100];

    // Minor grid
    ctx.strokeStyle = isDark ? "rgba(69,70,83,0.18)" : "rgba(226,232,240,0.8)";
    ctx.lineWidth = 0.5;
    minorTicks.forEach((db) => {
      const y = yFromDBm(db);
      ctx.beginPath();
      ctx.moveTo(marginLeft, y);
      ctx.lineTo(marginLeft + plotW, y);
      ctx.stroke();
    });

    // Major grid
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    dbTicks.forEach((db) => {
      const y = yFromDBm(db);
      ctx.strokeStyle = isDark ? "rgba(69,70,83,0.4)" : "rgba(203,213,225,0.9)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(marginLeft, y);
      ctx.lineTo(marginLeft + plotW, y);
      ctx.stroke();

      ctx.fillStyle = isDark ? "#8e919d" : "#475569";
      ctx.fillText(`${db} dBm`, marginLeft - 5, y);
    });

    // MDS (Minimum Detectable Signal) threshold line at -105 dBm
    const yMDS = yFromDBm(-105);
    ctx.strokeStyle = "rgba(244,63,94,0.35)";
    ctx.lineWidth = 0.8;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(marginLeft, yMDS);
    ctx.lineTo(marginLeft + plotW, yMDS);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(244,63,94,0.7)";
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = "right";
    ctx.fillText("MDS: -105 dBm", marginLeft + plotW - 6, yMDS - 4);

    // Vertical Frequency grid lines & bottom labels (0 to 18 GHz)
    const freqTicks = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000];
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    freqTicks.forEach((f) => {
      const x = marginLeft + ((f - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      ctx.strokeStyle = isDark ? "rgba(69,70,83,0.35)" : "rgba(203,213,225,0.75)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();

      ctx.fillStyle = isDark ? "#8e919d" : "#334155";
      ctx.font = '9px "JetBrains Mono", monospace';
      const label = (f / 1000).toFixed(0) + "G";
      ctx.fillText(label, x, marginTop + plotH + 5);
    });

    // Intermediate 1 GHz ticks
    for (let f = 1000; f < 18000; f += 2000) {
      const x = marginLeft + ((f - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      ctx.strokeStyle = isDark ? "rgba(69,70,83,0.18)" : "rgba(226,232,240,0.6)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();
    }

    // Tuned Receiver 500 MHz Aperture Beam
    if (currentBand !== undefined && currentBand !== null) {
      const bandStartF = currentBand * BAND_WIDTH_MHZ;
      const bandEndF = (currentBand + 1) * BAND_WIDTH_MHZ;
      const xStart = marginLeft + ((bandStartF - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      const xEnd = marginLeft + ((bandEndF - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      const xCenter = (xStart + xEnd) / 2;
      const bandW = xEnd - xStart;

      // Spotlight beam
      const beamGrad = ctx.createLinearGradient(0, marginTop, 0, marginTop + plotH);
      if (isDark) {
        beamGrad.addColorStop(0, "rgba(56,189,248,0.14)");
        beamGrad.addColorStop(1, "rgba(56,189,248,0.02)");
      } else {
        beamGrad.addColorStop(0, "rgba(37,99,235,0.12)");
        beamGrad.addColorStop(1, "rgba(37,99,235,0.01)");
      }
      ctx.fillStyle = beamGrad;
      ctx.fillRect(xStart, marginTop, bandW, plotH);

      // Aperture boundary lines
      ctx.strokeStyle = isDark ? "rgba(56,189,248,0.6)" : "rgba(37,99,235,0.5)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(xStart, marginTop);
      ctx.lineTo(xStart, marginTop + plotH);
      ctx.moveTo(xEnd, marginTop);
      ctx.lineTo(xEnd, marginTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Center frequency indicator hairline
      ctx.strokeStyle = isDark ? "#38bdf8" : "#2563eb";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xCenter, marginTop);
      ctx.lineTo(xCenter, marginTop + plotH);
      ctx.stroke();

      // Top bracket & tag
      ctx.fillStyle = isDark ? "#38bdf8" : "#1d4ed8";
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      const tuneGHz = (bandStartF + BAND_WIDTH_MHZ / 2) / 1000;
      ctx.fillText(`TUNED: B${currentBand} · ${tuneGHz.toFixed(3)}G [500M IBW]`, xCenter, marginTop + 3);
    }

    // Synthesize physical RF spectral envelope with thermal grass and pulse peaks
    const ptsCount = Math.floor(plotW);
    const instantaneousDBm = new Float32Array(ptsCount);

    // Collect active emitter & pulse peaks
    const signalPeaks = [];
    if (Array.isArray(liveTelemetry?.emitters) && liveTelemetry.emitters.length > 0) {
      liveTelemetry.emitters.forEach((em) => {
        if (em.frequency_mhz != null && em.frequency_mhz > 0) {
          signalPeaks.push({
            freqMHz: em.frequency_mhz,
            powerDBm: em.amplitude_db ? Number(em.amplitude_db) : -54.0,
            bwMHz: 24.0,
          });
        }
      });
    } else {
      // Default scenario signals
      [
        { freqMHz: 750, powerDBm: -52.0, bwMHz: 18.0 },
        { freqMHz: 1750, powerDBm: -58.0, bwMHz: 25.0 },
        { freqMHz: 4250, powerDBm: -64.0, bwMHz: 30.0 },
        { freqMHz: 8250, powerDBm: -42.0, bwMHz: 22.0 },
        { freqMHz: 10250, powerDBm: -50.0, bwMHz: 20.0 },
        { freqMHz: 14250, powerDBm: -62.0, bwMHz: 35.0 },
      ].forEach((p) => signalPeaks.push(p));
    }

    // Intercepted PDWs
    if (Array.isArray(liveTelemetry?.pdws)) {
      liveTelemetry.pdws.slice(0, 8).forEach((p) => {
        if (p.frequency_mhz != null && p.frequency_mhz > 0) {
          signalPeaks.push({
            freqMHz: p.frequency_mhz,
            powerDBm: p.amplitude_db ? Number(p.amplitude_db) : -45.0,
            bwMHz: 18.0,
          });
        }
      });
    }

    // Live intercept flare in tuned band
    if (liveTelemetry?.hit && currentBand !== undefined && currentBand !== null) {
      signalPeaks.push({
        freqMHz: currentBand * BAND_WIDTH_MHZ + 250,
        powerDBm: -34.0,
        bwMHz: 20.0,
      });
    }

    // Build instantaneous spectral trace points
    for (let i = 0; i < ptsCount; i++) {
      const fMHz = FREQ_START_MHZ + (i / ptsCount) * (FREQ_END_MHZ - FREQ_START_MHZ);
      const bIdx = Math.min(35, Math.floor(fMHz / BAND_WIDTH_MHZ));

      // Thermal noise baseline (-102.5 dBm) with RF grass jitter
      const grass = Math.sin(i * 14.1) * 1.6 + Math.cos(i * 37.9) * 1.1;
      let level = -102.5 + grass;

      // Regional band activity envelope lift
      const prio = bandHeights?.[bIdx] ?? 0;
      if (prio > 0) {
        level += (prio / 100) * 16.0;
      }

      // Gaussian emission profile for individual RF peaks
      signalPeaks.forEach((pk) => {
        const df = Math.abs(fMHz - pk.freqMHz);
        if (df < 120) {
          const sigma = pk.bwMHz || 22.0;
          const peakHeight = pk.powerDBm - (-102.5);
          const gaussian = peakHeight * Math.exp(-(df * df) / (2 * sigma * sigma));
          level = Math.max(level, -102.5 + gaussian);
        }
      });

      instantaneousDBm[i] = Math.min(-20, Math.max(-110, level));
    }

    // Defect 6: Authoritative telemetry frame identifier (never use object identity)
    const frameId = liveTelemetry
      ? (liveTelemetry.step != null
          ? `s-${liveTelemetry.step}-${liveTelemetry.clockUs ?? 0}`
          : (liveTelemetry.clockUs != null
              ? `c-${liveTelemetry.clockUs}`
              : (liveTelemetry.timestamp != null ? `t-${liveTelemetry.timestamp}` : null)))
      : null;

    const isNewFrame = frameId !== null && frameId !== lastFrameIdRef.current;
    if (isNewFrame) {
      lastFrameIdRef.current = frameId;
    }

    // Max-Hold (Peak Hold) Persistence Decay
    if (!peakHoldRef.current || peakHoldRef.current.length !== ptsCount) {
      peakHoldRef.current = new Float32Array(ptsCount);
      for (let i = 0; i < ptsCount; i++) peakHoldRef.current[i] = instantaneousDBm[i];
    } else if (isNewFrame) {
      // ONLY decay downward on new telemetry frame arrival, NEVER on mouse move!
      for (let i = 0; i < ptsCount; i++) {
        // Attenuate downward in dBm towards noise floor (-105 dBm)
        const decayed = peakHoldRef.current[i] - 0.45;
        peakHoldRef.current[i] = Math.max(instantaneousDBm[i], Math.max(-105, decayed));
      }
    }

    // 1. Draw Area Gradient under Instantaneous Curve
    if (traceMode === "LIVE_PEAK" || traceMode === "LIVE_ONLY") {
      ctx.beginPath();
      ctx.moveTo(marginLeft, marginTop + plotH);
      for (let i = 0; i < ptsCount; i++) {
        const x = marginLeft + i;
        const y = yFromDBm(instantaneousDBm[i]);
        ctx.lineTo(x, y);
      }
      ctx.lineTo(marginLeft + ptsCount, marginTop + plotH);
      ctx.closePath();

      const fillGrad = ctx.createLinearGradient(0, marginTop, 0, marginTop + plotH);
      if (isDark) {
        fillGrad.addColorStop(0, "rgba(56,189,248,0.30)");
        fillGrad.addColorStop(0.5, "rgba(73,223,157,0.12)");
        fillGrad.addColorStop(1, "rgba(73,223,157,0.01)");
      } else {
        fillGrad.addColorStop(0, "rgba(2,132,199,0.28)");
        fillGrad.addColorStop(0.5, "rgba(2,132,199,0.08)");
        fillGrad.addColorStop(1, "rgba(2,132,199,0.01)");
      }
      ctx.fillStyle = fillGrad;
      ctx.fill();
    }

    // 2. Draw Max-Hold / Peak-Hold Trace (Golden Amber Line)
    if (traceMode === "LIVE_PEAK" || traceMode === "PEAK_ONLY") {
      ctx.beginPath();
      for (let i = 0; i < ptsCount; i++) {
        const x = marginLeft + i;
        const y = yFromDBm(peakHoldRef.current[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = isDark ? "#f59e0b" : "#d97706";
      ctx.lineWidth = 1.0;
      ctx.stroke();
    }

    // 3. Draw Instantaneous Signal Trace (Crisp Tactical Stroke)
    if (traceMode === "LIVE_PEAK" || traceMode === "LIVE_ONLY") {
      ctx.beginPath();
      for (let i = 0; i < ptsCount; i++) {
        const x = marginLeft + i;
        const y = yFromDBm(instantaneousDBm[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = isDark ? "#38bdf8" : "#0284c7";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }

    // 4. Automatic Peak Markers (Top detected signal peaks > -80 dBm)
    const detectedPeaks = [];
    for (let i = 3; i < ptsCount - 3; i++) {
      const val = instantaneousDBm[i];
      if (
        val > -80 &&
        val >= instantaneousDBm[i - 1] &&
        val >= instantaneousDBm[i - 2] &&
        val >= instantaneousDBm[i + 1] &&
        val >= instantaneousDBm[i + 2]
      ) {
        const f = FREQ_START_MHZ + (i / ptsCount) * (FREQ_END_MHZ - FREQ_START_MHZ);
        // Ensure not too close to an existing found peak
        if (!detectedPeaks.some((p) => Math.abs(p.f - f) < 300)) {
          detectedPeaks.push({ x: marginLeft + i, y: yFromDBm(val), f, val });
        }
      }
    }

    detectedPeaks.sort((a, b) => b.val - a.val);
    detectedPeaks.slice(0, 3).forEach((pk, idx) => {
      // Inverted marker triangle
      ctx.fillStyle = isDark ? "#f59e0b" : "#d97706";
      ctx.beginPath();
      ctx.moveTo(pk.x, pk.y - 4);
      ctx.lineTo(pk.x - 4, pk.y - 11);
      ctx.lineTo(pk.x + 4, pk.y - 11);
      ctx.closePath();
      ctx.fill();

      // Peak label tag
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      const ghz = (pk.f / 1000).toFixed(2);
      ctx.fillText(`PK${idx + 1}:${ghz}G`, pk.x, pk.y - 14);
    });

    // 5. Active Sub-Band Activity Indicators along top edge
    if (activeBands && activeBands.size > 0) {
      activeBands.forEach((band) => {
        if (band === currentBand) return;
        const cx = marginLeft + ((band * BAND_WIDTH_MHZ + BAND_WIDTH_MHZ / 2 - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
        ctx.fillStyle = isDark ? "rgba(73,223,157,0.7)" : "rgba(2,132,199,0.8)";
        ctx.beginPath();
        ctx.arc(cx, marginTop + 4, 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Outer plot border
    ctx.strokeStyle = isDark ? "rgba(189,194,255,0.4)" : "rgba(203,213,225,0.9)";
    ctx.lineWidth = 0.8;
    ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

    // 6. Interactive Crosshair & Cursor Readout
    if (hoverInfo) {
      const { x, y, freqMHz, levelDBm, band } = hoverInfo;
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.45)" : "rgba(15,23,42,0.45)";
      ctx.lineWidth = 0.6;
      ctx.setLineDash([2, 2]);

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(marginLeft, y);
      ctx.lineTo(marginLeft + plotW, y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Floating readout tooltip
      const readout = `${freqMHz.toLocaleString()} MHz (${(freqMHz / 1000).toFixed(3)} GHz) [B${band}] · ${levelDBm.toFixed(1)} dBm`;
      ctx.font = '9px "JetBrains Mono", monospace';
      const textW = ctx.measureText(readout).width;
      const badgeX = Math.min(marginLeft + plotW - textW - 12, Math.max(marginLeft + 6, x + 8));
      const badgeY = Math.max(marginTop + 14, y - 10);

      ctx.fillStyle = isDark ? "rgba(15,23,42,0.92)" : "rgba(255,255,255,0.95)";
      ctx.strokeStyle = isDark ? "#38bdf8" : "#0284c7";
      ctx.lineWidth = 0.8;
      ctx.fillRect(badgeX - 4, badgeY - 10, textW + 8, 14);
      ctx.strokeRect(badgeX - 4, badgeY - 10, textW + 8, 14);

      ctx.fillStyle = isDark ? "#ffffff" : "#0f172a";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(readout, badgeX, badgeY - 3);
    }
  }, [bandHeights, currentBand, activeBands, liveTelemetry, traceMode, hoverInfo, isDark]);

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        position: "relative",
        width: "100%",
        height: 240,
        background: isDark ? "var(--panel-lowest, #060913)" : "var(--panel, #ffffff)",
        border: "1px solid var(--border, #454653)",
        cursor: "crosshair",
        transition: "background 0.15s ease, border-color 0.15s ease",
      }}
    >
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />
    </div>
  );
}

/**
 * Enhanced High-Density Waterfall Spectrogram Canvas.
 * 360 fine bins (50 MHz resolution), multi-palette colormaps (Tactical SDR, CRT Phosphor, Magma),
 * calibrated Y-axis time graduations, intercepted pulse indicators, and interactive inspect tool.
 */
function WaterfallCanvas({
  waterfall,
  currentBand,
  palette = "tactical",
  gain = 1.0,
}) {
  const { isDark } = useTheme();
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hoverInfo, setHoverInfo] = useState(null);

  useCanvasResize(canvasRef, containerRef);

  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const marginLeft = 52;
    const marginRight = 16;
    const marginTop = 6;
    const marginBottom = 20;
    const plotW = rect.width - marginLeft - marginRight;
    const plotH = rect.height - marginTop - marginBottom;

    if (x >= marginLeft && x <= marginLeft + plotW && y >= marginTop && y <= marginTop + plotH) {
      const rows = waterfall.length || WATERFALL_ROWS;
      const rowIdx = Math.min(rows - 1, Math.max(0, Math.floor(((y - marginTop) / plotH) * rows)));
      const normX = (x - marginLeft) / plotW;
      const freqMHz = Math.round(FREQ_START_MHZ + normX * (FREQ_END_MHZ - FREQ_START_MHZ));
      const binIdx = Math.min(WATERFALL_BINS - 1, Math.max(0, Math.floor(normX * WATERFALL_BINS)));
      const rowData = waterfall[rowIdx];
      const val = rowData?.bins ? rowData.bins[binIdx] : (rowData?.[binIdx] ?? 0);
      const isHit = rowData?.isHit;
      const timeOffsetSec = (rowIdx * 0.1).toFixed(1);
      setHoverInfo({ x, y, freqMHz, binIdx, val, isHit, timeOffsetSec });
    } else {
      setHoverInfo(null);
    }
  }, [waterfall]);

  const handleMouseLeave = useCallback(() => {
    setHoverInfo(null);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = isDark ? "#050711" : "#f8fafc";
    ctx.fillRect(0, 0, W, H);

    const marginLeft = 52;
    const marginRight = 16;
    const marginTop = 6;
    const marginBottom = 20;
    const plotW = Math.max(10, W - marginLeft - marginRight);
    const plotH = Math.max(10, H - marginTop - marginBottom);

    if (!waterfall || waterfall.length === 0) {
      // Empty state
      ctx.strokeStyle = isDark ? "rgba(69,70,83,0.3)" : "rgba(203,213,225,0.6)";
      ctx.lineWidth = 0.5;
      for (let i = 1; i < 4; i++) {
        const y = marginTop + (i / 4) * plotH;
        ctx.beginPath();
        ctx.moveTo(marginLeft, y);
        ctx.lineTo(marginLeft + plotW, y);
        ctx.stroke();
      }
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      ctx.fillStyle = isDark ? "#8e919d" : "#334155";
      ctx.fillText("NO CHANNEL ACTIVITY FRAMES RECORDED · AWAITING ACTIVE MISSION STREAM", marginLeft + plotW / 2, marginTop + plotH / 2);
      return;
    }

    const rows = waterfall.length;
    const rowH = plotH / rows;
    const cellW = plotW / WATERFALL_BINS;

    // Render High-Resolution Channel Activity Matrix
    for (let r = 0; r < rows; r++) {
      const rowData = waterfall[r];
      const y = marginTop + r * rowH;
      const pixelRowH = Math.ceil(rowH) + 0.3;
      const bins = rowData?.bins || (Array.isArray(rowData) ? rowData : null);

      if (bins) {
        for (let b = 0; b < WATERFALL_BINS; b++) {
          const val = bins[b] ?? 0;
          const x = marginLeft + b * cellW;
          ctx.fillStyle = getWaterfallColor(val, palette, isDark, gain);
          ctx.fillRect(x, y, cellW + 0.4, pixelRowH);
        }
      }

      // Discrete pulse intercept indicator (emerald diamond / halo) at tuned band
      if (rowData?.isHit && rowData?.band != null) {
        const xPip = marginLeft + ((rowData.band + 0.5) / NUM_BANDS) * plotW;
        ctx.fillStyle = "#49df9d";
        ctx.beginPath();
        ctx.arc(xPip, y + rowH / 2, Math.max(1.8, rowH * 0.7), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Vertical Tuned Receiver Aperture Guideline
    if (currentBand !== undefined && currentBand !== null) {
      const cx = marginLeft + (currentBand / NUM_BANDS) * plotW;
      const bandW = (1 / NUM_BANDS) * plotW;
      ctx.strokeStyle = isDark ? "rgba(56,189,248,0.4)" : "rgba(37,99,235,0.45)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(cx + bandW / 2, marginTop);
      ctx.lineTo(cx + bandW / 2, marginTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Vertical Frequency Guidelines at 3, 6, 9, 12, 15, 18 GHz
    const freqGrid = [3000, 6000, 9000, 12000, 15000, 18000];
    ctx.strokeStyle = isDark ? "rgba(69,70,83,0.3)" : "rgba(203,213,225,0.7)";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([3, 3]);
    freqGrid.forEach((f) => {
      const x = marginLeft + ((f - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Bottom Frequency Labels
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = isDark ? "#8e919d" : "#334155";
    [0, 3000, 6000, 9000, 12000, 15000, 18000].forEach((f) => {
      const x = marginLeft + ((f - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      const label = f >= 1000 ? (f / 1000).toFixed(0) + "G" : "0M";
      ctx.fillText(label, x, marginTop + plotH + 4);
    });

    // Calibrated Y-axis Time History Scale (Left Margin)
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = isDark ? "#8e919d" : "#475569";
    const timeGrads = [
      { rowPct: 0.02, label: "NOW" },
      { rowPct: 0.20, label: "-2.5s" },
      { rowPct: 0.40, label: "-5.0s" },
      { rowPct: 0.60, label: "-7.5s" },
      { rowPct: 0.80, label: "-10.0s" },
      { rowPct: 0.98, label: "-14.0s" },
    ];

    timeGrads.forEach((tg) => {
      const y = marginTop + tg.rowPct * plotH;
      ctx.fillText(tg.label, marginLeft - 5, y);

      // Subtle horizontal time fiducial
      ctx.strokeStyle = isDark ? "rgba(69,70,83,0.2)" : "rgba(226,232,240,0.7)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(marginLeft, y);
      ctx.lineTo(marginLeft + plotW, y);
      ctx.stroke();
    });

    // Outer plot border
    ctx.strokeStyle = isDark ? "rgba(189,194,255,0.4)" : "rgba(203,213,225,0.9)";
    ctx.lineWidth = 0.8;
    ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

    // Interactive Hover Crosshair & Tooltip
    if (hoverInfo) {
      const { x, y, freqMHz, val, isHit, timeOffsetSec } = hoverInfo;
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.45)" : "rgba(15,23,42,0.45)";
      ctx.lineWidth = 0.6;
      ctx.setLineDash([2, 2]);

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(marginLeft, y);
      ctx.lineTo(marginLeft + plotW, y);
      ctx.stroke();
      ctx.setLineDash([]);

      const hitTag = isHit ? " [INTERCEPT HIT]" : "";
      const text = `T - ${timeOffsetSec}s · ${freqMHz.toLocaleString()} MHz (B${Math.floor(freqMHz / 500)}) · OCCUPANCY PRIORITY: ${Math.round(val)}%${hitTag}`;

      ctx.font = '9px "JetBrains Mono", monospace';
      const textW = ctx.measureText(text).width;
      const badgeX = Math.min(marginLeft + plotW - textW - 12, Math.max(marginLeft + 6, x + 8));
      const badgeY = Math.max(marginTop + 14, y - 10);

      ctx.fillStyle = isDark ? "rgba(15,23,42,0.92)" : "rgba(255,255,255,0.95)";
      ctx.strokeStyle = isHit ? "#49df9d" : (isDark ? "#38bdf8" : "#0284c7");
      ctx.lineWidth = 0.8;
      ctx.fillRect(badgeX - 4, badgeY - 10, textW + 8, 14);
      ctx.strokeRect(badgeX - 4, badgeY - 10, textW + 8, 14);

      ctx.fillStyle = isHit ? "#49df9d" : (isDark ? "#ffffff" : "#0f172a");
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, badgeX, badgeY - 3);
    }
  }, [waterfall, currentBand, palette, gain, hoverInfo, isDark]);

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        position: "relative",
        width: "100%",
        height: 380,
        background: isDark ? "var(--panel-lowest, #050711)" : "var(--panel-2, #f8fafc)",
        border: "1px solid var(--border, #454653)",
        cursor: "crosshair",
        transition: "background 0.15s ease, border-color 0.15s ease",
      }}
    >
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />
    </div>
  );
}

function TelemetryInspector({ telemetry }) {
  if (!telemetry) {
    return (
      <div className="st-panel">
        <PanelHead icon="inventory_2" title="TELEMETRY INSPECTOR" badge="NO BACKEND PACKET" badgeColor="#f59e0b" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          No backend telemetry packet has been received. Synthetic RF telemetry remains active.
        </div>
      </div>
    );
  }

  const rows = [
    ["SOURCE", telemetry.source ?? "—"],
    ["LIVE", telemetry.live ? "YES" : "NO"],
    ["SCHEMA", telemetry.schemaVersion ?? "—"],
    ["TYPE", telemetry.type ?? "—"],
    ["MESSAGE", telemetry.message ?? "No message"],
    ["STEP", telemetry.step ?? "—"],
    ["EPISODE", telemetry.episode ?? "—"],
  ];

  const valid = telemetry.valid === true && telemetry.live === true;

  return (
    <div className="st-panel">
      <PanelHead
        icon="inventory_2"
        title="TELEMETRY INSPECTOR"
        badge={valid ? "VALID" : "INVALID"}
        badgeColor={valid ? "#49df9d" : "#f59e0b"}
      />
      <div className="st-grid-12" style={{ gap: 4 }}>
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="st-tsm"
            style={{
              gridColumn: "span 3 / span 3",
              display: "flex",
              flexDirection: "column",
              gap: 2,
              padding: "4px 6px",
              background: "var(--panel)",
              border: "1px solid var(--border)",
            }}
          >
            <span style={{ color: "var(--muted)" }}>{label}</span>
            <strong style={{ color: "var(--text)" }}>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LiveSpectrum() {
  const [selectedBand, setSelectedBand] = useState(
    syntheticSystem.spectrum.currentBand ?? 16
  );
  const [backendData, setBackendData] = useState(null);
  const [streamStatus, setStreamStatus] = useState("SYNTHETIC");
  const [liveTelemetry, setLiveTelemetry] = useState(null);
  const [liveEvents, setLiveEvents] = useState(SYNTHETIC_EVENTS);

  // Operator display controls
  const [traceMode, setTraceMode] = useState("LIVE_PEAK"); // "LIVE_PEAK", "LIVE_ONLY", "PEAK_ONLY"
  const [waterfallPalette, setWaterfallPalette] = useState("tactical"); // "tactical", "phosphor", "magma"
  const [waterfallGain, setWaterfallGain] = useState(1.0);

  // High-Resolution Spectrogram Waterfall Buffer (populated exclusively from real telemetry)
  const [waterfall, setWaterfall] = useState([]);

  const [userSelectedBand, setUserSelectedBand] = useState(false);
  const userSelectedBandRef = useRef(false);
  const hasRealTelemetryRef = useRef(false);
  const lastRestStepRef = useRef(null);

  // REST polling fallback
  useEffect(() => {
    let active = true;
    async function loadTelemetry() {
      try {
        const data = await loadLiveTelemetry();
        if (!active) return;
        setBackendData(data);
        if (data.telemetry && typeof data.telemetry === "object") {
          const raw = data.telemetry;
          if (raw.live === true) {
            hasRealTelemetryRef.current = true;
            setStreamStatus("CONNECTED");
            setLiveTelemetry((prev) => ({
              valid: true,
              source: raw.source ?? "publisher",
              live: true,
              step: raw.step ?? raw.metrics?.step ?? prev?.step,
              band: raw.band ?? raw.metrics?.band ?? prev?.band,
              bandPriorities: raw.bandPriorities ?? raw.metrics?.band_priorities ?? prev?.bandPriorities ?? [],
              mode: raw.mode ?? raw.metrics?.mode ?? prev?.mode,
              modeName: raw.mode_name ?? raw.metrics?.mode_name ?? prev?.modeName,
              hit: raw.hit ?? raw.metrics?.hit ?? prev?.hit,
              dwellTimeUs: raw.dwell_time_us ?? raw.metrics?.dwell_time_us ?? prev?.dwellTimeUs,
              rollingPd: raw.metrics?.system_metrics?.rolling_pd ?? raw.rolling_pd ?? prev?.rollingPd,
              rollingMedianLatencyUs: raw.metrics?.system_metrics?.rolling_median_latency_us ?? raw.rolling_median_latency_us ?? prev?.rollingMedianLatencyUs,
              cognitiveExplanation: raw.cognitive_explanation ?? raw.metrics?.cognitive_explanation ?? prev?.cognitiveExplanation,
              systemMetrics: raw.system_metrics ?? raw.metrics?.system_metrics ?? prev?.systemMetrics,
              clockUs: raw.clock_us ?? raw.metrics?.clock_us ?? prev?.clockUs,
              emitters: raw.emitters ?? raw.metrics?.emitters ?? prev?.emitters ?? [],
              pdws: raw.pdws ?? raw.metrics?.pdws ?? prev?.pdws ?? [],
              allIncidentPdws: raw.all_incident_pdws ?? raw.metrics?.all_incident_pdws ?? prev?.allIncidentPdws ?? [],
              raw: raw,
            }));

            const liveBand = raw.band ?? raw.metrics?.band;
            const currentStep = raw.step ?? raw.metrics?.step;
            if (liveBand !== undefined && liveBand !== null && currentStep !== lastRestStepRef.current) {
              lastRestStepRef.current = currentStep;
              if (!userSelectedBandRef.current) setSelectedBand(liveBand);
              const isHit = raw.hit ?? raw.metrics?.hit ?? false;
              const modeNames = ["SHORT_DWELL", "NORMAL_DWELL", "LONG_DWELL", "REVISIT", "PREEMPTIVE"];
              const mName = raw.mode_name ?? raw.metrics?.mode_name ?? modeNames[raw.mode ?? 1] ?? "NORMAL_DWELL";
              const clk = raw.clock_us ?? raw.metrics?.clock_us ?? 0;

              setLiveEvents((prev) => [
                { time: `T+${(clk / 1000).toFixed(1)} ms`, band: liveBand, frequencyMHz: liveBand * 500 + 250, type: isHit ? "HIT" : "SEARCH", mode: mName },
                ...prev.slice(0, 9),
              ]);

              setWaterfall((prev) => {
                const newRow = createWaterfallRow({
                  liveBand,
                  isHit,
                  telemetry: raw,
                  priorRow: prev[0],
                  step: currentStep,
                  clockUs: clk,
                  modeName: mName,
                });
                return [newRow, ...prev.slice(0, WATERFALL_ROWS - 1)];
              });
            }
          }
        }
      } catch {
        if (!active) return;
      }
    }
    loadTelemetry();
    const interval = setInterval(loadTelemetry, 1000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  // WebSocket live streaming
  useEffect(() => {
    let active = true;
    let backendConnection = null;
    let syntheticConnection = null;
    let fallbackTimer = null;

    function startSyntheticFallback() {
      if (!active || syntheticConnection || hasRealTelemetryRef.current) return;
      syntheticConnection = startSyntheticStream({
        intervalMs: 1000,
        currentBand: 16,
        onStatus() { if (active && !hasRealTelemetryRef.current) setStreamStatus("SYNTHETIC"); },
        onTelemetry(telemetry) { if (active && !hasRealTelemetryRef.current) setLiveTelemetry(telemetry); },
      });
    }

    function scheduleSyntheticFallback() {
      if (fallbackTimer) clearTimeout(fallbackTimer);
      if (hasRealTelemetryRef.current) return;
      fallbackTimer = setTimeout(() => {
        if (!active || hasRealTelemetryRef.current) return;
        startSyntheticFallback();
      }, 3500);
    }

    try {
      backendConnection = startTelemetryStream({
        onStatus(status) {
          if (!active) return;
          if (status === "CONNECTED") {
            if (fallbackTimer) clearTimeout(fallbackTimer);
            if (syntheticConnection) { syntheticConnection.close(); syntheticConnection = null; }
            setStreamStatus("CONNECTED");
          } else if (status === "RECONNECTING") {
            if (!hasRealTelemetryRef.current) {
              setStreamStatus("RECONNECTING");
              scheduleSyntheticFallback();
            }
          } else {
            setStreamStatus(status);
          }
        },
        onTelemetry(telemetry) {
          if (!active || !telemetry?.valid) return;
          if (fallbackTimer) clearTimeout(fallbackTimer);
          if (syntheticConnection) { syntheticConnection.close(); syntheticConnection = null; }

          setLiveTelemetry(telemetry);
          if (telemetry.live) {
            hasRealTelemetryRef.current = true;
            setStreamStatus("CONNECTED");

            const liveBand = telemetry.band ?? telemetry.metrics?.band;
            if (liveBand !== undefined && liveBand !== null) {
              if (!userSelectedBandRef.current) setSelectedBand(liveBand);
              const isHit = telemetry.hit ?? telemetry.metrics?.hit ?? false;
              const modeNames = ["SHORT_DWELL", "NORMAL_DWELL", "LONG_DWELL", "REVISIT", "PREEMPTIVE"];
              const mName = telemetry.modeName ?? telemetry.metrics?.mode_name ?? modeNames[telemetry.mode ?? 1] ?? "NORMAL_DWELL";
              const clk = telemetry.clockUs ?? telemetry.metrics?.clock_us ?? 0;
              const curStep = telemetry.step ?? telemetry.metrics?.step ?? 0;

              setLiveEvents((prev) => [
                { time: `T+${(clk / 1000).toFixed(1)} ms`, band: liveBand, frequencyMHz: liveBand * 500 + 250, type: isHit ? "HIT" : "SEARCH", mode: mName },
                ...prev.slice(0, 9),
              ]);

              setWaterfall((prev) => {
                const newRow = createWaterfallRow({
                  liveBand,
                  isHit,
                  telemetry,
                  priorRow: prev[0],
                  step: curStep,
                  clockUs: clk,
                  modeName: mName,
                });
                return [newRow, ...prev.slice(0, WATERFALL_ROWS - 1)];
              });
            }
          } else {
            if (!hasRealTelemetryRef.current) setStreamStatus("CONNECTED_NO_LIVE_DATA");
          }
        },
        onError() {
          if (!active) return;
          if (!hasRealTelemetryRef.current) scheduleSyntheticFallback();
        },
        onClose() {
          if (!active) return;
          if (!hasRealTelemetryRef.current) scheduleSyntheticFallback();
        },
      });
    } catch {
      scheduleSyntheticFallback();
    }

    return () => {
      active = false;
      if (fallbackTimer) clearTimeout(fallbackTimer);
      backendConnection?.close();
      syntheticConnection?.close();
    };
  }, []);

  const usingBackend = backendData?.connected === true || streamStatus === "CONNECTED";
  const hasRealTelemetry =
    (liveTelemetry?.source === "backend" || liveTelemetry?.source === "publisher" || liveTelemetry?.source?.startsWith("run:")) &&
    liveTelemetry?.live === true;

  const currentScheduledBand = hasRealTelemetry && liveTelemetry?.band !== undefined && liveTelemetry?.band !== null
    ? liveTelemetry.band : selectedBand;

  const currentFrequencyMHz = useMemo(
    () => (userSelectedBand ? selectedBand : currentScheduledBand) * 500 + 250,
    [userSelectedBand, selectedBand, currentScheduledBand]
  );

  const activeBands = useMemo(() => {
    if (hasRealTelemetry) {
      const set = new Set();
      if (liveEvents.length > 0) {
        liveEvents.forEach((e) => set.add(e.band));
      }
      if (Array.isArray(liveTelemetry?.emitters)) {
        liveTelemetry.emitters.forEach((em) => {
          if (em.freq_min_mhz != null) {
            const b = Math.floor(em.freq_min_mhz / 500);
            if (b >= 0 && b < NUM_BANDS) set.add(b);
          }
          if (em.frequency_mhz != null) {
            const b = Math.floor(em.frequency_mhz / 500);
            if (b >= 0 && b < NUM_BANDS) set.add(b);
          }
        });
      }
      if (Array.isArray(liveTelemetry?.pdws)) {
        liveTelemetry.pdws.forEach((p) => {
          if (p.frequency_mhz != null) {
            const b = Math.floor(p.frequency_mhz / 500);
            if (b >= 0 && b < NUM_BANDS) set.add(b);
          }
        });
      }
      if (set.size > 0) return set;
    }
    return new Set([1, 3, 8, 16, 20, 28]);
  }, [hasRealTelemetry, liveEvents, liveTelemetry]);

  const events = liveEvents.slice(0, 10);

  const bandHeights = useMemo(
    () =>
      Array.from({ length: NUM_BANDS }, (_, band) => {
        if (hasRealTelemetry) {
          const prio = liveTelemetry?.bandPriorities?.[band] ?? liveTelemetry?.metrics?.band_priorities?.[band];
          if (prio !== undefined && prio > 0) {
            return Math.min(95, Math.max(15, Math.round(prio * 100)));
          }
          if (band === currentScheduledBand) return 92;
          if (activeBands.has(band)) return 65;
          return 12 + ((band * 13) % 18);
        }
        const base = 18 + ((band * 17) % 45);
        return activeBands.has(band) ? Math.min(base + 30, 92) : base;
      }),
    [hasRealTelemetry, currentScheduledBand, activeBands, liveTelemetry]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* Header Banner */}
      <div className="st-panel" style={{ padding: 8 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <PanelHead
            icon="graphic_eq"
            title="0–18 GHz LIVE WIDEBAND SPECTRUM & INSTANTANEOUS RECEIVER APERTURE"
            badge="HIGH DENSITY RECORD"
            badgeColor="#49df9d"
          />
          <DataSourceBadge connected={hasRealTelemetry} />
        </div>
      </div>

      <div className="st-grid-12">
        {/* Left Column: Spectrum Trace & Waterfall History */}
        <div className="st-span-8" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {/* Real-Time Spectrum Trace Panel */}
          <div className="st-panel" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: "4px 8px",
                background: "var(--panel-2)",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 4,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="st-headline" style={{ color: "var(--accent)" }}>REAL-TIME SPECTRUM TRACE</span>
                <span className="st-mark" style={{ color: "var(--muted)" }}>-110 to -20 dBm · 500 MHz IBW</span>
              </div>
              {/* Trace Mode Selector */}
              <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
                <span style={{ fontSize: 9, color: "var(--muted)", marginRight: 4 }}>TRACE:</span>
                {[
                  ["LIVE_PEAK", "LIVE + MAX HOLD"],
                  ["LIVE_ONLY", "LIVE ONLY"],
                  ["PEAK_ONLY", "MAX HOLD"],
                ].map(([modeKey, modeLabel]) => (
                  <button
                    key={modeKey}
                    type="button"
                    style={{
                      background: traceMode === modeKey ? "var(--accent)" : "transparent",
                      color: traceMode === modeKey ? "#ffffff" : "var(--muted)",
                      border: `1px solid ${traceMode === modeKey ? "var(--accent)" : "var(--border)"}`,
                      fontSize: 9,
                      padding: "1px 6px",
                      cursor: "pointer",
                      fontFamily: '"JetBrains Mono", monospace',
                      fontWeight: 600,
                    }}
                    onClick={() => setTraceMode(modeKey)}
                  >
                    {modeLabel}
                  </button>
                ))}
              </div>
            </div>
            <SpectrumCanvas
              bandHeights={bandHeights}
              currentBand={userSelectedBand ? selectedBand : currentScheduledBand}
              activeBands={activeBands}
              liveTelemetry={liveTelemetry}
              traceMode={traceMode}
            />
          </div>

          {/* Waterfall History Spectrogram Record */}
          <div className="st-panel" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: "4px 8px",
                background: "var(--panel-2)",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 4,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="st-headline" style={{ color: "var(--accent)" }}>CHANNEL ACTIVITY WATERFALL RECORD</span>
                <span className="st-badge" style={{ color: "var(--success)" }}>36 BANDS (0–18 GHz) · OCCUPANCY HEATMAP</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                {/* Palette Selector */}
                <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <span style={{ fontSize: 9, color: "var(--muted)", marginRight: 2 }}>PALETTE:</span>
                  {[
                    ["tactical", "TACTICAL"],
                    ["phosphor", "PHOSPHOR"],
                    ["magma", "MAGMA"],
                  ].map(([pKey, pLabel]) => (
                    <button
                      key={pKey}
                      type="button"
                      style={{
                        background: waterfallPalette === pKey ? "var(--secondary)" : "transparent",
                        color: waterfallPalette === pKey ? "#ffffff" : "var(--muted)",
                        border: `1px solid ${waterfallPalette === pKey ? "var(--secondary)" : "var(--border)"}`,
                        fontSize: 9,
                        padding: "1px 5px",
                        cursor: "pointer",
                        fontFamily: '"JetBrains Mono", monospace',
                        fontWeight: 600,
                      }}
                      onClick={() => setWaterfallPalette(pKey)}
                    >
                      {pLabel}
                    </button>
                  ))}
                </div>

                {/* Gain Booster */}
                <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <span style={{ fontSize: 9, color: "var(--muted)", marginRight: 2 }}>GAIN:</span>
                  {[1.0, 1.5, 2.0].map((gVal) => (
                    <button
                      key={gVal}
                      type="button"
                      style={{
                        background: waterfallGain === gVal ? "var(--accent)" : "transparent",
                        color: waterfallGain === gVal ? "#ffffff" : "var(--muted)",
                        border: `1px solid ${waterfallGain === gVal ? "var(--accent)" : "var(--border)"}`,
                        fontSize: 9,
                        padding: "1px 4px",
                        cursor: "pointer",
                        fontFamily: '"JetBrains Mono", monospace',
                        fontWeight: 600,
                      }}
                      onClick={() => setWaterfallGain(gVal)}
                    >
                      {gVal.toFixed(1)}x
                    </button>
                  ))}
                </div>

                {/* Reset Buffer */}
                <button
                  type="button"
                  style={{
                    background: "transparent",
                    border: "1px solid var(--border)",
                    color: "var(--muted)",
                    fontSize: 9,
                    padding: "1px 6px",
                    cursor: "pointer",
                    fontFamily: '"JetBrains Mono", monospace',
                  }}
                  onClick={() => {
                    setWaterfall(
                      Array.from({ length: WATERFALL_ROWS }, (_, rowIndex) => ({
                        bins: new Float32Array(WATERFALL_BINS),
                        timeLabel: `T-${(rowIndex * 0.1).toFixed(1)}s`,
                        timestamp: Date.now() - rowIndex * 100,
                        clockUs: 0,
                        band: null,
                        isHit: false,
                        hitFreqs: [],
                        modeName: "NORMAL_DWELL",
                      }))
                    );
                  }}
                  title="Clear spectrogram history buffer"
                >
                  CLEAR
                </button>
              </div>
            </div>
            <div style={{ fontSize: 9, color: "var(--muted, #908f9e)", padding: "2px 8px", background: "var(--panel-3, #222428)", borderBottom: "1px solid var(--border-subtle, #333539)" }}>
              * Intensity represents backend channel occupancy / activity priority vector (36 bands × 500 MHz), not measured RF power in dBm.
            </div>
            <WaterfallCanvas
              waterfall={waterfall}
              currentBand={userSelectedBand ? selectedBand : currentScheduledBand}
              palette={waterfallPalette}
              gain={waterfallGain}
            />
          </div>

          {/* Canonical 36-Band Selection Grid */}
          <div className="st-panel" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: "4px 8px",
                background: "var(--panel-2)",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <span className="st-headline" style={{ color: "var(--secondary)" }}>BAND SELECT</span>
              {userSelectedBand && (
                <button
                  type="button"
                  style={{
                    background: "transparent",
                    border: "1px solid var(--accent)",
                    color: "var(--accent)",
                    fontSize: 10,
                    padding: "1px 6px",
                    cursor: "pointer",
                  }}
                  onClick={() => {
                    setUserSelectedBand(false);
                    userSelectedBandRef.current = false;
                  }}
                >
                  FOLLOW RECEIVER
                </button>
              )}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(18, 1fr)", gap: 1, padding: 3 }}>
              {Array.from({ length: NUM_BANDS }, (_, band) => {
                const isTuned = (userSelectedBand ? selectedBand : currentScheduledBand) === band;
                return (
                  <button
                    key={band}
                    style={{
                      cursor: "pointer",
                      padding: "1px 0",
                      font: '600 9px/1.2 "JetBrains Mono", monospace',
                      textAlign: "center",
                      background: isTuned ? "var(--accent)" : "var(--panel)",
                      color: isTuned ? "#ffffff" : activeBands.has(band) ? "var(--secondary)" : "var(--muted)",
                      border: `1px solid ${isTuned ? "var(--accent)" : "var(--border)"}`,
                      transition: "background 0.15s ease, color 0.15s ease",
                    }}
                    onClick={() => {
                      setSelectedBand(band);
                      setUserSelectedBand(true);
                      userSelectedBandRef.current = true;
                    }}
                  >
                    B{band}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Receiver State, Smart Scheduler, Recent Events */}
        <aside className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="st-panel">
            <PanelHead
              icon="settings_input_antenna"
              title="RECEIVER STATE"
              badge={hasRealTelemetry ? "STREAMING REAL RF" : "SYNTHETIC"}
              badgeColor={hasRealTelemetry ? "#49df9d" : "#f59e0b"}
            />
            <span className="st-tsm" style={{ color: "var(--muted)" }}>CENTER FREQUENCY</span>
            <div className="st-tlg" style={{ color: "var(--accent)" }}>
              {currentFrequencyMHz.toLocaleString()} MHz
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {(() => {
                const rollingPd = liveTelemetry?.rollingPd ?? liveTelemetry?.metrics?.system_metrics?.rolling_pd ?? liveTelemetry?.metrics?.rolling_pd;
                const rollingMedLat = liveTelemetry?.rollingMedianLatencyUs ?? liveTelemetry?.metrics?.system_metrics?.rolling_median_latency_us ?? liveTelemetry?.metrics?.rolling_median_latency_us;
                return [
                  ["BAND", `B${currentScheduledBand}`],
                  ["IBW", "500 MHz (Canonical)"],
                  ["THRESHOLD", "-105 dBm (MDS)"],
                  ["INTERCEPT RATE (Pd)", hasRealTelemetry && rollingPd != null ? `${(rollingPd * 100).toFixed(1)}%` : (usingBackend ? "0.0%" : "74.0%")],
                  ["MEDIAN LATENCY", hasRealTelemetry && rollingMedLat != null ? `${rollingMedLat.toFixed(1)} µs` : (usingBackend ? "0.0 µs" : "110 µs")],
                  ["TELEMETRY STREAM", streamStatus],
                  ["REST BACKEND", usingBackend ? "AVAILABLE" : "OFFLINE"],
                ].map(([label, value]) => (
                  <div key={label} className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "3px 6px", background: "var(--panel)", border: "1px solid var(--border)" }}>
                    <span style={{ color: "var(--muted)" }}>{label}</span>
                    <strong style={{ color: label === "TELEMETRY STREAM" && (streamStatus === "CONNECTED" || hasRealTelemetry) ? "var(--success)" : label === "INTERCEPT RATE (Pd)" ? "var(--success)" : "var(--text)" }}>
                      {value}
                    </strong>
                  </div>
                ));
              })()}
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="neurology" title="SMART SCHEDULER" badge="NEXT DECISION" badgeColor="#49df9d" />
            <div className="st-grid-12" style={{ gap: 4 }}>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "var(--muted)" }}>SELECTED</span>
                <strong className="st-tmd" style={{ color: "var(--success)" }}>B{currentScheduledBand}</strong>
                <span className="st-mark" style={{ color: "var(--text-muted)" }}>{currentFrequencyMHz.toLocaleString()} MHz</span>
              </div>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "var(--muted)" }}>MODE</span>
                <strong className="st-tmd" style={{ color: "var(--secondary)" }}>
                  {hasRealTelemetry ? (liveTelemetry?.modeName ?? liveTelemetry?.metrics?.mode_name ?? "NORMAL_DWELL") : (usingBackend ? "0.0" : syntheticSystem.scheduler.selectedMode)}
                </strong>
                <span className="st-mark" style={{ color: "var(--text-muted)" }}>
                  {(() => {
                    const dVal = liveTelemetry?.dwellTimeUs ?? liveTelemetry?.dwell_time_us ?? liveTelemetry?.metrics?.dwell_time_us;
                    return dVal != null && !isNaN(Number(dVal)) ? `${Number(dVal).toFixed(0)} µs` : (usingBackend ? "0.0 µs" : `${syntheticSystem.scheduler.dwellTimeUs} µs`);
                  })()}
                </span>
              </div>
            </div>
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "3px 6px", background: "var(--panel)", border: "1px solid var(--border)" }}>
              <span style={{ color: "var(--muted)" }}>PRIMARY DRIVER</span>
              <strong style={{ color: "var(--text)" }}>
                {hasRealTelemetry
                  ? (liveTelemetry?.cognitiveExplanation?.decision_reason ?? liveTelemetry?.metrics?.cognitive_explanation?.decision_reason ?? "DRQN Cognitive Policy")
                  : (usingBackend ? "0.0" : "Recent pulse activity")}
              </strong>
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="view_timeline" title="RECENT EVENTS" badge="SCAN TIMELINE" badgeColor="#96ccff" />
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {events.map((event, index) => (
                <div key={`${event.time}-${event.band}-${index}`} className="st-tsm" style={{ display: "flex", gap: 6, padding: "2px 6px", background: "var(--panel)", border: "1px solid var(--border-subtle)" }}>
                  <span style={{ color: "var(--muted)" }}>{event.time}</span>
                  <span style={{ color: "var(--accent)" }}>B{event.band}</span>
                  <span style={{ color: "var(--text)" }}>{event.frequencyMHz.toLocaleString()}</span>
                  <span style={{ color: "var(--text-muted)" }}>{event.mode}</span>
                  <strong style={{ color: event.type === "HIT" ? "var(--success)" : event.type === "DETECTION" ? "var(--secondary)" : event.type === "ARMED" ? "var(--accent)" : "var(--muted)", marginLeft: "auto" }}>
                    {event.type}
                  </strong>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <TelemetryInspector telemetry={liveTelemetry} />
    </div>
  );
}
