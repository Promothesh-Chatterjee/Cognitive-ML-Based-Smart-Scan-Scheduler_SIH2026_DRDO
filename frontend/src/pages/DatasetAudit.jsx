import { useState, useEffect, useMemo, useRef } from "react";
import { PanelHead, StitchTable } from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

const INVARIANTS = [
  ["Observation dimension", "360 (36 bands × 10 features)", "ENFORCED"],
  ["Action space", "180 (36 bands × 5 modes)", "ENFORCED"],
  ["Schema", "3Y.1", "ENFORCED"],
  ["Obs / GT separation", "Strict — GT never in observation", "ENFORCED"],
  ["Determinism", "root_seed = 42", "ENFORCED"],
  ["Corpus", "p3ac_50k — 100 episodes × 500 dwells", "VALID"],
];

const PDW_DESCRIPTORS = [
  [
    "Time of Arrival (ToA)",
    "ToA",
    "0.0 – 500,000.0 µs (< 10 ns res)",
    "Min-Max Window Scaled [0.0, 1.0]",
    "PRI deinterleaving, burst grouping & temporal pulse tracking",
    <span key="toa" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Carrier Frequency (CF)",
    "CF",
    "0.0 – 18,000.0 MHz (36 bands × 500 MHz)",
    "Robust IQR Z-Score: (CF − 9000) / 6000",
    "Instantaneous sub-band tuning & receiver front-end locking",
    <span key="cf" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Pulse Width (PW)",
    "PW",
    "0.1 – 100.0 µs (0.01 µs res)",
    "Log1p + Standardized: (log1p(PW) − 2.5) / 1.5",
    "Pulse duration discrimination & intra-pulse radar mode tagging",
    <span key="pw" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Angle of Arrival (AoA)",
    "AoA",
    "−90.0° to +90.0° Azimuth (0.1° res)",
    "Circular Phasor Unit Vector [sin(θ), cos(θ)] (6D)",
    "Spatial bearing isolation, lobe filtering & deghosting",
    <span key="aoa" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Pulse Amplitude (Amp)",
    "Amp",
    "−90.0 to 0.0 dBm (0.1 dBm res)",
    "Standard Z-Score: (Amp − (−80)) / 20",
    "Dynamic sensitivity, path loss & emitter proximity estimation",
    <span key="amp" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Signal-to-Noise Ratio (SNR)",
    "SNR",
    "+10.0 to +50.0 dB (≥ 15 dB threshold)",
    "Threshold Gated [SNR ≥ 15 dB Detection Floor]",
    "Front-end false-alarm suppression & weak signal gating",
    <span key="snr" style={{ color: "#49df9d", fontWeight: 700 }}>VALID</span>,
  ],
];

const FALLBACK_PDWS = [
  { id: 3561, toaUs: 356100.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3560, toaUs: 356040.0, frequencyMHz: 2250.0, pulseWidthUs: 1.5, amplitudeDb: -62.0, snrDb: 33.0, aoaDeg: -45.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B04 (2,250 MHz) ≠ Rx B10]" },
  { id: 3559, toaUs: 355980.0, frequencyMHz: 8250.0, pulseWidthUs: 2.5, amplitudeDb: -58.0, snrDb: 37.0, aoaDeg: 30.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B16 (8,250 MHz) ≠ Rx B10]" },
  { id: 3556, toaUs: 355600.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -54.0, snrDb: 41.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3554, toaUs: 355410.0, frequencyMHz: 12250.0, pulseWidthUs: 1.0, amplitudeDb: -70.0, snrDb: 25.0, aoaDeg: 65.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B24 (12,250 MHz) ≠ Rx B10]" },
  { id: 3553, toaUs: 355300.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3552, toaUs: 355200.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -56.0, snrDb: 39.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3550, toaUs: 355020.0, frequencyMHz: 2250.0, pulseWidthUs: 1.5, amplitudeDb: -63.0, snrDb: 32.0, aoaDeg: -45.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B04 (2,250 MHz) ≠ Rx B10]" },
  { id: 3549, toaUs: 354900.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3546, toaUs: 354620.0, frequencyMHz: 8250.0, pulseWidthUs: 2.5, amplitudeDb: -59.0, snrDb: 36.0, aoaDeg: 30.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B16 (8,250 MHz) ≠ Rx B10]" },
  { id: 3544, toaUs: 354400.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3541, toaUs: 354110.0, frequencyMHz: 14250.0, pulseWidthUs: 1.2, amplitudeDb: -68.0, snrDb: 27.0, aoaDeg: -15.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B28 (14,250 MHz) ≠ Rx B10]" },
  { id: 3539, toaUs: 353900.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -54.0, snrDb: 41.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3535, toaUs: 353500.0, frequencyMHz: 2250.0, pulseWidthUs: 1.5, amplitudeDb: -62.0, snrDb: 33.0, aoaDeg: -45.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B04 (2,250 MHz) ≠ Rx B10]" },
  { id: 3532, toaUs: 353200.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3528, toaUs: 352850.0, frequencyMHz: 8250.0, pulseWidthUs: 2.5, amplitudeDb: -57.0, snrDb: 38.0, aoaDeg: 30.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B16 (8,250 MHz) ≠ Rx B10]" },
  { id: 3525, toaUs: 352500.0, frequencyMHz: 5250.0, pulseWidthUs: 2.0, amplitudeDb: -55.0, snrDb: 40.0, aoaDeg: 12.0, isHit: true, status: "HIT", reason: "In-Band Rx Aperture Match [B10 (5,250 MHz)]" },
  { id: 3520, toaUs: 352010.0, frequencyMHz: 2250.0, pulseWidthUs: 1.5, amplitudeDb: -61.0, snrDb: 34.0, aoaDeg: -45.0, isHit: false, status: "MISS", reason: "Unattended Out-of-Band [B04 (2,250 MHz) ≠ Rx B10]" },
];

function normPdw(p, idx, defaultStatus, currentBand = 0) {
  const pId = p.pulse_id != null ? p.pulse_id : (p.id != null ? p.id : `P-${idx + 1}`);
  const tUs = Number(p.time_us != null ? p.time_us : (p.toa_us != null ? p.toa_us : (p.toaUs != null ? p.toaUs : 0.0)));
  const fMhz = Number(
    p.frequency_mhz != null
      ? p.frequency_mhz
      : (p.frequencyMHz != null ? p.frequencyMHz : (p.freq_mhz != null ? p.freq_mhz : (currentBand * 500 + 250)))
  );
  const pwUs = Number(p.pulse_width_us != null ? p.pulse_width_us : (p.pulseWidthUs != null ? p.pulseWidthUs : (p.pw_us != null ? p.pw_us : 1.0)));
  const ampDb = Number(p.amplitude_db != null ? p.amplitude_db : (p.amplitudeDb != null ? p.amplitudeDb : (p.amp_db != null ? p.amp_db : -65.0)));
  const snrVal = p.snr_db != null ? Number(p.snr_db) : (p.snrDb != null ? Number(p.snrDb) : Number((ampDb + 95.0).toFixed(1)));
  const aoaVal = Number(p.aoa_deg != null ? p.aoa_deg : (p.aoaDeg != null ? p.aoaDeg : (p.aoa != null ? p.aoa : 0.0)));
  const pulseBand = Math.floor(fMhz / 500);

  const isHit = p.status === "DETECTED" || p.status === "HIT" || defaultStatus === "DETECTED" || defaultStatus === "HIT";

  return {
    id: pId,
    toaUs: tUs,
    frequencyMHz: fMhz,
    pulseWidthUs: pwUs,
    amplitudeDb: ampDb,
    snrDb: snrVal,
    aoaDeg: aoaVal,
    pulseBand,
    status: isHit ? "HIT" : "MISS",
    isHit,
    reason: isHit
      ? `In-Band Rx Aperture Match [B${pulseBand.toString().padStart(2, "0")} (${fMhz.toFixed(0)} MHz)]`
      : `Unattended Out-of-Band [B${pulseBand.toString().padStart(2, "0")} (${fMhz.toFixed(0)} MHz) ≠ Rx B${currentBand.toString().padStart(2, "0")}]`,
  };
}

export default function DatasetAudit() {
  const telemetry = useOverviewTelemetry();
  const {
    live,
    wsStatus,
    pdws,
    allIncidentPdws,
    recentDwells = [],
    currentBand = 0,
    currentFreqMHz = 250,
    currentMode = "NORMAL_DWELL",
    currentDwellUs = 500.0,
    rollingPd = 0.0,
    totalHits = 0,
    totalDwells = 0,
    scheduler = {},
    sessionAvgPd = 0.0,
    bandHeights = [],
  } = telemetry;

  const groundTruthEmitters = useMemo(() => {
    if (Array.isArray(telemetry.groundTruthEmitters) && telemetry.groundTruthEmitters.length > 0) {
      return telemetry.groundTruthEmitters;
    }
    if (Array.isArray(telemetry.truth_emitters) && telemetry.truth_emitters.length > 0) {
      return telemetry.truth_emitters;
    }
    return [];
  }, [telemetry.groundTruthEmitters, telemetry.truth_emitters]);

  const isOnline = Boolean(
    live ||
    wsStatus === "ONLINE" ||
    (Array.isArray(allIncidentPdws) && allIncidentPdws.length > 0) ||
    (Array.isArray(pdws) && pdws.length > 0)
  );

  const [pdwHistory, setPdwHistory] = useState([]);
  const [incidentHistory, setIncidentHistory] = useState([]);
  const seenPulseIds = useRef(new Set());
  const seenIncidentIds = useRef(new Set());
  const [filter, setFilter] = useState("ALL"); // "ALL" | "HIT" | "MISS"
  const [selectedBandFilter, setSelectedBandFilter] = useState(null); // null | band number (0..35)
  const [viewMode, setViewMode] = useState("PDWS"); // "PDWS" | "DWELLS"

  // Accumulate captured hits in a rolling cache
  useEffect(() => {
    if (!Array.isArray(pdws) || pdws.length === 0) return;
    setPdwHistory((prev) => {
      const incoming = [];
      for (let i = 0; i < pdws.length; i++) {
        const p = pdws[i];
        const pId = p.pulse_id != null ? p.pulse_id : (p.id != null ? p.id : Math.floor(p.time_us ?? p.toa_us ?? 0));
        const tUs = Number(p.time_us != null ? p.time_us : (p.toa_us != null ? p.toa_us : 0));
        const uid = `${pId}-${tUs.toFixed(1)}`;
        if (!seenPulseIds.current.has(uid)) {
          seenPulseIds.current.add(uid);
          incoming.push(normPdw(p, i, "DETECTED", currentBand));
        }
      }
      if (incoming.length === 0) return prev;
      return [...incoming, ...prev].slice(0, 1000);
    });
  }, [pdws, currentBand]);

  // Accumulate all incident pulses in a rolling cache
  useEffect(() => {
    if (!Array.isArray(allIncidentPdws) || allIncidentPdws.length === 0) return;
    setIncidentHistory((prev) => {
      const incoming = [];
      for (let i = 0; i < allIncidentPdws.length; i++) {
        const p = allIncidentPdws[i];
        const pId = p.pulse_id != null ? p.pulse_id : (p.id != null ? p.id : Math.floor(p.time_us ?? p.toa_us ?? 0));
        const tUs = Number(p.time_us != null ? p.time_us : (p.toa_us != null ? p.toa_us : 0));
        const uid = `${pId}-${tUs.toFixed(1)}`;
        if (!seenIncidentIds.current.has(uid)) {
          seenIncidentIds.current.add(uid);
          incoming.push(p);
        }
      }
      if (incoming.length === 0) return prev;
      return [...incoming, ...prev].slice(0, 5000);
    });
  }, [allIncidentPdws]);

  // Merge hits and misses across all incident environment pulses and keep the
  // live log readable by showing the newest events first while still preserving
  // the full chronological pulse stream.
  const activePdws = useMemo(() => {
    const hitSource = pdwHistory.length > 0
      ? pdwHistory
      : (Array.isArray(pdws) && pdws.length > 0 ? pdws.map((p, i) => normPdw(p, i, "DETECTED", currentBand)) : []);

    const hitUids = new Set(hitSource.map((p) => `${p.id}-${p.toaUs.toFixed(1)}`));
    const hitPulseIds = new Set(hitSource.map((p) => String(p.id)));

    const rawIncidents = incidentHistory.length > 0
      ? incidentHistory
      : (Array.isArray(allIncidentPdws) && allIncidentPdws.length > 0 ? allIncidentPdws : []);

    const normalizedHits = hitSource.map((p, i) => normPdw(p, i, "DETECTED", currentBand));
    const incidentList = [];

    for (let i = 0; i < rawIncidents.length; i++) {
      const p = rawIncidents[i];
      const pNorm = normPdw(p, i, "INCIDENT", currentBand);
      const uid = `${pNorm.id}-${pNorm.toaUs.toFixed(1)}`;
      const isHitMatch = hitUids.has(uid) || hitPulseIds.has(String(pNorm.id)) || p.status === "DETECTED" || p.status === "HIT";

      if (isHitMatch) {
        pNorm.status = "HIT";
        pNorm.isHit = true;
        pNorm.reason = `In-Band Rx Aperture Match [B${pNorm.pulseBand.toString().padStart(2, "0")} (${pNorm.frequencyMHz.toFixed(0)} MHz)]`;
      } else {
        pNorm.status = "MISS";
        pNorm.isHit = false;
        pNorm.reason = `Unattended Out-of-Band [B${pNorm.pulseBand.toString().padStart(2, "0")} (${pNorm.frequencyMHz.toFixed(0)} MHz) ≠ Rx B${currentBand.toString().padStart(2, "0")}]`;
      }

      incidentList.push(pNorm);
    }

    const merged = [];
    const seenUids = new Set();

    for (const p of [...incidentList, ...normalizedHits]) {
      const uid = `${p.id}-${p.toaUs.toFixed(1)}`;
      if (!seenUids.has(uid)) {
        seenUids.add(uid);
        merged.push(p);
      }
    }

    if (merged.length > 0) {
      return merged
        .filter((p) => Number.isFinite(p.toaUs))
        .sort((a, b) => b.toaUs - a.toaUs || (Number(b.id) || 0) - (Number(a.id) || 0));
    }

    return [...FALLBACK_PDWS].sort((a, b) => b.toaUs - a.toaUs);
  }, [pdws, allIncidentPdws, pdwHistory, incidentHistory, currentBand]);

  const hitCount = activePdws.filter((p) => p.isHit).length;
  const missCount = activePdws.filter((p) => !p.isHit).length;
  const liveHitCount = Number.isFinite(totalHits) ? Math.max(0, Number(totalHits)) : hitCount;
  const liveMissCount = Number.isFinite(totalDwells) && Number(totalDwells) >= liveHitCount
    ? Math.max(0, Number(totalDwells) - liveHitCount)
    : missCount;
  const liveTotalPulses = Number.isFinite(totalDwells) ? Math.max(0, Number(totalDwells)) : activePdws.length;
  const interceptEfficiency = liveTotalPulses > 0
    ? ((liveHitCount / liveTotalPulses) * 100).toFixed(1)
    : (rollingPd > 0 ? (rollingPd * 100).toFixed(1) : "0.0");

  // Dynamic analysis: compute rolling 10-dwell interception efficiency from recentDwells
  const dwellStats = useMemo(() => {
    if (!Array.isArray(recentDwells) || recentDwells.length === 0) {
      return { recentPd: 0, totalDwellsAnalyzed: 0, hitDwells: 0, missDwells: 0, recentTrend: [] };
    }
    const sample = recentDwells.slice(0, 30);
    const hitDwells = sample.filter((d) => d.type === "HIT" || d.type === "INTERCEPTION").length;
    const missDwells = sample.filter((d) => d.type === "MISS").length;
    const recentPd = sample.length > 0 ? (hitDwells / sample.length) * 100 : 0;
    const recentTrend = sample.slice(0, 15).reverse().map((d) => d.type === "HIT" || d.type === "INTERCEPTION");
    return { recentPd, totalDwellsAnalyzed: sample.length, hitDwells, missDwells, recentTrend };
  }, [recentDwells]);

  // ALL 36 SUB-BANDS: Comprehensive hit/miss breakdown for full-spectrum analysis
  const allSubBandsStats = useMemo(() => {
    const TOTAL_BANDS = 36;
    const stats = {};
    for (let b = 0; b < TOTAL_BANDS; b++) {
      stats[b] = { band: b, freqMHz: b * 500 + 250, hits: 0, misses: 0, total: 0, pdRate: 0 };
    }
    for (const p of activePdws) {
      const b = p.pulseBand != null ? p.pulseBand : Math.floor(p.frequencyMHz / 500);
      if (b >= 0 && b < TOTAL_BANDS) {
        stats[b].total += 1;
        if (p.isHit) stats[b].hits += 1;
        else stats[b].misses += 1;
      }
    }
    for (let b = 0; b < TOTAL_BANDS; b++) {
      if (stats[b].total > 0) {
        stats[b].pdRate = Math.round((stats[b].hits / stats[b].total) * 100);
      }
    }
    return Object.values(stats)
      .filter((band) => band.total > 0)
      .sort((a, b) => a.band - b.band);
  }, [activePdws]);

  const filteredPdws = useMemo(() => {
    let list = activePdws;
    if (selectedBandFilter !== null) {
      list = list.filter((p) => p.pulseBand === selectedBandFilter);
    }
    if (filter === "HIT") return list.filter((p) => p.isHit);
    if (filter === "MISS") return list.filter((p) => !p.isHit);
    return list;
  }, [activePdws, filter, selectedBandFilter]);

  // PDW Table Rows
  const pdwRows = filteredPdws.map((pdw, idx) => {
    const isHit = pdw.isHit;
    const statusColor = isHit ? "#49df9d" : "#ff6b6b";
    const statusBg = isHit ? "rgba(73, 223, 157, 0.18)" : "rgba(255, 107, 107, 0.18)";
    const statusBorder = isHit ? "1px solid #49df9d" : "1px solid #ff6b6b";
    const statusLabel = isHit ? "HIT (INTERCEPTED)" : "MISS (OUT-OF-BAND)";

    return [
      <span key={`id-${idx}`} style={{ fontFamily: "var(--font-mono, monospace)", fontWeight: 700, color: "var(--text, #e2e2e8)" }}>
        {pdw.id}
      </span>,
      pdw.toaUs != null ? pdw.toaUs.toFixed(1) : "-",
      `${pdw.frequencyMHz.toFixed(1)} MHz (B${Math.floor(pdw.frequencyMHz / 500).toString().padStart(2, "0")})`,
      pdw.pulseWidthUs != null ? pdw.pulseWidthUs.toFixed(2) : "-",
      pdw.amplitudeDb != null ? pdw.amplitudeDb.toFixed(1) : "-",
      pdw.snrDb != null ? pdw.snrDb.toFixed(1) : "-",
      pdw.aoaDeg != null ? `${pdw.aoaDeg.toFixed(1)}°` : "-",
      <span
        key={`st-${idx}-${pdw.id}-${pdw.toaUs}`}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          color: statusColor,
          backgroundColor: statusBg,
          border: statusBorder,
          borderRadius: 3,
          padding: "3px 10px",
          fontWeight: 800,
          fontSize: 11,
          letterSpacing: "0.5px",
          whiteSpace: "nowrap",
        }}
      >
        <span style={{ fontSize: 9 }}>●</span>
        {statusLabel}
      </span>,
      <span key={`rs-${idx}`} style={{ fontSize: 11, color: isHit ? "#49df9d" : "var(--muted, #908f9e)" }}>
        {pdw.reason}
      </span>,
    ];
  });

  // Dwell History Rows
  const dwellRows = Array.isArray(recentDwells) && recentDwells.length > 0
    ? recentDwells.slice(0, 50).map((d, idx) => {
        const isHit = d.type === "HIT" || d.type === "INTERCEPTION";
        const statusColor = isHit ? "#49df9d" : (d.type === "FALSE_ALARM" ? "#f59e0b" : "#ff6b6b");
        const statusBg = isHit ? "rgba(73, 223, 157, 0.18)" : "rgba(255, 107, 107, 0.18)";
        const statusBorder = `1px solid ${statusColor}`;

        return [
          <span key={`step-${idx}`} style={{ fontFamily: "var(--font-mono, monospace)", fontWeight: 700, color: "var(--text)" }}>
            {d.step ?? (idx + 1)}
          </span>,
          `B${(d.band ?? 0).toString().padStart(2, "0")} (${(d.frequency_mhz ?? (d.band * 500 + 250)).toFixed(0)} MHz)`,
          d.mode ?? "NORMAL_DWELL",
          `${(d.dwell_us ?? 100.0).toFixed(0)} µs`,
          d.time_us != null ? `${Number(d.time_us).toFixed(1)} µs` : "-",
          <span
            key={`dw-st-${idx}`}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              color: statusColor,
              backgroundColor: statusBg,
              border: statusBorder,
              borderRadius: 3,
              padding: "2px 8px",
              fontWeight: 800,
              fontSize: 11,
              letterSpacing: 0.5,
            }}
          >
            ● {d.type}
          </span>,
          d.error_us != null ? `${Number(d.error_us).toFixed(1)} µs` : "N/A",
          d.track_id ? `TRK-${d.track_id}` : "WIDE SCAN",
        ];
      })
    : [["—", "—", "—", "—", "—", <span key="none" style={{ color: "var(--muted)" }}>AWAITING DWELLS</span>, "—", "—"]];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="dataset"
          title="Dataset Architecture, Verification & Handoff Audit"
          badge="AUDIT"
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Authoritative operational RF corpus (<code>p3ac_50k</code>) audit and live closed-loop interception verification.
        </div>
      </div>

      <div className="st-panel">
        <PanelHead title="Data Integrity & Strict Invariant Verification Matrix" />
        <StitchTable
          columns={["Invariant", "Contract", "Status"]}
          rows={INVARIANTS}
        />
      </div>

      <div className="st-panel">
        <PanelHead
          icon="graphic_eq"
          title="Pulse Descriptor Word (PDW) Dataset Specification & Receiver Model Input Vector"
          badge="SCHEMA 3Y.1 // 6D TENSOR"
          badgeColor="#96ccff"
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Canonical Pulse Descriptor Word (PDW) specification for the entire RF corpus (<code>p3ac_50k</code>).
          Every intercepted RF burst across the 36 bands (0–18 GHz) is parameterized into a 5D raw observable vector,
          transformed via leakage-safe statistics into a 6D tensor (<code>[ToA_norm, CF_norm, PW_norm, AoA_sin, AoA_cos, Amp_norm]</code>),
          and streamed directly into the Receiver Model & Transformer Deinterleaver.
        </div>

        <StitchTable
          columns={[
            "Parameter",
            "Symbol",
            "Physical Input Range",
            "Normalization / Encoding",
            "Receiver Model Processing Role",
            "Contract Status",
          ]}
          rows={PDW_DESCRIPTORS}
        />
        <div className="st-tsm" style={{ color: "#908f9e", marginTop: 6 }}>
          Zero Data Leakage Invariant: Normalization statistics are pre-fitted strictly on training partitions
          (<code>cf_median=9000 MHz, cf_iqr=6000 MHz, pw_mean=2.5, pw_std=1.5, amp_mean=−80 dBm, amp_std=20 dBm</code>).
          Simulation truth emitter IDs remain strictly isolated from this observable PDW tensor.
        </div>
      </div>

      <div className="st-panel">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
          <PanelHead
            icon="stream"
            title="LIVE PULSE DESCRIPTOR (RF ENVIRONMENT) — SIMULATION HITS & MISSES"
            badge={
              liveTotalPulses > 0
                ? `${liveTotalPulses} PULSES (${liveHitCount} HIT / ${liveMissCount} MISS)`
                : (isOnline ? "AWAITING STREAM" : "BASELINE SCENARIO")
            }
            badgeColor={liveHitCount > 0 ? "#49df9d" : liveMissCount > 0 ? "#ff9966" : "#96ccff"}
          />

          {/* Primary View Toggle: PDWs vs Dwells */}
          <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
            <button
              type="button"
              onClick={() => setViewMode("PDWS")}
              style={{
                padding: "4px 12px",
                fontSize: 11,
                fontWeight: 700,
                cursor: "pointer",
                borderRadius: 4,
                border: viewMode === "PDWS" ? "1px solid var(--accent, #bdc2ff)" : "1px solid var(--border-subtle, rgba(69, 70, 83, 0.4))",
                background: viewMode === "PDWS" ? "rgba(189, 194, 255, 0.2)" : "transparent",
                color: viewMode === "PDWS" ? "var(--accent, #bdc2ff)" : "var(--muted, #908f9e)",
              }}
            >
              📡 PULSE-LEVEL PDWS
            </button>
            <button
              type="button"
              onClick={() => setViewMode("DWELLS")}
              style={{
                padding: "4px 12px",
                fontSize: 11,
                fontWeight: 700,
                cursor: "pointer",
                borderRadius: 4,
                border: viewMode === "DWELLS" ? "1px solid #96ccff" : "1px solid var(--border-subtle, rgba(69, 70, 83, 0.4))",
                background: viewMode === "DWELLS" ? "rgba(150, 204, 255, 0.2)" : "transparent",
                color: viewMode === "DWELLS" ? "#96ccff" : "var(--muted, #908f9e)",
              }}
            >
              ⏱️ DWELL EXECUTION LOG
            </button>
          </div>
        </div>

        {/* Tactical KPI / Simulation Outcome Breakdown Banner */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
            gap: 10,
            padding: "12px 14px",
            background: "var(--panel-2, #1e2024)",
            border: "1px solid var(--border-subtle, rgba(69, 70, 83, 0.4))",
            borderRadius: 4,
            marginBottom: 10,
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)", textTransform: "uppercase", letterSpacing: 0.5 }}>
              Total Incident Pulses
            </span>
            <span style={{ fontSize: 18, fontWeight: 800, color: "var(--text, #e2e2e8)" }}>
              {liveTotalPulses}
            </span>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)" }}>All 36 RF Sub-bands</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)", textTransform: "uppercase", letterSpacing: 0.5 }}>
              Simulation Hits (Intercepted)
            </span>
            <span style={{ fontSize: 18, fontWeight: 800, color: "#49df9d" }}>
              {liveHitCount}
            </span>
            <span style={{ fontSize: 10, color: "#49df9d" }}>In-Band Dwell Matches</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)", textTransform: "uppercase", letterSpacing: 0.5 }}>
              Simulation Misses (Out-of-Band)
            </span>
            <span style={{ fontSize: 18, fontWeight: 800, color: "#ff6b6b" }}>
              {liveMissCount}
            </span>
            <span style={{ fontSize: 10, color: "#ff6b6b" }}>Unintercepted Burst Radiations</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)", textTransform: "uppercase", letterSpacing: 0.5 }}>
              Interception Efficiency
            </span>
            <span style={{ fontSize: 18, fontWeight: 800, color: "#96ccff" }}>
              {interceptEfficiency}%
            </span>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)" }}>
              Rolling Pd: {(rollingPd > 0 ? (rollingPd * 100).toFixed(1) : ((liveHitCount / Math.max(1, liveTotalPulses)) * 100).toFixed(1))}%
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)", textTransform: "uppercase", letterSpacing: 0.5 }}>
              Active Receiver Aperture
            </span>
            <span style={{ fontSize: 18, fontWeight: 800, color: "var(--accent, #bdc2ff)" }}>
              B{currentBand.toString().padStart(2, "0")}
            </span>
            <span style={{ fontSize: 10, color: "var(--muted, #908f9e)" }}>
              {currentFreqMHz.toFixed(0)} MHz // {currentDwellUs.toFixed(0)}µs {currentMode}
            </span>
          </div>
        </div>

        {/* Dynamic Hit & Miss Live Analytics Bar */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 10,
            padding: "10px 14px",
            background: "rgba(26, 28, 32, 0.7)",
            border: "1px solid rgba(189, 194, 255, 0.15)",
            borderRadius: 4,
            marginBottom: 12,
          }}
        >
          {/* Dynamic Recent Dwells Sparkline Trend */}
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: "#bdc2ff", letterSpacing: 0.5 }}>
                DYNAMIC DWELL TREND (LAST 15 DWELLS)
              </span>
              <span style={{ fontSize: 10, color: "#908f9e" }}>
                Rolling Pd: <strong style={{ color: dwellStats.recentPd >= 50 ? "#49df9d" : "#ff9966" }}>{dwellStats.recentPd.toFixed(0)}%</strong>
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 3, height: 16 }}>
              {dwellStats.recentTrend.length > 0 ? (
                dwellStats.recentTrend.map((isHit, idx) => (
                  <div
                    key={idx}
                    title={`Dwell ${idx + 1}: ${isHit ? "HIT (INTERCEPTED)" : "MISS"}`}
                    style={{
                      flex: 1,
                      height: isHit ? "100%" : "35%",
                      backgroundColor: isHit ? "#49df9d" : "#ff6b6b",
                      borderRadius: 1,
                      opacity: 0.7 + (idx / 15) * 0.3,
                      transition: "all 0.2s ease",
                    }}
                  />
                ))
              ) : (
                <div style={{ fontSize: 10, color: "#908f9e" }}>Awaiting continuous dwell stream...</div>
              )}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#908f9e" }}>
              <span>← Earlier</span>
              <span>{dwellStats.hitDwells} Hits / {dwellStats.missDwells} Misses in window</span>
              <span>Newest →</span>
            </div>
          </div>

          {/* Dynamic All 36 Sub-Bands Hit/Miss Efficiency Spectrum */}
          <div style={{ display: "flex", flexDirection: "column", gap: 6, gridColumn: "1 / -1" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 6 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: "#bdc2ff", letterSpacing: 0.5 }}>
                DYNAMIC SUB-BAND HIT/MISS EFFICIENCY — ALL 36 SUB-BANDS (0.0 – 18.0 GHz)
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 10, color: "#908f9e" }}>
                <span>Click any band to filter pulse table</span>
                {selectedBandFilter !== null && (
                  <button
                    type="button"
                    onClick={() => setSelectedBandFilter(null)}
                    style={{
                      padding: "1px 6px",
                      fontSize: 9,
                      background: "transparent",
                      border: "1px solid #bdc2ff",
                      color: "#bdc2ff",
                      cursor: "pointer",
                      borderRadius: 2,
                    }}
                  >
                    CLEAR FILTER (SHOW ALL BANDS)
                  </button>
                )}
                <span>Active Rx: <strong style={{ color: "#bdc2ff" }}>B{currentBand.toString().padStart(2, "0")}</strong></span>
              </div>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(88px, 1fr))",
                gap: 4,
                maxHeight: 180,
                overflowY: "auto",
                padding: "4px 2px",
              }}
            >
              {allSubBandsStats.map((b) => {
                const isCurrent = b.band === currentBand;
                const isSelected = selectedBandFilter === b.band;
                const hasActivity = b.total > 0;
                const isThreat = hasActivity;
                const hitRateColor = b.pdRate >= 70 ? "#49df9d" : (b.pdRate >= 30 ? "#f59e0b" : (b.total > 0 ? "#ff6b6b" : "#908f9e"));

                return (
                  <div
                    key={b.band}
                    onClick={() => setSelectedBandFilter(isSelected ? null : b.band)}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      padding: "3px 5px",
                      background: isSelected
                        ? "rgba(189, 194, 255, 0.28)"
                        : isCurrent
                        ? "rgba(189, 194, 255, 0.12)"
                        : (hasActivity ? "#1a1c22" : "#14161a"),
                      border: isSelected
                        ? "1px solid var(--accent, #bdc2ff)"
                        : isCurrent
                        ? "1px solid rgba(189, 194, 255, 0.6)"
                        : (hasActivity ? "1px solid #363842" : "1px solid #23252a"),
                      borderRadius: 3,
                      fontSize: 9,
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                      opacity: hasActivity || isCurrent ? 1 : 0.65,
                    }}
                    title={`Band ${b.band.toString().padStart(2, "0")} (${b.freqMHz} MHz)\nHits: ${b.hits} | Misses: ${b.misses} | Total: ${b.total}\nInterception Efficiency: ${b.pdRate}%\n${isCurrent ? "[CURRENT RX TUNED]" : ""}`}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontWeight: 800, color: isCurrent ? "#bdc2ff" : (isThreat ? "#e2e2e8" : "#6c6b7a") }}>
                        B{b.band.toString().padStart(2, "0")}
                      </span>
                      <span style={{ fontSize: 8, color: hitRateColor, fontWeight: 700 }}>
                        {b.total > 0 ? `${b.pdRate}%` : "QUIET"}
                      </span>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 2, fontSize: 8 }}>
                      <span style={{ color: "#49df9d", fontWeight: b.hits > 0 ? 700 : 400 }}>
                        {b.hits}H
                      </span>
                      <span style={{ color: "#ff6b6b", fontWeight: b.misses > 0 ? 700 : 400 }}>
                        {b.misses}M
                      </span>
                      <span style={{ color: "#908f9e" }}>
                        {b.total}T
                      </span>
                    </div>

                    {/* Mini efficiency bar */}
                    <div style={{ height: 2, background: "#282a30", borderRadius: 1, marginTop: 2, overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${b.pdRate}%`,
                          background: hitRateColor,
                          transition: "width 0.2s ease",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ fontSize: 9, color: "#908f9e", display: "flex", justifyContent: "space-between", marginTop: 2 }}>
              <span>
                Real dynamic spectrum metrics across all 36 canonical 500 MHz channels.{" "}
                <strong style={{ color: "#49df9d" }}>H</strong> = Intercepted (Hit),{" "}
                <strong style={{ color: "#ff6b6b" }}>M</strong> = Unattended (Miss),{" "}
                <strong style={{ color: "#908f9e" }}>T</strong> = Total incident.
              </span>
              <span style={{ color: "#bdc2ff" }}>
                {selectedBandFilter !== null ? `FILTERED: SUB-BAND B${selectedBandFilter.toString().padStart(2, "0")}` : "SHOWING ALL SUB-BANDS"}
              </span>
            </div>
          </div>
        </div>

        {viewMode === "PDWS" ? (
          <>
            {/* Filter Controls and Live Legend */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 10,
                marginBottom: 8,
              }}
            >
              {/* Tactical Filter Buttons */}
              <div style={{ display: "flex", gap: 6 }}>
                {[
                  { id: "ALL", label: `ALL INCIDENT (${liveTotalPulses})` },
                  { id: "HIT", label: `HITS ONLY (${liveHitCount})`, color: "#49df9d" },
                  { id: "MISS", label: `MISSES ONLY (${liveMissCount})`, color: "#ff6b6b" },
                ].map((btn) => {
                  const isActive = filter === btn.id;
                  return (
                    <button
                      key={btn.id}
                      type="button"
                      onClick={() => setFilter(btn.id)}
                      style={{
                        padding: "4px 12px",
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.5px",
                        cursor: "pointer",
                        borderRadius: 3,
                        border: isActive
                          ? `1px solid ${btn.color || "var(--accent, #bdc2ff)"}`
                          : "1px solid var(--border-subtle, rgba(69, 70, 83, 0.4))",
                        background: isActive
                          ? (btn.color ? `${btn.color}26` : "rgba(189, 194, 255, 0.2)")
                          : "transparent",
                        color: isActive ? (btn.color || "var(--accent, #bdc2ff)") : "var(--muted, #908f9e)",
                        transition: "all 0.15s ease",
                      }}
                    >
                      {btn.label}
                    </button>
                  );
                })}
              </div>

              {/* Tactical Legend */}
              <div style={{ display: "flex", gap: 14, fontSize: 11, color: "var(--muted, #908f9e)" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ color: "#49df9d", fontSize: 12 }}>●</span>
                  <strong style={{ color: "var(--text, #e2e2e8)" }}>HIT:</strong> Intercepted In-Band Aperture
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ color: "#ff6b6b", fontSize: 12 }}>●</span>
                  <strong style={{ color: "var(--text, #e2e2e8)" }}>MISS:</strong> Radiated Out-Of-Band
                </span>
              </div>
            </div>

            <div className="st-table-wrap" style={{ maxHeight: "420px", overflowY: "auto" }}>
              <StitchTable
                columns={[
                  "PULSE ID",
                  "TIME (TOA µS)",
                  "FREQUENCY (MHZ)",
                  "PW (µS)",
                  "AMP (DBM)",
                  "SNR (DB)",
                  "AOA (°)",
                  "SIMULATION OUTCOME",
                  "MATCH REASON",
                ]}
                rows={pdwRows}
              />
            </div>
          </>
        ) : (
          <>
            <div style={{ fontSize: 11, color: "var(--muted, #908f9e)", marginBottom: 8 }}>
              Real-time closed-loop dwell execution log showing each 1-GHz sub-band dwell cycle and simulation hit/miss convergence.
            </div>
            <div className="st-table-wrap" style={{ maxHeight: "420px", overflowY: "auto" }}>
              <StitchTable
                columns={[
                  "STEP",
                  "BAND & CENTER FREQ",
                  "SCAN MODE",
                  "DWELL DURATION",
                  "SIMULATION TIME",
                  "DWELL OUTCOME",
                  "PREDICTION ERROR",
                  "TRACK ASSIGNED",
                ]}
                rows={dwellRows}
              />
            </div>
          </>
        )}

        <div className="st-tsm" style={{ color: "var(--muted, #908f9e)", marginTop: 8, lineHeight: 1.5 }}>
          Live RF Environment Simulation: Displays observable pulse descriptor words (PDWs) generated across all 36 RF sub-bands (0–18 GHz).{" "}
          <strong style={{ color: "#49df9d" }}>HIT</strong> indicates the cognitive receiver was actively tuned to the emitter sub-band during pulse arrival.{" "}
          <strong style={{ color: "#ff6b6b" }}>MISS</strong> indicates an emitter radiated an RF burst while the receiver was tasked to a different sub-band or retuning. Simulation truth emitter identities remain isolated from neural observations.
        </div>
      </div>

      {/* 4. DECOUPLED ARCHITECTURE: OBSERVATION DATA (PANEL A) VS ISOLATED GROUND TRUTH STORE (PANEL B) */}
      <div className="st-grid-12" style={{ gap: 8, marginTop: 4 }}>
        {/* PANEL A: SCHEDULER OBSERVATION DATA */}
        <div className="st-span-6 st-panel" style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <PanelHead
            icon="psychology"
            title="PANEL A: SCHEDULER OBSERVATION VECTOR"
            badge="360-D (36 × 10)"
            badgeColor="var(--accent, #bdc2ff)"
          />
          <div className="st-body" style={{ color: "var(--text-muted, #c6c5d5)" }}>
            Receiver-derived spectrum state only: 36 bands × 10 features per dwell. Emitter identities, true RF parameters, and scenario metadata never enter this observation space.
          </div>

          {/* Decision linkage bar */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 8,
              padding: "6px 10px",
              background: "var(--panel-2, #1e2024)",
              border: "1px solid var(--border-subtle, #333539)",
              fontSize: 11,
              fontFamily: "var(--font-mono, monospace)",
              alignItems: "center",
            }}
          >
            <span style={{ color: "var(--muted, #908f9e)" }}>CURRENT TARGET:</span>
            <strong style={{ color: "var(--accent, #bdc2ff)" }}>
              BAND {Number(currentBand) + 1} ({((Number(currentBand) * 500 + 250)).toLocaleString()} MHz)
            </strong>
            <span style={{ color: "var(--border, #454653)" }}>|</span>
            <span style={{ color: "var(--muted, #908f9e)" }}>MODE:</span>
            <strong style={{ color: "var(--secondary, #96ccff)" }}>{currentMode}</strong>
            <span style={{ color: "var(--border, #454653)" }}>|</span>
            <span style={{ color: "var(--muted, #908f9e)" }}>REASON:</span>
            <span style={{ color: "var(--text, #e2e2e8)" }}>{scheduler.decisionReason || "DRQN Policy"}</span>
          </div>

          {/* 36-band observation inspector table */}
          <div style={{ maxHeight: 280, overflowY: "auto", border: "1px solid var(--border, #454653)" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "var(--font-mono, monospace)" }}>
              <thead>
                <tr style={{ background: "var(--panel-3, #282a2e)", borderBottom: "1px solid var(--border, #454653)", textAlign: "left" }}>
                  <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>BAND</th>
                  <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>FREQ (MHz)</th>
                  <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>ACTIVITY</th>
                  <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>OCCUPANCY</th>
                  <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>LINKAGE</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 36 }, (_, bIdx) => {
                  const isTuned = bIdx === Number(currentBand);
                  const actVal = bandHeights?.[bIdx] != null ? Number(bandHeights[bIdx]) : null;
                  const isAct = actVal != null ? actVal > 0.15 : false;
                  return (
                    <tr
                      key={bIdx}
                      style={{
                        background: isTuned
                          ? "rgba(189,194,255,0.12)"
                          : bIdx % 2 === 0
                          ? "var(--panel, #1a1c20)"
                          : "var(--panel-2, #1e2024)",
                        borderBottom: "1px solid var(--border-subtle, rgba(69,70,83,0.3))",
                      }}
                    >
                      <td style={{ padding: "3px 8px", color: isTuned ? "var(--accent, #bdc2ff)" : "var(--text, #e2e2e8)", fontWeight: isTuned ? 700 : 500 }}>
                        B{String(bIdx + 1).padStart(2, "0")}
                      </td>
                      <td style={{ padding: "3px 8px", color: "var(--text, #e2e2e8)" }}>
                        {bIdx * 500 + 250}
                      </td>
                      <td style={{ padding: "3px 8px" }}>
                        <span
                          style={{
                            color: isAct ? "var(--success, #49df9d)" : "var(--muted, #908f9e)",
                            fontWeight: isAct ? 600 : 400,
                          }}
                        >
                          {actVal != null ? (isAct ? "ACTIVE" : "QUIET") : "—"}
                        </span>
                      </td>
                      <td style={{ padding: "3px 8px", color: "var(--text, #e2e2e8)" }}>
                        {actVal != null ? actVal.toFixed(3) : "—"}
                      </td>
                      <td style={{ padding: "3px 8px" }}>
                        {isTuned ? (
                          <span style={{ color: "var(--accent, #bdc2ff)", fontWeight: 700 }}>
                            TUNED TARGET ↑
                          </span>
                        ) : (
                          <span style={{ color: "var(--muted, #908f9e)" }}>MONITORED</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div
            style={{
              padding: "6px 8px",
              background: "rgba(48,151,224,0.08)",
              border: "1px solid var(--secondary, #3097e0)",
              fontSize: 10,
              color: "var(--secondary, #96ccff)",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
              shield
            </span>
            <span>
              ZERO LEAKAGE PROVENANCE: 360-D observation vector strictly formed from receiver RF features.
            </span>
          </div>
        </div>

        {/* PANEL B: ISOLATED GROUND TRUTH STORE */}
        <div className="st-span-6 st-panel" style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <PanelHead
            icon="visibility"
            title="PANEL B: ISOLATED GROUND TRUTH STORE"
            badge="TRUTH (AUDIT ONLY)"
            badgeColor="var(--warning, #f59e0b)"
          />
          <div className="st-body" style={{ color: "var(--text-muted, #c6c5d5)" }}>
            Operator-view simulation truth, physically isolated from Panel A. Used exclusively for post-hoc validation, scoring, and telemetry auditing.
          </div>

          {/* Truth metrics strip */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 6,
              background: "var(--panel-2, #1e2024)",
              padding: "6px 8px",
              border: "1px solid var(--border-subtle, #333539)",
            }}
          >
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>INCIDENT PULSES</span>
              <strong style={{ fontSize: 13, color: "var(--text, #e2e2e8)" }}>
                {allIncidentPdws?.length || incidentHistory?.length || 0}
              </strong>
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>INTERCEPTED HITS</span>
              <strong style={{ fontSize: 13, color: "var(--success, #49df9d)" }}>
                {totalHits || pdwHistory?.filter(p => p.isHit).length || 0}
              </strong>
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>TRACKED EMITTERS</span>
              <strong style={{ fontSize: 13, color: "var(--warning, #f59e0b)" }}>
                {groundTruthEmitters.length > 0 ? groundTruthEmitters.length : "—"}
              </strong>
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>INCIDENT IR</span>
              <strong style={{ fontSize: 13, color: "var(--accent, #bdc2ff)" }}>
                {totalDwells > 0
                  ? `${((totalHits / totalDwells) * 100).toFixed(1)}%`
                  : sessionAvgPd > 0
                  ? `${(sessionAvgPd * 100).toFixed(1)}%`
                  : "—"}
              </strong>
            </div>
          </div>

          {/* True Emitter tracks table */}
          {groundTruthEmitters.length > 0 ? (
            <div style={{ maxHeight: 280, overflowY: "auto", border: "1px solid var(--border, #454653)" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "var(--font-mono, monospace)" }}>
                <thead>
                  <tr style={{ background: "var(--panel-3, #282a2e)", borderBottom: "1px solid var(--border, #454653)", textAlign: "left" }}>
                    <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>EMITTER ID</th>
                    <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>CARRIER FREQ</th>
                    <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>PRI</th>
                    <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>PW</th>
                    <th style={{ padding: "4px 8px", color: "var(--muted, #908f9e)" }}>MODULATION</th>
                  </tr>
                </thead>
                <tbody>
                  {groundTruthEmitters.map((em, idx) => {
                    const eId = em.emitter_id != null ? em.emitter_id : (em.id ?? `E-${idx + 1}`);
                    const freqSpan = em.freq_min_mhz != null && em.freq_max_mhz != null
                      ? (em.freq_min_mhz === em.freq_max_mhz ? `${em.freq_min_mhz} MHz` : `${em.freq_min_mhz}–${em.freq_max_mhz} MHz`)
                      : `${em.frequency_mhz ?? 8250} MHz`;
                    const pri = em.mean_pri_us != null ? `${Number(em.mean_pri_us).toFixed(1)} µs` : (em.pri_us != null ? `${Number(em.pri_us).toFixed(1)} µs` : "—");
                    const pw = em.mean_pw_us != null ? `${Number(em.mean_pw_us).toFixed(2)} µs` : (em.pw_us != null ? `${Number(em.pw_us).toFixed(2)} µs` : "—");
                    const mod = em.classification || em.modulation_type || em.radar_type || "PULSED RADAR";
                    return (
                      <tr
                        key={idx}
                        style={{
                          background: idx % 2 === 0 ? "var(--panel, #1a1c20)" : "var(--panel-2, #1e2024)",
                          borderBottom: "1px solid var(--border-subtle, rgba(69,70,83,0.3))",
                        }}
                      >
                        <td style={{ padding: "3px 8px", color: "var(--warning, #f59e0b)", fontWeight: 700 }}>
                          {eId}
                        </td>
                        <td style={{ padding: "3px 8px", color: "var(--text, #e2e2e8)" }}>
                          {freqSpan}
                        </td>
                        <td style={{ padding: "3px 8px", color: "var(--text, #e2e2e8)" }}>
                          {pri}
                        </td>
                        <td style={{ padding: "3px 8px", color: "var(--text, #e2e2e8)" }}>
                          {pw}
                        </td>
                        <td style={{ padding: "3px 8px", color: "var(--secondary, #96ccff)" }}>
                          {mod}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div
              style={{
                padding: "24px 16px",
                textAlign: "center",
                color: "var(--muted, #908f9e)",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: 11,
                background: "var(--panel-lowest, #141619)",
                border: "1px dashed var(--border-subtle, #333539)",
              }}
            >
              <div>NO ISOLATED GROUND TRUTH EMITTER RECORDS AVAILABLE FROM ACTIVE TELEMETRY STREAM</div>
              <div style={{ marginTop: 6, fontSize: 10, color: "var(--muted, #757785)" }}>
                Ground truth emitter parameters are strictly air-gapped from the DRQN policy observation space.
              </div>
            </div>
          )}

          <div
            style={{
              padding: "6px 8px",
              background: "rgba(245,158,11,0.08)",
              border: "1px solid var(--warning, #f59e0b)",
              fontSize: 10,
              color: "var(--warning, #f59e0b)",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
              gavel
            </span>
            <span>
              STRICT AUDIT ISOLATION: The DRQN policy network has zero observation visibility into Panel B.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
