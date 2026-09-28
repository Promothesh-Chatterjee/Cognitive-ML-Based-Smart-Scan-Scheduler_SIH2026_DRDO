/**
 * useOverviewTelemetry
 *
 * Single source-of-truth hook for the Mission Overview and live subsystem pages.
 * Powered by verified HTTP polling against the deployed FastAPI backend (/telemetry/latest,
 * /mission/status, /mission/stream/status, and /metrics).
 * Automatically stops polling when stopped or unmounted, with AbortController cancellation
 * and exponential backoff retry for cold starts.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { createTelemetryPoller, CONNECTION_STATES } from "./telemetryPoller";

const TOTAL_BANDS = 36;
const BAND_WIDTH_MHZ = 500;
const IBW_MHZ = 1000;
const DWELL_HISTORY_SIZE = 8;

const MODE_LABELS = {
  0: "SHORT_DWELL",
  1: "NORMAL_DWELL",
  2: "LONG_DWELL",
  3: "REVISIT",
  4: "PREEMPTIVE_INTERCEPT",
};

function bandToFreqMHz(band) {
  return band * BAND_WIDTH_MHZ + BAND_WIDTH_MHZ / 2;
}

function floatOr(val, fallback = 0.0) {
  const n = Number(val);
  return !isNaN(n) && isFinite(n) ? n : fallback;
}

function buildDefault() {
  return {
    connectionState: "CONNECTING",
    connectionLabel: CONNECTION_STATES.CONNECTING,
    wsStatus: "ONLINE",
    live: false,
    source: "none",
    streamRunning: false,
    missionActive: false,
    totalSpectrumGHz: 18,
    ibwMHz: IBW_MHZ,
    activeBands: 0,
    quietBands: TOTAL_BANDS,
    currentBand: 0,
    currentFreqMHz: 250,
    currentMode: "0.0",
    currentDwellUs: 0.0,
    totalHits: 0,
    totalDwells: 0,
    rollingPd: 0.0,
    rollingMedianLatencyUs: 0.0,
    bandHeights: Array(TOTAL_BANDS).fill(0.0),
    bandStates: Array(TOTAL_BANDS).fill("quiet"),
    scheduler: {
      chosenBand: 0,
      chosenFreqMHz: 250,
      scanMode: "0.0",
      dwellUs: 0.0,
      drqnScore: 0.0,
      interceptProbability: 0.0,
      predictedEtaUs: 0.0,
      actionSpace: 180,
      decisionReason: "0.0",
      explorationPressure: 0.0,
      qMargin: 0.0,
      moeGating: 0.0,
    },
    dwellHistory: [],
    cognitiveExplanation: {},
    systemMetrics: {},
    metrics: null,
    missionStatus: null,
    streamStatus: null,
    pdws: [],
    allIncidentPdws: [],
    emitters: [],
    recentDwells: [],
    modulationArchetypes: { periodic: 0, agile_hop: 0, strobe_cw: 0, total_species: 0 },
    threatTiers: { tier_1: 0, tier_2: 0, tier_3: 0, critical_count: 0 },
    fomMetrics: {
      mean_detection_latency_us: 0.0,
      median_detection_latency_us: 0.0,
      latency_delta_baseline_us: 0.0,
      missed_revisits_count: 0,
      total_revisits: 0,
      missed_revisits_pct: 0.0,
      cumulative_missed_revisits: 0,
      cumulative_total_revisits: 0,
      cumulative_missed_pct: 0.0,
      scheduler_omniscience_leak_pct: 0.0,
      antenna_azimuth_deg: 0.0,
      hop_trajectory: null,
    },
    benchmarkRows: null,
    modeRows: null,
    archetypeRows: null,
    lastError: null,
    instantaneousPd: 0.0,
    sessionAvgPd: 0.0,
    liveMetrics: null,
    waterfallHistory: [],
  };
}

function processTelemetry({ raw, missionStat, streamStat, metrics, connectionState }) {
  const next = buildDefault();

  next.connectionState = connectionState || "BACKEND_CONNECTED";
  next.connectionLabel = CONNECTION_STATES[next.connectionState] || next.connectionState;
  next.wsStatus = next.connectionState === "BACKEND_UNAVAILABLE" ? "OFFLINE" : "ONLINE";

  if (streamStat) {
    next.streamStatus = streamStat;
    next.streamRunning = Boolean(streamStat.running);
    if (streamStat.total_dwells != null) next.totalDwells = streamStat.total_dwells;
    if (streamStat.total_hits != null) next.totalHits = streamStat.total_hits;
    if (streamStat.rolling_pd != null) next.rollingPd = streamStat.rolling_pd;
    if (streamStat.rolling_median_latency_us != null) next.rollingMedianLatencyUs = streamStat.rolling_median_latency_us;
    if (streamStat.mission_clock_us != null) next.missionClockUs = streamStat.mission_clock_us;
  }

  if (missionStat) {
    next.missionStatus = missionStat;
    next.missionActive = Boolean(missionStat.is_mission_active);
    if (missionStat.mission_clock_us != null) next.missionClockUs = missionStat.mission_clock_us;
    if (missionStat.total_dwells != null) next.totalDwells = Math.max(next.totalDwells, missionStat.total_dwells);
    if (missionStat.total_hits != null) next.totalHits = Math.max(next.totalHits, missionStat.total_hits);
    if (missionStat.rolling_pd != null && next.rollingPd === 0) next.rollingPd = missionStat.rolling_pd;
    if (missionStat.rolling_median_latency_us != null && next.rollingMedianLatencyUs === 0) {
      next.rollingMedianLatencyUs = missionStat.rolling_median_latency_us;
    }
  }

  if (metrics) {
    next.metrics = metrics;
    next.systemMetrics = { ...metrics };
    if (metrics.rolling_pd != null && next.rollingPd === 0) next.rollingPd = metrics.rolling_pd;
    if (metrics.rolling_median_latency_us != null && next.rollingMedianLatencyUs === 0) {
      next.rollingMedianLatencyUs = metrics.rolling_median_latency_us;
    }
    if (metrics.operational_dwells != null) next.totalDwells = Math.max(next.totalDwells, metrics.operational_dwells);
    if (metrics.operational_hits != null) next.totalHits = Math.max(next.totalHits, metrics.operational_hits);
  }

  const isLive = Boolean(
    (raw && raw.live !== false && (raw.source === "publisher" || raw.step != null || raw.band != null)) ||
    next.streamRunning ||
    next.missionActive
  );

  if (!raw || !isLive) {
    next.live = isLive;
    next.source = raw?.source ?? (next.streamRunning ? "publisher" : "none");
    return next;
  }

  next.live = true;
  next.source = raw.source ?? "publisher";
  next.wsStatus = "ONLINE";

  const m = raw.metrics ?? raw;
  const band = m.band ?? 0;
  const modeIdx = m.mode ?? 1;
  const modeName = m.mode_name ?? MODE_LABELS[modeIdx] ?? "NORMAL_DWELL";
  const dwellUs = floatOr(m.dwell_time_us, floatOr(raw.dwell_time_us, 500.0));
  const clockUs = m.clock_us ?? raw.clock_us ?? next.missionClockUs ?? 0;

  next.currentBand = band;
  next.currentFreqMHz = bandToFreqMHz(band);
  next.currentMode = modeName;
  next.currentDwellUs = dwellUs;
  if (!missionStat && !streamStat) next.missionClockUs = clockUs;

  const ce = m.cognitive_explanation ?? raw.cognitive_explanation ?? {};
  next.cognitiveExplanation = ce;

  const sm = m.system_metrics ?? raw.system_metrics ?? {};
  next.systemMetrics = { ...next.systemMetrics, ...sm };

  if (next.rollingPd === 0) {
    next.rollingPd = sm.rolling_pd ?? raw.rolling_pd ?? m.rolling_pd ?? 0.0;
  }
  if (next.rollingMedianLatencyUs === 0) {
    next.rollingMedianLatencyUs = sm.rolling_median_latency_us ?? raw.rolling_median_latency_us ?? m.rolling_median_latency_us ?? 0.0;
  }

  const drqnScore = ce.drqn_score ?? m.drqn_score ?? raw.drqn_score ?? ce.action_score ?? 0.0;
  const interceptProb = ce.prediction_confidence ?? m.prediction_confidence ?? raw.prediction_confidence ?? 0.0;
  const etaUs = ce.predicted_eta_us ?? m.predicted_eta_us ?? raw.predicted_eta_us ?? ce.eta_us ?? 0.0;
  const reason = ce.decision_reason ?? m.decision_reason ?? raw.decision_reason ?? ce.reason ?? "DRQN Cognitive Policy";
  const explorationPressure = ce.exploration_pressure ?? m.exploration_pressure ?? raw.exploration_pressure ?? 0.0;
  const qMargin = ce.q_margin ?? m.q_margin ?? raw.q_margin ?? 0.0;

  next.scheduler = {
    chosenBand: band,
    chosenFreqMHz: next.currentFreqMHz,
    scanMode: modeName,
    dwellUs: dwellUs,
    drqnScore: drqnScore,
    interceptProbability: interceptProb,
    predictedEtaUs: Math.max(0, etaUs),
    actionSpace: 180,
    decisionReason: reason,
    explorationPressure: explorationPressure,
    qMargin: qMargin,
    moeGating: Math.max(0.0, 1.0 - explorationPressure),
  };

  const rawPriors = raw.band_priorities ?? m.band_priorities ?? raw.bandPriorities ?? m.bandPriorities;
  if (Array.isArray(rawPriors) && rawPriors.length === TOTAL_BANDS) {
    next.bandHeights = rawPriors.map((v) => Math.max(0.0, Math.min(1.0, Number(v) || 0.0)));
    next.bandStates = next.bandHeights.map((h, i) => (i === band ? "active" : (h > 0.1 ? "active" : "quiet")));
    next.activeBands = next.bandHeights.filter((h) => h > 0.1).length;
    next.quietBands = TOTAL_BANDS - next.activeBands;
  } else {
    next.bandHeights = Array(TOTAL_BANDS).fill(0.0);
    next.bandStates = Array(TOTAL_BANDS).fill("quiet").map((_, i) => (i === band ? "active" : "quiet"));
    next.activeBands = 1;
    next.quietBands = TOTAL_BANDS - 1;
  }

  const rawPdws = raw.pdws ?? m.pdws ?? raw.recent_pdws ?? m.recent_pdws ?? raw.detections ?? m.detections ?? [];
  next.pdws = Array.isArray(rawPdws) ? rawPdws : [];

  const rawIncidentPdws = raw.all_incident_pdws ?? m.all_incident_pdws ?? [];
  next.allIncidentPdws = Array.isArray(rawIncidentPdws) ? rawIncidentPdws : [];

  const rawDwells = raw.recent_dwells ?? m.recent_dwells ?? [];
  next.recentDwells = Array.isArray(rawDwells) ? rawDwells : [];

  const rawEmitters = raw.emitters ?? m.emitters ?? [];
  next.emitters = Array.isArray(rawEmitters) ? rawEmitters : [];

  const rawMod = raw.modulation_archetypes ?? m.modulation_archetypes ?? {};
  next.modulationArchetypes = {
    periodic: rawMod.periodic ?? 0,
    agile_hop: rawMod.agile_hop ?? 0,
    strobe_cw: rawMod.strobe_cw ?? 0,
    total_species: rawMod.total_species ?? 0,
  };

  const rawTiers = raw.threat_tiers ?? m.threat_tiers ?? {};
  next.threatTiers = {
    tier_1: rawTiers.tier_1 ?? 0,
    tier_2: rawTiers.tier_2 ?? 0,
    tier_3: rawTiers.tier_3 ?? 0,
    critical_count: rawTiers.critical_count ?? 0,
  };

  const rawFom = raw.fom_metrics ?? m.fom_metrics ?? {};
  next.fomMetrics = {
    mean_detection_latency_us: floatOr(rawFom.mean_detection_latency_us, 0.0),
    median_detection_latency_us: floatOr(rawFom.median_detection_latency_us, 0.0),
    latency_delta_baseline_us: floatOr(rawFom.latency_delta_baseline_us, 0.0),
    missed_revisits_count: rawFom.missed_revisits_count ?? 0,
    total_revisits: rawFom.total_revisits ?? 0,
    missed_revisits_pct: floatOr(rawFom.missed_revisits_pct, 0.0),
    cumulative_missed_revisits: rawFom.cumulative_missed_revisits ?? 0,
    cumulative_total_revisits: rawFom.cumulative_total_revisits ?? 0,
    cumulative_missed_pct: floatOr(rawFom.cumulative_missed_pct, 0.0),
    scheduler_omniscience_leak_pct: floatOr(rawFom.scheduler_omniscience_leak_pct, 0.0),
    antenna_azimuth_deg: floatOr(rawFom.antenna_azimuth_deg, 0.0),
    hop_trajectory: rawFom.hop_trajectory ?? null,
  };

  next.benchmarkRows = raw.benchmark_rows ?? m.benchmark_rows ?? null;
  next.modeRows = raw.mode_rows ?? m.mode_rows ?? null;
  next.archetypeRows = raw.archetype_rows ?? m.archetype_rows ?? null;

  return next;
}

export function useOverviewTelemetry(options = {}) {
  const defaultIntervalMs = options.intervalMs || 1000;
  const [pollingIntervalMs, setPollingIntervalMs] = useState(defaultIntervalMs);
  const [state, setState] = useState(buildDefault);

  const pollerRef = useRef(null);
  const dwellHistoryRef = useRef([]);
  const pdwsRef = useRef([]);
  const incidentPdwsRef = useRef([]);
  const recentDwellsRef = useRef([]);
  const recentDwellsWindowRef = useRef([]);
  const seenDwellKeysRef = useRef(new Set());
  const waterfallHistoryRef = useRef([]);

  const latestTelRef = useRef(null);
  const missionStatRef = useRef(null);
  const streamStatRef = useRef(null);
  const metricsRef = useRef(null);
  const connStateRef = useRef("CONNECTING");

  const ingest = useCallback(({ telemetry, missionStatus, streamStatus, metrics, connectionState }) => {
    if (telemetry) latestTelRef.current = telemetry;
    if (missionStatus) missionStatRef.current = missionStatus;
    if (streamStatus) streamStatRef.current = streamStatus;
    if (metrics) metricsRef.current = metrics;
    if (connectionState) connStateRef.current = connectionState;

    const resolved = processTelemetry({
      raw: latestTelRef.current,
      missionStat: missionStatRef.current,
      streamStat: streamStatRef.current,
      metrics: metricsRef.current,
      connectionState: connStateRef.current,
    });

    if (resolved.live) {
      if (Array.isArray(resolved.pdws) && resolved.pdws.length > 0) {
        const seen = new Set(
          pdwsRef.current.map((p) => `${p.pulse_id ?? p.id}-${Number(p.time_us ?? p.toa_us ?? 0).toFixed(1)}`),
        );
        const newItems = [];
        for (const p of resolved.pdws) {
          const uid = `${p.pulse_id ?? p.id}-${Number(p.time_us ?? p.toa_us ?? 0).toFixed(1)}`;
          if (!seen.has(uid)) {
            seen.add(uid);
            newItems.push(p);
          }
        }
        if (newItems.length > 0) {
          pdwsRef.current = [...newItems, ...pdwsRef.current].slice(0, 1000);
        } else if (pdwsRef.current.length === 0) {
          pdwsRef.current = [...resolved.pdws].slice(0, 1000);
        }
      }
      if (pdwsRef.current.length > 0 && (!resolved.pdws || resolved.pdws.length === 0)) {
        resolved.pdws = pdwsRef.current.slice(0, 1000);
      }

      if (Array.isArray(resolved.allIncidentPdws) && resolved.allIncidentPdws.length > 0) {
        const seenInc = new Set(
          incidentPdwsRef.current.map((p) => `${p.pulse_id ?? p.id}-${Number(p.time_us ?? p.toa_us ?? 0).toFixed(1)}`),
        );
        const newInc = [];
        for (const p of resolved.allIncidentPdws) {
          const uid = `${p.pulse_id ?? p.id}-${Number(p.time_us ?? p.toa_us ?? 0).toFixed(1)}`;
          if (!seenInc.has(uid)) {
            seenInc.add(uid);
            newInc.push(p);
          }
        }
        if (newInc.length > 0) {
          incidentPdwsRef.current = [...newInc, ...incidentPdwsRef.current].slice(0, 5000);
        } else if (incidentPdwsRef.current.length === 0) {
          incidentPdwsRef.current = [...resolved.allIncidentPdws].slice(0, 5000);
        }
      }
      resolved.allIncidentPdws = incidentPdwsRef.current.slice(0, 5000);

      if (Array.isArray(resolved.recentDwells) && resolved.recentDwells.length > 0) {
        const seen = new Set(recentDwellsRef.current.map((d) => d.id));
        const newItems = [];
        for (const d of resolved.recentDwells) {
          if (!seen.has(d.id)) {
            seen.add(d.id);
            newItems.push(d);
          }
          // Defect 2: Deduplicate distinct mission dwells into recentDwellsWindowRef using authoritative ID
          const authKey = d.id || `dwell-${d.step ?? 0}-${d.band ?? 0}-${Number(d.time_us ?? d.clock_us ?? 0).toFixed(1)}`;
          if (!seenDwellKeysRef.current.has(authKey)) {
            seenDwellKeysRef.current.add(authKey);
            const isHit = d.type === "HIT" || d.type === "INTERCEPTION" || Boolean(d.hit);
            recentDwellsWindowRef.current.push({
              id: authKey,
              hit: isHit,
              band: d.band,
              step: d.step ?? 0,
              error_us: d.error_us != null ? Number(d.error_us) : null,
            });
          }
        }
        if (newItems.length > 0) {
          recentDwellsRef.current = [...newItems, ...recentDwellsRef.current].slice(0, 150);
        } else if (recentDwellsRef.current.length === 0) {
          recentDwellsRef.current = [...resolved.recentDwells].slice(0, 150);
        }
      }
      if (recentDwellsRef.current.length > 0) {
        resolved.recentDwells = recentDwellsRef.current;
      }

      const m = (latestTelRef.current?.metrics ?? latestTelRef.current) ?? {};
      const band = m.band ?? resolved.currentBand;
      const hit = m.hit ?? false;
      const dwellUs = m.dwell_time_us ?? resolved.currentDwellUs;
      const clockUs = m.clock_us ?? resolved.missionClockUs;
      const modeName = m.mode_name ?? resolved.currentMode;
      const rawStep = latestTelRef.current?.step ?? latestTelRef.current?.metrics?.step ?? resolved.totalDwells;

      const liveAuthKey = m.id || `live-${rawStep}-${band}-${Number(clockUs).toFixed(1)}`;
      const entry = {
        id: liveAuthKey,
        band,
        freqMHz: bandToFreqMHz(band),
        mode: modeName,
        hit,
        dwellUs,
        clockUs,
        now: true,
      };

      const prev = dwellHistoryRef.current;
      const isDuplicate = prev.length > 0 && prev[prev.length - 1].band === band && Math.abs(prev[prev.length - 1].clockUs - clockUs) < 1;
      if (!isDuplicate) {
        const updated = prev.map((e) => ({ ...e, now: false }));
        updated.push(entry);
        dwellHistoryRef.current = updated.slice(-DWELL_HISTORY_SIZE);

        // Deduplicate into rolling 25-dwell window
        if (rawStep > 0 && !seenDwellKeysRef.current.has(liveAuthKey)) {
          seenDwellKeysRef.current.add(liveAuthKey);
          recentDwellsWindowRef.current.push({
            id: liveAuthKey,
            hit: Boolean(hit),
            band,
            step: rawStep,
            error_us: m.error_us != null ? Number(m.error_us) : null,
          });
        }

        const activeBandsList = [];
        if (Array.isArray(resolved.emitters) && resolved.emitters.length > 0) {
          resolved.emitters.forEach((e) => {
            const b = Math.floor((e.frequency_mhz ?? e.freq_mhz ?? 0) / 500);
            if (b >= 0 && b < 36 && !activeBandsList.includes(b)) activeBandsList.push(b);
          });
        }
        if (hit && !activeBandsList.includes(band)) activeBandsList.push(band);

        // Waterfall frame records authoritative 36-channel occupancy vector from telemetry
        const curTel = latestTelRef.current || {};
        const rawPriors = curTel.band_priorities ?? m?.band_priorities ?? curTel.bandPriorities ?? [];
        const channelOccupancy = Array(36).fill(0.0);
        if (Array.isArray(rawPriors) && rawPriors.length === 36) {
          for (let b = 0; b < 36; b++) {
            channelOccupancy[b] = Math.max(0.0, Math.min(1.0, Number(rawPriors[b]) || 0.0));
          }
        } else if (Array.isArray(resolved.bandHeights) && resolved.bandHeights.length === 36) {
          for (let b = 0; b < 36; b++) {
            channelOccupancy[b] = Math.max(0.0, Math.min(1.0, Number(resolved.bandHeights[b]) || 0.0));
          }
        }

        waterfallHistoryRef.current = [
          ...waterfallHistoryRef.current,
          {
            step: rawStep,
            timestamp: clockUs,
            tuned_band: band,
            last_band: band, // backward-compatibility alias
            active_bands: activeBandsList,
            channel_occupancy: channelOccupancy,
            band_priorities: channelOccupancy, // semantic alias
            band_powers: channelOccupancy, // backward-compatibility alias
            is_hit: Boolean(hit),
          },
        ].slice(-50);
      }
    }

    // Keep window strictly capped to latest 25 distinct completed dwells
    if (recentDwellsWindowRef.current.length > 25) {
      recentDwellsWindowRef.current = recentDwellsWindowRef.current.slice(-25);
    }

    const windowHits = recentDwellsWindowRef.current.filter((d) => d.hit).length;
    const windowTotal = recentDwellsWindowRef.current.length;
    const instantaneousPd = windowTotal > 0 ? windowHits / windowTotal : (resolved.rollingPd || 0.0);
    const sessionAvgPd = resolved.totalDwells > 0 ? resolved.totalHits / resolved.totalDwells : 0.0;

    resolved.instantaneousPd = instantaneousPd;
    resolved.sessionAvgPd = sessionAvgPd;
    if (windowTotal > 0) {
      resolved.rollingPd = instantaneousPd;
    }

    const rawStep = latestTelRef.current?.step ?? latestTelRef.current?.metrics?.step ?? resolved.totalDwells;
    const met = metricsRef.current || {};
    const rawTel = latestTelRef.current || {};
    const rawMet = rawTel.metrics || {};

    // Defect 1: Never fabricate live metric fallbacks!
    const pfaVal = met.Pfa ?? met.pfa ?? rawMet.Pfa ?? rawMet.pfa ?? rawTel.Pfa ?? rawTel.pfa ?? null;
    const irVal = met.avg_intercept_rate ?? met.intercept_rate ?? met.mean_ir ?? (resolved.totalDwells > 0 ? (resolved.totalHits / resolved.totalDwells) : null);
    const rewardVal = met.avg_reward ?? met.reward_per_dwell ?? rawMet.avg_reward ?? rawTel.avg_reward ?? null;
    const correctPredVal = met.pct_correct_predictions ?? met.correct_pct ?? rawMet.pct_correct_predictions ?? rawTel.pct_correct_predictions ?? null;

    let timeErrorVal = met.avg_intercept_time_error_us ?? met.avg_time_error ?? rawMet.avg_intercept_time_error_us ?? rawTel.avg_intercept_time_error_us ?? null;
    if (timeErrorVal === null && recentDwellsWindowRef.current.length > 0) {
      const errs = recentDwellsWindowRef.current.map((d) => d.error_us).filter((e) => e != null && !isNaN(e));
      if (errs.length > 0) {
        timeErrorVal = errs.reduce((sum, e) => sum + Math.abs(e), 0) / errs.length;
      }
    }

    resolved.liveMetrics = {
      connected: resolved.live,
      step: rawStep,
      pd: instantaneousPd,
      pfa: pfaVal !== null ? floatOr(pfaVal, null) : null,
      avg_intercept_rate: irVal !== null ? floatOr(irVal, null) : null,
      avg_reward: rewardVal !== null ? floatOr(rewardVal, null) : null,
      pct_correct_predictions: correctPredVal !== null ? floatOr(correctPredVal, null) : null,
      avg_intercept_time_error_us: timeErrorVal !== null ? floatOr(timeErrorVal, null) : null,
      last_band: resolved.currentBand != null ? `B${String(resolved.currentBand + 1).padStart(2, "0")}` : null,
      last_mode: resolved.currentMode ?? "NORMAL_DWELL",
      decision_reason: resolved.scheduler?.decisionReason || "DRQN Policy",
    };
    resolved.waterfallHistory = waterfallHistoryRef.current;

    resolved.dwellHistory = dwellHistoryRef.current;
    setState((prev) => ({
      ...resolved,
      wsStatus: prev.wsStatus === "ONLINE" || resolved.live ? "ONLINE" : prev.wsStatus,
    }));
  }, []);

  // Initialize unified HTTP poller
  useEffect(() => {
    let unmounted = false;

    const poller = createTelemetryPoller({
      intervalMs: pollingIntervalMs,
      onData(data) {
        if (!unmounted) ingest(data);
      },
      onStateChange(connState) {
        if (!unmounted) {
          connStateRef.current = connState;
          setState((prev) => ({
            ...prev,
            connectionState: connState,
            connectionLabel: CONNECTION_STATES[connState] || connState,
            wsStatus: connState === "BACKEND_UNAVAILABLE" ? "OFFLINE" : "ONLINE",
          }));
        }
      },
      onError(err) {
        if (!unmounted) {
          setState((prev) => ({
            ...prev,
            lastError: err?.message || String(err),
          }));
        }
      },
    });

    pollerRef.current = poller;

    return () => {
      unmounted = true;
      poller.close();
      pollerRef.current = null;
    };
  }, [pollingIntervalMs, ingest]);

  const updateInterval = useCallback((newMs) => {
    setPollingIntervalMs(newMs);
    pollerRef.current?.setIntervalMs(newMs);
  }, []);

  // Mission control functions
  const startStream = useCallback(async (params = {}) => {
    const res = await api.startMissionStream(params);
    pollerRef.current?.pollCycle();
    return res;
  }, []);

  const stopStream = useCallback(async () => {
    const res = await api.stopMissionStream();
    pollerRef.current?.pollCycle();
    return res;
  }, []);

  const startMission = useCallback(async (initialTimeUs = 0.0, params = {}) => {
    const res = await api.startMission(initialTimeUs, params);
    pollerRef.current?.pollCycle();
    return res;
  }, []);

  const stepMission = useCallback(async (pdws = null, obs = null) => {
    const res = await api.stepMission(pdws, obs);
    pollerRef.current?.pollCycle();
    return res;
  }, []);

  const stopMission = useCallback(async () => {
    const res = await api.stopMission();
    pollerRef.current?.pollCycle();
    return res;
  }, []);

  const resetMission = useCallback(async () => {
    const res = await api.resetMission();
    dwellHistoryRef.current = [];
    seenDwellKeysRef.current.clear();
    recentDwellsWindowRef.current = [];
    waterfallHistoryRef.current = [];
    pdwsRef.current = [];
    incidentPdwsRef.current = [];
    recentDwellsRef.current = [];
    pollerRef.current?.pollCycle();
    return res;
  }, []);

  return {
    ...state,
    pollingIntervalMs,
    setPollingInterval: updateInterval,
    startStream,
    stopStream,
    startMission,
    stepMission,
    stopMission,
    resetMission,
  };
}
