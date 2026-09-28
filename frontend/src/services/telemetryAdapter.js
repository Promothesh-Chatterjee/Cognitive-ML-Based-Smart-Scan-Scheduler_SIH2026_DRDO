function isObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

export function adaptTelemetryPayload(
  payload,
) {
  if (
    payload === null ||
    payload === undefined
  ) {
    return {
      valid: false,
      source: "none",
      live: false,
      message: "No telemetry payload",
      schemaVersion: null,
      step: null,
      episode: null,
      type: null,
      raw: null,
    };
  }

  if (!isObject(payload)) {
    return {
      valid: false,
      source: "unknown",
      live: false,
      message:
        "Telemetry payload is not an object",
      schemaVersion: null,
      step: null,
      episode: null,
      type: null,
      raw: payload,
    };
  }

  return {
    valid: true,

    source:
      typeof payload.source === "string"
        ? payload.source
        : "backend",

    live:
      payload.live === true,

    message:
      typeof payload.message === "string"
        ? payload.message
        : typeof payload.live_message ===
            "string"
          ? payload.live_message
          : null,

    schemaVersion:
      payload.telemetry_schema_version ??
      null,

    step:
      payload.step ??
      payload.metrics?.step ??
      null,

    episode:
      payload.episode ??
      payload.metrics?.episode ??
      null,

    type:
      payload.type ??
      null,

    metrics: payload.metrics || {},
    bandPriorities: payload.bandPriorities || payload.metrics?.band_priorities || [],
    pdws: payload.pdws || payload.metrics?.pdws || [],
    allIncidentPdws: payload.all_incident_pdws || payload.allIncidentPdws || payload.metrics?.all_incident_pdws || [],
    recentDwells: payload.recent_dwells || payload.recentDwells || payload.metrics?.recent_dwells || [],
    emitters: payload.emitters || payload.metrics?.emitters || [],
    band: payload.band ?? payload.metrics?.band ?? null,
    mode: payload.mode ?? payload.metrics?.mode ?? null,
    modeName: payload.mode_name ?? payload.metrics?.mode_name ?? null,
    hit: payload.hit ?? payload.metrics?.hit ?? false,
    dwellTimeUs: payload.dwell_time_us ?? payload.dwellTimeUs ?? payload.metrics?.dwell_time_us ?? null,
    rollingPd: payload.metrics?.system_metrics?.rolling_pd ?? payload.rolling_pd ?? null,
    rollingMedianLatencyUs: payload.metrics?.system_metrics?.rolling_median_latency_us ?? payload.rolling_median_latency_us ?? null,
    cognitiveExplanation: payload.metrics?.cognitive_explanation || {},
    systemMetrics: payload.metrics?.system_metrics || {},
    clockUs: payload.metrics?.clock_us ?? payload.clock_us ?? 0,
    observation: payload.observation || payload.metrics?.observation || payload.raw?.observation || null,

    raw: payload,
  };
}