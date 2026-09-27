export function getApiBaseUrl() {
  if (typeof window !== "undefined") {
    const isProd = window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";
    const override = localStorage.getItem("smartscan_api_url");
    if (override && override.trim()) {
      const cleanOverride = override.trim().replace(/\/+$/, "");
      if (isProd && (cleanOverride.includes("localhost") || cleanOverride.includes("127.0.0.1"))) {
        localStorage.removeItem("smartscan_api_url");
      } else {
        return cleanOverride;
      }
    }
  }
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, "");
  }
  const isLocal = typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
  return isLocal ? "http://localhost:8080" : "https://smartscan-backend-q6ay.onrender.com";
}

export function setApiBaseUrl(url) {
  if (typeof window !== "undefined") {
    if (url) {
      localStorage.setItem("smartscan_api_url", url.trim().replace(/\/+$/, ""));
    } else {
      localStorage.removeItem("smartscan_api_url");
    }
  }
}

const API_BASE_URL = getApiBaseUrl();

async function request(path, options = {}) {
  const base = getApiBaseUrl();
  const apiKey = (typeof window !== "undefined" && localStorage.getItem("smartscan_api_key")) || 
    import.meta.env.VITE_SMARTSCAN_API_KEY || 
    import.meta.env.VITE_API_KEY || 
    "";
  let response;

  try {
    response = await fetch(`${base}${path}`, {
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(apiKey ? { "X-API-Key": apiKey } : {}),
        ...(options.headers || {}),
      },
      ...options,
    });
  } catch (netErr) {
    if (netErr.name === "AbortError") {
      throw netErr;
    }
    const isNetwork = netErr.message?.includes("Failed to fetch") || netErr.message?.includes("NetworkError");
    const errMsg = isNetwork
      ? `Render backend may be spinning up from cold start or unreachable (${netErr.message}).`
      : `Network request to backend failed: ${netErr.message}`;
    const err = new Error(errMsg);
    err.name = "NetworkError";
    err.isNetworkError = true;
    err.original = netErr;
    throw err;
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status} (${response.statusText || "Error"})`;

    try {
      const body = await response.json();
      if (body && typeof body === "object") {
        if (body.detail) {
          detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        } else if (body.message) {
          detail = typeof body.message === "string" ? body.message : JSON.stringify(body.message);
        } else if (body.error) {
          detail = typeof body.error === "string" ? body.error : JSON.stringify(body.error);
        }
      }
    } catch {
      // Non-JSON error response.
    }

    const httpErr = new Error(detail);
    httpErr.status = response.status;
    httpErr.statusText = response.statusText;
    throw httpErr;
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

export const api = {
  getSystemStatus(options = {}) {
    return request("/health", options);
  },

  getMetrics(options = {}) {
    return request("/metrics", options);
  },

  getLatestTelemetry(options = {}) {
    return request("/telemetry/latest", options);
  },

  getTelemetryHistory(limit = 200, options = {}) {
    const parsedLimit = Number(limit);
    const safeLimit = Math.max(
      1,
      Math.min(1000, Number.isFinite(parsedLimit) ? parsedLimit : 200),
    );

    return request(`/telemetry/history?limit=${encodeURIComponent(safeLimit)}`, options);
  },

  getTelemetryRuns(options = {}) {
    return request("/telemetry/runs", options);
  },

  getEmitterMemory(options = {}) {
    return request("/memory/emitters", options);
  },

  startMission(initialTimeUs = 0.0, params = {}, options = {}) {
    return request("/mission/start", {
      method: "POST",
      body: JSON.stringify({
        initial_time_us: Number(initialTimeUs) || 0.0,
        scenario: params.scenario ?? "final_grc",
        speed_hz: Number(params.speed_hz) || 15.0,
        max_dwells: params.max_dwells ?? 4000,
        auto_stream: true,
      }),
      ...options,
    });
  },

  stepMission(pdws = null, obs = null, options = {}) {
    return request("/mission/step", {
      method: "POST",
      body: JSON.stringify({ pdws, obs }),
      ...options,
    });
  },

  stopMission(options = {}) {
    return request("/mission/stop", {
      method: "POST",
      ...options,
    });
  },

  getMissionStatus(options = {}) {
    return request("/mission/status", options);
  },

  startMissionStream(params = {}, options = {}) {
    const payload = {
      scenario: params.scenario ?? "final_grc",
      speed_hz: Number(params.speed_hz) || 15.0,
      max_dwells: params.max_dwells ?? 4000,
    };
    return request("/mission/stream/start", {
      method: "POST",
      body: JSON.stringify(payload),
      ...options,
    });
  },

  stopMissionStream(options = {}) {
    return request("/mission/stream/stop", {
      method: "POST",
      ...options,
    });
  },

  getMissionStreamStatus(options = {}) {
    return request("/mission/stream/status", options);
  },

  resetMission(options = {}) {
    return request("/reset", {
      method: "POST",
      ...options,
    });
  },

  evaluateBenchmark(params = {}, options = {}) {
    return request("/benchmark/evaluate", {
      method: "POST",
      body: JSON.stringify(params),
      ...options,
    });
  },

  getLatestBenchmark(options = {}) {
    return request("/benchmark/latest", options);
  },

  /**
   * Predict single best time-frequency action from the trained DRQN+MoE scheduler.
   *
   * Observation contract:
   * Exactly 360 numeric values (36 frequency bands × 10 features per band).
   * Flattened layout:
   *   [band_0_feat_0, ..., band_0_feat_9, band_1_feat_0, ..., band_35_feat_9]
   *
   * @param {Array<number>} obs - Exactly 360 numeric values.
   * @param {string} [policyMode="default"] - Policy mode ('default', 'operational', 'demo', 'fallback').
   * @param {Object} [options={}] - Fetch options like signal.
   * @returns {Promise<Object>} Real model output (selected_action, selected_band, selected_mode, etc.)
   */
  predictBands(obs, policyMode = "default", options = {}) {
    if (!Array.isArray(obs)) {
      return Promise.reject(new Error("Observation must be an array of numeric values."));
    }
    if (obs.length !== 360) {
      return Promise.reject(
        new Error(
          `Invalid observation dimension: expected exactly 360 numeric values (36 bands × 10 features), got ${obs.length}.`,
        ),
      );
    }
    for (let i = 0; i < obs.length; i++) {
      const val = Number(obs[i]);
      if (!Number.isFinite(val)) {
        return Promise.reject(
          new Error(`Invalid observation value at index ${i}: must be a finite number, got ${obs[i]}.`),
        );
      }
    }
    return request("/predict_bands", {
      method: "POST",
      body: JSON.stringify({
        obs: obs.map(Number),
        policy_mode: policyMode,
      }),
      ...options,
    });
  },

  getMissionScenarios(options = {}) {
    return request("/mission/scenarios", options);
  },

  getBenchmarkScenarios(options = {}) {
    return request("/benchmark/scenarios", options);
  },

  getGnuRfScenarios(options = {}) {
    return request("/gnu_rf/scenarios", options);
  },
};

export { API_BASE_URL };
