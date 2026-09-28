export function getWsBaseUrl() {
  if (typeof window !== "undefined") {
    const isProd = window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";
    const override = localStorage.getItem("smartscan_ws_url");
    if (override && override.trim()) {
      const clean = override.trim().replace(/\/+$/, "");
      if (isProd && (clean.includes("localhost") || clean.includes("127.0.0.1"))) {
        localStorage.removeItem("smartscan_ws_url");
      } else {
        return clean;
      }
    }
    const apiOverride = localStorage.getItem("smartscan_api_url");
    if (apiOverride && apiOverride.trim()) {
      const clean = apiOverride.trim().replace(/\/+$/, "");
      if (isProd && (clean.includes("localhost") || clean.includes("127.0.0.1"))) {
        localStorage.removeItem("smartscan_api_url");
      } else {
        if (clean.startsWith("https://")) {
          return clean.replace(/^https:\/\//, "wss://");
        }
        if (clean.startsWith("http://")) {
          return clean.replace(/^http:\/\//, "ws://");
        }
      }
    }
  }

  const envWs = import.meta.env.VITE_WS_BASE_URL;
  if (envWs && envWs.trim()) {
    return envWs.trim().replace(/\/+$/, "");
  }

  const envApi = import.meta.env.VITE_API_BASE_URL;
  if (envApi && envApi.trim()) {
    const clean = envApi.trim().replace(/\/+$/, "");
    if (clean.startsWith("https://")) {
      return clean.replace(/^https:\/\//, "wss://");
    }
    if (clean.startsWith("http://")) {
      return clean.replace(/^http:\/\//, "ws://");
    }
  }

  const isLocal = typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
  return isLocal ? "ws://localhost:8080" : "wss://cognitive-ew-backend-753709137146.asia-south1.run.app";
}

const WS_BASE_URL = getWsBaseUrl();

/**
 * Backend WebSocket status:
 * The deployed FastAPI backend does not expose a documented /ws/state endpoint in OpenAPI.
 * Live data transmission is reliably handled via HTTP polling (GET /telemetry/latest, /mission/status, etc.).
 * WEBSOCKET_ENABLED is set to false to safely disable socket attempts and prevent connection errors.
 */
export const WEBSOCKET_ENABLED = false;

export function createWebSocket(path, handlers = {}) {
  if (!WEBSOCKET_ENABLED) {
    // Safely disabled - report status and return no-op interface
    if (typeof window !== "undefined" && window.__SMARTSCAN_WS_WARNED !== true) {
      window.__SMARTSCAN_WS_WARNED = true;
      console.info(
        "[SmartScan WS] WebSocket connection disabled; FastAPI backend uses HTTP polling for live state and telemetry.",
      );
    }

    // Call onClose so subscribers know socket is offline
    handlers.onClose?.({ code: 1000, reason: "WebSocket disabled by configuration", wasClean: true });

    return {
      send() {
        throw new Error("WebSocket is disabled; please use HTTP endpoints for backend requests.");
      },
      close() {},
      getSocket() {
        return null;
      },
      getState() {
        return typeof WebSocket !== "undefined" ? WebSocket.CLOSED : 3;
      },
    };
  }

  let socket = null;
  let closedByUser = false;
  let reconnectTimer = null;
  let reconnectAttempt = 0;

  const maxReconnectDelay = 5000;

  function clearReconnectTimer() {
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function scheduleReconnect() {
    if (closedByUser) return;
    clearReconnectTimer();
    const delay = Math.min(500 * 2 ** reconnectAttempt, maxReconnectDelay);
    reconnectAttempt += 1;
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  }

  function connect() {
    if (closedByUser) return;
    try {
      const base = getWsBaseUrl();
      socket = new WebSocket(`${base}${path}`);
    } catch (error) {
      handlers.onError?.(error);
      scheduleReconnect();
      return;
    }

    socket.addEventListener("open", (event) => {
      reconnectAttempt = 0;
      handlers.onOpen?.(event);
    });

    socket.addEventListener("message", (event) => {
      let payload = event.data;
      try {
        payload = JSON.parse(event.data);
      } catch {
        // Non-JSON
      }
      handlers.onMessage?.(payload);
    });

    socket.addEventListener("error", (event) => {
      handlers.onError?.(event);
    });

    socket.addEventListener("close", (event) => {
      handlers.onClose?.(event);
      if (!closedByUser) scheduleReconnect();
    });
  }

  connect();

  return {
    send(payload) {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        throw new Error("WebSocket is not connected.");
      }
      socket.send(typeof payload === "string" ? payload : JSON.stringify(payload));
    },

    close(code = 1000, reason = "Client closed") {
      closedByUser = true;
      clearReconnectTimer();
      socket?.close(code, reason);
    },

    getSocket() {
      return socket;
    },

    getState() {
      return socket ? socket.readyState : WebSocket.CLOSED;
    },
  };
}

export function connectStateSocket(handlers = {}) {
  return createWebSocket("/ws/state", handlers);
}

export function connectTelemetrySocket(handlers = {}) {
  return connectStateSocket(handlers);
}

export { WS_BASE_URL };
