/**
 * Configuration module for SmartScan Frontend.
 * Automatically resolves API and WebSocket URLs from environment variables
 * or defaults for seamless deployment on Vercel connecting to Cloud Run.
 */

export function getApiBaseUrl() {
  if (typeof window !== "undefined") {
    const override = localStorage.getItem("smartscan_api_url");
    if (override && override.trim()) {
      return override.trim().replace(/\/+$/, "");
    }
  }

  // 1. Environment variables set at build time (e.g. VITE_API_URL or VITE_API_BASE_URL)
  const envUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, "");
  }

  // 2. Default fallback: localhost in dev, current origin in production
  if (typeof window !== "undefined") {
    const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    return isLocal ? "http://localhost:8000" : "https://cognitive-ew-backend-753709137146.asia-south1.run.app";
  }

  return "http://localhost:8000";
}

export function getWsBaseUrl() {
  // 1. Explicit WebSocket URL from env
  const envWs = import.meta.env.VITE_WS_URL;
  if (envWs && envWs.trim()) {
    return envWs.trim().replace(/\/+$/, "");
  }

  // 2. Derive from API URL if set
  const apiUrl = getApiBaseUrl();
  if (apiUrl.startsWith("http://")) {
    return apiUrl.replace("http://", "ws://");
  } else if (apiUrl.startsWith("https://")) {
    return apiUrl.replace("https://", "wss://");
  }

  // 3. Fallback based on current location
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    return isLocal ? "ws://localhost:8000" : `${proto}//${window.location.host}`;
  }

  return "ws://localhost:8000";
}

export const API_BASE_URL = getApiBaseUrl();
export const WS_BASE_URL = getWsBaseUrl();
