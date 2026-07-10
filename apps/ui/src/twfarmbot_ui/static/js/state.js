// Shared app state: health, position, telemetry, and the persisted session
// (chat history, camera gallery, garden world) stored via /ui/sessions.

import { api, ui, getSettings } from "./api.js";

export const store = {
  health: null,          // {status, actions, farmbot}
  position: null,        // {x, y, z}
  lastPositionRefresh: 0,
  messages: [],
  diag: {},              // last GET /status state
  lastStatsRefresh: 0,
  telemetry: [],         // ring buffer of {time, cpu, memory, disk, wifi, soc}
  session: null,         // persisted snapshot
};

const listeners = new Map();

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}

export function emit(event, payload) {
  for (const fn of listeners.get(event) || []) fn(payload);
}

// ── Refresh helpers ─────────────────────────────────────────────────────

export async function refreshPosition() {
  const r = await api("/position");
  if (r.ok && r.body?.xyz) {
    store.position = r.body.xyz;
    store.lastPositionRefresh = Date.now();
    emit("position", store.position);
  }
  return r;
}

export async function refreshHealth() {
  const r = await api("/health");
  if (r.ok && typeof r.body === "object") {
    store.health = r.body;
    emit("health", store.health);
  }
  return r;
}

export async function refreshMessages() {
  const r = await api("/messages");
  if (r.ok && Array.isArray(r.body?.last_messages)) {
    store.messages = r.body.last_messages.slice(-20).map(String);
    emit("messages", store.messages);
  }
  return r;
}

export async function refreshStatus() {
  const r = await api("/status");
  if (r.ok && typeof r.body === "object") {
    store.diag = r.body.state || {};
    store.lastStatsRefresh = Date.now();
    const info = store.diag.informational_settings || {};
    store.telemetry.push({
      time: new Date().toLocaleTimeString(),
      cpu: parseFloat(info.cpu_usage) || 0,
      memory: parseFloat(info.memory_usage) || 0,
      disk: parseFloat(info.disk_usage) || 0,
      wifi: parseFloat(info.wifi_level_percent) || 0,
      soc: parseFloat(info.soc_temp) || 0,
    });
    store.telemetry = store.telemetry.slice(-60);
    emit("status", store.diag);
  }
  return r;
}

// ── Session persistence (server-side JSON files) ───────────────────────

function blankSession() {
  const now = new Date().toISOString();
  const stamp = now.slice(0, 19).replaceAll(":", "-");
  const suffix = Math.random().toString(16).slice(2, 10);
  return {
    session_id: `${stamp}-${suffix}`,
    label: null,
    created_at: now,
    updated_at: now,
    assistant_messages: [],
    assistant_selected_model: null,
    assistant_metrics: {},
    executed_plans: [],
    camera_images: [],
    garden_world: null,
  };
}

export async function initSession() {
  const requested = new URLSearchParams(location.search).get("session");
  if (requested) {
    const r = await ui(`/sessions/${encodeURIComponent(requested)}`);
    if (r.ok) {
      store.session = { ...blankSession(), ...r.body };
      return;
    }
  }
  store.session = blankSession();
}

// A blank unsaved session isn't persisted until it has content.
function sessionHasContent() {
  const s = store.session;
  return s && (s.assistant_messages.length || s.camera_images.length ||
    s.garden_world || s.executed_plans.length || s.label);
}

export async function persistSession() {
  if (!sessionHasContent()) return;
  const s = store.session;
  await ui(`/sessions/${encodeURIComponent(s.session_id)}`, { method: "PUT", json: s });
}

export function loadSessionSnapshot(snapshot) {
  store.session = { ...blankSession(), ...snapshot };
  const params = new URLSearchParams(location.search);
  params.set("session", snapshot.session_id);
  history.replaceState(null, "", `?${params}`);
}

export function newSession() {
  store.session = blankSession();
  const params = new URLSearchParams(location.search);
  params.delete("session");
  history.replaceState(null, "", params.size ? `?${params}` : location.pathname);
}

// ── Background polling ──────────────────────────────────────────────────

export function startPolling(onTick) {
  setInterval(() => {
    const settings = getSettings();
    const now = Date.now();
    onTick?.();
    if (now - store.lastPositionRefresh >= settings.refreshPositionS * 1000) {
      refreshPosition();
    }
    if (now - store.lastStatsRefresh >= settings.refreshStatsS * 1000) {
      refreshHealth();
      refreshStatus();
    }
  }, 1000);
}
