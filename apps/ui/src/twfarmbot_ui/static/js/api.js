// Thin fetch wrapper mirroring the old Python ApiClient semantics:
// results are {ok, code, body}, POST /actions defaults to wait=false,
// and /chat/stream is consumed as Server-Sent Events.

async function parseBody(resp) {
  const text = await resp.text();
  try { return JSON.parse(text); } catch { return text; }
}

export function errorMessage(result) {
  const body = result.body;
  if (body && typeof body === "object") {
    if (body.detail !== undefined) return String(body.detail);
    if (body.error !== undefined) return String(body.error);
  }
  if (typeof body === "string") return body;
  return JSON.stringify(body);
}

export async function request(url, { method = "GET", json, params, timeoutMs = 10000 } = {}) {
  const qs = params ? `?${new URLSearchParams(params)}` : "";
  const opts = { method, signal: AbortSignal.timeout(timeoutMs) };
  if (json !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(json);
  }
  try {
    const resp = await fetch(`${url}${qs}`, opts);
    return { ok: resp.ok, code: resp.status, body: await parseBody(resp) };
  } catch (err) {
    return { ok: false, code: 0, body: { error: `${err.name}: ${err.message}` } };
  }
}

export const api = (path, opts) => request(`/api${path}`, opts);
export const resireg = (path, opts) => request(`/resireg${path}`, { timeoutMs: 120000, ...opts });
export const ui = (path, opts) => request(`/ui${path}`, opts);

// The FarmBot executes actions on a single worker queue; like the old UI we
// dispatch fire-and-forget by default and only wait for approved proposals.
export function postAction(kind, params = {}, { wait = false } = {}) {
  return api("/actions", {
    method: "POST",
    json: { kind, params },
    params: { wait: String(wait) },
    timeoutMs: wait ? 120000 : 10000,
  });
}

// POST-based SSE stream; yields parsed `data:` payloads.
export async function* sse(path, json, { timeoutMs = 90000 } = {}) {
  const resp = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(json),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${errorMessage({ body: await parseBody(resp) })}`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop();
    for (const chunk of events) {
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) yield JSON.parse(line.slice(6));
      }
    }
  }
}

// ── Local UI settings (auto-refresh intervals etc.) ─────────────────────

const SETTINGS_KEY = "twfb_ui_settings";
const DEFAULTS = { refreshPositionS: 5, refreshStatsS: 300, refreshCameraS: 0 };

export function getSettings() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; }
  catch { return { ...DEFAULTS }; }
}

export function saveSettings(patch) {
  const next = { ...getSettings(), ...patch };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
  return next;
}
