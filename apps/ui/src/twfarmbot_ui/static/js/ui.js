// Tiny DOM helpers + shared layout primitives for consistent page structure.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "html") el.innerHTML = value;
    else if (key in el && key !== "style" && typeof value !== "string") el[key] = value;
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const icon = (name) => h("md-icon", {}, name);

/**
 * Material button with a correctly slotted icon.
 * variant: "filled" | "filled-tonal" | "outlined" | "text"
 */
export function btn(variant, { icon: iconName, ...attrs } = {}, ...children) {
  const el = h(`md-${variant}-button`, attrs);
  if (iconName) el.append(h("md-icon", { slot: "icon" }, iconName));
  el.append(...children.flat().filter((c) => c !== null && c !== undefined)
    .map((c) => (c.nodeType ? c : document.createTextNode(String(c)))));
  return el;
}

/** Icon-only button (md-icon-button uses the default slot). */
export function iconBtn(iconName, attrs = {}) {
  return h("md-icon-button", attrs, icon(iconName));
}

export function snack(message, { error = false, ms = 3500 } = {}) {
  const host = document.getElementById("snackbar-host");
  const bar = h("div", { class: `snackbar${error ? " error" : ""}` }, message);
  host.append(bar);
  setTimeout(() => bar.remove(), ms);
}

// ── Layout primitives ───────────────────────────────────────────────────

/** Standard page shell: header + body. Pass `actions` for title-row buttons. */
export function page(title, eyebrow, { actions, bodyClass = "" } = {}) {
  const body = h("div", { class: `page-body${bodyClass ? ` ${bodyClass}` : ""}` });
  const header = h("header", { class: "page-header" },
    h("p", { class: "eyebrow" }, eyebrow ?? "TWFarmBot · UAS Technikum Wien"),
    h("div", { class: "page-title-row" },
      h("h1", { class: "page-title" }, title),
      actions ? h("div", { class: "page-actions" }, ...(Array.isArray(actions) ? actions : [actions])) : null));
  return { root: h("div", { class: "page" }, header, body), body };
}

/** Section with a consistent title and vertical rhythm. */
export function section(title, ...children) {
  return h("section", { class: "section" },
    title ? h("h2", { class: "section-title" }, title) : null,
    ...children);
}

/** Elevated surface card; optional title rendered as card header. */
export function card(title, ...children) {
  if (!title) {
    return h("div", { class: "card" }, h("div", { class: "card-body" }, ...children));
  }
  return h("div", { class: "card" },
    h("div", { class: "card-header" }, h("h3", { class: "card-title" }, title)),
    h("div", { class: "card-body" }, ...children));
}

/** Horizontal button / control row. */
export function toolbar(...children) {
  return h("div", { class: "toolbar" }, ...children);
}

/** Responsive metric grid (auto-fits 2–5 columns). */
export function metricRow(pairs) {
  return h("div", { class: "metric-grid" },
    pairs.map(([label, value]) => metric(label, value)));
}

export function metric(label, value) {
  return h("div", { class: "metric" },
    h("div", { class: "metric-label" }, label),
    h("div", { class: "metric-value" }, value ?? "—"));
}

/** Two-column layout for map + sidebar, camera + controls, etc. */
export function split(main, side, { ratio = "2fr 1fr" } = {}) {
  return h("div", { class: "split", style: `--split-ratio:${ratio}` }, main, side);
}

export function stack(...children) {
  return h("div", { class: "stack" }, ...children);
}

export function emptyState(message, { iconName = "info", compact = false } = {}) {
  return h("div", { class: compact ? "empty-panel" : "empty-state" }, icon(iconName), h("p", {}, message));
}

export function expander(label, ...children) {
  return h("details", { class: "expander" }, h("summary", {}, label), h("div", { class: "expander-body" }, ...children));
}

export function jsonBlock(data) {
  return h("pre", { class: "codeblock" }, JSON.stringify(data, null, 2));
}

export function dataTable(headers, rows) {
  return h("table", { class: "data-table" },
    h("thead", {}, h("tr", {}, headers.map((head) => h("th", {}, head)))),
    h("tbody", {}, rows.map((row) => h("tr", {}, row.map((cell) => h("td", {}, cell ?? "—"))))));
}

// ── Formatting helpers ────────────────────────────────────────────────

export const num = (value) => {
  const f = parseFloat(value);
  return Number.isFinite(f) ? f.toFixed(1) : "—";
};

export const flt = (value, fallback = 0) => {
  const f = parseFloat(value);
  return Number.isFinite(f) ? f : fallback;
};

export function parseNumber(value) {
  if (typeof value === "number") return value;
  const text = String(value).trim();
  if (!/^-?\d+(?:[.,]\d+)?$/.test(text)) return null;
  return parseFloat(text.replace(",", "."));
}

export function timeAgo(tsMs) {
  if (!tsMs) return "never";
  const delta = (Date.now() - tsMs) / 1000;
  if (delta < 2) return "just now";
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}min ago`;
  return `${(delta / 3600).toFixed(1)}h ago`;
}

const ACTION_ICONS = {
  move: "➡️", move_path: "🧭", water: "💧", find_home: "🏠", take_photo: "📷",
  read_pin: "📖", write_pin: "✏️", mount_tool: "🔧", dismount_tool: "🔧", e_stop: "🛑",
};

export function actionSummary(action) {
  const kind = action.kind || "action";
  const params = action.params || {};
  const prefix = `${ACTION_ICONS[kind] || "🛠️"} ${kind}`;
  switch (kind) {
    case "move": return `${prefix} → (${num(params.x)}, ${num(params.y)}, ${num(params.z)})`;
    case "move_path": return `${prefix} · ${(params.waypoints || []).length} waypoint(s)`;
    case "water": return `${prefix} · ${params.seconds ?? "—"}s`;
    case "find_home": return `${prefix} (axis=${params.axis || "all"})`;
    case "read_pin": return `${prefix} ${params.pin ?? "—"} (${params.mode || "digital"})`;
    case "write_pin": return `${prefix} ${params.pin ?? "—"} = ${params.value ?? "—"}`;
    case "mount_tool": return `${prefix} ${params.tool_name ?? "—"}`;
    default: return prefix;
  }
}
