import "@material/web/all.js";
import { styles as typescaleStyles } from "@material/web/typography/md-typescale-styles.js";

import { postAction, errorMessage } from "./api.js";
import { h, icon, snack, timeAgo, num } from "./ui.js";
import * as state from "./state.js";

import * as overview from "./views/overview.js";
import * as garden from "./views/garden.js";
import * as motion from "./views/motion.js";
import * as camera from "./views/camera.js";
import * as io from "./views/io.js";
import * as assistant from "./views/assistant.js";
import * as historyView from "./views/history.js";
import * as diagnostics from "./views/diagnostics.js";
import * as settings from "./views/settings.js";

document.adoptedStyleSheets.push(typescaleStyles.styleSheet);

const TABS = [
  { key: "overview", label: "Overview", icon: "monitoring", view: overview },
  { key: "garden", label: "Garden", icon: "psychiatry", view: garden },
  { key: "motion", label: "Motion", icon: "open_with", view: motion },
  { key: "camera", label: "Camera", icon: "photo_camera", view: camera },
  { key: "io", label: "I/O", icon: "settings_input_component", view: io },
  { key: "assistant", label: "Assistant", icon: "smart_toy", view: assistant },
  { key: "history", label: "History", icon: "history", view: historyView },
  { key: "diagnostics", label: "Diagnostics", icon: "troubleshoot", view: diagnostics },
  { key: "settings", label: "Settings", icon: "settings", view: settings },
];

// Legacy routes from the old Sensors / Operations tabs now live under I/O.
const LEGACY_TABS = { sensors: "io", operations: "io", "i/o": "io" };

let activeTab = null;
let teardown = null;

function tabFromUrl() {
  const raw = (new URLSearchParams(location.search).get("tab") || "").toLowerCase();
  const key = LEGACY_TABS[raw] || raw;
  return TABS.some((tab) => tab.key === key) ? key : TABS[0].key;
}

async function showTab(key) {
  if (key === activeTab) return;
  if (typeof teardown === "function") teardown();
  activeTab = key;
  const params = new URLSearchParams(location.search);
  params.set("tab", key);
  history.replaceState(null, "", `?${params}`);

  for (const item of document.querySelectorAll(".nav-item")) {
    item.classList.toggle("active", item.dataset.tab === key);
  }
  document.body.classList.toggle("has-chat-bar", key === "assistant");
  const content = document.getElementById("content");
  content.replaceChildren();
  const tab = TABS.find((t) => t.key === key);
  teardown = await tab.view.render(content);
}

function buildNav() {
  const nav = document.getElementById("nav");
  nav.append(h("div", { class: "nav-label" }, "Navigation"));
  for (const tab of TABS) {
    nav.append(h("div", {
      class: "nav-item",
      "data-tab": tab.key,
      onClick: () => showTab(tab.key),
    }, icon(tab.icon), tab.label));
  }
}

function bindSidebar() {
  const pill = document.getElementById("farmbot-pill");
  const livePos = document.getElementById("live-position");
  const posAge = document.getElementById("position-age");

  state.on("health", (health) => {
    const fb = String(health?.farmbot ?? "unknown");
    const css = fb === "connected" ? "ok" : fb.startsWith("failed") ? "bad" : "warn";
    pill.className = `pill ${css}`;
    pill.textContent = `● ${fb}`;
  });
  state.on("position", (pos) => {
    livePos.textContent = `X ${num(pos?.x)} · Y ${num(pos?.y)} · Z ${num(pos?.z)}`;
  });
  setInterval(() => {
    posAge.textContent = state.store.lastPositionRefresh
      ? `updated ${timeAgo(state.store.lastPositionRefresh)}` : "";
  }, 1000);

  const refreshAll = () => {
    state.refreshPosition();
    state.refreshHealth();
    state.refreshMessages();
  };
  document.getElementById("refresh-btn").addEventListener("click", refreshAll);
  document.getElementById("estop-btn").addEventListener("click", async () => {
    const r = await postAction("e_stop");
    if (r.ok) snack("🛑 ESTOP sent");
    else snack(errorMessage(r), { error: true });
  });
}

async function boot() {
  buildNav();
  bindSidebar();
  await state.initSession();
  state.refreshHealth();
  state.refreshPosition();
  state.refreshMessages();
  state.startPolling();
  showTab(tabFromUrl());
  window.addEventListener("popstate", () => showTab(tabFromUrl()));
}

boot();
