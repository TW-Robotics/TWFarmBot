import "./material.js";
import { styles as typescaleStyles } from "@material/web/typography/md-typescale-styles.js";

import { postAction, errorMessage } from "./api.js";
import { h, icon, snack, timeAgo, num } from "./ui.js";
import * as state from "./state.js";

document.adoptedStyleSheets.push(typescaleStyles.styleSheet);

const TABS = [
  { key: "overview", label: "Overview", icon: "monitoring" },
  { key: "garden", label: "Garden", icon: "psychiatry" },
  { key: "motion", label: "Motion", icon: "open_with" },
  { key: "camera", label: "Camera", icon: "photo_camera" },
  { key: "io", label: "I/O", icon: "settings_input_component" },
  { key: "assistant", label: "Assistant", icon: "smart_toy" },
  { key: "history", label: "History", icon: "history" },
  { key: "diagnostics", label: "Diagnostics", icon: "troubleshoot" },
  { key: "settings", label: "Settings", icon: "settings" },
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
  const view = await import(`./views/${tab.key}.js`);
  // Ignore a slow module import if the user already selected another tab.
  if (activeTab !== key) return;
  teardown = await view.render(content);
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
  const updatePositionAge = () => {
    posAge.textContent = state.store.lastPositionRefresh
      ? `updated ${timeAgo(state.store.lastPositionRefresh)}` : "";
  };

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
  return updatePositionAge;
}

async function boot() {
  buildNav();
  bindSidebarToggle();
  const updatePositionAge = bindSidebar();
  await state.initSession();
  state.refreshHealth();
  state.refreshPosition();
  state.refreshMessages();
  state.startPolling(updatePositionAge);
  showTab(tabFromUrl());
  window.addEventListener("popstate", () => showTab(tabFromUrl()));
}

const SIDEBAR_COLLAPSED_KEY = "twfb_sidebar_collapsed";

function setSidebarCollapsed(collapsed) {
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  const icon = document.querySelector("#sidebar-toggle md-icon");
  if (icon) icon.textContent = collapsed ? "left_panel_open" : "left_panel_close";
  const btn = document.getElementById("sidebar-toggle");
  if (btn) {
    const label = collapsed ? "Show navigation rail" : "Hide navigation rail";
    btn.setAttribute("aria-label", label);
    btn.setAttribute("title", label);
  }
}

function bindSidebarToggle() {
  const btn = document.getElementById("sidebar-toggle");
  if (!btn) return;
  setSidebarCollapsed(localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1");
  btn.addEventListener("click", () => {
    const next = !document.body.classList.contains("sidebar-collapsed");
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, next ? "1" : "0");
    setSidebarCollapsed(next);
  });
}

boot();
