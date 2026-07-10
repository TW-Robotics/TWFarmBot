import Chart from "chart.js/auto";
import { h, btn, page, section, card, toolbar, metricRow, num, timeAgo } from "../ui.js";
import * as state from "../state.js";

const EXPERIMENT_KEY = "twfb_experiment";

function chartConfig(datasets, { yMax } = {}) {
  return {
    type: "line",
    data: { labels: state.store.telemetry.map((s) => s.time), datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, max: yMax, grid: { color: "rgba(128,128,128,0.1)" } },
      },
      plugins: { legend: { labels: { boxWidth: 12, padding: 16 } } },
    },
  };
}

function series(key, label, color) {
  return {
    label,
    data: state.store.telemetry.map((s) => s[key]),
    borderColor: color,
    backgroundColor: color,
    tension: 0.3,
    pointRadius: 2,
    borderWidth: 2,
  };
}

export async function render(container) {
  const { root, body } = page("Research overview");
  const posBox = h("div");
  const statusBox = h("div");
  const resourceBox = h("div");
  const chartsArea = h("div");
  const networkBox = h("div", { class: "stack" });
  const eventsBox = h("div");
  const charts = [];
  let renderedTelemetryLast = null;
  let renderedTelemetryLength = -1;

  const refreshBtn = btn("filled-tonal", {
    icon: "refresh",
    onClick: async () => {
      await Promise.all([state.refreshHealth(), state.refreshPosition(), state.refreshStatus()]);
      update();
    },
  }, "Refresh status");
  const clearBtn = btn("outlined", {
    onClick: () => { state.store.telemetry = []; update(); },
  }, "Clear history");

  const experiment = JSON.parse(localStorage.getItem(EXPERIMENT_KEY) || "{}");
  const saveExp = (key, value) => {
    experiment[key] = value;
    localStorage.setItem(EXPERIMENT_KEY, JSON.stringify(experiment));
  };

  body.append(
    section(null, posBox),
    section("System status", toolbar(refreshBtn, clearBtn), statusBox),
    section("Resources over time", resourceBox, chartsArea),
    h("div", { class: "split" },
      card("Network & hardware", networkBox),
      card("Recent events", eventsBox)),
    section("Experiment", h("div", { class: "form-grid" },
      h("md-outlined-text-field", { label: "Run", value: experiment.run || "", class: "grow",
        onInput: (e) => saveExp("run", e.target.value) }),
      h("md-outlined-text-field", { label: "Operator", value: experiment.operator || "", class: "grow",
        onInput: (e) => saveExp("operator", e.target.value) }),
      h("md-outlined-text-field", { label: "Notes", type: "textarea", rows: 2, class: "grow",
        value: experiment.notes || "", onInput: (e) => saveExp("notes", e.target.value) }))),
  );
  container.append(root);

  function update() {
    const pos = state.store.position || {};
    const info = state.store.diag.informational_settings || {};
    const loc = state.store.diag.location_data || {};
    const axes = loc.axis_states || {};

    posBox.replaceChildren(
      metricRow([["X · mm", num(pos.x)], ["Y · mm", num(pos.y)], ["Z · mm", num(pos.z)]]),
      h("p", { class: "caption" }, `Position updated ${timeAgo(state.store.lastPositionRefresh)}`));

    statusBox.replaceChildren(metricRow([
      ["FarmBot", state.store.health?.farmbot ?? "—"],
      ["Uptime", `${num(info.uptime)} s`],
      ["Wi-Fi", `${num(info.wifi_level_percent)}%`],
      ["Sync", info.sync_status ?? "—"],
      ["Busy", info.busy ? "Yes" : "No"],
    ]));

    resourceBox.replaceChildren(metricRow([
      ["CPU", info.cpu_usage != null ? `${info.cpu_usage}%` : "—"],
      ["Memory", info.memory_usage != null ? `${info.memory_usage}%` : "—"],
      ["Disk", info.disk_usage != null ? `${info.disk_usage}%` : "—"],
      ["SoC temp", info.soc_temp != null ? `${info.soc_temp}°C` : "—"],
    ]));

    networkBox.replaceChildren(
      h("p", { class: "caption" }, "Private IP: ", h("span", { class: "mono" }, info.private_ip ?? "—")),
      h("p", { class: "caption" }, `Wi-Fi signal: ${num(info.wifi_level)} dBm`),
      h("p", { class: "caption" }, `Controller: ${info.controller_version ?? "—"}`),
      h("p", { class: "caption" }, `Firmware: ${info.firmware_version ?? "—"}`),
      h("p", { class: "caption" },
        `Axis states · X ${axes.x ?? "—"} · Y ${axes.y ?? "—"} · Z ${axes.z ?? "—"}`));

    const messages = state.store.messages;
    eventsBox.replaceChildren(messages.length
      ? h("pre", { class: "codeblock" }, messages.slice(-10).join("\n"))
      : h("p", { class: "caption" }, "No events recorded."));

    const telemetry = state.store.telemetry;
    const telemetryLast = telemetry.at(-1);
    if (telemetry.length === renderedTelemetryLength && telemetryLast === renderedTelemetryLast) {
      return;
    }
    renderedTelemetryLength = telemetry.length;
    renderedTelemetryLast = telemetryLast;

    charts.forEach((c) => c.destroy());
    charts.length = 0;
    if (telemetry.length) {
      const usageBox = h("div", { class: "chart-box tall" }, h("canvas"));
      const wifiBox = h("div", { class: "chart-box" }, h("canvas"));
      const socBox = h("div", { class: "chart-box" }, h("canvas"));
      chartsArea.replaceChildren(usageBox, h("div", { class: "split", style: "--split-ratio:1fr 1fr" }, wifiBox, socBox));
      charts.push(
        new Chart(usageBox.firstChild, chartConfig(
          [series("cpu", "CPU", "#554fd8"), series("memory", "Memory", "#2685c7"), series("disk", "Disk", "#d35d7b")],
          { yMax: 100 })),
        new Chart(wifiBox.firstChild, chartConfig([series("wifi", "Wi-Fi %", "#2685c7")], { yMax: 100 })),
        new Chart(socBox.firstChild, chartConfig([series("soc", "SoC °C", "#ba1a1a")])));
    } else {
      chartsArea.replaceChildren(h("p", { class: "caption" },
        "Click “Refresh status” to start collecting telemetry for the charts."));
    }
  }

  update();
  const offs = [state.on("position", update), state.on("status", update),
    state.on("health", update), state.on("messages", update)];
  return () => { offs.forEach((off) => off()); charts.forEach((c) => c.destroy()); };
}
