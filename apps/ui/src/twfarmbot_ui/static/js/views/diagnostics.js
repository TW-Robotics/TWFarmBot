import { errorMessage } from "../api.js";
import { h, icon, snack, page, section, card, toolbar, metricRow, dataTable, emptyState } from "../ui.js";
import * as state from "../state.js";

export async function render(container) {
  const { root, body } = page("Diagnostics");
  const contentBox = h("div");

  body.append(
    toolbar(h("md-filled-tonal-button", {
      onClick: async () => {
        const r = await state.refreshStatus();
        if (!r.ok) snack(`Read failed: ${errorMessage(r)}`, { error: true });
        draw();
      },
    }, icon("troubleshoot"), "Load /status")),
    contentBox,
  );
  container.append(root);

  function draw() {
    const payload = state.store.diag || {};
    const info = payload.informational_settings || {};
    const axes = payload.location_data?.axis_states || {};
    const pins = payload.pins || {};

    if (!Object.keys(info).length && !Object.keys(axes).length && !Object.keys(pins).length) {
      contentBox.replaceChildren(emptyState("Click “Load /status” to fetch diagnostic state.", { iconName: "troubleshoot", compact: true }));
      return;
    }

    contentBox.replaceChildren(
      metricRow([
        ["Controller", info.controller_version ?? "—"],
        ["Firmware", info.firmware_version ?? "—"],
        ["Wi-Fi", `${info.wifi_level_percent ?? "—"}%`],
        ["Uptime", `${info.uptime ?? "—"} s`],
      ]),
      h("div", { class: "card-grid" },
        card("Resources",
          h("p", { class: "caption" }, `CPU ${info.cpu_usage ?? "—"}%`),
          h("p", { class: "caption" }, `Memory ${info.memory_usage ?? "—"}% · Disk ${info.disk_usage ?? "—"}%`),
          h("p", { class: "caption" }, `SoC ${info.soc_temp ?? "—"} °C`)),
        card("Axis state",
          h("p", { class: "caption" }, `X ${axes.x ?? "—"} · Y ${axes.y ?? "—"} · Z ${axes.z ?? "—"}`),
          h("p", { class: "caption" }, `Busy: ${info.busy ?? "—"}`)),
        card("Network",
          h("p", { class: "caption" }, `${info.wifi_level ?? "—"} dBm`),
          h("p", { class: "caption" }, `${info.private_ip ?? "—"}`),
          h("p", { class: "caption" }, `Sync: ${info.sync_status ?? "—"}`)),
      ),
      Object.keys(pins).length
        ? section("Pin snapshot", dataTable(["Pin", "Value", "Mode"],
          Object.entries(pins).map(([pin, data]) => [pin, data?.value, data?.mode])))
        : null,
    );
  }

  draw();
  return state.on("status", draw);
}
