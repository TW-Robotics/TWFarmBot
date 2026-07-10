import { api, postAction, errorMessage } from "../api.js";
import { h, btn, snack, page, section, card, toolbar, stack, split, emptyState } from "../ui.js";

async function writePin(pin, value, mode, seconds) {
  const params = { pin, value, mode };
  if (seconds !== undefined) params.seconds = seconds;
  const r = await postAction("write_pin", params);
  if (r.ok) snack(`pin ${pin} = ${value}`);
  else snack(errorMessage(r), { error: true });
}

export async function render(container) {
  const { root, body } = page("I/O workspace");
  const r = await api("/pins");
  const pins = (r.ok && r.body?.pins) || [];
  const sensors = pins.filter((p) => p.kind === "sensor");
  const outputs = pins.filter((p) => p.kind !== "sensor");

  const sensorGrid = h("div", { class: "metric-grid" });
  body.append(
    section("Sensors", sensors.length ? sensorGrid : emptyState("No sensor pins configured.", { iconName: "sensors", compact: true })),
    section("Actuators", split(
      card("Irrigation", (() => {
        const secondsField = h("md-outlined-text-field", {
          label: "Duration (seconds)", type: "number", value: "2", min: "0.1", max: "300", step: "0.5", class: "field-sm",
        });
        return stack(
          toolbar(secondsField, btn("filled", {
            icon: "water_drop",
            onClick: async () => {
              const seconds = parseFloat(secondsField.value);
              if (!Number.isFinite(seconds) || seconds <= 0) { snack("Enter a positive duration.", { error: true }); return; }
              const res = await postAction("water", { seconds });
              if (res.ok) snack("Watering queued");
              else snack(errorMessage(res), { error: true });
            },
          }, "Water")),
          h("p", { class: "caption" }, "Runs the pump for the selected duration."));
      })()),
      card("Peripheral control", (() => {
        const peripheralBody = h("div", { class: "stack" });
        if (!outputs.length) return emptyState("No output pins configured.", { iconName: "electrical_services", compact: true });
        const select = h("md-outlined-select", { label: "Output", class: "grow" },
          outputs.map((p, i) => h("md-select-option", { value: String(i), selected: i === 0 },
            h("div", { slot: "headline" }, `${p.label} · pin ${p.pin}`))));
        select.addEventListener("change", () => drawControls(outputs[Number(select.value)]));
        function drawControls(sel) {
          const mode = sel.mode || "digital";
          const parts = [h("span", { class: "pill" }, mode)];
          if (mode === "analog") {
            const presets = sel.presets || {};
            if (Object.keys(presets).length) {
              parts.push(toolbar(...Object.entries(presets)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([value, label]) => h("md-filled-tonal-button", {
                  onClick: () => writePin(sel.pin, Number(value), mode),
                }, `${label} (${value})`))));
            }
            const slider = h("md-slider", { min: 0, max: 255, value: 0, labeled: true });
            parts.push(toolbar(slider, h("md-outlined-button", {
              onClick: () => writePin(sel.pin, Number(slider.value), mode),
            }, "Apply")));
          } else {
            const pulseSwitch = h("md-switch", { selected: true });
            const pulseField = h("md-outlined-text-field", {
              label: "Pulse (seconds)", type: "number", value: "2", min: "0.1", max: "300", step: "0.5", class: "field-sm",
            });
            pulseSwitch.addEventListener("change", () => { pulseField.style.display = pulseSwitch.selected ? "" : "none"; });
            parts.push(
              h("label", { class: "toolbar" }, pulseSwitch, h("span", {}, "Timed pulse")),
              pulseField,
              toolbar(
                btn("outlined", { icon: "power_settings_new", onClick: () => writePin(sel.pin, 0, mode) }, "OFF"),
                btn("filled", {
                  icon: "power",
                  onClick: () => {
                    if (pulseSwitch.selected) {
                      const seconds = parseFloat(pulseField.value);
                      if (!Number.isFinite(seconds) || seconds <= 0) { snack("Enter a positive pulse duration.", { error: true }); return; }
                      writePin(sel.pin, 1, mode, seconds);
                    } else writePin(sel.pin, 1, mode);
                  },
                }, "ON")));
          }
          peripheralBody.replaceChildren(...parts);
        }
        drawControls(outputs[0]);
        return stack(select, peripheralBody);
      })()),
    )),
  );
  container.append(root);

  sensorGrid.replaceChildren(...sensors.map((sensor) => {
    const valueBox = h("div", { class: "metric-value", style: "font-size:20px" }, "—");
    return card(sensor.label, stack(
      h("div", { class: "toolbar" },
        h("span", { class: "pill" }, sensor.mode || "analog"),
        h("span", { class: "caption" }, `pin ${sensor.pin}`)),
      valueBox,
      btn("outlined", {
        icon: "sensors",
        onClick: async () => {
          const res = await api(`/pin/${sensor.pin}`, { params: { mode: sensor.mode || "analog" } });
          valueBox.textContent = res.ok ? String(res.body?.value ?? "—") : "—";
          if (!res.ok) snack(errorMessage(res), { error: true });
        },
      }, "Read")));
  }));
}
