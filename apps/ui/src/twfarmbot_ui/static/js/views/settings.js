import { ui, postAction, errorMessage, getSettings, saveSettings } from "../api.js";
import { h, btn, snack, page, section, card, toolbar, expander, jsonBlock } from "../ui.js";
import * as state from "../state.js";

export async function render(container) {
  const { root, body } = page("Settings");
  const config = await ui("/config");
  const apiField = h("md-outlined-text-field", {
    label: "API URL", value: (config.ok && config.body?.api_url) || "", class: "grow",
  });
  const statusBox = h("div");

  const drawStatus = () => statusBox.replaceChildren(jsonBlock({
    farmbot: state.store.health?.farmbot ?? "?",
    api: apiField.value,
    actions: state.store.health?.actions ?? [],
  }));

  body.append(
    section("Connection", toolbar(apiField,
      btn("outlined", {
        onClick: async () => {
          const r = await ui("/config", { method: "PUT", json: { api_url: apiField.value } });
          if (!r.ok) { snack(errorMessage(r), { error: true }); return; }
          await state.refreshHealth();
          snack("API URL updated");
          drawStatus();
        },
      }, "Apply"),
      btn("filled-tonal", {
        icon: "ecg_heart",
        onClick: async () => { await state.refreshHealth(); drawStatus(); snack("Health checked"); },
      }, "Health check")),
      statusBox),
    section("Auto-refresh intervals", (() => {
      const settings = getSettings();
      const field = (label, key, min, max) => h("md-outlined-text-field", {
        label, type: "number", value: String(settings[key]), min: String(min), max: String(max), class: "field-sm",
        onChange: (e) => {
          const value = parseInt(e.target.value, 10);
          if (Number.isFinite(value)) { saveSettings({ [key]: value }); snack("Saved"); }
        },
      });
      return toolbar(
        field("Position (s)", "refreshPositionS", 1, 300),
        field("Stats (s)", "refreshStatsS", 10, 3600),
        field("Camera (s, 0=off)", "refreshCameraS", 0, 3600));
    })()),
    section("Raw action", expander("Fire a raw action", (() => {
      const kindField = h("md-outlined-text-field", { label: "Kind", value: "move", class: "field-sm" });
      const paramsField = h("md-outlined-text-field", {
        label: "Params (JSON)", type: "textarea", rows: 3,
        value: '{"x": 0, "y": 0, "z": 0}', class: "grow",
      });
      const resultBox = h("div");
      return card(null, toolbar(kindField, paramsField,
        btn("filled", {
          icon: "bolt",
          onClick: async () => {
            let params;
            try { params = JSON.parse(paramsField.value); }
            catch (err) { snack(`Bad JSON: ${err.message}`, { error: true }); return; }
            const r = await postAction(kindField.value.trim(), params);
            resultBox.replaceChildren(jsonBlock(r.body));
          },
        }, "Fire")),
        resultBox);
    })())),
  );
  container.append(root);
  drawStatus();
}
