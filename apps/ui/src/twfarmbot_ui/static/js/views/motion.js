import { api, postAction, errorMessage } from "../api.js";
import { h, btn, snack, page, section, card, toolbar, metricRow, num, flt, parseNumber } from "../ui.js";
import * as state from "../state.js";

async function doMove(x, y, z, label = "") {
  const r = await postAction("move", { x, y, z });
  if (r.ok) { snack(label ? `→ ${label}` : `→ (${x.toFixed(0)}, ${y.toFixed(0)}, ${z.toFixed(0)})`); state.refreshPosition(); }
  else snack(errorMessage(r), { error: true });
}

export async function render(container) {
  const { root, body } = page("Motion workspace");
  const posBox = h("div");
  const presetsBox = h("div", { class: "toolbar" });
  let step = 10;

  const cur = () => ({
    x: flt(state.store.position?.x), y: flt(state.store.position?.y), z: flt(state.store.position?.z),
  });

  const stepChips = h("md-chip-set", { class: "chip-row" }, [1, 10, 50, 100].map((value) =>
    h("md-filter-chip", {
      label: `${value} mm`, selected: value === step,
      onClick: (e) => {
        e.preventDefault();
        step = value;
        for (const chip of stepChips.querySelectorAll("md-filter-chip")) chip.selected = chip.label === `${value} mm`;
      },
    })));

  const jog = (dx, dy, dz, label) => () => {
    const { x, y, z } = cur();
    doMove(x + dx * step, y + dy * step, z + dz * step, `${label}${step}`);
  };

  // XY pad (3×3) plus a separate labelled Z column so nothing floats.
  const padBtn = (iconName, onClick, { filled = false, label } = {}) =>
    h(filled ? "md-filled-icon-button" : "md-filled-tonal-icon-button",
      { onClick, "aria-label": label, title: label },
      h("md-icon", {}, iconName));

  const xyPad = h("div", { class: "dpad" },
    h("span"), padBtn("arrow_upward", jog(0, 1, 0, "Y+"), { label: "Y+" }), h("span"),
    padBtn("arrow_back", jog(-1, 0, 0, "X-"), { label: "X−" }),
    padBtn("home", () => doMove(0, 0, 0, "Home"), { filled: true, label: "Home" }),
    padBtn("arrow_forward", jog(1, 0, 0, "X+"), { label: "X+" }),
    h("span"), padBtn("arrow_downward", jog(0, -1, 0, "Y-"), { label: "Y−" }), h("span"));

  const zCol = h("div", { class: "z-col" },
    padBtn("keyboard_arrow_up", jog(0, 0, 1, "Z+"), { label: "Z+" }),
    h("span", { class: "z-label" }, "Z"),
    padBtn("keyboard_arrow_down", jog(0, 0, -1, "Z-"), { label: "Z−" }));

  const fx = h("md-outlined-text-field", { label: "X", class: "field-sm" });
  const fy = h("md-outlined-text-field", { label: "Y", class: "field-sm" });
  const fz = h("md-outlined-text-field", { label: "Z", class: "field-sm" });
  const syncFields = () => {
    const { x, y, z } = cur();
    fx.value = x.toFixed(2); fy.value = y.toFixed(2); fz.value = z.toFixed(2);
  };

  body.append(
    section(null, posBox),
    h("div", { class: "split", style: "--split-ratio:1fr 1fr" },
      card("Jog controls",
        stepChips,
        h("div", { class: "jog-area" }, xyPad, zCol)),
      card("Absolute move",
        toolbar(fx, fy, fz),
        toolbar(
          btn("filled", {
            icon: "my_location",
            onClick: () => {
              const [x, y, z] = [parseNumber(fx.value), parseNumber(fy.value), parseNumber(fz.value)];
              if (x === null || y === null || z === null) {
                snack("Invalid coordinates. Use a number like 123 or 123.4.", { error: true });
                return;
              }
              doMove(x, y, z);
            },
          }, "Go to"),
          btn("outlined", {
            icon: "home_work",
            onClick: async () => {
              const r = await postAction("find_home");
              if (r.ok) snack("Homing queued");
              else snack(errorMessage(r), { error: true });
            },
          }, "Find home")))),
    section("Preset locations", presetsBox),
  );
  container.append(root);
  syncFields();

  function updatePos() {
    const pos = state.store.position || {};
    posBox.replaceChildren(metricRow([["X · mm", num(pos.x)], ["Y · mm", num(pos.y)], ["Z · mm", num(pos.z)]]));
    syncFields();
  }
  updatePos();

  const r = await api("/positions");
  const presets = (r.ok && r.body?.positions) || [];
  presetsBox.replaceChildren(...(presets.length
    ? presets.map((p) => btn("filled-tonal", {
      icon: "place",
      onClick: () => doMove(flt(p.x), flt(p.y), flt(p.z), p.label),
    }, p.label || "?"))
    : [h("p", { class: "caption" }, "No preset locations configured.")]));

  return state.on("position", updatePos);
}
