import { api, ui, errorMessage } from "../api.js";
import { h, btn, snack, page, section, card, toolbar, metricRow, split, stack, num, dataTable } from "../ui.js";
import * as state from "../state.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const GRID_STEP = 25;
const PALETTE = ["#554fd8", "#2685c7", "#d35d7b", "#8b5ed7", "#e57a44", "#00a59b", "#b54fc8"];
const KINDS = ["plant", "obstacle", "tool", "marker", "sensor", "valve", "custom"];

const kindColor = (kind, offset = 0) => {
  let hash = offset;
  for (const ch of String(kind)) hash = (hash * 31 + ch.charCodeAt(0)) % 997;
  return PALETTE[hash % PALETTE.length];
};

function svgEl(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  el.append(...children);
  return el;
}

export async function render(container) {
  const { root, body } = page("Garden map", "Spatial model · configured world state");
  const metricsBox = h("div");
  const mapBox = h("div");
  const sideBox = h("div", { class: "stack" });
  let selected = [];

  body.append(
    toolbar(btn("filled-tonal", { icon: "refresh", onClick: () => loadWorld(true) }, "Refresh map")),
    metricsBox,
    split(mapBox, sideBox),
  );
  container.append(root);

  async function loadWorld(force = false) {
    let world = force ? null : state.store.session?.garden_world;
    if (!world) {
      const r = await api("/garden");
      if (!r.ok || typeof r.body !== "object") {
        mapBox.replaceChildren(card(null, h("p", { class: "caption" },
          `Garden model unavailable: ${errorMessage(r)}`)));
        return;
      }
      world = r.body;
      state.store.session.garden_world = world;
      state.persistSession();
    }
    selected = [];
    draw(world);
  }

  function draw(world) {
    const bounds = world.bounds || {};
    const camera = world.camera || {};
    const robot = world.robot || {};
    const entities = world.entities || [];
    const zones = world.zones || [];
    const [x0, y0] = [bounds.x || 0, bounds.y || 0];
    const [width, height] = [bounds.width || 1, bounds.height || 1];
    const pad = Math.max(width, height) * 0.03;
    const flipY = (y) => y0 + height - (y - y0);

    metricsBox.replaceChildren(metricRow([
      ["Garden X", `${num(width)} mm`], ["Garden Y", `${num(height)} mm`],
      ["Known objects", entities.length], ["Mapped zones", zones.length],
    ]));

    const svg = svgEl("svg", {
      class: "garden-map",
      viewBox: `${x0 - pad} ${y0 - pad} ${width + 2 * pad} ${height + 2 * pad}`,
    });
    svg.append(svgEl("rect", {
      x: x0, y: y0, width, height,
      fill: "none", stroke: "var(--md-sys-color-outline)", "stroke-width": pad * 0.2,
    }));
    for (const zone of zones) {
      const zb = zone.bounds || {};
      const color = kindColor(zone.kind);
      svg.append(svgEl("rect", {
        x: zb.x, y: flipY(zb.y + zb.height), width: zb.width, height: zb.height,
        fill: color, "fill-opacity": 0.15, stroke: color, "stroke-width": pad * 0.12, rx: 6,
      }, svgEl("title", {}, `${zone.name} (${zone.kind})`)));
    }
    const marker = (x, y, r, color, label) => svgEl("circle", {
      cx: x, cy: flipY(y), r, fill: color, stroke: "white", "stroke-width": r * 0.15,
    }, svgEl("title", {}, label));
    for (const entity of entities) {
      const p = entity.position || {};
      svg.append(marker(p.x, p.y, Math.max(entity.radius_mm || 20, 12), kindColor(entity.kind, 3),
        `${entity.name} (${entity.kind})`));
    }
    svg.append(marker(robot.x || 0, robot.y || 0, 20, "#554fd8", "FarmBot"));
    const camPos = camera.position || {};
    svg.append(marker(camPos.x || 0, camPos.y || 0, 14, "#d35d7b", "Camera"));

    const selectionLayer = svgEl("g", {});
    svg.append(selectionLayer);
    svg.addEventListener("click", (event) => {
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
      const gx = Math.round(point.x / GRID_STEP) * GRID_STEP;
      const gy = Math.round((y0 + height - (point.y - y0)) / GRID_STEP) * GRID_STEP;
      if (gx < x0 || gx > x0 + width || gy < y0 || gy > y0 + height) return;
      const idx = selected.findIndex(([sx, sy]) => sx === gx && sy === gy);
      if (idx >= 0) selected.splice(idx, 1);
      else selected.push([gx, gy]);
      drawSelection();
    });

    function drawSelection() {
      selectionLayer.replaceChildren(...selected.map(([sx, sy]) => svgEl("circle", {
        cx: sx, cy: flipY(sy), r: 12, fill: "none", stroke: "#e53935", "stroke-width": 3,
      })));
      drawSide();
    }

    mapBox.replaceChildren(svg,
      h("div", { class: "garden-legend" },
        ...zones.map((z) => h("span", {},
          h("span", { class: "swatch", style: `background:${kindColor(z.kind)}` }), z.name)),
        h("span", {}, h("span", { class: "swatch", style: "background:#554fd8" }), "FarmBot"),
        h("span", {}, h("span", { class: "swatch", style: "background:#d35d7b" }), "Camera"),
        h("span", { class: "caption" }, `Click to select ${GRID_STEP} mm grid points`)));

    function drawSide() {
      const parts = [];
      if (selected.length) {
        let kind = "plant";
        const customField = h("md-outlined-text-field", { label: "Custom kind", class: "grow", style: "display:none" });
        const chips = h("md-chip-set", { class: "chip-row" }, KINDS.map((k) =>
          h("md-filter-chip", {
            label: k, selected: k === kind,
            onClick: (e) => {
              e.preventDefault();
              kind = k;
              for (const chip of chips.querySelectorAll("md-filter-chip")) chip.selected = chip.label === k;
              customField.style.display = k === "custom" ? "" : "none";
            },
          })));
        const nameField = h("md-outlined-text-field", { label: "Name prefix", placeholder: "e.g. Tomato", class: "grow" });
        parts.push(card(`${selected.length} point(s) selected`, stack(
          chips, customField, nameField,
          toolbar(
            h("md-filled-button", {
              onClick: async () => {
                const name = nameField.value.trim();
                if (!name) { snack("Please enter a name prefix.", { error: true }); return; }
                const finalKind = kind === "custom" && customField.value.trim() ? customField.value.trim() : kind;
                for (const [i, [px, py]] of selected.entries()) {
                  const res = await ui("/garden/entities", {
                    method: "POST", json: { x: px, y: py, kind: finalKind, name: `${name}-${i + 1}` },
                  });
                  if (!res.ok) { snack(errorMessage(res), { error: true }); return; }
                }
                snack(`Added ${selected.length} ${finalKind}(s)`);
                loadWorld(true);
              },
            }, `Assign ${selected.length}`),
            h("md-outlined-button", { onClick: () => { selected = []; drawSelection(); } }, "Clear")))));
      }
      parts.push(
        section("Live pose", metricRow([["X", num(robot.x)], ["Y", num(robot.y)], ["Z", num(robot.z)]])),
        section("Camera pose",
          h("p", { class: "caption" }, `X ${num(camPos.x)} · Y ${num(camPos.y)} · Z ${num(camPos.z)} mm`),
          h("p", { class: "caption" },
            `Yaw ${num(camera.yaw_deg)}° · Pitch ${num(camera.pitch_deg)}° · Roll ${num(camera.roll_deg)}°`)),
        section("Mapped objects", entities.length
          ? dataTable(["Name", "Kind"], entities.map((e) => [e.name, e.kind]))
          : h("p", { class: "caption" }, "No objects mapped yet.")));
      sideBox.replaceChildren(...parts);
    }
    drawSelection();
  }

  await loadWorld();
  const off = state.on("position", (pos) => {
    const world = state.store.session?.garden_world;
    if (!world || !pos) return;
    world.robot = { ...pos };
    const offset = world.camera_offset || {};
    world.camera = { ...(world.camera || {}), position: {
      x: (pos.x || 0) + (offset.x || 0), y: (pos.y || 0) + (offset.y || 0), z: (pos.z || 0) + (offset.z || 0),
    }};
    if (!selected.length) draw(world);
  });
  return off;
}
