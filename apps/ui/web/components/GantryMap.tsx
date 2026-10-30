import { useState, type MouseEvent, type PointerEvent } from "react";
import { useDevice } from "../device";

export type Pose = { x?: number; y?: number; z?: number };

export type AxisLimit = { min?: number | null; max?: number | null };

export type Workspace = {
  x?: AxisLimit;
  y?: AxisLimit;
  z?: AxisLimit;
};

const DEFAULT_X = { min: 0, max: 1900 };
const DEFAULT_Y = { min: 0, max: 650 };

function num(value: unknown): number | undefined {
  if (value == null || value === "") return undefined;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function span(axis: AxisLimit | undefined, fallback: { min: number; max: number }) {
  const min = num(axis?.min) ?? fallback.min;
  const max = num(axis?.max) ?? Math.max(fallback.max, min + 1);
  return { min, max, size: Math.max(max - min, 1) };
}

export function poseFromUnknown(value: unknown): Pose | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const xyz = raw.xyz && typeof raw.xyz === "object" ? (raw.xyz as Record<string, unknown>) : raw;
  const x = num(xyz.x);
  const y = num(xyz.y);
  const z = num(xyz.z);
  if (x === undefined && y === undefined && z === undefined) return null;
  return { x, y, z };
}

export function fmt(value: number | undefined): string {
  return value === undefined ? "—" : Math.round(value).toString();
}

export type GardenCell = {
  x?: number;
  y?: number;
  z?: number;
  ndre_mean?: number | null;
  stress_fraction?: number | null;
  plant?: number | null;
  weed?: number | null;
  soil?: number | null;
  dominant?: string | null;
  action_hint?: string | null;
};

export type Rect = { x1: number; y1: number; x2: number; y2: number };
export type Area = Rect & { id?: string; label: string; pending?: boolean };

export type MapLayer = "vigor" | "cover";

export type LegendEntry = { label: string; color: string };

/** One colour table per layer; the sidebar legend renders from the same entries. */
export const LAYER_LEGEND: Record<MapLayer, Record<string, LegendEntry>> = {
  vigor: {
    healthy: { label: "Healthy", color: "var(--bed-healthy)" },
    monitor: { label: "Monitor", color: "var(--bed-monitor)" },
    water: { label: "Needs water", color: "var(--bed-water)" },
    unclear: { label: "Unclear", color: "var(--bed-unclear)" },
  },
  cover: {
    plant: { label: "Plant", color: "var(--bed-healthy)" },
    weed: { label: "Weed", color: "var(--bed-weed)" },
    soil: { label: "Soil", color: "var(--bed-soil)" },
  },
};

function vigorClass(cell: GardenCell): string | null {
  const hint = String(cell.action_hint || "").toLowerCase();
  if (hint === "ok") return "healthy";
  if (hint === "monitor") return "monitor";
  if (hint === "consider_water" || hint === "recheck") return "water";
  if (hint === "reposition") return "unclear";
  const n = num(cell.ndre_mean);
  const s = num(cell.stress_fraction);
  if (n !== undefined) {
    if (n >= 0.25) return "healthy";
    if (n >= 0.15) return "monitor";
    if (n >= 0.05) return "water";
    return "unclear";
  }
  if (s !== undefined) {
    if (s >= 0.4) return "water";
    if (s >= 0.2) return "monitor";
    return "healthy";
  }
  return null;
}

function coverClass(cell: GardenCell): string | null {
  const dominant = String(cell.dominant || "").toLowerCase();
  if (dominant.includes("weed")) return "weed";
  if (dominant.includes("plant")) return "plant";
  if (dominant.includes("soil")) return "soil";
  const plant = num(cell.plant) ?? 0;
  const weed = num(cell.weed) ?? 0;
  const soil = num(cell.soil) ?? 0;
  const best = Math.max(plant, weed, soil);
  if (best <= 0) return null;
  if (best === weed) return "weed";
  if (best === plant) return "plant";
  return "soil";
}

/** Number of cells per legend key for the given layer. */
export function layerCounts(cells: GardenCell[], layer: MapLayer): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const cell of cells) {
    const key = cellClass(cell, layer);
    if (key) counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

export function cellClass(cell: GardenCell, layer: MapLayer): string | null {
  return layer === "cover" ? coverClass(cell) : vigorClass(cell);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function svgEventPoint(svg: SVGSVGElement, event: MouseEvent): { x: number; y: number } | null {
  const ctm = svg.getScreenCTM();
  if (!ctm) return null;
  const pt = svg.createSVGPoint();
  pt.x = event.clientX;
  pt.y = event.clientY;
  const mapped = pt.matrixTransform(ctm.inverse());
  return { x: mapped.x, y: mapped.y };
}

export function GantryMap({
  pose,
  workspace,
  live = false,
  cells = [],
  cellMm = 250,
  layer = "vigor",
  onGoto,
  areas = [],
  onArea,
  onAreaClick,
}: {
  pose: Pose;
  workspace?: Workspace | null;
  live?: boolean;
  cells?: GardenCell[];
  cellMm?: number;
  layer?: MapLayer;
  onGoto?: (x: number, y: number) => void;
  /** Planted areas (gantry mm), drawn as labelled rectangles. */
  areas?: Area[];
  /** Drag on the bed to outline an area; a plain click gives a small square. */
  onArea?: (area: Rect) => void;
  /** A click on an existing area (one with an `id`) selects it instead of drawing. */
  onAreaClick?: (id: string) => void;
}) {
  const xBox = span(workspace?.x, DEFAULT_X);
  const yBox = span(workspace?.y, DEFAULT_Y);
  // The viewBox matches the bed's aspect, so the SVG scales the bed to fill the panel.
  const pad = 12;
  const plotW = 300;
  const plotH = plotW * (xBox.size / yBox.size);
  const vbW = plotW + pad * 2;
  const vbH = plotH + pad * 2;
  const ox = pad;
  const oy = pad;
  const x = Math.min(xBox.max, Math.max(xBox.min, num(pose.x) ?? xBox.min));
  const y = Math.min(yBox.max, Math.max(yBox.min, num(pose.y) ?? yBox.min));
  const px = ox + ((y - yBox.min) / yBox.size) * plotW;
  const py = oy + ((x - xBox.min) / xBox.size) * plotH;
  const cellPx = Math.min((cellMm / yBox.size) * plotW, (cellMm / xBox.size) * plotH);

  const bedPoint = (event: MouseEvent<SVGSVGElement>) => {
    const pt = svgEventPoint(event.currentTarget, event);
    if (!pt) return null;
    if (pt.x < ox || pt.x > ox + plotW || pt.y < oy || pt.y > oy + plotH) return null;
    const gantryX = clamp(
      xBox.min + ((pt.y - oy) / plotH) * xBox.size,
      xBox.min,
      xBox.max,
    );
    const gantryY = clamp(
      yBox.min + ((pt.x - ox) / plotW) * yBox.size,
      yBox.min,
      yBox.max,
    );
    return { x: Math.round(gantryX), y: Math.round(gantryY) };
  };
  const gotoFromEvent = (event: MouseEvent<SVGSVGElement>) => {
    const spot = onGoto && bedPoint(event);
    if (spot) onGoto(spot.x, spot.y);
  };
  const [drag, setDrag] = useState<{ a: { x: number; y: number }; b: { x: number; y: number } } | null>(null);
  const rectOf = (a: { x: number; y: number }, b: { x: number; y: number }): Rect => ({
    x1: Math.min(a.x, b.x),
    y1: Math.min(a.y, b.y),
    x2: Math.max(a.x, b.x),
    y2: Math.max(a.y, b.y),
  });
  const dragEvents = onArea
    ? {
        onPointerDown: (event: PointerEvent<SVGSVGElement>) => {
          const pt = bedPoint(event);
          if (!pt) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          setDrag({ a: pt, b: pt });
        },
        onPointerMove: (event: PointerEvent<SVGSVGElement>) => {
          const pt = bedPoint(event);
          if (drag && pt) setDrag({ ...drag, b: pt });
        },
        onPointerUp: () => {
          if (!drag) return;
          const r = rectOf(drag.a, drag.b);
          setDrag(null);
          const isClick = r.x2 - r.x1 < 50 && r.y2 - r.y1 < 50;
          const hit = isClick
            ? [...areas].reverse().find((a) => a.id && r.x1 >= a.x1 && r.x1 <= a.x2 && r.y1 >= a.y1 && r.y1 <= a.y2)
            : undefined;
          if (hit?.id && onAreaClick) return onAreaClick(hit.id);
          // A click (no real drag) outlines a small square around the spot.
          if (r.x2 - r.x1 < 50 && r.y2 - r.y1 < 50) {
            const cx = (r.x1 + r.x2) / 2;
            const cy = (r.y1 + r.y2) / 2;
            onArea({ x1: cx - 50, y1: cy - 50, x2: cx + 50, y2: cy + 50 });
          } else {
            onArea(r);
          }
        },
      }
    : {};
  const toSvgX = (gy: number) => ox + ((clamp(gy, yBox.min, yBox.max) - yBox.min) / yBox.size) * plotW;
  const toSvgY = (gx: number) => oy + ((clamp(gx, xBox.min, xBox.max) - xBox.min) / xBox.size) * plotH;
  const shownAreas: Area[] = drag ? [...areas, { ...rectOf(drag.a, drag.b), label: "", pending: true }] : areas;

  const { isTablet } = useDevice();
  const hint = isTablet ? "Tap the bed to move there" : "Double-click the bed to move there";
  return (
    <div className={`gantry-map${live ? " gantry-map--live" : ""}${onGoto ? " gantry-map--goto" : ""}${onArea ? " gantry-map--draw" : ""}`}>
      <svg
        viewBox={`0 0 ${vbW} ${vbH}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Gantry map. ${hint}.`}
        {...(onArea
          ? dragEvents
          : onGoto
            ? isTablet
              ? { onClick: gotoFromEvent }
              : { onDoubleClick: gotoFromEvent }
            : {})}
      >
        <title>{onGoto ? `${hint} (Z stays put)` : "Bed map"}</title>
        <rect x={ox} y={oy} width={plotW} height={plotH} rx={14} className="gantry-bed" />
        {cells.map((cell, index) => {
          const cx = num(cell.x);
          const cy = num(cell.y);
          if (cx === undefined || cy === undefined) return null;
          const key = cellClass(cell, layer);
          const entry = key ? LAYER_LEGEND[layer][key] : undefined;
          if (!entry) return null;
          return (
            <circle
              key={`cell-${index}-${cx}-${cy}`}
              cx={ox + ((clamp(cy, yBox.min, yBox.max) - yBox.min) / yBox.size) * plotW}
              cy={oy + ((clamp(cx, xBox.min, xBox.max) - xBox.min) / xBox.size) * plotH}
              r={Math.max(5, cellPx * 0.22)}
              className="gantry-cell"
              fill={entry.color}
            >
              <title>{`${entry.label} · x ${Math.round(cx)} y ${Math.round(cy)}`}</title>
            </circle>
          );
        })}
        {shownAreas.map((r, index) => (
          <g key={`area-${index}-${r.x1}-${r.y1}`}>
            <rect
              x={toSvgX(r.y1)}
              y={toSvgY(r.x1)}
              width={toSvgX(r.y2) - toSvgX(r.y1)}
              height={toSvgY(r.x2) - toSvgY(r.x1)}
              rx={4}
              className={`gantry-area${r.pending ? " gantry-area--pending" : ""}`}
            />
            <text
              x={(toSvgX(r.y1) + toSvgX(r.y2)) / 2}
              y={(toSvgY(r.x1) + toSvgY(r.x2)) / 2}
              className="gantry-area-label"
              textAnchor="middle"
              dy="0.35em"
            >
              {r.label}
            </text>
            <title>{r.label}</title>
          </g>
        ))}
        <line x1={ox} y1={py} x2={ox + plotW} y2={py} className="gantry-beam" />
        <circle cx={px} cy={py} r={9} className="gantry-head" />
      </svg>
    </div>
  );
}
