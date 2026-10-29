import type { MouseEvent } from "react";
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
}: {
  pose: Pose;
  workspace?: Workspace | null;
  live?: boolean;
  cells?: GardenCell[];
  cellMm?: number;
  layer?: MapLayer;
  onGoto?: (x: number, y: number) => void;
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

  const gotoFromEvent = (event: MouseEvent<SVGSVGElement>) => {
    if (!onGoto) return;
    const pt = svgEventPoint(event.currentTarget, event);
    if (!pt) return;
    if (pt.x < ox || pt.x > ox + plotW || pt.y < oy || pt.y > oy + plotH) return;
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
    onGoto(Math.round(gantryX), Math.round(gantryY));
  };

  const { isTablet } = useDevice();
  const hint = isTablet ? "Tap the bed to move there" : "Double-click the bed to move there";
  return (
    <div className={`gantry-map${live ? " gantry-map--live" : ""}${onGoto ? " gantry-map--goto" : ""}`}>
      <svg
        viewBox={`0 0 ${vbW} ${vbH}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Gantry map. ${hint}.`}
        {...(onGoto ? (isTablet ? { onClick: gotoFromEvent } : { onDoubleClick: gotoFromEvent }) : {})}
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
        <line x1={ox} y1={py} x2={ox + plotW} y2={py} className="gantry-beam" />
        <circle cx={px} cy={py} r={9} className="gantry-head" />
      </svg>
    </div>
  );
}
