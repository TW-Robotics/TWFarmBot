import { useState, type ReactNode } from "react";
import { ClickableCard } from "@astryxdesign/core/ClickableCard";
import { Dialog, DialogHeader } from "@astryxdesign/core/Dialog";
import { Heading, Text } from "@astryxdesign/core/Text";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { Spinner } from "@astryxdesign/core/Spinner";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type SeriesPoint = { t: number; value: number | null; reference?: number | null } & {
  /** Extra series (e.g. regression fits) keyed by `Overlay.key`. */
  [overlay: string]: number | null | undefined;
};

export type Overlay = { key: string; label: string; color: string };

export type ChartSpec = {
  title: string;
  unit: string;
  points: SeriesPoint[];
  /** "time" plots a clock time (hours of day) per day instead of a value. */
  kind?: "value" | "time";
  /** Bars suit per-hour totals like rain; lines suit continuous readings. */
  mark?: "line" | "bar";
  /** Label for the optional dashed `reference` series (same unit and axis). */
  referenceLabel?: string;
  /** Series colour, a theme token like "var(--color-icon-blue)". */
  color?: string;
  /** X-axis tick granularity; defaults to hourly. */
  axis?: "hour" | "day" | "month";
};

export type ChartRange = "day" | "month" | "year";

const RANGE_SUBTITLE: Record<ChartRange, string> = {
  day: "Hourly · yesterday to tomorrow",
  month: "Daily · last 31 days and 2-day forecast",
  year: "Daily · last 12 months",
};

const AXIS = "var(--color-text-secondary)";
const GRID = "var(--color-border)";
const LINE = "var(--color-accent)";

function hourLabel(t: number): string {
  return new Date(t * 1000).toLocaleString([], { weekday: "short", hour: "2-digit" });
}

function dayLabel(t: number): string {
  return new Date(t * 1000).toLocaleDateString([], { weekday: "short", day: "numeric" });
}

function monthLabel(t: number): string {
  return new Date(t * 1000).toLocaleDateString([], { month: "short", year: "2-digit" });
}

const AXIS_LABEL = { hour: hourLabel, day: dayLabel, month: monthLabel };

function clockFromHours(hours: number): string {
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function formatValue(spec: ChartSpec, value: number): string {
  if (spec.kind === "time") return clockFromHours(value);
  return `${value.toFixed(1)}${spec.unit ? ` ${spec.unit}` : ""}`;
}

function currentValue(spec: ChartSpec): number | null {
  const now = Date.now() / 1000;
  let best: SeriesPoint | undefined;
  for (const point of spec.points) {
    if (point.t <= now && point.value != null) best = point;
  }
  return best?.value ?? null;
}

export function MetricLineChart({
  spec,
  height,
  isCompact,
  overlays,
  baseOpacity = 1,
}: {
  spec: ChartSpec;
  height: number | string;
  isCompact?: boolean;
  overlays?: Overlay[];
  /** Opacity of the underlying series, so overlays can stand out. */
  baseOpacity?: number;
}) {
  const isTime = spec.kind === "time";
  const color = spec.color ?? LINE;
  const xLabel = AXIS_LABEL[spec.axis ?? (isTime ? "day" : "hour")];
  const tooltipLabel = spec.axis && spec.axis !== "hour" ? dayLabel : xLabel;
  const now = Date.now() / 1000;

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <ComposedChart data={spec.points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeOpacity={0.5} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={xLabel}
            stroke={AXIS}
            tickLine={false}
            fontSize={12}
            minTickGap={isCompact ? 60 : 40}
          />
          <YAxis
            stroke={AXIS}
            tickLine={false}
            axisLine={false}
            fontSize={12}
            width={isCompact ? 40 : 56}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) => (isTime ? clockFromHours(v) : String(Number(v.toFixed(2))))}
          />
          <Tooltip
            labelFormatter={(t) => tooltipLabel(Number(t))}
            formatter={(v, name) => [
              formatValue(spec, Number(v)),
              name === "reference"
                ? spec.referenceLabel ?? "Reference"
                : overlays?.find((o) => o.key === name)?.label ?? spec.title,
            ]}
            contentStyle={{
              background: "var(--color-background-popover)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              color: "var(--color-text-primary)",
            }}
            cursor={spec.mark === "bar" ? { fill: GRID, fillOpacity: 0.3 } : { stroke: AXIS, strokeDasharray: "3 3" }}
          />
          {!isTime && spec.axis !== "month" ? (
            <ReferenceLine
              x={now}
              stroke={AXIS}
              strokeDasharray="4 4"
              label={isCompact ? undefined : { value: "now", fill: AXIS, fontSize: 12, position: "top" }}
            />
          ) : null}
          {spec.points.some((p) => p.reference != null) ? (
            <Line
              dataKey="reference"
              type="monotone"
              stroke={AXIS}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
          ) : null}
          {spec.mark === "bar" ? (
            <Bar dataKey="value" fill={color} fillOpacity={baseOpacity} radius={[2, 2, 0, 0]} isAnimationActive={false} />
          ) : (
            <Line
              dataKey="value"
              type="monotone"
              stroke={color}
              strokeOpacity={baseOpacity}
              strokeWidth={2}
              dot={isTime ? { r: 4, fill: color, stroke: color } : false}
              activeDot={{ r: 5, fill: color, stroke: "var(--color-background-card)", strokeWidth: 2 }}
              connectNulls
              isAnimationActive={false}
            />
          )}
          {overlays?.map((o) => (
            <Line
              key={o.key}
              dataKey={o.key}
              type="monotone"
              stroke={o.color}
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrendCard({ spec, onOpen }: { spec: ChartSpec; onOpen: () => void }) {
  const current = currentValue(spec);
  return (
    <div className="trend-cell">
      <ClickableCard label={`${spec.title} history`} padding={4} onClick={onOpen}>
        <div className="trend-body">
          <HStack gap={2} vAlign="end">
            <div style={{ flex: 1 }}>
              <HStack gap={2} vAlign="center">
                <span className="trend-swatch" style={{ background: spec.color ?? LINE }} />
                <Text weight="semibold">{spec.title}</Text>
              </HStack>
            </div>
            <Heading level={3}>{current != null ? formatValue(spec, current) : "—"}</Heading>
          </HStack>
          <div className="trend-chart">
            <MetricLineChart spec={spec} height="100%" isCompact />
          </div>
        </div>
      </ClickableCard>
    </div>
  );
}

export function MetricChartDialog({
  title,
  spec,
  range,
  note,
  subtitle,
  content,
  onRangeChange,
  onClose,
}: {
  title: string | null;
  spec: ChartSpec | null;
  range: ChartRange;
  note?: string;
  subtitle?: string;
  /** Custom chart body (e.g. the multi-axis pose chart); replaces `spec` rendering. */
  content?: ReactNode;
  onRangeChange: (range: ChartRange) => void;
  onClose: () => void;
}) {
  const hasData = spec?.points.some((p) => p.value != null);
  return (
    <Dialog isOpen={title !== null} onOpenChange={(open) => !open && onClose()} width={860} purpose="info">
      {title ? (
        <VStack gap={3}>
          <DialogHeader
            title={title}
            subtitle={subtitle ?? RANGE_SUBTITLE[range]}
            onOpenChange={(open) => !open && onClose()}
          />
          <SegmentedControl
            label="Time range"
            value={range}
            onChange={(value) => onRangeChange(value as ChartRange)}
          >
            <SegmentedControlItem value="day" label="Day" />
            <SegmentedControlItem value="month" label="Month" />
            <SegmentedControlItem value="year" label="Year" />
          </SegmentedControl>
          {content !== undefined ? (
            content
          ) : !spec ? (
            <div style={{ height: 360, display: "grid", placeItems: "center" }}>
              <Spinner />
            </div>
          ) : hasData ? (
            <MetricLineChart spec={spec} height={360} />
          ) : (
            <div style={{ height: 360, display: "grid", placeItems: "center" }}>
              <Text color="secondary">No data for this range.</Text>
            </div>
          )}
          {spec?.referenceLabel || note ? (
            <Text type="supporting" color="secondary">
              {[spec?.referenceLabel ? `Dashed line: ${spec.referenceLabel}` : null, note]
                .filter(Boolean)
                .join(" · ")}
            </Text>
          ) : null}
        </VStack>
      ) : null}
    </Dialog>
  );
}

export type PoseSample = { t: number; x: number; y: number; z: number };

type PoseAxis = "x" | "y" | "z";

const POSE_AXES: { key: PoseAxis; label: string; color: string }[] = [
  { key: "x", label: "X", color: "var(--chart-blue)" },
  { key: "y", label: "Y", color: "var(--chart-orange)" },
  { key: "z", label: "Z", color: "var(--chart-aqua)" },
];

/** X/Y/Z on one mm axis; the gantry holds position between moves, so lines step. */
export function PoseChart({
  samples,
  height,
  axis = "hour",
  isCompact,
}: {
  samples: PoseSample[] | null;
  height: number;
  axis?: ChartSpec["axis"];
  isCompact?: boolean;
}) {
  const [hidden, setHidden] = useState<Set<PoseAxis>>(new Set());
  const toggle = (key: PoseAxis) =>
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  if (!samples) {
    return (
      <div style={{ height, display: "grid", placeItems: "center" }}>
        <Spinner />
      </div>
    );
  }
  if (samples.length < 2) {
    return (
      <div style={{ height, display: "grid", placeItems: "center" }}>
        <Text color="secondary">No moves recorded in this range yet.</Text>
      </div>
    );
  }
  const xLabel = AXIS_LABEL[axis ?? "hour"];
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <ComposedChart data={samples} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeOpacity={0.5} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={xLabel}
            stroke={AXIS}
            tickLine={false}
            fontSize={12}
            minTickGap={isCompact ? 60 : 40}
          />
          <YAxis
            stroke={AXIS}
            tickLine={false}
            axisLine={false}
            fontSize={12}
            width={isCompact ? 44 : 56}
            tickFormatter={(v: number) => String(Math.round(v))}
          />
          <Tooltip
            labelFormatter={(t) =>
              new Date(Number(t) * 1000).toLocaleString([], {
                weekday: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })
            }
            formatter={(v, name) => [`${Math.round(Number(v))} mm`, String(name)]}
            contentStyle={{
              background: "var(--color-background-popover)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              color: "var(--color-text-primary)",
            }}
            cursor={{ stroke: AXIS, strokeDasharray: "3 3" }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            height={28}
            iconType="circle"
            iconSize={8}
            onClick={(entry) => toggle(entry.dataKey as PoseAxis)}
            formatter={(value, entry) => (
              <span
                style={{
                  color: "var(--color-text-secondary)",
                  opacity: hidden.has(entry.dataKey as PoseAxis) ? 0.4 : 1,
                  textDecoration: hidden.has(entry.dataKey as PoseAxis) ? "line-through" : "none",
                  cursor: "pointer",
                  fontSize: 12,
                }}
              >
                {value}
              </span>
            )}
            wrapperStyle={{ cursor: "pointer" }}
          />
          {POSE_AXES.map((item) => (
            <Line
              key={item.key}
              dataKey={item.key}
              name={item.label}
              type="stepAfter"
              stroke={item.color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 5, fill: item.color, stroke: "var(--color-background-card)", strokeWidth: 2 }}
              hide={hidden.has(item.key)}
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
