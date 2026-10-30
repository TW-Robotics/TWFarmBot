import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { linearRegression, max, mean, min } from "simple-statistics";
import { Badge } from "@astryxdesign/core/Badge";
import { Button } from "@astryxdesign/core/Button";
import { ClickableCard } from "@astryxdesign/core/ClickableCard";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Spinner } from "@astryxdesign/core/Spinner";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api } from "../api";
import { PageHeader } from "../components/PageHeader";
import { TrendDialog } from "../components/TrendDialog";
import { MetricLineChart, type ChartRange, type ChartSpec } from "../components/MetricChart";
import { DLI_PER_WM2_DAY, GDD_BASE_C, GLASS_TRANSMISSION } from "../agronomy";
import { buildSpec, CHARTS, type ChartRequest, type MeteoHistory } from "./OverviewPage";

/** Days covered by one point: hourly for the day range, daily otherwise. */
const stepDays = (data: MeteoHistory) => (data.range === "day" ? 1 / 24 : 1);

/** Cumulative growing degree days: heat above the base temperature, summed over the range. */
const GDD: ChartRequest = {
  title: "Growing degree days",
  unit: "°C·d",
  color: "var(--chart-orange)",
  dailyNote: `Cumulative heat above ${GDD_BASE_C} °C, from outdoor air temperature`,
  points: (data) => {
    let total = 0;
    return data.hourly.map((p) => {
      if (p.temperature_c != null) total += Math.max(0, p.temperature_c - GDD_BASE_C) * stepDays(data);
      return { t: p.t, value: Math.round(total * 10) / 10 };
    });
  },
};

/**
 * Daily light integral (mol/m²/day) behind glass. Day range shows the running total,
 * reset at local midnight; month/year show one value per day.
 */
const DLI: ChartRequest = {
  title: "Daily light integral",
  unit: "mol/m²",
  color: "var(--chart-yellow)",
  dailyNote: `Photosynthetic light per day, assuming ${GLASS_TRANSMISSION * 100}% glass transmission`,
  points: (data) => {
    const molPerWm2Day = DLI_PER_WM2_DAY;
    let running = 0;
    let day = "";
    return data.hourly.map((p) => {
      if (data.range !== "day") {
        return { t: p.t, value: p.total_radiation_w_m2 == null ? null : p.total_radiation_w_m2 * molPerWm2Day };
      }
      const key = new Date(p.t * 1000).toDateString();
      if (key !== day) [day, running] = [key, 0];
      running += (p.total_radiation_w_m2 ?? 0) * molPerWm2Day * stepDays(data);
      return { t: p.t, value: Math.round(running * 100) / 100 };
    });
  },
};

const METRICS = [
  GDD,
  DLI,
  CHARTS.temperature,
  CHARTS.humidity,
  CHARTS.clouds,
  CHARTS.rain,
  CHARTS.solar,
  CHARTS.uv,
  CHARTS.wind,
  CHARTS.gusts,
  CHARTS.pressure,
  CHARTS.dewPoint,
  CHARTS.vpd,
  CHARTS.et0,
  CHARTS.soilTemp,
  CHARTS.soilMoisture,
];

const DAY = 86400;
const PER_RANGE: Record<ChartRange, string> = { day: "per day", month: "per day", year: "per month" };
// A fitted change smaller than this share of the observed spread counts as steady.
const STEADY_SHARE = 0.15;

type Analysis = {
  min: number;
  max: number;
  mean: number;
  /** Fitted change per day (per month for the year view). */
  slope: number;
  direction: "rising" | "falling" | "steady";
};

/** Min / max / mean and a least-squares trend over the observed (non-forecast) points. */
function analyse(spec: ChartSpec, range: ChartRange): Analysis | null {
  const now = Date.now() / 1000;
  const rows = spec.points
    .filter((p): p is typeof p & { value: number } => p.value != null && p.t <= now)
    .map((p) => [p.t / DAY, p.value]);
  if (rows.length < 3) return null;
  const values = rows.map(([, v]) => v);
  const { m } = linearRegression(rows);
  const spread = max(values) - min(values);
  const change = Math.abs(m * (rows[rows.length - 1][0] - rows[0][0]));
  const direction = spread === 0 || change < spread * STEADY_SHARE ? "steady" : m > 0 ? "rising" : "falling";
  return {
    min: min(values),
    max: max(values),
    mean: mean(values),
    slope: range === "year" ? m * 30 : m,
    direction,
  };
}

const TONE = { rising: "warning", falling: "neutral", steady: "success" } as const;

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <VStack gap={0}>
      <Text type="supporting" color="secondary">
        {label}
      </Text>
      <Text weight="semibold">{value}</Text>
    </VStack>
  );
}

const TrendPanel = memo(function TrendPanel({
  request,
  spec,
  range,
  onOpen,
}: {
  request: ChartRequest;
  spec: ChartSpec;
  range: ChartRange;
  onOpen: (request: ChartRequest) => void;
}) {
  const stats = useMemo(() => analyse(spec, range), [spec, range]);
  const unit = spec.unit ? ` ${spec.unit}` : "";
  const n = (v: number) => `${v.toFixed(1)}${unit}`;
  return (
    <ClickableCard label={`${spec.title} analysis`} padding={4} onClick={() => onOpen(request)}>
      <VStack gap={3}>
        <HStack gap={2} vAlign="center">
          <div style={{ flex: 1 }}>
            <Text weight="semibold">{spec.title}</Text>
          </div>
          {stats ? <Badge variant={TONE[stats.direction]} label={stats.direction} /> : null}
        </HStack>
        <MetricLineChart spec={spec} height={160} isCompact />
        {stats ? (
          <HStack gap={4} wrap="wrap">
            <Stat label="Min" value={n(stats.min)} />
            <Stat label="Mean" value={n(stats.mean)} />
            <Stat label="Max" value={n(stats.max)} />
            <Stat label="Trend" value={`${stats.slope >= 0 ? "+" : ""}${n(stats.slope)} ${PER_RANGE[range]}`} />
          </HStack>
        ) : (
          <Text type="supporting" color="secondary">
            Not enough data for this range.
          </Text>
        )}
      </VStack>
    </ClickableCard>
  );
});

export function TrendsPage() {
  const toast = useToast();
  const [range, setRange] = useState<ChartRange>("month");
  const [open, setOpen] = useState<ChartRequest | null>(null);
  const [histories, setHistories] = useState<Partial<Record<ChartRange, MeteoHistory>>>({});

  const load = useCallback(
    async (target: ChartRange) => {
      try {
        const data = await api<MeteoHistory>(`/meteorology/history?range=${target}`);
        setHistories((current) => ({ ...current, [target]: data }));
      } catch (error) {
        toast({ body: error instanceof Error ? error.message : "History unavailable", type: "error" });
      }
    },
    [toast],
  );

  useEffect(() => {
    if (!histories[range]) void load(range);
  }, [range, histories, load]);

  const history = histories[range];
  const specs = useMemo(
    () => (history ? METRICS.map((request) => buildSpec(request, history, range)) : []),
    [history, range],
  );
  const openSpec = useMemo(() => specs.find((s) => s.title === open?.title) ?? null, [specs, open]);

  return (
    <VStack gap={5}>
      <HStack gap={3} vAlign="end" wrap="wrap">
        <div style={{ flex: 1, minWidth: 200 }}>
          <PageHeader kicker="Weather" title="Trend analysis" />
        </div>
        <SegmentedControl label="Time range" value={range} onChange={(v) => setRange(v as ChartRange)}>
          <SegmentedControlItem value="day" label="Day" />
          <SegmentedControlItem value="month" label="Month" />
          <SegmentedControlItem value="year" label="Year" />
        </SegmentedControl>
        <Button label="Refresh" variant="secondary" onClick={() => void load(range)} />
      </HStack>
      {history ? (
        <div className="trends-grid">
          {METRICS.map((request, i) => (
            <TrendPanel key={request.title} request={request} spec={specs[i]} range={range} onOpen={setOpen} />
          ))}
        </div>
      ) : (
        <div style={{ height: 240, display: "grid", placeItems: "center" }}>
          <Spinner />
        </div>
      )}
      <TrendDialog
        spec={openSpec}
        range={range}
        onClose={() => setOpen(null)}
      />
    </VStack>
  );
}
