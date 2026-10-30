import { useMemo, useState } from "react";
import regression from "regression";
import { linearRegression, linearRegressionLine } from "simple-statistics";
import { BollingerBands, EMA, SMA } from "technicalindicators";
import { Dialog, DialogHeader } from "@astryxdesign/core/Dialog";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { ToggleButton } from "@astryxdesign/core/ToggleButton";
import { MetricLineChart, type ChartRange, type ChartSpec, type Overlay, type SeriesPoint } from "./MetricChart";

type FitId = "ema" | "sma" | "rolling" | "bands" | "linear" | "log" | "poly";

const DAY = 86400;
const PERIODS = [5, 10, 20, 50];

// Bollinger bands draw two series (upper/lower) from one toggle.
const BAND_KEYS = ["bb_upper", "bb_lower"];

const FITS: Record<FitId, Overlay> = {
  ema: { key: "ema", label: "EMA", color: "var(--chart-aqua)" },
  sma: { key: "sma", label: "SMA", color: "var(--chart-blue)" },
  rolling: { key: "rolling", label: "Rolling regression", color: "var(--chart-red)" },
  bands: { key: "bb_upper", label: "Bollinger bands (±2σ)", color: "var(--chart-gray)" },
  linear: { key: "linear", label: "Linear regression", color: "var(--chart-violet)" },
  log: { key: "log", label: "Logarithmic regression", color: "var(--chart-yellow)" },
  poly: { key: "poly", label: "Polynomial (deg 3)", color: "var(--chart-orange)" },
};

type Observed = { index: number; x: number; value: number };

/** Add one overlay series to every point; null where the fit has no value. */
function withOverlays(points: SeriesPoint[], active: FitId[], period: number): SeriesPoint[] {
  const now = Date.now() / 1000;
  const observed: Observed[] = [];
  points.forEach((p, index) => {
    if (p.value != null && p.t <= now) observed.push({ index, x: 0, value: p.value });
  });
  if (observed.length < 3) return points;
  const t0 = points[observed[0].index].t;
  // Regression x is days since the first observation (+1 so log is defined).
  observed.forEach((o) => (o.x = (points[o.index].t - t0) / DAY + 1));
  const data = observed.map((o) => [o.x, o.value] as [number, number]);
  const out = points.map((p) => ({ ...p }));
  const set = (id: string, index: number, value: number) => {
    out[index][id] = Number.isFinite(value) ? value : null;
  };

  if (active.includes("ema")) {
    const series = EMA.calculate({ period, values: observed.map((o) => o.value) });
    series.forEach((v, i) => set("ema", observed[i + period - 1].index, v));
  }
  if (active.includes("sma")) {
    const series = SMA.calculate({ period, values: observed.map((o) => o.value) });
    series.forEach((v, i) => set("sma", observed[i + period - 1].index, v));
  }
  if (active.includes("rolling")) {
    // Least-squares line over each trailing window, evaluated at the window's end.
    for (let i = period - 1; i < data.length; i++) {
      const window = data.slice(i - period + 1, i + 1);
      set("rolling", observed[i].index, linearRegressionLine(linearRegression(window))(data[i][0]));
    }
  }
  if (active.includes("bands")) {
    const bands = BollingerBands.calculate({ period, stdDev: 2, values: observed.map((o) => o.value) });
    bands.forEach((b, i) => {
      set("bb_upper", observed[i + period - 1].index, b.upper);
      set("bb_lower", observed[i + period - 1].index, b.lower);
    });
  }
  // Regressions are fitted on observed points and drawn across the whole range,
  // so linear/poly extend into the forecast.
  const fits: [FitId, regression.Result][] = [];
  if (active.includes("linear")) fits.push(["linear", regression.linear(data)]);
  if (active.includes("log")) fits.push(["log", regression.logarithmic(data)]);
  if (active.includes("poly")) fits.push(["poly", regression.polynomial(data, { order: 3 })]);
  for (const [id, result] of fits) {
    points.forEach((p, index) => set(id, index, result.predict((p.t - t0) / DAY + 1)[1]));
  }
  return out;
}

export function TrendDialog({
  spec,
  range,
  onClose,
}: {
  spec: ChartSpec | null;
  range: ChartRange;
  onClose: () => void;
}) {
  const [active, setActive] = useState<FitId[]>(["linear"]);
  const [opacity, setOpacity] = useState(1);
  const [period, setPeriod] = useState(PERIODS[1]);
  const toggle = (id: FitId) => (on: boolean) =>
    setActive((current) => (on ? [...current, id] : current.filter((x) => x !== id)));

  const shown = useMemo(
    () => (spec ? { ...spec, points: withOverlays(spec.points, active, period) } : null),
    [spec, active, period],
  );
  const overlays = (Object.keys(FITS) as FitId[])
    .filter((id) => active.includes(id))
    .flatMap((id) =>
      id === "bands"
        ? BAND_KEYS.map((key) => ({ ...FITS.bands, key, label: `Bollinger ${key.slice(3)}` }))
        : [FITS[id]],
    );
  const needsPeriod = ["ema", "sma", "rolling", "bands"].some((id) => active.includes(id as FitId));

  return (
    <Dialog isOpen={spec !== null} onOpenChange={(open) => !open && onClose()} width={1400} purpose="info">
      {shown ? (
        <VStack gap={3}>
          <DialogHeader
            title={`${shown.title} · analysis`}
            subtitle={`${range === "year" ? "Daily · last 12 months" : range === "month" ? "Daily · last 31 days" : "Hourly"} · fits use observed data only`}
            onOpenChange={(open) => !open && onClose()}
          />
          <HStack gap={2} wrap="wrap" vAlign="center">
            {(Object.keys(FITS) as FitId[]).map((id) => (
              <ToggleButton
                key={id}
                label={FITS[id].label}
                size="sm"
                isPressed={active.includes(id)}
                onPressedChange={toggle(id)}
              />
            ))}
            <div style={{ visibility: needsPeriod ? "visible" : "hidden" }} aria-hidden={!needsPeriod}>
              <SegmentedControl label="Period" value={String(period)} onChange={(v) => setPeriod(Number(v))}>
                {PERIODS.map((n) => (
                  <SegmentedControlItem key={n} value={String(n)} label={`${n}`} />
                ))}
              </SegmentedControl>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Text type="supporting" color="secondary">
                Data opacity
              </Text>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                style={{ width: 200 }}
              />
            </label>
          </HStack>
          <MetricLineChart spec={shown} height="max(320px, calc(100vh - 340px))" overlays={overlays} baseOpacity={opacity} />
          <Text type="supporting" color="secondary">
            EMA/SMA/rolling/Bollinger window is in samples (hours on Day, days on Month/Year). Regression x-axis is days since the first
            observation.
          </Text>
        </VStack>
      ) : null}
    </Dialog>
  );
}
