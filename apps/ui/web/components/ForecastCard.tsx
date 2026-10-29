import { useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Card } from "@astryxdesign/core/Card";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { MetricLineChart, type ChartSpec } from "./MetricChart";

type Block = {
  t: number;
  temperature_c: number | null;
  relative_humidity_pct: number | null;
  cloud_cover_pct: number | null;
  precipitation_mm: number | null;
  total_radiation_w_m2: number | null;
};

type Day = {
  date: string;
  temp_min_c: number | null;
  temp_max_c: number | null;
  rain_mm: number | null;
  cloud_mean_pct: number | null;
  solar_peak_w_m2: number | null;
};

/** Shape returned by the assistant's `get_forecast` tool. */
export type ForecastResult = {
  timezone?: string;
  hourly: Block[];
  daily?: Day[];
  next_rain_at?: number | null;
  rain_total_mm?: number | null;
  display?: boolean;
};

const METRICS: {
  id: string;
  label: string;
  title: string;
  unit: string;
  key: keyof Omit<Block, "t">;
  color: string;
  mark?: "bar";
}[] = [
  { id: "temp", label: "Temp", title: "Air temperature", unit: "°C", key: "temperature_c", color: "var(--chart-orange)" },
  { id: "rain", label: "Rain", title: "Rain", unit: "mm", key: "precipitation_mm", color: "var(--chart-blue)", mark: "bar" },
  { id: "sun", label: "Sun", title: "Solar radiation", unit: "W/m²", key: "total_radiation_w_m2", color: "var(--chart-yellow)" },
  { id: "cloud", label: "Clouds", title: "Cloud cover", unit: "%", key: "cloud_cover_pct", color: "var(--color-icon-gray)" },
];

export function isForecastResult(value: unknown): value is ForecastResult {
  const record = value as Record<string, unknown> | null;
  return Boolean(record && Array.isArray(record.hourly) && record.hourly.length > 1 && !record.error);
}

function dayName(date: string): string {
  return new Date(`${date}T12:00:00`).toLocaleDateString([], { weekday: "short" });
}

function rainBadge(result: ForecastResult) {
  if (result.next_rain_at == null) return <Badge variant="success" label="Dry" />;
  const when = new Date(result.next_rain_at * 1000).toLocaleString([], {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
  return <Badge variant="neutral" label={`Rain from ${when}`} />;
}

export function ForecastCard({ result }: { result: ForecastResult }) {
  const [metricId, setMetricId] = useState("temp");
  const metric = METRICS.find((item) => item.id === metricId) ?? METRICS[0];
  const spec: ChartSpec = {
    title: metric.title,
    unit: metric.unit,
    color: metric.color,
    mark: metric.mark,
    axis: "hour",
    points: result.hourly.map((block) => ({ t: block.t, value: block[metric.key] })),
  };
  const hours = Math.round((result.hourly[result.hourly.length - 1].t - result.hourly[0].t) / 3600);

  return (
    <Card padding={3}>
      <VStack gap={3}>
        <HStack gap={2} vAlign="center" wrap="wrap">
          <div style={{ flex: 1, minWidth: 0 }}>
            <Text weight="semibold">Forecast · next {hours + 3} hours</Text>
          </div>
          {rainBadge(result)}
        </HStack>
        <SegmentedControl label="Forecast metric" value={metricId} onChange={setMetricId} size="sm">
          {METRICS.map((item) => (
            <SegmentedControlItem key={item.id} value={item.id} label={item.label} />
          ))}
        </SegmentedControl>
        <MetricLineChart spec={spec} height={190} isCompact />
        {result.daily?.length ? (
          <div className="forecast-days">
            {result.daily.map((day) => (
              <div key={day.date} className="forecast-day">
                <Text type="supporting" color="secondary">
                  {dayName(day.date)}
                </Text>
                <Text weight="semibold">
                  {day.temp_min_c != null && day.temp_max_c != null
                    ? `${Math.round(day.temp_min_c)}–${Math.round(day.temp_max_c)} °C`
                    : "—"}
                </Text>
                <Text type="supporting" color="secondary">
                  {day.rain_mm ? `${day.rain_mm} mm rain` : "dry"}
                </Text>
              </div>
            ))}
          </div>
        ) : null}
      </VStack>
    </Card>
  );
}
