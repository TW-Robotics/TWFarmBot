import { useCallback, useEffect, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt } from "../api";
import { Metric, PageHeader } from "../components/PageHeader";
import {
  MetricChartDialog,
  PoseChart,
  TrendCard,
  type PoseSample,
  type ChartRange,
  type ChartSpec,
  type SeriesPoint,
} from "../components/MetricChart";

type Meteorology = {
  location?: { timezone?: string };
  sun?: {
    altitude_deg?: number;
    azimuth_deg?: number;
    sun_phase?: string;
    clear_sky_irradiance_w_m2?: number;
  };
  weather?: {
    temperature_c?: number;
    relative_humidity_pct?: number;
    precipitation_mm?: number;
    cloud_cover_pct?: number;
    total_radiation_w_m2?: number;
    uv_index?: number;
    sunrise?: string | null;
    sunset?: string | null;
    source?: string;
  };
  agronomic_advice?: {
    camera_lighting_quality?: string;
    camera_lighting_advice?: string;
    watering_suitability?: string;
    watering_advice?: string;
  };
};

type HourlyPoint = {
  t: number;
  temperature_c: number | null;
  relative_humidity_pct: number | null;
  cloud_cover_pct: number | null;
  precipitation_mm: number | null;
  total_radiation_w_m2: number | null;
  uv_index: number | null;
  altitude_deg: number;
  azimuth_deg: number;
  clear_sky_irradiance_w_m2: number;
};

type MeteoHistory = {
  range?: ChartRange;
  hourly: HourlyPoint[];
  daily: { t: number; sunrise: number; sunset: number }[];
};

type HourlyKey = keyof Omit<HourlyPoint, "t">;

const CHARTS = {
  temperature: hourlyRequest("Air temperature", "°C", "temperature_c", {
    color: "var(--chart-orange)",
    dailyNote: "Daily mean",
  }),
  humidity: hourlyRequest("Humidity", "%", "relative_humidity_pct", {
    color: "var(--chart-aqua)",
    dailyNote: "Daily mean",
  }),
  clouds: hourlyRequest("Cloud cover", "%", "cloud_cover_pct", {
    color: "var(--color-icon-gray)",
    dailyNote: "Daily mean",
  }),
  rain: hourlyRequest("Rain", "mm", "precipitation_mm", {
    mark: "bar",
    color: "var(--chart-blue)",
    dailyNote: "Daily total",
  }),
  solar: hourlyRequest("Solar radiation", "W/m²", "total_radiation_w_m2", {
    referenceLabel: "Clear-sky irradiance",
    color: "var(--chart-yellow)",
    dailyNote: "Daily mean over 24 h",
  }),
  uv: hourlyRequest("UV index", "", "uv_index", {
    color: "var(--chart-violet)",
    dailyNote: "Daily maximum · not available in the yearly archive",
  }),
  altitude: hourlyRequest("Sun altitude", "°", "altitude_deg", {
    color: "var(--chart-orange)",
    dailyNote: "Noon peak altitude",
  }),
  azimuth: hourlyRequest("Sun azimuth", "°", "azimuth_deg", {
    color: "var(--chart-orange)",
    dailyNote: "Direction of sunrise",
  }),
  clearSky: hourlyRequest("Clear-sky irradiance", "W/m²", "clear_sky_irradiance_w_m2", {
    color: "var(--chart-yellow)",
    dailyNote: "Daily mean over 24 h",
  }),
  sunrise: { ...sunTimeRequest("Sunrise", "sunrise"), color: "var(--chart-orange)" },
  sunset: { ...sunTimeRequest("Sunset", "sunset"), color: "var(--chart-violet)" },
};

const POSE_SUBTITLE: Record<ChartRange, string> = {
  day: "Last 24 hours · one point per move",
  month: "Last 31 days · one point per move",
  year: "Last 12 months · one point per move",
};

const TRENDS = [CHARTS.temperature, CHARTS.humidity, CHARTS.solar, CHARTS.rain];

/** How to build one chart's series from any range's history payload. */
type ChartRequest = Omit<ChartSpec, "points" | "axis"> & {
  points: (data: MeteoHistory) => SeriesPoint[];
  /** Explains how the daily (month/year) value is aggregated. */
  dailyNote?: string;
};

function hourlyRequest(
  title: string,
  unit: string,
  key: HourlyKey,
  extra?: Partial<ChartRequest>,
): ChartRequest {
  const withReference = Boolean(extra?.referenceLabel);
  return {
    title,
    unit,
    ...extra,
    points: (data) =>
      data.hourly.map((p) => ({
        t: p.t,
        value: p[key],
        reference: withReference ? p.clear_sky_irradiance_w_m2 : undefined,
      })),
  };
}

function sunTimeRequest(title: string, key: "sunrise" | "sunset"): ChartRequest {
  return {
    title,
    unit: "",
    kind: "time",
    points: (data) => data.daily.map((d) => ({ t: d.t, value: hoursOfDay(d[key]) })),
  };
}

const RANGE_AXIS: Record<ChartRange, ChartSpec["axis"]> = { day: "hour", month: "day", year: "month" };

function buildSpec(request: ChartRequest, data: MeteoHistory, range: ChartRange): ChartSpec {
  return {
    ...request,
    axis: request.kind === "time" && range === "day" ? "day" : RANGE_AXIS[range],
    points: request.points(data),
  };
}

function hoursOfDay(t: number): number {
  const date = new Date(t * 1000);
  return date.getHours() + date.getMinutes() / 60;
}

function labelize(value: string | undefined): string {
  return (value || "—").replace(/_/g, " ");
}

function clock(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function adviceTone(value: string | undefined): "success" | "warning" | "neutral" {
  if (!value) return "neutral";
  if (value === "favorable" || value === "optimal_diffuse" || value === "acceptable") {
    return "success";
  }
  if (
    value === "dark" ||
    value.includes("warning") ||
    value.includes("risk") ||
    value.includes("rain") ||
    value.includes("harsh")
  ) {
    return "warning";
  }
  return "neutral";
}

export function OverviewPage({
  pose,
  farmbot,
}: {
  pose: { x?: number; y?: number; z?: number };
  farmbot: string;
}) {
  const toast = useToast();
  const [meteo, setMeteo] = useState<Meteorology | null>(null);
  const [histories, setHistories] = useState<Partial<Record<ChartRange, MeteoHistory>>>({});
  const [chart, setChart] = useState<ChartRequest | null>(null);
  const [chartRange, setChartRange] = useState<ChartRange>("day");
  const [poseHistories, setPoseHistories] = useState<Partial<Record<ChartRange, PoseSample[]>>>({});
  const [poseOpen, setPoseOpen] = useState(false);
  const [poseRange, setPoseRange] = useState<ChartRange>("day");

  const loadPoseHistory = useCallback(
    async (range: ChartRange) => {
      try {
        const data = await api<{ range?: ChartRange; samples: PoseSample[] }>(
          `/pose/history?range=${range}`,
        );
        if (data.range !== range) throw new Error("The API is outdated — restart twfarmbot-api.");
        setPoseHistories((current) => ({ ...current, [range]: data.samples }));
      } catch (error) {
        setPoseHistories((current) => ({ ...current, [range]: [] }));
        toast({ body: error instanceof Error ? error.message : "Position history unavailable", type: "error" });
      }
    },
    [toast],
  );

  useEffect(() => {
    if (poseOpen && !poseHistories[poseRange]) void loadPoseHistory(poseRange);
  }, [poseOpen, poseRange, poseHistories, loadPoseHistory]);

  const openPose = () => {
    setPoseRange("day");
    setPoseOpen(true);
  };

  /** Recorded moves plus the live pose, so the step line runs up to now. */
  const withLivePose = (samples: PoseSample[] | undefined): PoseSample[] | null => {
    if (!samples) return null;
    if (pose.x == null || pose.y == null || pose.z == null) return samples;
    return [...samples, { t: Date.now() / 1000, x: pose.x, y: pose.y, z: pose.z }];
  };

  const loadHistory = useCallback(
    async (range: ChartRange) => {
      try {
        const data = await api<MeteoHistory>(`/meteorology/history?range=${range}`);
        if (data.range !== range) {
          throw new Error("The API is outdated and ignores the time range — restart twfarmbot-api.");
        }
        setHistories((current) => ({ ...current, [range]: data }));
      } catch (error) {
        toast({ body: error instanceof Error ? error.message : "History unavailable", type: "error" });
      }
    },
    [toast],
  );

  useEffect(() => {
    if (chart && !histories[chartRange]) void loadHistory(chartRange);
  }, [chart, chartRange, histories, loadHistory]);

  const openChart = (request: ChartRequest) => () => {
    setChartRange("day");
    setChart(request);
  };

  const loadMeteo = useCallback(async () => {
    setHistories({});
    setPoseHistories({});
    const [current] = await Promise.allSettled([
      api<Meteorology>("/meteorology"),
      loadHistory("day"),
      loadPoseHistory("day"),
    ]);
    if (current.status === "fulfilled") {
      setMeteo(current.value);
    } else {
      setMeteo(null);
      toast({
        body: current.reason instanceof Error ? current.reason.message : "Weather unavailable",
        type: "error",
      });
    }
  }, [toast, loadHistory, loadPoseHistory]);

  useEffect(() => {
    void loadMeteo();
  }, [loadMeteo]);

  const dayHistory = histories.day;
  const trends = dayHistory ? TRENDS.map((request) => buildSpec(request, dayHistory, "day")) : [];
  const chartHistory = histories[chartRange];
  const chartSpec = chart && chartHistory ? buildSpec(chart, chartHistory, chartRange) : null;

  const sun = meteo?.sun;
  const weather = meteo?.weather;
  const advice = meteo?.agronomic_advice;

  return (
    <VStack gap={5}>
      <HStack gap={3} vAlign="end" wrap="wrap">
        <div style={{ flex: 1, minWidth: 200 }}>
          <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Overview" />
        </div>
        <Button label="Refresh weather" variant="secondary" onClick={() => void loadMeteo()} />
      </HStack>
      <div className="metric-grid">
        <Metric label="Robot" value={farmbot} />
        <Metric label="X · mm" value={fmt(pose.x)} onClick={openPose} />
        <Metric label="Y · mm" value={fmt(pose.y)} onClick={openPose} />
        <Metric label="Z · mm" value={fmt(pose.z)} onClick={openPose} />
      </div>
      {trends.length ? (
        <div className="trend-section">
          <Text weight="semibold">Weather trends · yesterday to tomorrow</Text>
          <div className="trend-grid">
            {trends.map((spec, index) => (
              <TrendCard key={spec.title} spec={spec} onOpen={openChart(TRENDS[index])} />
            ))}
          </div>
        </div>
      ) : null}
      {meteo ? (
        <VStack gap={3}>
          <div className="metric-grid">
            <Metric
              label="Air temp"
              onClick={openChart(CHARTS.temperature)}
              value={weather?.temperature_c != null ? `${fmt(weather.temperature_c)} °C` : "—"}
            />
            <Metric
              label="Humidity"
              onClick={openChart(CHARTS.humidity)}
              value={
                weather?.relative_humidity_pct != null ? `${fmt(weather.relative_humidity_pct)}%` : "—"
              }
            />
            <Metric
              label="Clouds"
              onClick={openChart(CHARTS.clouds)}
              value={weather?.cloud_cover_pct != null ? `${fmt(weather.cloud_cover_pct)}%` : "—"}
            />
            <Metric
              label="Rain"
              onClick={openChart(CHARTS.rain)}
              value={weather?.precipitation_mm != null ? `${fmt(weather.precipitation_mm)} mm` : "—"}
            />
            <Metric
              label="Solar"
              onClick={openChart(CHARTS.solar)}
              value={
                weather?.total_radiation_w_m2 != null ? `${fmt(weather.total_radiation_w_m2)} W/m²` : "—"
              }
            />
            <Metric label="UV" onClick={openChart(CHARTS.uv)} value={weather?.uv_index != null ? fmt(weather.uv_index) : "—"} />
          </div>
          <div className="metric-grid">
            <Metric label="Sun phase" onClick={openChart(CHARTS.altitude)} value={labelize(sun?.sun_phase)} />
            <Metric label="Altitude" onClick={openChart(CHARTS.altitude)} value={sun?.altitude_deg != null ? `${fmt(sun.altitude_deg)}°` : "—"} />
            <Metric label="Azimuth" onClick={openChart(CHARTS.azimuth)} value={sun?.azimuth_deg != null ? `${fmt(sun.azimuth_deg)}°` : "—"} />
            <Metric label="Sunrise" onClick={openChart(CHARTS.sunrise)} value={clock(weather?.sunrise)} />
            <Metric label="Sunset" onClick={openChart(CHARTS.sunset)} value={clock(weather?.sunset)} />
            <Metric
              label="Clear-sky"
              onClick={openChart(CHARTS.clearSky)}
              value={
                sun?.clear_sky_irradiance_w_m2 != null
                  ? `${fmt(sun.clear_sky_irradiance_w_m2)} W/m²`
                  : "—"
              }
            />
          </div>
          <HStack gap={3} wrap="wrap">
            <Card padding={4} style={{ flex: 1, minWidth: 240 }}>
              <VStack gap={2}>
                <HStack gap={2} vAlign="center" wrap="wrap">
                  <Text weight="semibold">Camera lighting</Text>
                  <Badge
                    variant={adviceTone(advice?.camera_lighting_quality)}
                    label={labelize(advice?.camera_lighting_quality)}
                  />
                </HStack>
                <Text type="supporting">{advice?.camera_lighting_advice || "—"}</Text>
              </VStack>
            </Card>
            <Card padding={4} style={{ flex: 1, minWidth: 240 }}>
              <VStack gap={2}>
                <HStack gap={2} vAlign="center" wrap="wrap">
                  <Text weight="semibold">Watering</Text>
                  <Badge
                    variant={adviceTone(advice?.watering_suitability)}
                    label={labelize(advice?.watering_suitability)}
                  />
                </HStack>
                <Text type="supporting">{advice?.watering_advice || "—"}</Text>
              </VStack>
            </Card>
          </HStack>
          <Text type="supporting" color="secondary">
            {meteo.location?.timezone || "local"}
            {weather?.source ? ` · ${weather.source}` : ""}
          </Text>
        </VStack>
      ) : (
        <Text color="secondary">Weather is unavailable. Restart the API if GET /meteorology is missing.</Text>
      )}
      <MetricChartDialog
        title={poseOpen ? "Gantry position" : null}
        spec={null}
        range={poseRange}
        subtitle={POSE_SUBTITLE[poseRange]}
        note="Click X, Y or Z in the legend to hide or show an axis."
        content={
          <PoseChart
            samples={withLivePose(poseHistories[poseRange])}
            height={360}
            axis={poseRange === "day" ? "hour" : poseRange === "month" ? "day" : "month"}
          />
        }
        onRangeChange={setPoseRange}
        onClose={() => setPoseOpen(false)}
      />
      <MetricChartDialog
        title={chart?.title ?? null}
        spec={chartSpec}
        range={chartRange}
        note={chartRange !== "day" ? chart?.dailyNote : undefined}
        onRangeChange={setChartRange}
        onClose={() => setChart(null)}
      />
    </VStack>
  );
}
