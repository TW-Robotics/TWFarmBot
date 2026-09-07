import { useCallback, useEffect, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt } from "../api";
import { Metric, PageHeader } from "../components/PageHeader";

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

  const loadMeteo = useCallback(async () => {
    try {
      setMeteo(await api<Meteorology>("/meteorology"));
    } catch (error) {
      setMeteo(null);
      toast({
        body: error instanceof Error ? error.message : "Weather unavailable",
        type: "error",
      });
    }
  }, [toast]);

  useEffect(() => {
    void loadMeteo();
  }, [loadMeteo]);

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
        <Metric label="X · mm" value={fmt(pose.x)} />
        <Metric label="Y · mm" value={fmt(pose.y)} />
        <Metric label="Z · mm" value={fmt(pose.z)} />
      </div>
      {meteo ? (
        <VStack gap={3}>
          <div className="metric-grid">
            <Metric
              label="Air temp"
              value={weather?.temperature_c != null ? `${fmt(weather.temperature_c)} °C` : "—"}
            />
            <Metric
              label="Humidity"
              value={
                weather?.relative_humidity_pct != null ? `${fmt(weather.relative_humidity_pct)}%` : "—"
              }
            />
            <Metric
              label="Clouds"
              value={weather?.cloud_cover_pct != null ? `${fmt(weather.cloud_cover_pct)}%` : "—"}
            />
            <Metric
              label="Rain"
              value={weather?.precipitation_mm != null ? `${fmt(weather.precipitation_mm)} mm` : "—"}
            />
            <Metric
              label="Solar"
              value={
                weather?.total_radiation_w_m2 != null ? `${fmt(weather.total_radiation_w_m2)} W/m²` : "—"
              }
            />
            <Metric label="UV" value={weather?.uv_index != null ? fmt(weather.uv_index) : "—"} />
          </div>
          <div className="metric-grid">
            <Metric label="Sun phase" value={labelize(sun?.sun_phase)} />
            <Metric label="Altitude" value={sun?.altitude_deg != null ? `${fmt(sun.altitude_deg)}°` : "—"} />
            <Metric label="Azimuth" value={sun?.azimuth_deg != null ? `${fmt(sun.azimuth_deg)}°` : "—"} />
            <Metric label="Sunrise" value={clock(weather?.sunrise)} />
            <Metric label="Sunset" value={clock(weather?.sunset)} />
            <Metric
              label="Clear-sky"
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
    </VStack>
  );
}
