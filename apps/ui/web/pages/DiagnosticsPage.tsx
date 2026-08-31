import { useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt } from "../api";
import { Metric, PageHeader } from "../components/PageHeader";

export function DiagnosticsPage() {
  const toast = useToast();
  const [state, setState] = useState<any>({});

  const load = async () => {
    try {
      const r = await api<{ state?: any }>("/status");
      setState(r.state || {});
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Read failed", type: "error" });
    }
  };

  const info = state.informational_settings || {};
  const axes = state.location_data?.axis_states || {};
  const pins = state.pins || {};

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Diagnostics" />
      <Button label="Load /status" variant="primary" onClick={() => void load()} />
      {!info.controller_version && !Object.keys(pins).length ? (
        <Text color="secondary">Load /status to fetch diagnostic state.</Text>
      ) : (
        <>
          <div className="metric-grid">
            <Metric label="Controller" value={String(info.controller_version ?? "—")} />
            <Metric label="Firmware" value={String(info.firmware_version ?? "—")} />
            <Metric label="Wi-Fi" value={`${fmt(info.wifi_level_percent)}%`} />
            <Metric label="Uptime" value={`${fmt(info.uptime)} s`} />
          </div>
          <HStack gap={3}>
            <Card padding={3}>
              <Text>
                CPU {info.cpu_usage ?? "—"}% · Memory {info.memory_usage ?? "—"}% · Disk {info.disk_usage ?? "—"}%
              </Text>
            </Card>
            <Card padding={3}>
              <Text>
                Axis X {axes.x ?? "—"} · Y {axes.y ?? "—"} · Z {axes.z ?? "—"}
              </Text>
            </Card>
          </HStack>
        </>
      )}
    </VStack>
  );
}
