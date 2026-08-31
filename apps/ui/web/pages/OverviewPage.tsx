import { useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api } from "../api";
import { Metric, PageHeader } from "../components/PageHeader";
import { fmt } from "../api";

type Sample = { time: string; cpu: number; memory: number; disk: number; wifi: number; soc: number };

export function OverviewPage({
  pose,
  farmbot,
}: {
  pose: { x?: number; y?: number; z?: number };
  farmbot: string;
}) {
  const toast = useToast();
  const [info, setInfo] = useState<Record<string, any>>({});
  const [history, setHistory] = useState<Sample[]>([]);
  const [messages, setMessages] = useState<string[]>([]);

  const refresh = async () => {
    try {
      const [status, events] = await Promise.all([
        api<{ state?: any }>("/status"),
        api<{ last_messages?: string[] }>("/messages"),
      ]);
      const next = status.state?.informational_settings || {};
      setInfo(next);
      setMessages(events.last_messages || []);
      setHistory((prev) =>
        [
          ...prev,
          {
            time: new Date().toLocaleTimeString(),
            cpu: Number(next.cpu_usage) || 0,
            memory: Number(next.memory_usage) || 0,
            disk: Number(next.disk_usage) || 0,
            wifi: Number(next.wifi_level_percent) || 0,
            soc: Number(next.soc_temp) || 0,
          },
        ].slice(-60),
      );
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Refresh failed", type: "error" });
    }
  };

  return (
    <VStack gap={5}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Research overview" />
      <div className="metric-grid">
        <Metric label="X · mm" value={fmt(pose.x)} />
        <Metric label="Y · mm" value={fmt(pose.y)} />
        <Metric label="Z · mm" value={fmt(pose.z)} />
      </div>
      <HStack gap={2}>
        <Button label="Refresh status" variant="primary" onClick={() => void refresh()} />
        <Button label="Clear history" variant="secondary" onClick={() => setHistory([])} />
      </HStack>
      <div className="metric-grid">
        <Metric label="FarmBot" value={farmbot} />
        <Metric label="Uptime" value={info.uptime != null ? `${fmt(info.uptime)} s` : "—"} />
        <Metric label="Wi-Fi" value={info.wifi_level_percent != null ? `${fmt(info.wifi_level_percent)}%` : "—"} />
        <Metric label="Sync" value={String(info.sync_status ?? "—")} />
        <Metric label="Busy" value={info.busy ? "Yes" : "No"} />
      </div>
      <div className="metric-grid">
        <Metric label="CPU" value={info.cpu_usage != null ? `${info.cpu_usage}%` : "—"} />
        <Metric label="Memory" value={info.memory_usage != null ? `${info.memory_usage}%` : "—"} />
        <Metric label="Disk" value={info.disk_usage != null ? `${info.disk_usage}%` : "—"} />
        <Metric label="SoC temp" value={info.soc_temp != null ? `${info.soc_temp}°C` : "—"} />
      </div>
      {history.length ? (
        <svg className="chart" viewBox="0 0 600 220">
          {(["cpu", "memory", "disk"] as const).map((key, idx) => {
            const color = ["#628bff", "#72d7ac", "#c7a15b"][idx];
            const pts = history
              .map((s, i) => `${(i / Math.max(history.length - 1, 1)) * 580 + 10},${210 - s[key] * 1.8}`)
              .join(" ");
            return <polyline key={key} fill="none" stroke={color} strokeWidth="2" points={pts} />;
          })}
        </svg>
      ) : (
        <Text color="secondary">Refresh status to start collecting chart data.</Text>
      )}
      <HStack gap={3} wrap>
        <Card padding={4} style={{ flex: 1, minWidth: 240 }}>
          <VStack gap={2}>
            <HeadingLike>Network & hardware</HeadingLike>
            <Text type="supporting">Private IP: {String(info.private_ip ?? "—")}</Text>
            <Text type="supporting">Wi-Fi: {fmt(info.wifi_level)} dBm</Text>
            <Text type="supporting">Controller: {String(info.controller_version ?? "—")}</Text>
            <Text type="supporting">Firmware: {String(info.firmware_version ?? "—")}</Text>
          </VStack>
        </Card>
        <Card padding={4} style={{ flex: 1, minWidth: 240 }}>
          <VStack gap={2}>
            <HeadingLike>Recent events</HeadingLike>
            <Text type="supporting">{messages.slice(-10).join("\n") || "No events recorded."}</Text>
          </VStack>
        </Card>
      </HStack>
    </VStack>
  );
}

function HeadingLike({ children }: { children: string }) {
  return <Text weight="semibold">{children}</Text>;
}
