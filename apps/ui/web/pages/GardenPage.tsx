import { useEffect, useMemo, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Selector } from "@astryxdesign/core/Selector";
import { TextInput } from "@astryxdesign/core/TextInput";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt, localApi } from "../api";
import { Metric, PageHeader } from "../components/PageHeader";

type Entity = { name: string; kind: string; position: { x: number; y: number }; radius_mm: number };
type Zone = { name: string; kind: string; bounds: { x: number; y: number; width: number; height: number } };

const KINDS = ["plant", "obstacle", "tool", "marker", "sensor", "valve", "custom"];

export function GardenPage() {
  const toast = useToast();
  const [world, setWorld] = useState<any>(null);
  const [selected, setSelected] = useState<{ x: number; y: number } | null>(null);
  const [kind, setKind] = useState("plant");
  const [customKind, setCustomKind] = useState("");
  const [name, setName] = useState("");

  const load = async () => {
    try {
      setWorld(await api("/garden"));
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Garden unavailable", type: "error" });
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const bounds = world?.bounds || { x: 0, y: 0, width: 1, height: 1 };
  const entities: Entity[] = world?.entities || [];
  const zones: Zone[] = world?.zones || [];
  const robot = world?.robot || {};
  const camera = world?.camera || {};

  const points = useMemo(
    () => [
      ...entities.map((e) => ({ ...e.position, kind: e.kind, name: e.name, r: e.radius_mm })),
      { x: robot.x || 0, y: robot.y || 0, kind: "robot", name: "FarmBot", r: 35 },
      {
        x: camera.position?.x || 0,
        y: camera.position?.y || 0,
        kind: "camera",
        name: "Camera",
        r: 25,
      },
    ],
    [entities, robot, camera],
  );

  const assign = async () => {
    if (!selected || !name) {
      toast({ body: "Select a point and enter a name", type: "error" });
      return;
    }
    try {
      await localApi("/local/garden-entities", {
        method: "POST",
        body: JSON.stringify({
          x: selected.x,
          y: selected.y,
          kind: kind === "custom" ? customKind || "plant" : kind,
          name,
        }),
      });
      toast({ body: "Entity added" });
      setSelected(null);
      await load();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Save failed", type: "error" });
    }
  };

  if (!world) {
    return (
      <VStack gap={4}>
        <PageHeader kicker="Spatial model" title="Garden map" />
        <Text color="secondary">Loading garden…</Text>
      </VStack>
    );
  }

  const x0 = bounds.x || 0;
  const y0 = bounds.y || 0;
  const w = bounds.width || 1;
  const h = bounds.height || 1;

  const toX = (x: number) => ((x - x0) / w) * 800;
  const toY = (y: number) => 520 - ((y - y0) / h) * 520;
  const fromEvent = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * w + x0;
    const py = ((box.bottom - event.clientY) / box.height) * h + y0;
    setSelected({ x: Math.round(px / 25) * 25, y: Math.round(py / 25) * 25 });
  };

  return (
    <VStack gap={4}>
      <PageHeader kicker="Spatial model · configured world state" title="Garden map" />
      <Button label="Refresh map" onClick={() => void load()} />
      <div className="metric-grid">
        <Metric label="Garden X" value={`${fmt(w)} mm`} />
        <Metric label="Garden Y" value={`${fmt(h)} mm`} />
        <Metric label="Known objects" value={String(entities.length)} />
        <Metric label="Mapped zones" value={String(zones.length)} />
      </div>
      <HStack gap={4}>
        <svg className="garden-map" viewBox="0 0 800 520" onClick={fromEvent}>
          <rect x="0" y="0" width="800" height="520" fill="none" stroke="currentColor" strokeOpacity="0.3" />
          {zones.map((zone) => (
            <rect
              key={zone.name}
              x={toX(zone.bounds.x)}
              y={toY(zone.bounds.y + zone.bounds.height)}
              width={(zone.bounds.width / w) * 800}
              height={(zone.bounds.height / h) * 520}
              fill="#225BFF"
              fillOpacity="0.18"
              stroke="#628bff"
            />
          ))}
          {points.map((p) => (
            <circle key={`${p.name}-${p.x}-${p.y}`} cx={toX(p.x)} cy={toY(p.y)} r={Math.max(4, p.r / 12)}>
              <title>{`${p.name} (${p.kind})`}</title>
            </circle>
          ))}
          {selected && (
            <circle cx={toX(selected.x)} cy={toY(selected.y)} r="8" fill="#ff4b4b" />
          )}
        </svg>
        <Card padding={4} style={{ minWidth: 240, flex: "0 0 280px" }}>
          <VStack gap={3}>
            <Text weight="semibold">Live pose</Text>
            <Text type="supporting">
              X {fmt(robot.x)} · Y {fmt(robot.y)} · Z {fmt(robot.z)}
            </Text>
            <Text weight="semibold">Camera pose</Text>
            <Text type="supporting">
              X {fmt(camera.position?.x)} · Y {fmt(camera.position?.y)} · Z {fmt(camera.position?.z)}
            </Text>
            {selected && (
              <>
                <Text>
                  Selected {selected.x}, {selected.y}
                </Text>
                <Selector label="Kind" options={KINDS} value={kind} onChange={setKind} />
                {kind === "custom" && (
                  <TextInput label="Custom kind" value={customKind} onChange={setCustomKind} />
                )}
                <TextInput label="Name" value={name} onChange={setName} />
                <Button label="Assign" variant="primary" onClick={() => void assign()} />
              </>
            )}
            {entities.map((e) => (
              <Text key={e.name} type="supporting">
                {e.name} · {e.kind}
              </Text>
            ))}
          </VStack>
        </Card>
      </HStack>
    </VStack>
  );
}
