import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt, postAction } from "../api";
import { JogPad } from "../components/JogPad";
import { Metric, PageHeader } from "../components/PageHeader";

export function MotionPage({
  pose,
  onMoved,
}: {
  pose: { x?: number; y?: number; z?: number };
  onMoved: () => Promise<void>;
}) {
  const toast = useToast();
  const [presets, setPresets] = useState<any[]>([]);

  useEffect(() => {
    void api<{ positions?: any[] }>("/positions")
      .then((r) => setPresets(r.positions || []))
      .catch(() => setPresets([]));
  }, []);

  const move = async (nx: number, ny: number, nz: number, label?: string) => {
    try {
      await postAction("move", { x: nx, y: ny, z: nz, speed: 100 }, true);
      toast({ body: label || `→ (${nx.toFixed(0)}, ${ny.toFixed(0)}, ${nz.toFixed(0)})` });
      await onMoved();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Move failed", type: "error" });
    }
  };

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Motion workspace" />
      <div className="metric-grid">
        <Metric label="X · mm" value={fmt(pose.x)} />
        <Metric label="Y · mm" value={fmt(pose.y)} />
        <Metric label="Z · mm" value={fmt(pose.z)} />
      </div>
      <JogPad pose={pose} onMoved={onMoved} />
      <Button
        label="Find home"
        onClick={() =>
          void postAction("find_home", {}, true)
            .then(() => toast({ body: "Homing complete" }))
            .then(() => onMoved())
            .catch((error) => toast({ body: String(error.message), type: "error" }))
        }
      />
      {presets.length > 0 && (
        <VStack gap={2}>
          <Text weight="semibold">Locations</Text>
          <HStack gap={2} wrap>
            {presets.map((p) => (
              <Button
                key={p.label}
                label={p.label}
                onClick={() => void move(p.x, p.y, p.z, p.label)}
              />
            ))}
          </HStack>
        </VStack>
      )}
    </VStack>
  );
}
