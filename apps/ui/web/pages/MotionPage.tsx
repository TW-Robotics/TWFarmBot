import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { TextInput } from "@astryxdesign/core/TextInput";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt, parseNumber, postAction } from "../api";
import { Metric, PageHeader } from "../components/PageHeader";

export function MotionPage({
  pose,
  onMoved,
}: {
  pose: { x?: number; y?: number; z?: number };
  onMoved: () => Promise<void>;
}) {
  const toast = useToast();
  const [step, setStep] = useState("10");
  const [gx, setGx] = useState(fmt(pose.x));
  const [gy, setGy] = useState(fmt(pose.y));
  const [gz, setGz] = useState(fmt(pose.z));
  const [presets, setPresets] = useState<any[]>([]);

  useEffect(() => {
    void api<{ positions?: any[] }>("/positions")
      .then((r) => setPresets(r.positions || []))
      .catch(() => setPresets([]));
  }, []);

  const x = Number(pose.x) || 0;
  const y = Number(pose.y) || 0;
  const z = Number(pose.z) || 0;
  const d = Number(step);

  const move = async (nx: number, ny: number, nz: number, label?: string) => {
    try {
      await postAction("move", { x: nx, y: ny, z: nz });
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
      <SegmentedControl value={step} onChange={setStep} label="Jog step · mm">
        {["1", "10", "50", "100"].map((s) => (
          <SegmentedControlItem key={s} value={s} label={s} />
        ))}
      </SegmentedControl>
      <HStack gap={2} justify="center">
        <Button label={`Y+ ${d}`} onClick={() => void move(x, y + d, z, `Y+${d}`)} />
      </HStack>
      <HStack gap={2} justify="center">
        <Button label={`X− ${d}`} onClick={() => void move(x - d, y, z, `X-${d}`)} />
        <Button label="Home" variant="primary" onClick={() => void move(0, 0, 0, "Home")} />
        <Button label={`X+ ${d}`} onClick={() => void move(x + d, y, z, `X+${d}`)} />
      </HStack>
      <HStack gap={2} justify="center">
        <Button label={`Y− ${d}`} onClick={() => void move(x, y - d, z, `Y-${d}`)} />
      </HStack>
      <HStack gap={2}>
        <Button label={`Z+ ${d}`} onClick={() => void move(x, y, z + d, `Z+${d}`)} />
        <Button label={`Z− ${d}`} onClick={() => void move(x, y, z - d, `Z-${d}`)} />
      </HStack>
      <HStack gap={2}>
        <TextInput label="X" value={gx} onChange={setGx} />
        <TextInput label="Y" value={gy} onChange={setGy} />
        <TextInput label="Z" value={gz} onChange={setGz} />
        <Button
          label="Go to"
          variant="primary"
          onClick={() => {
            const nx = parseNumber(gx);
            const ny = parseNumber(gy);
            const nz = parseNumber(gz);
            if (nx == null || ny == null || nz == null) {
              toast({ body: "Use a plain number like 123.4", type: "error" });
              return;
            }
            void move(nx, ny, nz);
          }}
        />
      </HStack>
      <Button
        label="Find home"
        onClick={() =>
          void postAction("find_home")
            .then(() => toast({ body: "Homing queued" }))
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
