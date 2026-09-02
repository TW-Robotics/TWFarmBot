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
  const [speed, setSpeed] = useState("100");
  const [gx, setGx] = useState(fmt(pose.x));
  const [gy, setGy] = useState(fmt(pose.y));
  const [gz, setGz] = useState(fmt(pose.z));
  const [presets, setPresets] = useState<any[]>([]);

  useEffect(() => {
    void api<{ positions?: any[] }>("/positions")
      .then((r) => setPresets(r.positions || []))
      .catch(() => setPresets([]));
  }, []);

  const d = Number(step);

  const speedParams = (): Record<string, number> => {
    const speedPct = Number(speed);
    if (Number.isFinite(speedPct) && speedPct > 0 && speedPct <= 100) {
      return { speed: speedPct };
    }
    return { speed: 100 };
  };

  const jog = async (axis: "x" | "y" | "z", distance: number, label: string) => {
    try {
      await postAction("move_axis", { axis, distance, ...speedParams() }, true);
      toast({ body: label });
      await onMoved();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Move failed", type: "error" });
    }
  };

  const move = async (nx: number, ny: number, nz: number, label?: string) => {
    try {
      await postAction("move", { x: nx, y: ny, z: nz, ...speedParams() }, true);
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
      <SegmentedControl value={speed} onChange={setSpeed} label="Move speed · %">
        {["25", "50", "75", "100"].map((s) => (
          <SegmentedControlItem key={s} value={s} label={s} />
        ))}
      </SegmentedControl>
      <HStack gap={2} justify="center">
        <Button label={`Y+ ${d}`} onClick={() => void jog("y", d, `Y+${d}`)} />
      </HStack>
      <HStack gap={2} justify="center">
        <Button label={`X− ${d}`} onClick={() => void jog("x", -d, `X-${d}`)} />
        <Button label="Home" variant="primary" onClick={() => void move(0, 0, 0, "Home")} />
        <Button label={`X+ ${d}`} onClick={() => void jog("x", d, `X+${d}`)} />
      </HStack>
      <HStack gap={2} justify="center">
        <Button label={`Y− ${d}`} onClick={() => void jog("y", -d, `Y-${d}`)} />
      </HStack>
      <HStack gap={2}>
        <Button label={`Z+ ${d}`} onClick={() => void jog("z", d, `Z+${d}`)} />
        <Button label={`Z− ${d}`} onClick={() => void jog("z", -d, `Z-${d}`)} />
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
