import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { TextInput } from "@astryxdesign/core/TextInput";
import { useToast } from "@astryxdesign/core/Toast";
import { fmt, parseNumber, postAction } from "../api";

type Pose = { x?: number; y?: number; z?: number };

export function JogPad({
  pose,
  onMoved,
  showGoto = true,
}: {
  pose?: Pose;
  onMoved?: () => Promise<void> | void;
  showGoto?: boolean;
}) {
  const toast = useToast();
  const [step, setStep] = useState("10");
  const [speed, setSpeed] = useState("100");
  const [gx, setGx] = useState(fmt(pose?.x));
  const [gy, setGy] = useState(fmt(pose?.y));
  const [gz, setGz] = useState(fmt(pose?.z));
  const [busy, setBusy] = useState(false);
  const d = Number(step);

  useEffect(() => {
    setGx(fmt(pose?.x));
    setGy(fmt(pose?.y));
    setGz(fmt(pose?.z));
  }, [pose?.x, pose?.y, pose?.z]);

  const speedParams = (): Record<string, number> => {
    const speedPct = Number(speed);
    if (Number.isFinite(speedPct) && speedPct > 0 && speedPct <= 100) {
      return { speed: speedPct };
    }
    return { speed: 100 };
  };

  const afterMove = async () => {
    if (onMoved) await onMoved();
  };

  const jog = async (axis: "x" | "y" | "z", distance: number, label: string) => {
    setBusy(true);
    try {
      await postAction("move_axis", { axis, distance, ...speedParams() }, true);
      toast({ body: label });
      await afterMove();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Move failed", type: "error" });
    } finally {
      setBusy(false);
    }
  };

  const move = async (nx: number, ny: number, nz: number, label?: string) => {
    setBusy(true);
    try {
      await postAction("move", { x: nx, y: ny, z: nz, ...speedParams() }, true);
      toast({ body: label || `→ (${nx.toFixed(0)}, ${ny.toFixed(0)}, ${nz.toFixed(0)})` });
      await afterMove();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Move failed", type: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <VStack gap={3}>
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
        <Button label={`Y+ ${d}`} isDisabled={busy} onClick={() => void jog("y", d, `Y+${d}`)} />
      </HStack>
      <HStack gap={2} justify="center">
        <Button label={`X− ${d}`} isDisabled={busy} onClick={() => void jog("x", -d, `X-${d}`)} />
        <Button
          label="Home"
          variant="primary"
          isDisabled={busy}
          onClick={() => void move(0, 0, 0, "Home")}
        />
        <Button label={`X+ ${d}`} isDisabled={busy} onClick={() => void jog("x", d, `X+${d}`)} />
      </HStack>
      <HStack gap={2} justify="center">
        <Button label={`Y− ${d}`} isDisabled={busy} onClick={() => void jog("y", -d, `Y-${d}`)} />
      </HStack>
      <HStack gap={2}>
        <Button label={`Z+ ${d}`} isDisabled={busy} onClick={() => void jog("z", d, `Z+${d}`)} />
        <Button label={`Z− ${d}`} isDisabled={busy} onClick={() => void jog("z", -d, `Z-${d}`)} />
      </HStack>
      {showGoto ? (
        <HStack gap={2}>
          <TextInput label="X" value={gx} onChange={setGx} />
          <TextInput label="Y" value={gy} onChange={setGy} />
          <TextInput label="Z" value={gz} onChange={setGz} />
          <Button
            label="Go to"
            variant="primary"
            isDisabled={busy}
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
      ) : null}
    </VStack>
  );
}
