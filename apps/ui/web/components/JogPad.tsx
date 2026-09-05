import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Center } from "@astryxdesign/core/Center";
import { IconButton } from "@astryxdesign/core/IconButton";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Slider } from "@astryxdesign/core/Slider";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import {
  ArrowDownIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowUpIcon,
  ArrowsPointingInIcon,
} from "@heroicons/react/24/outline";
import { postAction } from "../api";

type Pose = { x?: number; y?: number; z?: number };

const icon = { width: 20, height: 20 };
const STEPS = ["1", "10", "50", "100", "250", "500"];

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
  const [step, setStep] = useState(10);
  const [speed, setSpeed] = useState(100);
  const [gx, setGx] = useState(pose?.x ?? 0);
  const [gy, setGy] = useState(pose?.y ?? 0);
  const [gz, setGz] = useState(pose?.z ?? 0);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setGx(pose?.x ?? 0);
    setGy(pose?.y ?? 0);
    setGz(pose?.z ?? 0);
  }, [pose?.x, pose?.y, pose?.z]);

  const afterMove = async () => {
    if (onMoved) await onMoved();
  };

  const jog = async (axis: "x" | "y" | "z", distance: number, label: string) => {
    setBusy(true);
    try {
      await postAction("move_axis", { axis, distance, speed }, true);
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
      await postAction("move", { x: nx, y: ny, z: nz, speed }, true);
      toast({ body: label || `→ (${nx.toFixed(0)}, ${ny.toFixed(0)}, ${nz.toFixed(0)})` });
      await afterMove();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Move failed", type: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <VStack gap={4}>
      <HStack gap={3} align="end" wrap>
        <NumberInput
          label="Jog distance · mm"
          value={step}
          onChange={setStep}
          min={0.5}
          max={2000}
          step={1}
        />
        <HStack gap={1} wrap>
          {STEPS.map((s) => (
            <Button
              key={s}
              label={s}
              variant={step === Number(s) ? "primary" : "secondary"}
              onClick={() => setStep(Number(s))}
            />
          ))}
        </HStack>
      </HStack>
      <Slider
        label="Speed"
        value={speed}
        onChange={setSpeed}
        min={10}
        max={100}
        step={5}
        valueDisplay="text"
        formatValue={(v) => `${v}%`}
      />
      <Center axis="horizontal">
        <div className="jog-grid">
          <div className="jog-yplus">
            <IconButton label={`Y +${step} mm`} icon={<ArrowUpIcon style={icon} />} width="100%" isDisabled={busy} onClick={() => void jog("y", step, `Y+${step}`)} />
          </div>
          <div className="jog-xmin">
            <IconButton label={`X −${step} mm`} icon={<ArrowLeftIcon style={icon} />} width="100%" isDisabled={busy} onClick={() => void jog("x", -step, `X-${step}`)} />
          </div>
          <div className="jog-origin">
            <IconButton
              label="Go to origin"
              icon={<ArrowsPointingInIcon style={icon} />}
              variant="secondary"
              width="100%"
              isDisabled={busy}
              onClick={() => void move(0, 0, 0, "Origin")}
            />
          </div>
          <div className="jog-xplus">
            <IconButton label={`X +${step} mm`} icon={<ArrowRightIcon style={icon} />} width="100%" isDisabled={busy} onClick={() => void jog("x", step, `X+${step}`)} />
          </div>
          <div className="jog-yminus">
            <IconButton label={`Y −${step} mm`} icon={<ArrowDownIcon style={icon} />} width="100%" isDisabled={busy} onClick={() => void jog("y", -step, `Y-${step}`)} />
          </div>
          <div className="jog-zplus">
            <Button label="Z+" width="100%" isDisabled={busy} onClick={() => void jog("z", step, `Z+${step}`)} />
          </div>
          <div className="jog-zminus">
            <Button label="Z−" width="100%" isDisabled={busy} onClick={() => void jog("z", -step, `Z-${step}`)} />
          </div>
        </div>
      </Center>
      {showGoto ? (
        <VStack gap={2}>
          <Text type="supporting" color="secondary">
            Absolute target · mm
          </Text>
          <HStack gap={2} wrap align="end">
            <NumberInput label="X" value={gx} onChange={setGx} />
            <NumberInput label="Y" value={gy} onChange={setGy} />
            <NumberInput label="Z" value={gz} onChange={setGz} />
            <Button
              label="Go to"
              variant="primary"
              isDisabled={busy}
              onClick={() => void move(gx, gy, gz)}
            />
          </HStack>
        </VStack>
      ) : null}
    </VStack>
  );
}
