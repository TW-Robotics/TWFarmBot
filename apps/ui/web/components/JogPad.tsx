import { useEffect, useRef, useState, type ReactNode } from "react";
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
import { useDevice } from "../device";

type Pose = { x?: number; y?: number; z?: number };

const ICON_DESKTOP = { width: 20, height: 20 };
const ICON_TABLET = { width: 34, height: 34 };
const STEPS_DESKTOP = ["1", "10", "50", "100", "250", "500"];
const STEPS_TABLET = [10, 50, 100, 250, 500];

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
  const { isTablet } = useDevice();
  const icon = isTablet ? ICON_TABLET : ICON_DESKTOP;
  const [step, setStep] = useState(isTablet ? 50 : 10);
  const [speed, setSpeed] = useState(100);
  const [gx, setGx] = useState(pose?.x ?? 0);
  const [gy, setGy] = useState(pose?.y ?? 0);
  const [gz, setGz] = useState(pose?.z ?? 0);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const holdRef = useRef<number | null>(null);
  const lastPointerRef = useRef(0);

  useEffect(() => {
    setGx(pose?.x ?? 0);
    setGy(pose?.y ?? 0);
    setGz(pose?.z ?? 0);
  }, [pose?.x, pose?.y, pose?.z]);

  const stopHold = () => {
    if (holdRef.current !== null) window.clearInterval(holdRef.current);
    holdRef.current = null;
  };
  useEffect(() => stopHold, []);

  const jog = async (axis: "x" | "y" | "z", distance: number) => {
    if (busyRef.current) return; // one move in flight; hold-ticks just skip
    busyRef.current = true;
    setBusy(true);
    try {
      await postAction("move_axis", { axis, distance, speed }, true);
      await onMoved?.();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Move failed", type: "error" });
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  const move = async (nx: number, ny: number, nz: number, label?: string) => {
    setBusy(true);
    try {
      await postAction("move", { x: nx, y: ny, z: nz, speed }, true);
      toast({ body: label || `→ (${nx.toFixed(0)}, ${ny.toFixed(0)}, ${nz.toFixed(0)})` });
      await onMoved?.();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Move failed", type: "error" });
    } finally {
      setBusy(false);
    }
  };

  // Tap = one step; press-and-hold = repeat steps until released (tablet/gloves:
  // no double-taps, no precise clicking). ponytail: stepped repeat via the
  // existing move_axis, not a continuous-motion protocol.
  // Desktop keeps plain clicks (hold would surprise a mouse user).
  const holdProps = isTablet
    ? (fn: () => void) => ({
        onPointerDown: () => {
          lastPointerRef.current = Date.now();
          fn();
          stopHold();
          holdRef.current = window.setInterval(fn, 650);
          window.addEventListener("pointerup", stopHold, { once: true });
        },
        onPointerLeave: stopHold,
        onPointerCancel: stopHold,
        onContextMenu: (e: { preventDefault: () => void }) => e.preventDefault(),
      })
    : (_fn: () => void) => ({});
  // Keyboard/screen-reader clicks still jog; ignore the click that follows a press.
  const clickJog = (axis: "x" | "y" | "z", distance: number) => () => {
    if (isTablet && Date.now() - lastPointerRef.current < 600) return;
    void jog(axis, distance);
  };

  const jogButton = (
    label: string,
    axis: "x" | "y" | "z",
    distance: number,
    children: ReactNode,
  ) => (
    <div {...holdProps(() => void jog(axis, distance))}>
      <IconButton
        label={label}
        icon={children}
        width="100%"
        isDisabled={busy}
        onClick={clickJog(axis, distance)}
      />
    </div>
  );

  return (
    <VStack gap={4}>
      {isTablet ? (
        <VStack gap={2}>
          <Text type="supporting" color="secondary">
            Step · mm (tap a direction, or press and hold)
          </Text>
          <HStack gap={2} wrap="wrap" className="jog-steps jog-steps--tablet">
            {STEPS_TABLET.map((s) => (
              <Button
                key={s}
                label={String(s)}
                size="lg"
                variant={step === s ? "primary" : "secondary"}
                onClick={() => setStep(s)}
              />
            ))}
          </HStack>
        </VStack>
      ) : (
        <HStack gap={3} align="end" wrap="wrap">
          <NumberInput
            label="Jog distance · mm"
            value={step}
            onChange={setStep}
            min={0.5}
            max={2000}
            step={1}
          />
          <HStack gap={1} wrap="wrap">
            {STEPS_DESKTOP.map((s) => (
              <Button
                key={s}
                label={s}
                variant={step === Number(s) ? "primary" : "secondary"}
                onClick={() => setStep(Number(s))}
              />
            ))}
          </HStack>
        </HStack>
      )}
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
        <div className={`jog-grid${isTablet ? " jog-grid--tablet" : ""}`}>
          <div className="jog-yplus">{jogButton(`Y +${step} mm`, "y", step, <ArrowUpIcon style={icon} />)}</div>
          <div className="jog-xmin">{jogButton(`X −${step} mm`, "x", -step, <ArrowLeftIcon style={icon} />)}</div>
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
          <div className="jog-xplus">{jogButton(`X +${step} mm`, "x", step, <ArrowRightIcon style={icon} />)}</div>
          <div className="jog-yminus">{jogButton(`Y −${step} mm`, "y", -step, <ArrowDownIcon style={icon} />)}</div>
          <div className="jog-zplus" {...holdProps(() => void jog("z", step))}>
            <Button
              label={isTablet ? `Z +${step}` : "Z+"}
              {...(isTablet ? { size: "lg" } : {})}
              width="100%"
              isDisabled={busy}
              onClick={clickJog("z", step)}
            />
          </div>
          <div className="jog-zminus" {...holdProps(() => void jog("z", -step))}>
            <Button
              label={isTablet ? `Z −${step}` : "Z−"}
              {...(isTablet ? { size: "lg" } : {})}
              width="100%"
              isDisabled={busy}
              onClick={clickJog("z", -step)}
            />
          </div>
        </div>
      </Center>
      {isTablet ? (
        <Text type="supporting" color="secondary">
          ↑ Y+ · ↓ Y− · ← X− · → X+
        </Text>
      ) : null}
      {showGoto ? (
        <VStack gap={2}>
          <Text type="supporting" color="secondary">
            Absolute target · mm
          </Text>
          <HStack gap={2} wrap="wrap" align="end">
            <NumberInput label="X" value={gx} onChange={setGx} />
            <NumberInput label="Y" value={gy} onChange={setGy} />
            <NumberInput label="Z" value={gz} onChange={setGz} />
            <Button
              label="Go to"
              variant="primary"
              {...(isTablet ? { size: "lg" } : {})}
              isDisabled={busy}
              onClick={() => void move(gx, gy, gz)}
            />
          </HStack>
        </VStack>
      ) : null}
    </VStack>
  );
}
