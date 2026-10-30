import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { Divider } from "@astryxdesign/core/Divider";
import { Grid } from "@astryxdesign/core/Grid";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { useToast } from "@astryxdesign/core/Toast";
import { fmt, postAction } from "../api";
import { JogPad } from "../components/JogPad";
import { PositionPresets } from "../components/PositionPresets";
import { Metric, PageHeader } from "../components/PageHeader";
import { useDevice } from "../device";
import type { Pose } from "../components/GantryMap";

export function MotionPage({
  pose,
  onMoved,
}: {
  pose: Pose;
  onMoved: () => Promise<void>;
}) {
  const toast = useToast();
  const { isTablet } = useDevice();

  const run = (kind: string, params: Record<string, unknown>, ok: string) =>
    void postAction(kind, params, true)
      .then(() => toast({ body: ok }))
      .then(() => onMoved())
      .catch((error) => toast({ body: error instanceof Error ? error.message : String(error), type: "error" }));

  const stop = () =>
    void postAction("e_stop")
      .then(() => toast({ body: "STOP sent" }))
      .catch((e) => toast({ body: e instanceof Error ? e.message : "STOP failed", type: "error" }));

  // Tablet: STOP + jog + presets, nothing else. No metric cards, no bed map
  // (tap-to-move proved too easy to fat-finger next to a live gantry), no
  // extra rows — everything must fit without hunting through cards.
  if (isTablet) {
    return (
      <VStack gap={3}>
        <Button label="STOP" variant="primary" size="lg" width="100%" onClick={stop} />
        <JogPad pose={pose} onMoved={onMoved} />
        <Divider />
        <PositionPresets onMoved={onMoved} />
      </VStack>
    );
  }

  return (
    <VStack gap={4}>
      <HStack justify="between" align="end" wrap="wrap">
        <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Motion" />
        <HStack gap={2} wrap="wrap">
          <Button label="Unlock" onClick={() => run("unlock", {}, "Unlocked")} />
          <Button
            label="Find home"
            variant="primary"
            onClick={() => run("find_home", { axis: "all" }, "Homing complete")}
          />
        </HStack>
      </HStack>
      <Grid columns={3} gap={3}>
        <Metric label="X · mm" value={fmt(pose.x)} />
        <Metric label="Y · mm" value={fmt(pose.y)} />
        <Metric label="Z · mm" value={fmt(pose.z)} />
      </Grid>
      <Card padding={4}>
        <JogPad pose={pose} onMoved={onMoved} />
      </Card>
      <PositionPresets onMoved={onMoved} />
      <HStack gap={2} wrap="wrap">
        <Button label="Home X" onClick={() => run("find_home", { axis: "x" }, "X homed")} />
        <Button label="Home Y" onClick={() => run("find_home", { axis: "y" }, "Y homed")} />
        <Button label="Home Z" onClick={() => run("find_home", { axis: "z" }, "Z homed")} />
        <Button label="Photo here" onClick={() => run("take_photo", {}, "Photo queued")} />
      </HStack>
    </VStack>
  );
}
