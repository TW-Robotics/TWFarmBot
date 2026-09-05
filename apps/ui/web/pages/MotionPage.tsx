import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { Grid } from "@astryxdesign/core/Grid";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { useToast } from "@astryxdesign/core/Toast";
import { fmt, postAction } from "../api";
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

  const run = (kind: string, params: Record<string, unknown>, ok: string) =>
    void postAction(kind, params, true)
      .then(() => toast({ body: ok }))
      .then(() => onMoved())
      .catch((error) => toast({ body: error instanceof Error ? error.message : String(error), type: "error" }));

  return (
    <VStack gap={4}>
      <HStack justify="between" align="end" wrap>
        <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Motion" />
        <HStack gap={2} wrap>
          <Button label="Unlock" onClick={() => run("unlock", {}, "Unlocked")} />
          <Button label="Find home" variant="primary" onClick={() => run("find_home", { axis: "all" }, "Homing complete")} />
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
      <HStack gap={2} wrap>
        <Button label="Home X" onClick={() => run("find_home", { axis: "x" }, "X homed")} />
        <Button label="Home Y" onClick={() => run("find_home", { axis: "y" }, "Y homed")} />
        <Button label="Home Z" onClick={() => run("find_home", { axis: "z" }, "Z homed")} />
        <Button label="Photo here" onClick={() => run("take_photo", {}, "Photo queued")} />
      </HStack>
    </VStack>
  );
}
