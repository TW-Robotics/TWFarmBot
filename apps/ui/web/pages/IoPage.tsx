import { useCallback, useEffect, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Switch } from "@astryxdesign/core/Switch";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api } from "../api";
import { PageHeader } from "../components/PageHeader";

type WateringState = {
  on?: boolean | null;
  power_w?: number | null;
  host?: string;
  backend?: string;
};

export function IoPage() {
  const toast = useToast();
  const [state, setState] = useState<WateringState | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setState(await api<WateringState>("/watering"));
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not read the pump",
        type: "error",
      });
    }
  }, [toast]);

  useEffect(() => {
    void load();
    const id = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(id);
  }, [load]);

  const setOn = async (on: boolean) => {
    setBusy(true);
    try {
      setState(await api<WateringState>("/watering", { method: "POST", body: JSON.stringify({ on }) }));
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Pump command failed",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const on = Boolean(state?.on);
  const watts = typeof state?.power_w === "number" ? `${Math.round(state.power_w)} W` : null;

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="I/O" />
      <Card padding={4}>
        <VStack gap={3}>
          <HStack gap={3} vAlign="center" wrap="wrap">
            <Text type="title">Bed pump</Text>
            <Badge
              label={state?.on ? "On" : state?.on === false ? "Off" : "Unknown"}
              variant={state?.on ? "success" : "neutral"}
            />
            {watts ? (
              <Text type="supporting" color="secondary">
                {watts}
              </Text>
            ) : null}
          </HStack>
          <Text type="supporting" color="secondary">
            One Shelly plug for every sprayer. Stays on until you turn it off (or ESTOP).
          </Text>
          <Switch
            label={on ? "On" : "Off"}
            value={on}
            isDisabled={busy}
            isLoading={busy}
            onChange={(checked) => void setOn(checked)}
          />
        </VStack>
      </Card>
    </VStack>
  );
}
