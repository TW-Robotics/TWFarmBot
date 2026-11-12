import { useCallback, useEffect, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Switch } from "@astryxdesign/core/Switch";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api } from "../api";
import { PageHeader } from "../components/PageHeader";

type PlugState = {
  on?: boolean | null;
  power_w?: number | null;
  host?: string;
  backend?: string;
};

function PlugCard({
  title,
  hint,
  path,
}: {
  title: string;
  hint: string;
  path: "/tool";
}) {
  const toast = useToast();
  const [state, setState] = useState<PlugState | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    // Background poll: never toast. A dead backend just shows "Unknown"
    // until it answers. Manual toggles below still report errors.
    try {
      setState(await api<PlugState>(path));
    } catch {
      setState(null);
    }
  }, [path]);

  useEffect(() => {
    void load();
    const id = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(id);
  }, [load]);

  const setOn = async (on: boolean) => {
    setBusy(true);
    try {
      setState(await api<PlugState>(path, { method: "POST", body: JSON.stringify({ on }) }));
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : `${title} command failed`,
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const on = Boolean(state?.on);
  const watts = typeof state?.power_w === "number" ? `${Math.round(state.power_w)} W` : null;

  return (
    <Card padding={4}>
      <VStack gap={3}>
        <HStack gap={3} vAlign="center" wrap="wrap">
          <Text type="title">{title}</Text>
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
          {hint}
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
  );
}

export function IoPage() {
  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="I/O" />
      <PlugCard
        title="Tool head"
        hint="RevPi Modbus register 0 — simple on/off end effector. Stays on until you turn it off."
        path="/tool"
      />
    </VStack>
  );
}
