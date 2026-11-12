import { useCallback, useEffect, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { IconButton } from "@astryxdesign/core/IconButton";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Switch } from "@astryxdesign/core/Switch";
import { Text } from "@astryxdesign/core/Text";
import { TextInput } from "@astryxdesign/core/TextInput";
import { useToast } from "@astryxdesign/core/Toast";
import { api } from "../api";
import { PageHeader } from "../components/PageHeader";
import { PencilSquareIcon } from "@heroicons/react/24/outline";

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
  onRename,
}: {
  title: string;
  hint: string;
  path: string;
  onRename?: (label: string) => Promise<void>;
}) {
  const toast = useToast();
  const [state, setState] = useState<PlugState | null>(null);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState<string | null>(null);

  const saveName = async () => {
    if (!onRename || draft === null || !draft.trim()) return;
    try {
      await onRename(draft.trim());
      setDraft(null);
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Rename failed", type: "error" });
    }
  };

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
          {draft === null ? (
            <Text type="title">{title}</Text>
          ) : (
            <>
              <TextInput
                label="Name"
                isLabelHidden
                value={draft}
                onChange={setDraft}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void saveName();
                  if (e.key === "Escape") setDraft(null);
                }}
              />
              <Button label="Save" variant="primary" onClick={() => void saveName()} />
              <Button label="Cancel" variant="secondary" onClick={() => setDraft(null)} />
            </>
          )}
          {onRename && draft === null ? (
            <IconButton
              label="Rename"
              icon={<PencilSquareIcon style={{ width: 18, height: 18 }} />}
              variant="secondary"
              onClick={() => setDraft(title)}
            />
          ) : null}
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

type ToolInfo = { name: string; label: string; hint?: string | null };

export function IoPage() {
  const [tools, setTools] = useState<ToolInfo[]>([]);

  useEffect(() => {
    void api<ToolInfo[]>("/tools")
      .then((r) => setTools(Array.isArray(r) ? r : []))
      .catch(() => setTools([]));
  }, []);

  const rename = async (name: string, label: string) => {
    await api(`/tool/${encodeURIComponent(name)}/label`, {
      method: "PUT",
      body: JSON.stringify({ label }),
    });
    setTools((list) => list.map((t) => (t.name === name ? { ...t, label } : t)));
  };

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="I/O" />
      {tools.map((t) => (
        <PlugCard
          key={t.name}
          title={t.label}
          hint={t.hint || "RevPi on/off output. Stays on until you turn it off."}
          path={`/tool/${encodeURIComponent(t.name)}`}
          onRename={(label) => rename(t.name, label)}
        />
      ))}
    </VStack>
  );
}
