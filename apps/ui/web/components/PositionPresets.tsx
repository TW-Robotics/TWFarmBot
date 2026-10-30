import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { Selector } from "@astryxdesign/core/Selector";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { TextInput } from "@astryxdesign/core/TextInput";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt, postAction, refreshXyz } from "../api";
import { useDevice } from "../device";

export type NamedPosition = {
  label: string;
  group: string;
  x?: number | null;
  y?: number | null;
  z?: number | null;
  set?: boolean;
};

const GROUPS = ["Travel", "Tools", "Camera"];

function isSet(item: NamedPosition): boolean {
  return item.set !== false && item.x != null && item.y != null && item.z != null;
}

function coords(item: NamedPosition): string {
  if (!isSet(item)) return "not set";
  return `${fmt(item.x)}, ${fmt(item.y)}, ${fmt(item.z)}`;
}

function payloadOf(items: NamedPosition[]): NamedPosition[] {
  return items.map((item) =>
    isSet(item)
      ? { label: item.label, group: item.group || "Travel", x: item.x, y: item.y, z: item.z }
      : { label: item.label, group: item.group || "Travel" },
  );
}

export function PositionPresets({ onMoved }: { onMoved: () => Promise<void> }) {
  const toast = useToast();
  const { isTablet } = useDevice();
  const [items, setItems] = useState<NamedPosition[]>([]);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draftLabel, setDraftLabel] = useState("");
  const [draftGroup, setDraftGroup] = useState("Tools");

  const load = useCallback(async () => {
    const body = await api<{ positions: NamedPosition[] }>("/positions");
    setItems(body.positions || []);
  }, []);

  useEffect(() => {
    void load().catch((error) =>
      toast({
        body: error instanceof Error ? error.message : "Could not load presets",
        type: "error",
      }),
    );
    // Load once on mount. toast is not a dep — it changes every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const persist = async (next: NamedPosition[], ok: string) => {
    setBusy(true);
    try {
      const body = await api<{ positions: NamedPosition[] }>("/positions", {
        method: "PUT",
        body: JSON.stringify({ positions: payloadOf(next) }),
      });
      setItems(body.positions || next);
      toast({ body: ok });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not save presets",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const go = async (item: NamedPosition) => {
    if (!isSet(item)) return;
    setBusy(true);
    try {
      await postAction("goto_named", { name: item.label }, true);
      toast({ body: `At ${item.label}` });
      await onMoved();
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Move failed",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const saveHere = async (item: NamedPosition) => {
    setBusy(true);
    try {
      const xyz = await refreshXyz();
      const next = items.map((row) =>
        row.label === item.label ? { ...row, ...xyz, set: true } : row,
      );
      const body = await api<{ positions: NamedPosition[] }>("/positions", {
        method: "PUT",
        body: JSON.stringify({ positions: payloadOf(next) }),
      });
      setItems(body.positions || next);
      toast({
        body: `Saved ${item.label} at ${fmt(xyz.x)}, ${fmt(xyz.y)}, ${fmt(xyz.z)}`,
      });
      await onMoved();
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not save pose",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const groups = useMemo(() => {
    const extra = items.map((item) => item.group).filter(Boolean);
    return [...new Set([...GROUPS, ...extra])];
  }, [items]);

  const add = () => {
    const label = draftLabel.trim();
    if (!label) return;
    if (items.some((item) => item.label.toLowerCase() === label.toLowerCase())) {
      toast({ body: `Preset ${label} already exists`, type: "error" });
      return;
    }
    void persist([...items, { label, group: draftGroup, set: false }], `Added ${label}`).then(
      () => setDraftLabel(""),
    );
  };

  const body = (
    <VStack gap={3}>
      <HStack justify="between" align="center" wrap>
        <Text weight="semibold">Named positions</Text>
        <Button
          label={editing ? "Done" : "Edit"}
          onClick={() => setEditing((value) => !value)}
        />
      </HStack>
      {groups.map((group) => {
        const rows = items.filter((item) => (item.group || "Travel") === group);
        if (!rows.length) return null;
        return (
          <VStack key={group} gap={2}>
            <Text type="supporting">{group}</Text>
            {editing ? (
              rows.map((item) => (
                <HStack key={item.label} gap={2} wrap align="center">
                  <Text>{item.label}</Text>
                  <Text type="supporting" color="secondary">
                    {coords(item)}
                  </Text>
                  <Button
                    label="Save here"
                    isDisabled={busy}
                    onClick={() => void saveHere(item)}
                  />
                  <Button
                    label="Remove"
                    isDisabled={busy}
                    onClick={() =>
                      void persist(
                        items.filter((row) => row.label !== item.label),
                        `Removed ${item.label}`,
                      )
                    }
                  />
                </HStack>
              ))
            ) : (
              <HStack gap={2} wrap>
                {rows.map((item) => (
                  <Button
                    key={item.label}
                    label={isSet(item) ? item.label : `${item.label} · unset`}
                    variant={isSet(item) ? "primary" : "secondary"}
                    isDisabled={busy || !isSet(item)}
                    onClick={() => void go(item)}
                  />
                ))}
              </HStack>
            )}
          </VStack>
        );
      })}
      {editing ? (
        <HStack gap={2} wrap align="end">
          <TextInput label="New preset" value={draftLabel} onChange={setDraftLabel} />
          <Selector
            label="Group"
            options={GROUPS}
            value={draftGroup}
            onChange={setDraftGroup}
          />
          <Button label="Add" isDisabled={busy || !draftLabel.trim()} onClick={add} />
        </HStack>
      ) : isTablet ? null : (
        <Text type="supporting" color="secondary">
          Edit → jog to a tool bay → Save here. That pose is what the assistant uses.
        </Text>
      )}
    </VStack>
  );
  // Tablet: flat on the page, no card chrome.
  return isTablet ? body : <Card padding={4}>{body}</Card>;
}
