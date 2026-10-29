import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { TextInput } from "@astryxdesign/core/TextInput";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, fmt, postAction, refreshXyz } from "../api";
import { PageHeader } from "../components/PageHeader";

type Waypoint = { x: number; y: number; z: number };
type Script = { label: string; waypoints: Waypoint[] };

function wp(item: Waypoint): string {
  return `${fmt(item.x)}, ${fmt(item.y)}, ${fmt(item.z)}`;
}

function WaypointRow({
  index,
  point,
  busy,
  onChange,
  onRemove,
}: {
  index: number;
  point: Waypoint;
  busy: boolean;
  onChange: (point: Waypoint) => void;
  onRemove: () => void;
}) {
  const [x, setX] = useState(point.x);
  const [y, setY] = useState(point.y);
  const [z, setZ] = useState(point.z);

  const setAxis = (axis: keyof Waypoint, value: number) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return;
    const next = { x, y, z, [axis]: value };
    if (axis === "x") setX(value);
    if (axis === "y") setY(value);
    if (axis === "z") setZ(value);
    onChange(next);
  };

  return (
    <HStack gap={2} wrap align="end">
      <Text type="supporting">{index + 1}.</Text>
      <NumberInput label="X" value={x} onChange={(value) => setAxis("x", value)} step={1} />
      <NumberInput label="Y" value={y} onChange={(value) => setAxis("y", value)} step={1} />
      <NumberInput label="Z" value={z} onChange={(value) => setAxis("z", value)} step={1} />
      <Button label="Remove step" isDisabled={busy} onClick={onRemove} />
    </HStack>
  );
}

export function ScriptsPage({ onMoved }: { onMoved: () => Promise<void> }) {
  const toast = useToast();
  const [items, setItems] = useState<Script[]>([]);
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draftLabel, setDraftLabel] = useState("");

  const load = useCallback(async () => {
    const body = await api<{ scripts: Script[] }>("/scripts");
    setItems(body.scripts || []);
  }, []);

  useEffect(() => {
    void load().catch((error) =>
      toast({
        body: error instanceof Error ? error.message : "Could not load scripts",
        type: "error",
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const persist = async (next: Script[], ok?: string) => {
    setBusy(true);
    try {
      const body = await api<{ scripts: Script[] }>("/scripts", {
        method: "PUT",
        body: JSON.stringify({ scripts: next }),
      });
      setItems(body.scripts || next);
      if (ok) toast({ body: ok });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not save scripts",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const patchWaypoint = (label: string, index: number, point: Waypoint) => {
    setItems((prev) =>
      prev.map((row) =>
        row.label !== label
          ? row
          : {
              ...row,
              waypoints: row.waypoints.map((current, i) => (i === index ? point : current)),
            },
      ),
    );
  };

  const run = async (item: Script) => {
    if (!item.waypoints.length) return;
    if (editing) await persist(itemsRef.current);
    setBusy(true);
    try {
      await postAction("run_script", { name: item.label }, true);
      toast({ body: `Ran ${item.label}` });
      await onMoved();
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Script failed",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const addHere = async (item: Script) => {
    setBusy(true);
    try {
      const xyz = await refreshXyz();
      const next = itemsRef.current.map((row) =>
        row.label === item.label
          ? { ...row, waypoints: [...row.waypoints, xyz] }
          : row,
      );
      await persist(next, `Added ${fmt(xyz.x)}, ${fmt(xyz.y)}, ${fmt(xyz.z)} to ${item.label}`);
      await onMoved();
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not read pose",
        type: "error",
      });
      setBusy(false);
    }
  };

  return (
    <VStack gap={4}>
      <HStack justify="between" align="end" wrap>
        <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Scripts" />
        <Button
          label={editing ? "Done" : "Edit"}
          onClick={() => {
            if (editing) void persist(itemsRef.current, "Saved scripts").then(() => setEditing(false));
            else setEditing(true);
          }}
        />
      </HStack>
      <Text type="supporting" color="secondary">
        Ordered moves for tool pickup. In Edit you can type millimetres or Add
        current pose. The assistant runs these with run_script.
      </Text>
      {items.map((item) => (
        <Card key={item.label} padding={4}>
          <VStack gap={3}>
            <HStack justify="between" align="center" wrap>
              <Text weight="semibold">{item.label}</Text>
              <HStack gap={2} wrap>
                <Button
                  label="Run"
                  variant="primary"
                  isDisabled={busy || item.waypoints.length === 0}
                  onClick={() => void run(item)}
                />
                {editing ? (
                  <Button
                    label="Remove"
                    isDisabled={busy}
                    onClick={() =>
                      void persist(
                        itemsRef.current.filter((row) => row.label !== item.label),
                        `Removed ${item.label}`,
                      )
                    }
                  />
                ) : null}
              </HStack>
            </HStack>
            {item.waypoints.length ? (
              <VStack gap={2}>
                {item.waypoints.map((point, index) =>
                  editing ? (
                    <WaypointRow
                      key={`${item.label}-${index}`}
                      index={index}
                      point={point}
                      busy={busy}
                      onChange={(next) => patchWaypoint(item.label, index, next)}
                      onRemove={() =>
                        void persist(
                          itemsRef.current.map((row) =>
                            row.label === item.label
                              ? {
                                  ...row,
                                  waypoints: row.waypoints.filter((_, i) => i !== index),
                                }
                              : row,
                          ),
                          `Removed step ${index + 1}`,
                        )
                      }
                    />
                  ) : (
                    <Text key={`${item.label}-${index}`} type="supporting" color="secondary">
                      {index + 1}. {wp(point)}
                    </Text>
                  ),
                )}
              </VStack>
            ) : (
              <Text type="supporting" color="secondary">
                No waypoints yet
              </Text>
            )}
            {editing ? (
              <Button
                label="Add current pose"
                isDisabled={busy}
                onClick={() => void addHere(item)}
              />
            ) : null}
          </VStack>
        </Card>
      ))}
      {editing ? (
        <HStack gap={2} wrap align="end">
          <TextInput label="New script" value={draftLabel} onChange={setDraftLabel} />
          <Button
            label="Add"
            isDisabled={busy || !draftLabel.trim()}
            onClick={() => {
              const label = draftLabel.trim();
              if (itemsRef.current.some((row) => row.label.toLowerCase() === label.toLowerCase())) {
                toast({ body: `Script ${label} already exists`, type: "error" });
                return;
              }
              void persist([...itemsRef.current, { label, waypoints: [] }], `Added ${label}`).then(
                () => setDraftLabel(""),
              );
            }}
          />
        </HStack>
      ) : null}
    </VStack>
  );
}
