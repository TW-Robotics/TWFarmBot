import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Button } from "@astryxdesign/core/Button";
import { ClickableCard } from "@astryxdesign/core/ClickableCard";
import { DateInput } from "@astryxdesign/core/DateInput";
import { Dialog, DialogHeader } from "@astryxdesign/core/Dialog";
import { ProgressBar } from "@astryxdesign/core/ProgressBar";
import { Selector } from "@astryxdesign/core/Selector";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { TextInput } from "@astryxdesign/core/TextInput";
import { useToast } from "@astryxdesign/core/Toast";
import { api } from "../api";
import { CROPS, cropByName, gddSince, recentDli, type DailyPoint } from "../agronomy";
import { GantryMap, type Area, type GardenCell, type Pose, type Rect, type Workspace } from "../components/GantryMap";
import { Metric, PageHeader } from "../components/PageHeader";

type Planting = Rect & { id: string; crop: string; planted: string; note: string };
/** The dialog edits an existing planting (has `id`) or adds one on a freshly drawn area. */
type Draft = Rect & { id?: string; crop: string; planted: string; note: string };
type History = { hourly: DailyPoint[] };

const DAY = 86400;
const today = () => new Date().toISOString().slice(0, 10);
const startOf = (iso: string) => new Date(`${iso}T00:00:00`).getTime() / 1000;
const size = (r: Rect) => `${Math.round(r.x2 - r.x1)} × ${Math.round(r.y2 - r.y1)} mm`;

/** Daily points for the last year: archive (year) up to the recent month, which is fresher. */
function mergeDaily(year: DailyPoint[], month: DailyPoint[]): DailyPoint[] {
  const cut = month[0]?.t ?? Infinity;
  return [...year.filter((p) => p.t < cut), ...month];
}

/** Progress to harvest and light status for one planting, from outdoor weather as a proxy. */
function growth(p: Draft, daily: DailyPoint[]) {
  const crop = cropByName(p.crop);
  if (!crop || !daily.length || !p.planted) return null;
  const plantedT = startOf(p.planted);
  const days = Math.max(0, Math.floor((Date.now() / 1000 - plantedT) / DAY));
  const gdd = gddSince(daily, crop.baseC, plantedT);
  const progress = Math.min(1, gdd / crop.gddToHarvest);
  const recent = daily.slice(-14).filter((d) => d.temperature_c != null);
  const rate = recent.length
    ? recent.reduce((s, d) => s + Math.max(0, (d.temperature_c as number) - crop.baseC), 0) / recent.length
    : 0;
  const daysLeft = progress >= 1 ? 0 : rate > 0 ? Math.ceil((crop.gddToHarvest - gdd) / rate) : null;
  const dli = recentDli(daily);
  const light: "low" | "ok" | "good" | null = dli == null ? null : dli < crop.dliMin ? "low" : dli < crop.dliIdeal ? "ok" : "good";
  return { crop, days, gdd, progress, daysLeft, dli, light };
}

const SOIL_ROLE = {
  heavy: "Hungry plant: uses lots of nutrients (nitrogen)",
  medium: "Uses a moderate amount of nutrients",
  light: "Undemanding: uses few nutrients",
  fixer: "Adds nitrogen to the soil",
} as const;

const LIGHT_WORD = { low: "Too low", ok: "OK", good: "Good" } as const;

/** Plain-language read of the camera's NDRE vegetation index. */
function plantHealth(ndre: number | null): string {
  if (ndre == null) return "No scan yet";
  return ndre < 0.15 ? "Weak signal" : ndre < 0.3 ? "Fair" : "Healthy";
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="planting-section">
      <Text weight="semibold">{title}</Text>
      {children}
    </div>
  );
}

function Neighbours({ good, bad }: { good: string[]; bad: string[] }) {
  const list = (title: string, items: string[], tone: "success" | "warning") => (
    <VStack gap={2}>
      <Text type="supporting" color="secondary">
        {title}
      </Text>
      <HStack gap={2} wrap="wrap">
        {items.map((item) => (
          <Badge key={item} variant={tone} label={item} />
        ))}
      </HStack>
    </VStack>
  );
  return (
    <div className="planting-two">
      {list("Plant next to", good, "success")}
      {list("Keep away from", bad, "warning")}
    </div>
  );
}

function PlantingDialog({
  draft,
  daily,
  cells,
  cellMm,
  onChange,
  onSave,
  onRemove,
  onClose,
}: {
  draft: Draft | null;
  daily: DailyPoint[];
  cells: GardenCell[];
  cellMm: number;
  onChange: (draft: Draft) => void;
  onSave: () => void;
  onRemove: () => void;
  onClose: () => void;
}) {
  const info = useMemo(() => (draft ? growth(draft, daily) : null), [draft, daily]);
  // Mean NDRE of the scanned cells inside the area (a margin of half a cell counts).
  const ndre = useMemo(() => {
    if (!draft) return null;
    const pad = cellMm / 2;
    const values = cells
      .filter(
        (c) =>
          c.x != null && c.y != null && c.ndre_mean != null &&
          c.x >= draft.x1 - pad && c.x <= draft.x2 + pad &&
          c.y >= draft.y1 - pad && c.y <= draft.y2 + pad,
      )
      .map((c) => c.ndre_mean as number);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  }, [draft, cells, cellMm]);

  const editing = Boolean(draft?.id);
  const facts = draft ? cropByName(draft.crop)?.facts : undefined;
  const care: [string, string][] = facts
    ? [
        ["Soil", SOIL_ROLE[facts.feeder]],
        ["Sun", facts.sun],
        ["Water", facts.water],
        ["Spacing", `${facts.spacingCm} cm apart`],
        ["Soil pH", facts.ph],
        ["Attracts", facts.attracts.join(", ")],
        ["Watch for", facts.pests.join(", ")],
      ]
    : [];

  return (
    <Dialog isOpen={draft !== null} onOpenChange={(open) => !open && onClose()} width={900} maxHeight="90dvh" purpose="info">
      {draft ? (
        <VStack gap={5}>
          <DialogHeader
            title={editing ? draft.crop : "Add planting"}
            subtitle={`Area ${size(draft)}`}
            onOpenChange={(open) => !open && onClose()}
          />
          <div className="planting-pinned">
            <div className="planting-fields">
              <Selector label="Crop" options={CROPS.map((c) => c.name)} value={draft.crop} onChange={(crop) => onChange({ ...draft, crop })} />
              <DateInput label="Planted" value={draft.planted as never} onChange={(v) => v && onChange({ ...draft, planted: String(v) })} />
              <TextInput label="Note" value={draft.note} onChange={(note) => onChange({ ...draft, note })} />
            </div>
          <HStack gap={2}>
            {editing ? <Button label="Remove" variant="secondary" onClick={onRemove} /> : null}
            <div style={{ flex: 1 }} />
            <Button label="Cancel" variant="secondary" onClick={onClose} />
            <Button
              label={editing ? "Save" : "Add planting"}
              variant="primary"
              isDisabled={!/^\d{4}-\d{2}-\d{2}$/.test(draft.planted)}
              onClick={onSave}
            />
          </HStack>
          </div>
          <div className="planting-body">
          {info ? (
            <Section title="How it's doing">
              <div className="planting-stats">
                <Metric label="Planted" value={`${info.days} days ago`} />
                <Metric label="Grown" value={`${Math.round(info.progress * 100)}%`} />
                <Metric
                  label="Harvest in"
                  value={info.daysLeft === 0 ? "Ready" : info.daysLeft == null ? "Not growing" : `~${info.daysLeft} days`}
                />
              </div>
              <ProgressBar label="Progress to harvest" isLabelHidden value={info.progress} max={1} />
              <div className="planting-stats">
                <Metric label="Light" value={info.light ? LIGHT_WORD[info.light] : "—"} />
                <Metric label="Plant health" value={plantHealth(ndre)} />
              </div>
              <Text type="supporting" color="secondary">
                Light: gets about {info.dli?.toFixed(0)} of the {info.crop.dliMin}–{info.crop.dliIdeal} this crop likes (mol/m² a
                day, outdoor estimate behind glass).
                {ndre != null ? ` Plant health comes from the camera scan (NDRE ${ndre.toFixed(2)}).` : ""}
              </Text>
            </Section>
          ) : null}
          {facts ? (
            <>
              <Section title="Neighbours">
                <Neighbours good={facts.companions} bad={facts.avoid} />
              </Section>
              <Section title="Care">
                <div className="planting-care">
                  {care.map(([label, value]) => (
                    <div key={label} className="planting-care-row">
                      <Text type="supporting" color="secondary">
                        {label}
                      </Text>
                      <Text>{value}</Text>
                    </div>
                  ))}
                </div>
              </Section>
            </>
          ) : null}
          </div>
        </VStack>
      ) : null}
    </Dialog>
  );
}

export function GardenPage({
  pose,
  workspace,
  cells,
  cellMm,
}: {
  pose: Pose;
  workspace: Workspace | null;
  cells: GardenCell[];
  cellMm: number;
}) {
  const toast = useToast();
  const [plantings, setPlantings] = useState<Planting[]>([]);
  const [daily, setDaily] = useState<DailyPoint[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);

  useEffect(() => {
    const fail = (error: unknown) =>
      toast({ body: error instanceof Error ? error.message : "Request failed", type: "error" });
    api<{ plantings: Planting[] }>("/plantings").then((r) => setPlantings(r.plantings), fail);
    Promise.all([
      api<History>("/meteorology/history?range=year"),
      api<History>("/meteorology/history?range=month"),
    ]).then(([year, month]) => setDaily(mergeDaily(year.hourly, month.hourly)), () => setDaily([]));
  }, [toast]);

  const persist = useCallback(
    async (next: Planting[]) => {
      try {
        const r = await api<{ plantings: Planting[] }>("/plantings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ plantings: next }),
        });
        setPlantings(r.plantings);
        return true;
      } catch (error) {
        toast({ body: error instanceof Error ? error.message : "Could not save", type: "error" });
        return false;
      }
    },
    [toast],
  );

  const save = async () => {
    if (!draft) return;
    const { id, ...fields } = draft;
    const next = id
      ? plantings.map((p) => (p.id === id ? { ...p, ...fields } : p))
      : [...plantings, { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, ...fields }];
    if (await persist(next)) setDraft(null);
  };

  const remove = async () => {
    if (draft?.id && (await persist(plantings.filter((p) => p.id !== draft.id)))) setDraft(null);
  };

  const sorted = useMemo(() => [...plantings].sort((a, b) => a.planted.localeCompare(b.planted)), [plantings]);
  const areas: Area[] = [
    ...plantings.map((p) => ({ ...p, label: p.crop })),
    ...(draft && !draft.id ? [{ ...draft, label: draft.crop, pending: true }] : []),
  ];

  return (
    <div className="garden-page">
      <PageHeader kicker="Garden" title="What's planted where" />
      <div className="garden-layout">
        <div className="garden-map">
          <GantryMap
            pose={pose}
            workspace={workspace}
            areas={areas}
            onArea={(rect) => setDraft({ ...rect, crop: CROPS[0].name, planted: today(), note: "" })}
            onAreaClick={(id) => {
              const p = plantings.find((x) => x.id === id);
              if (p) setDraft(p);
            }}
          />
          <Text type="supporting" color="secondary">
            Drag to outline a new area, or tap an area to see and edit it.
          </Text>
        </div>
        <div className="garden-list">
          {sorted.length ? (
            sorted.map((p) => {
              const g = growth(p, daily);
              return (
                <ClickableCard key={p.id} label={`Edit ${p.crop}`} padding={3} onClick={() => setDraft(p)}>
                  <VStack gap={2}>
                    <HStack gap={2} vAlign="center">
                      <Text weight="semibold">{p.crop}</Text>
                      <div style={{ flex: 1 }}>
                        <Text type="supporting" color="secondary">
                          {g ? `${g.days} days` : `planted ${p.planted}`} · {size(p)}
                          {p.note ? ` · ${p.note}` : ""}
                        </Text>
                      </div>
                      {g?.light === "low" ? <Badge variant="warning" label="Low light" /> : null}
                    </HStack>
                    {g ? (
                      <ProgressBar
                        label={`${p.crop} progress`}
                        isLabelHidden
                        value={g.progress}
                        max={1}
                        hasValueLabel
                        formatValueLabel={(v) => `${Math.round(v * 100)}%`}
                      />
                    ) : null}
                  </VStack>
                </ClickableCard>
              );
            })
          ) : (
            <Text color="secondary">Nothing planted yet. Drag on the bed to outline your first area.</Text>
          )}
        </div>
      </div>
      <PlantingDialog
        draft={draft}
        daily={daily}
        cells={cells}
        cellMm={cellMm}
        onChange={setDraft}
        onSave={() => void save()}
        onRemove={() => void remove()}
        onClose={() => setDraft(null)}
      />
    </div>
  );
}
