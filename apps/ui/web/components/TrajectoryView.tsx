import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { CodeBlock } from "@astryxdesign/core/CodeBlock";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { HStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { TextInput } from "@astryxdesign/core/TextInput";
import { ChatBubbleLeftRightIcon, ArrowDownTrayIcon, ClockIcon, CpuChipIcon, MagnifyingGlassIcon, WrenchIcon } from "@heroicons/react/24/outline";
import { api } from "../api";

type Rec = Record<string, any> & { seq: number; ts: string; type: string; thread_id: string };

type Row = {
  key: string;
  lane: "USER" | "ASSISTANT" | "TOOL" | "SYSTEM" | "CONTEXT";
  title: string;
  detail?: string;
  turn: number;
  record: Rec;
  running?: boolean;
  error?: boolean;
};

export type TrajectoryStats = { turns: number; steps: number; tokens: number; tokPerS: number | null };

function isError(record: Rec): boolean {
  const r = record.result;
  if (record.type === "tool_call") return !!(r && typeof r === "object" && (r.error || r.status === "error"));
  if (record.type === "run_end") return /error|max_/.test(String(record.stop_reason || ""));
  return false;
}

function oneLine(value: unknown): string {
  if (value == null) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
}

function coalesce(records: Rec[]): Rec[] {
  const out: Rec[] = [];
  for (const record of records) {
    const prev = out[out.length - 1];
    if (
      prev &&
      (record.type === "delta" || record.type === "thinking") &&
      prev.type === record.type &&
      prev.thread_id === record.thread_id
    ) {
      prev.content = `${prev.content || ""}${record.content || ""}`;
    } else {
      out.push({ ...record });
    }
  }
  return out;
}

function toRows(records: Rec[]): Row[] {
  const rows: Row[] = [];
  const finished = new Set(
    records.filter((r) => r.type === "tool_call" && r.id).map((r) => `${r.thread_id}:${r.id}`),
  );
  let turn = 0;
  for (const record of coalesce(records)) {
    const key = `${record.thread_id}:${record.seq}`;
    switch (record.type) {
      case "run_start":
        turn += 1;
        rows.push({
          key,
          lane: record.prompt ? "USER" : "SYSTEM",
          title: record.prompt || "Run resumed after approval",
          turn,
          record,
        });
        break;
      case "followup":
        rows.push({ key, lane: "USER", title: String(record.text || ""), turn, record });
        break;
      case "thinking":
      case "delta":
        if (String(record.content || "").trim())
          rows.push({ key, lane: "ASSISTANT", title: String(record.content), turn, record });
        break;
      case "tool_start":
        if (!finished.has(`${record.thread_id}:${record.id}`))
          rows.push({
            key,
            lane: "TOOL",
            title: `${record.name} ${oneLine(record.args)}`,
            detail: "running…",
            turn,
            record,
            running: true,
          });
        break;
      case "tool_call":
        rows.push({
          key,
          lane: "TOOL",
          title: `${record.name} ${oneLine(record.args)}`,
          detail: oneLine(record.result),
          turn,
          record,
          error: isError(record),
        });
        break;
      case "position":
        rows.push({
          key,
          lane: "CONTEXT",
          title: record.xyz
            ? `pose x ${Math.round(record.xyz.x)} · y ${Math.round(record.xyz.y)} · z ${Math.round(record.xyz.z)} mm`
            : "position updated",
          turn,
          record,
        });
        break;
      case "goal":
        rows.push({ key, lane: "CONTEXT", title: `goal ${oneLine(record.goal)}`, turn, record });
        break;
      case "retry":
        rows.push({
          key,
          lane: "SYSTEM",
          title: `Retry · ${String(record.content || "").replace(/\s+/g, " ")}`,
          turn,
          record,
          error: true,
        });
        break;
      case "approval":
        rows.push({
          key,
          lane: "SYSTEM",
          title: `Approval requested: ${(record.pending_approvals || []).map((a: any) => a.name).join(", ")}`,
          turn,
          record,
        });
        break;
      case "run_end":
        rows.push({
          key,
          lane: "SYSTEM",
          title: [
            `Run finished · ${record.stop_reason || "?"}`,
            record.metrics?.total_tokens != null ? `${record.metrics.total_tokens} tok` : null,
            record.metrics?.total_latency_s != null ? `${record.metrics.total_latency_s}s` : null,
          ]
            .filter(Boolean)
            .join(" · "),
          turn,
          record,
          error: isError(record),
        });
        break;
      default:
        break; // messages snapshots etc. stay out of the ledger
    }
  }
  return rows;
}

type Slot = { key: string; lane: 0 | 1 | 2; color: string; think?: number; out?: number };

const INPUT = { system: "var(--chart-gray)", user: "var(--chart-blue)", context: "var(--chart-aqua)" };

/** dsh-style lanes: one fixed-width block per step, laid out in order, turn dividers between. */
function toSlots(records: Rec[]) {
  const slots: Slot[] = [];
  const dividers: number[] = [];
  const finished = new Set(
    records.filter((r) => r.type === "tool_call" && r.id).map((r) => `${r.thread_id}:${r.id}`),
  );
  for (const record of coalesce(records)) {
    const key = `${record.thread_id}:${record.seq}`;
    const last = slots[slots.length - 1];
    const text = String(record.content || "").length;
    switch (record.type) {
      case "run_start":
        if (!record.resume && slots.length) dividers.push(slots.length);
        slots.push({ key, lane: 0, color: INPUT.system });
        if (record.prompt) slots.push({ key: `${key}:u`, lane: 0, color: INPUT.user });
        break;
      case "followup":
        slots.push({ key, lane: 0, color: INPUT.user });
        break;
      case "position":
      case "goal":
        slots.push({ key, lane: 0, color: INPUT.context });
        break;
      case "approval":
        slots.push({ key, lane: 0, color: INPUT.system });
        break;
      case "retry":
        slots.push({ key, lane: 0, color: "var(--chart-red)" });
        break;
      case "thinking":
      case "delta":
        if (!text) break;
        if (last?.lane === 1) {
          if (record.type === "thinking") last.think! += text;
          else last.out! += text;
        } else {
          slots.push({
            key,
            lane: 1,
            color: "var(--chart-violet)",
            think: record.type === "thinking" ? text : 0,
            out: record.type === "delta" ? text : 0,
          });
        }
        break;
      case "tool_start":
        if (!finished.has(`${record.thread_id}:${record.id}`))
          slots.push({ key, lane: 2, color: "color-mix(in srgb, var(--chart-orange) 50%, transparent)" });
        break;
      case "tool_call":
        slots.push({ key, lane: 2, color: "var(--chart-orange)" });
        break;
      default:
        break;
    }
  }
  return { slots, dividers };
}

function blockBackground(slot: Slot): string {
  if (slot.lane !== 1) return slot.color;
  const total = (slot.think || 0) + (slot.out || 0);
  if (!slot.think) return slot.color;
  const split = slot.out ? Math.min(75, Math.max(25, (slot.think / total) * 100)) : 100;
  const light = "color-mix(in srgb, var(--chart-violet) 45%, transparent)";
  return `linear-gradient(to right, ${light} 0 ${split}%, ${slot.color} ${split}% 100%)`;
}

function Lanes({ records }: { records: Rec[] }) {
  const { slots, dividers } = useMemo(() => toSlots(records), [records]);
  const unit = `min(${100 / Math.max(1, slots.length)}%, 72px)`;
  const at = (index: number) => `calc(${unit} * ${index})`;
  return (
    <div className="traj-lanes">
      {(["Input", "Model", "Tools"] as const).map((label, lane) => (
        <Fragment key={label}>
          <span>{label}</span>
          <div className="traj-track">
            {slots.map((slot, index) =>
              slot.lane === lane ? (
                <div
                  key={slot.key}
                  className="traj-block"
                  style={{ left: at(index), width: `calc(${unit} - 2px)`, background: blockBackground(slot) }}
                />
              ) : null,
            )}
            {lane === 0
              ? dividers.map((index) => (
                  <div key={index} className="traj-divider" style={{ left: `calc(${at(index)} - 1px)`, bottom: -40 }} />
                ))
              : null}
          </div>
        </Fragment>
      ))}
    </div>
  );
}

function Stat({ icon: Icon, label, value }: { icon: typeof ClockIcon; label: string; value: string | number }) {
  return (
    <HStack gap={1} vAlign="center">
      <Icon width={14} height={14} color="var(--color-text-secondary)" />
      <Text type="supporting" color="secondary">
        {label} {value}
      </Text>
    </HStack>
  );
}

export function TrajectoryView({
  threadIds,
  live,
  onStats,
}: {
  threadIds: string[];
  live: boolean;
  onStats?: (stats: TrajectoryStats) => void;
}) {
  const [records, setRecords] = useState<Rec[]>([]);
  const [query, setQuery] = useState("");
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const idsKey = threadIds.join(",");
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const load = useCallback(async () => {
    try {
      let ids = idsKey ? idsKey.split(",") : [];
      if (!ids.length) {
        const list = await api<{ trajectories?: { thread_id: string }[] }>("/trajectories?limit=1");
        ids = list.trajectories?.[0] ? [list.trajectories[0].thread_id] : [];
      }
      const parts = await Promise.all(
        ids.map((id) =>
          api<{ records?: Rec[] }>(`/trajectories/${encodeURIComponent(id)}`).catch(() => null),
        ),
      );
      const merged = parts
        .flatMap((part) => part?.records || [])
        .sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts));
      setRecords(merged);
    } catch {
      setRecords([]);
    } finally {
      setLoaded(true);
    }
  }, [idsKey]);

  useEffect(() => {
    void load();
  }, [load]);

  const running = live || (records.length > 0 && !records.some((r) => r.type === "run_end"));
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => void load(), 2000);
    return () => window.clearInterval(id);
  }, [running, load]);

  useEffect(() => {
    if (running) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [records.length, running]);

  const rows = useMemo(() => toRows(records), [records]);
  const visible = query
    ? rows.filter((row) =>
        `${row.lane} ${row.title} ${row.detail || ""}`.toLowerCase().includes(query.toLowerCase()),
      )
    : rows;

  const stats = useMemo(() => {
    const ends = records.filter((r) => r.type === "run_end");
    const sum = (field: string) => ends.reduce((n, r) => n + (Number(r.metrics?.[field]) || 0), 0);
    const llmS = sum("llm_latency_s");
    const t0 = records.length ? Date.parse(records[0].ts) : 0;
    const t1 = records.length ? Date.parse(records[records.length - 1].ts) : 0;
    return {
      durationS: (t1 - t0) / 1000,
      turns: records.filter((r) => r.type === "run_start" && !r.resume).length,
      steps: toSlots(records).slots.filter((slot) => slot.lane === 1).length,
      calls: records.filter((r) => r.type === "tool_call").length,
      tokens: sum("total_tokens"),
      tokPerS: llmS ? Math.round(sum("completion_tokens") / llmS) : null,
    };
  }, [records]);

  useEffect(() => {
    onStats?.({ turns: stats.turns, steps: stats.steps, tokens: stats.tokens, tokPerS: stats.tokPerS });
  }, [stats, onStats]);

  if (loaded && !records.length) {
    return (
      <EmptyState
        title="No trajectory yet"
        description="Run the assistant once — every prompt, thinking step, tool call, pose, and approval is logged here as it happens."
      />
    );
  }

  const exportJsonl = () => {
    const url = URL.createObjectURL(
      new Blob([records.map((r) => JSON.stringify(r)).join("\n") + "\n"], { type: "application/x-ndjson" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `trajectory-${records[0]?.thread_id || "run"}.jsonl`;
    link.click();
    URL.revokeObjectURL(url);
  };

  let lastTurn = 0;
  return (
    <div style={{ flex: 1, minHeight: 0, overflowY: "auto", width: "100%" }}>
      <HStack gap={4} vAlign="center" style={{ padding: "6px 12px", borderBottom: "1px solid var(--color-border)" }}>
        <Stat icon={ClockIcon} label="Duration" value={`${stats.durationS.toFixed(1)}s`} />
        <Stat icon={ChatBubbleLeftRightIcon} label="Turns" value={stats.turns} />
        <Stat icon={CpuChipIcon} label="Steps" value={stats.steps} />
        <Stat icon={WrenchIcon} label="Calls" value={stats.calls} />
        <div style={{ flex: 1 }} />
        <TextInput label="Search" isLabelHidden placeholder="Search" startIcon={MagnifyingGlassIcon} size="sm" width={240} value={query} onChange={setQuery} />
        <Button label="Export" size="sm" icon={<ArrowDownTrayIcon width={14} height={14} />} isDisabled={!records.length} onClick={exportJsonl} />
      </HStack>

      <Lanes records={records} />

      <div>
        {visible.map((row) => {
          const turnStart = row.turn !== lastTurn && (row.lane === "USER" || row.lane === "SYSTEM");
          lastTurn = row.turn;
          const open = openKey === row.key;
          const dot = row.error || row.lane === "ASSISTANT";
          return (
            <Fragment key={row.key}>
              <div
                className="traj-row"
                data-open={open || undefined}
                data-turn={turnStart || undefined}
                onClick={() => setOpenKey(open ? null : row.key)}
              >
                <div className="traj-gutter">
                  {turnStart ? <span className="traj-turn">Turn {row.turn}</span> : null}
                  {dot ? <span className="traj-dot" data-error={row.error || undefined} /> : null}
                </div>
                <div>
                  <span className="traj-chip" data-role={row.lane}>
                    {row.lane}
                  </span>
                </div>
                {row.lane === "TOOL" ? (
                  <div className="traj-tool traj-mono">
                    <span className="traj-text">{row.title}</span>
                    <span className="traj-text">→ {row.detail}</span>
                  </div>
                ) : (
                  <span className="traj-text">{row.title}</span>
                )}
              </div>
              {open ? (
                <div style={{ padding: "8px 12px 8px 188px" }}>
                  <CodeBlock
                    code={JSON.stringify(row.record, null, 2)}
                    language="json"
                    size="sm"
                    container="section"
                    width="100%"
                    isWrapped
                    isCollapsible
                    maxHeight={320}
                    hasCopyButton
                  />
                </div>
              ) : null}
            </Fragment>
          );
        })}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}
