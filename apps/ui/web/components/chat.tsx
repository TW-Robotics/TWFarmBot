/**
 * Rendering building blocks for the agent chat (Assistant page).
 *
 * Visual hierarchy, most → least prominent:
 *   1. Assistant prose (Markdown, flush with tool rows / thumbnails)
 *   2. Tool activity (grouped ChatToolCalls rows + thumbnail grid)
 *   3. Reasoning / execution trace (muted Collapsibles, closed by default)
 *   4. Message metadata (small secondary timestamp · model row)
 *
 * Everything is an Astryx component so colors, scale and spacing stay on
 * theme tokens — no ad-hoc hex colors or font sizes.
 */
import { useState } from "react";
import {
  ChatMessageMetadata,
  ChatToolCalls,
  type ChatToolCallItem,
  type ChatToolCallStatus,
} from "@astryxdesign/core/Chat";
import { Markdown } from "@astryxdesign/core/Markdown";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { CodeBlock } from "@astryxdesign/core/CodeBlock";
import { Collapsible } from "@astryxdesign/core/Collapsible";
import { Lightbox } from "@astryxdesign/core/Lightbox";
import { Link } from "@astryxdesign/core/Link";
import { Link } from "@astryxdesign/core/Link";
import { Spinner } from "@astryxdesign/core/Spinner";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { Thumbnail } from "@astryxdesign/core/Thumbnail";
import { Timestamp } from "@astryxdesign/core/Timestamp";
import { InformationCircleIcon, SparklesIcon } from "@heroicons/react/24/outline";
import { apiUrl } from "../api";

// ---------------------------------------------------------------------------
// Shared chat types (used by AssistantPage for state + streaming)
// ---------------------------------------------------------------------------

export type ToolImage = { label: string; url: string };

export type ProgramItem = {
  call_id?: string;
  code?: string;
  result?: unknown;
  status?: string;
};

export type ToolTranscriptItem = {
  kind: "tool";
  /** Provider tool-call id; matches tool_start → tool_call events. */
  id?: string;
  name: string;
  args?: unknown;
  /** Undefined while the tool is still running (row shows a spinner). */
  result?: unknown;
  images?: ToolImage[];
};

export type TranscriptItem =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string }
  | ToolTranscriptItem;

export type ChatMsg = {
  role: "user" | "assistant";
  content: string;
  ts?: number;
  thinking?: string;
  tool_calls?: {
    id?: string;
    name: string;
    args?: any;
    result?: any;
    images?: ToolImage[];
    caller?: { type?: string } | null;
  }[];
  programs?: ProgramItem[];
  proposed_actions?: { kind: string; params?: Record<string, unknown> }[];
  streaming?: boolean;
  error?: string;
  approved?: boolean;
  rejected?: boolean;
  executing?: boolean;
  weave_url?: string;
  metrics?: {
    total_latency_s?: number;
    llm_latency_s?: number;
    ttft_s?: number;
    tokens_per_s?: number | null;
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
    total_tokens?: number | null;
    turn_count?: number;
  };
  transcript?: TranscriptItem[];
};

export type PendingApproval = { id: string; name: string; args?: unknown };

export type RunGoal = {
  objective?: string;
  remaining?: string[];
  done?: string[];
};

export function isActiveGoal(goal: RunGoal | null | undefined): boolean {
  if (!goal) return false;
  return Boolean(
    goal.objective?.trim() || (goal.remaining && goal.remaining.length) || (goal.done && goal.done.length),
  );
}

export function normalizeGoal(value: unknown): RunGoal | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const inner =
    record.goal && typeof record.goal === "object"
      ? (record.goal as Record<string, unknown>)
      : record;
  const goal: RunGoal = {
    objective: typeof inner.objective === "string" ? inner.objective : "",
    remaining: Array.isArray(inner.remaining) ? inner.remaining.map(String) : [],
    done: Array.isArray(inner.done) ? inner.done.map(String) : [],
  };
  return isActiveGoal(goal) ? goal : null;
}

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

export function captureSrc(url: string): string {
  if (url.startsWith("data:") || url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  return `${apiUrl()}${url}`;
}

/** Rewrite relative /captures//photos/ image URLs in markdown to the API origin. */
export function resolveChatImages(text: string): string {
  const base = apiUrl();
  return text.replace(
    /!\[([^\]]*)\]\((\/(?:captures|photos)\/[^)]+)\)/g,
    (_match, label: string, path: string) => `![${label}](${base}${path})`,
  );
}

const MARKDOWN_IMAGE_RE = /!\[([^\]]*)\]\(([^)]+)\)/g;

function imageUrlKey(url: string): string {
  const captures = url.indexOf("/captures/");
  if (captures >= 0) return url.slice(captures);
  const photos = url.indexOf("/photos/");
  if (photos >= 0) return url.slice(photos);
  return url;
}

export function extractMarkdownImages(text: string): ToolImage[] {
  return [...text.matchAll(MARKDOWN_IMAGE_RE)]
    .map((match) => ({ label: match[1] || "Image", url: match[2].trim() }))
    .filter((image) => image.url);
}

/** Chat shows tool thumbnails instead of inline markdown images. */
export function stripMarkdownImages(text: string): string {
  return text
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Drop leaked Gemini/OpenRouter thought labels from visible assistant prose. */
export function stripThoughtPrefix(text: string): string {
  return text
    .replace(/<(think|thought|thinking)>[\s\S]*?<\/\1>/gi, "")
    .replace(/^(?:thought|thinking)\s+(?=[A-Z])/i, "")
    .trim();
}

function ndrePreviewFromSample(sample: Record<string, unknown>): string | null {
  const preview = sample.ndre_preview;
  if (typeof preview === "string" && preview.startsWith("/captures/")) {
    return preview;
  }
  const nir = sample.nir;
  if (nir && typeof nir === "object") {
    const artifactId = (nir as Record<string, unknown>).artifact_id;
    if (typeof artifactId === "string" && artifactId) {
      return `/captures/${artifactId}/ndre`;
    }
  }
  return null;
}

function toolResultImages(name: string, result: unknown): ToolImage[] {
  if (!result || typeof result !== "object") return [];
  const record = result as Record<string, unknown>;
  const params =
    record.params && typeof record.params === "object"
      ? (record.params as Record<string, unknown>)
      : null;
  const images: ToolImage[] = [];

  if (name === "scan_ndre" && params && Array.isArray(params.samples)) {
    params.samples.forEach((sample, index) => {
      if (!sample || typeof sample !== "object") return;
      const url = ndrePreviewFromSample(sample as Record<string, unknown>);
      if (!url) return;
      const axisPos =
        (sample as Record<string, unknown>).y ?? (sample as Record<string, unknown>).x;
      const label =
        typeof axisPos === "number" ? `NDRE ${index + 1} (${axisPos} mm)` : `NDRE ${index + 1}`;
      images.push({ label, url });
    });
  }

  if (name === "capture_ndre" && params) {
    const preview = params.ndre_preview;
    if (typeof preview === "string" && preview) {
      images.push({ label: "NDRE map", url: preview });
    }
  }

  return images;
}

export function mergeToolImages(
  images: ToolImage[] | undefined,
  name: string,
  result: unknown,
): ToolImage[] {
  const fromResult = toolResultImages(name, result);
  if (!images?.length) return fromResult;
  if (!fromResult.length) return images;
  const seen = new Set(images.map((item) => item.url));
  return [...images, ...fromResult.filter((item) => !seen.has(item.url))];
}

// ---------------------------------------------------------------------------
// Short one-line summaries so tool rows scan like a modern agent UI
// ---------------------------------------------------------------------------

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : {};

const mm = (value: unknown): string | null =>
  typeof value === "number" && Number.isFinite(value) ? `${Math.round(value)}` : null;

/** Short "target" for a tool row, e.g. "y 0→1900 mm · step 100". */
function describeToolCall(name: string, args: unknown): string {
  const a = asRecord(args);
  switch (name) {
    case "move": {
      const coords = [a.x, a.y, a.z].map(mm);
      return coords.every((c) => c !== null) ? `to (${coords.join(", ")}) mm` : "";
    }
    case "move_path": {
      const count = Array.isArray(a.waypoints) ? a.waypoints.length : 0;
      if (!count) return "";
      return `${count} waypoint${count === 1 ? "" : "s"}${a.photo_at_waypoints ? " · photos" : ""}`;
    }
    case "water":
      return typeof a.seconds === "number" ? `for ${a.seconds}s` : "";
    case "find_home":
      return a.axis && a.axis !== "all" ? `axis ${String(a.axis)}` : "all axes";
    case "read_pin":
    case "write_pin": {
      const pin = a.pin ?? a.pin_id;
      const base = pin !== undefined ? `pin ${String(pin)}` : "";
      return name === "write_pin" && a.value !== undefined
        ? `${base} = ${String(a.value)}`
        : base;
    }
    case "mount_tool":
    case "dismount_tool":
      return typeof a.tool_name === "string"
        ? a.tool_name
        : typeof a.tool === "string"
          ? a.tool
          : "";
    case "capture":
      return typeof a.band === "string" ? `${a.band} band` : "";
    case "scan_ndre": {
      const axis =
        typeof a.axis === "string" ? a.axis : a.y_end !== undefined ? "y" : a.x_end !== undefined ? "x" : "y";
      const start = a.start_mm ?? a[`${axis}_start`] ?? 0;
      const end = a.end_mm ?? a[`${axis}_end`];
      const step = a.step_mm ?? a.step;
      const parts = [`${axis} ${mm(start) ?? "0"}→${mm(end) ?? "?"} mm`];
      if (typeof step === "number") parts.push(`step ${mm(step)}`);
      const fixed = axis === "y" ? a.x : a.y;
      if (typeof fixed === "number") parts.push(`${axis === "y" ? "x" : "y"} ${mm(fixed)}`);
      return parts.join(" · ");
    }
    default:
      return genericArgsSummary(a);
  }
}

function genericArgsSummary(args: Record<string, unknown>): string {
  const entries = Object.entries(args).filter(
    ([, value]) => value !== null && typeof value !== "object",
  );
  const parts = entries.slice(0, 3).map(([key, value]) => `${key}: ${String(value).slice(0, 24)}`);
  if (Object.keys(args).length > parts.length) parts.push("…");
  return parts.join(" · ");
}

/** One-line result hint rendered after the tool name (scan counts, NDRE mean…). */
function summarizeToolResult(name: string, result: unknown): string | null {
  if (!result || typeof result !== "object") return null;
  const record = result as Record<string, unknown>;
  if (record.status === "error" || record.error) return null;
  const params = asRecord(record.params);

  if (name === "capture_ndre") {
    const ndre = asRecord(params.ndre ?? record.ndre);
    const interpretation = asRecord(params.interpretation ?? record.interpretation);
    const parts: string[] = [];
    if (typeof ndre.mean === "number") parts.push(`NDRE ${ndre.mean.toFixed(2)}`);
    if (typeof interpretation.label === "string") parts.push(interpretation.label);
    return parts.length ? parts.join(" · ") : null;
  }
  if (name === "scan_ndre") {
    const samples = Array.isArray(params.samples) ? params.samples : [];
    if (!samples.length) return null;
    const means = samples
      .map((sample) => asRecord(asRecord(sample).ndre).mean)
      .filter((value): value is number => typeof value === "number");
    const average = means.length
      ? means.reduce((sum, value) => sum + value, 0) / means.length
      : null;
    return [
      `${samples.length} sample${samples.length === 1 ? "" : "s"}`,
      average !== null ? `mean NDRE ${average.toFixed(2)}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }
  if (name === "get_position") {
    const position = asRecord(params.position ?? params);
    const coords = [position.x, position.y, position.z].map(mm);
    return coords.every((c) => c !== null) ? `(${coords.join(", ")}) mm` : null;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Tool calls
// ---------------------------------------------------------------------------

function toolFailure(result: unknown): string | null {
  if (!result || typeof result !== "object") return null;
  const record = result as Record<string, unknown>;
  if (record.status === "error" || record.error) {
    return String(record.error ?? record.message ?? "Tool call failed");
  }
  return null;
}

function toolCallStatus(item: ToolTranscriptItem): ChatToolCallStatus {
  if (toolFailure(item.result)) return "error";
  return item.result === undefined ? "running" : "complete";
}

function JsonDetail({ label, value }: { label: string; value: unknown }) {
  const code = typeof value === "string" ? value : JSON.stringify(value ?? null, null, 2);
  return (
    <VStack gap={1}>
      <Text type="supporting" color="secondary">
        {label}
      </Text>
      <CodeBlock
        code={code}
        language="json"
        size="sm"
        container="section"
        width="100%"
        isWrapped
        isCollapsible
        maxHeight={260}
        hasCopyButton
      />
    </VStack>
  );
}

function toCallItem(item: ToolTranscriptItem, index: number): ChatToolCallItem {
  const failure = toolFailure(item.result);
  const summary = summarizeToolResult(item.name, item.result);
  return {
    key: item.id ?? `${index}:${item.name}`,
    name: item.name,
    target: describeToolCall(item.name, item.args) || undefined,
    status: toolCallStatus(item),
    errorMessage: failure ?? undefined,
    stats: summary ? (
      <Text type="supporting" color="secondary">
        {summary}
      </Text>
    ) : undefined,
    resultDetail: (
      <VStack gap={3} style={{ padding: "4px 0 10px" }}>
        <JsonDetail label="Arguments" value={item.args ?? {}} />
        <JsonDetail label={failure ? "Error" : "Result"} value={item.result ?? {}} />
      </VStack>
    ),
  };
}

function toolGroupImages(items: ToolTranscriptItem[]): ToolImage[] {
  return items.flatMap((item) => mergeToolImages(item.images, item.name, item.result));
}

/** Consecutive tool calls render as one grouped, collapsible ChatToolCalls. */
function ToolCallsGroup({ items }: { items: ToolTranscriptItem[] }) {
  return (
    <VStack gap={2} style={{ minWidth: 0 }}>
      <ChatToolCalls calls={items.map(toCallItem)} />
      <ToolImageGrid images={toolGroupImages(items)} />
    </VStack>
  );
}

// ---------------------------------------------------------------------------
// Tool result images — thumbnails + zoomable lightbox gallery
// ---------------------------------------------------------------------------

export function ToolImageGrid({ images }: { images: ToolImage[] }) {
  const [index, setIndex] = useState<number | null>(null);
  if (!images.length) return null;
  return (
    <>
      <HStack gap={2} wrap="wrap">
        {images.map((image, imageIndex) => (
          <VStack key={`${image.label}:${image.url}`} gap={1} style={{ width: 120 }}>
            <Thumbnail
              src={captureSrc(image.url)}
              alt={image.label}
              label={image.label}
              onClick={() => setIndex(imageIndex)}
              style={{ width: 120, height: 120 }}
            />
            <Text type="supporting" color="secondary" maxLines={1}>
              {image.label}
            </Text>
          </VStack>
        ))}
      </HStack>
      <Lightbox
        isOpen={index !== null}
        onOpenChange={() => setIndex(null)}
        media={images.map((image) => ({
          src: captureSrc(image.url),
          alt: image.label,
          caption: image.label,
        }))}
        index={index ?? 0}
        onIndexChange={setIndex}
        hasZoom
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// Reasoning + execution trace — muted, collapsed by default
// ---------------------------------------------------------------------------

function MutedDisclosure({
  icon,
  label,
  children,
}: {
  icon?: React.ReactNode;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <Collapsible
      defaultIsOpen={false}
      trigger={
        <HStack gap={1} vAlign="center">
          {icon}
          <Text type="supporting" color="secondary">
            {label}
          </Text>
        </HStack>
      }
    >
      <div
        style={{
          borderLeft: "2px solid var(--color-border)",
          margin: "4px 0 8px 2px",
          paddingLeft: 12,
        }}
      >
        {children}
      </div>
    </Collapsible>
  );
}

function ThinkingDisclosure({ text }: { text: string }) {
  return (
    <MutedDisclosure
      label="Reasoning"
      icon={
        <SparklesIcon
          style={{ width: 14, height: 14, color: "var(--color-text-secondary)" }}
        />
      }
    >
      <Text type="supporting" color="secondary" style={{ whiteSpace: "pre-wrap" }} as="p">
        {text}
      </Text>
    </MutedDisclosure>
  );
}

function formatRunStats(metrics: ChatMsg["metrics"]): string {
  if (!metrics) return "";
  const parts: string[] = [];
  const latency = metrics.total_latency_s;
  if (typeof latency === "number" && latency > 0) {
    parts.push(`${latency < 10 ? latency.toFixed(1) : Math.round(latency)}s`);
  }
  if (metrics.total_tokens) parts.push(`${metrics.total_tokens} tok`);
  if (metrics.tokens_per_s) parts.push(`${metrics.tokens_per_s} tok/s`);
  return parts.join(" · ");
}

function WeaveTraceRow({
  url,
  metrics,
}: {
  url?: string;
  metrics?: ChatMsg["metrics"];
}) {
  const stats = formatRunStats(metrics);
  if (!url && !stats) return null;
  return (
    <HStack gap={2} vAlign="center" wrap="wrap">
      {url ? (
        <Link href={url} isExternalLink type="supporting" color="secondary">
          W&B trace
        </Link>
      ) : null}
      {stats ? (
        <Text type="supporting" color="secondary">
          {stats}
        </Text>
      ) : null}
    </HStack>
  );
}

// ---------------------------------------------------------------------------
// Assistant message content
// ---------------------------------------------------------------------------

type TranscriptGroup =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string }
  | { kind: "tools"; items: ToolTranscriptItem[] };

/** Fall back to tool_calls + content for messages persisted before transcripts. */
function normalizeTranscript(message: ChatMsg): TranscriptItem[] {
  if (message.transcript?.length) return message.transcript;
  const items: TranscriptItem[] = [];
  for (const tool of message.tool_calls ?? []) {
    items.push({
      kind: "tool",
      name: tool.name,
      args: tool.args,
      result: tool.result,
      images: tool.images,
    });
  }
  const text = message.content || (message.streaming && !message.error ? "…" : "");
  if (text) items.push({ kind: "text", text });
  return items;
}

function groupTranscript(items: TranscriptItem[]): TranscriptGroup[] {
  const groups: TranscriptGroup[] = [];
  for (const item of items) {
    const last = groups[groups.length - 1];
    if (item.kind === "tool") {
      if (last?.kind === "tools") last.items.push(item);
      else groups.push({ kind: "tools", items: [item] });
    } else {
      groups.push({ kind: item.kind, text: item.text });
    }
  }
  return groups;
}

function WorkingIndicator() {
  return (
    <HStack gap={2} vAlign="center">
      <Spinner size="sm" shade="subtle" />
      <Text type="supporting" color="secondary">
        Working…
      </Text>
    </HStack>
  );
}

export function AssistantContent({ message }: { message: ChatMsg }) {
  const items = normalizeTranscript(message);
  const groups = groupTranscript(items);
  const lastItem = items[items.length - 1];
  // A running tool row already shows a spinner — don't double up.
  const lastToolRunning = lastItem?.kind === "tool" && lastItem.result === undefined;
  const isWorking = Boolean(message.streaming) && lastItem?.kind !== "text" && !lastToolRunning;
  const toolUrls = new Set(
    groups.flatMap((group) =>
      group.kind === "tools"
        ? toolGroupImages(group.items).map((image) => imageUrlKey(image.url))
        : [],
    ),
  );
  return (
    <VStack gap={2} style={{ minWidth: 0 }}>
      {message.error ? (
        <Banner
          status="error"
          title="Assistant error"
          description={message.error}
          collapsible={false}
        />
      ) : null}
      {groups.map((group, groupIndex) => {
        const key = `${group.kind}:${groupIndex}`;
        if (group.kind === "tools") {
          return <ToolCallsGroup key={key} items={group.items} />;
        }
        if (group.kind === "thinking") {
          return <ThinkingDisclosure key={key} text={group.text} />;
        }
        const prose = stripThoughtPrefix(stripMarkdownImages(group.text));
        const leftover = extractMarkdownImages(group.text).filter(
          (image) => !toolUrls.has(imageUrlKey(image.url)),
        );
        if (!prose && !leftover.length) return null;
        return (
          <VStack key={key} gap={2} style={{ minWidth: 0, width: "100%" }}>
            {prose ? (
              <Markdown
                density="compact"
                isStreaming={Boolean(message.streaming) && groupIndex === groups.length - 1}
              >
                {resolveChatImages(prose)}
              </Markdown>
            ) : null}
            <ToolImageGrid images={leftover} />
          </VStack>
        );
      })}
      {isWorking ? <WorkingIndicator /> : null}
      <WeaveTraceRow url={message.weave_url} metrics={message.metrics} />
    </VStack>
  );
}

/** Small secondary row under a message: time · model. */
export function MessageMetadata({ message, model }: { message: ChatMsg; model?: string }) {
  if (!message.ts || message.streaming) return null;
  return (
    <ChatMessageMetadata
      timestamp={<Timestamp value={new Date(message.ts).toISOString()} format="time" />}
      footer={message.role === "assistant" && model ? model : undefined}
    />
  );
}

// ---------------------------------------------------------------------------
// Approval prompt above the composer
// ---------------------------------------------------------------------------

function approvalChip(item: PendingApproval): string {
  const detail = describeToolCall(item.name, item.args);
  return detail ? `${item.name} · ${detail}` : item.name;
}

// ---------------------------------------------------------------------------
// Goal card above the composer (same bubble language as the input)
// ---------------------------------------------------------------------------

export function GoalBubble({
  goal,
  approvals,
  collapsed,
  onToggle,
  onApprove,
  onReject,
}: {
  goal: RunGoal | null;
  approvals: PendingApproval[];
  collapsed: boolean;
  onToggle: () => void;
  onApprove: () => void;
  onReject: () => void;
}) {
  const pendingGoal = approvals.find((item) => item.name === "set_goal");
  const actionApprovals = approvals.filter((item) => item.name !== "set_goal");
  const shown = pendingGoal ? normalizeGoal(pendingGoal.args) : goal;
  const needsApproval = approvals.length > 0;
  if (!isActiveGoal(shown) && !needsApproval) return null;
  const remaining = shown?.remaining || [];
  const done = shown?.done || [];
  const title = shown?.objective?.trim() || (needsApproval ? "Approval required" : "Current goal");
  const countLabel = [
    remaining.length ? `${remaining.length} remaining` : done.length ? "Checklist complete" : "",
    needsApproval ? "Approval required" : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const expanded = needsApproval || !collapsed;

  return (
    <div style={{ padding: "0 4px 8px" }}>
      <div className={needsApproval ? "twfb-goal-bubble twfb-goal-bubble-pending" : "twfb-goal-bubble"}>
        <HStack gap={3} vAlign="start" style={{ width: "100%", justifyContent: "space-between" }}>
          <div
            style={{ minWidth: 0, flex: 1, cursor: needsApproval ? "default" : "pointer" }}
            onClick={needsApproval ? undefined : onToggle}
          >
            <HStack gap={3} vAlign="start">
            <InformationCircleIcon
              style={{ width: 18, height: 18, flexShrink: 0, marginTop: 2, opacity: 0.7 }}
            />
            <VStack gap={1} style={{ minWidth: 0 }}>
              <Text>{title}</Text>
              {countLabel ? (
                <Text type="supporting" color="secondary">
                  {countLabel}
                </Text>
              ) : null}
            </VStack>
            </HStack>
          </div>
          {needsApproval ? (
            <HStack gap={2}>
              <Button label="Reject" variant="secondary" onClick={onReject} />
              <Button label="Approve" variant="primary" onClick={onApprove} />
            </HStack>
          ) : null}
        </HStack>
        {expanded ? (
          <HStack gap={1} wrap="wrap">
            {actionApprovals.map((item) => (
              <span key={item.id} className="twfb-goal-chip twfb-goal-chip-action">
                {approvalChip(item)}
              </span>
            ))}
            {remaining.map((step, index) => (
              <span key={`r-${index}-${step}`} className="twfb-goal-chip">
                {step}
              </span>
            ))}
            {done.map((step, index) => (
              <span key={`d-${index}-${step}`} className="twfb-goal-chip twfb-goal-chip-done">
                {step}
              </span>
            ))}
          </HStack>
        ) : null}
      </div>
    </div>
  );
}
