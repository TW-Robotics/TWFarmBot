import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChatComposer,
  ChatLayout,
  ChatMessage,
  ChatMessageBubble,
  ChatMessageList,
  ChatToolCalls,
} from "@astryxdesign/core/Chat";
import { Markdown } from "@astryxdesign/core/Markdown";
import { Button } from "@astryxdesign/core/Button";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Selector } from "@astryxdesign/core/Selector";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, apiUrl, postAction, streamChat } from "../api";

function resolveChatImages(text: string): string {
  const base = apiUrl();
  return text.replace(
    /!\[([^\]]*)\]\((\/(?:captures|photos)\/[^)]+)\)/g,
    (_match, label: string, path: string) => `![${label}](${base}${path})`,
  );
}
import {
  SETTINGS_CHANGED_EVENT,
  getActiveLlmProfile,
  llmOverridesFromProfile,
  profileLabel,
} from "../settings";

type ProgramItem = {
  call_id?: string;
  code?: string;
  result?: unknown;
  status?: string;
};
type TranscriptItem =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string }
  | {
      kind: "tool";
      name: string;
      args?: unknown;
      result?: unknown;
      images?: { label: string; url: string }[];
    };

function captureSrc(url: string): string {
  if (url.startsWith("data:") || url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  return `${apiUrl()}${url}`;
}

function ToolStillGrid({ images }: { images: { label: string; url: string }[] }) {
  if (!images.length) return null;
  return (
    <HStack gap={2} style={{ flexWrap: "wrap", marginTop: 8 }}>
      {images.map((image) => (
        <VStack key={`${image.label}:${image.url}`} gap={1}>
          <img
            alt={image.label}
            src={captureSrc(image.url)}
            style={{
              maxWidth: 240,
              maxHeight: 180,
              objectFit: "contain",
              borderRadius: 6,
            }}
          />
          <Text type="supporting" color="secondary">
            {image.label}
          </Text>
        </VStack>
      ))}
    </HStack>
  );
}

type ToolImage = { label: string; url: string };

function ndrePreviewFromSample(sample: Record<string, unknown>): string | null {
  const preview = sample.ndre_preview;
  if (typeof preview === "string" && preview.startsWith("/captures/")) {
    return preview;
  }
  const nir = sample.nir;
  if (nir && typeof nir === "object" && nir !== null) {
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
        typeof axisPos === "number"
          ? `NDRE ${index + 1} (${axisPos} mm)`
          : `NDRE ${index + 1}`;
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

function mergeToolImages(
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

function attachMetaToolImages(
  assistant: ChatMsg,
  toolCalls: { name: string; args?: unknown; result?: unknown; images?: ToolImage[] }[],
) {
  assistant.tool_calls = toolCalls.map((tool) => ({
    ...tool,
    images: mergeToolImages(tool.images, tool.name, tool.result),
  }));
  if (!assistant.transcript?.length) return;
  for (const tool of assistant.tool_calls) {
    if (!tool.images?.length) continue;
    for (let index = assistant.transcript.length - 1; index >= 0; index -= 1) {
      const item = assistant.transcript[index];
      if (item.kind === "tool" && item.name === tool.name) {
        if (!item.images?.length) item.images = tool.images;
        break;
      }
    }
  }
}

type ChatMsg = {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  tool_calls?: {
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
  trace?: { turn?: number; response_id?: string; status?: string; output?: Record<string, unknown>[] }[];
  transcript?: TranscriptItem[];
};

export function AssistantPage() {
  const toast = useToast();
  // Bump this when the assistant wire format/provider defaults change so old
  // conversations cannot keep the UI in an incompatible state.
  const storageKey = `twfarmbot:assistant:v6:${apiUrl()}`;
  const [messages, setMessages] = useState<ChatMsg[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(storageKey) || "[]");
    } catch {
      return [];
    }
  });
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState(false);
  const stopRef = useRef<AbortController | null>(null);
  const threadIdRef = useRef<string | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<
    { id: string; name: string; args?: unknown }[]
  >([]);
  const [activeProfileName, setActiveProfileName] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    const profile = getActiveLlmProfile();
    setActiveProfileName(profile ? profileLabel(profile) : null);
    const llm = llmOverridesFromProfile(profile);
    try {
      const result = llm
        ? await api<{ models?: string[]; current?: string }>("/models/list", {
            method: "POST",
            body: JSON.stringify({ llm }),
          })
        : await (async () => {
            const server = await api<{ provider?: string }>("/settings/llm");
            const provider = profile?.provider || server.provider || "openai";
            return api<{ models?: string[]; current?: string; provider?: string }>(
              `/models?provider=${encodeURIComponent(provider)}`,
            );
          })();
      const list = result.models || [];
      setModels(list);
      const remembered = localStorage.getItem(`${storageKey}:model`);
      setModel(
        remembered && list.includes(remembered)
          ? remembered
          : profile?.model || result.current || list[0] || "",
      );
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not load models",
        type: "error",
      });
    }
  }, [storageKey, toast]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  useEffect(() => {
    const onSettingsChanged = () => void loadModels();
    window.addEventListener(SETTINGS_CHANGED_EVENT, onSettingsChanged);
    return () => window.removeEventListener(SETTINGS_CHANGED_EVENT, onSettingsChanged);
  }, [loadModels]);

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(messages));
  }, [messages, storageKey]);

  const stopRun = async () => {
    const threadId = threadIdRef.current;
    stopRef.current?.abort();
    if (threadId) {
      try {
        await api("/chat/cancel", {
          method: "POST",
          body: JSON.stringify({ thread_id: threadId }),
        });
      } catch {
        /* stream abort is enough if cancel is unreachable */
      }
    }
    setPendingApprovals([]);
    setBusy(false);
  };

  const send = async (text: string) => {
    const value = text.trim();
    if (!value || busy) return;
    setBusy(true);
    const controller = new AbortController();
    stopRef.current = controller;
    const request = [...messages, { role: "user" as const, content: value }];
    const assistant: ChatMsg = {
      role: "assistant",
      content: "",
      tool_calls: [],
      programs: [],
      proposed_actions: [],
      streaming: true,
    };
    setMessages([...request, assistant]);
    let awaitingApproval = false;
    const pushTranscript = (item: TranscriptItem) => {
      const current = assistant.transcript ?? [];
      const last = current[current.length - 1];
      if (
        (item.kind === "text" || item.kind === "thinking") &&
        last !== undefined &&
        last.kind === item.kind
      ) {
        last.text += item.text;
      } else {
        current.push(item);
      }
      assistant.transcript = current;
    };
    try {
      await streamChat(
        {
          messages: request,
          model: model || null,
          allow_actions: true,
          llm: llmOverridesFromProfile(getActiveLlmProfile()),
        },
        (event) => {
          if (event.type === "delta") {
            assistant.content += event.content || "";
            pushTranscript({ kind: "text", text: event.content || "" });
          } else if (event.type === "thinking") {
            assistant.thinking = (assistant.thinking || "") + (event.content || "");
            pushTranscript({ kind: "thinking", text: event.content || "" });
          } else if (event.type === "tool_call") {
            assistant.tool_calls = [
              ...(assistant.tool_calls || []),
              { name: event.name, args: event.args, result: event.result, caller: event.caller },
            ];
            pushTranscript({
              kind: "tool",
              name: event.name,
              args: event.args,
              result: event.result,
              images: mergeToolImages(event.images, event.name, event.result),
            });
            setPendingApprovals([]);
          } else if (event.type === "program") {
            const next = [...(assistant.programs || [])];
            const index = next.findIndex((item) => item.call_id && item.call_id === event.call_id);
            const recorded = { call_id: event.call_id, code: event.code, result: event.result, status: event.status };
            if (index >= 0) next[index] = { ...next[index], ...recorded };
            else next.push(recorded);
            assistant.programs = next;
          } else if (event.type === "thread" && event.thread_id) {
            threadIdRef.current = event.thread_id;
          } else if (event.type === "approval") {
            threadIdRef.current = event.thread_id || threadIdRef.current;
            setPendingApprovals(event.pending_approvals || []);
            awaitingApproval = true;
            assistant.streaming = false;
          } else if (event.type === "meta") {
            assistant.proposed_actions = event.proposed_actions || [];
            if (event.tool_calls?.length) {
              attachMetaToolImages(assistant, event.tool_calls);
            } else {
              assistant.tool_calls = event.tool_calls || assistant.tool_calls;
            }
            assistant.programs = event.programs || assistant.programs;
            assistant.trace = event.trace || assistant.trace;
            setPendingApprovals([]);
          } else if (event.type === "error") assistant.error = event.error;
          setMessages([...request, { ...assistant }]);
        },
        { signal: controller.signal },
      );
      assistant.streaming = false;
      if (controller.signal.aborted) {
        pushTranscript({ kind: "text", text: "\n\n*Stopped by user.*" });
      } else if (!assistant.content && !assistant.error && !awaitingApproval) {
        assistant.content = "I could not produce a response. Try again with a more specific request.";
      }
      setMessages([...request, { ...assistant }]);
    } catch (error) {
      assistant.streaming = false;
      assistant.error = error instanceof Error ? error.message : "Assistant request failed";
      setMessages([...request, { ...assistant }]);
      toast({ body: assistant.error, type: "error" });
    } finally {
      stopRef.current = null;
      setBusy(false);
    }
  };

  const resumeApprovals = async (ids: string[]) => {
    const threadId = threadIdRef.current;
    if (!threadId || busy) return;
    setPendingApprovals([]);
    setBusy(true);
    const controller = new AbortController();
    stopRef.current = controller;
    const history = messages.filter((item) => item.role === "user" || item.role === "assistant");
    const request = history.filter((item) => item.role === "user" || !item.streaming);
    const assistant: ChatMsg = {
      ...(history[history.length - 1] || { role: "assistant" as const, content: "" }),
      role: "assistant",
      streaming: true,
      error: undefined,
    };
    setMessages([...request.slice(0, -1), assistant]);
    try {
      await streamChat(
        {
          messages: request.filter((item) => item.role === "user"),
          model: model || null,
          allow_actions: true,
          llm: llmOverridesFromProfile(getActiveLlmProfile()),
          thread_id: threadId,
          approved_ids: ids,
        },
        (event) => {
          if (event.type === "delta") {
            assistant.content = (assistant.content || "") + (event.content || "");
            const current = assistant.transcript ?? [];
            const last = current[current.length - 1];
            const piece = event.content || "";
            if (last?.kind === "text") last.text += piece;
            else current.push({ kind: "text", text: piece });
            assistant.transcript = current;
          } else if (event.type === "thinking") {
            assistant.thinking = (assistant.thinking || "") + (event.content || "");
          } else if (event.type === "tool_call") {
            assistant.tool_calls = [
              ...(assistant.tool_calls || []),
              { name: event.name, args: event.args, result: event.result, caller: event.caller },
            ];
            const current = assistant.transcript ?? [];
            current.push({
              kind: "tool",
              name: event.name,
              args: event.args,
              result: event.result,
              images: mergeToolImages(event.images, event.name, event.result),
            });
            assistant.transcript = current;
            setPendingApprovals([]);
          } else if (event.type === "approval") {
            const incoming = event.pending_approvals || [];
            const replay =
              ids.length > 0 &&
              incoming.length === ids.length &&
              incoming.every((item: { id: string }) => ids.includes(item.id));
            if (!replay) {
              threadIdRef.current = event.thread_id || threadIdRef.current;
              setPendingApprovals(incoming);
              assistant.streaming = false;
            }
          } else if (event.type === "meta") {
            if (event.tool_calls?.length) {
              attachMetaToolImages(assistant, event.tool_calls);
            } else {
              assistant.tool_calls = event.tool_calls || assistant.tool_calls;
            }
            assistant.proposed_actions = event.proposed_actions || assistant.proposed_actions;
            setPendingApprovals([]);
          } else if (event.type === "error") assistant.error = event.error;
          setMessages([...request.slice(0, -1), { ...assistant }]);
        },
        { signal: controller.signal },
      );
      assistant.streaming = false;
      setMessages([...request.slice(0, -1), { ...assistant }]);
    } catch (error) {
      assistant.streaming = false;
      assistant.error = error instanceof Error ? error.message : "Approval resume failed";
      setMessages([...request.slice(0, -1), { ...assistant }]);
      toast({ body: assistant.error, type: "error" });
    } finally {
      stopRef.current = null;
      setBusy(false);
    }
  };

  return (
    <VStack height="100%" style={{ minHeight: 0, height: "100%" }}>
      <HStack gap={3} style={{ padding: "12px 20px" }}>
        {activeProfileName ? (
          <Text type="supporting" color="secondary">
            {activeProfileName}
          </Text>
        ) : null}
        {models.length > 0 && (
          <Selector
            label="Model"
            isLabelHidden
            options={models}
            value={model}
            onChange={(value) => {
              setModel(value);
              localStorage.setItem(`${storageKey}:model`, value);
            }}
          />
        )}
        <Button label="Clear" onClick={() => setMessages([])} />
        {busy || pendingApprovals.length > 0 ? (
          <Button label="Stop" onClick={() => void stopRun()} />
        ) : null}
      </HStack>
      <ChatLayout
        density="spacious"
        style={{ flex: 1, minHeight: 0 }}
        emptyState={
          <EmptyState
            title="What would you like to do?"
            description="Ask about the garden, inspect robot state, analyze a camera image, or prepare a safe action."
          />
        }
        composer={
          <ChatComposer
            placeholder="Ask about your garden…"
            onSubmit={(value) => void send(value)}
            onStop={() => void stopRun()}
            isStopShown={busy}
            isDisabled={pendingApprovals.length > 0}
          />
        }
      >
        <ChatMessageList isStreaming={busy}>
          {messages.map((message, index) => (
            <ChatMessage key={index} sender={message.role}>
              {message.role === "assistant" && message.transcript?.length ? (
                <VStack gap={2}>
                  {message.error ? (
                    <ChatMessageBubble variant="ghost">
                      <Markdown density="compact">{message.error}</Markdown>
                    </ChatMessageBubble>
                  ) : null}
                  {message.transcript.map((item, itemIndex) => {
                    if (item.kind === "tool") {
                      const resultRecord =
                        typeof item.result === "object" && item.result !== null
                          ? (item.result as Record<string, unknown>)
                          : null;
                      const failed =
                        resultRecord !== null &&
                        (resultRecord.status === "error" ||
                          Boolean(resultRecord.error));
                      return (
                        <VStack key={itemIndex} gap={2}>
                          <ChatToolCalls
                            calls={[
                              {
                                name: item.name,
                                target: JSON.stringify(item.args ?? {}),
                                status: failed ? ("error" as const) : ("complete" as const),
                                errorMessage: failed
                                  ? String(resultRecord?.error ?? "failed")
                                  : undefined,
                                resultDetail: (
                                  <pre
                                    style={{
                                      whiteSpace: "pre-wrap",
                                      overflowWrap: "anywhere",
                                      margin: "4px 0",
                                    }}
                                  >
                                    {typeof item.result === "string"
                                      ? item.result
                                      : JSON.stringify(item.result, null, 2)}
                                  </pre>
                                ),
                              },
                            ]}
                          />
                          <ToolStillGrid
                            images={mergeToolImages(item.images, item.name, item.result)}
                          />
                        </VStack>
                      );
                    }
                    if (item.kind === "thinking") {
                      return (
                        <details key={itemIndex} style={{ fontSize: 13 }}>
                          <summary style={{ cursor: "pointer" }}>Reasoning</summary>
                          <pre
                            style={{
                              whiteSpace: "pre-wrap",
                              overflowWrap: "anywhere",
                            }}
                          >
                            {item.text}
                          </pre>
                        </details>
                      );
                    }
                    return (
                      <ChatMessageBubble key={itemIndex} variant="ghost">
                        <Markdown density="compact">
                          {resolveChatImages(item.text)}
                        </Markdown>
                      </ChatMessageBubble>
                    );
                  })}
                </VStack>
              ) : (
                <>
                  {message.tool_calls?.length ? (
                    <VStack gap={2}>
                      <ChatToolCalls
                        calls={message.tool_calls.map((tool) => ({
                          name: tool.name,
                          target: `${tool.caller ? "program" : "direct"} · ${JSON.stringify(tool.args || {})}`,
                          status: "complete" as const,
                        }))}
                      />
                      {message.tool_calls.some(
                        (tool) =>
                          mergeToolImages(tool.images, tool.name, tool.result).length > 0,
                      ) ? (
                        <ToolStillGrid
                          images={message.tool_calls.flatMap((tool) =>
                            mergeToolImages(tool.images, tool.name, tool.result),
                          )}
                        />
                      ) : null}
                    </VStack>
                  ) : null}
                  <ChatMessageBubble variant={message.role === "assistant" ? "ghost" : "filled"}>
                    <Markdown density="compact">
                      {resolveChatImages(
                        message.error || message.content || (message.streaming ? "…" : ""),
                      )}
                    </Markdown>
                  </ChatMessageBubble>
                </>
              )}
              {message.trace?.length ? (
                <details style={{ margin: "8px 12px", fontSize: 13 }}>
                  <summary style={{ cursor: "pointer" }}>Execution trace ({message.trace.length} turn{message.trace.length === 1 ? "" : "s"})</summary>
                  <VStack gap={2} style={{ marginTop: 8 }}>
                    {message.trace.map((turn, traceIndex) => (
                      <div key={traceIndex}>
                        <div>Turn {turn.turn || traceIndex + 1} · {turn.status || "completed"}</div>
                        {turn.output?.map((item, itemIndex) => (
                          <pre key={itemIndex} style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", margin: "4px 0" }}>
                            {JSON.stringify(item, null, 2)}
                          </pre>
                        ))}
                      </div>
                    ))}
                  </VStack>
                </details>
              ) : null}
            </ChatMessage>
          ))}
        </ChatMessageList>
        {pendingApprovals.length ? (
          <HStack gap={2} style={{ padding: "8px 20px" }}>
            <Text type="supporting" color="secondary">
              Approve {pendingApprovals.map((item) => item.name).join(", ")}?
            </Text>
            <Button
              label="Approve"
              onClick={() => void resumeApprovals(pendingApprovals.map((item) => item.id))}
            />
            <Button label="Reject" onClick={() => void resumeApprovals([])} />
          </HStack>
        ) : null}
      </ChatLayout>
    </VStack>
  );
}
