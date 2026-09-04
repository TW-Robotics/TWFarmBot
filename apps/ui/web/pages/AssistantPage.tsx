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
  | { kind: "tool"; name: string; args?: unknown; result?: unknown };

type ChatMsg = {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  tool_calls?: { name: string; args?: any; result?: any; caller?: { type?: string } | null }[];
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
            pushTranscript({ kind: "tool", name: event.name, args: event.args, result: event.result });
          } else if (event.type === "program") {
            const next = [...(assistant.programs || [])];
            const index = next.findIndex((item) => item.call_id && item.call_id === event.call_id);
            const recorded = { call_id: event.call_id, code: event.code, result: event.result, status: event.status };
            if (index >= 0) next[index] = { ...next[index], ...recorded };
            else next.push(recorded);
            assistant.programs = next;
          } else if (event.type === "meta") {
            assistant.proposed_actions = event.proposed_actions || [];
            assistant.tool_calls = event.tool_calls || assistant.tool_calls;
            assistant.programs = event.programs || assistant.programs;
            assistant.trace = event.trace || assistant.trace;
          } else if (event.type === "error") assistant.error = event.error;
          setMessages([...request, { ...assistant }]);
        },
        { signal: controller.signal },
      );
      assistant.streaming = false;
      if (controller.signal.aborted) {
        pushTranscript({ kind: "text", text: "\n\n*Stopped by user.*" });
      } else if (!assistant.content && !assistant.error) {
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
            onStop={() => stopRef.current?.abort()}
            isStopShown={busy}
            isDisabled={busy}
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
                        resultRecord !== null && "error" in resultRecord;
                      return (
                        <ChatToolCalls
                          key={itemIndex}
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
                    <ChatToolCalls
                      calls={message.tool_calls.map((tool) => ({
                        name: tool.name,
                        target: `${tool.caller ? "program" : "direct"} · ${JSON.stringify(tool.args || {})}`,
                        status: "complete" as const,
                      }))}
                    />
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
      </ChatLayout>
    </VStack>
  );
}
