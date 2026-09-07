import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChatComposer,
  ChatLayout,
  ChatMessage,
  ChatMessageBubble,
  ChatMessageList,
  ChatSendButton,
} from "@astryxdesign/core/Chat";
import { Markdown } from "@astryxdesign/core/Markdown";
import { Button } from "@astryxdesign/core/Button";
import { DropdownMenu } from "@astryxdesign/core/DropdownMenu";
import { IconButton } from "@astryxdesign/core/IconButton";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { MicrophoneIcon, SpeakerWaveIcon, StopIcon } from "@heroicons/react/24/outline";
import { ApiError, api, apiUrl, streamChat } from "../api";
import { useMicTranscribe } from "../useMicTranscribe";
import { useRealtimeVoice } from "../useRealtimeVoice";
import { VoiceOverlay } from "../components/VoiceOverlay";
import {
  GoalBubble,
  AssistantContent,
  MessageMetadata,
  mergeToolImages,
  normalizeGoal,
  resolveChatImages,
  type ChatMsg,
  type PendingApproval,
  type RunGoal,
  type ToolImage,
  type TranscriptItem,
} from "../components/chat";
import { ReasoningEffortMenu } from "../components/ReasoningEffortControl";
import {
  SETTINGS_CHANGED_EVENT,
  getActiveLlmProfile,
  llmOverridesFromProfile,
  profileLabel,
} from "../settings";

function applyTrace(assistant: ChatMsg, event: { weave_url?: string; metrics?: ChatMsg["metrics"] }) {
  if (typeof event.weave_url === "string" && event.weave_url) {
    assistant.weave_url = event.weave_url;
  }
  if (event.metrics && typeof event.metrics === "object") {
    assistant.metrics = event.metrics;
  }
}

function applyRunGoal(event: { type?: string; name?: string; goal?: unknown; result?: unknown }, setGoal: (goal: ReturnType<typeof normalizeGoal>) => void) {
  if (event.type === "goal") {
    setGoal(normalizeGoal(event.goal));
    return;
  }
  if (event.type === "tool_call" && event.name === "set_goal") {
    setGoal(normalizeGoal(event.result));
    return;
  }
  if (event.type === "meta" && event.goal !== undefined) {
    setGoal(normalizeGoal(event.goal));
  }
}
function applyToolStart(
  assistant: ChatMsg,
  event: { id?: string; name: string; args?: unknown },
) {
  assistant.tool_calls = [
    ...(assistant.tool_calls || []),
    { id: event.id, name: event.name, args: event.args },
  ];
  const current = assistant.transcript ?? [];
  current.push({ kind: "tool", id: event.id, name: event.name, args: event.args });
  assistant.transcript = current;
}

/** A tool finished: patch its running row in place, or append for legacy streams. */
function applyToolResult(
  assistant: ChatMsg,
  event: {
    id?: string;
    name: string;
    args?: unknown;
    result?: unknown;
    images?: ToolImage[];
    caller?: { type?: string } | null;
  },
) {
  const images = mergeToolImages(event.images, event.name, event.result);
  const transcript = assistant.transcript ?? [];
  if (event.id) {
    for (let index = transcript.length - 1; index >= 0; index -= 1) {
      const item = transcript[index];
      if (item.kind === "tool" && item.id === event.id && item.result === undefined) {
        item.result = event.result;
        item.images = images;
        assistant.transcript = transcript;
        const call = (assistant.tool_calls || []).find((tool) => tool.id === event.id);
        if (call) {
          call.result = event.result;
          call.images = images;
        }
        return;
      }
    }
  }
  assistant.tool_calls = [
    ...(assistant.tool_calls || []),
    {
      id: event.id,
      name: event.name,
      args: event.args,
      result: event.result,
      images,
      caller: event.caller,
    },
  ];
  transcript.push({
    kind: "tool",
    id: event.id,
    name: event.name,
    args: event.args,
    result: event.result,
    images,
  });
  assistant.transcript = transcript;
}

/** Stream ended: tools that never reported a result were interrupted. */
function settlePendingTools(assistant: ChatMsg) {
  for (const item of assistant.transcript ?? []) {
    if (item.kind === "tool" && item.result === undefined) {
      item.result = { status: "error", error: "Interrupted before completion" };
    }
  }
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

function voiceActivity(message?: ChatMsg): string | undefined {
  const running = message?.transcript?.find(
    (item) => item.kind === "tool" && item.result === undefined,
  );
  if (running && running.kind === "tool") return running.name;
  const last = message?.tool_calls?.at(-1);
  if (last && last.result === undefined) return last.name;
  return undefined;
}

function shortModelName(id: string): string {
  const slash = id.lastIndexOf("/");
  return slash >= 0 ? id.slice(slash + 1) : id;
}

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
  const [reasoningBudget, setReasoningBudget] = useState(() => {
    const stored = Number(localStorage.getItem(`twfarmbot:assistant:v6:${apiUrl()}:reasoningBudget`));
    return Number.isFinite(stored) ? stored : getActiveLlmProfile()?.reasoningBudget ?? 8192;
  });
  const [busy, setBusy] = useState(false);
  const stopRef = useRef<AbortController | null>(null);
  const threadIdRef = useRef<string | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<PendingApproval[]>([]);
  const [goal, setGoal] = useState<RunGoal | null>(null);
  const [goalCollapsed, setGoalCollapsed] = useState(false);
  const [activeProfileName, setActiveProfileName] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const mic = useMicTranscribe();
  const voice = useRealtimeVoice();
  const voiceApiRef = useRef(voice);
  voiceApiRef.current = voice;
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const busyRef = useRef(busy);
  busyRef.current = busy;
  const midRunUsersRef = useRef<ChatMsg[]>([]);
  const leftoverRef = useRef<string[]>([]);
  const pendingRef = useRef(pendingApprovals);
  pendingRef.current = pendingApprovals;
  const voiceOnRef = useRef(false);
  voiceOnRef.current = voice.listening;
  const autoApproveRef = useRef<string[] | null>(null);

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

  const chatLlm = useCallback(() => {
    const profile = llmOverridesFromProfile(getActiveLlmProfile()) || {};
    return {
      ...profile,
      model: model || profile.model,
      reasoning_budget: reasoningBudget,
    };
  }, [model, reasoningBudget]);

  const chooseModel = (value: string) => {
    setModel(value);
    localStorage.setItem(`${storageKey}:model`, value);
  };

  const chooseBudget = (value: number) => {
    setReasoningBudget(value);
    localStorage.setItem(`${storageKey}:reasoningBudget`, String(value));
  };

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
    voiceApiRef.current.cancelSpeak();
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
    leftoverRef.current = [];
    midRunUsersRef.current = [];
  };

  const paint = (request: ChatMsg[], assistant: ChatMsg) => {
    setMessages([...request, ...midRunUsersRef.current, { ...assistant }]);
  };

  const send = async (text: string) => {
    const value = text.trim();
    if (!value || busyRef.current) return;
    setBusy(true);
    busyRef.current = true;
    const controller = new AbortController();
    stopRef.current = controller;
    midRunUsersRef.current = [];
    const history = messagesRef.current.filter((item) => item.role === "user" || item.role === "assistant");
    const lastUser = [...history].reverse().find((item) => item.role === "user");
    const already = lastUser?.content === value;
    const request = already
      ? history
      : [...history, { role: "user" as const, content: value, ts: Date.now() }];
    const assistant: ChatMsg = {
      role: "assistant",
      content: "",
      ts: Date.now(),
      tool_calls: [],
      programs: [],
      proposed_actions: [],
      streaming: true,
    };
    setGoal(null);
    setGoalCollapsed(false);
    paint(request, assistant);
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
          llm: chatLlm(),
          skip_approval: voiceOnRef.current,
        },
        (event) => {
          applyRunGoal(event, setGoal);
          if (
            (event.type === "goal" || (event.type === "tool_call" && event.name === "set_goal")) &&
            voiceOnRef.current
          ) {
            setGoalCollapsed(true);
          }
          if (event.type === "delta") {
            assistant.content += event.content || "";
            pushTranscript({ kind: "text", text: event.content || "" });
            if (voiceOnRef.current) voiceApiRef.current.speakDelta(event.content || "");
          } else if (event.type === "thinking") {
            assistant.thinking = (assistant.thinking || "") + (event.content || "");
            pushTranscript({ kind: "thinking", text: event.content || "" });
          } else if (event.type === "tool_start") {
            applyToolStart(assistant, event);
          } else if (event.type === "tool_call") {
            applyToolResult(assistant, event);
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
            applyTrace(assistant, event);
          } else if (event.type === "approval") {
            threadIdRef.current = event.thread_id || threadIdRef.current;
            const incoming = event.pending_approvals || [];
            if (voiceOnRef.current) {
              autoApproveRef.current = incoming.map((item: { id: string }) => item.id);
            } else {
              setPendingApprovals(incoming);
            }
            awaitingApproval = true;
            assistant.streaming = false;
            if (incoming.length) {
              setGoalCollapsed(false);
            }
          } else if (event.type === "meta") {
            assistant.proposed_actions = event.proposed_actions || [];
            if (event.tool_calls?.length) {
              attachMetaToolImages(assistant, event.tool_calls);
            } else {
              assistant.tool_calls = event.tool_calls || assistant.tool_calls;
            }
            assistant.programs = event.programs || assistant.programs;
            applyTrace(assistant, event);
            setPendingApprovals([]);
          } else if (event.type === "error") assistant.error = event.error;
          paint(request, assistant);
        },
        { signal: controller.signal },
      );
      assistant.streaming = false;
      settlePendingTools(assistant);
      if (controller.signal.aborted) {
        pushTranscript({ kind: "text", text: "\n\n*Stopped by user.*" });
      } else if (!assistant.content && !assistant.error && !awaitingApproval) {
        assistant.content = "I could not produce a response. Try again with a more specific request.";
      }
      paint(request, assistant);
    } catch (error) {
      assistant.streaming = false;
      settlePendingTools(assistant);
      assistant.error = error instanceof Error ? error.message : "Assistant request failed";
      paint(request, assistant);
      toast({ body: assistant.error, type: "error" });
    } finally {
      if (voiceOnRef.current) voiceApiRef.current.flushSpeak();
      stopRef.current = null;
      setBusy(false);
      busyRef.current = false;
      midRunUsersRef.current = [];
      const autoIds = autoApproveRef.current;
      autoApproveRef.current = null;
      if (autoIds?.length) {
        void resumeApprovals(autoIds);
        return;
      }
      const next = leftoverRef.current.shift();
      if (next) void send(next);
    }
  };

  const submitSpoken = async (text: string) => {
    const value = text.trim();
    if (!value) return;
    voiceApiRef.current.cancelSpeak();
    if (pendingRef.current.length) {
      leftoverRef.current.push(value);
      setMessages((current) => [...current, { role: "user", content: value, ts: Date.now() }]);
      return;
    }
    if (busyRef.current && threadIdRef.current) {
      const userMsg: ChatMsg = { role: "user", content: value, ts: Date.now() };
      midRunUsersRef.current = [...midRunUsersRef.current, userMsg];
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last?.role === "assistant" && last.streaming) {
          return [...current.slice(0, -1), userMsg, last];
        }
        return [...current, userMsg];
      });
      try {
        await api("/chat/followup", {
          method: "POST",
          body: JSON.stringify({ thread_id: threadIdRef.current, text: value }),
        });
      } catch (error) {
        midRunUsersRef.current = midRunUsersRef.current.filter((item) => item !== userMsg);
        setMessages((current) => current.filter((item) => item !== userMsg));
        if (error instanceof ApiError && error.status === 409) {
          leftoverRef.current.push(value);
          if (!busyRef.current) void send(value);
          return;
        }
        toast({
          body: error instanceof Error ? error.message : "Could not queue follow-up",
          type: "error",
        });
      }
      return;
    }
    leftoverRef.current = leftoverRef.current.filter((item) => item !== value);
    void send(value);
  };
  const spokenRef = useRef(submitSpoken);
  spokenRef.current = submitSpoken;

  const toggleVoice = () => {
    if (voice.listening) {
      voice.stopVoice();
      return;
    }
    void voice.startVoice(
      (text) => void spokenRef.current(text),
      (message) => toast({ body: message, type: "error" }),
    );
  };

  const resumeApprovals = async (ids: string[]) => {
    const threadId = threadIdRef.current;
    if (!threadId || busyRef.current) return;
    setPendingApprovals([]);
    setBusy(true);
    busyRef.current = true;
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
          llm: chatLlm(),
          thread_id: threadId,
          approved_ids: ids,
          skip_approval: voiceOnRef.current,
        },
        (event) => {
          applyRunGoal(event, setGoal);
          if (event.type === "delta") {
            assistant.content = (assistant.content || "") + (event.content || "");
            const current = assistant.transcript ?? [];
            const last = current[current.length - 1];
            const piece = event.content || "";
            if (last?.kind === "text") last.text += piece;
            else current.push({ kind: "text", text: piece });
            assistant.transcript = current;
            if (voiceOnRef.current) voiceApiRef.current.speakDelta(piece);
          } else if (event.type === "thinking") {
            assistant.thinking = (assistant.thinking || "") + (event.content || "");
          } else if (event.type === "tool_start") {
            applyToolStart(assistant, event);
          } else if (event.type === "tool_call") {
            applyToolResult(assistant, event);
            setPendingApprovals([]);
          } else if (event.type === "thread" && event.thread_id) {
            threadIdRef.current = event.thread_id;
            applyTrace(assistant, event);
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
            applyTrace(assistant, event);
            setPendingApprovals([]);
          } else if (event.type === "error") assistant.error = event.error;
          setMessages([...request.slice(0, -1), { ...assistant }]);
        },
        { signal: controller.signal },
      );
      assistant.streaming = false;
      settlePendingTools(assistant);
      if (voiceOnRef.current) voiceApiRef.current.flushSpeak();
      setMessages([...request.slice(0, -1), { ...assistant }]);
    } catch (error) {
      assistant.streaming = false;
      settlePendingTools(assistant);
      assistant.error = error instanceof Error ? error.message : "Approval resume failed";
      setMessages([...request.slice(0, -1), { ...assistant }]);
      toast({ body: assistant.error, type: "error" });
    } finally {
      stopRef.current = null;
      setBusy(false);
      busyRef.current = false;
      const next = leftoverRef.current.shift();
      if (next) void send(next);
    }
  };

  return (
    <VStack height="100%" style={{ minHeight: 0, height: "100%" }}>
      <HStack gap={3} vAlign="center" style={{ padding: "12px 20px" }}>
        {activeProfileName ? (
          <Text type="supporting" color="secondary">
            {activeProfileName}
          </Text>
        ) : null}
        <Button label="Clear" onClick={() => { setMessages([]); setGoal(null); setGoalCollapsed(false); setPendingApprovals([]); }} />
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
          <VStack gap={2} style={{ width: "100%" }}>
            {voice.listening ? null : (
              <GoalBubble
                goal={goal}
                approvals={pendingApprovals}
                collapsed={goalCollapsed}
                onToggle={() => setGoalCollapsed((value) => !value)}
                onApprove={() => {
                  if (!pendingApprovals.length) return;
                  setGoalCollapsed(true);
                  void resumeApprovals(pendingApprovals.map((item) => item.id));
                }}
                onReject={() => {
                  if (pendingApprovals.every((item) => item.name === "set_goal")) {
                    setGoal(null);
                  }
                  setGoalCollapsed(true);
                  void resumeApprovals([]);
                }}
              />
            )}
            <ChatComposer
            placeholder="Ask about your garden…"
            value={draft}
            onChange={setDraft}
            onSubmit={(value) => {
              setDraft("");
              void send(value);
            }}
            onStop={() => void stopRun()}
            isStopShown={busy || pendingApprovals.length > 0}
            footerActions={
              <HStack gap={3} vAlign="center" wrap="wrap">
                {models.length > 0 ? (
                  <DropdownMenu
                    hasChevron
                    placement="above"
                    menuWidth={320}
                    button={{
                      label: shortModelName(model) || "Model",
                      variant: "ghost",
                      size: "sm",
                    }}
                    items={models.map((id) => ({
                      id,
                      label: id,
                      onClick: () => chooseModel(id),
                    }))}
                  />
                ) : null}
                <ReasoningEffortMenu
                  model={model}
                  budget={reasoningBudget}
                  onChange={chooseBudget}
                />
              </HStack>
            }
            sendActions={
              <>
                {voice.listening || mic.busy ? (
                  <Text type="supporting" color="secondary">
                    {voice.listening ? "Listening…" : "Transcribing…"}
                  </Text>
                ) : null}
                <IconButton
                  label={voice.listening ? "Stop voice mode" : "Voice mode"}
                  icon={<SpeakerWaveIcon style={{ width: 18, height: 18 }} />}
                  size="md"
                  variant={voice.listening ? "primary" : "secondary"}
                  onClick={() => toggleVoice()}
                />
                <IconButton
                  label={
                    voice.listening
                      ? "Voice mode is on"
                      : mic.busy
                        ? "Transcribing…"
                        : mic.recording
                          ? "Stop recording"
                          : mic.liveMic
                            ? "Dictate into the draft"
                            : "Transcribe audio from this computer"
                  }
                  icon={
                    mic.recording ? (
                      <StopIcon style={{ width: 18, height: 18 }} />
                    ) : (
                      <MicrophoneIcon style={{ width: 18, height: 18 }} />
                    )
                  }
                  size="md"
                  variant={mic.recording ? "primary" : "secondary"}
                  isDisabled={voice.listening || mic.busy}
                  onClick={() =>
                    void mic.toggle(
                      llmOverridesFromProfile(getActiveLlmProfile()),
                      (text) =>
                        setDraft((current) =>
                          current.trim() ? `${current.trim()} ${text}` : text,
                        ),
                      (message) => toast({ body: message, type: "error" }),
                    )
                  }
                />
              </>
            }
            sendButton={
              <ChatSendButton
                isStopShown={busy || pendingApprovals.length > 0}
                isDisabled={false}
                onStop={() => void stopRun()}
              />
            }
          />
          </VStack>
        }
      >
        <ChatMessageList isStreaming={busy}>
          {messages.map((message, index) => (
            <ChatMessage
              key={index}
              sender={message.role}
              metadata={
                message.role === "assistant" ? (
                  <MessageMetadata message={message} model={model} />
                ) : undefined
              }
            >
              {message.role === "user" ? (
                <ChatMessageBubble
                  variant="filled"
                  metadata={<MessageMetadata message={message} />}
                >
                  <Markdown density="compact">{resolveChatImages(message.content)}</Markdown>
                </ChatMessageBubble>
              ) : (
                <AssistantContent message={message} />
              )}
            </ChatMessage>
          ))}
        </ChatMessageList>
      </ChatLayout>
      {voice.listening ? (
        <VoiceOverlay
          paused={voice.paused}
          status={
            voice.paused
              ? "On hold"
              : voice.speaking
                ? "Speaking…"
                : busy
                  ? "Working…"
                  : "Listening"
          }
          you={[...messages].reverse().find((item) => item.role === "user")?.content}
          reply={
            [...messages].reverse().find((item) => item.role === "assistant")?.content
          }
          activity={voiceActivity(
            [...messages].reverse().find((item) => item.role === "assistant"),
          )}
          onHold={() => (voice.paused ? voice.resumeVoice() : voice.pauseVoice())}
          onEnd={() => voice.stopVoice()}
        />
      ) : null}
    </VStack>
  );
}
