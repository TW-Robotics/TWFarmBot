import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { List, ListItem } from "@astryxdesign/core/List";
import { Tab, TabList } from "@astryxdesign/core/TabList";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { StatusDot } from "@astryxdesign/core/StatusDot";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { Selector } from "@astryxdesign/core/Selector";
import { TextInput } from "@astryxdesign/core/TextInput";
import { Heading, Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, apiUrl, setApiUrl } from "../api";
import { setDevicePref, useDevice, type DevicePref } from "../device";
import { ReasoningEffortSelector } from "../components/ReasoningEffortControl";
import {
  AdjustmentsHorizontalIcon,
  ChatBubbleLeftRightIcon,
  CloudIcon,
  ComputerDesktopIcon,
  GlobeAltIcon,
  MicrophoneIcon,
} from "@heroicons/react/24/outline";
import {
  AppSettings,
  LLM_PROVIDERS,
  LlmProfile,
  PROVIDER_DEFAULTS,
  SETTINGS_CHANGED_EVENT,
  createEmptyProfile,
  loadSettings,
  profileLabel,
  saveSettings,
} from "../settings";

type ServerLlmSettings = {
  provider?: string;
  base_url?: string;
  model?: string;
  timeout_s?: number;
  temperature?: number;
  reasoning_budget?: number;
  api_key_configured?: boolean;
  keys_configured?: Record<string, boolean>;
  vertex?: { project?: string | null; location?: string | null };
  vertex_env?: { project?: boolean; location?: boolean };
  voice?: { configured?: boolean; realtime?: boolean };
  env_overrides?: {
    provider?: boolean;
    base_url?: boolean;
    model?: boolean;
    api_key?: boolean;
    timeout_s?: boolean;
    temperature?: boolean;
    reasoning_budget?: boolean;
  };
};

type ChatDraft = {
  provider: string;
  baseUrl: string;
  model: string;
  timeoutS: number;
  temperature: number;
  reasoningBudget: number;
  apiKey: string;
};

const PROVIDER_HELP: Record<string, string> = {
  openai: "ChatGPT API. Create a secret key at platform.openai.com.",
  openrouter: "One key for many models (Claude, GPT, Gemini, …). openrouter.ai",
  vertex: "Uses Google Cloud on this machine. No chat API key — set the project in step 4.",
  local: "LFM3 via scripts/lfm3_server.sh, Ollama, or any OpenAI-compatible server. No key needed.",
};

const ENV_LOCK_LABEL: Record<string, string> = {
  provider: "PLANNING_LLM_PROVIDER",
  base_url: "PLANNING_LLM_BASE_URL",
  model: "PLANNING_LLM_MODEL",
  api_key: "PLANNING_LLM_API_KEY",
  timeout_s: "PLANNING_LLM_TIMEOUT_S",
  temperature: "PLANNING_LLM_TEMPERATURE",
  reasoning_budget: "PLANNING_LLM_REASONING_BUDGET",
};

function providerLabel(id: string): string {
  return LLM_PROVIDERS.find((item) => item.id === id)?.label || id;
}

function chatReady(server: ServerLlmSettings | null, provider: string): boolean {
  if (provider === "vertex" || provider === "local") return true;
  return Boolean(server?.api_key_configured || server?.keys_configured?.[provider]);
}

function profileToPayload(profile: LlmProfile) {
  return {
    llm: {
      provider: profile.provider,
      base_url: profile.baseUrl,
      api_key: profile.apiKey.trim() || undefined,
      model: profile.model,
      timeout_s: profile.timeoutS,
      temperature: profile.temperature,
      reasoning_budget: profile.reasoningBudget,
    },
  };
}

export function SettingsPage({
  farmbot,
  onSaved,
}: {
  farmbot: string;
  onSaved: () => Promise<void>;
}) {
  const toast = useToast();
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [url, setUrl] = useState(apiUrl());
  const [serverLlm, setServerLlm] = useState<ServerLlmSettings | null>(null);
  const [chatDraft, setChatDraft] = useState<ChatDraft>({
    provider: "openai",
    baseUrl: PROVIDER_DEFAULTS.openai.baseUrl,
    model: PROVIDER_DEFAULTS.openai.model,
    timeoutS: 120,
    temperature: 0,
    reasoningBudget: 8192,
    apiKey: "",
  });
  const [vertexDraft, setVertexDraft] = useState({ project: "", location: "" });
  const [savingChat, setSavingChat] = useState(false);
  const [savingVertex, setSavingVertex] = useState(false);
  const [testingModels, setTestingModels] = useState(false);
  const [checkingApi, setCheckingApi] = useState(false);
  const [section, setSection] = useState<SectionId>("general");
  const device = useDevice();

  const activeProfile = useMemo(
    () => settings.llmProfiles.find((profile) => profile.id === settings.activeLlmProfileId) ?? null,
    [settings],
  );
  const [draft, setDraft] = useState<LlmProfile | null>(activeProfile);

  useEffect(() => {
    setDraft(activeProfile ? { ...activeProfile } : null);
  }, [activeProfile?.id]);

  const applyServer = useCallback((result: ServerLlmSettings) => {
    setServerLlm(result);
    setChatDraft((current) => ({
      provider: result.provider || current.provider,
      baseUrl: result.base_url || current.baseUrl,
      model: result.model || current.model,
      timeoutS: result.timeout_s ?? current.timeoutS,
      temperature: result.temperature ?? current.temperature,
      reasoningBudget: result.reasoning_budget ?? current.reasoningBudget,
      apiKey: "",
    }));
    setVertexDraft({
      project: result.vertex?.project || "",
      location: result.vertex?.location || "",
    });
  }, []);

  const refreshServerLlm = useCallback(async () => {
    try {
      applyServer(await api<ServerLlmSettings>("/settings/llm"));
    } catch {
      setServerLlm(null);
    }
  }, [applyServer]);

  useEffect(() => {
    void refreshServerLlm();
  }, [refreshServerLlm]);

  const locked = serverLlm?.env_overrides || {};
  const lockedNames = Object.entries(ENV_LOCK_LABEL)
    .filter(([field]) => locked[field as keyof typeof locked])
    .map(([, name]) => name);

  const robotReady = farmbot !== "unreachable" && farmbot !== "unknown";
  const assistantReady = chatReady(serverLlm, chatDraft.provider);
  const voiceReady = Boolean(serverLlm?.voice?.realtime);
  const vertexReady = Boolean(serverLlm?.vertex?.project);

  const updateSettings = (next: AppSettings) => {
    setSettings(next);
    saveSettings(next);
  };

  const notifyAssistant = () => {
    window.dispatchEvent(new CustomEvent(SETTINGS_CHANGED_EVENT));
  };

  const updateChat = (patch: Partial<ChatDraft>) => {
    setChatDraft((current) => ({ ...current, ...patch }));
  };

  const saveChat = async () => {
    setSavingChat(true);
    try {
      const keys: Record<string, string> = {};
      if (chatDraft.apiKey.trim() && chatDraft.provider !== "vertex") {
        keys[chatDraft.provider] = chatDraft.apiKey.trim();
      }
      applyServer(
        await api<ServerLlmSettings>("/settings/llm", {
          method: "PUT",
          body: JSON.stringify({
            keys,
            planning: {
              provider: chatDraft.provider,
              base_url: chatDraft.baseUrl,
              model: chatDraft.model,
              timeout_s: chatDraft.timeoutS,
              temperature: chatDraft.temperature,
              reasoning_budget: chatDraft.reasoningBudget,
            },
          }),
        }),
      );
      notifyAssistant();
      toast({ body: "Assistant settings saved on this robot" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Saving assistant settings failed",
        type: "error",
      });
    } finally {
      setSavingChat(false);
    }
  };

  const clearChatKey = async () => {
    setSavingChat(true);
    try {
      applyServer(
        await api<ServerLlmSettings>("/settings/llm", {
          method: "PUT",
          body: JSON.stringify({ keys: { [chatDraft.provider]: "" } }),
        }),
      );
      toast({ body: `Cleared ${providerLabel(chatDraft.provider)} key` });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Clearing key failed",
        type: "error",
      });
    } finally {
      setSavingChat(false);
    }
  };

  const saveVertexSettings = async () => {
    setSavingVertex(true);
    try {
      applyServer(
        await api<ServerLlmSettings>("/settings/llm", {
          method: "PUT",
          body: JSON.stringify({
            keys: {},
            vertex: {
              project: vertexDraft.project.trim(),
              location: vertexDraft.location.trim(),
            },
          }),
        }),
      );
      toast({ body: "Vertex AI settings saved" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Saving Vertex settings failed",
        type: "error",
      });
    } finally {
      setSavingVertex(false);
    }
  };

  const testLlmConnection = async () => {
    setTestingModels(true);
    try {
      const result = draft
        ? await api<{ models?: string[]; provider?: string; api_key_configured?: boolean }>(
            "/models/list",
            { method: "POST", body: JSON.stringify(profileToPayload(draft)) },
          )
        : await api<{ models?: string[]; provider?: string; api_key_configured?: boolean }>(
            "/models/list",
            {
              method: "POST",
              body: JSON.stringify({
                llm: {
                  provider: chatDraft.provider,
                  base_url: chatDraft.baseUrl || undefined,
                  api_key: chatDraft.apiKey.trim() || undefined,
                  model: chatDraft.model,
                  timeout_s: chatDraft.timeoutS,
                  temperature: chatDraft.temperature,
                  reasoning_budget: chatDraft.reasoningBudget,
                },
              }),
            },
          );
      const count = result.models?.length ?? 0;
      const needsKey =
        result.api_key_configured === false && result.provider !== "vertex";
      toast({
        body: `${result.provider || "provider"}: ${count} models available${
          needsKey ? " (no API key configured)" : ""
        }`,
      });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Model discovery failed",
        type: "error",
      });
    } finally {
      setTestingModels(false);
    }
  };

  const checkApi = async () => {
    setCheckingApi(true);
    try {
      await api("/health");
      await onSaved();
      toast({ body: "API reachable" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "API unreachable",
        type: "error",
      });
    } finally {
      setCheckingApi(false);
    }
  };

  const profileOptions = useMemo(
    () => ["Use server settings", ...settings.llmProfiles.map((profile) => profileLabel(profile))],
    [settings.llmProfiles],
  );
  const activeProfileLabel = activeProfile ? profileLabel(activeProfile) : "Use server settings";

  const updateDraft = (patch: Partial<LlmProfile>) => {
    setDraft((current) => (current ? { ...current, ...patch } : current));
  };

  const saveProfile = () => {
    if (!draft) return;
    const trimmed = { ...draft, name: draft.name.trim() || profileLabel(draft) };
    const exists = settings.llmProfiles.some((profile) => profile.id === trimmed.id);
    const llmProfiles = exists
      ? settings.llmProfiles.map((profile) => (profile.id === trimmed.id ? trimmed : profile))
      : [...settings.llmProfiles, trimmed];
    updateSettings({ ...settings, activeLlmProfileId: trimmed.id, llmProfiles });
    toast({ body: `Saved browser profile "${trimmed.name}"` });
  };

  const addProfile = () => {
    const profile = createEmptyProfile(`Profile ${settings.llmProfiles.length + 1}`);
    setDraft(profile);
    updateSettings({
      ...settings,
      activeLlmProfileId: profile.id,
      llmProfiles: [...settings.llmProfiles, profile],
    });
  };

  const deleteProfile = () => {
    if (!activeProfile) return;
    updateSettings({
      activeLlmProfileId: null,
      llmProfiles: settings.llmProfiles.filter((profile) => profile.id !== activeProfile.id),
    });
    setDraft(null);
    toast({ body: "Browser profile deleted" });
  };


  const keyStored = Boolean(serverLlm?.keys_configured?.[chatDraft.provider]);

  const sections: { id: SectionId; label: string; icon: typeof GlobeAltIcon; ready?: boolean }[] = [
    { id: "general", label: "General", icon: AdjustmentsHorizontalIcon },
    { id: "assistant", label: "Assistant", icon: ChatBubbleLeftRightIcon, ready: assistantReady },
    { id: "voice", label: "Voice", icon: MicrophoneIcon, ready: voiceReady },
    { id: "vertex", label: "Vertex AI", icon: CloudIcon, ready: vertexReady },
    { id: "profile", label: "Browser profile", icon: ComputerDesktopIcon },
  ];
  const current = sections.find((item) => item.id === section) ?? sections[0];

  const general = (
    <>
      <SettingsGroup title="Robot connection">
        <SettingRow
          label="FarmBot API URL"
          description="On the robot itself this is usually correct. Change it only if you opened the UI from another computer."
        >
          <TextInput label="FarmBot API URL" isLabelHidden width={320} value={url} onChange={setUrl} />
        </SettingRow>
        <SettingRow
          label="Status"
          description={
            farmbot === "unreachable"
              ? "Cannot reach the API — check the URL and that twfarmbot-api is running."
              : `FarmBot ${farmbot} · ${apiUrl()}`
          }
        >
          <HStack gap={2}>
            <Button
              label={checkingApi ? "Checking…" : "Health check"}
              onClick={() => void checkApi()}
              isDisabled={checkingApi}
            />
            <Button
              label="Save"
              variant="primary"
              onClick={() => {
                setApiUrl(url);
                void onSaved();
                toast({ body: "API URL saved" });
              }}
            />
          </HStack>
        </SettingRow>
      </SettingsGroup>
      <SettingsGroup title="This device">
        <SettingRow
          label="Layout"
          description={`Tablet layout collapses the sidebars and makes the Assistant voice-only. Detected: ${device.detectedTablet ? "tablet (touch only)" : "desktop"}.`}
        >
          <Selector
            label="Layout"
            isLabelHidden
            width={CONTROL_WIDTH}
            options={Object.values(DEVICE_LABELS)}
            value={DEVICE_LABELS[device.pref]}
            onChange={(label) => {
              const next = (Object.keys(DEVICE_LABELS) as DevicePref[]).find(
                (key) => DEVICE_LABELS[key] === label,
              );
              if (next) setDevicePref(next);
            }}
          />
        </SettingRow>
      </SettingsGroup>
      <SettingsGroup title="Overview">
        <StatusRow label="Robot" description="Connection to the FarmBot API." ready={robotReady} />
        <StatusRow
          label="Chat assistant"
          description={`${providerLabel(chatDraft.provider)} · ${chatDraft.model}`}
          ready={assistantReady}
        />
        <StatusRow label="Voice" description="OpenAI Realtime voice and dictation." ready={voiceReady} optional />
        <StatusRow label="Vertex AI" description="Only needed for the Vertex provider." ready={vertexReady} optional />
      </SettingsGroup>
    </>
  );

  const assistant = (
    <>
      {lockedNames.length ? (
        <Banner
          status="info"
          title="Some values are locked by the server environment"
          description={`${lockedNames.join(", ")} ${lockedNames.length === 1 ? "is" : "are"} set on the machine. Remove ${lockedNames.length === 1 ? "it" : "them"} from .env to change ${lockedNames.length === 1 ? "it" : "them"} here.`}
          collapsible={false}
        />
      ) : null}
      <SettingsGroup title="Model">
        <SettingRow label="Provider" description={PROVIDER_HELP[chatDraft.provider] || ""}>
          <Selector
            label="Provider"
            isLabelHidden
            width={CONTROL_WIDTH}
            options={LLM_PROVIDERS.map((item) => item.label)}
            value={providerLabel(chatDraft.provider)}
            isDisabled={Boolean(locked.provider)}
            disabledMessage="Locked by PLANNING_LLM_PROVIDER"
            onChange={(label) => {
              const provider = LLM_PROVIDERS.find((item) => item.label === label)?.id || "openai";
              const defaults = PROVIDER_DEFAULTS[provider] || PROVIDER_DEFAULTS.openai;
              updateChat({ provider, baseUrl: defaults.baseUrl, model: defaults.model });
            }}
          />
        </SettingRow>
        {chatDraft.provider !== "vertex" ? (
          <SettingRow label="API base URL" description="OpenAI-compatible endpoint for this provider.">
            <TextInput
              label="API base URL"
              isLabelHidden
              width={CONTROL_WIDTH}
              value={chatDraft.baseUrl}
              onChange={(baseUrl) => updateChat({ baseUrl })}
              isDisabled={Boolean(locked.base_url)}
              disabledMessage="Locked by PLANNING_LLM_BASE_URL"
            />
          </SettingRow>
        ) : null}
        <SettingRow label="Default model" description="Used by the Assistant tab unless you pick another one there.">
          <TextInput
            label="Default model"
            isLabelHidden
            width={CONTROL_WIDTH}
            value={chatDraft.model}
            onChange={(model) => updateChat({ model })}
            isDisabled={Boolean(locked.model)}
            disabledMessage="Locked by PLANNING_LLM_MODEL"
          />
        </SettingRow>
        {chatDraft.provider !== "vertex" ? (
          <SettingRow
            label={`${providerLabel(chatDraft.provider)} API key`}
            description={
              locked.api_key
                ? "A key is set in the server environment."
                : chatDraft.provider === "local"
                  ? "Optional — only if your local server requires one."
                  : keyStored
                  ? "A key is saved on this robot. Paste a new one to replace it."
                  : "No key yet — chat will fail until you save one."
            }
          >
            <HStack gap={2}>
              <TextInput
                label="API key"
                isLabelHidden
                width={keyStored && !locked.api_key ? 220 : CONTROL_WIDTH}
                value={chatDraft.apiKey}
                onChange={(apiKey) => updateChat({ apiKey })}
                type="password"
                placeholder={keyStored ? "••••••••" : "Paste the secret key"}
                isDisabled={Boolean(locked.api_key)}
                disabledMessage="Locked by PLANNING_LLM_API_KEY"
              />
              {keyStored && !locked.api_key ? (
                <Button label="Clear" onClick={() => void clearChatKey()} isDisabled={savingChat} />
              ) : null}
            </HStack>
          </SettingRow>
        ) : null}
      </SettingsGroup>
      <SettingsGroup title="Behaviour">
        <SettingRow label="Reasoning" description="Effort level, when the selected model supports it.">
          <ReasoningEffortSelector
            isLabelHidden
            width={CONTROL_WIDTH}
            model={chatDraft.model}
            budget={chatDraft.reasoningBudget}
            isDisabled={Boolean(locked.reasoning_budget)}
            onChange={(reasoningBudget) => updateChat({ reasoningBudget })}
          />
        </SettingRow>
        <SettingRow label="Temperature" description="0 keeps planning deterministic.">
          <NumberInput
            label="Temperature"
            isLabelHidden
            width={120}
            value={chatDraft.temperature}
            onChange={(temperature) => updateChat({ temperature: temperature ?? chatDraft.temperature })}
            isDisabled={Boolean(locked.temperature)}
          />
        </SettingRow>
        <SettingRow label="Timeout" description="Seconds to wait for a model response. Raise for slow models.">
          <NumberInput
            label="Timeout (s)"
            isLabelHidden
            width={120}
            value={chatDraft.timeoutS}
            onChange={(timeoutS) => updateChat({ timeoutS: timeoutS ?? chatDraft.timeoutS })}
            isDisabled={Boolean(locked.timeout_s)}
          />
        </SettingRow>
      </SettingsGroup>
      <HStack gap={2} hAlign="end">
        <Button
          label={testingModels ? "Testing…" : "Test connection"}
          onClick={() => void testLlmConnection()}
          isDisabled={testingModels}
        />
        <Button
          label={savingChat ? "Saving…" : "Save"}
          variant="primary"
          onClick={() => void saveChat()}
          isDisabled={savingChat}
        />
      </HStack>
    </>
  );

  const voice = (
    <SettingsGroup title="Realtime voice">
      <StatusRow
        label="OpenAI Realtime"
        description="Voice and dictate use the native OpenAI key from Assistant. OpenRouter keys cannot do this."
        ready={voiceReady}
        optional
      />
      {!voiceReady ? (
        <SettingRow label="Set up" description="Save a native OpenAI API key in Assistant. You can still type meanwhile.">
          <Button label="Open Assistant settings" onClick={() => setSection("assistant")} />
        </SettingRow>
      ) : null}
    </SettingsGroup>
  );

  const vertex = (
    <>
      <SettingsGroup title="Google Cloud">
        <SettingRow
          label="GCP project id"
          description="Enable the Vertex AI API on this project. The id is not secret."
        >
          <TextInput
            label="GCP project id"
            isLabelHidden
            width={CONTROL_WIDTH}
            value={vertexDraft.project}
            onChange={(project) => setVertexDraft((v) => ({ ...v, project }))}
            placeholder="my-gcp-project"
            isDisabled={Boolean(serverLlm?.vertex_env?.project)}
            disabledMessage="Locked by GOOGLE_CLOUD_PROJECT"
          />
        </SettingRow>
        <SettingRow label="Location" description="Vertex region. Defaults to global.">
          <TextInput
            label="Location"
            isLabelHidden
            width={CONTROL_WIDTH}
            value={vertexDraft.location}
            onChange={(location) => setVertexDraft((v) => ({ ...v, location }))}
            placeholder={serverLlm?.vertex?.location || "global"}
            isDisabled={Boolean(serverLlm?.vertex_env?.location)}
            disabledMessage="Locked by GOOGLE_CLOUD_LOCATION"
          />
        </SettingRow>
        <SettingRow
          label="Credentials"
          description="This machine needs a service-account JSON at GOOGLE_APPLICATION_CREDENTIALS."
        >
          <Badge variant={vertexReady ? "success" : "neutral"} label={vertexReady ? "Ready" : "Optional"} />
        </SettingRow>
      </SettingsGroup>
      <HStack gap={2} hAlign="end">
        <Button
          label={savingVertex ? "Saving…" : "Save"}
          variant="primary"
          onClick={() => void saveVertexSettings()}
          isDisabled={savingVertex}
        />
      </HStack>
    </>
  );

  const profile = (
    <>
      {activeProfile ? (
        <Banner
          status="warning"
          title="This browser is overriding the robot"
          description={`Profile “${profileLabel(activeProfile)}” is active. The Assistant tab uses it instead of the robot settings.`}
          collapsible={false}
        />
      ) : null}
      <SettingsGroup title="Profile">
        <SettingRow
          label="Active profile"
          description="Override the robot for yourself only. Keys stay in this browser."
        >
          <Selector
            label="Active profile"
            isLabelHidden
            width={CONTROL_WIDTH}
            options={profileOptions}
            value={activeProfileLabel}
            onChange={(label) => {
              if (label === "Use server settings") {
                updateSettings({ ...settings, activeLlmProfileId: null });
                return;
              }
              const match = settings.llmProfiles.find((item) => profileLabel(item) === label);
              if (match) updateSettings({ ...settings, activeLlmProfileId: match.id });
            }}
          />
        </SettingRow>
        <SettingRow label="Manage" description="Create, save or delete browser-only profiles.">
          <HStack gap={2}>
            <Button label="New" onClick={addProfile} />
            {activeProfile ? <Button label="Delete" onClick={deleteProfile} /> : null}
          </HStack>
        </SettingRow>
      </SettingsGroup>
      {draft ? (
        <>
          <SettingsGroup title="Profile model">
            <SettingRow label="Name" description="Shown in the profile picker.">
              <TextInput label="Profile name" isLabelHidden width={CONTROL_WIDTH} value={draft.name} onChange={(name) => updateDraft({ name })} />
            </SettingRow>
            <SettingRow label="Provider" description={PROVIDER_HELP[draft.provider] || ""}>
              <Selector
                label="Provider"
                isLabelHidden
                width={CONTROL_WIDTH}
                options={LLM_PROVIDERS.map((item) => item.label)}
                value={providerLabel(draft.provider)}
                onChange={(label) => {
                  const provider = LLM_PROVIDERS.find((item) => item.label === label)?.id || "openai";
                  const defaults = PROVIDER_DEFAULTS[provider] || PROVIDER_DEFAULTS.openai;
                  updateDraft({ provider, baseUrl: defaults.baseUrl, model: defaults.model });
                }}
              />
            </SettingRow>
            <SettingRow label="Base URL" description="OpenAI-compatible endpoint.">
              <TextInput
                label="Base URL"
                isLabelHidden
                width={CONTROL_WIDTH}
                value={draft.baseUrl}
                onChange={(baseUrl) => updateDraft({ baseUrl })}
                placeholder={draft.provider === "vertex" ? "auto: project + location" : undefined}
              />
            </SettingRow>
            {draft.provider !== "vertex" ? (
              <SettingRow label="API key" description="Stored only in this browser.">
                <TextInput label="API key" isLabelHidden width={CONTROL_WIDTH} type="password" value={draft.apiKey} onChange={(apiKey) => updateDraft({ apiKey })} />
              </SettingRow>
            ) : null}
            <SettingRow label="Default model" description="Model id for this profile.">
              <TextInput label="Default model" isLabelHidden width={CONTROL_WIDTH} value={draft.model} onChange={(model) => updateDraft({ model })} />
            </SettingRow>
            <SettingRow label="Reasoning" description="Effort level, when supported.">
              <ReasoningEffortSelector
            isLabelHidden
            width={CONTROL_WIDTH}
                model={draft.model}
                budget={draft.reasoningBudget}
                onChange={(reasoningBudget) => updateDraft({ reasoningBudget })}
              />
            </SettingRow>
            <SettingRow label="Temperature" description="0 keeps planning deterministic.">
              <NumberInput label="Temperature" isLabelHidden width={120} value={draft.temperature} onChange={(temperature) => updateDraft({ temperature: temperature ?? draft.temperature })} />
            </SettingRow>
            <SettingRow label="Timeout" description="Seconds to wait for a response.">
              <NumberInput label="Timeout (s)" isLabelHidden width={120} value={draft.timeoutS} onChange={(timeoutS) => updateDraft({ timeoutS: timeoutS ?? draft.timeoutS })} />
            </SettingRow>
          </SettingsGroup>
          <HStack gap={2} hAlign="end">
            <Button
              label={testingModels ? "Testing…" : "Test connection"}
              onClick={() => void testLlmConnection()}
              isDisabled={testingModels}
            />
            <Button label="Save profile" variant="primary" onClick={saveProfile} />
          </HStack>
        </>
      ) : null}
    </>
  );

  const content = { general, assistant, voice, vertex, profile }[current.id];

  return (
    <div className="settings-page">
      <header className="settings-header">
        <Heading level={2}>Settings</Heading>
        <TabList value={current.id} onChange={(id) => setSection(id as SectionId)}>
          {sections.map((item) => (
            <Tab
              key={item.id}
              value={item.id}
              label={item.label}
              icon={<item.icon style={{ width: 16, height: 16 }} />}
              endContent={
                item.ready === undefined ? null : (
                  <StatusDot
                    variant={item.ready ? "success" : "neutral"}
                    label={item.ready ? "Ready" : "Not set up"}
                  />
                )
              }
            />
          ))}
        </TabList>
      </header>
      <div className="settings-scroll">
        <div className="settings-content">
          <VStack gap={6}>{content}</VStack>
        </div>
      </div>
    </div>
  );
}

const CONTROL_WIDTH = 300;

const DEVICE_LABELS: Record<DevicePref, string> = {
  auto: "Detect automatically",
  tablet: "Tablet",
  desktop: "Desktop",
};

type SectionId = "general" | "assistant" | "voice" | "vertex" | "profile";

function SettingsGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <VStack gap={3}>
      <Heading level={3}>{title}</Heading>
      <Card padding={0}>
        <List hasDividers density="spacious">
          {children}
        </List>
      </Card>
    </VStack>
  );
}

function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <ListItem
      label={label}
      description={
        description ? (
          <Text type="supporting" color="secondary">
            {description}
          </Text>
        ) : undefined
      }
      endContent={<div className="settings-control">{children}</div>}
    />
  );
}

function StatusRow({
  label,
  description,
  ready,
  optional,
}: {
  label: string;
  description: string;
  ready: boolean;
  optional?: boolean;
}) {
  return (
    <SettingRow label={label} description={description}>
      <Badge
        variant={ready ? "success" : optional ? "neutral" : "warning"}
        label={ready ? "Ready" : optional ? "Optional" : "Needs setup"}
      />
    </SettingRow>
  );
}
