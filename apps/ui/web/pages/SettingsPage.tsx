import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { Selector } from "@astryxdesign/core/Selector";
import { TextInput } from "@astryxdesign/core/TextInput";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, apiUrl, setApiUrl } from "../api";
import { PageHeader } from "../components/PageHeader";
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
  api_key_configured?: boolean;
  keys_configured?: Record<string, boolean>;
  vertex?: { project?: string | null; location?: string | null };
  vertex_env?: { project?: boolean; location?: boolean };
  voice?: { configured?: boolean; realtime?: boolean; stored?: boolean; env?: boolean };
  env_overrides?: {
    provider?: boolean;
    base_url?: boolean;
    model?: boolean;
    api_key?: boolean;
    timeout_s?: boolean;
    temperature?: boolean;
  };
};

type ChatDraft = {
  provider: string;
  baseUrl: string;
  model: string;
  timeoutS: number;
  temperature: number;
  apiKey: string;
};

const PROVIDER_HELP: Record<string, string> = {
  openai: "ChatGPT API. Create a secret key at platform.openai.com.",
  openrouter: "One key for many models (Claude, GPT, Gemini, …). openrouter.ai",
  vertex: "Uses Google Cloud on this machine. No chat API key — set the project in step 4.",
  local: "Ollama or any OpenAI-compatible server on this network.",
};

const ENV_LOCK_LABEL: Record<string, string> = {
  provider: "PLANNING_LLM_PROVIDER",
  base_url: "PLANNING_LLM_BASE_URL",
  model: "PLANNING_LLM_MODEL",
  api_key: "PLANNING_LLM_API_KEY",
  timeout_s: "PLANNING_LLM_TIMEOUT_S",
  temperature: "PLANNING_LLM_TEMPERATURE",
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
    },
  };
}

function StepCard({
  step,
  title,
  status,
  children,
}: {
  step: number;
  title: string;
  status: "ready" | "needed" | "optional";
  children: ReactNode;
}) {
  const badge =
    status === "ready"
      ? { variant: "success" as const, label: `${step} · ready` }
      : status === "optional"
        ? { variant: "neutral" as const, label: `${step} · optional` }
        : { variant: "warning" as const, label: `${step} · needed` };
  return (
    <Card padding={4}>
      <VStack gap={3}>
        <HStack gap={2}>
          <Badge variant={badge.variant} label={badge.label} />
          <Text weight="semibold">{title}</Text>
        </HStack>
        {children}
      </VStack>
    </Card>
  );
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
    apiKey: "",
  });
  const [voiceDraft, setVoiceDraft] = useState("");
  const [vertexDraft, setVertexDraft] = useState({ project: "", location: "" });
  const [savingChat, setSavingChat] = useState(false);
  const [savingVoice, setSavingVoice] = useState(false);
  const [savingVertex, setSavingVertex] = useState(false);
  const [testingModels, setTestingModels] = useState(false);
  const [checkingApi, setCheckingApi] = useState(false);

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
      apiKey: "",
    }));
    setVertexDraft({
      project: result.vertex?.project || "",
      location: result.vertex?.location || "",
    });
    setVoiceDraft("");
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

  const saveVoice = async () => {
    if (!voiceDraft.trim()) {
      toast({ body: "Paste a Muse Voice key first" });
      return;
    }
    setSavingVoice(true);
    try {
      applyServer(
        await api<ServerLlmSettings>("/settings/llm", {
          method: "PUT",
          body: JSON.stringify({ keys: {}, voice: { api_key: voiceDraft.trim() } }),
        }),
      );
      toast({ body: "Voice key saved on this robot" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Saving voice key failed",
        type: "error",
      });
    } finally {
      setSavingVoice(false);
    }
  };

  const clearVoice = async () => {
    setSavingVoice(true);
    try {
      applyServer(
        await api<ServerLlmSettings>("/settings/llm", {
          method: "PUT",
          body: JSON.stringify({ keys: {}, voice: { api_key: "" } }),
        }),
      );
      toast({ body: "Cleared voice key" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Clearing voice key failed",
        type: "error",
      });
    } finally {
      setSavingVoice(false);
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

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Settings" />
      <Text type="supporting" color="secondary">
        Work top to bottom. Chat and voice keys are saved on this robot and never shown again
        after you save.
      </Text>

      <HStack gap={2} wrap="wrap">
        <Badge variant={robotReady ? "success" : "warning"} label={robotReady ? "Robot connected" : "Robot"} />
        <Badge
          variant={assistantReady ? "success" : "warning"}
          label={assistantReady ? "Chat ready" : "Chat needs a key"}
        />
        <Badge variant={voiceReady ? "success" : "neutral"} label={voiceReady ? "Voice ready" : "Voice optional"} />
        <Badge
          variant={vertexReady ? "success" : "neutral"}
          label={vertexReady ? "Vertex ready" : "Vertex optional"}
        />
      </HStack>

      {activeProfile ? (
        <Banner
          status="warning"
          title="This browser is overriding the robot"
          description={`Profile “${profileLabel(activeProfile)}” is active. The Assistant tab will use it instead of the server settings below. Switch back to “Use server settings” in Advanced if you want everyone to share the same setup.`}
          collapsible={false}
        />
      ) : null}

      <StepCard step={1} title="Connect this browser to the robot" status={robotReady ? "ready" : "needed"}>
        <Text type="supporting" color="secondary">
          On the robot itself this is usually already correct. Change it only if you opened the UI
          from another computer.
        </Text>
        <TextInput label="FarmBot API URL" value={url} onChange={setUrl} />
        <HStack gap={2}>
          <Button
            label="Save connection"
            variant="primary"
            onClick={() => {
              setApiUrl(url);
              void onSaved();
              toast({ body: "API URL saved" });
            }}
          />
          <Button
            label={checkingApi ? "Checking…" : "Health check"}
            onClick={() => void checkApi()}
            isDisabled={checkingApi}
          />
        </HStack>
        <Text type="supporting">
          {farmbot === "unreachable"
            ? "Cannot reach the API — check the URL and that twfarmbot-api is running."
            : `FarmBot ${farmbot} · ${apiUrl()}`}
        </Text>
      </StepCard>

      <StepCard
        step={2}
        title="Chat assistant"
        status={assistantReady ? "ready" : "needed"}
      >
        <Text type="supporting" color="secondary">
          The model that plans and talks on the Assistant tab. Saved for everyone using this robot.
        </Text>
        {lockedNames.length ? (
          <Banner
            status="info"
            title="Some values are locked by the server environment"
            description={`${lockedNames.join(", ")} ${lockedNames.length === 1 ? "is" : "are"} set on the machine. Remove ${lockedNames.length === 1 ? "it" : "them"} from .env to change ${lockedNames.length === 1 ? "it" : "them"} here.`}
            collapsible={false}
          />
        ) : null}
        <Selector
          label="Provider"
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
        <Text type="supporting" color="secondary">
          {PROVIDER_HELP[chatDraft.provider] || ""}
        </Text>
        {chatDraft.provider !== "vertex" ? (
          <TextInput
            label="API base URL"
            value={chatDraft.baseUrl}
            onChange={(baseUrl) => updateChat({ baseUrl })}
            isDisabled={Boolean(locked.base_url)}
            disabledMessage="Locked by PLANNING_LLM_BASE_URL"
          />
        ) : null}
        <TextInput
          label="Default model"
          value={chatDraft.model}
          onChange={(model) => updateChat({ model })}
          isDisabled={Boolean(locked.model)}
          disabledMessage="Locked by PLANNING_LLM_MODEL"
        />
        {chatDraft.provider !== "vertex" ? (
          <VStack gap={2}>
            <TextInput
              label={`${providerLabel(chatDraft.provider)} API key`}
              value={chatDraft.apiKey}
              onChange={(apiKey) => updateChat({ apiKey })}
              type="password"
              placeholder={keyStored ? "Saved on this robot — paste a new key to replace" : "Paste the secret key"}
              isDisabled={Boolean(locked.api_key)}
              disabledMessage="Locked by PLANNING_LLM_API_KEY"
            />
            <Text type="supporting">
              {locked.api_key
                ? "A key is set in the server environment."
                : keyStored
                  ? "A key is saved on this robot."
                  : "No key yet — chat will fail until you save one."}
            </Text>
          </VStack>
        ) : null}
        <HStack gap={3}>
          <NumberInput
            label="Timeout (s)"
            value={chatDraft.timeoutS}
            onChange={(timeoutS) => updateChat({ timeoutS: timeoutS ?? chatDraft.timeoutS })}
            isDisabled={Boolean(locked.timeout_s)}
          />
          <NumberInput
            label="Temperature"
            value={chatDraft.temperature}
            onChange={(temperature) =>
              updateChat({ temperature: temperature ?? chatDraft.temperature })
            }
            isDisabled={Boolean(locked.temperature)}
          />
        </HStack>
        <HStack gap={2} wrap="wrap">
          <Button
            label={savingChat ? "Saving…" : "Save assistant"}
            variant="primary"
            onClick={() => void saveChat()}
            isDisabled={savingChat}
          />
          {keyStored && !locked.api_key ? (
            <Button label="Clear key" onClick={() => void clearChatKey()} isDisabled={savingChat} />
          ) : null}
          <Button
            label={testingModels ? "Testing…" : "Test connection"}
            onClick={() => void testLlmConnection()}
            isDisabled={testingModels}
          />
        </HStack>
      </StepCard>

      <StepCard step={3} title="Voice" status={voiceReady ? "ready" : "optional"}>
        <Text type="supporting" color="secondary">
          The Voice button uses OpenAI Realtime: server-side listening plus instant spoken
          replies. It needs a native OpenAI key from step 2 (OpenRouter keys cannot do this).
        </Text>
        {voiceReady ? (
          <Text type="supporting">OpenAI Realtime is ready. No extra voice key is required.</Text>
        ) : (
          <Banner
            status="warning"
            title="Save an OpenAI key in step 2"
            description="Voice will stay off until a native OpenAI API key is stored. You can still type in the Assistant tab."
            collapsible={false}
          />
        )}
        <Text type="supporting" color="secondary">
          Optional: a Muse Voice key is only used if OpenAI is unavailable for the dictate
          button.
        </Text>
        <TextInput
          label="Muse Voice API key (optional fallback)"
          value={voiceDraft}
          onChange={setVoiceDraft}
          type="password"
          placeholder={
            serverLlm?.voice?.stored
              ? "Saved on this robot — paste a new key to replace"
              : "Not required when OpenAI is configured"
          }
          isDisabled={Boolean(serverLlm?.voice?.env)}
          disabledMessage="Locked by MUSE_VOICE_API_KEY"
        />
        <HStack gap={2}>
          <Button
            label={savingVoice ? "Saving…" : "Save fallback key"}
            onClick={() => void saveVoice()}
            isDisabled={savingVoice || Boolean(serverLlm?.voice?.env)}
          />
          {serverLlm?.voice?.stored && !serverLlm.voice.env ? (
            <Button label="Clear" onClick={() => void clearVoice()} isDisabled={savingVoice} />
          ) : null}
        </HStack>
      </StepCard>

      <StepCard step={4} title="Vertex AI" status={vertexReady ? "ready" : "optional"}>
        <Text type="supporting" color="secondary">
          Only needed if the chat provider is Vertex AI. Enable the Vertex AI API on the GCP
          project and give this machine Google credentials (service-account JSON at
          GOOGLE_APPLICATION_CREDENTIALS). The project id is not secret.
        </Text>
        <TextInput
          label="GCP project id"
          value={vertexDraft.project}
          onChange={(project) => setVertexDraft((current) => ({ ...current, project }))}
          placeholder="my-gcp-project"
          isDisabled={Boolean(serverLlm?.vertex_env?.project)}
          disabledMessage="Locked by GOOGLE_CLOUD_PROJECT"
        />
        <TextInput
          label="Location"
          value={vertexDraft.location}
          onChange={(location) => setVertexDraft((current) => ({ ...current, location }))}
          placeholder={serverLlm?.vertex?.location || "global"}
          isDisabled={Boolean(serverLlm?.vertex_env?.location)}
          disabledMessage="Locked by GOOGLE_CLOUD_LOCATION"
        />
        <Button
          label={savingVertex ? "Saving…" : "Save Vertex settings"}
          variant="primary"
          onClick={() => void saveVertexSettings()}
          isDisabled={savingVertex}
        />
      </StepCard>

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">Advanced · this browser only</Text>
          <Text type="supporting" color="secondary">
            Override the robot for yourself — for example to try another model — without changing
            what others use. Keys stay in this browser and never reach the server store.
          </Text>
          <Selector
            label="Active profile"
            options={profileOptions}
            value={activeProfileLabel}
            onChange={(label) => {
              if (label === "Use server settings") {
                updateSettings({ ...settings, activeLlmProfileId: null });
                return;
              }
              const profile = settings.llmProfiles.find((item) => profileLabel(item) === label);
              if (profile) updateSettings({ ...settings, activeLlmProfileId: profile.id });
            }}
          />
          <HStack gap={2} wrap="wrap">
            <Button label="New profile" onClick={addProfile} />
            {activeProfile ? (
              <>
                <Button label="Save profile" variant="primary" onClick={saveProfile} />
                <Button label="Delete profile" onClick={deleteProfile} />
              </>
            ) : null}
          </HStack>
          {draft ? (
            <VStack gap={3}>
              <TextInput label="Profile name" value={draft.name} onChange={(name) => updateDraft({ name })} />
              <Selector
                label="Provider"
                options={LLM_PROVIDERS.map((item) => item.label)}
                value={providerLabel(draft.provider)}
                onChange={(label) => {
                  const provider = LLM_PROVIDERS.find((item) => item.label === label)?.id || "openai";
                  const defaults = PROVIDER_DEFAULTS[provider] || PROVIDER_DEFAULTS.openai;
                  updateDraft({ provider, baseUrl: defaults.baseUrl, model: defaults.model });
                }}
              />
              <TextInput
                label="Base URL"
                value={draft.baseUrl}
                onChange={(baseUrl) => updateDraft({ baseUrl })}
                placeholder={draft.provider === "vertex" ? "auto: built from project + location" : undefined}
              />
              {draft.provider === "vertex" ? (
                <Text type="supporting" color="secondary">
                  Vertex AI uses the server&apos;s Google credentials plus the project in step 4.
                </Text>
              ) : (
                <TextInput
                  label="API key"
                  value={draft.apiKey}
                  onChange={(apiKey) => updateDraft({ apiKey })}
                  type="password"
                />
              )}
              <TextInput
                label="Default model"
                value={draft.model}
                onChange={(model) => updateDraft({ model })}
              />
              <HStack gap={3}>
                <NumberInput
                  label="Timeout (s)"
                  value={draft.timeoutS}
                  onChange={(timeoutS) => updateDraft({ timeoutS: timeoutS ?? draft.timeoutS })}
                />
                <NumberInput
                  label="Temperature"
                  value={draft.temperature}
                  onChange={(temperature) =>
                    updateDraft({ temperature: temperature ?? draft.temperature })
                  }
                />
              </HStack>
            </VStack>
          ) : (
            <Text color="secondary">
              Using the robot settings from steps 2–4. Create a profile only if you need a
              personal override in this browser.
            </Text>
          )}
        </VStack>
      </Card>
    </VStack>
  );
}
