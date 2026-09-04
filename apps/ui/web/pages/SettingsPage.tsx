import { useCallback, useEffect, useMemo, useState } from "react";
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
};

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
  const [keyDrafts, setKeyDrafts] = useState<Record<string, string>>({});
  const [savingKeys, setSavingKeys] = useState(false);
  const [testingModels, setTestingModels] = useState(false);

  const activeProfile = useMemo(
    () => settings.llmProfiles.find((profile) => profile.id === settings.activeLlmProfileId) ?? null,
    [settings],
  );

  const [draft, setDraft] = useState<LlmProfile | null>(activeProfile);

  useEffect(() => {
    setDraft(activeProfile ? { ...activeProfile } : null);
  }, [activeProfile?.id]);

  const refreshServerLlm = useCallback(async () => {
    try {
      const result = await api<ServerLlmSettings>("/settings/llm");
      setServerLlm(result);
    } catch {
      setServerLlm(null);
    }
  }, []);

  useEffect(() => {
    void refreshServerLlm();
  }, [refreshServerLlm]);

  const profileOptions = useMemo(
    () => [
      "Server defaults (env / configs/dev.yaml)",
      ...settings.llmProfiles.map((profile) => profileLabel(profile)),
    ],
    [settings.llmProfiles],
  );

  const activeProfileLabel = activeProfile
    ? profileLabel(activeProfile)
    : "Server defaults (env / configs/dev.yaml)";

  const updateSettings = (next: AppSettings) => {
    setSettings(next);
    saveSettings(next);
  };

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
    updateSettings({
      ...settings,
      activeLlmProfileId: trimmed.id,
      llmProfiles,
    });
    toast({ body: `Saved profile "${trimmed.name}"` });
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
    const llmProfiles = settings.llmProfiles.filter((profile) => profile.id !== activeProfile.id);
    updateSettings({
      activeLlmProfileId: null,
      llmProfiles,
    });
    setDraft(null);
    toast({ body: "Profile deleted" });
  };

  const testLlmConnection = async () => {
    setTestingModels(true);
    try {
      const result = draft
        ? await api<{ models?: string[]; provider?: string; api_key_configured?: boolean }>(
            "/models/list",
            { method: "POST", body: JSON.stringify(profileToPayload(draft)) },
          )
        : await api<{ models?: string[]; provider?: string; api_key_configured?: boolean }>("/models");
      const count = result.models?.length ?? 0;
      toast({
        body: `${result.provider || "provider"}: ${count} models available${
          result.api_key_configured === false ? " (no API key configured)" : ""
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
  const saveServerKeys = async () => {
    const keys: Record<string, string> = {};
    for (const [provider, value] of Object.entries(keyDrafts)) {
      if (value.trim()) keys[provider] = value.trim();
    }
    if (Object.keys(keys).length === 0) {
      toast({ body: "Enter at least one key first" });
      return;
    }
    setSavingKeys(true);
    try {
      await api("/settings/llm", {
        method: "PUT",
        body: JSON.stringify({ keys }),
      });
      setKeyDrafts({});
      await refreshServerLlm();
      toast({ body: "Server keys saved" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Saving keys failed",
        type: "error",
      });
    } finally {
      setSavingKeys(false);
    }
  };

  const clearServerKey = async (provider: string) => {
    setSavingKeys(true);
    try {
      await api("/settings/llm", {
        method: "PUT",
        body: JSON.stringify({ keys: { [provider]: "" } }),
      });
      await refreshServerLlm();
      toast({ body: `Cleared ${provider} key` });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Clearing key failed",
        type: "error",
      });
    } finally {
      setSavingKeys(false);
    }
  };

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Settings" />

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">API connection</Text>
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
              label="Health check"
              onClick={() =>
                void api("/health")
                  .then(() => toast({ body: "API reachable" }))
                  .catch((error) => toast({ body: String(error.message), type: "error" }))
              }
            />
          </HStack>
          <Text type="supporting">
            FarmBot {farmbot} · {apiUrl()}
          </Text>
        </VStack>
      </Card>

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">LLM profiles</Text>
          <Text type="supporting" color="secondary">
            Store API keys in this browser and switch providers without editing server env files.
            Keys stay in local storage on this machine only.
          </Text>

          {serverLlm ? (
            <Text type="supporting">
              Server default: {serverLlm.provider} · {serverLlm.model} ·{" "}
              {serverLlm.api_key_configured ? "API key configured" : "no API key in env"}
            </Text>
          ) : null}

          <Selector
            label="Active profile"
            options={profileOptions}
            value={activeProfileLabel}
            onChange={(label) => {
              if (label.startsWith("Server defaults")) {
                updateSettings({ ...settings, activeLlmProfileId: null });
                return;
              }
              const profile = settings.llmProfiles.find((item) => profileLabel(item) === label);
              if (profile) updateSettings({ ...settings, activeLlmProfileId: profile.id });
            }}
          />

          <HStack gap={2}>
            <Button label="New profile" onClick={addProfile} />
            {activeProfile ? (
              <>
                <Button label="Save profile" variant="primary" onClick={saveProfile} />
                <Button label="Delete profile" onClick={deleteProfile} />
              </>
            ) : null}
            <Button
              label={testingModels ? "Testing…" : "Test models"}
              onClick={() => void testLlmConnection()}
              isDisabled={testingModels}
            />
          </HStack>

          {draft ? (
            <VStack gap={3}>
              <TextInput label="Profile name" value={draft.name} onChange={(name) => updateDraft({ name })} />
              <Selector
                label="Provider"
                options={LLM_PROVIDERS.map((item) => item.label)}
                value={LLM_PROVIDERS.find((item) => item.id === draft.provider)?.label || draft.provider}
                onChange={(label) => {
                  const provider = LLM_PROVIDERS.find((item) => item.label === label)?.id || "openai";
                  const defaults = PROVIDER_DEFAULTS[provider] || PROVIDER_DEFAULTS.openai;
                  updateDraft({
                    provider,
                    baseUrl: defaults.baseUrl,
                    model: defaults.model,
                  });
                }}
              />
              <TextInput
                label="Base URL"
                value={draft.baseUrl}
                onChange={(baseUrl) => updateDraft({ baseUrl })}
              />
              <TextInput
                label="API key"
                value={draft.apiKey}
                onChange={(apiKey) => updateDraft({ apiKey })}
                type="password"
              />
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
                  onChange={(temperature) => updateDraft({ temperature: temperature ?? draft.temperature })}
                />
              </HStack>
            </VStack>
          ) : (
            <Text color="secondary">
              Using server defaults from <code>PLANNING_LLM_*</code> and <code>configs/dev.yaml</code>.
              Create a profile to override provider, API key, and model from the UI.
            </Text>
          )}
        </VStack>
      </Card>
      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">Server API keys</Text>
          <Text type="supporting" color="secondary">
            Stored on the server (mode 0600, never committed). Used whenever a
            request has no key of its own — env vars still win when set.
          </Text>
          {LLM_PROVIDERS.map((item) => (
            <HStack gap={2} key={item.id}>
              <TextInput
                label={`${item.label} key`}
                value={keyDrafts[item.id] ?? ""}
                onChange={(value) =>
                  setKeyDrafts((current) => ({ ...current, [item.id]: value }))
                }
                type="password"
              />
              <Text type="supporting">
                {serverLlm?.keys_configured?.[item.id] ? "stored" : "not set"}
              </Text>
              {serverLlm?.keys_configured?.[item.id] ? (
                <Button
                  label="Clear"
                  onClick={() => void clearServerKey(item.id)}
                  isDisabled={savingKeys}
                />
              ) : null}
            </HStack>
          ))}
          <HStack gap={2}>
            <Button
              label={savingKeys ? "Saving…" : "Save server keys"}
              variant="primary"
              onClick={() => void saveServerKeys()}
              isDisabled={savingKeys}
            />
          </HStack>
        </VStack>
      </Card>
    </VStack>
  );
}
