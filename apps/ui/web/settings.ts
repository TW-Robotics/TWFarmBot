export type LlmProfile = {
  id: string;
  name: string;
  provider: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  timeoutS: number;
  temperature: number;
};

export type LlmOverrides = {
  provider?: string;
  base_url?: string;
  api_key?: string;
  model?: string;
  timeout_s?: number;
  temperature?: number;
};

export type AppSettings = {
  activeLlmProfileId: string | null;
  llmProfiles: LlmProfile[];
};

export const SETTINGS_STORAGE_KEY = "twfarmbot:settings:v1";
export const SETTINGS_CHANGED_EVENT = "twfarmbot:settings-changed";

export const LLM_PROVIDERS = [
  { id: "openai", label: "OpenAI" },
  { id: "openrouter", label: "OpenRouter" },
  { id: "local", label: "Local (OpenAI-compatible)" },
] as const;

export const PROVIDER_DEFAULTS: Record<string, { baseUrl: string; model: string }> = {
  openai: { baseUrl: "https://api.openai.com/v1", model: "gpt-5.6" },
  openrouter: { baseUrl: "https://openrouter.ai/api/v1", model: "anthropic/claude-sonnet-4" },
  local: { baseUrl: "http://127.0.0.1:11434/v1", model: "llama3.2" },
};

const DEFAULT_SETTINGS: AppSettings = {
  activeLlmProfileId: null,
  llmProfiles: [],
};

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS, llmProfiles: [] };
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      activeLlmProfileId: parsed.activeLlmProfileId ?? null,
      llmProfiles: Array.isArray(parsed.llmProfiles) ? parsed.llmProfiles : [],
    };
  } catch {
    return { ...DEFAULT_SETTINGS, llmProfiles: [] };
  }
}

export function saveSettings(settings: AppSettings): void {
  localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent(SETTINGS_CHANGED_EVENT));
}

export function getActiveLlmProfile(settings: AppSettings = loadSettings()): LlmProfile | null {
  if (!settings.activeLlmProfileId) return null;
  return settings.llmProfiles.find((profile) => profile.id === settings.activeLlmProfileId) ?? null;
}

export function llmOverridesFromProfile(profile: LlmProfile | null): LlmOverrides | null {
  if (!profile) return null;
  const overrides: LlmOverrides = {
    provider: profile.provider,
    base_url: profile.baseUrl,
    model: profile.model,
    timeout_s: profile.timeoutS,
    temperature: profile.temperature,
  };
  if (profile.apiKey.trim()) overrides.api_key = profile.apiKey.trim();
  return overrides;
}

export function createEmptyProfile(name = "New profile"): LlmProfile {
  const defaults = PROVIDER_DEFAULTS.openai;
  return {
    id: crypto.randomUUID(),
    name,
    provider: "openai",
    baseUrl: defaults.baseUrl,
    apiKey: "",
    model: defaults.model,
    timeoutS: 120,
    temperature: 0,
  };
}

export function profileLabel(profile: LlmProfile): string {
  return profile.name.trim() || profile.provider;
}
