export type Json = Record<string, unknown> | unknown[] | string | number | boolean | null;

import type { LlmOverrides } from "./settings";

const API_STORAGE = "twfarmbot:apiUrl:v3";

function sameOriginApi(): string {
  return `${window.location.origin}/api`;
}

export function apiUrl(): string {
  const stored = localStorage.getItem(API_STORAGE);
  if (stored) {
    const url = stored.replace(/\/$/, "");
    if (window.isSecureContext && url.startsWith("http://")) {
      return sameOriginApi();
    }
    return url;
  }
  return (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "")
    || sameOriginApi();
}

export function setApiUrl(url: string): void {
  localStorage.setItem(API_STORAGE, url.replace(/\/$/, ""));
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function parseBody(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return await response.text();
  }
}

function errorMessage(body: any, status: number): string {
  if (body && typeof body === "object") {
    if ("detail" in body) return String(body.detail);
    if ("error" in body) return String(body.error);
  }
  if (typeof body === "string" && body) return body;
  return `Request failed (${status})`;
}

export async function api<T = any>(
  path: string,
  options: RequestInit & { timeoutMs?: number } = {},
): Promise<T> {
  const { timeoutMs, ...init } = options;
  const controller = new AbortController();
  const timer = timeoutMs
    ? window.setTimeout(() => controller.abort(), timeoutMs)
    : undefined;
  try {
    const response = await fetch(`${apiUrl()}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(init.headers || {}),
      },
    });
    const body = await parseBody(response);
    if (!response.ok) throw new ApiError(errorMessage(body, response.status), response.status);
    return body as T;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

export async function localApi<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const body = await parseBody(response);
  if (!response.ok) throw new ApiError(errorMessage(body, response.status), response.status);
  return body as T;
}

export function postAction(kind: string, params: Record<string, unknown> = {}, wait = false) {
  const query = wait ? "?wait=true" : "?wait=false";
  return api(`/actions${query}`, {
    method: "POST",
    body: JSON.stringify({ kind, params }),
  });
}

export async function streamChat(
  payload: {
    messages: unknown[];
    model: string | null;
    allow_actions: boolean;
    llm?: LlmOverrides | null;
    thread_id?: string | null;
    approved_ids?: string[] | null;
  },
  onEvent: (event: any) => void,
  options?: { signal?: AbortSignal },
): Promise<void> {
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  try {
    const response = await fetch(`${apiUrl()}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal: options?.signal,
    });
    if (!response.ok) {
      const body = await parseBody(response);
      throw new ApiError(errorMessage(body, response.status), response.status);
    }
    if (!response.body) throw new Error("The API returned no stream");
    reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const handle = (line: string) => {
    if (!line.startsWith("data: ")) return;
    const raw = line.slice(6).trim();
    if (!raw || raw === "[DONE]") return;
    onEvent(JSON.parse(raw));
  };
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      lines.forEach(handle);
    }
    if (buffer) handle(buffer);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      await reader?.cancel().catch(() => undefined);
      return;
    }
    throw error;
  }
}

export function parseNumber(value: string): number | null {
  const text = value.trim();
  if (!/^\s*-?\d+(?:[.,]\d+)?\s*$/.test(text)) return null;
  const n = Number(text.replace(",", "."));
  return Number.isFinite(n) ? n : null;
}

export function fmt(value: unknown): string {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(1) : "—";
}
