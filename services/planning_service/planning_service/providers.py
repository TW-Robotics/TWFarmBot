"""LLM provider abstraction.

Every provider speaks OpenAI-compatible ``/chat/completions`` JSON
function/tool calling. Physical ACT tools still go through the approval
gate; read tools are ordinary JSON tools.
"""

from __future__ import annotations

import os
from abc import ABC

import requests
from langchain_core.language_models import BaseChatModel

from .client import build_chat_model
from .config import DEFAULT_MODEL, PlannerConfig, resolve_vertex_settings


class LLMProvider(ABC):
    """Abstract base for an LLM backend."""

    name: str

    def build_chat_model(self, model: str, config: PlannerConfig) -> BaseChatModel:
        """Return a configured LangChain chat model for ``model``."""
        return build_chat_model(
            base_url=config.base_url,
            model=model,
            api_key=config.api_key,
            timeout_s=config.timeout_s,
            temperature=config.temperature,
            extra_body=config.extra_body,
        )

    def list_models(self, _config: PlannerConfig) -> list[str]:
        """Return a list of model ids available from this provider."""
        return []


# Curated OpenRouter models known to support tool/function calling. Used as a
# fallback when the live /models endpoint cannot be reached.
_OPENROUTER_TOOL_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3-opus",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3",
    "mistralai/mistral-nemo",
    "mistralai/mistral-large",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "nousresearch/hermes-3-llama-3.1-405b",
]


class OpenRouterProvider(LLMProvider):
    """OpenRouter (https://openrouter.ai) provider."""

    name = "openrouter"

    def list_models(self, config: PlannerConfig) -> list[str]:
        try:
            r = requests.get(
                f"{config.base_url}/models",
                headers={"Authorization": f"Bearer {config.api_key}"}
                if config.api_key
                else {},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            models = data.get("data", [])
            ids = sorted(
                {m.get("id") for m in models if m.get("id")},
                key=lambda s: s.lower(),
            )
            if ids:
                return ids
        except Exception:  # noqa: BLE001
            pass
        return list(_OPENROUTER_TOOL_MODELS)


class OpenAICompatibleProvider(LLMProvider):
    """Generic OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, TGI, …)."""

    name = "local"

    def list_models(self, config: PlannerConfig) -> list[str]:
        try:
            r = requests.get(
                f"{config.base_url}/models",
                headers={"Authorization": f"Bearer {config.api_key}"}
                if config.api_key
                else {},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            models = data.get("data", [])
            return sorted(
                {m.get("id") for m in models if m.get("id")},
                key=lambda s: s.lower(),
            )
        except Exception:  # noqa: BLE001
            return []


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions provider with JSON function/tool calling."""

    name = "openai"

    def list_models(self, _config: PlannerConfig) -> list[str]:
        return ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]


class VertexProvider(LLMProvider):
    """Google Vertex AI via the native Gemini endpoint.

    Uses ``langchain_google_vertexai.ChatVertexAI`` (proper Gemini tool
    calling) with Application Default Credentials — no API keys, no token
    plumbing. Auth is your gcloud login or a service-account key at
    ``GOOGLE_APPLICATION_CREDENTIALS``.

    Required: ``GOOGLE_CLOUD_PROJECT`` (or the Vertex AI card in Settings).
    Optional: ``GOOGLE_CLOUD_LOCATION`` (default ``global`` — the global
    endpoint serves every current Gemini generation; regional locations
    such as ``europe-west3`` serve only some of them),
    ``PLANNING_LLM_MODEL`` (default ``gemini-3.8-flash``; a ``google/``
    prefix is stripped).
    """

    name = "vertex"

    def build_chat_model(self, model: str, config: PlannerConfig) -> BaseChatModel:
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as err:
            raise RuntimeError(
                "Vertex provider needs the 'langchain-google-vertexai' package: "
                "uv sync"
            ) from err
        settings = resolve_vertex_settings()
        if not settings["project"]:
            raise ValueError(
                "Vertex provider needs a GCP project: set it in Settings "
                "(Vertex AI card) or via GOOGLE_CLOUD_PROJECT"
            )
        native_model = model
        if native_model == DEFAULT_MODEL:
            # Built-in default is an OpenAI id — meaningless on Vertex.
            native_model = _VERTEX_DEFAULT_MODEL
        elif "/" in native_model:
            native_model = native_model.rsplit("/", 1)[-1]
        return ChatVertexAI(
            model_name=native_model,
            project=str(settings["project"]),
            location=str(settings["location"]),
            temperature=config.temperature,
        )

    def list_models(self, config: PlannerConfig) -> list[str]:
        del config
        return list(_VERTEX_TOOL_MODELS)


# Gemini models verified serverless against the global Vertex endpoint
# (2026-09, project farmbot-507612). Regional endpoints serve a subset —
# set Location = global in the Vertex settings card for all of these.
_VERTEX_TOOL_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite",
]

_VERTEX_DEFAULT_MODEL = _VERTEX_TOOL_MODELS[0]


_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    OpenAIProvider.name: OpenAIProvider,
    OpenRouterProvider.name: OpenRouterProvider,
    OpenAICompatibleProvider.name: OpenAICompatibleProvider,
    VertexProvider.name: VertexProvider,
}


DEFAULT_PROVIDER = OpenAIProvider.name


def get_provider(name: str | None = None, *, permissive: bool = False) -> LLMProvider:
    """Return a provider instance by name."""
    del permissive
    key = (name or os.getenv("PLANNING_LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if key not in _PROVIDER_REGISTRY:
        raise ValueError(f"unknown LLM provider: {key!r}")
    return _PROVIDER_REGISTRY[key]()


def list_provider_names() -> list[str]:
    """Return the ids of all registered providers."""
    return sorted(_PROVIDER_REGISTRY)
