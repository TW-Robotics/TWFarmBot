"""LLM provider abstraction.

Every provider speaks OpenAI-compatible ``/chat/completions`` JSON
function/tool calling. Physical ACT tools still go through the approval
gate; read tools are ordinary JSON tools.
"""

from __future__ import annotations

import os
from abc import ABC
from typing import Any

import requests
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider as _PAIOpenAI

from .config import DEFAULT_MODEL, PlannerConfig, resolve_vertex_settings
from .reasoning import effort_from_budget, model_uses_reasoning, openai_effort


class LLMProvider(ABC):
    """Abstract base for an LLM backend."""

    name: str

    def build_model(self, model: str, config: PlannerConfig) -> Model:
        """Return a Pydantic AI model for ``model`` on an OpenAI-compatible API."""
        effort = effort_from_budget(config.reasoning_budget)
        settings: dict[str, Any] = {
            "temperature": config.temperature,
            "timeout": config.timeout_s,
        }
        if config.extra_body:
            settings["extra_body"] = dict(config.extra_body)
        base_url = config.base_url.rstrip("/")
        if "openrouter.ai" in base_url:
            from pydantic_ai.models.openrouter import OpenRouterModel
            from pydantic_ai.providers.openrouter import OpenRouterProvider as _PAIRouter

            if model_uses_reasoning(model):
                settings["thinking"] = False if effort == "none" else openai_effort(model, effort)
            return OpenRouterModel(
                model, provider=_PAIRouter(api_key=config.api_key), settings=settings
            )
        # No key is needed by local servers (Ollama/vLLM), but the client wants one.
        provider = _PAIOpenAI(base_url=config.base_url, api_key=config.api_key or "not-required")
        if (
            base_url == "https://api.openai.com/v1"
            and "gpt-5" in model.lower()
            and "chat" not in model.lower()
        ):
            # GPT-5 needs Responses to combine reasoning with tools.
            settings["thinking"] = False if effort == "none" else openai_effort(model, effort)
            return OpenAIResponsesModel(model, provider=provider, settings=settings)
        return OpenAIChatModel(model, provider=provider, settings=settings)

    def list_models(self, _config: PlannerConfig) -> list[str]:
        """Return a list of model ids available from this provider."""
        return []


# Fallback when OpenRouter /models cannot be reached: reasoning + tools.
_OPENROUTER_REASONING_TOOL_MODELS = [
    "openai/gpt-5.6",
    "google/gemini-2.5-flash",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4-flash",
    "anthropic/claude-sonnet-4",
]


class OpenRouterProvider(LLMProvider):
    """OpenRouter (https://openrouter.ai) provider."""

    name = "openrouter"

    def list_models(self, config: PlannerConfig) -> list[str]:
        try:
            r = requests.get(
                f"{config.base_url}/models",
                params={"supported_parameters": "tools,reasoning"},
                headers={"Authorization": f"Bearer {config.api_key}"}
                if config.api_key
                else {},
                timeout=15,
            )
            r.raise_for_status()
            ids = _openrouter_reasoning_tool_ids(r.json().get("data", []))
            if ids:
                return ids
        except Exception:  # noqa: BLE001
            pass
        return list(_OPENROUTER_REASONING_TOOL_MODELS)


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

    Uses Pydantic AI's ``GoogleModel`` (proper Gemini tool
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

    def build_model(self, model: str, config: PlannerConfig) -> Model:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google_cloud import GoogleCloudProvider

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
        effort = effort_from_budget(config.reasoning_budget)
        return GoogleModel(
            native_model,
            provider=GoogleCloudProvider(
                project=str(settings["project"]), location=str(settings["location"])
            ),
            settings={
                "temperature": config.temperature,
                "thinking": False if effort == "none" else effort,
            },
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


def _openrouter_reasoning_tool_ids(models: list) -> list[str]:
    ids: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not model_id or ":batch" in str(model_id):
            continue
        params = {str(p) for p in (item.get("supported_parameters") or [])}
        if "tools" in params and "reasoning" in params:
            ids.add(str(model_id))
    return sorted(ids, key=str.lower)


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
