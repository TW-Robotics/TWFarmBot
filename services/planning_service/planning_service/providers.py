"""LLM provider abstraction.

The planning service supports multiple OpenAI-compatible backends. A provider
knows how to build a LangChain chat model and, optionally, how to list
available models for a UI picker.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests
from langchain_core.language_models import BaseChatModel

from .client import build_chat_model
from .config import PlannerConfig


class LLMProvider(ABC):
    """Abstract base for an LLM backend."""

    name: str

    @abstractmethod
    def build_chat_model(self, model: str, config: PlannerConfig) -> BaseChatModel:
        """Return a configured LangChain chat model for ``model``."""
        ...

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

    def build_chat_model(self, model: str, config: PlannerConfig) -> BaseChatModel:
        return build_chat_model(
            base_url=config.base_url,
            model=model,
            api_key=config.api_key,
            timeout_s=config.timeout_s,
            temperature=config.temperature,
            extra_body=config.extra_body,
        )

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

    def build_chat_model(self, model: str, config: PlannerConfig) -> BaseChatModel:
        return build_chat_model(
            base_url=config.base_url,
            model=model,
            api_key=config.api_key,
            timeout_s=config.timeout_s,
            temperature=config.temperature,
            extra_body=config.extra_body,
        )

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


_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    OpenRouterProvider.name: OpenRouterProvider,
    OpenAICompatibleProvider.name: OpenAICompatibleProvider,
}


DEFAULT_PROVIDER = OpenRouterProvider.name


def get_provider(name: str | None = None) -> LLMProvider:
    """Return a provider instance by name."""
    key = (name or os.getenv("PLANNING_LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if key not in _PROVIDER_REGISTRY:
        raise ValueError(f"unknown LLM provider: {key!r}")
    return _PROVIDER_REGISTRY[key]()


def list_provider_names() -> list[str]:
    """Return the ids of all registered providers."""
    return sorted(_PROVIDER_REGISTRY.keys())
