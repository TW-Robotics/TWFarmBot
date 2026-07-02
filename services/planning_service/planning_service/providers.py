"""LLM provider abstraction.

The planning service supports multiple OpenAI-compatible backends. A provider
knows how to build a LangChain chat model and, optionally, how to list
available models for a UI picker.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import replace

import requests
from langchain_core.language_models import BaseChatModel

from .client import build_chat_model
from .config import PlannerConfig


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://100.102.103.44:11434/v1"
OLLAMA_MODEL = "gemma4:e4b"


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

    def configure(self, config: PlannerConfig) -> PlannerConfig:
        """Return config adjusted for this provider."""
        return replace(config, provider=self.name)


class OpenRouterProvider(LLMProvider):
    """OpenRouter (https://openrouter.ai) provider."""

    name = "openrouter"

    def configure(self, config: PlannerConfig) -> PlannerConfig:
        if config.provider == self.name:
            return config
        return replace(config, provider=self.name, base_url=OPENROUTER_BASE_URL)

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
            return []


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


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama via its OpenAI-compatible `/v1` API."""

    name = "ollama"

    def configure(self, config: PlannerConfig) -> PlannerConfig:
        if config.provider == self.name:
            return config
        return replace(
            config,
            provider=self.name,
            base_url=os.getenv("PLANNING_OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            model=os.getenv("PLANNING_OLLAMA_MODEL", OLLAMA_MODEL),
            api_key=None,
        )

    def list_models(self, config: PlannerConfig) -> list[str]:
        ids = super().list_models(config)
        if ids:
            return ids
        try:
            base_url = config.base_url.removesuffix("/v1")
            r = requests.get(f"{base_url}/api/tags", timeout=10)
            r.raise_for_status()
            data = r.json()
            return sorted(
                {
                    model.get("name") or model.get("model")
                    for model in data.get("models", [])
                    if model.get("name") or model.get("model")
                },
                key=lambda s: s.lower(),
            )
        except Exception:  # noqa: BLE001
            return []


_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    OpenRouterProvider.name: OpenRouterProvider,
    OpenAICompatibleProvider.name: OpenAICompatibleProvider,
    OllamaProvider.name: OllamaProvider,
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
