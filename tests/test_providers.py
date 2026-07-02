"""Tests for the LLM provider abstraction and API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from planning_service.config import PlannerConfig
from twfarmbot_api_server.app import create_app
from planning_service.providers import (
    OpenAICompatibleProvider,
    OllamaProvider,
    OpenRouterProvider,
    get_provider,
    list_provider_names,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_list_provider_names() -> None:
    names = list_provider_names()
    assert "openrouter" in names
    assert "local" in names
    assert "ollama" in names


def test_get_provider_returns_instance() -> None:
    assert isinstance(get_provider("openrouter"), OpenRouterProvider)
    assert isinstance(get_provider("local"), OpenAICompatibleProvider)
    assert isinstance(get_provider("ollama"), OllamaProvider)


def test_provider_override_uses_provider_defaults() -> None:
    cfg = PlannerConfig(
        provider="ollama",
        base_url="http://100.102.103.44:11434/v1",
        model="gemma4:e4b",
        api_key=None,
        timeout_s=120.0,
        temperature=0.0,
    )

    openrouter_cfg = OpenRouterProvider().configure(cfg)
    assert openrouter_cfg.provider == "openrouter"
    assert openrouter_cfg.base_url == "https://openrouter.ai/api/v1"

    ollama_cfg = OllamaProvider().configure(openrouter_cfg)
    assert ollama_cfg.provider == "ollama"
    assert ollama_cfg.base_url == "http://100.102.103.44:11434/v1"
    assert ollama_cfg.model == "gemma4:e4b"


def test_openrouter_does_not_return_fallback_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get(*_args, **_kwargs):  # noqa: ANN202, ANN002, ANN003
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("planning_service.providers.requests.get", fail_get)
    cfg = PlannerConfig(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-sonnet",
        api_key="dummy-key",
        timeout_s=30.0,
        temperature=0.0,
    )

    assert OpenRouterProvider().list_models(cfg) == []


def test_ollama_lists_native_tags_when_openai_models_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "models": [
                    {"name": "gemma4:e4b"},
                    {"model": "qwen2.5:7b"},
                ]
            }

    def fake_get(url: str, **_kwargs):  # noqa: ANN202, ANN003
        if url.endswith("/v1/models"):
            raise RuntimeError("openai models unavailable")
        assert url == "http://ollama:11434/api/tags"
        return Response()

    monkeypatch.setattr("planning_service.providers.requests.get", fake_get)
    cfg = PlannerConfig(
        provider="ollama",
        base_url="http://ollama:11434/v1",
        model="gemma4:e4b",
        api_key=None,
        timeout_s=30.0,
        temperature=0.0,
    )

    assert OllamaProvider().list_models(cfg) == ["gemma4:e4b", "qwen2.5:7b"]


def test_providers_build_model() -> None:
    cfg = PlannerConfig(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-sonnet",
        api_key="dummy-key",
        timeout_s=30.0,
        temperature=0.0,
    )
    openrouter = OpenRouterProvider()
    model = openrouter.build_chat_model("openai/gpt-4o", cfg)
    assert model is not None
    assert model.model_name == "openai/gpt-4o"  # type: ignore[attr-defined]

    local_cfg = PlannerConfig(
        provider="local",
        base_url="http://localhost:8000/v1",
        model="llama3",
        api_key="dummy-key",
        timeout_s=60.0,
        temperature=0.0,
    )
    local = OpenAICompatibleProvider()
    local_model = local.build_chat_model("qwen2.5", local_cfg)
    assert local_model is not None
    assert local_model.model_name == "qwen2.5"  # type: ignore[attr-defined]

    ollama_cfg = PlannerConfig(
        provider="ollama",
        base_url="http://100.102.103.44:11434/v1",
        model="gemma4:e4b",
        api_key=None,
        timeout_s=120.0,
        temperature=0.0,
    )
    ollama = OllamaProvider()
    ollama_model = ollama.build_chat_model("gemma4:e4b", ollama_cfg)
    assert ollama_model is not None
    assert ollama_model.model_name == "gemma4:e4b"  # type: ignore[attr-defined]


def test_providers_endpoints(client) -> None:  # noqa: ANN001
    r = client.get("/providers")
    assert r.status_code == 200
    body = r.json()
    assert "openrouter" in body["providers"]
    assert "ollama" in body["providers"]

    r = client.get("/models?provider=local")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local"
    assert isinstance(body["models"], list)
