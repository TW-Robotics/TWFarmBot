"""Tests for the LLM provider abstraction and API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from planning_service.config import PlannerConfig
from twfarmbot_api_server.app import create_app
from planning_service.providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
    get_provider,
    list_provider_names,
)


def test_list_provider_names() -> None:
    names = list_provider_names()
    assert names == ["local", "openai", "openrouter"]


def test_get_provider_returns_instance() -> None:
    assert get_provider("openai").name == "openai"
    assert isinstance(get_provider("openrouter"), OpenRouterProvider)
    assert isinstance(get_provider("local"), OpenAICompatibleProvider)


def test_providers_build_chat_completions_models() -> None:
    openai_cfg = PlannerConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.6",
        api_key="dummy-key",
        timeout_s=30.0,
        temperature=0.0,
    )
    openai = OpenAIProvider()
    openai_model = openai.build_chat_model("gpt-5.6", openai_cfg)
    assert openai_model is not None
    assert openai_model.model_name == "gpt-5.6"  # type: ignore[attr-defined]
    assert hasattr(openai_model, "bind_tools")
    assert not hasattr(openai_model, "configure_tools")

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


def test_providers_endpoints() -> None:
    client = TestClient(create_app())
    r = client.get("/providers")
    assert r.status_code == 200
    body = r.json()
    assert body["providers"] == ["local", "openai", "openrouter"]

    r = client.get("/models?provider=local")
    assert r.status_code == 200
    assert r.json()["provider"] == "local"
