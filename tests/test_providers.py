"""Tests for the LLM provider abstraction and API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from planning_service.config import PlannerConfig
from twfarmbot_api_server.app import create_app
from planning_service.providers import (
    OpenAICompatibleProvider,
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


def test_get_provider_returns_instance() -> None:
    assert isinstance(get_provider("openrouter"), OpenRouterProvider)
    assert isinstance(get_provider("local"), OpenAICompatibleProvider)


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
    assert model.model_name == "openai/gpt-4o"

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
    assert local_model.model_name == "qwen2.5"


def test_providers_endpoints(client) -> None:  # noqa: ANN001
    r = client.get("/providers")
    assert r.status_code == 200
    body = r.json()
    assert "openrouter" in body["providers"]

    r = client.get("/models?provider=local")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local"
    assert isinstance(body["models"], list)
