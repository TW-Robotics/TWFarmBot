"""Tests for the LLM provider abstraction and API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

import pytest

from planning_service.config import PlannerConfig
from twfarmbot_api_server.app import create_app
from planning_service.providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
    VertexProvider,
    get_provider,
    list_provider_names,
)


def test_list_provider_names() -> None:
    names = list_provider_names()
    assert names == ["local", "openai", "openrouter", "vertex"]


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
    assert openai_model.reasoning_effort == "none"  # type: ignore[attr-defined]
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
    assert body["providers"] == ["local", "openai", "openrouter", "vertex"]

    r = client.get("/models?provider=local")
    assert r.status_code == 200
    assert r.json()["provider"] == "local"


def test_vertex_needs_project(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    monkeypatch.delenv("PLANNING_LLM_BASE_URL", raising=False)
    cfg = PlannerConfig(
        provider="vertex",
        base_url="https://api.openai.com/v1",
        model="gpt-5.6",
        api_key=None,
        timeout_s=30.0,
        temperature=0.0,
    )
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        VertexProvider().build_chat_model("gpt-5.6", cfg)


def test_vertex_builds_native_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west3")
    monkeypatch.delenv("PLANNING_LLM_BASE_URL", raising=False)
    cfg = PlannerConfig(
        provider="vertex",
        base_url="https://api.openai.com/v1",
        model="gpt-5.6",
        api_key=None,
        timeout_s=30.0,
        temperature=0.0,
    )
    model = VertexProvider().build_chat_model("gpt-5.6", cfg)
    assert model.model_name == "gemini-3.8-flash"  # type: ignore[attr-defined]
    assert getattr(model, "project", None) == "demo-project"
    assert getattr(model, "location", None) == "europe-west3"


def test_vertex_strips_google_prefix(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    cfg = PlannerConfig(
        provider="vertex",
        base_url="",
        model="google/gemini-2.5-flash",
        api_key=None,
        timeout_s=30.0,
        temperature=0.0,
    )
    model = VertexProvider().build_chat_model("google/gemini-2.5-flash", cfg)
    assert model.model_name == "gemini-2.5-flash"  # type: ignore[attr-defined]
