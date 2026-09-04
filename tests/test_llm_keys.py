"""Tests for server-side LLM API key storage and resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from planning_service import config as planning_config
from twfarmbot_api_server.app import create_app


@pytest.fixture
def keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "llm_keys.json"
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(path))
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)
    return path


def test_store_round_trip_and_permissions(keys_file: Path) -> None:
    planning_config.write_stored_keys({"openai": "sk-test", "openrouter": "  "})
    assert planning_config.read_stored_keys() == {"openai": "sk-test"}
    assert keys_file.stat().st_mode & 0o777 == 0o600


def test_blank_value_deletes_entry(keys_file: Path) -> None:
    planning_config.write_stored_keys({"openai": "sk-test"})
    planning_config.write_stored_keys({"openai": ""})
    assert planning_config.read_stored_keys() == {}


def test_env_beats_stored_key(keys_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    planning_config.write_stored_keys({"openai": "sk-stored"})
    monkeypatch.setenv("PLANNING_LLM_API_KEY", "sk-env")
    cfg = planning_config.load_config(yaml_data={"planning": {"provider": "openai"}})
    assert cfg.api_key == "sk-env"


def test_stored_key_used_when_no_env(keys_file: Path) -> None:
    planning_config.write_stored_keys({"openrouter": "sk-stored"})
    cfg = planning_config.load_config(
        yaml_data={"planning": {"provider": "openrouter"}}
    )
    assert cfg.api_key == "sk-stored"


def test_blank_override_key_is_ignored(keys_file: Path) -> None:
    planning_config.write_stored_keys({"openai": "sk-stored"})
    cfg = planning_config.load_config(yaml_data={"planning": {"provider": "openai"}})
    updated = planning_config.apply_overrides(cfg, api_key="   ")
    assert updated.api_key == "sk-stored"


def test_settings_endpoints_never_leak_values(
    keys_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)
    client = TestClient(create_app())
    put = client.put("/settings/llm", json={"keys": {"openai": "sk-secret-123"}})
    assert put.status_code == 200
    assert put.json()["keys_configured"].get("openai") is True

    get = client.get("/settings/llm")
    assert get.status_code == 200
    body = get.json()
    assert body["keys_configured"].get("openai") is True
    assert "sk-secret-123" not in json.dumps(body)

    stored = json.loads(keys_file.read_text(encoding="utf-8"))
    assert stored["keys"]["openai"] == "sk-secret-123"


def test_settings_delete_clears_key(keys_file: Path) -> None:
    client = TestClient(create_app())
    client.put("/settings/llm", json={"keys": {"openai": "sk-x"}})
    delete = client.put("/settings/llm", json={"keys": {"openai": ""}})
    assert delete.json()["keys_configured"].get("openai") is not True
    assert planning_config.read_stored_keys() == {}


def test_override_provider_picks_up_stored_key(keys_file: Path) -> None:
    from twfarmbot_api_server.app import _resolve_llm_config
    from twfarmbot_api_server.app import LlmOverrides

    planning_config.write_stored_keys({"openrouter": "sk-stored"})
    cfg = _resolve_llm_config(
        LlmOverrides(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    )
    assert cfg.provider == "openrouter"
    assert cfg.api_key == "sk-stored"


def test_vertex_settings_round_trip(keys_file: Path) -> None:
    out = planning_config.write_vertex_settings(project="demo-proj", location="")
    assert out == {"project": "demo-proj", "location": "global"}
    assert planning_config.read_vertex_settings() == {
        "project": "demo-proj",
        "location": None,
    }
    # Keys co-exist in the same file.
    planning_config.write_stored_keys({"openai": "sk-x"})
    assert planning_config.read_vertex_settings()["project"] == "demo-proj"
    assert planning_config.read_stored_keys() == {"openai": "sk-x"}


def test_vertex_settings_env_wins(keys_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    planning_config.write_vertex_settings(project="stored-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
    assert planning_config.resolve_vertex_settings() == {
        "project": "env-proj",
        "location": "global",
    }


def test_settings_vertex_endpoints(keys_file: Path) -> None:
    client = TestClient(create_app())
    put = client.put(
        "/settings/llm", json={"keys": {}, "vertex": {"project": "demo-proj"}}
    )
    assert put.status_code == 200
    assert put.json()["vertex"] == {
        "project": "demo-proj",
        "location": "global",
    }
    get = client.get("/settings/llm")
    assert get.status_code == 200
    assert get.json()["vertex"]["project"] == "demo-proj"
