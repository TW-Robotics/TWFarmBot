"""Offline tests for OpenAI Realtime session minting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from planning_service import config as planning_config
from planning_service.realtime import CLIENT_SECRETS_URL, create_client_secret
from twfarmbot_api_server.app import create_app


@pytest.fixture
def keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "llm_keys.json"
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)
    return path


def test_create_client_secret_uses_openai_key(
    keys_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planning_config.write_stored_keys({"openai": "sk-live-secret"})
    seen: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        content = b'{"value":"ek-ephemeral","expires_at":1}'
        text = '{"value":"ek-ephemeral","expires_at":1}'

        def json(self) -> dict[str, Any]:
            return {"value": "ek-ephemeral", "expires_at": 1}

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        return _Resp()

    monkeypatch.setattr("planning_service.realtime.requests.post", fake_post)
    out = create_client_secret()
    assert out["value"] == "ek-ephemeral"
    assert seen["url"] == CLIENT_SECRETS_URL
    assert seen["headers"]["Authorization"] == "Bearer sk-live-secret"


def test_voice_session_endpoint_never_leaks_long_lived_key(
    keys_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planning_config.write_stored_keys({"openai": "sk-live-secret"})

    class _Resp:
        status_code = 200
        content = b'{"value":"ek-ephemeral"}'
        text = '{"value":"ek-ephemeral"}'

        def json(self) -> dict[str, str]:
            return {"value": "ek-ephemeral"}

    monkeypatch.setattr("planning_service.realtime.requests.post", lambda *_a, **_k: _Resp())
    client = TestClient(create_app())
    put_ready = client.get("/settings/llm")
    assert put_ready.json()["voice"]["realtime"] is True
    response = client.post("/voice/session")
    assert response.status_code == 200
    body = response.json()
    assert body["value"] == "ek-ephemeral"
    assert "sk-live-secret" not in response.text


def test_voice_session_requires_openai_key(keys_file: Path) -> None:
    client = TestClient(create_app())
    response = client.post("/voice/session")
    assert response.status_code == 400
