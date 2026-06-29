"""Tests for the POST /actions endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from farmbot_serial import FarmduinoConnectionError
from safety_service import UnsafeActionError
from twfarmbot_api_server.app import create_app
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action


@pytest.fixture
def client() -> TestClient:
    registry = ActionRegistry()
    registry.register("noop", lambda a: a)
    app = create_app(registry=registry)
    return TestClient(app)


def test_post_action_returns_200_for_valid_action(client: TestClient) -> None:
    r = client.post("/actions", json={"kind": "noop", "params": {"foo": "bar"}})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["action"]["kind"] == "noop"
    assert body["action"]["params"] == {"foo": "bar"}


def test_post_action_returns_404_for_unknown_kind(client: TestClient) -> None:
    r = client.post("/actions", json={"kind": "unknown", "params": {}})
    assert r.status_code == 404
    assert "no handler registered" in r.json()["detail"].lower()


def test_post_action_returns_400_for_unsafe_action(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_validate(action: Action) -> None:
        raise UnsafeActionError("too dangerous")

    monkeypatch.setattr("twfarmbot_core.actions.safety_validate", fake_validate)
    r = client.post("/actions", json={"kind": "noop", "params": {}})
    assert r.status_code == 400
    assert "too dangerous" in r.json()["detail"]


def test_post_action_returns_502_on_farmduino_connection_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(action: Action) -> Action:
        raise FarmduinoConnectionError("port missing")

    monkeypatch.setattr(client.app.state.registry, "dispatch", explode)  # type: ignore[attr-defined]
    r = client.post("/actions", json={"kind": "noop", "params": {}})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "Farmduino not connected" in detail
    assert "port missing" in detail


def test_post_action_returns_500_with_real_error_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(action: Action) -> Action:
        raise RuntimeError("handler exploded")

    monkeypatch.setattr(client.app.state.registry, "dispatch", explode)  # type: ignore[attr-defined]
    r = client.post("/actions", json={"kind": "noop", "params": {}})
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "RuntimeError" in detail
    assert "handler exploded" in detail


def test_post_action_non_wait_returns_404_for_unknown_kind(client: TestClient) -> None:
    r = client.post("/actions?wait=false", json={"kind": "unknown", "params": {}})
    assert r.status_code == 404
    assert "unknown action kind" in r.json()["detail"].lower()
