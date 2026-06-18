"""Tests for the read-only GET routes (position, status, pin, messages)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import twfarmbot_api_server.read as read_module
from twfarmbot_api_server.app import app


class _StubBot:
    def read_status(self, path=None):
        return {"connected": True, "path": path}


class _StubBackend:
    def get_xyz(self):
        return {"x": 100.0, "y": 200.0, "z": 50.0}

    def read_pin(self, pin, mode="digital"):
        return 42 if pin == 13 else 0

    def get_last_messages(self):
        return ["msg1", "msg2"]

    def get_images(self, limit=10, refresh=False):
        return [{
            "id": 1,
            "created_at": "2026-06-18T10:36:51.909Z",
            "attachment_url": "https://example.test/photo.jpg",
            "meta": {"x": 2.2, "y": 99.4, "z": 0.0},
        }][:limit]

    def _bot(self):
        return _StubBot()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(read_module.farmbot, "backend", _StubBackend())
    return TestClient(app)


def test_get_position(client: TestClient) -> None:
    r = client.get("/position")
    assert r.status_code == 200
    assert r.json() == {"xyz": {"x": 100.0, "y": 200.0, "z": 50.0}}


def test_get_status_default_path(client: TestClient) -> None:
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["path"] is None
    assert body["state"]["connected"] is True


def test_get_status_with_path(client: TestClient) -> None:
    r = client.get("/status", params={"path": "position"})
    assert r.status_code == 200
    assert r.json()["path"] == "position"


def test_get_pin(client: TestClient) -> None:
    r = client.get("/pin/13")
    assert r.status_code == 200
    assert r.json() == {"pin": 13, "mode": "digital", "value": 42}


def test_get_pin_with_mode(client: TestClient) -> None:
    r = client.get("/pin/13", params={"mode": "analog"})
    assert r.status_code == 200
    assert r.json()["mode"] == "analog"


def test_get_messages(client: TestClient) -> None:
    r = client.get("/messages")
    assert r.status_code == 200
    assert r.json() == {"last_messages": ["msg1", "msg2"]}


def test_get_images(client: TestClient) -> None:
    r = client.get("/images", params={"limit": 1, "refresh": "true"})
    assert r.status_code == 200
    assert r.json()["images"][0]["attachment_url"] == "https://example.test/photo.jpg"


def test_read_error_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenBackend:
        def get_xyz(self):
            raise RuntimeError("broker down")

    monkeypatch.setattr(read_module.farmbot, "backend", _BrokenBackend())
    r = client.get("/position")
    assert r.status_code == 502
    assert "broker down" in r.json()["detail"]


def test_get_pins(client: TestClient) -> None:
    r = client.get("/pins")
    assert r.status_code == 200
    body = r.json()
    assert "pins" in body
    # The dev.yaml in the repo has at least one named pin
    assert isinstance(body["pins"], list)
    if body["pins"]:
        first = body["pins"][0]
        assert "label" in first
        assert "pin" in first
        assert "mode" in first


def test_get_positions(client: TestClient) -> None:
    r = client.get("/positions")
    assert r.status_code == 200
    body = r.json()
    assert "positions" in body
    assert isinstance(body["positions"], list)
    if body["positions"]:
        first = body["positions"][0]
        assert "label" in first
        for axis in ("x", "y", "z"):
            assert axis in first
