"""Tests for api_server startup-connect and /health farmbot field."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from farmbot_client import FarmBotConnectionError
from twfarmbot_api_server import app as app_module
from twfarmbot_api_server.app import (
    create_app,
    connect_to_farmbot,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_reports_unknown_farmbot_before_boot(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["farmbot"] == "unknown"
    assert "water" in body["actions"]


def test_connect_to_farmbot_marks_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """When get_farmbot succeeds, status flips to 'connected'."""
    monkeypatch.setattr(
        "farmbot_gateway.get_farmbot", lambda: object()
    )
    test_app = create_app()
    app_module.app = test_app  # so connect_to_farmbot updates the right instance
    try:
        status = connect_to_farmbot(required=False)
        assert status == "connected"
        assert test_app.state.farmbot_status == "connected"
    finally:
        app_module.app = create_app()  # restore for other tests


def test_connect_to_farmbot_marks_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed connect (non-required) records the failure on state, does not raise."""
    def boom() -> None:
        raise FarmBotConnectionError("nope") from RuntimeError("auth bad")

    monkeypatch.setattr("farmbot_gateway.get_farmbot", boom)
    test_app = create_app()
    app_module.app = test_app
    try:
        status = connect_to_farmbot(required=False)
        assert status.startswith("failed:")
        assert "nope" in test_app.state.farmbot_status
    finally:
        app_module.app = create_app()


def test_connect_to_farmbot_required_exits_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> None:
        raise FarmBotConnectionError("nope") from RuntimeError("auth bad")

    monkeypatch.setattr("farmbot_gateway.get_farmbot", boom)
    test_app = create_app()
    app_module.app = test_app
    try:
        with pytest.raises(SystemExit) as excinfo:
            connect_to_farmbot(required=True)
        assert "FATAL" in str(excinfo.value)
    finally:
        app_module.app = create_app()


def test_connect_to_farmbot_skipped_when_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FARMBOT_REQUIRED", "0")
    test_app = create_app()
    app_module.app = test_app
    try:
        status = connect_to_farmbot(required=False)
        assert status == "skipped"
        assert test_app.state.farmbot_status == "skipped"
    finally:
        app_module.app = create_app()
