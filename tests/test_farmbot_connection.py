"""Smoke tests for the local FarmBot HTTP client."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from farmbot_client import FarmBotConnectionError, LocalFarmbot, connect


def test_connect_fails_fast_on_bad_server() -> None:
    with pytest.raises(FarmBotConnectionError):
        connect("http://127.0.0.1:1", timeout_s=2.0)


def test_move_sends_absolute_target() -> None:
    calls: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/move":
            body = json.loads(request.content.decode())
            calls.append((request.method, path, body))
            return httpx.Response(200, json={"success": True, "state": body})
        return httpx.Response(404)

    bot = LocalFarmbot(
        "http://farmbot.local",
        client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://farmbot.local",
        ),
    )
    bot.move(10, 20, 0)
    assert calls == [("POST", "/move", {"x": 10.0, "y": 20.0, "z": 0.0})]


@pytest.mark.skipif(
    not os.getenv("FARMBOT_LIVE_TEST"),
    reason="set FARMBOT_LIVE_TEST=1 to run against a real local FarmBot",
)
def test_farmbot_connection_live() -> None:
    fb = connect()
    xyz = fb.get_xyz()
    assert xyz is not None
    assert "x" in xyz
