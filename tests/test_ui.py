"""Tests for the UI's HTTP helper.

The Streamlit app itself is hard to unit-test (renders DOM), but the
``_request`` helper that talks to the API is testable in isolation.
"""

from __future__ import annotations

from typing import Any


def test_request_helper_hits_health(monkeypatch: "pytest.MonkeyPatch") -> None:
    import httpx

    def fake_request(method, url, **kwargs):
        return httpx.Response(
            200,
            json={"status": "ok", "actions": ["water"], "farmbot": "connected"},
        )

    monkeypatch.setattr(httpx, "request", fake_request)
    from twfarmbot_ui import _request

    code, body = _request("GET", "/health")
    assert code == 200
    assert body["status"] == "ok"
    assert "water" in body["actions"]


def test_request_helper_returns_error_on_connection_failure(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import httpx

    def fake_request(method, url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", fake_request)
    from twfarmbot_ui import _request

    code, body = _request("GET", "/health")
    assert code == 0
    assert "ConnectError" in str(body.get("error", ""))


def test_request_helper_passes_query_params(monkeypatch: "pytest.MonkeyPatch") -> None:
    import httpx

    seen: list[dict[str, Any]] = []

    def fake_request(method, url, **kwargs):
        seen.append({"method": method, "url": str(url), "kwargs": kwargs})
        return httpx.Response(200, json={"pin": 13, "mode": "analog", "value": 1})

    monkeypatch.setattr(httpx, "request", fake_request)
    from twfarmbot_ui import _request

    code, body = _request("GET", "/pin/13", params={"mode": "analog"})
    assert code == 200
    assert seen[0]["url"].endswith("/pin/13")
    assert seen[0]["kwargs"]["params"] == {"mode": "analog"}
