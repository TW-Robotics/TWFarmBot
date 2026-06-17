"""Tests for the UI's API client.

The Streamlit app itself is hard to unit-test (renders DOM), but the
``ApiClient`` it uses to talk to the API is testable in isolation.
"""

from __future__ import annotations

from typing import Any


def test_api_client_returns_ok_on_2xx(monkeypatch: "pytest.MonkeyPatch") -> None:
    import httpx

    def fake_request(method, url, **kwargs):
        return httpx.Response(
            200,
            json={"status": "ok", "actions": ["water"], "farmbot": "connected"},
        )

    monkeypatch.setattr(httpx, "request", fake_request)
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api")
    r = c.request("GET", "/health")
    assert r.ok is True
    assert r.code == 200
    assert r.body["status"] == "ok"
    assert "water" in r.body["actions"]


def test_api_client_returns_not_ok_on_connection_failure(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import httpx

    def fake_request(method, url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", fake_request)
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api")
    r = c.request("GET", "/health")
    assert r.ok is False
    assert r.code == 0
    assert "ConnectError" in str(r.body.get("error", ""))


def test_api_client_passes_query_params(monkeypatch: "pytest.MonkeyPatch") -> None:
    import httpx

    seen: list[dict[str, Any]] = []

    def fake_request(method, url, **kwargs):
        seen.append({"method": method, "url": str(url), "kwargs": kwargs})
        return httpx.Response(200, json={"pin": 13, "mode": "analog", "value": 1})

    monkeypatch.setattr(httpx, "request", fake_request)
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api")
    r = c.request("GET", "/pin/13", params={"mode": "analog"})
    assert r.ok is True
    assert seen[0]["url"].endswith("/pin/13")
    assert seen[0]["kwargs"]["params"] == {"mode": "analog"}


def test_api_client_strips_trailing_slash() -> None:
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api/")
    assert c.base_url == "http://api"
