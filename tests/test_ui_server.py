"""Tests for the Material 3 UI server (static frontend + API proxy).

The proxy is exercised against an in-process stub upstream via httpx's
ASGITransport, so no sockets are opened.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient


def _upstream() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "actions": ["move"], "farmbot": "connected"}

    @app.get("/pin/{pin}")
    def pin(pin: int, mode: str = "digital") -> dict[str, Any]:
        return {"pin": pin, "mode": mode, "value": 1}

    @app.post("/actions")
    def actions(payload: dict[str, Any], wait: bool = True) -> dict[str, Any]:
        return {"status": "ok" if wait else "queued", "action": payload}

    @app.post("/chat/stream")
    def chat_stream() -> StreamingResponse:
        def events() -> Iterator[str]:
            yield f"data: {json.dumps({'type': 'delta', 'content': 'hi'})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'metrics': {}})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path / "sessions"))
    from twfarmbot_ui.server import create_app

    proxy_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=_upstream()))
    with TestClient(create_app(http_client=proxy_http)) as test_client:
        yield test_client


def test_serves_material_frontend(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "@material/web" in r.text

    r = client.get("/js/main.js")
    assert r.status_code == 200


def test_proxies_api_get(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["farmbot"] == "connected"


def test_proxies_query_params(client: TestClient) -> None:
    r = client.get("/api/pin/13", params={"mode": "analog"})
    assert r.json() == {"pin": 13, "mode": "analog", "value": 1}


def test_proxies_post_body_and_wait_param(client: TestClient) -> None:
    r = client.post(
        "/api/actions",
        params={"wait": "false"},
        json={"kind": "move", "params": {"x": 1}},
    )
    assert r.json()["status"] == "queued"
    assert r.json()["action"]["kind"] == "move"


def test_proxies_sse_stream(client: TestClient) -> None:
    with client.stream("POST", "/api/chat/stream", json={"messages": []}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes()).decode()
    assert '"type": "delta"' in body
    assert '"type": "meta"' in body


def test_proxy_returns_502_on_connection_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path / "sessions"))
    from twfarmbot_ui.server import create_app

    class FailingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

    proxy_http = httpx.AsyncClient(transport=FailingTransport())
    with TestClient(create_app(http_client=proxy_http)) as test_client:
        r = test_client.get("/api/health")
    assert r.status_code == 502
    assert "ConnectError" in r.json()["detail"]


def test_session_crud_roundtrip(client: TestClient) -> None:
    snapshot = client.post("/ui/sessions").json()
    session_id = snapshot["session_id"]

    snapshot["label"] = "watering experiment"
    snapshot["assistant_messages"] = [{"role": "user", "content": "water bed 1"}]
    assert client.put(f"/ui/sessions/{session_id}", json=snapshot).status_code == 200

    listed = client.get("/ui/sessions").json()["sessions"]
    assert [s["session_id"] for s in listed] == [session_id]
    assert listed[0]["preview"] == "water bed 1"

    loaded = client.get(f"/ui/sessions/{session_id}").json()
    assert loaded["label"] == "watering experiment"

    delete_resp = client.delete(f"/ui/sessions/{session_id}")
    assert delete_resp.json() == {"deleted": True}
    assert client.get(f"/ui/sessions/{session_id}").status_code == 404


def test_garden_entity_appended_to_yaml(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "dev.yaml"
    config.write_text(
        "# keep this comment\nspatial:\n  bounds: {x: 0, y: 0, width: 100, height: 100}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TWFB_CONFIG", str(config))

    r = client.post(
        "/ui/garden/entities",
        json={"x": 25.0, "y": 50.0, "kind": "plant", "name": "Tomato 1"},
    )
    assert r.status_code == 200
    assert r.json()["entity"]["id"] == "tomato_1"

    text = config.read_text(encoding="utf-8")
    assert "# keep this comment" in text  # round-trip preserves comments
    assert "tomato_1" in text


def test_config_endpoint_updates_upstream(client: TestClient) -> None:
    r = client.put("/ui/config", json={"api_url": "http://other:9000/"})
    assert r.json()["api_url"] == "http://other:9000"
    assert client.get("/ui/config").json()["api_url"] == "http://other:9000"
