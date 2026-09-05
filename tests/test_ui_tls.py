from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from twfarmbot_ui.server import create_app
from twfarmbot_ui.tls import ensure_certs


def test_ensure_certs_reuses_existing_files(tmp_path) -> None:
    cert, key = ensure_certs(tmp_path)
    assert cert.is_file()
    assert key.is_file()
    first = cert.read_bytes()
    again, _ = ensure_certs(tmp_path)
    assert again.read_bytes() == first


def test_api_proxy_forwards_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWFB_API_URL", "http://127.0.0.1:8000")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"text": "hello"})

    app = create_app()
    app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        response = client.post("/api/chat/transcribe", json={"format": "webm"})
    assert response.status_code == 200
    assert response.json() == {"text": "hello"}
    assert seen["url"] == "http://127.0.0.1:8000/chat/transcribe"
    assert "webm" in seen["body"]
