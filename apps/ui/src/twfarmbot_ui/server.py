"""Astryx SPA host plus local-only garden/vision/session endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from twfarmbot_ui import history
from twfarmbot_ui.local import add_garden_entity, run_vision

STATIC_DIR = Path(__file__).parent / "static"
_HOP = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_LOG = logging.getLogger(__name__)


class GardenEntityPayload(BaseModel):
    x: float
    y: float
    kind: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)


class VisionPayload(BaseModel):
    image_url: str
    mode: str
    prompt: str = ""
    classes: str = ""
    negative: str = ""
    negatives: str = ""
    n_clusters: int = 6


def _api_base() -> str:
    return os.getenv("TWFB_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _http(app: FastAPI) -> httpx.AsyncClient:
    client = getattr(app.state, "http", None)
    if client is None:
        client = httpx.AsyncClient(timeout=None, follow_redirects=False)
        app.state.http = client
    return client


def create_app() -> FastAPI:
    app = FastAPI(title="TWFarmBot UI")

    @app.post("/local/garden-entities")
    def garden_entities(payload: GardenEntityPayload) -> dict[str, str]:
        try:
            add_garden_entity(payload.x, payload.y, payload.kind, payload.name)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/local/vision")
    def vision(payload: VisionPayload) -> dict[str, Any]:
        try:
            return run_vision(payload.model_dump())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/local/sessions")
    def sessions() -> dict[str, Any]:
        return {"sessions": history.list_sessions()}

    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def api_proxy(path: str, request: Request) -> StreamingResponse:
        url = f"{_api_base()}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {key: value for key, value in request.headers.items() if key.lower() not in _HOP}
        client = _http(request.app)
        upstream = await client.send(
            client.build_request(
                request.method,
                url,
                headers=headers,
                content=await request.body(),
            ),
            stream=True,
        )
        out_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _HOP
        }
        return StreamingResponse(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=out_headers,
            background=BackgroundTask(upstream.aclose),
        )

    if STATIC_DIR.is_dir():
        assets = STATIC_DIR / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            candidate = STATIC_DIR / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            index = STATIC_DIR / "index.html"
            if not index.is_file():
                raise HTTPException(
                    status_code=503,
                    detail="UI assets are missing. Run `npm run build` in apps/ui.",
                )
            return FileResponse(index)

    return app


def _tls_enabled() -> bool:
    return os.getenv("TWFB_UI_TLS", "1").strip().lower() not in {"0", "false", "no", "off"}


def run() -> None:
    import uvicorn

    from twfarmbot_core.logging import configure_logging
    from twfarmbot_ui.tls import ensure_certs

    configure_logging()
    kwargs: dict[str, Any] = {
        "host": os.getenv("TWFB_UI_HOST", "0.0.0.0"),
        "port": int(os.getenv("TWFB_UI_PORT", "8501")),
    }
    if _tls_enabled():
        cert, key = ensure_certs()
        kwargs["ssl_certfile"] = str(cert)
        kwargs["ssl_keyfile"] = str(key)
        _LOG.info("UI TLS enabled (%s); open https://<this-host>:%s", cert, kwargs["port"])
    uvicorn.run(create_app(), **kwargs)
