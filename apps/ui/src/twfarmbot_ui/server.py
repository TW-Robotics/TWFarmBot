"""FastAPI server for the Material 3 web UI.

Serves the static single-page frontend and reverse-proxies every API call:

* ``/api/*``     → the twfarmbot api_server (``TWFB_API_URL``)
* ``/resireg/*`` → the ReSiReg vision server (``TWFB_RESIREG_URL``)
* ``/ui/*``      → small UI-local endpoints (session persistence and
                   garden-entity YAML writes) that used to live inside the
                   Streamlit process.

Proxying keeps the frontend same-origin (no CORS on the api_server) and
keeps the upstream URLs server-side, exactly like the old Streamlit app.
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from twfarmbot_ui import history

STATIC_DIR = Path(__file__).parent / "static"

# Hop-by-hop headers must not be forwarded by a proxy (RFC 9110 §7.6.1).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

# Generous read timeout: /chat and /plan can block on the LLM for minutes.
_PROXY_TIMEOUT = httpx.Timeout(300.0, connect=5.0)


class GardenEntityPayload(BaseModel):
    x: float
    y: float
    kind: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)


class ConfigPayload(BaseModel):
    api_url: str | None = None
    resireg_url: str | None = None


def _entity_id(name: str) -> str:
    base = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return base or "entity"


def add_garden_entity(x: float, y: float, kind: str, name: str) -> dict[str, Any]:
    """Append a new entity to the ``TWFB_CONFIG`` YAML, preserving comments."""
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    path = Path(os.getenv("TWFB_CONFIG", "configs/dev.yaml"))
    with path.open(encoding="utf-8") as fh:
        data = yaml.load(fh)
    spatial = data.setdefault("spatial", {})
    entities = spatial.setdefault("entities", [])
    entity = {
        "id": _entity_id(name),
        "kind": kind,
        "name": name,
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "radius_mm": 50,
        "metadata": {},
    }
    entities.append(entity)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    return entity


def create_app(http_client: httpx.AsyncClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        await application.state.http.aclose()

    app = FastAPI(title="TWFarmBot UI", version="0.2.0", lifespan=lifespan)
    app.state.api_base = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000").rstrip("/")
    app.state.resireg_base = os.getenv(
        "TWFB_RESIREG_URL", "http://127.0.0.1:8080"
    ).rstrip("/")
    app.state.http = http_client or httpx.AsyncClient(timeout=_PROXY_TIMEOUT)

    async def _proxy(request: Request, base: str, path: str) -> StreamingResponse:
        client: httpx.AsyncClient = app.state.http
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        upstream = client.build_request(
            request.method,
            f"{base}/{path}",
            params=request.query_params,
            headers=headers,
            content=await request.body(),
        )
        try:
            resp = await client.send(upstream, stream=True)
        except httpx.HTTPError as err:
            raise HTTPException(
                status_code=502, detail=f"{type(err).__name__}: {err}"
            ) from err
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
            },
            background=BackgroundTask(resp.aclose),
        )

    @app.api_route(
        "/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
    )
    async def api_proxy(request: Request, path: str) -> StreamingResponse:
        return await _proxy(request, app.state.api_base, path)

    @app.api_route("/resireg/{path:path}", methods=["GET", "POST"])
    async def resireg_proxy(request: Request, path: str) -> StreamingResponse:
        return await _proxy(request, app.state.resireg_base, path)

    # ---- UI-local endpoints ---------------------------------------------

    @app.get("/ui/config")
    def get_config() -> dict[str, Any]:
        return {
            "api_url": app.state.api_base,
            "resireg_url": app.state.resireg_base,
        }

    @app.put("/ui/config")
    def put_config(payload: ConfigPayload) -> dict[str, Any]:
        if payload.api_url:
            app.state.api_base = payload.api_url.rstrip("/")
        if payload.resireg_url:
            app.state.resireg_base = payload.resireg_url.rstrip("/")
        return get_config()

    @app.get("/ui/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": history.list_sessions()}

    @app.post("/ui/sessions")
    def new_session() -> dict[str, Any]:
        return history.empty_snapshot()

    @app.get("/ui/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        snapshot = history.load_session(session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"no session: {session_id}")
        return snapshot

    @app.put("/ui/sessions/{session_id}")
    async def put_session(session_id: str, request: Request) -> dict[str, Any]:
        snapshot = await request.json()
        if not isinstance(snapshot, dict):
            raise HTTPException(status_code=400, detail="snapshot must be an object")
        snapshot["session_id"] = session_id
        history.save_session(snapshot)
        return {"status": "ok", "session_id": session_id}

    @app.delete("/ui/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        return {"deleted": history.delete_session(session_id)}

    @app.post("/ui/garden/entities")
    def post_garden_entity(payload: GardenEntityPayload) -> dict[str, Any]:
        try:
            entity = add_garden_entity(payload.x, payload.y, payload.kind, payload.name)
        except (OSError, KeyError, TypeError) as err:
            raise HTTPException(
                status_code=500, detail=f"config write failed: {err}"
            ) from err
        return {"status": "ok", "entity": entity}

    # ---- Static frontend -------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-cache"},
        )

    @app.middleware("http")
    async def cache_static_assets(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/app.css" or path.startswith("/js/") or path.startswith("/vendor/"):
            # /vendor/ files are SHA256-pinned, so a long cache is safe.
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")

    return app


app = create_app()


def main() -> None:
    from twfarmbot_core.logging import configure_logging

    configure_logging()
    import uvicorn

    port = int(os.getenv("TWFB_UI_PORT", "8501"))
    uvicorn.run("twfarmbot_ui.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
