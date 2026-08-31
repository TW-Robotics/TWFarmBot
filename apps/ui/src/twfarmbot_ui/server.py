"""Astryx SPA host plus local-only garden/vision/session endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twfarmbot_ui import history
from twfarmbot_ui.local import add_garden_entity, run_vision

STATIC_DIR = Path(__file__).parent / "static"


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


def run() -> None:
    import uvicorn

    from twfarmbot_core.logging import configure_logging

    configure_logging()
    uvicorn.run(
        create_app(),
        host=os.getenv("TWFB_UI_HOST", "0.0.0.0"),
        port=int(os.getenv("TWFB_UI_PORT", "8501")),
    )
