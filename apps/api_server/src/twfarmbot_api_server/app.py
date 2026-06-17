"""FastAPI app exposing the FarmBot / sensor / experiment API.

HTTP transport only. Action dispatch logic lives in ``core.actions`` and
is shared with apps/worker (see docs/architecture.md).

The api_server is the canonical entry point for the whole system, so it
connects to the FarmBot on startup (see ``connect_to_farmbot``) and
surfaces the live status via ``GET /health``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from farmbot_client import FarmBotConnectionError
from safety_service import UnsafeActionError
from twfarmbot_api_server.handlers import register_default_handlers
from twfarmbot_core.actions import (
    ActionRegistry,
    UnknownActionError,
)
from twfarmbot_core.domain import Action
from twfarmbot_core.logging import configure_logging

log = logging.getLogger("twfarmbot.api_server")


class ActionPayload(BaseModel):
    kind: str = Field(..., min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    def to_action(self) -> Action:
        return Action(kind=self.kind, params=self.params)


def create_app(registry: ActionRegistry | None = None) -> FastAPI:
    app = FastAPI(title="TWFarmBot API", version="0.4.0")
    app.state.registry = registry or _default_registry()
    app.state.farmbot_status = "unknown"  # updated by main() at boot

    from twfarmbot_api_server.read import router as read_router
    app.include_router(read_router)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "actions": app.state.registry.kinds(),
            "farmbot": app.state.farmbot_status,
        }

    @app.post("/actions")
    def post_action(payload: ActionPayload) -> dict[str, Any]:
        action = payload.to_action()
        log.info("POST /actions kind=%s params=%s", action.kind, action.params)
        try:
            executed = app.state.registry.dispatch(action)
        except UnknownActionError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err
        except UnsafeActionError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return {
            "status": "ok",
            "action": {"kind": executed.kind, "params": executed.params},
        }

    return app


def _default_registry() -> ActionRegistry:
    r = ActionRegistry()
    register_default_handlers(r)
    return r


def connect_to_farmbot(required: bool = True) -> str:
    """Eager-connect to the FarmBot at server boot.

    Returns a short status string (``"connected"`` / ``"skipped"`` /
    ``"failed: <reason>"``) and stashes it on ``app.state.farmbot_status``
    so ``GET /health`` can report it.

    Set ``FARMBOT_REQUIRED=0`` to allow boot without a live bot (useful
    for UI-only or offline dev). When ``required=True`` (the default),
    a failed connection raises ``SystemExit`` so uvicorn never starts
    with a dead upstream.
    """
    if not required and os.getenv("FARMBOT_REQUIRED", "1") == "0":
        log.warning("FarmBot connection skipped (FARMBOT_REQUIRED=0)")
        app.state.farmbot_status = "skipped"
        return "skipped"

    log.info("connecting to FarmBot (startup probe)…")
    try:
        from farmbot_gateway import get_farmbot

        get_farmbot()
    except FarmBotConnectionError as err:
        cause = type(err.__cause__).__name__ if err.__cause__ else "?"
        status = f"failed: {err} (cause: {cause})"
        app.state.farmbot_status = status
        log.error("FarmBot startup connect failed: %s", err)
        if required:
            raise SystemExit(
                f"\nFATAL: could not connect to FarmBot at boot.\n  {status}\n"
                f"Fix credentials/network, or set FARMBOT_REQUIRED=0 to boot anyway.\n"
            ) from err
        return status

    app.state.farmbot_status = "connected"
    log.info("FarmBot connected")
    return "connected"


app = create_app()


def main() -> None:
    from twfarmbot_core.config import load_settings
    settings = load_settings()
    configure_logging(settings.log_level)
    logging.getLogger("twfarmbot.api_server").info(
        "Starting TWFarmBot API server in env=%s (actions=%s)",
        settings.env, app.state.registry.kinds(),
    )
    connect_to_farmbot(required=True)
    import uvicorn
    uvicorn.run("twfarmbot_api_server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
