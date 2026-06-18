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
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from farmbot_client import FarmBotConnectionError
from safety_service import UnsafeActionError, validate
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
    # FarmBot commands are blocking and must not overlap. Queue them through a
    # single worker so HTTP clients can receive an acknowledgement immediately.
    app.state.action_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="farmbot-actions"
    )
    app.state.position_poll_stop = Event()
    app.state.position_poll_thread = None
    app.state.position_poll_future = None
    app.state.farmbot_status = "unknown"  # updated by main() at boot

    from twfarmbot_api_server.read import router as read_router
    app.include_router(read_router)

    @app.on_event("startup")
    def start_position_polling() -> None:
        if app.state.farmbot_status != "connected":
            return
        app.state.position_poll_thread = Thread(
            target=_position_poll_loop,
            args=(app,),
            name="farmbot-position-poll",
            daemon=True,
        )
        app.state.position_poll_thread.start()

    @app.on_event("shutdown")
    def stop_position_polling() -> None:
        app.state.position_poll_stop.set()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "actions": app.state.registry.kinds(),
            "farmbot": app.state.farmbot_status,
        }

    @app.post("/actions")
    def post_action(payload: ActionPayload, wait: bool = True) -> dict[str, Any]:
        action = payload.to_action()
        log.info("POST /actions kind=%s params=%s", action.kind, action.params)
        if not wait:
            if action.kind not in app.state.registry.kinds():
                raise HTTPException(status_code=404, detail=f"unknown action kind: {action.kind}")
            try:
                validate(action)
            except UnsafeActionError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err
            app.state.action_executor.submit(_dispatch_queued, app.state.registry, action)
            return {
                "status": "queued",
                "action": {"kind": action.kind, "params": action.params},
            }
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


def _dispatch_queued(registry: ActionRegistry, action: Action) -> None:
    try:
        registry.dispatch(action)
    except Exception:  # noqa: BLE001
        log.exception("queued action failed kind=%s params=%s", action.kind, action.params)
    finally:
        if action.kind in {"move", "find_home"}:
            _refresh_position()


def _refresh_position() -> None:
    from watering_service.backends import farmbot

    try:
        farmbot.backend.refresh_xyz()
    except Exception:  # noqa: BLE001
        log.warning("background position refresh failed", exc_info=True)


def _position_poll_loop(app: FastAPI) -> None:
    """Keep one position refresh queued, never overlapping robot actions."""
    while not app.state.position_poll_stop.is_set():
        future = app.state.position_poll_future
        if future is None or future.done():
            app.state.position_poll_future = app.state.action_executor.submit(
                _refresh_position
            )
        app.state.position_poll_stop.wait(2.0)


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
