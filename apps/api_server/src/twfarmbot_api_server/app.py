"""FastAPI app exposing the FarmBot / sensor / experiment API.

HTTP transport only. Action dispatch logic lives in ``core.actions`` and
is shared with apps/worker (see docs/architecture.md).

The api_server is the canonical entry point for the whole system, so it
connects to the FarmBot on startup (see ``connect_to_farmbot``) and
surfaces the live status via ``GET /health``.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from farmbot_client import FarmBotConnectionError
from safety_service import UnsafeActionError, validate
from twfarmbot_api_server.handlers import register_default_handlers
from planning_service.config import load_config
from planning_service.providers import get_provider, list_provider_names
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


class PlanPayload(BaseModel):
    request: str = Field(..., min_length=1, description="Natural-language task.")
    debug: bool = Field(
        default=False,
        description=(
            "If true, the response includes the raw LLM text and any "
            "introspection tool calls the model made, for debugging."
        ),
    )
    model: str | None = Field(
        default=None,
        description="Optional model override. Uses the configured default if omitted.",
    )


class ChatPayload(BaseModel):
    messages: list[dict[str, Any]] = Field(
        ...,
        description="Conversation history in OpenAI format (user/assistant turns).",
    )
    allow_actions: bool = Field(
        default=True,
        description="If false, the model only has read-only introspection tools.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override. Uses the configured default if omitted.",
    )


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

    @app.get("/providers")
    def list_providers() -> dict[str, Any]:
        cfg = load_config()
        return {
            "providers": list_provider_names(),
            "current": cfg.provider,
        }

    @app.get("/models")
    def list_models(provider: str | None = None) -> dict[str, Any]:
        cfg = load_config()
        try:
            prov = get_provider(provider or cfg.provider)
            models = prov.list_models(cfg)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return {
            "provider": provider or cfg.provider,
            "models": models,
            "current": cfg.model,
        }

    @app.post("/actions")
    def post_action(payload: ActionPayload, wait: bool = True) -> dict[str, Any]:
        action = payload.to_action()
        log.info("POST /actions kind=%s params=%s", action.kind, action.params)
        if not wait:
            if action.kind not in app.state.registry.kinds():
                raise HTTPException(
                    status_code=404, detail=f"unknown action kind: {action.kind}"
                )
            try:
                validate(action)
            except UnsafeActionError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err
            app.state.action_executor.submit(
                _dispatch_queued, app.state.registry, action
            )
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
        except FarmBotConnectionError as err:
            raise HTTPException(
                status_code=502,
                detail=f"FarmBot not connected: {err}",
            ) from err
        except Exception as err:  # noqa: BLE001 — surface real cause to the UI
            log.exception("action failed kind=%s params=%s", action.kind, action.params)
            raise HTTPException(
                status_code=500,
                detail=f"{type(err).__name__}: {err}",
            ) from err
        return {
            "status": "ok",
            "action": {"kind": executed.kind, "params": executed.params},
        }

    @app.post("/plan")
    def post_plan(
        payload: PlanPayload,
        execute: bool = False,
    ) -> dict[str, Any]:
        """Translate a natural-language request into a validated Action[].

        With ``execute=false`` (the default), returns the proposed plan
        for UI preview. With ``execute=true``, dispatches each action
        through the same registry + safety gate as ``POST /actions``.
        """
        log.info("POST /plan request=%r execute=%s", payload.request, execute)
        try:
            from planning_service import plan as planner_plan
            from planning_service import PlanError
            from planning_service.introspection import HttpSystemStateProvider

            # The introspection provider points back at ourselves so
            # the planner can read state through our own endpoints.
            api_base = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")
            system_state = HttpSystemStateProvider(api_base)
            world = _world_snapshot()

            result = planner_plan(
                payload.request,
                registry=app.state.registry,
                world=world,
                system_state=system_state,
                model_name=payload.model,
            )
        except PlanError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except UnsafeActionError as err:
            raise HTTPException(status_code=400, detail=f"unsafe plan: {err}") from err
        except Exception as err:
            # LLM timeouts, network errors, missing API keys, etc. all
            # land here. Surface the type + message in the HTTP response
            # so the UI's debug expander shows the real cause.
            cfg = load_config()
            log.exception("planner call failed for request=%r", payload.request)
            raise HTTPException(
                status_code=504
                if "timeout" in str(err).lower() or "Timeout" in type(err).__name__
                else 500,
                detail=(
                    f"planner call to {cfg.base_url}/{cfg.model} failed: "
                    f"{type(err).__name__}: {err}. "
                    f"Current timeout: {cfg.timeout_s}s. "
                    f"Raise PLANNING_LLM_TIMEOUT_S or the YAML 'planning.timeout_s' "
                    f"if this is a slow model."
                ),
            ) from err

        actions = list(result.actions)
        body: dict[str, Any] = {
            "status": "ok",
            "request": payload.request,
            "actions": [{"kind": a.kind, "params": a.params} for a in actions],
            "rationale": result.rationale,
        }
        if payload.debug:
            body["debug"] = {
                "raw_text": result.raw_text,
                "action_count": len(actions),
                "model": os.getenv("PLANNING_LLM_MODEL"),
                "base_url": os.getenv("PLANNING_LLM_BASE_URL"),
            }
        if execute and actions:
            results: list[dict[str, Any]] = []
            for action in actions:
                try:
                    app.state.registry.dispatch(action)
                    results.append(
                        {"kind": action.kind, "status": "ok", "params": action.params}
                    )
                except UnknownActionError as err:
                    results.append(
                        {
                            "kind": action.kind,
                            "status": "error",
                            "error": f"unknown: {err}",
                        }
                    )
                except UnsafeActionError as err:
                    results.append(
                        {
                            "kind": action.kind,
                            "status": "error",
                            "error": f"unsafe: {err}",
                        }
                    )
                except Exception as err:  # noqa: BLE001 — surface as per-action error, don't 500
                    log.exception(
                        "planner action failed kind=%s params=%s",
                        action.kind,
                        action.params,
                    )
                    results.append(
                        {
                            "kind": action.kind,
                            "status": "error",
                            "error": f"{type(err).__name__}: {err}",
                        }
                    )
            body["results"] = results
        return body

    @app.post("/chat")
    def post_chat(payload: ChatPayload) -> dict[str, Any]:
        """Conversational assistant that can read state and execute actions."""
        log.info(
            "POST /chat messages=%d allow_actions=%s",
            len(payload.messages),
            payload.allow_actions,
        )
        try:
            from planning_service import ChatResult, chat as planner_chat
            from planning_service.introspection import HttpSystemStateProvider

            api_base = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")
            system_state = HttpSystemStateProvider(api_base)
            world = _world_snapshot()

            result: ChatResult = planner_chat(
                payload.messages,
                registry=app.state.registry,
                world=world,
                system_state=system_state,
                allow_actions=payload.allow_actions,
                propose_only=True,
                model_name=payload.model,
            )
        except Exception as err:  # noqa: BLE001
            cfg = load_config()
            log.exception("chat call failed")
            raise HTTPException(
                status_code=504
                if "timeout" in str(err).lower() or "Timeout" in type(err).__name__
                else 500,
                detail=(
                    f"chat call to {cfg.base_url}/{cfg.model} failed: "
                    f"{type(err).__name__}: {err}. "
                    f"Current timeout: {cfg.timeout_s}s."
                ),
            ) from err

        return {
            "status": "ok",
            "response": result.response,
            "thinking": result.thinking,
            "tool_calls": result.tool_calls,
            "messages": result.messages,
            "metrics": result.metrics,
        }

    @app.post("/chat/stream")
    def post_chat_stream(payload: ChatPayload) -> StreamingResponse:
        """Streaming conversational assistant (Server-Sent Events)."""
        log.info("POST /chat/stream messages=%d", len(payload.messages))

        from planning_service import stream_chat as planner_stream_chat
        from planning_service.introspection import HttpSystemStateProvider

        api_base = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")
        system_state = HttpSystemStateProvider(api_base)
        world = _world_snapshot()

        def event_generator():
            try:
                for event in planner_stream_chat(
                    payload.messages,
                    registry=app.state.registry,
                    world=world,
                    system_state=system_state,
                    allow_actions=payload.allow_actions,
                    propose_only=True,
                    model_name=payload.model,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as err:  # noqa: BLE001
                log.exception("chat stream failed")
                yield f"data: {json.dumps({'type': 'error', 'error': f'{type(err).__name__}: {err}'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


def _dispatch_queued(registry: ActionRegistry, action: Action) -> None:
    try:
        registry.dispatch(action)
    except Exception:  # noqa: BLE001
        log.exception(
            "queued action failed kind=%s params=%s", action.kind, action.params
        )
    finally:
        if action.kind in {"move", "find_home"}:
            _refresh_position()


def _world_snapshot() -> Any:
    """Build a world-model snapshot including the last known robot position."""
    from spatial_service import get_snapshot
    from watering_service.backends import farmbot

    try:
        xyz = farmbot.backend.get_xyz()
    except Exception:  # noqa: BLE001
        xyz = None
    return get_snapshot(xyz)


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
        settings.env,
        app.state.registry.kinds(),
    )
    connect_to_farmbot(required=True)
    import uvicorn

    uvicorn.run("twfarmbot_api_server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
