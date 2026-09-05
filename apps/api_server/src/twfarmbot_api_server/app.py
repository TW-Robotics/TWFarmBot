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
from dataclasses import replace
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from farmbot_client import FarmBotConnectionError
from safety_service import UnsafeActionError, validate
from twfarmbot_api_server.handlers import register_default_handlers
from planning_service.config import PlannerConfig, apply_overrides, load_config
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


class LlmOverrides(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_s: float | None = None
    temperature: float | None = None


class VertexPayload(BaseModel):
    project: str | None = Field(
        default=None,
        description="GCP project id. Blank clears the stored value.",
    )
    location: str | None = Field(
        default=None,
        description="Vertex region. Blank clears the stored value.",
    )


class VoicePayload(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="Muse Voice Transcribe key. Blank clears the stored value.",
    )


class PlanningPayload(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_s: float | None = None
    temperature: float | None = None


class LlmKeysPayload(BaseModel):
    keys: dict[str, str | None] = Field(
        default_factory=dict,
        description=(
            "Per-provider API keys. Blank or null values delete the stored key."
        ),
    )
    vertex: VertexPayload | None = Field(
        default=None,
        description="Vertex AI project/location (project id is not secret).",
    )
    voice: VoicePayload | None = Field(
        default=None,
        description="Muse Voice Transcribe key (never returned on GET).",
    )
    planning: PlanningPayload | None = Field(
        default=None,
        description="Default chat provider/model saved from the Settings tab.",
    )


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
    llm: LlmOverrides | None = Field(
        default=None,
        description="Optional per-request LLM credentials/settings from the UI.",
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
    llm: LlmOverrides | None = Field(
        default=None,
        description="Optional per-request LLM credentials/settings from the UI.",
    )
    thread_id: str | None = Field(
        default=None,
        description="LangGraph thread to resume after an approval interrupt.",
    )
    approved_ids: list[str] | None = Field(
        default=None,
        description="Tool-call ids to run after approval. Empty list rejects them.",
    )
    skip_approval: bool = Field(
        default=False,
        description="If true, run risky tools without an operator interrupt.",
    )


class ChatCancelPayload(BaseModel):
    thread_id: str | None = None


class ChatFollowupPayload(BaseModel):
    text: str = Field(..., min_length=1)
    thread_id: str | None = None


class ChatTranscribePayload(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    format: str = Field(default="webm")
    llm: LlmOverrides | None = None


class ModelsPayload(BaseModel):
    llm: LlmOverrides | None = None


def _resolve_llm_config(
    llm: LlmOverrides | None,
    model_name: str | None = None,
) -> PlannerConfig:
    from planning_service.config import resolve_api_key_for

    cfg = load_config()
    if llm is not None:
        cfg = apply_overrides(cfg, **llm.model_dump(exclude_none=True))
    if model_name:
        cfg = replace(cfg, model=model_name)
    if cfg.api_key is None:
        cfg = replace(cfg, api_key=resolve_api_key_for(cfg.provider))
    return cfg


def create_app(registry: ActionRegistry | None = None) -> FastAPI:
    app = FastAPI(title="TWFarmBot API", version="0.4.0")
    # The native Assistant runs in the dashboard browser component.  Keep the
    # browser bridge opt-in for deployments, while allowing local/LAN demos by
    # default (the API has no cookie-authenticated browser session).
    configured_origins = os.getenv("TWFB_CORS_ORIGINS", "")
    cors_origins = [
        origin.strip() for origin in configured_origins.split(",") if origin.strip()
    ] or [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://localhost:8501",
        "https://127.0.0.1:8501",
        "null",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=(
            r"^https?://(?:(?:localhost)|(?:127\.0\.0\.1)|"
            r"(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3})|"
            r"(?:192\.168\.\d{1,3}\.\d{1,3})|"
            r"(?:172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})|"
            r"(?:100\.(?:6[4-9]|[78]\d|9\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})"
            r")(?:\:\d+)?$"
        ),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
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
    from twfarmbot_api_server.spectral import router as spectral_router

    app.include_router(read_router)
    app.include_router(spectral_router)

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
            prov = get_provider(
                provider or cfg.provider, permissive=cfg.permissive_provider
            )
            models = prov.list_models(cfg)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return {
            "provider": provider or cfg.provider,
            "models": models,
            "current": cfg.model,
            "api_key_configured": bool(cfg.api_key),
        }

    @app.post("/models/list")
    def list_models_with_overrides(payload: ModelsPayload) -> dict[str, Any]:
        cfg = _resolve_llm_config(payload.llm)
        try:
            prov = get_provider(cfg.provider, permissive=cfg.permissive_provider)
            models = prov.list_models(cfg)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return {
            "provider": cfg.provider,
            "models": models,
            "current": cfg.model,
            "api_key_configured": bool(cfg.api_key),
        }

    def _llm_settings_view() -> dict[str, Any]:
        from planning_service.config import (
            planning_env_overrides,
            resolve_vertex_settings,
            stored_keys_configured,
            vertex_env_overrides,
            voice_key_status,
        )

        cfg = load_config()
        return {
            "provider": cfg.provider,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "timeout_s": cfg.timeout_s,
            "temperature": cfg.temperature,
            "api_key_configured": bool(cfg.api_key),
            "keys_configured": stored_keys_configured(),
            "providers": list_provider_names(),
            "vertex": resolve_vertex_settings(),
            "vertex_env": vertex_env_overrides(),
            "voice": voice_key_status(),
            "env_overrides": planning_env_overrides(),
        }

    @app.get("/settings/llm")
    def get_llm_settings() -> dict[str, Any]:
        return _llm_settings_view()

    @app.put("/settings/llm")
    def put_llm_settings(payload: LlmKeysPayload) -> dict[str, Any]:
        from planning_service.config import (
            write_stored_keys,
            write_stored_planning,
            write_stored_voice_key,
            write_vertex_settings,
        )

        log.info("PUT /settings/llm providers=%s", sorted(str(k) for k in payload.keys))
        write_stored_keys(payload.keys)
        if payload.planning is not None:
            write_stored_planning(**payload.planning.model_dump())
        if payload.voice is not None:
            write_stored_voice_key(payload.voice.api_key)
        if payload.vertex is not None:
            write_vertex_settings(
                project=payload.vertex.project,
                location=payload.vertex.location,
            )
        return {"status": "ok", **_llm_settings_view()}

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
        if action.kind in {"move", "move_axis", "find_home"}:
            _refresh_position()
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

            llm_cfg = _resolve_llm_config(payload.llm, payload.model)

            result = planner_plan(
                payload.request,
                registry=app.state.registry,
                world=world,
                system_state=system_state,
                config=llm_cfg,
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

            llm_cfg = _resolve_llm_config(payload.llm, payload.model)

            result: ChatResult = planner_chat(
                payload.messages,
                registry=app.state.registry,
                world=world,
                system_state=system_state,
                allow_actions=payload.allow_actions,
                propose_only=False,
                config=llm_cfg,
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
                llm_cfg = _resolve_llm_config(payload.llm, payload.model)
                for event in planner_stream_chat(
                    payload.messages,
                    registry=app.state.registry,
                    world=world,
                    system_state=system_state,
                    allow_actions=payload.allow_actions,
                    propose_only=False,
                    config=llm_cfg,
                    model_name=payload.model,
                    thread_id=payload.thread_id,
                    approved_ids=payload.approved_ids,
                    skip_approval=payload.skip_approval,
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

    @app.post("/chat/cancel")
    def post_chat_cancel(payload: ChatCancelPayload) -> dict[str, Any]:
        """Stop the in-flight agent loop and halt the gantry."""
        from planning_service.harness.cancel import cancel_run
        from watering_service.backends import farmbot

        thread_id = cancel_run(payload.thread_id)
        try:
            farmbot.backend.e_stop()
        except Exception:  # noqa: BLE001
            log.warning("e_stop after chat cancel failed", exc_info=True)
        return {"status": "ok", "thread_id": thread_id}

    @app.post("/chat/followup")
    def post_chat_followup(payload: ChatFollowupPayload) -> dict[str, Any]:
        """Queue a user utterance on the in-flight run. Does not e-stop."""
        from planning_service.harness.cancel import FollowupRejected, enqueue_followup

        try:
            thread_id = enqueue_followup(payload.thread_id, payload.text)
        except FollowupRejected as err:
            raise HTTPException(status_code=409, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return {"status": "ok", "thread_id": thread_id}

    @app.post("/voice/session")
    def post_voice_session() -> dict[str, Any]:
        """Mint a short-lived OpenAI Realtime token for the browser."""
        from planning_service.realtime import create_client_secret

        try:
            return create_client_secret()
        except RuntimeError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    @app.post("/chat/transcribe")
    def post_chat_transcribe(payload: ChatTranscribePayload) -> dict[str, str]:
        """Speech-to-text via OpenAI, with Muse Voice as fallback."""
        import base64

        from planning_service.transcribe import TRANSCRIBE_MODEL, transcribe_audio

        try:
            audio = base64.b64decode(payload.audio_base64, validate=False)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="invalid audio_base64") from err
        cfg = _resolve_llm_config(payload.llm)
        try:
            text = transcribe_audio(audio, fmt=payload.format, config=cfg)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except RuntimeError as err:
            raise HTTPException(status_code=502, detail=str(err)) from err
        return {"text": text, "model": TRANSCRIBE_MODEL}

    return app


def _dispatch_queued(registry: ActionRegistry, action: Action) -> None:
    try:
        registry.dispatch(action)
    except Exception:  # noqa: BLE001
        log.exception(
            "queued action failed kind=%s params=%s", action.kind, action.params
        )
    finally:
        if action.kind in {"move", "move_axis", "find_home"}:
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

    log.info("connecting to local FarmBot HTTP API…")
    try:
        from farmbot_gateway import get_farmbot
        from farmbot_client import connect

        connect()
        get_farmbot()
    except FarmBotConnectionError as err:
        cause = type(err.__cause__).__name__ if err.__cause__ else "?"
        status = f"failed: {err} (cause: {cause})"
        app.state.farmbot_status = status
        log.error("FarmBot startup connect failed: %s", err)
        if required:
            raise SystemExit(
                f"\nFATAL: could not connect to local FarmBot at boot.\n  {status}\n"
                f"Set FARMBOT_LOCAL_URL, or FARMBOT_REQUIRED=0 to boot anyway.\n"
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
    connect_to_farmbot(required=os.getenv("FARMBOT_REQUIRED", "1") != "0")
    import uvicorn

    uvicorn.run("twfarmbot_api_server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
