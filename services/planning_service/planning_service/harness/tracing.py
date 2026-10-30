"""Optional W&B Weave tracing for the planning harness.

Tracing is enabled by setting ``WEAVE_PROJECT`` or the YAML
``planning.weave_project`` value. When enabled, every tool invocation and reasoning trace
is logged to Weave under one parent call per chat stream.
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from typing import Any

import weave

_log = logging.getLogger(__name__)

_weave_initialized: bool = False
_parent_call: ContextVar[Any] = ContextVar("twfb_weave_parent", default=None)


def init_weave(project_name: str | None) -> None:
    """Initialize the Weave client once.

    No-op if ``project_name`` is empty, Weave is already initialized, or
    the code is running inside pytest.
    """
    global _weave_initialized
    if _weave_initialized or not project_name:
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        weave.init(project_name, settings={"print_call_link": False})
        _weave_initialized = True
        _log.info("weave tracing initialized for project %s", project_name)
    except Exception as err:  # noqa: BLE001
        _log.warning("weave initialization failed: %s", err)


def is_enabled() -> bool:
    return _weave_initialized


def start_agent_call(*, thread_id: str, prompt: str = "") -> Any | None:
    """Open a parent Weave call for one chat stream so the UI can link it."""
    if not is_enabled():
        return None
    try:
        client = weave.get_client()
        if client is None:
            return None
        call = client.create_call(
            "twfarmbot.agent",
            {"thread_id": thread_id, "prompt": prompt[:2000]},
            display_name="TWFarmBot chat",
        )
        _parent_call.set(call)
        return call
    except Exception as err:  # noqa: BLE001
        _log.warning("weave create_call failed: %s", err)
        return None


def call_url(call: Any) -> str | None:
    url = getattr(call, "ui_url", None)
    return str(url) if url else None


def finish_agent_call(
    call: Any,
    output: dict[str, Any] | None = None,
    exception: BaseException | None = None,
) -> None:
    if call is None or not is_enabled():
        return
    try:
        client = weave.get_client()
        if client is None:
            return
        client.finish_call(call, output=_safe(output), exception=exception)
    except Exception as err:  # noqa: BLE001
        _log.warning("weave finish_call failed: %s", err)
    finally:
        _parent_call.set(None)


def _parent() -> Any | None:
    return weave.get_current_call() or _parent_call.get()


def _safe(value: Any) -> Any:
    """JSON-friendly values so Weave does not drop the span."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def log_span(
    op: str,
    inputs: dict[str, Any],
    output: Any,
    *,
    display_name: str | None = None,
    parent: Any | None = None,
) -> None:
    if not is_enabled():
        return
    try:
        weave.log_call(
            op,
            _safe(inputs),
            _safe(output),
            parent=parent if parent is not None else _parent(),
            display_name=display_name,
            use_stack=False,
        )
    except Exception as err:  # noqa: BLE001
        _log.warning("weave log_call failed (%s): %s", op, err)


def log_tool_call(
    name: str,
    args: dict[str, Any],
    result: Any,
    *,
    latency_s: float,
    parent: Any | None = None,
) -> None:
    log_span(
        "twfarmbot.tool",
        {"name": name, "args": args},
        {"result": result, "latency_s": round(latency_s, 3)},
        display_name=name,
        parent=parent,
    )
