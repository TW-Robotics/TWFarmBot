"""Optional W&B Weave tracing for the planning harness.

Tracing is enabled by setting ``WEAVE_PROJECT`` or the YAML
``planning.weave_project`` value. When enabled, every model call and tool
invocation is logged to Weave, including token usage, latency, and
reasoning traces.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Iterator

import weave

from .metrics import Metrics
from .reasoning_controller import ReasoningController, content_text

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


def _summarize_model_output(response: Any) -> dict[str, Any]:
    reasoning = ReasoningController()
    thinking = reasoning.extract(response) or ""
    text = reasoning.strip_from_text(content_text(getattr(response, "content", "")) or "")
    calls: list[dict[str, Any]] = []
    for call in getattr(response, "tool_calls", None) or []:
        if isinstance(call, dict):
            name = call.get("name")
            args = call.get("args") or {}
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "args", None) or {}
        if name:
            calls.append({"name": str(name), "args": args if isinstance(args, dict) else {}})
    return {"text": text, "thinking": thinking, "tool_calls": calls}


def log_model_turn(
    response: Any, *, latency_s: float, model: str, parent: Any | None = None
) -> None:
    summary = _summarize_model_output(response)
    summary["latency_s"] = round(latency_s, 3)
    summary["model"] = model
    usage = _extract_usage(response)
    if usage:
        summary.update(usage)
    log_span("twfarmbot.model", {"model": model}, summary, display_name=model, parent=parent)


def _extract_usage(response: Any) -> dict[str, int] | None:
    """Best-effort extraction of token usage from a LangChain response."""
    if response is None:
        return None
    metadata = getattr(response, "response_metadata", {}) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage")
    if isinstance(token_usage, dict):
        out: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = token_usage.get(key)
            if isinstance(value, (int, float)):
                out[key] = int(value)
        return out or None
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        out = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage_metadata.get(key)
            if isinstance(value, (int, float)):
                out[key] = int(value)
        return out or None
    return None


def timed_invoke(
    model: Any,
    messages: Any,
    model_name: str,
    metrics: Metrics | None = None,
    parent: Any | None = None,
) -> Any:
    """Invoke a model and trace it, returning the response."""
    start = time.perf_counter()
    try:
        response = model.invoke(messages)
    except Exception:
        _log.exception("model invoke failed")
        raise
    latency = time.perf_counter() - start
    if is_enabled():
        log_model_turn(response, latency_s=latency, model=model_name, parent=parent)
    if metrics is not None:
        metrics.add_llm_usage(response, latency_s=latency)
    return response


def timed_stream(
    model: Any,
    messages: Any,
    model_name: str,
    metrics: Metrics | None = None,
    parent: Any | None = None,
) -> Iterator[Any]:
    """Stream a model response and trace the aggregated result."""
    start = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[Any] = []
    try:
        for chunk in model.stream(messages):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            chunks.append(chunk)
            yield chunk
    except Exception:
        _log.exception("model stream failed")
        raise
    latency = time.perf_counter() - start
    if is_enabled() and chunks:
        assembled = chunks[0]
        for chunk in chunks[1:]:
            try:
                assembled = assembled + chunk
            except TypeError:
                assembled = chunk
        log_model_turn(assembled, latency_s=latency, model=model_name, parent=parent)
    if metrics is not None:
        ttft = first_token_time or 0.0
        usage_source = chunks[-1] if chunks else None
        metrics.add_llm_usage(usage_source, latency_s=latency, ttft_s=ttft)
