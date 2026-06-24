"""Optional W&B Weave tracing for the planning harness.

Tracing is enabled by setting ``WEAVE_PROJECT`` or the YAML
``planning.weave_project`` value. When enabled, every model call and tool
invocation is logged to Weave, including token usage, latency, and
reasoning traces.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterator

import weave

_log = logging.getLogger(__name__)

_weave_initialized: bool = False


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
        weave.init(project_name)
        _weave_initialized = True
        _log.info("weave tracing initialized for project %s", project_name)
    except Exception as err:  # noqa: BLE001
        _log.warning("weave initialization failed: %s", err)


def is_enabled() -> bool:
    return _weave_initialized


def langchain_tracer() -> Any | None:
    """Return a fresh LangChain WeaveTracer, or None if Weave is off."""
    if not _weave_initialized:
        return None
    try:
        from weave.integrations.langchain.langchain import WeaveTracer

        return WeaveTracer()
    except Exception as err:  # noqa: BLE001
        _log.warning("could not create WeaveTracer: %s", err)
        return None


@weave.op()  # type: ignore[misc]
def trace_tool_call(
    name: str, args: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Trace a single tool invocation.

    This is a no-op at runtime unless Weave has been initialized; the
    decorator simply records inputs/outputs when tracing is active.
    """
    return {"name": name, "args": args, "result": result}


@weave.op()  # type: ignore[misc]
def trace_model_invoke(response: Any, *, latency_s: float, model: str) -> Any:
    """Trace a non-streaming model response.

    We attach token usage and latency as attributes when available.
    """
    usage = _extract_usage(response)
    attrs: dict[str, Any] = {"model": model, "latency_s": latency_s}
    if usage:
        attrs.update(usage)
    with weave.attributes(attrs):
        return response


@weave.op()  # type: ignore[misc]
def trace_model_stream(chunks: list[Any], *, latency_s: float, model: str) -> list[Any]:
    """Trace a streaming model response.

    Chunks are collected before this op is called so the whole stream is
    captured as a single trace with aggregated metadata.
    """
    usage = _extract_usage(chunks[-1]) if chunks else None
    attrs: dict[str, Any] = {"model": model, "latency_s": latency_s, "stream": True}
    if usage:
        attrs.update(usage)
    with weave.attributes(attrs):
        return chunks


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


def timed_invoke(model: Any, messages: Any, model_name: str) -> Any:
    """Invoke a model and trace it, returning the response."""
    start = time.perf_counter()
    tracer = langchain_tracer()
    kwargs: dict[str, Any] = {"config": {"callbacks": [tracer]}} if tracer else {}
    try:
        response = model.invoke(messages, **kwargs)
    except Exception:
        _log.exception("model invoke failed")
        raise
    latency = time.perf_counter() - start
    if is_enabled():
        trace_model_invoke(response, latency_s=latency, model=model_name)
    return response


def timed_stream(model: Any, messages: Any, model_name: str) -> Iterator[Any]:
    """Stream a model response and trace the aggregated result."""
    start = time.perf_counter()
    tracer = langchain_tracer()
    kwargs: dict[str, Any] = {"config": {"callbacks": [tracer]}} if tracer else {}
    chunks: list[Any] = []
    try:
        for chunk in model.stream(messages, **kwargs):
            chunks.append(chunk)
            yield chunk
    except Exception:
        _log.exception("model stream failed")
        raise
    latency = time.perf_counter() - start
    if is_enabled():
        trace_model_stream(chunks, latency_s=latency, model=model_name)
