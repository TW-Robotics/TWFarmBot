"""LangGraph runtime for the TWFarmBot agent loop.

One reason-then-act graph replaces the three hand-rolled turn loops:

- ``agent`` node: invoke (or stream) the model with bound tools.
- ``tools`` node: execute calls through ``ApprovalGate`` + ``safety_service``.
- Conditional edges route between them until the model stops calling tools
  or a stop condition fires.

Tool calls, reasoning extraction, result compaction, metrics, and tracing
are reused unchanged — only the turn plumbing moved onto LangGraph.
"""

from __future__ import annotations

import json
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel

from ..tool_results import compact_tool_result, provider_tool_content, _sse_images
from .approval_gate import ApprovalGate
from .metrics import Metrics
from .reasoning_controller import ReasoningController, content_text
from .tool_policy import ToolDescriptor
from .tool_registry import ToolRegistry
from .tracing import is_enabled, timed_invoke, timed_stream, trace_tool_call

from .cancel import (
    STOP_CANCELLED,
    drain_followup,
    is_cancelled,
    peek_followup,
)

STOP_DONE = "done"
STOP_MAX_TURNS = "max_turns"
STOP_MAX_ERRORS = "max_errors"
JOG_APPROVAL_MM = 50.0
CHECKPOINTER = MemorySaver()


class HarnessState(TypedDict, total=False):
    """Graph state. Lists use reducers so concurrent node writes merge."""

    messages: Annotated[list[Any], add_messages]
    turn: int
    tool_calls_made: Annotated[int, operator.add]
    consecutive_errors: int
    stop_reason: str
    tool_log: Annotated[list[dict[str, Any]], operator.add]
    proposed: Annotated[list[dict[str, Any]], operator.add]
    saw_action_call: bool


def _llm_friendly_result(result: Any) -> Any:
    """Replace large binary payloads with placeholders for the LLM context."""
    if isinstance(result, dict) and "image_url" in result:
        out = dict(result)
        out["image_url"] = "[image data shown to user in chat]"
        return out
    return result


def _normalize_tool_args(value: Any) -> Any:
    """Recursively convert Pydantic model instances to plain dicts."""
    if isinstance(value, BaseModel):
        return _normalize_tool_args(value.model_dump())
    if isinstance(value, list):
        return [_normalize_tool_args(item) for item in value]
    if isinstance(value, dict):
        return {k: _normalize_tool_args(v) for k, v in value.items()}
    return value


def _tool_calls_from_message(response: Any) -> list[dict[str, Any]]:
    """Read LangChain JSON function/tool calls off an AI message."""
    raw = getattr(response, "tool_calls", None) or []
    if not raw:
        raw = [
            {
                "name": chunk.get("name")
                if isinstance(chunk, dict)
                else getattr(chunk, "name", None),
                "args": chunk.get("args")
                if isinstance(chunk, dict)
                else getattr(chunk, "args", None),
                "id": chunk.get("id")
                if isinstance(chunk, dict)
                else getattr(chunk, "id", None),
            }
            for chunk in (getattr(response, "tool_call_chunks", None) or [])
        ]
    out: list[dict[str, Any]] = []
    for index, call in enumerate(raw):
        if isinstance(call, dict):
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{index}"
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "args", None) or {}
            call_id = getattr(call, "id", None) or f"call_{index}"
        if not name:
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({"name": str(name), "args": dict(args), "id": str(call_id)})
    return out


def _ai_from_stream(assembled: Any, buffer: str) -> AIMessage:
    """Keep streamed tool_call_chunks, not just the final tool_calls list."""
    if assembled is None:
        return AIMessage(content=buffer)
    calls = _tool_calls_from_message(assembled)
    content = content_text(getattr(assembled, "content", "")) or buffer
    if isinstance(assembled, AIMessage) and getattr(assembled, "tool_calls", None):
        return assembled
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": call["name"],
                "args": call["args"],
                "id": call["id"],
                "type": "tool_call",
            }
            for call in calls
        ],
        additional_kwargs=dict(getattr(assembled, "additional_kwargs", None) or {}),
    )


def _last_ai_message(messages: list[Any]) -> Any | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _tool_succeeded(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if result.get("status") == "error" or result.get("error"):
        return False
    return True


def _bind_model_tools(model: Runnable, tools: list[Any]) -> Runnable:
    """Attach JSON tool schemas when the backend supports bind_tools."""
    bind = getattr(model, "bind_tools", None)
    if bind is None or not tools:
        return model
    try:
        return bind(tools)
    except (NotImplementedError, TypeError, ValueError):
        return model


def _last_ai_text(messages: list[Any]) -> str:
    """Return the most recent non-empty AI message content."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = content_text(message.content)
            if text:
                return text
    return ""


@dataclass
class RunDeps:
    """Per-run collaborators shared by every node (kept out of the state)."""

    model: Any
    registry: ToolRegistry
    gate: ApprovalGate
    reasoning: ReasoningController
    model_name: str = "unknown"
    propose_only: bool = False
    allow_actions: bool = True
    max_turns: int = 100
    max_errors: int = 5
    stop_after_propose: bool = False
    streaming: bool = False
    metrics: Metrics = field(default_factory=Metrics)
    action_names: set[str] = field(default_factory=set)
    skip_approval: bool = False


def _resolve_tool(
    deps: RunDeps, descriptor: ToolDescriptor, params: dict[str, Any]
) -> dict[str, Any]:
    if descriptor.is_introspection:
        if descriptor.execute is not None:
            return descriptor.execute(params)
        return {"error": f"introspection tool {descriptor.name!r} has no executor"}
    result = deps.gate.resolve(
        descriptor,
        params,
        propose_only=deps.propose_only,
        allow_actions=deps.allow_actions,
    )
    payload: dict[str, Any] = {
        "status": result.status,
        "kind": result.kind,
        "params": result.params,
    }
    if result.note:
        payload["note"] = result.note
    if result.error:
        payload["error"] = result.error
    return payload


def _invoke_tool(
    deps: RunDeps, name: str | None, args: dict[str, Any]
) -> dict[str, Any]:
    if name is None:
        return {"error": "tool call missing name"}
    descriptor = deps.registry.by_name().get(name)
    if descriptor is None:
        return {"error": f"unknown tool {name!r}"}
    args = _normalize_tool_args(args)
    start = time.perf_counter()
    try:
        result = _resolve_tool(deps, descriptor, args)
    except Exception as err:  # noqa: BLE001
        result = {"error": f"{type(err).__name__}: {err}"}
    result = _normalize_tool_args(result)
    latency = time.perf_counter() - start
    if is_enabled():
        trace_tool_call(name, args, _llm_friendly_result(result), latency_s=latency)
    resi_latency = (
        result.get("_resireg_latency_s") if isinstance(result, dict) else None
    )
    if resi_latency:
        deps.metrics.add_resireg_latency(float(resi_latency))
    return result


def _needs_approval(
    deps: RunDeps, name: str, args: dict[str, Any] | None = None
) -> bool:
    """Interrupt chat for risky acts; jogs and reads run immediately."""
    if deps.skip_approval or not deps.streaming:
        return False
    descriptor = deps.registry.by_name().get(name)
    if descriptor is None:
        return False
    if descriptor.policy.requires_approval:
        return True
    args = args or {}
    if name == "move_axis":
        try:
            return abs(float(args.get("distance") or 0)) > JOG_APPROVAL_MM
        except (TypeError, ValueError):
            return True
    if name == "move":
        return _move_travel_mm(args) > JOG_APPROVAL_MM
    return False


def _move_travel_mm(args: dict[str, Any]) -> float:
    try:
        target = [float(args[axis]) for axis in ("x", "y", "z")]
    except (KeyError, TypeError, ValueError):
        return JOG_APPROVAL_MM + 1
    current = _current_xyz()
    if current is None:
        return JOG_APPROVAL_MM + 1
    return sum((target[i] - current[i]) ** 2 for i in range(3)) ** 0.5


def _current_xyz() -> list[float] | None:
    try:
        from watering_service.backends import farmbot

        xyz = farmbot.backend.get_xyz()
    except Exception:  # noqa: BLE001
        return None
    try:
        return [float(xyz["x"]), float(xyz["y"]), float(xyz["z"])]
    except (KeyError, TypeError, ValueError):
        return None


def interrupt_payload(payload: Any) -> dict[str, Any] | None:
    """Read pending approvals out of a LangGraph stream updates chunk."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("__interrupt__")
    if not raw:
        return None
    first = raw[0] if isinstance(raw, (list, tuple)) else raw
    value = getattr(first, "value", first)
    if isinstance(value, dict) and value.get("pending_approvals"):
        return value
    return None


def build_graph(deps: RunDeps):

    def agent_node(state: HarnessState) -> dict[str, Any]:
        """Invoke the model once and record the turn."""
        if is_cancelled() or state.get("turn", 0) >= deps.max_turns:
            return {
                "stop_reason": STOP_CANCELLED if is_cancelled() else STOP_MAX_TURNS
            }
        response = timed_invoke(
            deps.model,
            state["messages"],
            deps.model_name,
            metrics=deps.metrics,
        )
        update: dict[str, Any] = {
            "messages": [response],
            "turn": state.get("turn", 0) + 1,
        }
        if _tool_calls_from_message(response):
            update["stop_reason"] = ""
        else:
            update["stop_reason"] = STOP_DONE
        return update

    def agent_stream_node(state: HarnessState) -> dict[str, Any]:
        """Streaming twin of ``agent_node``: forwards deltas as custom events."""
        if is_cancelled() or state.get("turn", 0) >= deps.max_turns:
            return {
                "stop_reason": STOP_CANCELLED if is_cancelled() else STOP_MAX_TURNS
            }
        writer = get_stream_writer()
        buffer = ""
        streamed_reasoning: list[str] = []
        reasoning_emitted = False
        assembled: Any = None
        for chunk in timed_stream(
            deps.model,
            state["messages"],
            deps.model_name,
            metrics=deps.metrics,
        ):
            assembled = chunk if assembled is None else assembled + chunk
            kwargs = getattr(chunk, "additional_kwargs", {}) or {}
            stream_event = kwargs.get("stream_event")
            if isinstance(stream_event, dict) and stream_event.get("type"):
                writer(stream_event)
                continue
            for event in deps.reasoning.stream_chunks(
                chunk,
                accumulated_reasoning=streamed_reasoning,
                emitted=reasoning_emitted,
            ):
                reasoning_emitted = True
                writer(event)
            content = getattr(chunk, "content", None)
            if content:
                buffer += content_text(content)
                events, buffer = deps.reasoning.split_stream(buffer)
                for event in events:
                    writer(event)
        if buffer:
            writer({"type": "delta", "content": buffer})
        response = _ai_from_stream(assembled, buffer)
        update = {
            "messages": [response],
            "turn": state.get("turn", 0) + 1,
        }
        if _tool_calls_from_message(response):
            update["stop_reason"] = ""
        else:
            update["stop_reason"] = STOP_DONE
        return update

    def tools_node(state: HarnessState) -> dict[str, Any]:
        """Execute the latest tool calls and record compacted results."""
        last = (state.get("messages") or [None])[-1]
        calls = (
            _tool_calls_from_message(last) if isinstance(last, AIMessage) else []
        )
        pending = [
            c for c in calls if _needs_approval(deps, str(c["name"]), dict(c["args"]))
        ]
        pending_ids = {c["id"] for c in pending}
        approved_ids: set[str] = set()
        if pending:
            decision = interrupt(
                {
                    "pending_approvals": [
                        {"name": c["name"], "args": c["args"], "id": c["id"]}
                        for c in pending
                    ]
                }
            )
            if isinstance(decision, dict):
                approved_ids = set(decision.get("approved_ids", []))
        records: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []
        saw_action = False
        # Stream tool lifecycle events so the UI can show a row the moment a
        # call starts (often minutes for scans) instead of after the fact.
        writer = get_stream_writer() if deps.streaming else None
        for call in calls:
            name = str(call["name"])
            args = dict(call["args"])
            if name in deps.action_names:
                saw_action = True
            if peek_followup():
                result = {
                    "status": "skipped",
                    "kind": name,
                    "params": args,
                    "note": "skipped: new user instruction",
                }
            elif is_cancelled():
                result = {
                    "status": "error",
                    "kind": name,
                    "params": args,
                    "error": "stopped by operator",
                }
            elif call["id"] in pending_ids and call["id"] not in approved_ids:
                result = {
                    "status": "error",
                    "kind": name,
                    "params": args,
                    "error": "rejected by approver",
                }
            else:
                if writer:
                    writer(
                        {
                            "type": "tool_start",
                            "id": str(call["id"]),
                            "name": name,
                            "args": args,
                        }
                    )
                result = _invoke_tool(deps, name, args)
            if writer:
                writer(
                    {
                        "type": "tool_call",
                        "id": str(call["id"]),
                        "name": name,
                        "args": args,
                        "result": compact_tool_result(result),
                        "images": _sse_images(name, result),
                    }
                )
            records.append({"name": name, "args": args, "result": result})
            if isinstance(result, dict) and result.get("status") == "proposed":
                proposed.append(
                    {
                        "kind": result.get("kind", name),
                        "params": result.get("params", args),
                    }
                )
            tool_messages.append(
                ToolMessage(
                    content=provider_tool_content(name, result),
                    tool_call_id=str(call.get("id") or name),
                    name=name,
                )
            )
        ok = any(_tool_succeeded(r["result"]) for r in records) if records else True
        errors = 0 if ok else state.get("consecutive_errors", 0) + 1
        outgoing: list[Any] = list(tool_messages)
        followup = None if is_cancelled() else drain_followup()
        if is_cancelled():
            drain_followup()
        elif followup:
            outgoing.append(HumanMessage(content=followup))
            if writer:
                writer({"type": "followup", "text": followup})
        update = {
            "messages": outgoing,
            "tool_log": records,
            "proposed": proposed,
            "tool_calls_made": len(records),
            "consecutive_errors": errors,
            "saw_action_call": saw_action,
        }
        if is_cancelled():
            update["stop_reason"] = STOP_CANCELLED
        elif errors >= deps.max_errors:
            update["stop_reason"] = STOP_MAX_ERRORS
        return update

    def route_after_agent(state: HarnessState) -> Literal["tools", "__end__"]:
        """Run tools when the latest message is a tool-calling AI turn."""
        if state.get("stop_reason"):
            return "__end__"
        last = (state.get("messages") or [None])[-1]
        if isinstance(last, AIMessage) and _tool_calls_from_message(last):
            return "tools"
        return "__end__"

    def route_after_tools(state: HarnessState) -> Literal["agent", "__end__"]:
        """Return to the agent unless stopped or a proposal landed."""
        if state.get("stop_reason"):
            return "__end__"
        if deps.stop_after_propose and state.get("saw_action_call"):
            return "__end__"
        return "agent"

    graph = StateGraph(HarnessState)
    graph.add_node("agent", agent_stream_node if deps.streaming else agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "__end__": END},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "__end__": END},
    )
    return graph.compile(checkpointer=CHECKPOINTER)
