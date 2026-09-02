"""Generic multi-turn agent loop for both chat and planner modes.

The model emits JSON tool/function calls. This loop:
- binds registered tools on the chat-completions client,
- executes each call through the approval gate,
- extracts reasoning/thinking,
- emits events (in streaming mode) or returns a result object.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Iterator, Sequence

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from ..tool_results import append_result_images, compact_tool_result
from .approval_gate import ApprovalGate
from .context_builder import ContextBuilder
from .metrics import Metrics
from .reasoning_controller import ReasoningController
from .tool_policy import ToolCategory, ToolDescriptor
from .tool_registry import ToolRegistry
from .tracing import is_enabled, timed_invoke, timed_stream, trace_tool_call

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

# Absolute safety backstop to prevent a misbehaving model from looping forever.
_MAX_TOOL_TURNS = 100


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


def _bind_model_tools(model: Runnable, tools: list[Any]) -> Runnable:
    """Attach JSON tool schemas when the backend supports bind_tools."""
    bind = getattr(model, "bind_tools", None)
    if bind is None or not tools:
        return model
    try:
        return bind(tools)
    except (NotImplementedError, TypeError, ValueError):
        return model


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of one agent turn."""

    response: str
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    programs: list[dict[str, Any]] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)


class AgentLoop:
    """Multi-turn JSON function/tool-calling loop."""

    def __init__(
        self,
        model: Runnable,
        tool_registry: ToolRegistry,
        approval_gate: ApprovalGate,
        context_builder: ContextBuilder,
        reasoning: ReasoningController | None = None,
        *,
        model_name: str = "unknown",
        propose_only: bool = False,
        allow_actions: bool = True,
        include_reasoning: bool = False,
    ) -> None:
        self._registry = tool_registry
        self._approval_gate = approval_gate
        self._context_builder = context_builder
        self._reasoning = reasoning or ReasoningController()
        self._model_name = model_name
        self._propose_only = propose_only
        self._allow_actions = allow_actions
        self._include_reasoning = include_reasoning
        self._action_tool_names = {
            d.name
            for d in tool_registry.descriptors()
            if d.policy.category == ToolCategory.ACT
        }
        self._model = _bind_model_tools(model, tool_registry.langchain_tools())

    def run(self, messages: list[dict[str, Any]]) -> AgentTurnResult:
        """Run the loop synchronously and return the final result."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        tool_log: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []
        last_response: Any = None
        final_text = ""
        final_thinking: str | None = None

        for _ in range(_MAX_TOOL_TURNS):
            response = timed_invoke(
                self._model, lc_messages, self._model_name, metrics=metrics
            )
            last_response = response
            calls = _tool_calls_from_message(response)
            if not calls:
                final_text = str(response.content or "")
                final_thinking = self._reasoning.extract(response)
                break

            lc_messages.append(response)
            self._apply_json_tool_calls(
                calls,
                lc_messages,
                tool_log=tool_log,
                proposed=proposed,
                metrics=metrics,
            )
        else:
            final_text = str(getattr(last_response, "content", "") or "")
            final_thinking = self._reasoning.extract(last_response)
            if not final_text:
                final_text = (
                    "I ran too many tool calls without finishing. Please try again."
                )

        final_text = self._reasoning.strip_from_text(final_text)
        final_text = append_result_images(final_text, tool_log)
        metrics.total_latency_s = time.perf_counter() - total_start
        return AgentTurnResult(
            response=final_text,
            thinking=final_thinking,
            tool_calls=tool_log,
            proposed_actions=proposed,
            metrics=metrics,
        )

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Run the loop and yield SSE-style events."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        tool_log: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []

        for _ in range(_MAX_TOOL_TURNS):
            final_msg = yield from self._stream_turn(lc_messages, metrics=metrics)
            calls = _tool_calls_from_message(final_msg)
            if not calls:
                text = str(final_msg.content or "")
                imaged = append_result_images(text, tool_log)
                extra = imaged[len(text) :] if imaged.startswith(text) else ""
                if extra:
                    yield {"type": "delta", "content": extra}
                break

            lc_messages.append(final_msg)
            events = self._apply_json_tool_calls(
                calls,
                lc_messages,
                tool_log=tool_log,
                proposed=proposed,
                metrics=metrics,
            )
            for event in events:
                yield event

        metrics.total_latency_s = time.perf_counter() - total_start
        yield {
            "type": "meta",
            "tool_calls": [
                {
                    "name": call.get("name"),
                    "args": call.get("args"),
                    "result": compact_tool_result(call.get("result")),
                }
                for call in tool_log
            ],
            "proposed_actions": proposed,
            "programs": [],
            "metrics": metrics.to_dict(),
        }

    def _stream_turn(
        self,
        lc_messages: Sequence[BaseMessage],
        *,
        metrics: Metrics,
    ) -> Generator[dict[str, Any], None, AIMessage]:
        """Stream one model turn and return the assembled message."""
        buffer = ""
        streamed_reasoning: list[str] = []
        streamed_reasoning_emitted = False
        assembled: Any = None

        for chunk in timed_stream(
            self._model, lc_messages, self._model_name, metrics=metrics
        ):
            assembled = chunk if assembled is None else assembled + chunk
            kwargs = getattr(chunk, "additional_kwargs", {}) or {}
            stream_event = kwargs.get("stream_event")
            if isinstance(stream_event, dict) and stream_event.get("type"):
                yield stream_event
                continue

            for event in self._reasoning.stream_chunks(
                chunk,
                accumulated_reasoning=streamed_reasoning,
                emitted=streamed_reasoning_emitted,
            ):
                streamed_reasoning_emitted = True
                yield event

            content = getattr(chunk, "content", None)
            if content:
                buffer += str(content)
                for event in self._reasoning.split_text(buffer):
                    if event["type"] == "delta":
                        if event["content"]:
                            yield event
                        buffer = ""
                    elif event["type"] == "thinking":
                        yield event

        if buffer:
            yield {"type": "delta", "content": buffer}

        if assembled is None:
            return AIMessage(content="")
        if isinstance(assembled, AIMessage):
            return assembled
        return AIMessage(
            content=str(getattr(assembled, "content", "") or ""),
            tool_calls=list(getattr(assembled, "tool_calls", None) or []),
            additional_kwargs=dict(getattr(assembled, "additional_kwargs", None) or {}),
        )

    def plan_request(self, request: str) -> AgentTurnResult:
        """Planner-mode loop: gather introspection, collect action proposals."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.planner_messages(request)
        tool_log: list[dict[str, Any]] = []
        last_response: Any = None
        final_text = ""
        final_thinking: str | None = None

        for _ in range(_MAX_TOOL_TURNS):
            response = timed_invoke(
                self._model, lc_messages, self._model_name, metrics=metrics
            )
            last_response = response
            calls = _tool_calls_from_message(response)
            if not calls:
                final_text = str(response.content or "")
                final_thinking = self._reasoning.extract(response)
                break

            proposed_holder: list[dict[str, Any]] = []
            lc_messages.append(response)
            self._apply_json_tool_calls(
                calls,
                lc_messages,
                tool_log=tool_log,
                proposed=proposed_holder,
                metrics=metrics,
            )
            if any(c.get("name") in self._action_tool_names for c in calls):
                final_text = str(response.content or "")
                final_thinking = self._reasoning.extract(response)
                break
        else:
            final_text = str(getattr(last_response, "content", "") or "")
            final_thinking = self._reasoning.extract(last_response)
            if not final_text:
                final_text = (
                    "I ran too many tool calls without finishing. Please try again."
                )

        final_text = self._reasoning.strip_from_text(final_text)
        metrics.total_latency_s = time.perf_counter() - total_start
        return AgentTurnResult(
            response=final_text,
            thinking=final_thinking,
            tool_calls=tool_log,
            proposed_actions=[],
            metrics=metrics,
        )

    def _apply_json_tool_calls(
        self,
        calls: Sequence[dict[str, Any]],
        lc_messages: list[Any],
        *,
        tool_log: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
        metrics: Metrics | None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for call in calls:
            name = call.get("name")
            args = call.get("args", {})
            result = self._invoke_tool(name, args, metrics=metrics)
            recorded = {"name": name, "args": args, "result": result}
            tool_log.append(recorded)
            if isinstance(result, dict) and result.get("status") == "proposed":
                proposed.append(
                    {
                        "kind": result.get("kind", name),
                        "params": result.get("params", args),
                    }
                )
            events.append(
                {
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                    "result": compact_tool_result(result),
                }
            )
            lc_messages.append(
                ToolMessage(
                    content=json.dumps(compact_tool_result(result), default=str),
                    tool_call_id=str(call.get("id") or name or "tool"),
                    name=str(name or ""),
                )
            )
        return events

    def _resolve_tool(
        self, descriptor: ToolDescriptor, params: dict[str, Any]
    ) -> dict[str, Any]:
        if descriptor.is_introspection:
            if descriptor.execute is not None:
                return descriptor.execute(params)
            return {"error": f"introspection tool {descriptor.name!r} has no executor"}
        result = self._approval_gate.resolve(
            descriptor,
            params,
            propose_only=self._propose_only,
            allow_actions=self._allow_actions,
        )
        return {
            "status": result.status,
            "kind": result.kind,
            "params": result.params,
            "note": result.note,
            "error": result.error,
        }

    def _invoke_tool(
        self,
        name: str | None,
        args: dict[str, Any],
        *,
        metrics: Metrics | None = None,
    ) -> dict[str, Any]:
        if name is None:
            return {"error": "tool call missing name"}
        descriptor = self._registry.by_name().get(name)
        if descriptor is None:
            return {"error": f"unknown tool {name!r}"}
        args = _normalize_tool_args(args)
        start = time.perf_counter()
        try:
            result = self._resolve_tool(descriptor, args)
        except Exception as err:  # noqa: BLE001
            result = {"error": f"{type(err).__name__}: {err}"}
        result = _normalize_tool_args(result)
        latency = time.perf_counter() - start
        if is_enabled():
            trace_tool_call(name, args, _llm_friendly_result(result), latency_s=latency)
        if metrics is not None:
            resi_latency = (
                result.get("_resireg_latency_s") if isinstance(result, dict) else None
            )
            if resi_latency:
                metrics.add_resireg_latency(float(resi_latency))
        return result
