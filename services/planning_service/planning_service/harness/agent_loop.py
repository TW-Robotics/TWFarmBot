"""Generic multi-turn agent loop for both chat and planner modes.

The model writes Python that calls registered tools. This loop:
- extracts fenced farm scripts from the model output,
- runs them in a restricted interpreter,
- routes every tool call through the approval gate,
- extracts reasoning/thinking,
- emits events (in streaming mode) or returns a result object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Iterator, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .approval_gate import ApprovalGate
from .context_builder import ContextBuilder
from .farm_script import (
    FarmScriptRuntime,
    ScriptResult,
    extract_farm_scripts,
    format_script_feedback,
)
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
    """Multi-turn programmatic tool-calling loop."""

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
        self._model = model
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
        configure_tools = getattr(model, "configure_tools", None)
        if configure_tools is not None:
            configure_tools(
                tool_registry.descriptors(),
                lambda name, args: self._invoke_tool(name, args),
            )

    def run(self, messages: list[dict[str, Any]]) -> AgentTurnResult:
        """Run the loop synchronously and return the final result."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        runtime = self._runtime()
        tool_log: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []
        programs: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        last_response: Any = None
        final_text = ""
        final_thinking: str | None = None

        for _ in range(_MAX_TOOL_TURNS):
            response = timed_invoke(
                self._model, lc_messages, self._model_name, metrics=metrics
            )
            last_response = response
            self._absorb_native_response(
                response,
                tool_log=tool_log,
                proposed=proposed,
                programs=programs,
                trace=trace,
                metrics=metrics,
            )
            text = str(response.content or "")
            scripts = extract_farm_scripts(text, self._registry.by_name())
            if not scripts:
                final_text = text
                final_thinking = self._reasoning.extract(response)
                break

            lc_messages.append(response)
            for source in scripts:
                script_result = runtime.run(source)
                self._absorb_script(
                    script_result, tool_log=tool_log, proposed=proposed, metrics=metrics
                )
                lc_messages.append(HumanMessage(content=format_script_feedback(script_result)))
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
            proposed_actions=proposed,
            programs=programs,
            metrics=metrics,
        )

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Run the loop and yield SSE-style events."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        runtime = self._runtime()
        tool_log: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []
        programs: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []

        for _ in range(_MAX_TOOL_TURNS):
            final_msg = yield from self._stream_turn(lc_messages, metrics=metrics)
            self._absorb_native_response(
                final_msg,
                tool_log=tool_log,
                proposed=proposed,
                programs=programs,
                trace=trace,
                metrics=metrics,
            )
            text = str(final_msg.content or "")
            scripts = extract_farm_scripts(text, self._registry.by_name())
            if not scripts:
                break

            lc_messages.append(final_msg)
            for source in scripts:
                script_result = runtime.run(source)
                events = self._absorb_script(
                    script_result, tool_log=tool_log, proposed=proposed, metrics=metrics
                )
                for event in events:
                    yield event
                lc_messages.append(HumanMessage(content=format_script_feedback(script_result)))

        metrics.total_latency_s = time.perf_counter() - total_start
        yield {
            "type": "meta",
            "tool_calls": tool_log,
            "proposed_actions": proposed,
            "programs": programs,
            "trace": trace,
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
        content_parts: list[str] = []
        additional_kwargs: dict[str, Any] = {}

        for chunk in timed_stream(
            self._model, lc_messages, self._model_name, metrics=metrics
        ):
            kwargs = getattr(chunk, "additional_kwargs", {}) or {}
            additional_kwargs.update(kwargs)
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
                content_parts.append(str(content))
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

        return AIMessage(
            content="".join(content_parts), additional_kwargs=additional_kwargs
        )

    def _absorb_native_response(
        self,
        response: Any,
        *,
        tool_log: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
        programs: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        metrics: Metrics,
    ) -> None:
        """Copy metadata produced by native Responses/PTC adapters."""
        metadata = getattr(response, "additional_kwargs", {}) or {}
        for call in metadata.get("tool_calls", []):
            tool_log.append(call)
            result = call.get("result", {})
            if isinstance(result, dict):
                latency = result.get("_resireg_latency_s")
                if latency:
                    metrics.add_resireg_latency(float(latency))
        proposed.extend(metadata.get("proposed_actions", []))
        programs.extend(metadata.get("programs", []))
        trace.extend(metadata.get("trace", []))

    def plan_request(self, request: str) -> AgentTurnResult:
        """Planner-mode loop: gather introspection, collect action proposals."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.planner_messages(request)
        runtime = self._runtime()
        tool_log: list[dict[str, Any]] = []
        last_response: Any = None
        final_text = ""
        final_thinking: str | None = None

        for _ in range(_MAX_TOOL_TURNS):
            response = timed_invoke(
                self._model, lc_messages, self._model_name, metrics=metrics
            )
            last_response = response
            text = str(response.content or "")
            scripts = extract_farm_scripts(text, self._registry.by_name())
            if not scripts:
                final_text = text
                final_thinking = self._reasoning.extract(response)
                break

            proposed_holder: list[dict[str, Any]] = []
            feedbacks: list[str] = []
            for source in scripts:
                script_result = runtime.run(source)
                self._absorb_script(
                    script_result,
                    tool_log=tool_log,
                    proposed=proposed_holder,
                    metrics=metrics,
                )
                feedbacks.append(format_script_feedback(script_result))

            action_calls = [
                c for c in tool_log if c.get("name") in self._action_tool_names
            ]
            if action_calls:
                final_text = text
                final_thinking = self._reasoning.extract(response)
                break

            lc_messages.append(response)
            lc_messages.append(HumanMessage(content="\n\n".join(feedbacks)))
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

    def _runtime(self) -> FarmScriptRuntime:
        return FarmScriptRuntime(
            self._registry.descriptors(),
            lambda name, args: self._invoke_tool(name, args),
        )

    def _absorb_script(
        self,
        script_result: ScriptResult,
        *,
        tool_log: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
        metrics: Metrics | None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for call in script_result.calls:
            name = call.get("name")
            args = call.get("args", {})
            result = call.get("result")
            tool_log.append({"name": name, "args": args, "result": result})
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
                    "result": result,
                }
            )
            if metrics is not None and isinstance(result, dict):
                resi_latency = result.get("_resireg_latency_s")
                if resi_latency:
                    metrics.add_resireg_latency(float(resi_latency))
        if not script_result.ok:
            events.append(
                {
                    "type": "tool_call",
                    "name": "farm_script",
                    "args": {},
                    "result": {"error": script_result.error},
                }
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
