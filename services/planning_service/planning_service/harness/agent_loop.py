"""Generic multi-turn agent loop for both chat and planner modes.

The loop owns:
- binding the model to the unified tool set,
- invoking the model and detecting tool calls,
- resolving each tool call through the approval gate,
- extracting reasoning/thinking,
- emitting events (in streaming mode) or returning a result object.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Iterator, Sequence

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .approval_gate import ApprovalGate
from .context_builder import ContextBuilder
from .reasoning_controller import ReasoningController
from .tool_policy import ToolCategory, ToolDescriptor
from .tool_registry import ToolRegistry
from .tracing import is_enabled, timed_invoke, timed_stream, trace_tool_call

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

# Absolute safety backstop to prevent a misbehaving model from looping forever.
# This is not a user-facing iteration budget; normal flows stop as soon as the
# model returns text instead of tool calls.
_MAX_TOOL_TURNS = 100


def _llm_friendly_result(result: Any) -> Any:
    """Replace large binary payloads with placeholders for the LLM context."""
    if isinstance(result, dict) and "image_url" in result:
        out = dict(result)
        out["image_url"] = "[image data shown to user in chat]"
        return out
    return result


def _normalize_tool_args(value: Any) -> Any:
    """Recursively convert Pydantic model instances to plain JSON-serializable dicts.

    LangChain validates tool inputs against Pydantic schemas, so nested args
    (e.g. ``Waypoint`` inside ``move_path``) arrive as model instances. The
    rest of the harness expects plain dicts.
    """
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


class AgentLoop:
    """Multi-turn tool-calling loop."""

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

    def run(self, messages: list[dict[str, Any]]) -> AgentTurnResult:
        """Run the loop synchronously and return the final result."""
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        tool_map = self._tool_map()
        tool_log: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []
        last_response: Any = None
        final_text = ""
        final_thinking: str | None = None

        for _ in range(_MAX_TOOL_TURNS):
            response = timed_invoke(self._model, lc_messages, self._model_name)
            last_response = response
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                final_text = str(response.content or "")
                final_thinking = self._reasoning.extract(response)
                break

            lc_messages.append(response)
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args", {})
                tool_call_id = call.get("id", "")
                result = self._invoke_tool(name, args, tool_map)
                tool_log.append({"name": name, "args": args, "result": result})
                if isinstance(result, dict) and result.get("status") == "proposed":
                    proposed.append(
                        {
                            "kind": result.get("kind", name),
                            "params": result.get("params", args),
                        }
                    )
                lc_messages.append(
                    ToolMessage(
                        content=json.dumps(_llm_friendly_result(result)),
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                )
        else:
            final_text = str(getattr(last_response, "content", "") or "")
            final_thinking = self._reasoning.extract(last_response)
            if not final_text:
                final_text = (
                    "I ran too many tool calls without finishing. Please try again."
                )

        final_text = self._reasoning.strip_from_text(final_text)
        return AgentTurnResult(
            response=final_text,
            thinking=final_thinking,
            tool_calls=tool_log,
            proposed_actions=proposed,
        )

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Run the loop and yield SSE-style events.

        Every model turn is streamed, including tool-decision turns, so the
        user sees reasoning and content as it is produced instead of waiting
        for a complete response.
        """
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        tool_map = self._tool_map()
        tool_log: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []

        for _ in range(_MAX_TOOL_TURNS):
            final_msg, tool_calls = yield from self._stream_turn(lc_messages)
            if not tool_calls:
                # The final answer has already been streamed; nothing left to do.
                break

            lc_messages.append(final_msg)
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args", {})
                tool_call_id = call.get("id", "")
                result = self._invoke_tool(name, args, tool_map)
                tool_log.append({"name": name, "args": args, "result": result})
                if isinstance(result, dict) and result.get("status") == "proposed":
                    proposed.append(
                        {
                            "kind": result.get("kind", name),
                            "params": result.get("params", args),
                        }
                    )
                yield {
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                    "result": result,
                }
                lc_messages.append(
                    ToolMessage(
                        content=json.dumps(_llm_friendly_result(result)),
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                )

        yield {"type": "meta", "tool_calls": tool_log, "proposed_actions": proposed}

    def _stream_turn(
        self, lc_messages: Sequence[BaseMessage]
    ) -> Generator[dict[str, Any], None, tuple[AIMessage, list[dict[str, Any]]]]:
        """Stream one model turn and return the final message + tool calls.

        Yields content/thinking deltas as they arrive. When the turn ends with
        tool calls, those calls are returned so the caller can execute them.
        """
        buffer = ""
        streamed_reasoning: list[str] = []
        streamed_reasoning_emitted = False
        tool_accum: dict[int, dict[str, Any]] = {}
        content_parts: list[str] = []

        for chunk in timed_stream(self._model, lc_messages, self._model_name):
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
                        # Any trailing text in the buffer after the think block
                        # will be re-processed in the next iteration.

            for tc in getattr(chunk, "tool_call_chunks", []) or []:
                if isinstance(tc, dict):
                    idx = int(tc.get("index", 0) or 0)
                    entry = tool_accum.setdefault(idx, {"args": ""})
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    if tc.get("name"):
                        entry["name"] = tc["name"]
                    if tc.get("args"):
                        entry["args"] += tc["args"]
                else:
                    idx = int(getattr(tc, "index", 0) or 0)
                    entry = tool_accum.setdefault(idx, {"args": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    if getattr(tc, "name", None):
                        entry["name"] = tc.name
                    if getattr(tc, "args", None):
                        entry["args"] += tc.args

        if buffer:
            yield {"type": "delta", "content": buffer}

        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_accum):
            entry = tool_accum[idx]
            args_str = entry.get("args", "")
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {
                    "id": entry.get("id", ""),
                    "name": entry.get("name", ""),
                    "args": args,
                }
            )

        final_content = "".join(content_parts)
        final_msg = AIMessage(
            content=final_content,
            tool_calls=tool_calls,
            additional_kwargs={},
        )
        return final_msg, tool_calls

    def plan_request(self, request: str) -> AgentTurnResult:
        """Planner-mode loop: gather introspection, collect action proposals.

        Action tools are resolved through the approval gate; callers should
        construct the loop with ``propose_only=True`` (or ``allow_actions=False``)
        so physical actions are not executed during planning.
        """
        lc_messages = self._context_builder.planner_messages(request)
        tool_map = self._tool_map()
        tool_log: list[dict[str, Any]] = []
        last_response: Any = None
        final_text = ""
        final_thinking: str | None = None

        for _ in range(_MAX_TOOL_TURNS):
            response = timed_invoke(self._model, lc_messages, self._model_name)
            last_response = response
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                final_text = str(response.content or "")
                final_thinking = self._reasoning.extract(response)
                break

            # In planning mode, once the model emits action tool calls we
            # have the plan and should stop rather than asking it to continue.
            action_calls = [
                c for c in tool_calls if c.get("name") in self._action_tool_names
            ]
            if action_calls:
                for call in tool_calls:
                    name = call.get("name")
                    args = call.get("args", {})
                    result = self._invoke_tool(name, args, tool_map)
                    tool_log.append({"name": name, "args": args, "result": result})
                final_text = str(response.content or "")
                final_thinking = self._reasoning.extract(response)
                break

            lc_messages.append(response)
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args", {})
                tool_call_id = call.get("id", "")
                result = self._invoke_tool(name, args, tool_map)
                tool_log.append({"name": name, "args": args, "result": result})
                lc_messages.append(
                    ToolMessage(
                        content=json.dumps(_llm_friendly_result(result)),
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                )
        else:
            final_text = str(getattr(last_response, "content", "") or "")
            final_thinking = self._reasoning.extract(last_response)
            if not final_text:
                final_text = (
                    "I ran too many tool calls without finishing. Please try again."
                )

        final_text = self._reasoning.strip_from_text(final_text)
        return AgentTurnResult(
            response=final_text,
            thinking=final_thinking,
            tool_calls=tool_log,
            proposed_actions=[],
        )

    def _tool_map(self) -> dict[str, BaseTool]:
        return {
            t.name: t
            for t in self._registry.langchain_tools(resolve=self._resolve_tool)
        }

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
        self, name: str | None, args: dict[str, Any], tool_map: dict[str, BaseTool]
    ) -> dict[str, Any]:
        if name is None:
            return {"error": "tool call missing name"}
        tool = tool_map.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        args = _normalize_tool_args(args)
        start = time.perf_counter()
        try:
            result = tool.invoke(args)
        except Exception as err:  # noqa: BLE001
            result = {"error": f"{type(err).__name__}: {err}"}
        result = _normalize_tool_args(result)
        latency = time.perf_counter() - start
        if is_enabled():
            trace_tool_call(name, args, _llm_friendly_result(result), latency_s=latency)
        return result
