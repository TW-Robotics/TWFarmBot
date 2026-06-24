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
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from .approval_gate import ApprovalGate
from .context_builder import ContextBuilder
from .reasoning_controller import ReasoningController
from .tool_policy import ToolCategory, ToolDescriptor
from .tool_registry import ToolRegistry


def _llm_friendly_result(result: Any) -> Any:
    """Replace large binary payloads with placeholders for the LLM context."""
    if isinstance(result, dict) and "image_url" in result:
        out = dict(result)
        out["image_url"] = "[image data shown to user in chat]"
        return out
    return result


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
        model: BaseChatModel,
        tool_registry: ToolRegistry,
        approval_gate: ApprovalGate,
        context_builder: ContextBuilder,
        reasoning: ReasoningController | None = None,
        *,
        propose_only: bool = False,
        allow_actions: bool = True,
        max_iterations: int = 5,
        include_reasoning: bool = False,
    ) -> None:
        self._model = model
        self._registry = tool_registry
        self._approval_gate = approval_gate
        self._context_builder = context_builder
        self._reasoning = reasoning or ReasoningController()
        self._propose_only = propose_only
        self._allow_actions = allow_actions
        self._max_iterations = max_iterations
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

        for _ in range(self._max_iterations):
            response = self._model.invoke(lc_messages)
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
        """Run the loop and yield SSE-style events."""
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        tool_map = self._tool_map()
        tool_log: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []
        last_response: Any = None

        for _ in range(self._max_iterations):
            response = self._model.invoke(lc_messages)
            last_response = response
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
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

        tool_turn_thinking = self._reasoning.extract(last_response)
        if tool_turn_thinking:
            yield {"type": "thinking", "content": tool_turn_thinking}

        buffer = ""
        streamed_reasoning: list[str] = []
        streamed_reasoning_emitted = bool(tool_turn_thinking)
        for chunk in self._model.stream(lc_messages):
            for event in self._reasoning.stream_chunks(
                chunk,
                accumulated_reasoning=streamed_reasoning,
                emitted=streamed_reasoning_emitted,
            ):
                streamed_reasoning_emitted = True
                yield event

            content = getattr(chunk, "content", None)
            if not content:
                continue
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

        if buffer:
            yield {"type": "delta", "content": buffer}

    def plan_request(self, request: str, *, max_iterations: int = 3) -> AgentTurnResult:
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

        for _ in range(max_iterations):
            response = self._model.invoke(lc_messages)
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
        try:
            return tool.invoke(args)
        except Exception as err:  # noqa: BLE001
            return {"error": f"{type(err).__name__}: {err}"}
