"""Agent loop facade over the LangGraph reason-then-act runtime.

``AgentLoop`` keeps its public shape (``run`` / ``stream`` / ``plan_request``
plus ``AgentTurnResult`` and the SSE event dicts); the turn plumbing lives in
:mod:`planning_service.harness.graph`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from ..tool_results import append_result_images, compact_tool_result
from .approval_gate import ApprovalGate
from .context_builder import ContextBuilder
from .graph import (
    STOP_DONE,
    RunDeps,
    _bind_model_tools,
    _last_ai_text,
    build_graph,
)
from .metrics import Metrics
from .reasoning_controller import ReasoningController
from .tool_policy import ToolCategory
from .tool_registry import ToolRegistry

# Default cap on model turns per call (API-compatible alias for the old const).
_MAX_TOOL_TURNS = 100
_FALLBACK_TEXT = "I ran too many tool calls without finishing. Please try again."


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of one agent run."""

    response: str
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    programs: list[dict[str, Any]] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    stop_reason: str = STOP_DONE


class AgentLoop:
    """Multi-turn JSON function/tool-calling loop (LangGraph-backed)."""

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
        max_turns: int = _MAX_TOOL_TURNS,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._tool_registry = tool_registry
        self._approval_gate = approval_gate
        self._context_builder = context_builder
        self._reasoning = reasoning or ReasoningController()
        self._model_name = model_name
        self._propose_only = propose_only
        self._allow_actions = allow_actions
        self._include_reasoning = include_reasoning
        self._max_turns = max_turns
        self._max_errors = max_consecutive_errors
        self._action_tool_names = {
            d.name
            for d in tool_registry.descriptors()
            if d.policy.category == ToolCategory.ACT
        }
        self._model = _bind_model_tools(model, tool_registry.langchain_tools())

    def _deps(
        self,
        metrics: Metrics,
        *,
        stop_after_propose: bool,
        streaming: bool,
    ) -> RunDeps:
        return RunDeps(
            model=self._model,
            registry=self._tool_registry,
            gate=self._approval_gate,
            reasoning=self._reasoning,
            model_name=self._model_name,
            propose_only=self._propose_only,
            allow_actions=self._allow_actions,
            max_turns=self._max_turns,
            max_errors=self._max_errors,
            stop_after_propose=stop_after_propose,
            streaming=streaming,
            metrics=metrics,
            action_names=set(self._action_tool_names),
        )

    @staticmethod
    def _config(max_turns: int) -> dict[str, Any]:
        return {
            "recursion_limit": max_turns * 2 + 8,
            "configurable": {"thread_id": uuid.uuid4().hex},
        }

    @staticmethod
    def _initial_state(messages: list[Any]) -> dict[str, Any]:
        return {
            "messages": list(messages),
            "turn": 0,
            "tool_calls_made": 0,
            "consecutive_errors": 0,
            "stop_reason": "",
            "tool_log": [],
            "proposed": [],
            "saw_action_call": False,
        }

    def run(self, messages: list[dict[str, Any]]) -> AgentTurnResult:
        """Run the loop synchronously and return the final result."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        graph = build_graph(
            self._deps(metrics, stop_after_propose=False, streaming=False)
        )
        final = graph.invoke(
            self._initial_state(lc_messages), self._config(self._max_turns)
        )
        state_messages = list(final.get("messages", []))
        tool_log = list(final.get("tool_log", []))
        proposed = list(final.get("proposed", []))
        stop_reason = str(final.get("stop_reason") or STOP_DONE)
        final_text = _last_ai_text(state_messages)
        final_thinking = self._reasoning.extract(_last_ai_message(state_messages))
        if not final_text and stop_reason != STOP_DONE:
            final_text = _FALLBACK_TEXT
        final_text = self._reasoning.strip_from_text(final_text)
        final_text = append_result_images(final_text, tool_log)
        metrics.total_latency_s = time.perf_counter() - total_start
        return AgentTurnResult(
            response=final_text,
            thinking=final_thinking,
            tool_calls=tool_log,
            proposed_actions=proposed,
            metrics=metrics,
            stop_reason=stop_reason,
        )

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Run the loop and yield SSE-style events."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.chat_messages(
            messages, include_reasoning=self._include_reasoning
        )
        graph = build_graph(
            self._deps(metrics, stop_after_propose=False, streaming=True)
        )
        config = self._config(self._max_turns)
        for mode, payload in graph.stream(
            self._initial_state(lc_messages), config, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                yield dict(payload)
            elif mode == "updates":
                tools_update = payload.get("tools")
                if isinstance(tools_update, dict):
                    for record in tools_update.get("tool_log", []):
                        yield {
                            "type": "tool_call",
                            "name": record.get("name"),
                            "args": record.get("args"),
                            "result": compact_tool_result(record.get("result")),
                        }
        state = graph.get_state(config).values
        state_messages = list(state.get("messages", []))
        tool_log = list(state.get("tool_log", []))
        proposed = list(state.get("proposed", []))
        stop_reason = str(state.get("stop_reason") or STOP_DONE)
        text = _last_ai_text(state_messages)
        imaged = append_result_images(text, tool_log)
        extra = imaged[len(text) :] if imaged.startswith(text) else ""
        if extra:
            yield {"type": "delta", "content": extra}
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
            "stop_reason": stop_reason,
        }

    def plan_request(self, request: str) -> AgentTurnResult:
        """Planner-mode loop: gather introspection, collect action proposals."""
        total_start = time.perf_counter()
        metrics = Metrics()
        lc_messages = self._context_builder.planner_messages(request)
        graph = build_graph(
            self._deps(metrics, stop_after_propose=True, streaming=False)
        )
        final = graph.invoke(
            self._initial_state(lc_messages), self._config(self._max_turns)
        )
        state_messages = list(final.get("messages", []))
        tool_log = list(final.get("tool_log", []))
        stop_reason = str(final.get("stop_reason") or STOP_DONE)
        final_text = _last_ai_text(state_messages)
        final_thinking = self._reasoning.extract(_last_ai_message(state_messages))
        if not final_text and stop_reason != STOP_DONE:
            final_text = _FALLBACK_TEXT
        final_text = self._reasoning.strip_from_text(final_text)
        metrics.total_latency_s = time.perf_counter() - total_start
        return AgentTurnResult(
            response=final_text,
            thinking=final_thinking,
            tool_calls=tool_log,
            proposed_actions=[],
            metrics=metrics,
            stop_reason=stop_reason,
        )


def _last_ai_message(messages: list[Any]) -> Any:
    """Return the most recent AI message (empty AIMessage if none)."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return AIMessage(content="")
