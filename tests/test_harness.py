"""Tests for the planning_service harness primitives.

These tests do not need a live LLM or FarmBot; they exercise policy,
approval, reasoning extraction, prompt generation, and the agent loop
with small fake models.
"""

from __future__ import annotations

from typing import Any, Sequence

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.tools import BaseTool

from planning_service.harness import (
    AgentLoop,
    ApprovalGate,
    ContextBuilder,
    ReasoningController,
    ToolCategory,
    ToolDescriptor,
    ToolPolicy,
    ToolRegistry,
)
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action


class _ToolAwareFake(FakeListChatModel):
    """Fake model that supports ``bind_tools`` and can emit tool_calls."""

    _custom_responses: list[Any]
    _custom_index: int = 0

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        **kwargs: Any,
    ) -> "_ToolAwareFake":
        return self

    def set_responses(self, responses: list[Any]) -> None:
        """Provide a sequence of AIMessage or string responses."""
        self._custom_responses = responses
        self._custom_index = 0

    def invoke(self, *_args: Any, **_kwargs: Any) -> AIMessage:
        if not getattr(self, "_custom_responses", None):
            return super().invoke(*_args, **_kwargs)
        response = self._custom_responses[self._custom_index]
        self._custom_index = min(
            self._custom_index + 1, len(self._custom_responses) - 1
        )
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=str(response))

    def stream(self, *_args: Any, **_kwargs: Any):
        msg = self.invoke(*_args, **_kwargs)
        if getattr(msg, "tool_calls", None):
            yield AIMessageChunk(
                content=msg.content or "",
                tool_call_chunks=[
                    ToolCallChunk(
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        args=json.dumps(tc.get("args", {})),
                        index=i,
                    )
                    for i, tc in enumerate(msg.tool_calls)
                ],
            )
        else:
            for word in str(msg.content or "").split():
                yield AIMessageChunk(content=word + " ")


def _make_registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg.register("move", lambda a: a)
    reg.register("water", lambda a: a)
    reg.register("take_photo", lambda a: a)
    reg.register("e_stop", lambda a: a)
    return reg


# ───────────────────────────── ToolPolicy / Registry ─────────────────────────


def test_tool_registry_contains_all_action_kinds() -> None:
    reg = _make_registry()
    tool_registry = ToolRegistry(reg)
    names = {d.name for d in tool_registry.descriptors()}
    assert names >= {"move", "water", "take_photo", "e_stop"}


def test_action_policies_are_categorized() -> None:
    reg = _make_registry()
    tool_registry = ToolRegistry(reg)
    by_name = tool_registry.by_name()
    assert by_name["move"].policy.category == ToolCategory.ACT
    assert by_name["move"].policy.requires_approval is True
    assert by_name["take_photo"].policy.category == ToolCategory.READ
    assert by_name["take_photo"].policy.requires_approval is False
    assert by_name["e_stop"].policy.category == ToolCategory.ACT
    assert by_name["e_stop"].policy.requires_approval is False


# ───────────────────────────────── ApprovalGate ──────────────────────────────


def test_approval_gate_proposes_dangerous_actions_in_chat() -> None:
    reg = _make_registry()
    gate = ApprovalGate(reg)
    descriptor = ToolRegistry(reg).by_name()["move"]
    result = gate.resolve(
        descriptor, {"x": 0, "y": 0, "z": 0}, propose_only=True, allow_actions=True
    )
    assert result.status == "proposed"


def test_approval_gate_executes_safe_read_actions() -> None:
    reg = _make_registry()
    gate = ApprovalGate(reg)
    descriptor = ToolRegistry(reg).by_name()["take_photo"]
    result = gate.resolve(descriptor, {}, propose_only=True, allow_actions=True)
    assert result.status == "ok"


def test_approval_gate_never_executes_in_planning_mode() -> None:
    reg = _make_registry()
    gate = ApprovalGate(reg, planning_mode=True)
    descriptor = ToolRegistry(reg).by_name()["e_stop"]
    result = gate.resolve(descriptor, {}, propose_only=False, allow_actions=True)
    assert result.status == "proposed"


# ───────────────────────────── ReasoningController ───────────────────────────


def test_reasoning_controller_extracts_think_tags() -> None:
    rc = ReasoningController()
    thinking = rc.extract(
        type("Msg", (), {"content": "before <think>step 1</think> after"})()
    )
    assert thinking == "step 1"
    assert rc.strip_from_text("before <think>x</think> after") == "before  after"


def test_reasoning_controller_extracts_provider_reasoning() -> None:
    rc = ReasoningController()
    msg = type(
        "Msg",
        (),
        {
            "content": "answer",
            "response_metadata": {},
            "additional_kwargs": {"reasoning": "step by step"},
        },
    )()
    assert rc.extract(msg) == "step by step"


def test_reasoning_controller_splits_text_events() -> None:
    rc = ReasoningController()
    events = list(rc.split_text("hi <think>reason</think> bye"))
    assert events == [
        {"type": "delta", "content": "hi "},
        {"type": "thinking", "content": "reason"},
        {"type": "delta", "content": " bye"},
    ]


# ─────────────────────────────── ContextBuilder ──────────────────────────────


def test_context_builder_lists_tools_in_prompt() -> None:
    reg = _make_registry()
    tool_registry = ToolRegistry(reg)
    builder = ContextBuilder(tool_registry)
    prompt = builder.chat_system_prompt()
    assert "Read-only tools" in prompt
    assert "Execution tools" in prompt
    assert "take_photo" in prompt
    assert "move" in prompt


# ────────────────────────────────── AgentLoop ────────────────────────────────


def _make_loop(
    model: _ToolAwareFake,
    reg: ActionRegistry,
    propose_only: bool = True,
) -> AgentLoop:
    tool_registry = ToolRegistry(reg)
    approval_gate = ApprovalGate(reg)
    builder = ContextBuilder(tool_registry)
    bound = model.bind_tools(tool_registry.langchain_tools())
    return AgentLoop(
        model=bound,
        tool_registry=tool_registry,
        approval_gate=approval_gate,
        context_builder=builder,
        propose_only=propose_only,
        allow_actions=True,
    )


def test_agent_loop_runs_multiple_introspection_turns() -> None:
    reg = _make_registry()
    # First turn calls get_position; second turn answers.
    fake = _ToolAwareFake(responses=["unused"])
    fake.set_responses(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_position", "id": "1", "args": {}}],
            ),
            "done",
        ]
    )

    loop = _make_loop(fake, reg)
    result = loop.run([{"role": "user", "content": "where am I"}])
    assert result.response == "done"
    assert any(tc["name"] == "get_position" for tc in result.tool_calls)


def test_agent_loop_proposes_move_without_executing() -> None:
    reg = _make_registry()
    fake = _ToolAwareFake(responses=["unused"])
    fake.set_responses(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "move", "id": "1", "args": {"x": 100, "y": 200, "z": 0}}
                ],
            ),
            "proposed",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=True)
    result = loop.run([{"role": "user", "content": "move to 100,200"}])
    assert len(result.proposed_actions) == 1
    assert result.proposed_actions[0]["kind"] == "move"
    assert result.response == "proposed"


def test_agent_loop_streams_tool_call_and_delta_events() -> None:
    reg = _make_registry()
    fake = _ToolAwareFake(responses=["unused"])
    fake.set_responses(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "take_photo", "id": "1", "args": {}}],
            ),
            "photo taken",
        ]
    )
    loop = _make_loop(fake, reg)
    events = list(loop.stream([{"role": "user", "content": "take a photo"}]))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "meta" in types
    assert "delta" in types
