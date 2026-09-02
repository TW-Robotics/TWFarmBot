"""Tests for the planning_service harness primitives.

These tests do not need a live LLM or FarmBot; they exercise policy,
approval, reasoning extraction, prompt generation, and the agent loop
with small fake models.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

from planning_service.harness import (
    AgentLoop,
    ApprovalGate,
    ContextBuilder,
    ReasoningController,
    ToolCategory,
    ToolRegistry,
)
from planning_service.introspection import InMemorySystemStateProvider
from twfarmbot_core.actions import ActionRegistry


class _ScriptFake(FakeListChatModel):
    """Fake model that returns a sequence of AIMessage or string responses."""

    _custom_responses: list[Any] | None = None
    _custom_index: int = 0

    def set_responses(self, responses: list[Any]) -> None:
        self._custom_responses = responses
        self._custom_index = 0

    def invoke(self, *_args: Any, **_kwargs: Any) -> AIMessage:
        custom_responses = getattr(self, "_custom_responses", None)
        if not custom_responses:
            return super().invoke(*_args, **_kwargs)
        response = custom_responses[self._custom_index]
        self._custom_index = min(self._custom_index + 1, len(custom_responses) - 1)
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=str(response))

    def stream(self, *_args: Any, **_kwargs: Any):
        msg = self.invoke(*_args, **_kwargs)
        content = str(msg.content or "")
        tool_calls = list(getattr(msg, "tool_calls", None) or [])
        if tool_calls:
            chunks = []
            for index, call in enumerate(tool_calls):
                name = call["name"] if isinstance(call, dict) else call.name
                args = call["args"] if isinstance(call, dict) else call.args
                call_id = (
                    call.get("id")
                    if isinstance(call, dict)
                    else getattr(call, "id", None)
                ) or str(index)
                chunks.append(
                    {
                        "name": name,
                        "args": json.dumps(args or {}),
                        "id": str(call_id),
                        "index": index,
                        "type": "tool_call_chunk",
                    }
                )
            yield AIMessageChunk(content=content, tool_call_chunks=chunks)  # type: ignore[arg-type]
            return
        if not content:
            yield AIMessageChunk(content="")
            return
        mid = max(1, len(content) // 2)
        yield AIMessageChunk(content=content[:mid])
        yield AIMessageChunk(content=content[mid:])


def _tool_call(
    name: str, args: dict[str, Any] | None = None, call_id: str = "call_1"
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _make_registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg.register("move", lambda a: a)
    reg.register("water", lambda a: a)
    reg.register("take_photo", lambda a: a)
    reg.register("capture", lambda a: a)
    reg.register("e_stop", lambda a: a)
    return reg


# ───────────────────────────── ToolPolicy / Registry ─────────────────────────


def test_tool_registry_contains_all_action_kinds() -> None:
    reg = _make_registry()
    tool_registry = ToolRegistry(reg)
    names = {d.name for d in tool_registry.descriptors()}
    assert names >= {"move", "water", "take_photo", "capture", "e_stop"}


def test_action_policies_are_categorized() -> None:
    reg = _make_registry()
    tool_registry = ToolRegistry(reg)
    by_name = tool_registry.by_name()
    assert by_name["move"].policy.category == ToolCategory.ACT
    assert by_name["move"].policy.requires_approval is True
    assert by_name["take_photo"].policy.category == ToolCategory.READ
    assert by_name["take_photo"].policy.requires_approval is False
    assert by_name["capture"].policy.category == ToolCategory.READ
    assert by_name["capture"].policy.requires_approval is False
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
    assert "Available tools" in prompt
    assert "take_photo" in prompt
    assert "move" in prompt


# ────────────────────────────────── AgentLoop ────────────────────────────────


def _make_loop(
    model: _ScriptFake,
    reg: ActionRegistry,
    propose_only: bool = True,
    system_state: InMemorySystemStateProvider | None = None,
) -> AgentLoop:
    tool_registry = ToolRegistry(reg, system_state)
    approval_gate = ApprovalGate(reg)
    builder = ContextBuilder(tool_registry)
    return AgentLoop(
        model=model,
        tool_registry=tool_registry,
        approval_gate=approval_gate,
        context_builder=builder,
        propose_only=propose_only,
        allow_actions=True,
    )


def test_agent_loop_runs_json_read_and_physical_tool_calls() -> None:
    reg = _make_registry()
    state = InMemorySystemStateProvider(position={"x": 11, "y": 22, "z": 33})
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("get_position", {}, "read_1"),
            _tool_call("move", {"x": 100, "y": 200, "z": 0}, "act_1"),
            "done",
        ]
    )

    loop = _make_loop(fake, reg, system_state=state)
    result = loop.run([{"role": "user", "content": "where am I, then move"}])
    assert result.response == "done"
    names = [tc["name"] for tc in result.tool_calls]
    assert names == ["get_position", "move"]
    assert result.tool_calls[0]["result"]["x"] == 11
    assert result.tool_calls[1]["result"]["status"] == "proposed"
    assert result.proposed_actions == [
        {"kind": "move", "params": {"x": 100, "y": 200, "z": 0}}
    ]


def test_agent_loop_does_not_run_python_fences() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(["```python\nmove(x=1, y=2, z=0)\n```"])
    loop = _make_loop(fake, reg)
    result = loop.run([{"role": "user", "content": "move"}])
    assert result.tool_calls == []
    assert result.proposed_actions == []
    assert "move(x=1" in result.response


def test_agent_loop_proposes_move_without_executing() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("move", {"x": 100, "y": 200, "z": 0}),
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
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("take_photo"),
            "photo taken",
        ]
    )
    loop = _make_loop(fake, reg)
    events = list(loop.stream([{"role": "user", "content": "take a photo"}]))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "meta" in types
    assert "delta" in types
    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert tool_events[0]["name"] == "take_photo"
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["programs"] == []


def test_agent_loop_json_calls_capture() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("capture", {"band": "rgb"}, "cap_1"),
            "captured",
        ]
    )
    loop = _make_loop(fake, reg)
    result = loop.run([{"role": "user", "content": "capture rgb"}])
    assert result.tool_calls[0]["name"] == "capture"
    assert result.tool_calls[0]["args"] == {"band": "rgb"}
    assert result.response == "captured"


def test_agent_loop_runs_parallel_json_tool_calls() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "move",
                        "args": {"x": x, "y": 0, "z": 0},
                        "id": f"m{x}",
                        "type": "tool_call",
                    }
                    for x in (0, 100, 200)
                ],
            ),
            "queued three moves",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=True)
    result = loop.run([{"role": "user", "content": "visit three points"}])
    moves = [tc for tc in result.tool_calls if tc["name"] == "move"]
    assert len(moves) == 3
    assert len(result.proposed_actions) == 3
