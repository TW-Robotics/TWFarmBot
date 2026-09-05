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
    reg.register("capture_ndre", lambda a: a)
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
    assert by_name["move"].policy.requires_approval is False
    assert by_name["water"].policy.requires_approval is True
    assert by_name["take_photo"].policy.category == ToolCategory.READ
    assert by_name["take_photo"].policy.requires_approval is False
    assert by_name["capture"].policy.category == ToolCategory.READ
    assert by_name["capture"].policy.requires_approval is False
    assert by_name["capture_ndre"].policy.category == ToolCategory.ANALYZE
    assert by_name["capture_ndre"].policy.requires_approval is False
    assert by_name["e_stop"].policy.category == ToolCategory.ACT
    assert by_name["e_stop"].policy.requires_approval is False


# ───────────────────────────────── ApprovalGate ──────────────────────────────


def test_approval_gate_executes_actions_when_allowed() -> None:
    reg = _make_registry()
    gate = ApprovalGate(reg)
    descriptor = ToolRegistry(reg).by_name()["move"]
    result = gate.resolve(
        descriptor, {"x": 0, "y": 0, "z": 0}, propose_only=False, allow_actions=True
    )
    assert result.status == "ok"


def test_approval_gate_blocks_actions_when_disabled() -> None:
    reg = _make_registry()
    gate = ApprovalGate(reg)
    descriptor = ToolRegistry(reg).by_name()["move"]
    result = gate.resolve(
        descriptor, {"x": 0, "y": 0, "z": 0}, propose_only=False, allow_actions=False
    )
    assert result.status == "error"


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
    max_turns: int = 100,
    max_consecutive_errors: int = 5,
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
        max_turns=max_turns,
        max_consecutive_errors=max_consecutive_errors,
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
    assert result.tool_calls[1]["result"]["status"] == "ok"
    assert result.proposed_actions == []


def test_agent_loop_does_not_run_python_fences() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(["```python\nmove(x=1, y=2, z=0)\n```"])
    loop = _make_loop(fake, reg)
    result = loop.run([{"role": "user", "content": "move"}])
    assert result.tool_calls == []
    assert result.proposed_actions == []
    assert "move(x=1" in result.response


def test_agent_loop_executes_move_immediately() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("move", {"x": 100, "y": 200, "z": 0}),
            "moved",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False)
    result = loop.run([{"role": "user", "content": "move to 100,200"}])
    assert len(result.proposed_actions) == 0
    assert result.tool_calls[0]["result"]["status"] == "ok"
    assert "error" not in result.tool_calls[0]["result"]
    assert result.response == "moved"


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
    loop = _make_loop(fake, reg, propose_only=False)
    result = loop.run([{"role": "user", "content": "visit three points"}])
    moves = [tc for tc in result.tool_calls if tc["name"] == "move"]
    assert len(moves) == 3
    assert len(result.proposed_actions) == 0


# ───────────────────────────── Long-running turns ────────────────────────────


def test_long_run_reasons_over_many_tool_rounds() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    script: list[Any] = [
        _tool_call("take_photo", {}, f"read_{i}") for i in range(6)
    ] + ["finished"]
    fake.set_responses(script)

    loop = _make_loop(fake, reg)
    result = loop.run([{"role": "user", "content": "watch the bed"}])
    assert result.response == "finished"
    assert [tc["name"] for tc in result.tool_calls] == ["take_photo"] * 6
    assert result.stop_reason == "done"


def test_max_turns_stops_endless_tool_loop() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("move", {"x": 1, "y": 2, "z": 0}, "act_1"),
            _tool_call("move", {"x": 1, "y": 2, "z": 0}, "act_2"),
            _tool_call("move", {"x": 1, "y": 2, "z": 0}, "act_3"),
        ]
    )
    loop = _make_loop(fake, reg, max_turns=2)
    result = loop.run([{"role": "user", "content": "keep moving"}])
    assert result.stop_reason == "max_turns"
    assert len(result.tool_calls) == 2


def test_consecutive_errors_stop_the_run() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("no_such_tool", {}, "bad_1"),
            _tool_call("no_such_tool", {}, "bad_2"),
            _tool_call("no_such_tool", {}, "bad_3"),
        ]
    )
    loop = _make_loop(fake, reg, max_consecutive_errors=2)
    result = loop.run([{"role": "user", "content": "do the thing"}])
    assert result.stop_reason == "max_errors"
    assert len(result.tool_calls) == 2


def test_tool_names_are_unique_for_strict_providers() -> None:
    # Action and introspection vocabularies overlap (read_pin). Providers
    # like Gemini reject duplicate function declarations, so the registry
    # must emit each name once.
    reg = ActionRegistry()
    reg.register("move", lambda a: a)
    reg.register("read_pin", lambda a: a)
    state = InMemorySystemStateProvider(position={"x": 0, "y": 0, "z": 0})
    tools = ToolRegistry(reg, state).langchain_tools()
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    assert names.count("read_pin") == 1


def test_silent_model_still_gets_closing_summary() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("no_such_tool", {}, "bad_1"),
            AIMessage(content=""),
        ]
    )
    loop = _make_loop(fake, reg)
    result = loop.run([{"role": "user", "content": "do the thing"}])
    assert result.stop_reason == "done"
    assert result.response
    assert "no_such_tool" in result.response


def test_stream_emits_thinking_then_tool_call_then_meta() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            AIMessage(
                content="<think>checking light</think>looking",
                tool_calls=[
                    {
                        "name": "take_photo",
                        "args": {},
                        "id": "read_1",
                        "type": "tool_call",
                    }
                ],
            ),
            "shot taken",
        ]
    )

    loop = _make_loop(fake, reg)
    events = list(loop.stream([{"role": "user", "content": "look around"}]))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_type.setdefault(str(event["type"]), []).append(event)
    assert "checking light" in [
        str(e.get("content", "")) for e in by_type.get("thinking", [])
    ]
    assert [e["name"] for e in by_type.get("tool_call", [])] == ["take_photo"]
    metas = by_type.get("meta", [])
    assert len(metas) == 1
    assert metas[0]["stop_reason"] == "done"


def test_gemini_block_content_flattens_to_text() -> None:
    # Gemini returns content blocks; internal fields must never leak.
    from planning_service.harness.graph import _last_ai_text

    msg = AIMessage(
        content=[
            {
                "type": "text",
                "text": "Hello! How can I help you",
                "thought_signature": "CnsBjz1rX/9m7==",
            }
        ]
    )
    assert _last_ai_text([msg]) == "Hello! How can I help you"


def test_block_content_run_returns_clean_text() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "hi there",
                        "thought_signature": "s3cr3t",
                    }
                ]
            ),
        ]
    )
    loop = _make_loop(fake, reg)
    result = loop.run([{"role": "user", "content": "hello"}])
    assert result.response == "hi there"
    assert "thought_signature" not in result.response


def test_stream_pauses_water_for_approval() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("water", {"seconds": 2}, "w1"),
            "watered",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False)
    events = list(loop.stream([{"role": "user", "content": "water 2s"}]))
    types = [e["type"] for e in events]
    assert "approval" in types
    assert "meta" not in types
    pending = next(e for e in events if e["type"] == "approval")
    assert pending["pending_approvals"][0]["name"] == "water"
    thread_id = pending["thread_id"]
    resumed = list(
        loop.stream(
            [],
            thread_id=thread_id,
            resume={"approved_ids": [pending["pending_approvals"][0]["id"]]},
        )
    )
    assert any(e.get("name") == "water" for e in resumed if e["type"] == "tool_call")
    assert any(e["type"] == "meta" for e in resumed)


def test_agent_loop_keeps_going_after_get_position_only() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("get_position", {}, "p1"),
            "I have the current pose.",
            _tool_call("move", {"x": 0, "y": 100, "z": 0}, "m1"),
            "moved",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False)
    result = loop.run(
        [{"role": "user", "content": "move in y 100mm then take a photo"}]
    )
    assert [tc["name"] for tc in result.tool_calls] == ["get_position", "move"]
    assert result.response == "moved"


def test_provider_tool_content_attaches_still(tmp_path, monkeypatch) -> None:
    from planning_service.tool_results import provider_tool_content

    jpeg = tmp_path / "shot-rgb.jpg"
    jpeg.write_bytes(b"\xff\xd8" + b"\x00" * 32 + b"\xff\xd9")
    monkeypatch.setattr(
        "planning_service.tool_results._capture_file_uri",
        lambda *_args: f"data:image/jpeg;base64,{jpeg.read_bytes().hex()[:8]}",
    )
    content = provider_tool_content(
        "capture",
        {"status": "ok", "params": {"band": "rgb", "artifact_id": "shot"}},
    )
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"
