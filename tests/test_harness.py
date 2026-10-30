"""Tests for the planning_service harness primitives.

These tests do not need a live LLM or FarmBot; they exercise policy,
approval, reasoning extraction, prompt generation, and the agent loop
with small fake models.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fakes import AIMessage, ScriptModel
from planning_service.harness import (
    AgentLoop,
    ApprovalGate,
    ContextBuilder,
    ToolCategory,
    ToolRegistry,
)
from planning_service.harness.tool_registry import CORE_TOOLS
from planning_service.introspection import InMemorySystemStateProvider
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

_Call = tuple[str, dict[str, Any], str]
_ScriptFake = ScriptModel


def _tool_call(name: str, args: dict[str, Any] | None = None, call_id: str = "call_1") -> list[_Call]:
    return [(name, args or {}, call_id)]


def _make_registry(**handlers: Any) -> ActionRegistry:
    defaults: dict[str, Any] = {
        "move": lambda a: a,
        "water": lambda a: a,
        "take_photo": lambda a: a,
        "capture": lambda a: a,
        "capture_ndre": lambda a: a,
        "e_stop": lambda a: a,
    }
    defaults.update(handlers)
    reg = ActionRegistry()
    for kind, handler in defaults.items():
        reg.register(kind, handler)
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


# ─────────────────────────────── ContextBuilder ──────────────────────────────


def test_context_builder_prompt_is_slim_and_points_to_tools() -> None:
    prompt = ContextBuilder(ToolRegistry(_make_registry())).chat_system_prompt()
    assert "Available tools" not in prompt  # schemas are sent with the request
    assert "search_tools" in prompt and "get_help" in prompt
    assert "get_position" in prompt  # verify moves
    assert len(prompt) < 4500



# ────────────────────────────────── AgentLoop ────────────────────────────────


def _make_loop(
    model: _ScriptFake,
    reg: ActionRegistry,
    propose_only: bool = True,
    system_state: InMemorySystemStateProvider | None = None,
    max_turns: int = 100,
    max_consecutive_errors: int = 5,
    skip_approval: bool = False,
    core_tools: frozenset[str] | None = None,
    force_think: bool = False,
) -> AgentLoop:
    tool_registry = ToolRegistry(reg, system_state)
    approval_gate = ApprovalGate(reg)
    builder = ContextBuilder(tool_registry)
    return AgentLoop(
        model=model,
        core_tools=core_tools if core_tools is not None else frozenset(tool_registry.by_name()),
        force_think=force_think,
        tool_registry=tool_registry,
        approval_gate=approval_gate,
        context_builder=builder,
        propose_only=propose_only,
        allow_actions=True,
        skip_approval=skip_approval,
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


def test_agent_loop_streams_tool_start_before_result() -> None:
    """tool_start fires when execution begins; tool_call lands with the result."""
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("take_photo", {}, "photo_1"),
            "photo taken",
        ]
    )
    loop = _make_loop(fake, reg)
    events = list(loop.stream([{"role": "user", "content": "take a photo"}]))
    starts = [e for e in events if e["type"] == "tool_start"]
    calls = [e for e in events if e["type"] == "tool_call"]
    assert [e["id"] for e in starts] == ["photo_1"]
    assert starts[0]["name"] == "take_photo"
    assert "result" not in starts[0]
    assert [e["id"] for e in calls] == ["photo_1"]
    assert calls[0]["result"]["status"] == "ok"
    assert events.index(starts[0]) < events.index(calls[0])


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
    def broken(_action: Any) -> Any:
        raise RuntimeError("jammed")

    fake = _ScriptFake(responses=["unused"])
    fake.set_responses([_tool_call("move", {"x": 1, "y": 2, "z": 0}, f"bad_{i}") for i in range(3)])
    loop = _make_loop(fake, _make_registry(move=broken), max_consecutive_errors=2)
    result = loop.run([{"role": "user", "content": "do the thing"}])
    assert result.stop_reason == "max_errors"
    assert len(result.tool_calls) == 2


def test_unknown_tool_is_retried_then_stops() -> None:
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses([_tool_call("no_such_tool", {}, f"bad_{i}") for i in range(5)])
    result = _make_loop(fake, _make_registry()).run([{"role": "user", "content": "x"}])
    assert result.stop_reason == "max_errors"


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
    def broken(_action: Any) -> Any:
        raise RuntimeError("jammed")

    fake = _ScriptFake(responses=["unused"])
    fake.set_responses([_tool_call("move", {"x": 1, "y": 2, "z": 0}, "bad_1"), ""])
    loop = _make_loop(fake, _make_registry(move=broken))
    result = loop.run([{"role": "user", "content": "do the thing"}])
    assert result.stop_reason == "done"
    assert "move failed" in result.response


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
    assert "approval" not in [e["type"] for e in resumed]


def test_stream_skips_approval_when_requested() -> None:
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("water", {"seconds": 2}, "w1"),
            "watered",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False, skip_approval=True)
    events = list(loop.stream([{"role": "user", "content": "water 2s"}]))
    assert "approval" not in [e["type"] for e in events]
    assert any(e.get("name") == "water" for e in events if e["type"] == "tool_call")
    assert any(e["type"] == "meta" for e in events)


def test_stream_includes_weave_url_when_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Call:
        ui_url = "https://wandb.ai/org/proj/r/call/abc"

    monkeypatch.setattr(
        "planning_service.harness.loop.start_agent_call",
        lambda **_: _Call(),
    )
    monkeypatch.setattr(
        "planning_service.harness.loop.call_url",
        lambda call: getattr(call, "ui_url", None),
    )
    monkeypatch.setattr(
        "planning_service.harness.loop.finish_agent_call",
        lambda *_, **__: None,
    )
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(["hello"])
    loop = _make_loop(fake, _make_registry(), skip_approval=True)
    events = list(loop.stream([{"role": "user", "content": "hi"}]))
    thread = next(e for e in events if e["type"] == "thread")
    meta = next(e for e in events if e["type"] == "meta")
    assert thread["weave_url"] == _Call.ui_url
    assert meta["weave_url"] == _Call.ui_url


def test_stream_weave_output_includes_tools_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Call:
        ui_url = "https://wandb.ai/org/proj/r/call/abc"

    monkeypatch.setattr(
        "planning_service.harness.loop.start_agent_call",
        lambda **_: _Call(),
    )
    monkeypatch.setattr(
        "planning_service.harness.loop.call_url",
        lambda call: getattr(call, "ui_url", None),
    )
    monkeypatch.setattr(
        "planning_service.harness.loop.finish_agent_call",
        lambda _call, output=None, exception=None: captured.update({"output": output}),
    )
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("take_photo", {}, "p1"),
            "Here is a photo.",
        ]
    )
    loop = _make_loop(fake, _make_registry(), skip_approval=True)
    list(loop.stream([{"role": "user", "content": "where am I"}]))
    output = captured["output"]
    assert output["response"] == "Here is a photo."
    assert output["tools"][0]["name"] == "take_photo"


def test_tracing_helpers_noop_when_disabled() -> None:
    from planning_service.harness.tracing import (
        call_url,
        finish_agent_call,
        start_agent_call,
    )

    assert start_agent_call(thread_id="t", prompt="hi") is None
    assert call_url(None) is None
    finish_agent_call(None)


def test_stream_pauses_stepped_moves_for_approval() -> None:
    """A 50 mm photo grid is not a jog — the batch must pause."""
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
                    for x in (0, 50, 100, 150, 200)
                ],
            ),
            "moved",
        ]
    )
    loop = _make_loop(fake, _make_registry(), propose_only=False)
    events = list(loop.stream([{"role": "user", "content": "photo every 50 mm"}]))
    pending = next(e for e in events if e["type"] == "approval")
    names = [item["name"] for item in pending["pending_approvals"]]
    assert names == ["move"] * 5
    assert "tool_call" not in [e["type"] for e in events]


def test_stream_set_goal_with_moves_pauses_the_moves() -> None:
    state = InMemorySystemStateProvider()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "set_goal",
                        "args": {
                            "objective": "photos every 50 mm",
                            "remaining": ["x=50", "x=100"],
                        },
                        "id": "g1",
                        "type": "tool_call",
                    },
                    {
                        "name": "move",
                        "args": {"x": 50, "y": 0, "z": 0},
                        "id": "m1",
                        "type": "tool_call",
                    },
                    {
                        "name": "move",
                        "args": {"x": 100, "y": 0, "z": 0},
                        "id": "m2",
                        "type": "tool_call",
                    },
                ],
            ),
            "ok",
        ]
    )
    loop = _make_loop(fake, _make_registry(), system_state=state, propose_only=False)
    events = list(loop.stream([{"role": "user", "content": "grid"}]))
    pending = next(e for e in events if e["type"] == "approval")
    names = [item["name"] for item in pending["pending_approvals"]]
    assert names == ["set_goal", "move", "move"]
    assert "tool_call" not in [e["type"] for e in events]


def test_stream_scan_ndre_emits_images() -> None:
    reg = _make_registry()
    reg.register(
        "scan_ndre",
        lambda _action: Action(
            kind="scan_ndre",
            params={
                "samples": [
                    {"y": 0.0, "ndre_preview": "/captures/a/ndre"},
                    {"y": 50.0, "ndre_preview": "/captures/b/ndre"},
                ],
                "count": 2,
            },
        ),
    )
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call(
                "scan_ndre",
                {"axis": "y", "end_mm": 50, "step_mm": 50, "start_mm": 0},
                "scan_1",
            ),
            "done",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False)
    events = list(loop.stream([{"role": "user", "content": "scan y"}]))
    pending = next(e for e in events if e["type"] == "approval")
    resumed = list(
        loop.stream(
            [],
            thread_id=pending["thread_id"],
            resume={"approved_ids": [pending["pending_approvals"][0]["id"]]},
        )
    )
    tool = next(
        e for e in resumed if e["type"] == "tool_call" and e.get("name") == "scan_ndre"
    )
    assert tool["images"] == [
        {"label": "NDRE 1 (0 mm)", "url": "/captures/a/ndre"},
        {"label": "NDRE 2 (50 mm)", "url": "/captures/b/ndre"},
    ]
    meta = next(e for e in resumed if e["type"] == "meta")
    scan = next(tc for tc in meta["tool_calls"] if tc["name"] == "scan_ndre")
    assert scan["images"] == tool["images"]


def test_agent_loop_runs_tools_then_model_again() -> None:
    """After a tool result, the graph returns to the model for another turn."""
    reg = _make_registry()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call("get_position", {}, "p1"),
            _tool_call("move", {"x": 0, "y": 100, "z": 0}, "m1"),
            "moved",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False, system_state=InMemorySystemStateProvider())
    result = loop.run(
        [{"role": "user", "content": "move in y 100mm then take a photo"}]
    )
    assert [tc["name"] for tc in result.tool_calls] == ["get_position", "move"]
    assert result.response == "moved"




def test_agent_hides_resi_except_segment() -> None:
    tools = ToolRegistry(_make_registry(), InMemorySystemStateProvider())
    names = set(tools.by_name())
    assert "segment_image" in names
    assert "get_garden" in names
    assert "find_locations" in names
    assert "analyze_image" not in names
    assert "visualize_image_features" not in names
    assert "estimate_traversability" not in names




def _batch_tool_calls(*calls: _Call) -> list[_Call]:
    return list(calls)


def test_followup_mailbox_rejects_without_active_run() -> None:
    from planning_service.harness.cancel import (
        FollowupRejected,
        begin_run,
        cancel_run,
        drain_followup,
        end_run,
        enqueue_followup,
        peek_followup,
    )

    end_run("t1")
    with pytest.raises(FollowupRejected):
        enqueue_followup("t1", "hello")
    begin_run("t1")
    enqueue_followup("t1", "hello")
    enqueue_followup("t1", "again")
    assert peek_followup("t1")
    assert drain_followup("t1") == "hello again"
    assert drain_followup("t1") is None
    enqueue_followup("t1", "keep")
    cancel_run("t1")
    with pytest.raises(FollowupRejected):
        enqueue_followup("t1", "nope")
    assert drain_followup("t1") is None
    end_run("t1")


# Motion tools run one at a time in order; read tools may overlap.
def test_stream_followup_skips_remaining_tools() -> None:
    ran: list[str] = []

    def photo(action: Any) -> Any:
        from planning_service.harness.cancel import enqueue_followup

        ran.append("photo")
        enqueue_followup(None, "do not capture")
        return action

    def capture(action: Any) -> Any:
        ran.append("capture")
        return action

    reg = _make_registry(e_stop=photo, move=capture)
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _batch_tool_calls(
                ("e_stop", {}, "p1"),
                ("move", {"x": 0, "y": 0, "z": 0}, "c1"),
            ),
            "got the follow-up",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False, skip_approval=True)
    events = list(loop.stream([{"role": "user", "content": "photo then capture"}]))
    assert ran == ["photo"]
    results = {e["name"]: e["result"] for e in events if e["type"] == "tool_call"}
    assert results["e_stop"]["status"] == "ok"
    assert results["move"]["status"] == "skipped"
    follow = next(e for e in events if e["type"] == "followup")
    assert follow["text"] == "do not capture"
    text = "".join(e.get("content") or "" for e in events if e["type"] == "delta")
    assert "got the follow-up" in text


def test_stream_followup_before_tools_skips_batch() -> None:
    ran: list[str] = []
    reg = _make_registry(
        e_stop=lambda action: ran.append("photo") or action,
        move=lambda action: ran.append("capture") or action,
    )
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _batch_tool_calls(
                ("e_stop", {}, "p1"),
                ("move", {"x": 0, "y": 0, "z": 0}, "c1"),
            ),
            "ready",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False, skip_approval=True)
    thread: dict[str, str] = {}
    from planning_service.harness.cancel import enqueue_followup

    fake.on_call = lambda i: i == 0 and enqueue_followup(thread["id"], "just say ready")
    events = []
    for e in loop.stream([{"role": "user", "content": "photo and capture"}]):
        thread.setdefault("id", e.get("thread_id", ""))
        events.append(e)
    assert ran == []
    skipped = [e for e in events if e["type"] == "tool_call"]
    assert {e["result"]["status"] for e in skipped} == {"skipped"}
    assert any(e.get("text") == "just say ready" for e in events if e["type"] == "followup")
    text = "".join(e.get("content") or "" for e in events if e["type"] == "delta")
    assert "ready" in text


def test_cancel_discards_followup_and_stops() -> None:
    ran: list[str] = []

    def photo(action: Any) -> Any:
        from planning_service.harness.cancel import cancel_run, enqueue_followup

        ran.append("photo")
        enqueue_followup(None, "keep going")
        cancel_run()
        return action

    def capture(action: Any) -> Any:
        ran.append("capture")
        return action

    reg = _make_registry(e_stop=photo, move=capture)
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _batch_tool_calls(
                ("e_stop", {}, "p1"),
                ("move", {"x": 0, "y": 0, "z": 0}, "c1"),
            ),
            "should not run",
        ]
    )
    loop = _make_loop(fake, reg, propose_only=False, skip_approval=True)
    events = list(loop.stream([{"role": "user", "content": "photo then capture"}]))
    assert ran == ["photo"]
    assert not any(e["type"] == "followup" for e in events)
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["stop_reason"] == "cancelled"




def test_record_tool_fact_stores_water_and_ndre_observations() -> None:
    from planning_service.harness.facts import record_tool_fact
    from planning_service.memory import MemoryStore

    store = MemoryStore(":memory:")
    record_tool_fact("water", {}, {"status": "ok"}, xyz=[10.0, 20.0, 0.0], store=store)
    record_tool_fact(
        "capture_ndre",
        {},
        {
            "status": "ok",
            "params": {
                "interpretation": {"action_hint": "Canopy looks healthy"},
                "ndre": {"mean": 0.31},
                "gantry_mm": {"x": 11, "y": 22, "z": 0},
            },
        },
        store=store,
    )
    record_tool_fact("water", {}, {"status": "proposed"}, xyz=[1, 2, 0], store=store)
    recents = store.list_recent(limit=8)
    texts = " ".join(m["content"] for m in recents)
    assert "Pump ran" in texts and "~60s" in texts
    assert "Canopy looks healthy" in texts and "mean=0.31" in texts
    assert all("auto" in (m.get("tags") or []) for m in recents)


def test_agent_loop_set_goal_then_continues() -> None:
    state = InMemorySystemStateProvider(position={"x": 1, "y": 2, "z": 3})
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call(
                "set_goal",
                {
                    "objective": "NDRE then decide",
                    "remaining": ["y=200", "y=300"],
                },
                "g1",
            ),
            _tool_call("get_position", {}, "p1"),
            "survey next",
        ]
    )
    loop = _make_loop(fake, _make_registry(), system_state=state, skip_approval=True)
    result = loop.run([{"role": "user", "content": "survey the bed"}])
    assert result.tool_calls[0]["name"] == "set_goal"
    assert result.tool_calls[0]["result"]["goal"]["remaining"] == ["y=200", "y=300"]
    assert result.tool_calls[1]["name"] == "get_position"
    assert result.response == "survey next"


def test_agent_loop_streams_goal_like_approvals() -> None:
    state = InMemorySystemStateProvider()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call(
                "set_goal",
                {"objective": "survey the bed", "remaining": ["y=200", "y=300"]},
                "g1",
            ),
            "ok",
        ]
    )
    loop = _make_loop(fake, _make_registry(), system_state=state, skip_approval=True)
    events = list(loop.stream([{"role": "user", "content": "survey"}]))
    goals = [e for e in events if e["type"] == "goal"]
    assert goals
    assert goals[0]["goal"]["objective"] == "survey the bed"
    assert goals[0]["goal"]["remaining"] == ["y=200", "y=300"]
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["goal"]["remaining"] == ["y=200", "y=300"]


def test_stream_pauses_new_goal_for_approval() -> None:
    state = InMemorySystemStateProvider()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call(
                "set_goal",
                {"objective": "survey", "remaining": ["y=200", "y=300"]},
                "g1",
            ),
            "surveying",
        ]
    )
    loop = _make_loop(fake, _make_registry(), system_state=state)
    events = list(loop.stream([{"role": "user", "content": "survey"}]))
    pending = next(e for e in events if e["type"] == "approval")
    assert pending["pending_approvals"][0]["name"] == "set_goal"
    resumed = list(
        loop.stream(
            [],
            thread_id=pending["thread_id"],
            resume={"approved_ids": [pending["pending_approvals"][0]["id"]]},
        )
    )
    assert any(e["type"] == "goal" for e in resumed)
    assert any(e.get("name") == "set_goal" for e in resumed if e["type"] == "tool_call")


def test_stream_goal_completed_does_not_pause() -> None:
    state = InMemorySystemStateProvider()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call(
                "set_goal",
                {"objective": "survey", "remaining": ["y=200", "y=300"]},
                "g1",
            ),
            _tool_call("set_goal", {"completed": "y=200"}, "g2"),
            "still y=300",
        ]
    )
    loop = _make_loop(fake, _make_registry(), system_state=state)
    first = list(loop.stream([{"role": "user", "content": "survey"}]))
    pending = next(e for e in first if e["type"] == "approval")
    resumed = list(
        loop.stream(
            [],
            thread_id=pending["thread_id"],
            resume={"approved_ids": [pending["pending_approvals"][0]["id"]]},
        )
    )
    assert "approval" not in [e["type"] for e in resumed]
    completed = [e for e in resumed if e["type"] == "tool_call" and e.get("id") == "g2"]
    assert completed
    assert completed[0]["result"]["goal"]["remaining"] == ["y=300"]


def test_stream_approved_plan_runs_later_moves_without_reprompt() -> None:
    state = InMemorySystemStateProvider()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call(
                "set_goal",
                {
                    "objective": "photos every 50 mm",
                    "remaining": ["x=50", "x=100"],
                },
                "g1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "move",
                        "args": {"x": 50, "y": 0, "z": 0},
                        "id": "m1",
                        "type": "tool_call",
                    },
                    {
                        "name": "move",
                        "args": {"x": 100, "y": 0, "z": 0},
                        "id": "m2",
                        "type": "tool_call",
                    },
                ],
            ),
            "done",
        ]
    )
    loop = _make_loop(fake, _make_registry(), system_state=state, propose_only=False)
    first = list(loop.stream([{"role": "user", "content": "grid"}]))
    pending = next(e for e in first if e["type"] == "approval")
    resumed = list(
        loop.stream(
            [],
            thread_id=pending["thread_id"],
            resume={"approved_ids": [pending["pending_approvals"][0]["id"]]},
        )
    )
    assert "approval" not in [e["type"] for e in resumed]
    assert [e["name"] for e in resumed if e["type"] == "tool_call"] == [
        "set_goal",
        "move",
        "move",
    ]


def test_agent_loop_set_goal_merges_completed_steps() -> None:
    state = InMemorySystemStateProvider()
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses(
        [
            _tool_call(
                "set_goal",
                {"objective": "survey", "remaining": ["y=200", "y=300"]},
                "g1",
            ),
            _tool_call("set_goal", {"completed": "y=200"}, "g2"),
            "still y=300",
        ]
    )
    loop = _make_loop(fake, _make_registry(), system_state=state, skip_approval=True)
    result = loop.run([{"role": "user", "content": "keep going"}])
    assert result.tool_calls[1]["result"]["goal"]["remaining"] == ["y=300"]
    assert result.tool_calls[1]["result"]["goal"]["done"] == ["y=200"]


def test_water_runs_even_when_a_watering_fact_exists() -> None:
    from planning_service.memory import MemoryStore

    store = MemoryStore(":memory:")
    store.remember("Pump ran for the whole bed (~60s).", category="treatment", tags=["auto", "water"])
    state = InMemorySystemStateProvider(memory_store=store)
    fake = _ScriptFake(responses=["unused"])
    fake.set_responses([_tool_call("water", {}, "w1"), "watered again"])
    loop = _make_loop(
        fake,
        _make_registry(),
        propose_only=False,
        system_state=state,
        skip_approval=True,
    )
    result = loop.run([{"role": "user", "content": "water the bed"}])
    assert result.tool_calls[0]["name"] == "water"
    assert result.tool_calls[0]["result"]["status"] == "ok"
    texts = " ".join(m["content"] for m in store.list_recent(limit=8))
    assert texts.count("Pump ran") >= 2





def test_forced_think_runs_first_then_model_is_free() -> None:
    fake = _ScriptFake(answer_think=True)
    fake.set_responses(["hi"])
    loop = _make_loop(fake, _make_registry(), system_state=InMemorySystemStateProvider(), force_think=True)
    result = loop.run([{"role": "user", "content": "hello"}])
    assert [tc["name"] for tc in result.tool_calls] == ["think"]
    assert result.response == "hi"


def test_deferred_tools_hidden_until_searched() -> None:
    seen: list[set[str]] = []

    class Spy(_ScriptFake):
        pass

    fake = _ScriptFake()
    fake.set_responses(["hi"])
    loop = _make_loop(fake, _make_registry(), system_state=InMemorySystemStateProvider(), core_tools=CORE_TOOLS)
    visible = {t.name for t in loop._tools if not t.defer_loading}
    deferred = {t.name for t in loop._tools if t.defer_loading}
    assert "capture_ndre" in deferred and not (deferred & CORE_TOOLS)
    assert {"think", "move", "get_position"} <= visible


def test_capture_still_reaches_model_as_image(monkeypatch) -> None:
    monkeypatch.setattr(
        "planning_service.tool_results._capture_file_uri",
        lambda *_args: "data:image/jpeg;base64,/9j/4AAQ",
    )
    fake = _ScriptFake()
    fake.set_responses([_tool_call("capture", {"band": "rgb"}, "c1"), "coverage looks patchy"])

    def capture(action: Any) -> Any:
        action.params["artifact_id"] = "shot"
        return action

    loop = _make_loop(fake, _make_registry(capture=capture), propose_only=False)
    result = loop.run([{"role": "user", "content": "check coverage"}])
    assert "coverage looks patchy" in result.response
    assert "BinaryImage" in repr(fake.seen[-1]) or "BinaryContent" in repr(fake.seen[-1])
