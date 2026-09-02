"""End-to-end test of ``planning_service.plan`` against a fake LLM.

Uses a small tool-aware fake model so no network is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from safety_service import UnsafeActionError

from planning_service import plan
from planning_service.introspection import InMemorySystemStateProvider
from twfarmbot_core.actions import ActionRegistry


class _ToolAwareFake(FakeListChatModel):
    """Offline fake model. JSON fallback is still accepted by ``plan()``."""


class _ToolCallFake(_ToolAwareFake):
    """Fake model that returns canned AIMessages, including JSON tool calls."""

    def __init__(self, messages: list[AIMessage | str]) -> None:
        super().__init__(responses=["unused"])
        self._messages = messages
        self._index = 0

    def invoke(self, *_args: Any, **_kwargs: Any) -> AIMessage:
        item = self._messages[min(self._index, len(self._messages) - 1)]
        self._index += 1
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=str(item))


@pytest.fixture
def world() -> SimpleNamespace:
    return SimpleNamespace(
        to_dict=lambda: {
            "zones": [
                {
                    "id": "tomato",
                    "name": "Tomato Zone",
                    "kind": "zone",
                    "bounds": {"x": 1100, "y": 100, "width": 800, "height": 300},
                }
            ],
            "entities": [
                {
                    "id": "herbs",
                    "name": "Herbs",
                    "kind": "plant",
                    "position": {"x": 120, "y": 180, "z": 0},
                }
            ],
        }
    )


def test_plan_returns_actions_from_fake_model() -> None:
    fake = _ToolAwareFake(
        responses=[
            '{"actions": ['
            '{"kind": "water", "params": {"seconds": 60}},'
            '{"kind": "move", "params": {"x": 0, "y": 0, "z": 0}}'
            '], "rationale": "water then home"}'
        ]
    )
    result = plan("water bed 1 for a minute then go home", model=fake)
    assert [a.kind for a in result.actions] == ["water", "move"]
    assert result.rationale == "water then home"
    assert result.raw_text  # captured for debugging


def test_plan_extracts_json_from_markdown_fence() -> None:
    fake = _ToolAwareFake(
        responses=['```json\n{"actions": [{"kind": "take_photo", "params": {}}]}\n```']
    )
    result = plan("snap a pic", model=fake)
    assert result.actions == [
        __import__("twfarmbot_core.domain", fromlist=["Action"]).Action(
            kind="take_photo", params={}
        )
    ]


def test_plan_propagates_safety_violation(monkeypatch) -> None:
    monkeypatch.setenv("FARMBOT_ENFORCE_WORKSPACE", "1")
    monkeypatch.setenv("FARMBOT_MAX_AXIS_X", "650")
    fake = _ToolAwareFake(
        responses=[
            '{"actions": [{"kind": "move", "params": {"x": 9000, "y": 0, "z": 0}}]}'
        ]
    )
    with pytest.raises(UnsafeActionError):
        plan("go to x=9000", model=fake)


def test_plan_returns_empty_plan_on_unparseable_output() -> None:
    """When the model emits prose, return an empty plan with the prose as rationale.

    A chatty model is no longer a 500 — the user sees the model's
    explanation in the UI instead.
    """
    fake = _ToolAwareFake(responses=["I refuse to comply."])
    result = plan("do something", model=fake)
    assert result.actions == []
    assert "I refuse to comply" in result.rationale


def test_plan_uses_world_context_when_provided(world: SimpleNamespace) -> None:
    fake = _ToolAwareFake(
        responses=['{"actions": [{"kind": "water", "params": {"seconds": 30}}]}']
    )
    result = plan("water the tomato zone", world=world, model=fake)
    assert len(result.actions) == 1
    assert result.actions[0].kind == "water"


def test_plan_preserves_rationale_for_empty_plan() -> None:
    """Empty plans should still carry the model's explanation."""
    fake = _ToolAwareFake(
        responses=[
            '{"actions": [], "rationale": "request is ambiguous: which tomato?"}'
        ]
    )
    result = plan("water the tomato", model=fake)
    assert result.actions == []
    assert "ambiguous" in result.rationale


def test_plan_extracts_actions_from_json_tool_calls() -> None:
    """Physical JSON tool calls become the plan; read tools are executed too."""
    reg = ActionRegistry()
    reg.register("move", lambda a: a)
    reg.register("water", lambda a: a)
    state = InMemorySystemStateProvider(position={"x": 5, "y": 6, "z": 7})

    fake = _ToolCallFake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_position",
                        "args": {},
                        "id": "read_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "water",
                        "args": {"seconds": 10},
                        "id": "act_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "move",
                        "args": {"x": 0, "y": 0, "z": 0},
                        "id": "act_2",
                        "type": "tool_call",
                    },
                ],
            ),
        ]
    )
    result = plan(
        "water then home",
        model=fake,
        registry=reg,
        system_state=state,
    )
    assert [a.kind for a in result.actions] == ["water", "move"]
    assert result.actions[0].params["seconds"] == 10
