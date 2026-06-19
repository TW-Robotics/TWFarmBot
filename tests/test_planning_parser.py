"""Tests for the LLM plan parser — pure, no network."""

from __future__ import annotations

import pytest
from safety_service import UnsafeActionError
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

from planning_service.parser import PlanError, parse_plan


def _registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg.register("move", lambda a: a)
    reg.register("water", lambda a: a)
    reg.register("take_photo", lambda a: a)
    return reg


def test_parses_clean_json() -> None:
    text = '{"actions": [{"kind": "move", "params": {"x": 1, "y": 2, "z": 3}}], "rationale": "ok"}'
    out = parse_plan(text, _registry())
    assert out == [Action(kind="move", params={"x": 1, "y": 2, "z": 3})]


def test_parses_json_in_markdown_fence() -> None:
    text = 'Here you go:\n```json\n{"actions": [{"kind": "take_photo", "params": {}}]}\n```\nDone.'
    out = parse_plan(text, _registry())
    assert out == [Action(kind="take_photo", params={})]


def test_parses_json_embedded_in_prose() -> None:
    text = 'Sure, the plan is {"actions": [{"kind": "water", "params": {"seconds": 60}}]}. Hope that helps.'
    out = parse_plan(text, _registry())
    assert out == [Action(kind="water", params={"seconds": 60})]


def test_unknown_kind_is_rejected() -> None:
    text = '{"actions": [{"kind": "fly_to_mars", "params": {}}]}'
    with pytest.raises(PlanError, match="unknown action kind"):
        parse_plan(text, _registry())


def test_malformed_json_raises_plan_error() -> None:
    with pytest.raises(PlanError, match="no JSON object"):
        parse_plan("sorry, I cannot help with that", _registry())


def test_schema_violation_is_rejected() -> None:
    # 'params' missing entirely — must be an object, default {} in our model,
    # but 'actions' must be a list. We feed a top-level scalar.
    with pytest.raises(PlanError):
        parse_plan('"just a string"', _registry())


def test_unsafe_action_is_rejected_by_safety_gate() -> None:
    # x=9000 exceeds the default safety cap of 3000mm on x.
    text = '{"actions": [{"kind": "move", "params": {"x": 9000, "y": 0, "z": 0}}]}'
    with pytest.raises(UnsafeActionError):
        parse_plan(text, _registry())


def test_empty_actions_is_valid() -> None:
    text = '{"actions": [], "rationale": "request was unsafe"}'
    assert parse_plan(text, _registry()) == []


def test_multiple_actions_in_order() -> None:
    text = (
        '{"actions": ['
        '{"kind": "water", "params": {"seconds": 30}},'
        '{"kind": "move", "params": {"x": 0, "y": 0, "z": 0}}'
        ']}'
    )
    out = parse_plan(text, _registry())
    assert len(out) == 2
    assert out[0].kind == "water"
    assert out[1].kind == "move"
