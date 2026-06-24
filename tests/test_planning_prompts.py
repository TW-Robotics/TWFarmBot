"""Tests for the planner/chat prompt builders."""

from __future__ import annotations

from planning_service.harness import ContextBuilder, ToolRegistry
from planning_service.prompt import build_user_prompt
from twfarmbot_core.actions import ActionRegistry


def _make_registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg.register("move", lambda a: a)
    reg.register("find_home", lambda a: a)
    reg.register("water", lambda a: a)
    return reg


def test_system_prompt_mentions_action_kinds() -> None:
    reg = _make_registry()
    prompt = ContextBuilder(ToolRegistry(reg)).planner_system_prompt()
    assert "move" in prompt
    assert "find_home" in prompt
    assert "water" in prompt


def test_chat_prompt_lists_execution_tools() -> None:
    reg = _make_registry()
    prompt = ContextBuilder(ToolRegistry(reg)).chat_system_prompt()
    assert "Execution tools" in prompt
    assert "move" in prompt


def test_user_prompt_includes_world_context() -> None:
    prompt = build_user_prompt(
        "water the tomato",
        world_context="- zone Tomato (id=tomato)",
    )
    assert "zone Tomato" in prompt
    assert "Request: water the tomato" in prompt


def test_user_prompt_omits_world_when_none() -> None:
    prompt = build_user_prompt("snap a pic")
    assert "Current world model" not in prompt
    assert "Request: snap a pic" in prompt
