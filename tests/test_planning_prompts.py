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
    reg.register("goto_named", lambda a: a)
    reg.register("run_script", lambda a: a)
    return reg


def test_planner_prompt_is_tool_first() -> None:
    prompt = ContextBuilder(ToolRegistry(_make_registry())).planner_system_prompt()
    assert "JSON tool calls" in prompt
    assert "goto_named" in prompt and "find_locations" in prompt
    assert "search_tools" in prompt
    assert "Hard travel box" in prompt


def test_chat_prompt_is_slim_no_catalog() -> None:
    prompt = ContextBuilder(ToolRegistry(_make_registry())).chat_system_prompt()
    assert "Available tools" not in prompt and "NDRE = (" not in prompt
    assert "search_tools" in prompt and "get_help" in prompt


def test_ndre_notes_live_in_get_help_not_the_prompt() -> None:
    from planning_service.help_topics import get_help

    notes = get_help("spectral")
    assert "NDRE = (NIR − red-edge) / (NIR + red-edge)" in notes
    assert "not NDVI" in notes and "not thermal" in notes
    assert "interpretation.advice" in notes
    prompt = ContextBuilder(ToolRegistry(_make_registry())).chat_system_prompt()
    assert "NDRE = (" not in prompt and "Hard travel box" in prompt


def test_voice_prompt_asks_for_short_spoken_ndre() -> None:
    prompt = ContextBuilder(ToolRegistry(_make_registry())).chat_system_prompt(
        voice=True
    )
    assert "Voice mode" in prompt
    assert "NDRE" in prompt
    assert "two short sentences" in prompt


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


def test_chat_prompt_injects_no_live_data_and_points_to_tools() -> None:
    """World model/weather/memories are never in the prompt; the model uses tools."""
    builder = ContextBuilder(ToolRegistry(_make_registry()))
    text = builder.chat_system_prompt()
    for bad in ("Current world model", "<background>", "Long-Term Memor", "Solar Context"):
        assert bad not in text
    for tool in ("get_positions", "goto_named", "search_tools"):
        assert tool in text
    assert builder.chat_system_prompt() == text  # identical every turn: prefix caching holds


def test_chat_prompt_forces_fresh_tool_call_for_live_state() -> None:
    text = str(ContextBuilder(ToolRegistry(_make_registry())).chat_system_prompt())
    assert "get_position" in text and "not from memory" not in text and "never from memory" in text
