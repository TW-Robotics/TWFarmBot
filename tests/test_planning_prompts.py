"""Tests for the planner prompt builders."""

from __future__ import annotations

from planning_service.prompt import build_system_prompt, build_user_prompt


def test_system_prompt_mentions_move_vs_find_home() -> None:
    prompt = build_system_prompt(["move", "find_home", "water"])
    assert "move(x=0, y=0, z=0)" in prompt
    assert "find_home" in prompt
    assert "end-stop homing" in prompt or "physical limits" in prompt


def test_user_prompt_includes_bed_ids() -> None:
    prompt = build_user_prompt(
        "water the tomato",
        world_context="- zone Tomato (id=tomato)",
        bed_ids=["b1", "b2", "b3"],
    )
    assert "b1, b2, b3" in prompt
    assert "bed_id" in prompt
    assert "Request: water the tomato" in prompt


def test_user_prompt_omits_bed_ids_when_none() -> None:
    prompt = build_user_prompt("snap a pic")
    assert "bed_id" not in prompt
    assert "Request: snap a pic" in prompt


def test_user_prompt_omits_world_when_none() -> None:
    prompt = build_user_prompt("snap a pic", bed_ids=["b1"])
    assert "Current world model" not in prompt
    assert "b1" in prompt
