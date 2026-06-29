"""Tests for path-planning helpers and the move_path action."""

from __future__ import annotations

from typing import Any

import pytest

from planning_service import path_planning
from planning_service.harness import ToolRegistry
from planning_service.introspection import (
    InMemorySystemStateProvider,
    build_introspection_tools,
)
from safety_service import UnsafeActionError, validate
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action


def test_plan_path_samples_straight_line() -> None:
    waypoints = path_planning.plan_path(
        {"x": 0, "y": 0},
        {"x": 300, "y": 0},
        step_mm=100,
        z=50,
    )
    assert [w["x"] for w in waypoints] == [0.0, 100.0, 200.0, 300.0]
    assert all(w["z"] == 50 for w in waypoints)
    assert len(waypoints) == 4


def test_plan_path_clamps_to_bounds() -> None:
    waypoints = path_planning.plan_path(
        {"x": -100, "y": 50},
        {"x": 3000, "y": 50},
        step_mm=2000,
        z=0,
    )
    assert waypoints[0]["x"] >= 0
    assert waypoints[-1]["x"] <= 1900


def test_scan_zone_returns_raster() -> None:
    waypoints = path_planning.scan_zone("tomato", step_mm=200, z=100)
    assert len(waypoints) > 0
    assert all(w["z"] == 100 for w in waypoints)
    # Rows should alternate direction: first row increasing X, second decreasing.
    row1_x = [w["x"] for w in waypoints if w["y"] == 200.0]
    row2_x = [w["x"] for w in waypoints if w["y"] == 300.0]
    if row1_x and row2_x:
        assert row1_x == sorted(row1_x)
        assert row2_x == sorted(row2_x, reverse=True)


def test_scan_zone_unknown_zone() -> None:
    with pytest.raises(ValueError, match="not found"):
        path_planning.scan_zone("mars", step_mm=100)


def test_introspection_tools_expose_path_planners() -> None:
    provider = InMemorySystemStateProvider()
    tools = build_introspection_tools(provider)
    names = {t.name for t in tools}
    assert "plan_path" in names
    assert "scan_zone" in names


def test_tool_registry_exposes_move_path() -> None:
    reg = ActionRegistry()
    reg.register("move", lambda a: a)
    reg.register("move_path", lambda a: a)
    tool_registry = ToolRegistry(reg)
    names = {d.name for d in tool_registry.descriptors()}
    assert "move_path" in names


def test_move_path_safety_rejects_out_of_bounds() -> None:
    action = Action(
        kind="move_path",
        params={"waypoints": [{"x": 0, "y": 0, "z": 0}, {"x": 9999, "y": 0, "z": 0}]},
    )
    with pytest.raises(UnsafeActionError):
        validate(action)


def test_move_path_safety_accepts_valid_waypoints() -> None:
    action = Action(
        kind="move_path",
        params={
            "waypoints": [{"x": 0, "y": 0, "z": 0}, {"x": 500, "y": 200, "z": 100}]
        },
    )
    assert validate(action) is action


def test_handle_move_path_calls_backend(monkeypatch: Any) -> None:
    from twfarmbot_api_server.handlers import path as path_handler

    moves: list[tuple[float, float, float]] = []
    photos: list[Any] = []

    class _StubBackend:
        def move(self, x: float, y: float, z: float, speed: float | None = None) -> None:
            moves.append((x, y, z))

        def take_photo(self) -> None:
            photos.append(None)

        def get_xyz(self) -> dict[str, float]:
            return {"x": 40.0, "y": 50.0, "z": 60.0}

    monkeypatch.setattr("watering_service.get_backend", _StubBackend)

    action = Action(
        kind="move_path",
        params={
            "waypoints": [
                {"x": 10, "y": 20, "z": 30},
                {"x": 40, "y": 50, "z": 60},
            ],
            "photo_at_waypoints": True,
        },
    )
    result = path_handler.handle_move_path(action)
    assert result is action
    assert moves == [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)]
    assert len(photos) == 2
