"""Tests for the Homing Guard and gantry coordinate datum tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fakes import ScriptModel

from farmbot_gateway import farmbot
from farmbot_gateway.farmbot import FarmBotBackend
from planning_service import run_autonomous_cycle
from safety_service import validate as safety_validate
from twfarmbot_api_server.app import create_app
from twfarmbot_core.actions import ActionRegistry


class _MockBot:
    def __init__(self) -> None:
        self.homed_axes: list[str] = []
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def find_home(self, axis: str = "all", speed: float = 100) -> None:
        self.homed_axes.append(axis)
        self.x, self.y, self.z = 0.0, 0.0, 0.0

    def move(self, x: float, y: float, z: float, speed: float | None = None) -> None:
        self.x, self.y, self.z = x, y, z

    def move_axis(self, axis: str, distance: float, speed: float | None = None) -> None:
        if axis == "x":
            self.x += distance
        elif axis == "y":
            self.y += distance
        elif axis == "z":
            self.z += distance

    def set_home(self, axis: str = "all") -> None:
        self.homed_axes.append(axis)

    def e_stop(self) -> None:
        pass

    def get_xyz(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


def test_backend_homing_state_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Track unhomed on start, homed after find_home('all'), unhomed on e_stop."""
    backend = FarmBotBackend()
    bot = _MockBot()
    monkeypatch.setattr(backend, "_bot", lambda: bot)

    assert backend.is_homed() is False
    assert backend.last_homed_at() is None

    backend.find_home(axis="all")
    assert backend.is_homed() is True
    assert backend.last_homed_at() is not None
    assert bot.homed_axes == ["all"]

    # E-stop un-verifies coordinate calibration
    backend.e_stop()
    assert backend.is_homed() is False


def test_backend_ensure_homed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_homed runs homing only when unhomed."""
    backend = FarmBotBackend()
    bot = _MockBot()
    monkeypatch.setattr(backend, "_bot", lambda: bot)

    assert backend.is_homed() is False
    backend.ensure_homed()
    assert backend.is_homed() is True
    assert bot.homed_axes == ["all"]

    # Second call is a no-op
    backend.ensure_homed()
    assert len(bot.homed_axes) == 1


def test_pose_snapshot_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gantry pose and homing state are saved to gantry_pose.json."""
    monkeypatch.setenv("TWFB_DATA_DIR", str(tmp_path))
    backend = FarmBotBackend()
    bot = _MockBot()
    monkeypatch.setattr(backend, "_bot", lambda: bot)

    backend.find_home(axis="all")
    pose_file = tmp_path / "gantry_pose.json"
    assert pose_file.is_file()
    data = json.loads(pose_file.read_text())
    assert data["is_homed"] is True
    assert data["x"] == 0.0

    backend.move(150.0, 200.0, 10.0)
    data = json.loads(pose_file.read_text())
    assert data["x"] == 150.0
    assert data["y"] == 200.0


def test_autonomy_homing_guard_homes_unhomed_bot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Autonomous cycle ensures the robot is homed before executing actions."""
    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TWFB_DATA_DIR", str(tmp_path))

    bot = _MockBot()
    monkeypatch.setattr(farmbot.backend, "_bot", lambda: bot)
    farmbot.backend._is_homed = False

    dispatched: list[str] = []
    registry = ActionRegistry(validate=safety_validate)
    registry.register("find_home", lambda a: dispatched.append("find_home") or farmbot.backend.find_home("all") or a)
    registry.register("capture_ndre", lambda a: dispatched.append("capture_ndre") or a)

    fake_model = ScriptModel(
        responses=[
            "Bed is healthy.",
        ]
    )

    result = run_autonomous_cycle(
        registry=registry,
        model=fake_model,
        save_session=False,
        ensure_homed=True,
    )

    assert result.status == "ok"
    # find_home was automatically called before any cycle reasoning!
    assert "find_home" in dispatched
    assert farmbot.backend.is_homed() is True


def test_api_health_includes_homed_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health reports homed boolean and timestamp."""
    bot = _MockBot()
    monkeypatch.setattr(farmbot.backend, "_bot", lambda: bot)
    farmbot.backend._is_homed = False
    farmbot.backend._last_homed_at = None

    registry = ActionRegistry(validate=safety_validate)
    app = create_app(registry=registry)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["homed"] is False

    farmbot.backend.find_home(axis="all")
    res2 = client.get("/health")
    assert res2.json()["homed"] is True
    assert res2.json()["last_homed_at"] is not None
