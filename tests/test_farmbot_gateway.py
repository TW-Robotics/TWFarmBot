"""Tests for cached FarmBot gateway state."""

from __future__ import annotations

from types import SimpleNamespace

from farmbot_gateway import FarmBotLink


def test_cached_xyz_reads_latest_status_without_connecting() -> None:
    state = SimpleNamespace(last_messages={
        "status": [
            {"content": {"location_data": {"position": {"x": 1, "y": 2, "z": 3}}}},
            {"content": {"location_data": {"position": {"x": 4, "y": 5, "z": 6}}}},
        ]
    })
    link = FarmBotLink(SimpleNamespace(state=state))

    assert link.cached_xyz() == {"x": 4.0, "y": 5.0, "z": 6.0}


def test_cached_xyz_is_empty_before_first_status() -> None:
    link = FarmBotLink(SimpleNamespace(
        state=SimpleNamespace(last_messages={"status": []})
    ))

    assert link.cached_xyz() == {}
