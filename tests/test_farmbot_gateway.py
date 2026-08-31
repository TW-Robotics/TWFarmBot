"""Tests for local FarmBot gateway handle."""

from __future__ import annotations

from types import SimpleNamespace

from farmbot_gateway import FarmBotLink


def test_cached_xyz_reads_client_position(monkeypatch) -> None:
    monkeypatch.setattr(
        "farmbot_gateway.get_client",
        lambda: SimpleNamespace(get_xyz=lambda: {"x": 4.0, "y": 5.0, "z": 6.0}),
    )
    assert FarmBotLink().cached_xyz() == {"x": 4.0, "y": 5.0, "z": 6.0}


def test_cached_xyz_is_empty_when_disconnected(monkeypatch) -> None:
    from farmbot_client import FarmBotConnectionError

    def boom():
        raise FarmBotConnectionError("down")

    monkeypatch.setattr(
        "farmbot_gateway.get_client",
        lambda: SimpleNamespace(get_xyz=boom),
    )
    assert FarmBotLink().cached_xyz() == {}
