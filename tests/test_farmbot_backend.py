"""Tests for the real FarmBot backend + handler arg translation.

The backend is tested with a mocked ``farmbot_client.get_client``.
"""

from __future__ import annotations

from typing import Any

import pytest

from twfarmbot_core.domain import Action
from watering_service.backends import farmbot


class _FakeBot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str):
        def _call(*args, **kwargs):
            self.calls.append((name, {"args": args, **kwargs}))
            return {"ok": True, "method": name}

        return _call


@pytest.fixture
def fake_bot(monkeypatch: pytest.MonkeyPatch) -> _FakeBot:
    bot = _FakeBot()
    monkeypatch.setattr("farmbot_gateway.get_farmbot", lambda: bot)
    return bot


def _new_backend(monkeypatch: pytest.MonkeyPatch, pump_pin: int | None = None) -> Any:
    if pump_pin is not None:
        monkeypatch.setenv("FARMBOT_PUMP_PIN", str(pump_pin))
    return farmbot.FarmBotBackend()


def test_backend_water_opens_then_closes(fake_bot, monkeypatch):
    backend = _new_backend(monkeypatch, pump_pin=13)
    backend.water(0.01)
    methods = [m for m, _ in fake_bot.calls]
    assert methods[0] == "water_on"
    assert methods[-1] == "water_off"


def test_backend_move(fake_bot, monkeypatch):
    backend = _new_backend(monkeypatch)
    backend.move(100.0, 200.0, 50.0)
    method, kwargs = fake_bot.calls[0]
    assert method == "move"
    assert kwargs["args"][:3] == (100.0, 200.0, 50.0)


def test_backend_mount_and_dismount_are_noop(fake_bot, monkeypatch):
    backend = _new_backend(monkeypatch)
    backend.mount_tool("weeder")
    backend.dismount_tool()
    assert fake_bot.calls == []


def test_backend_e_stop(fake_bot, monkeypatch):
    backend = _new_backend(monkeypatch)
    backend.e_stop()
    assert any(m == "e_stop" for m, _ in fake_bot.calls)


def test_backend_take_photo_calls_client(fake_bot, monkeypatch):
    backend = _new_backend(monkeypatch)
    backend.take_photo()
    assert any(m == "take_photo" for m, _ in fake_bot.calls)


def test_handler_water_translates_args(monkeypatch):
    from twfarmbot_api_server.handlers import watering as h

    seen = []
    monkeypatch.setattr("watering_service.water", lambda seconds: seen.append(seconds))

    out = h.handle_water(Action(kind="water", params={"seconds": 7}))
    assert seen == [7.0]
    assert out.kind == "water"


def test_handler_move_translates_args(monkeypatch):
    from twfarmbot_api_server.handlers import move as h

    seen = []

    class _StubBackend:
        def move(self, x, y, z, speed=None):
            seen.append((x, y, z, speed))

    monkeypatch.setattr(farmbot, "backend", _StubBackend())

    h.handle_move(Action(kind="move", params={"x": 100, "y": 200, "z": 50}))
    h.handle_move(Action(kind="move", params={"x": 1, "y": 2, "z": 3, "speed": 50}))
    assert seen == [(100.0, 200.0, 50.0, None), (1.0, 2.0, 3.0, 50.0)]


def test_handler_read_pin_returns_value_in_params(monkeypatch):
    from twfarmbot_api_server.handlers import pin as h

    class _StubBackend:
        def read_pin(self, pin, mode="digital"):
            return 42

    monkeypatch.setattr(farmbot, "backend", _StubBackend())

    out = h.handle_read_pin(Action(kind="read_pin", params={"pin": 13}))
    assert out.params == {"pin": 13, "mode": "digital", "value": 42}


def test_handler_send_message_translates(monkeypatch):
    from twfarmbot_api_server.handlers import feedback as h

    seen = []

    class _StubBackend:
        def send_message(self, message, message_type="info", channels=None):
            seen.append((message, message_type, channels))

    monkeypatch.setattr(farmbot, "backend", _StubBackend())

    h.handle_send_message(Action(kind="send_message", params={"message": "hi"}))
    h.handle_send_message(
        Action(kind="send_message", params={"message": "warn", "type": "warn"})
    )
    assert seen == [("hi", "info", None), ("warn", "warn", None)]


def test_handler_e_stop_calls_backend(monkeypatch):
    from twfarmbot_api_server.handlers import feedback as h

    called = []

    class _StubBackend:
        def e_stop(self):
            called.append(True)

    monkeypatch.setattr(farmbot, "backend", _StubBackend())

    h.handle_e_stop(Action(kind="e_stop", params={}))
    assert called == [True]


def test_handler_find_home_calls_backend(monkeypatch):
    from twfarmbot_api_server.handlers import find_home as h

    called = []

    class _StubBackend:
        def find_home(self):
            called.append(True)

    monkeypatch.setattr(farmbot, "backend", _StubBackend())

    h.handle_find_home(Action(kind="find_home", params={}))
    assert called == [True]


def test_handler_take_photo_calls_backend(monkeypatch):
    from twfarmbot_api_server.handlers import camera as h

    called = []

    class _StubBackend:
        def take_photo(self):
            called.append(True)

    monkeypatch.setattr(farmbot, "backend", _StubBackend())
    h.handle_take_photo(Action(kind="take_photo", params={}))
    assert called == [True]


def test_safety_rejects_move_outside_bounds():
    from safety_service import SafetyLimits, UnsafeActionError, validate

    limits = SafetyLimits(max_axis_mm={"x": 1000, "y": 1000, "z": 1000})
    with pytest.raises(UnsafeActionError, match="exceeds"):
        validate(Action(kind="move", params={"x": 9999, "y": 0, "z": 0}), limits=limits)


def test_safety_accepts_move_within_bounds():
    from safety_service import SafetyLimits, validate

    limits = SafetyLimits(max_axis_mm={"x": 1000, "y": 1000, "z": 1000})
    out = validate(
        Action(kind="move", params={"x": 500, "y": 500, "z": 500}), limits=limits
    )
    assert out.kind == "move"
