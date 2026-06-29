"""Tests for the direct serial backend + handler arg translation."""

from __future__ import annotations

from typing import Any, cast

import pytest

from twfarmbot_core.domain import Action
from watering_service.backends.direct_serial import DirectSerialBackend


class _FakeSerial:
    """Stand-in for ``FarmduinoSerial`` that records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.connected = True
        self._position = {"x": 0.0, "y": 0.0, "z": 0.0}

    def connect(self) -> None:
        self.calls.append(("connect", {}))

    def move(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        speed_mm_s: float | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"speed_mm_s": speed_mm_s}
        if x is not None:
            kwargs["x"] = x
        if y is not None:
            kwargs["y"] = y
        if z is not None:
            kwargs["z"] = z
        self.calls.append(("move", kwargs))
        for axis in ("x", "y", "z"):
            if axis in kwargs:
                self._position[axis] = kwargs[axis]

    def home(self, axis: str = "all") -> None:
        self.calls.append(("home", {"axis": axis}))

    def set_home(self, axis: str = "all") -> None:
        self.calls.append(("set_home", {"axis": axis}))

    def read_pin(self, pin: int, mode: str = "digital") -> int:
        self.calls.append(("read_pin", {"pin": pin, "mode": mode}))
        return 42

    def write_pin(self, pin: int, value: int, mode: str = "digital") -> None:
        self.calls.append(("write_pin", {"pin": pin, "value": value, "mode": mode}))

    def write_pin_timed(
        self, pin: int, value: int, seconds: float, mode: str = "digital"
    ) -> None:
        self.calls.append(
            ("write_pin_timed", {"pin": pin, "value": value, "seconds": seconds, "mode": mode})
        )

    def get_position(self) -> dict[str, float]:
        self.calls.append(("get_position", {}))
        return dict(self._position)

    def get_endstops(self) -> dict[str, tuple[int, int]]:
        self.calls.append(("get_endstops", {}))
        return {"x": (0, 0), "y": (0, 0), "z": (0, 0)}

    def e_stop(self) -> None:
        self.calls.append(("e_stop", {}))

    def abort(self) -> None:
        self.calls.append(("abort", {}))

    def approve_config(self) -> None:
        self.calls.append(("approve_config", {}))


def _fake_serial(backend: DirectSerialBackend) -> Any:
    """Return the monkey-patched fake serial object attached to ``backend``."""
    return cast(Any, backend._ensure_serial())


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> DirectSerialBackend:
    """Fresh DirectSerialBackend wired to a recording fake serial object."""
    fake_serial = _FakeSerial()
    monkeypatch.setattr(
        "watering_service.backends.direct_serial.DirectSerialBackend._ensure_serial",
        lambda self: fake_serial,
    )
    return DirectSerialBackend()


# ----------------------------------------------------------------------- backend


def test_backend_water_uses_timed_pin(backend: DirectSerialBackend) -> None:
    backend.water(0.5)
    methods = [c[0] for c in _fake_serial(backend).calls]
    assert "write_pin_timed" in methods
    call = [c for c in _fake_serial(backend).calls if c[0] == "write_pin_timed"][0]
    assert call[1]["pin"] == 8
    assert call[1]["value"] == 1
    assert call[1]["seconds"] == 0.5


def test_backend_move_translates_args(backend: DirectSerialBackend) -> None:
    backend.move(100.0, 200.0, 50.0)
    call = _fake_serial(backend).calls[0]
    assert call[0] == "move"
    assert call[1]["x"] == 100.0
    assert call[1]["y"] == 200.0
    assert call[1]["z"] == 50.0


def test_backend_find_home(backend: DirectSerialBackend) -> None:
    backend.find_home("all")
    assert _fake_serial(backend).calls[0] == ("home", {"axis": "all"})


def test_backend_read_pin_returns_value(backend: DirectSerialBackend) -> None:
    assert backend.read_pin(14, "analog") == 42


def test_backend_write_pin_with_seconds_uses_timed(
    backend: DirectSerialBackend,
) -> None:
    backend.write_pin(8, 1, "digital", seconds=2.0)
    methods = [c[0] for c in _fake_serial(backend).calls]
    assert "write_pin_timed" in methods


def test_backend_write_pin_without_seconds_uses_plain(
    backend: DirectSerialBackend,
) -> None:
    backend.write_pin(8, 0, "digital")
    methods = [c[0] for c in _fake_serial(backend).calls]
    assert "write_pin" in methods
    assert "write_pin_timed" not in methods


def test_backend_e_stop(backend: DirectSerialBackend) -> None:
    backend.e_stop()
    assert _fake_serial(backend).calls[0] == ("e_stop", {})


def test_backend_get_xyz_caches_position(backend: DirectSerialBackend) -> None:
    _fake_serial(backend)._position = {"x": 1.0, "y": 2.0, "z": 3.0}
    assert backend.get_xyz() == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert _fake_serial(backend).calls[0] == ("get_position", {})


# ----------------------------------------------------------------------- handlers


def test_handler_water_translates_args(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import watering as h

    seen = []

    class _StubBackend:
        def water(self, seconds):
            seen.append(seconds)

    monkeypatch.setattr("watering_service.get_backend", lambda: _StubBackend())

    out = h.handle_water(Action(kind="water", params={"seconds": 7}))
    assert seen == [7.0]
    assert out.kind == "water"


def test_handler_move_translates_args(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import move as h

    seen = []

    class _StubBackend:
        def move(self, x, y, z, speed=None):
            seen.append((x, y, z, speed))

    monkeypatch.setattr("watering_service.get_backend", lambda: _StubBackend())

    h.handle_move(Action(kind="move", params={"x": 100, "y": 200, "z": 50}))
    h.handle_move(Action(kind="move", params={"x": 1, "y": 2, "z": 3, "speed": 50}))
    assert seen == [(100.0, 200.0, 50.0, None), (1.0, 2.0, 3.0, 50.0)]


def test_handler_read_pin_returns_value_in_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import pin as h

    class _StubBackend:
        def read_pin(self, pin, mode="digital"):
            return 42

    monkeypatch.setattr("watering_service.get_backend", lambda: _StubBackend())

    out = h.handle_read_pin(Action(kind="read_pin", params={"pin": 13}))
    assert out.params == {"pin": 13, "mode": "digital", "value": 42}


def test_handler_send_message_translates(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import feedback as h

    seen = []

    class _StubBackend:
        def send_message(self, message, message_type="info", channels=None):
            seen.append((message, message_type, channels))

    monkeypatch.setattr("watering_service.get_backend", lambda: _StubBackend())

    h.handle_send_message(Action(kind="send_message", params={"message": "hi"}))
    h.handle_send_message(
        Action(kind="send_message", params={"message": "warn", "type": "warn"})
    )
    assert seen == [("hi", "info", None), ("warn", "warn", None)]


def test_handler_e_stop_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import feedback as h

    called = []

    class _StubBackend:
        def e_stop(self):
            called.append(True)

    monkeypatch.setattr("watering_service.get_backend", lambda: _StubBackend())

    h.handle_e_stop(Action(kind="e_stop", params={}))
    assert called == [True]


def test_handler_find_home_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import find_home as h

    called = []

    class _StubBackend:
        def find_home(self):
            called.append(True)

    monkeypatch.setattr("watering_service.get_backend", lambda: _StubBackend())

    h.handle_find_home(Action(kind="find_home", params={}))
    assert called == [True]


def test_handler_take_photo_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import camera as h

    called = []

    class _StubBackend:
        def take_photo(self):
            called.append(True)

    monkeypatch.setattr("watering_service.get_backend", lambda: _StubBackend())
    h.handle_take_photo(Action(kind="take_photo", params={}))
    assert called == [True]


# ----------------------------------------------------------------------- safety


def test_safety_rejects_move_outside_bounds() -> None:
    from safety_service import SafetyLimits, UnsafeActionError, validate

    limits = SafetyLimits(max_axis_mm={"x": 1000, "y": 1000, "z": 1000})
    with pytest.raises(UnsafeActionError, match="exceeds"):
        validate(Action(kind="move", params={"x": 9999, "y": 0, "z": 0}), limits=limits)


def test_safety_accepts_move_within_bounds() -> None:
    from safety_service import SafetyLimits, validate

    limits = SafetyLimits(max_axis_mm={"x": 1000, "y": 1000, "z": 1000})
    out = validate(
        Action(kind="move", params={"x": 500, "y": 500, "z": 500}), limits=limits
    )
    assert out.kind == "move"
