"""Tests for the low-level Farmduino serial driver."""

from __future__ import annotations

import pytest

from farmbot_serial.client import (
    FarmduinoCommandError,
    FarmduinoSerial,
    FarmduinoTimeoutError,
    build_command,
    parse_response,
)


class _FakeSerial:
    """Mock pyserial.Serial that replays queued responses."""

    def __init__(self, responses: list[bytes]) -> None:
        self.responses = responses
        self.writes: list[bytes] = []
        self.is_open = True

    def readline(self) -> bytes:
        if self.responses:
            return self.responses.pop(0)
        return b""

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        pass

    def reset_output_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


@pytest.fixture
def serial_client(monkeypatch: pytest.MonkeyPatch) -> FarmduinoSerial:
    """Return a FarmduinoSerial wired to a fake port with motion params cached."""
    client = FarmduinoSerial(config={})
    # Pre-populate motion params so tests don't need to read them from firmware.
    client._steps_per_mm = {"x": 80.0, "y": 80.0, "z": 400.0}
    client._max_speed_steps_s = {"x": 6400.0, "y": 6400.0, "z": 6400.0}
    client._home_speed_steps_s = {"x": 3200.0, "y": 3200.0, "z": 3200.0}
    # Wire a default fake serial so the client looks connected without real hardware.
    client._serial = _FakeSerial([])
    client._queue_id = 0
    return client


def _connect(
    client: FarmduinoSerial,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[bytes],
) -> _FakeSerial:
    """Attach a fresh fake serial with the given responses and reset queue state."""
    fake = _FakeSerial(responses)
    monkeypatch.setattr("farmbot_serial.client.Serial", lambda **kwargs: fake)
    client._serial = fake
    client._queue_id = 0
    return fake


# ----------------------------------------------------------------------- protocol


def test_build_command_with_params_and_queue() -> None:
    cmd = build_command("G00", {"X": 100, "Y": 200.5, "Z": 0}, queue_id=7)
    assert cmd == "G00 X100 Y200.50 Z0 Q7"


def test_parse_response_ok() -> None:
    resp = parse_response("R02 Q3")
    assert resp.code == "R02"
    assert resp.queue_id == 3
    assert resp.ok is True


def test_parse_response_error() -> None:
    resp = parse_response("R03 V2 Q0")
    assert resp.code == "R03"
    assert resp.params["V"] == "2"
    assert resp.ok is False


def test_parse_response_position() -> None:
    resp = parse_response("R82 X100 Y200 Z50 Q0")
    assert resp.code == "R82"
    assert resp.params == {"X": "100", "Y": "200", "Z": "50"}


# ----------------------------------------------------------------------- commands


def test_send_command_success(serial_client, monkeypatch) -> None:
    client = serial_client
    fake = _connect(client, monkeypatch, [b"R01 Q0\r\n", b"R02 Q0\r\n"])
    resp = client.send_command("F41", {"P": 8, "V": 1, "M": 0})
    assert resp.code == "R02"
    assert fake.writes == [b"F41 P8 V1 M0 Q0\r\n"]


def test_send_command_error_raises(serial_client, monkeypatch) -> None:
    client = serial_client
    _connect(client, monkeypatch, [b"R01 Q0\r\n", b"R03 V14 Q0\r\n"])
    with pytest.raises(FarmduinoCommandError) as excinfo:
        client.send_command("G00", {"X": 99999})
    assert excinfo.value.error_value == 14


def test_send_command_timeout(serial_client, monkeypatch) -> None:
    client = serial_client
    _connect(client, monkeypatch, [])
    with pytest.raises(FarmduinoTimeoutError):
        client.send_command("F82", timeout_s=0.05)


def test_move_uses_speed_letters(serial_client, monkeypatch) -> None:
    client = serial_client
    fake = _connect(client, monkeypatch, [b"R01 Q0\r\n", b"R02 Q0\r\n"])
    client.move(100, 200, 50, speed_mm_s=10)
    written = fake.writes[0].decode()
    assert "X100" in written
    assert "Y200" in written
    assert "Z50" in written
    assert "A" in written and "B" in written and "C" in written


def test_read_pin_returns_value(serial_client, monkeypatch) -> None:
    client = serial_client
    _connect(client, monkeypatch, [b"R41 P14 V512 Q0\r\n"])
    assert client.read_pin(14, mode="analog") == 512


def test_write_pin_timed_uses_f44(serial_client, monkeypatch) -> None:
    client = serial_client
    fake = _connect(client, monkeypatch, [b"R01 Q0\r\n", b"R02 Q0\r\n"])
    client.write_pin_timed(8, 1, seconds=1.5, mode="digital")
    written = fake.writes[0].decode()
    assert written.startswith("F44 P8 V1 W0 T1500 M0")


def test_get_position_parses_r82(serial_client, monkeypatch) -> None:
    client = serial_client
    _connect(client, monkeypatch, [b"R82 X10 Y20 Z5 Q0\r\n"])
    assert client.get_position() == {"x": 10.0, "y": 20.0, "z": 5.0}


def test_e_stop_writes_raw_e(serial_client, monkeypatch) -> None:
    client = serial_client
    fake = _connect(client, monkeypatch, [])
    client.e_stop()
    assert fake.writes == [b"E\r\n"]
