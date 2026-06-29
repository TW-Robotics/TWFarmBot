"""Minimal USB-serial driver for the Farmduino G-code/F-code protocol.

The Farmduino firmware speaks plain ASCII over a serial line:
- Commands: ``G00``, ``F22``, ``F41``, ``F42``, ``E``, ``@``, ...
- Responses: ``R01`` started, ``R02`` ok, ``R03`` error, ``R82`` position, ...
- Every command can carry a ``Q<id>`` queue tag; responses echo it.

This module intentionally contains no domain logic (watering, cameras,
UI). It only knows how to talk to the microcontroller.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from serial import Serial, SerialException

from twfarmbot_core.config import load_yaml_config

log = logging.getLogger(__name__)


class FarmduinoError(Exception):
    """Base class for Farmduino serial errors."""


class FarmduinoConnectionError(FarmduinoError):
    """Raised when the serial link cannot be opened or is lost."""


class FarmduinoTimeoutError(FarmduinoError):
    """Raised when a command does not finish in time."""


class FarmduinoCommandError(FarmduinoError):
    """Raised when the firmware responds with ``R03`` (command failed)."""

    def __init__(
        self, message: str, *, response_code: str | None = None, error_value: int | None = None
    ) -> None:
        super().__init__(message)
        self.response_code = response_code
        self.error_value = error_value


@dataclass(frozen=True)
class GCodeResponse:
    """One line returned by the Farmduino firmware."""

    raw: str
    code: str | None  # e.g. "R02"
    queue_id: int | None
    params: dict[str, str]

    @property
    def ok(self) -> bool:
        return self.code == "R02"


# Board defaults for a Genesis v1.8 (Farmduino v1.32).
# These are used when the YAML config or the firmware itself don't provide values.
_DEFAULT_CONFIG: dict[str, Any] = {
    "hardware": {
        "version": "genesis_v1.8",
        "board": "farmduino_v32",
        "serial": {
            "port": "/dev/ttyACM0",
            "baud": 115200,
            "connect_timeout_s": 10.0,
            "command_timeout_s": 60.0,
            "move_timeout_s": 300.0,
        },
        "movement": {
            # Fallback values only; the driver tries to read the real values
            # from firmware parameters 55-57 and 65-67 and 71-73 on connect.
            "steps_per_mm": {"x": 80.0, "y": 80.0, "z": 400.0},
            "max_speed_mm_s": {"x": 80.0, "y": 80.0, "z": 16.0},
            "home_speed_mm_s": {"x": 40.0, "y": 40.0, "z": 8.0},
        },
        "peripherals": {
            "water": {"pin": 8, "mode": "digital"},
            "vacuum": {"pin": 9, "mode": "digital"},
            "lighting": {"pin": 7, "mode": "digital"},
        },
    }
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _mode_int(mode: str) -> int:
    """Convert human-readable pin mode to firmware integer."""
    mode = str(mode).lower()
    if mode in ("digital", "output", "0"):
        return 0
    if mode in ("analog", "input", "1"):
        return 1
    raise FarmduinoError(f"unsupported pin mode {mode!r}")


def _axis_letter(axis: str) -> str:
    axis = axis.lower()
    if axis in ("x", "y", "z"):
        return axis.upper()
    raise FarmduinoError(f"unsupported axis {axis!r}")


def build_command(
    code: str, params: dict[str, Any] | None = None, queue_id: int | None = None
) -> str:
    """Build an ASCII command string such as ``G00 X100 Y200 Z0 Q0``."""
    parts = [code]
    if params:
        for key, value in params.items():
            if isinstance(value, float):
                # Strip trailing .0 for readability; firmware accepts floats too.
                text = str(int(value)) if value == int(value) else f"{value:.2f}"
            else:
                text = str(value)
            parts.append(f"{key}{text}")
    if queue_id is not None:
        parts.append(f"Q{queue_id}")
    return " ".join(parts)


def parse_response(line: str) -> GCodeResponse:
    """Parse one line from the Farmduino into a :class:`GCodeResponse`."""
    line = line.strip()
    tokens = line.split()
    if not tokens or not tokens[0].startswith("R"):
        return GCodeResponse(raw=line, code=None, queue_id=None, params={})

    code = tokens[0]
    queue_id: int | None = None
    params: dict[str, str] = {}
    for tok in tokens[1:]:
        if tok.startswith("Q") and tok[1:].isdigit():
            queue_id = int(tok[1:])
            continue
        # Token like X100, V1, P8. Key is leading letter(s), value is rest.
        match = re.match(r"^([A-Za-z]+)(.*)$", tok)
        if match:
            params[match.group(1)] = match.group(2)
    return GCodeResponse(raw=line, code=code, queue_id=queue_id, params=params)


class FarmduinoSerial:
    """Thread-safe serial client for the Farmduino firmware."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._user_config = config or {}
        self._serial: Serial | None = None
        self._lock = threading.Lock()
        self._queue_id = 0
        self._steps_per_mm: dict[str, float] = {}
        self._max_speed_steps_s: dict[str, float] = {}
        self._home_speed_steps_s: dict[str, float] = {}

    # ------------------------------------------------------------------ config

    @property
    def config(self) -> dict[str, Any]:
        """Return merged config: defaults < YAML < user-supplied dict."""
        yaml_config = self._user_config or load_yaml_config()
        return _deep_merge(_DEFAULT_CONFIG, yaml_config)

    def _hw(self) -> dict[str, Any]:
        return self.config.get("hardware", {})

    def _serial_cfg(self) -> dict[str, Any]:
        return self._hw().get("serial", {})

    def _movement_cfg(self) -> dict[str, Any]:
        return self._hw().get("movement", {})

    # ------------------------------------------------------------------ lifecycle

    @property
    def connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        """Open the serial port, approve config, and cache motion parameters."""
        cfg = self._serial_cfg()
        port = cfg.get("port", "/dev/ttyACM0")
        baud = int(cfg.get("baud", 115200))
        timeout = float(cfg.get("connect_timeout_s", 10.0))

        log.info("farmduino: opening %s @ %d baud", port, baud)
        try:
            self._serial = Serial(port=port, baudrate=baud, timeout=1.0)
        except SerialException as err:
            raise FarmduinoConnectionError(f"could not open {port}: {err}") from err

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._read_line(deadline)
            if line is None:
                continue
            if "ARDUINO STARTUP COMPLETE" in line:
                break
            if "R99" not in line:
                log.debug("farmduino boot line: %s", line.strip())
        else:
            # Some firmware builds don't emit the startup banner; keep going.
            log.warning("farmduino: startup banner not seen, continuing anyway")

        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

        # Firmware rejects motion until config is approved.
        self.approve_config()
        self._load_motion_params()
        log.info(
            "farmduino: connected; steps/mm=%s max_speed_steps_s=%s",
            self._steps_per_mm,
            self._max_speed_steps_s,
        )

    def close(self) -> None:
        """Close the serial port."""
        with self._lock:
            if self._serial is not None and self._serial.is_open:
                try:
                    self._serial.close()
                except SerialException:
                    log.exception("farmduino: error closing port")
            self._serial = None

    def _ensure_connected(self) -> Serial:
        if not self.connected:
            self.connect()
        assert self._serial is not None
        return self._serial

    # ------------------------------------------------------------------ parameters

    def _read_param(self, param_id: int, timeout_s: float = 5.0) -> int | None:
        resp = self._send_and_expect("F21", {"P": param_id}, "R21", timeout_s=timeout_s)
        try:
            return int(resp.params.get("V", "0"))
        except ValueError:
            return None

    def _load_motion_params(self) -> None:
        """Read steps/mm and speed limits from the firmware if possible."""
        fallback = self._movement_cfg()

        steps_per_mm = {}
        for axis, pid in (("x", 55), ("y", 56), ("z", 57)):
            value = self._read_param(pid)
            if value is None:
                value = int(fallback.get("steps_per_mm", {}).get(axis, 80))
            steps_per_mm[axis] = float(value)
        self._steps_per_mm = steps_per_mm

        max_speed = {}
        for axis, pid in (("x", 71), ("y", 72), ("z", 73)):
            value = self._read_param(pid)
            if value is None:
                value = int(
                    fallback.get("max_speed_mm_s", {}).get(axis, 80)
                    * steps_per_mm[axis]
                )
            max_speed[axis] = float(value)
        self._max_speed_steps_s = max_speed

        home_speed = {}
        for axis, pid in (("x", 65), ("y", 66), ("z", 67)):
            value = self._read_param(pid)
            if value is None:
                value = int(
                    fallback.get("home_speed_mm_s", {}).get(axis, 40)
                    * steps_per_mm[axis]
                )
            home_speed[axis] = float(value)
        self._home_speed_steps_s = home_speed

    def steps_per_mm(self) -> dict[str, float]:
        return dict(self._steps_per_mm)

    # ------------------------------------------------------------------ transport

    def _read_line(self, deadline: float) -> str | None:
        """Read one line from the serial port, respecting an absolute deadline."""
        ser = self._ensure_connected()
        while time.monotonic() < deadline:
            try:
                raw = ser.readline()
            except SerialException as err:
                raise FarmduinoConnectionError(f"serial read failed: {err}") from err
            if raw:
                try:
                    return raw.decode("utf-8", errors="replace").rstrip("\r\n")
                except UnicodeDecodeError:
                    return raw.decode("ascii", errors="replace").rstrip("\r\n")
            # No data yet; short sleep to avoid busy-wait when timeout is long.
            time.sleep(0.01)
        return None

    def send_command(
        self,
        code: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> GCodeResponse:
        """Send a command and block until the matching ``R02``/``R03`` arrives."""
        with self._lock:
            cfg = self._serial_cfg()
            timeout = (
                timeout_s
                if timeout_s is not None
                else float(cfg.get("command_timeout_s", 60.0))
            )
            queue_id = self._queue_id
            self._queue_id += 1

            cmd = build_command(code, params, queue_id=queue_id)
            ser = self._ensure_connected()
            log.debug("farmduino -> %s", cmd)
            try:
                ser.write((cmd + "\r\n").encode("ascii"))
            except SerialException as err:
                raise FarmduinoConnectionError(f"serial write failed: {err}") from err

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = self._read_line(deadline)
                if line is None:
                    break
                log.debug("farmduino <- %s", line)
                resp = parse_response(line)

                if resp.code == "R88":
                    raise FarmduinoCommandError(
                        "firmware reports config not approved; call approve_config() first",
                        response_code="R88",
                        error_value=15,
                    )

                if resp.queue_id is not None and resp.queue_id != queue_id:
                    # Async or stale response; log and keep waiting.
                    log.debug("farmduino: ignoring response for queue %s", resp.queue_id)
                    continue

                if resp.code in {"R02", "R03"}:
                    if resp.code == "R03":
                        err_value: int | None = None
                        try:
                            err_value = int(resp.params.get("V", "0"))
                        except ValueError:
                            pass
                        raise FarmduinoCommandError(
                            f"command {cmd!r} failed: {line!r}",
                            response_code=resp.code,
                            error_value=err_value,
                        )
                    return resp

                # R01, R04, R82, R41, R99, etc. are logged but don't finish the command.
                continue

            raise FarmduinoTimeoutError(
                f"command {cmd!r} timed out after {timeout}s"
            )

    def _send_and_expect(
        self,
        code: str,
        params: dict[str, Any] | None,
        expected_code: str,
        *,
        timeout_s: float | None = None,
    ) -> GCodeResponse:
        """Send a command and wait for a specific response code (e.g. R41, R82)."""
        with self._lock:
            cfg = self._serial_cfg()
            timeout = (
                timeout_s
                if timeout_s is not None
                else float(cfg.get("command_timeout_s", 60.0))
            )
            queue_id = self._queue_id
            self._queue_id += 1

            cmd = build_command(code, params, queue_id=queue_id)
            ser = self._ensure_connected()
            log.debug("farmduino -> %s", cmd)
            try:
                ser.write((cmd + "\r\n").encode("ascii"))
            except SerialException as err:
                raise FarmduinoConnectionError(f"serial write failed: {err}") from err

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = self._read_line(deadline)
                if line is None:
                    break
                log.debug("farmduino <- %s", line)
                resp = parse_response(line)

                if resp.queue_id is not None and resp.queue_id != queue_id:
                    log.debug("farmduino: ignoring response for queue %s", resp.queue_id)
                    continue

                if resp.code == expected_code:
                    return resp

                # R03 is still an error for any transaction.
                if resp.code == "R03":
                    err_value: int | None = None
                    try:
                        err_value = int(resp.params.get("V", "0"))
                    except ValueError:
                        pass
                    raise FarmduinoCommandError(
                        f"command {cmd!r} failed: {line!r}",
                        response_code=resp.code,
                        error_value=err_value,
                    )

                continue

            raise FarmduinoTimeoutError(
                f"command {cmd!r} timed out after {timeout}s"
            )

    # ------------------------------------------------------------------ high level

    def approve_config(self) -> None:
        """Send ``F22 P2 V1`` so the firmware accepts motion commands."""
        self.send_command("F22", {"P": 2, "V": 1}, timeout_s=5.0)

    def move(
        self,
        x: float,
        y: float,
        z: float,
        speed_mm_s: float | None = None,
    ) -> None:
        """Move to absolute coordinates (mm). Speed is in mm/s."""
        params: dict[str, Any] = {"X": x, "Y": y, "Z": z}
        speed_letters = {"x": "A", "y": "B", "z": "C"}
        for axis in ("x", "y", "z"):
            steps_per_mm = self._steps_per_mm.get(axis)
            max_steps_s = self._max_speed_steps_s.get(axis)
            if steps_per_mm is None or max_steps_s is None:
                continue
            target_mm_s = speed_mm_s
            if target_mm_s is None:
                target_mm_s = max_steps_s / steps_per_mm
            target_steps_s = int(target_mm_s * steps_per_mm)
            cap = int(max_steps_s)
            if target_steps_s > cap:
                target_steps_s = cap
            params[speed_letters[axis]] = target_steps_s
        self.send_command("G00", params, timeout_s=self._serial_cfg().get("move_timeout_s", 300.0))

    def home(self, axis: str = "all") -> None:
        """Home one axis or all axes."""
        axis = axis.lower()
        if axis == "all":
            self.send_command("G28", timeout_s=self._serial_cfg().get("move_timeout_s", 300.0))
            return
        mapping = {"x": "F11", "y": "F12", "z": "F13"}
        if axis not in mapping:
            raise FarmduinoError(f"unknown axis {axis!r}")
        self.send_command(mapping[axis], timeout_s=self._serial_cfg().get("move_timeout_s", 300.0))

    def set_home(self, axis: str = "all") -> None:
        """Set the current position as zero for the given axis/axes."""
        axis = axis.lower()
        flags = {"x": 0, "y": 0, "z": 0}
        if axis == "all":
            flags = {"x": 1, "y": 1, "z": 1}
        elif axis in flags:
            flags[axis] = 1
        else:
            raise FarmduinoError(f"unknown axis {axis!r}")
        self.send_command("F84", {k.upper(): v for k, v in flags.items()})

    def read_pin(self, pin: int, mode: str = "digital") -> int:
        """Read a pin. Returns an int (digital 0/1 or analog 0-1023)."""
        resp = self._send_and_expect("F42", {"P": pin, "M": _mode_int(mode)}, "R41")
        if resp.code != "R41":
            raise FarmduinoCommandError(
                f"unexpected response to read_pin: {resp.raw!r}",
                response_code=resp.code,
            )
        try:
            return int(resp.params.get("V", "0"))
        except ValueError as err:
            raise FarmduinoError(f"could not parse pin value from {resp.raw!r}") from err

    def write_pin(self, pin: int, value: int, mode: str = "digital") -> None:
        """Set a pin value."""
        self.send_command(
            "F41",
            {"P": pin, "V": int(value), "M": _mode_int(mode)},
        )

    def write_pin_timed(
        self, pin: int, value: int, seconds: float, mode: str = "digital"
    ) -> None:
        """Set a pin value, wait ``seconds``, then set it to 0."""
        ms = max(0, int(seconds * 1000))
        self.send_command(
            "F44",
            {"P": pin, "V": int(value), "W": 0, "T": ms, "M": _mode_int(mode)},
        )

    def get_position(self) -> dict[str, float]:
        """Query the current position (mm)."""
        resp = self._send_and_expect("F82", None, "R82")
        if resp.code != "R82":
            raise FarmduinoCommandError(
                f"unexpected response to get_position: {resp.raw!r}",
                response_code=resp.code,
            )
        result: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            letter = _axis_letter(axis)
            try:
                result[axis] = float(resp.params.get(letter, "0"))
            except ValueError as err:
                raise FarmduinoError(f"bad position value in {resp.raw!r}") from err
        return result

    def get_endstops(self) -> dict[str, tuple[int, int]]:
        """Query endstop states: ``{'x': (xa, xb), ...}``."""
        resp = self._send_and_expect("F81", None, "R81")
        if resp.code != "R81":
            raise FarmduinoCommandError(
                f"unexpected response to get_endstops: {resp.raw!r}",
                response_code=resp.code,
            )
        result: dict[str, tuple[int, int]] = {}
        for axis in ("x", "y", "z"):
            a = int(resp.params.get(f"{axis.upper()}A", "0") or "0")
            b = int(resp.params.get(f"{axis.upper()}B", "0") or "0")
            result[axis] = (a, b)
        return result

    def e_stop(self) -> None:
        """Trigger the emergency stop."""
        with self._lock:
            ser = self._ensure_connected()
            try:
                ser.write(b"E\r\n")
            except SerialException as err:
                raise FarmduinoConnectionError(f"e-stop write failed: {err}") from err

    def abort(self) -> None:
        """Abort the current movement."""
        with self._lock:
            ser = self._ensure_connected()
            try:
                ser.write(b"@\r\n")
            except SerialException as err:
                raise FarmduinoConnectionError(f"abort write failed: {err}") from err

    def reset_emergency_stop(self) -> None:
        """Unlock after an emergency stop (``F09``)."""
        self.send_command("F09")


__all__ = [
    "FarmduinoSerial",
    "GCodeResponse",
    "FarmduinoError",
    "FarmduinoConnectionError",
    "FarmduinoTimeoutError",
    "FarmduinoCommandError",
    "build_command",
    "parse_response",
]
