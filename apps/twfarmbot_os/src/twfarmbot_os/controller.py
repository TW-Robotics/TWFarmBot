"""Serializes Farmduino commands and keeps last-known status from R codes."""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any

from farmduino import Report, encode
from farmduino.protocol import PARAM_CONFIG_OK

from twfarmbot_os.transport import (
    BaseTransport,
    SerialTransport,
    SimulatedTransport,
    TransportError,
)

log = logging.getLogger(__name__)

_HOME = {"x": "F11", "y": "F12", "z": "F13", "all": "G28"}
_CAL = {"x": "F14", "y": "F15", "z": "F16"}
_CONFIG_APPROVE_RETRIES = 3


class FarmduinoController:
    def __init__(self, transport: BaseTransport) -> None:
        self.transport = transport
        self._lock = Lock()
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.locked = False
        self.config_ok = False
        self.pins: dict[int, float] = {}
        self.last_reports: list[str] = []
        self.message = "ready"
        if isinstance(transport, SerialTransport):
            self._ensure_config_approved()

    def _config_is_approved(self) -> bool:
        try:
            reports = self.execute(encode("F21", P=PARAM_CONFIG_OK), timeout_s=10)
        except TransportError:
            return False
        for report in reports:
            if report.number == 21 and int(report.params.get("P", -1)) == PARAM_CONFIG_OK:
                return report.params.get("V", 0) >= 1
        return False

    def _ensure_config_approved(self) -> None:
        if os.getenv("FARMBOT_APPROVE_CONFIG", "1") in {"0", "false"}:
            self.config_ok = True
            return
        if self._config_is_approved():
            self.config_ok = True
            self.message = "config already approved"
            log.info("Farmduino config already approved (P2=1)")
            return
        cmd = encode("F22", P=PARAM_CONFIG_OK, V=1)
        for attempt in range(1, _CONFIG_APPROVE_RETRIES + 1):
            try:
                self.execute(cmd, timeout_s=15)
            except TransportError as err:
                log.warning(
                    "config approve attempt %s/%s failed: %s",
                    attempt,
                    _CONFIG_APPROVE_RETRIES,
                    err,
                )
            else:
                if self._config_is_approved():
                    self.config_ok = True
                    self.message = "config approved"
                    log.info("Farmduino config approved (%s)", cmd)
                    return
            time.sleep(0.5 * attempt)
        msg = f"Farmduino config not approved after {cmd}"
        log.error(msg)
        if os.getenv("FARMBOT_REQUIRE_CONFIG", "1") not in {"0", "false"}:
            raise TransportError(msg)

    def snapshot(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "stopped": self.locked,
            "config_ok": self.config_ok,
            "pins": dict(self.pins),
            "lastCommand": self.last_reports[-1] if self.last_reports else "",
            "message": self.message,
            "reports": list(self.last_reports),
        }

    def execute(self, gcode: str, timeout_s: float = 120.0) -> list[Report]:
        with self._lock:
            reports = self.transport.request(gcode, timeout_s)
            self.last_reports = [r.raw for r in reports]
            self._apply(reports, gcode)
            if any(
                r.number == 87 for r in reports
            ) and not gcode.strip().upper().startswith("E"):
                self.message = "R87 emergency lock"
                raise TransportError("R87 emergency lock")
            failed = next((r for r in reports if r.number in {3, 9, 86, 88}), None)
            if failed:
                self.message = failed.raw
                raise TransportError(failed.raw)
            self.message = gcode
            return reports

    def _apply(self, reports: list[Report], gcode: str) -> None:
        head = gcode.strip().upper()
        if head == "E" or head.startswith("E "):
            self.locked = True
        elif head.startswith("F09"):
            self.locked = False
        for report in reports:
            if report.number == 82:
                self.x = report.params.get("X", self.x)
                self.y = report.params.get("Y", self.y)
                self.z = report.params.get("Z", self.z)
            elif report.number == 41:
                pin = int(report.params.get("P", 0))
                self.pins[pin] = report.params.get("V", 0)
            elif report.number == 87:
                self.locked = True
        if isinstance(self.transport, SimulatedTransport):
            self.locked = self.transport.mcu.locked
            self.x = self.transport.mcu.x
            self.y = self.transport.mcu.y
            self.z = self.transport.mcu.z

    def move_absolute(
        self,
        x: float | None,
        y: float | None,
        z: float | None,
        speed: float | None = None,
    ) -> list[Report]:
        params: dict[str, float] = {}
        if x is not None:
            params["X"] = x
        if y is not None:
            params["Y"] = y
        if z is not None:
            params["Z"] = z
        if speed is not None:
            params["A"] = speed
            params["B"] = speed
            params["C"] = speed
        return self.execute(encode("G00", **params))

    def move_axis(self, axis: str, distance: float) -> list[Report]:
        axis = axis.lower()
        current = {"x": self.x, "y": self.y, "z": self.z}
        current[axis] = current[axis] + float(distance)
        return self.move_absolute(current["x"], current["y"], current["z"])

    def find_home(self, axis: str = "all") -> list[Report]:
        code = _HOME.get(axis.lower())
        if code is None:
            raise TransportError(f"unknown axis {axis}")
        return self.execute(code)

    def calibrate(self, axis: str) -> list[Report]:
        code = _CAL.get(axis.lower())
        if code is None:
            raise TransportError(f"unknown axis {axis}")
        return self.execute(code)

    def set_zero(self, axis: str = "all") -> list[Report]:
        flags = {"X": 0, "Y": 0, "Z": 0}
        if axis == "all":
            flags = {"X": 1, "Y": 1, "Z": 1}
        else:
            flags[axis.upper()] = 1
        return self.execute(encode("F84", **flags))

    def e_stop(self) -> list[Report]:
        return self.execute("E", timeout_s=5)

    def unlock(self) -> list[Report]:
        return self.execute("F09", timeout_s=5)

    def write_pin(self, pin: int, value: float, mode: int = 0) -> list[Report]:
        return self.execute(encode("F41", P=pin, V=value, M=mode), timeout_s=10)

    def read_pin(self, pin: int, mode: int = 0) -> float:
        reports = self.execute(encode("F42", P=pin, M=mode), timeout_s=10)
        for report in reversed(reports):
            if report.number == 41:
                return report.params.get("V", 0.0)
        return self.pins.get(pin, 0.0)

    def set_pin_mode(self, pin: int, mode: int) -> list[Report]:
        return self.execute(encode("F43", P=pin, M=mode), timeout_s=10)

    def pulse_pin(
        self, pin: int, value: float, wait_ms: float, then: float, mode: int = 0
    ) -> list[Report]:
        return self.execute(
            encode("F44", P=pin, V=value, W=then, T=wait_ms, M=mode),
            timeout_s=max(10.0, wait_ms / 1000.0 + 5),
        )

    def servo(self, pin: int, angle: float) -> list[Report]:
        return self.execute(encode("F61", P=pin, V=angle), timeout_s=10)

    def endstops(self) -> list[Report]:
        return self.execute("F81", timeout_s=5)

    def refresh_position(self) -> list[Report]:
        return self.execute("F82", timeout_s=5)
