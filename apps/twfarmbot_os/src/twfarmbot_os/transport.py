"""Serial or simulated Farmduino transport."""

from __future__ import annotations

import logging
import os
import threading
import time
from queue import Empty, Queue

from farmduino import Report, parse_report

from twfarmbot_os.simulate import SimulatedFarmduino

log = logging.getLogger(__name__)

_DONE_CODES = {2, 3, 9, 20, 86, 87}


class TransportError(RuntimeError):
    pass


class BaseTransport:
    def request(self, gcode: str, timeout_s: float) -> list[Report]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class SimulatedTransport(BaseTransport):
    def __init__(self) -> None:
        self.mcu = SimulatedFarmduino()

    def request(self, gcode: str, timeout_s: float) -> list[Report]:
        del timeout_s
        lines = self.mcu.process(gcode)
        return [r for line in lines if (r := parse_report(line)) is not None]


class SerialTransport(BaseTransport):
    def __init__(self, path: str, baud: int = 115200) -> None:
        try:
            import serial
        except ImportError as err:
            raise TransportError("pyserial is required for serial mode") from err
        # Avoid toggling DTR on open — that resets the Farmduino.
        self._port = serial.Serial(
            path,
            baudrate=baud,
            timeout=0.2,
            dsrdtr=False,
            rtscts=False,
        )
        self._port.dtr = False
        self._inbox: Queue[str] = Queue()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.wait_until_idle(timeout_s=30.0)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._port.readline()
            except Exception:  # noqa: BLE001
                if self._stop.is_set():
                    return
                continue
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                log.debug("farmduino << %s", line)
                self._inbox.put(line)

    def _drain(self) -> None:
        while True:
            try:
                self._inbox.get_nowait()
            except Empty:
                return

    def wait_until_idle(self, timeout_s: float = 30.0) -> None:
        """Wait for Farmduino boot to finish before sending commands."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                line = self._inbox.get(timeout=min(0.25, remaining))
            except Empty:
                continue
            if line.startswith("R00") or "STARTUP COMPLETE" in line:
                log.info("Farmduino boot complete (%s)", line)
                self._drain()
                return
        log.warning(
            "Farmduino boot idle wait timed out after %.0fs; continuing",
            timeout_s,
        )
        self._drain()

    def request(self, gcode: str, timeout_s: float) -> list[Report]:
        self._drain()
        payload = gcode.strip() + "\r\n"
        log.debug("farmduino >> %s", gcode)
        self._port.write(payload.encode("ascii", errors="replace"))
        deadline = time.monotonic() + timeout_s
        reports: list[Report] = []
        saw_r01 = False
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                line = self._inbox.get(timeout=min(0.25, remaining))
            except Empty:
                continue
            report = parse_report(line)
            if report is None:
                continue
            reports.append(report)
            if report.number == 1:
                saw_r01 = True
            if report.number == 88 and not saw_r01:
                # Boot-time "no config yet" — keep reading for this command.
                continue
            if report.number in _DONE_CODES or report.number == 88:
                return reports
        raise TransportError(f"timeout waiting for R02 after {gcode!r}")

    def close(self) -> None:
        self._stop.set()
        try:
            self._port.close()
        except Exception:  # noqa: BLE001
            pass


def build_transport() -> BaseTransport:
    mode = os.getenv("FARMBOT_TRANSPORT", "sim").strip().lower()
    if mode in {"sim", "simulate", "simulation"}:
        log.info("TWFarmbotOS using simulated Farmduino")
        return SimulatedTransport()
    path = os.getenv("FARMBOT_SERIAL_PATH", "/dev/ttyACM0")
    baud = int(os.getenv("FARMBOT_SERIAL_BAUD", "115200"))
    log.info("TWFarmbotOS opening serial %s @ %s", path, baud)
    return SerialTransport(path, baud)
