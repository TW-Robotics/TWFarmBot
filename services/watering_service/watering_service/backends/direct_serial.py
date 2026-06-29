"""Direct serial backend for Genesis v1.8 (Farmduino v1.32).

Drives the robot over USB serial (/dev/ttyACM0) using the stock Farmduino
firmware's G-code/F-code protocol. No cloud, no MQTT, no farmbot-py.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from farmbot_serial import (
    FarmduinoCommandError,
    FarmduinoConnectionError,
    FarmduinoSerial,
    FarmduinoTimeoutError,
)
from twfarmbot_core.config import load_yaml_config

log = logging.getLogger(__name__)


class DirectSerialBackend:
    """Local-only backend that talks directly to the Farmduino over USB serial."""

    def __init__(self) -> None:
        self._serial: FarmduinoSerial | None = None
        self._serial_lock = Lock()
        self._images_dir = Path("data/images")
        self._images_cache: list[dict[str, Any]] = []
        self._images_cached_at = 0.0
        self._images_lock = Lock()
        self._last_position: dict[str, float] | None = None

    # ------------------------------------------------------------------ config

    def _config(self) -> dict[str, Any]:
        return load_yaml_config()

    def _hw(self) -> dict[str, Any]:
        return self._config().get("hardware", {})

    @property
    def pump_pin(self) -> int:
        """Water peripheral pin from config, with legacy env override."""
        from watering_service import _pump_pin

        return _pump_pin()

    def _peripheral_pin(self, name: str) -> int:
        peripherals = self._hw().get("peripherals", {})
        entry = peripherals.get(name, {})
        pin = entry.get("pin")
        if pin is None:
            raise RuntimeError(f"no peripheral configured for {name!r}")
        return int(pin)

    def _camera_dir(self) -> Path:
        camera = self._hw().get("camera", {})
        return Path(camera.get("save_dir", "data/images"))

    # ------------------------------------------------------------------ serial

    def _ensure_serial(self) -> FarmduinoSerial:
        with self._serial_lock:
            if self._serial is None:
                self._serial = FarmduinoSerial()
            if not self._serial.connected:
                self._serial.connect()
            return self._serial

    def connected(self) -> bool:
        return self._serial is not None and self._serial.connected

    def connect(self) -> None:
        """Eagerly open the serial connection."""
        self._ensure_serial()

    # ------------------------------------------------------------------ motion

    def move(self, x: float, y: float, z: float, speed: float | None = None) -> None:
        log.info("direct: move x=%s y=%s z=%s speed=%s", x, y, z, speed)
        self._ensure_serial().move(x, y, z, speed_mm_s=speed)
        self._last_position = {"x": x, "y": y, "z": z}

    def find_home(self, axis: str = "all", speed: float = 100) -> None:
        log.info("direct: find_home axis=%s", axis)
        self._ensure_serial().home(axis=axis)

    def set_home(self, axis: str = "all") -> None:
        log.info("direct: set_home axis=%s", axis)
        self._ensure_serial().set_home(axis=axis)

    # ------------------------------------------------------------------ pins / peripherals

    def read_pin(self, pin: int, mode: str = "digital") -> Any:
        log.info("direct: read_pin pin=%s mode=%s", pin, mode)
        return self._ensure_serial().read_pin(pin, mode=mode)

    def write_pin(
        self,
        pin: int,
        value: int,
        mode: str = "digital",
        seconds: float | None = None,
    ) -> None:
        log.info("direct: write_pin pin=%s value=%s mode=%s", pin, value, mode)
        ser = self._ensure_serial()
        if value == 1 and seconds is not None and seconds > 0:
            ser.write_pin_timed(pin, value, seconds, mode=mode)
        else:
            ser.write_pin(pin, value, mode=mode)

    def control_peripheral(
        self, peripheral_name: str, value: int, mode: str | None = None
    ) -> None:
        log.info("direct: control_peripheral %s=%s", peripheral_name, value)
        pin = self._peripheral_pin(peripheral_name)
        cfg_mode = mode or self._hw().get("peripherals", {}).get(
            peripheral_name, {}
        ).get("mode", "digital")
        self.write_pin(pin, value, cfg_mode)

    # ------------------------------------------------------------------ watering

    def water(self, seconds: float) -> None:
        log.info("direct: water seconds=%s", seconds)
        self.write_pin(self.pump_pin, 1, "digital", seconds=seconds)

    # ------------------------------------------------------------------ sensors / state

    def get_xyz(self) -> Any:
        if self._last_position is not None:
            return dict(self._last_position)
        return self.refresh_xyz()

    def refresh_xyz(self) -> Any:
        pos = self._ensure_serial().get_position()
        self._last_position = pos
        return pos

    def get_status(self) -> dict[str, Any]:
        ser = self._ensure_serial()
        return {
            "position": ser.get_position(),
            "endstops": ser.get_endstops(),
            "steps_per_mm": ser.steps_per_mm(),
            "connected": True,
        }

    def get_last_messages(self) -> Any:
        # Local-only stack has no MQTT message queue.
        return []

    def measure_soil_height(self) -> Any:
        raise NotImplementedError(
            "soil height measurement not implemented for direct serial backend"
        )

    # ------------------------------------------------------------------ camera

    def take_photo(self) -> None:
        save_dir = self._camera_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = save_dir / f"twfarmbot_{timestamp}.jpg"

        try:
            import cv2
        except ImportError as err:
            raise RuntimeError("opencv is required for camera capture") from err

        camera_index = int(self._hw().get("camera", {}).get("index", 0))
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {camera_index}")
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("camera capture failed")
            cv2.imwrite(str(path), frame)
            log.info("direct: saved photo %s", path)
        finally:
            cap.release()

    def get_images(
        self, limit: int = 10, *, refresh: bool = False
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        with self._images_lock:
            age = time.monotonic() - self._images_cached_at
            if refresh or not self._images_cache or age >= 10:
                self._images_cache = self._list_images()
                self._images_cached_at = time.monotonic()
            return self._images_cache[:limit]

    def _list_images(self) -> list[dict[str, Any]]:
        save_dir = self._camera_dir()
        if not save_dir.exists():
            return []
        files = sorted(
            (
                p
                for p in save_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [
            {
                "id": str(path.name),
                "created_at": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "attachment_url": f"/static/images/{path.name}",
                "path": str(path),
            }
            for path in files
        ]

    # ------------------------------------------------------------------ tools

    def mount_tool(self, tool_name: str) -> None:
        cfg = self._hw().get("tools", {})
        sequence = cfg.get("mount", {}).get(tool_name)
        if sequence is None:
            raise NotImplementedError(
                f"mount_tool {tool_name!r} not configured; "
                "add hardware.tools.mount.{tool_name} to configs/dev.yaml"
            )
        self._run_tool_sequence(sequence)

    def dismount_tool(self) -> None:
        cfg = self._hw().get("tools", {})
        sequence = cfg.get("dismount")
        if sequence is None:
            raise NotImplementedError(
                "dismount_tool not configured; add hardware.tools.dismount to configs/dev.yaml"
            )
        self._run_tool_sequence(sequence)

    def _run_tool_sequence(self, sequence: list[dict[str, Any]]) -> None:
        for step in sequence:
            kind = step.get("kind")
            if kind == "move":
                self.move(step["x"], step["y"], step["z"], step.get("speed"))
            elif kind == "write_pin":
                self.write_pin(
                    int(step["pin"]), int(step["value"]), step.get("mode", "digital")
                )
            elif kind == "sleep":
                time.sleep(float(step["seconds"]))
            else:
                log.warning("direct: unknown tool sequence step %s", kind)

    # ------------------------------------------------------------------ feedback / control

    def send_message(
        self,
        message: str,
        message_type: str = "info",
        channels: list[str] | None = None,
    ) -> None:
        log.log(
            logging.WARNING if message_type == "error" else logging.INFO,
            "direct: message [%s] %s (channels=%s)",
            message_type,
            message,
            channels,
        )

    def toast(self, message: str, message_type: str = "info") -> None:
        self.send_message(message, message_type=message_type)

    def e_stop(self) -> None:
        log.warning("direct: EMERGENCY STOP")
        try:
            self._ensure_serial().e_stop()
        except (FarmduinoConnectionError, FarmduinoTimeoutError, FarmduinoCommandError):
            log.exception("direct: e-stop failed")
            raise

    def reboot(self) -> None:
        log.warning("direct: reboot not supported in local serial mode")


backend: DirectSerialBackend = DirectSerialBackend()

__all__ = ["DirectSerialBackend", "backend"]
