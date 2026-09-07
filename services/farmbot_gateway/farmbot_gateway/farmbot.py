"""Local FarmBot hardware backend.

Translates high-level robot operations into the TWFarmbotOS HTTP API via
``FarmBotLink``. Includes the Homing Guard to track absolute hardware datum.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from twfarmbot_core.config import load_yaml_config

log = logging.getLogger(__name__)


class FarmBotBackend:
    def __init__(self) -> None:
        self._images_cache: list[dict[str, Any]] = []
        self._images_cached_at = 0.0
        self._images_lock = Lock()
        self._photo_baseline_ids: set[Any] = set()
        self._is_homed: bool = False
        self._last_homed_at: str | None = None
        self._homing_lock = Lock()

    @property
    def pump_pin(self) -> int:
        cfg = load_yaml_config().get("watering", {}) or {}
        return int(os.getenv("FARMBOT_PUMP_PIN", cfg.get("pump_pin", 7)))

    def _bot(self) -> Any:
        import farmbot_gateway

        return farmbot_gateway.get_farmbot()

    def is_homed(self) -> bool:
        """Return True if the robot has been homed to its physical hardware endstops."""
        return self._is_homed

    def last_homed_at(self) -> str | None:
        """Return the ISO timestamp of the last successful hardware homing."""
        return self._last_homed_at

    def ensure_homed(self, speed: float = 100) -> None:
        """Run find_home('all') if the robot is unverified / unhomed after boot."""
        with self._homing_lock:
            if not self._is_homed:
                log.warning(
                    "FarmBot is unhomed after restart; calibrating to hardware endstops..."
                )
                self.find_home(axis="all", speed=speed)

    def _save_pose_snapshot(self) -> None:
        try:
            pose = self.get_xyz()
            if isinstance(pose, dict):
                data = {
                    "x": pose.get("x", 0.0),
                    "y": pose.get("y", 0.0),
                    "z": pose.get("z", 0.0),
                    "is_homed": self._is_homed,
                    "last_homed_at": self._last_homed_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                data_dir = Path(os.getenv("TWFB_DATA_DIR", Path.cwd() / "data"))
                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / "gantry_pose.json").write_text(
                    json.dumps(data, indent=2), encoding="utf-8"
                )
        except Exception:  # noqa: BLE001
            pass

    def water(self, seconds: float) -> None:
        log.info("farmbot: water seconds=%s", seconds)
        self._bot().water_on()
        time.sleep(float(seconds))
        self._bot().water_off()

    def move(self, x: float, y: float, z: float, speed: float | None = None) -> None:
        self.ensure_homed()
        log.info("farmbot: move x=%s y=%s z=%s speed=%s", x, y, z, speed)
        self._bot().move(x, y, z, speed=speed)
        self._save_pose_snapshot()

    def move_axis(self, axis: str, distance: float, speed: float | None = None) -> None:
        self.ensure_homed()
        log.info(
            "farmbot: move_axis axis=%s distance=%s speed=%s",
            axis,
            distance,
            speed,
        )
        self._bot().move_axis(axis, distance, speed=speed)
        self._save_pose_snapshot()

    def find_home(self, axis: str = "all", speed: float = 100) -> None:
        log.info("farmbot: find_home axis=%s", axis)
        self._bot().find_home(axis=axis, speed=speed)
        if axis.lower() == "all":
            self._is_homed = True
            self._last_homed_at = datetime.now(timezone.utc).isoformat()
            self._save_pose_snapshot()

    def set_home(self, axis: str = "all") -> None:
        log.info("farmbot: set_home axis=%s", axis)
        self._bot().set_home(axis=axis)
        if axis.lower() == "all":
            self._is_homed = True
            self._last_homed_at = datetime.now(timezone.utc).isoformat()
            self._save_pose_snapshot()

    def read_pin(self, pin: int, mode: str = "digital") -> Any:
        return self._bot().read_pin(pin, mode)

    def write_pin(
        self,
        pin: int,
        value: int,
        mode: str = "digital",
        seconds: float | None = None,
    ) -> None:
        log.info("farmbot: write_pin pin=%s value=%s", pin, value)
        self._bot().write_pin(pin, value, mode)
        if value and seconds is not None and seconds > 0:
            time.sleep(float(seconds))
            self._bot().write_pin(pin, 0, mode)

    def control_peripheral(
        self, peripheral_name: str, value: int, mode: str | None = None
    ) -> None:
        log.info(
            "farmbot: control_peripheral unsupported %s=%s", peripheral_name, value
        )

    def mount_tool(self, tool_name: str) -> None:
        log.info("farmbot: mount_tool unsupported %s", tool_name)

    def dismount_tool(self) -> None:
        log.info("farmbot: dismount_tool unsupported")

    def get_xyz(self) -> Any:
        return self._bot().get_xyz()

    def refresh_xyz(self) -> Any:
        state = self._bot().refresh_state()
        return {axis: float(state.get(axis, 0) or 0) for axis in ("x", "y", "z")}

    def get_last_messages(self) -> Any:
        state = self._bot().get_state()
        message = state.get("message")
        return [message] if message else []

    def take_photo(self) -> None:
        photo = self._bot().take_photo()
        info = photo.get("photo") if isinstance(photo, dict) else None
        if isinstance(info, dict):
            with self._images_lock:
                self._images_cache.insert(0, info)

    def wait_for_new_photo(self) -> bool:
        return True

    def get_images(
        self, limit: int = 10, *, refresh: bool = False
    ) -> list[dict[str, Any]]:
        images = []
        try:
            raw = self._bot().get_images(limit)
            if isinstance(raw, list):
                images = raw
        except Exception:  # noqa: BLE001
            images = []
        if images:
            return images[:limit]
        return self._images_cache[:limit]

    def measure_soil_height(self) -> Any:
        log.info("farmbot: measure_soil_height unsupported")
        return None

    def send_message(
        self,
        message: str,
        message_type: str = "info",
        channels: list[str] | None = None,
    ) -> None:
        log.info("farmbot: %s %s", message_type, message)

    def toast(self, message: str, message_type: str = "info") -> None:
        log.info("farmbot: toast %s: %s", message_type, message)

    def e_stop(self) -> None:
        log.warning("farmbot: EMERGENCY STOP")
        self._is_homed = False
        self._bot().e_stop()
        self._save_pose_snapshot()

    def unlock(self) -> None:
        log.info("farmbot: unlock")
        self._bot().unlock()

    def reboot(self) -> None:
        log.warning("farmbot: reboot unsupported on local OS")


backend: FarmBotBackend = FarmBotBackend()
