"""Local FarmBot backend.

Translates TWFarmBot actions into the TWFarmbotOS HTTP API via
``farmbot_gateway``.
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

log = logging.getLogger(__name__)


class FarmBotBackend:
    def __init__(self) -> None:
        self._images_cache: list[dict[str, Any]] = []
        self._images_cached_at = 0.0
        self._images_lock = Lock()
        self._photo_baseline_ids: set[Any] = set()

    @property
    def pump_pin(self) -> int:
        from watering_service import _pump_pin

        return _pump_pin()

    def _bot(self) -> Any:
        from farmbot_gateway import get_farmbot

        return get_farmbot()

    def water(self, seconds: float) -> None:
        log.info("farmbot: water seconds=%s", seconds)
        self._bot().water_on()
        time.sleep(float(seconds))
        self._bot().water_off()

    def move(self, x: float, y: float, z: float, speed: float | None = None) -> None:
        log.info("farmbot: move x=%s y=%s z=%s speed=%s", x, y, z, speed)
        self._bot().move(x, y, z, speed=speed)

    def find_home(self, axis: str = "all", speed: float = 100) -> None:
        log.info("farmbot: find_home axis=%s", axis)
        self._bot().find_home(axis=axis, speed=speed)

    def set_home(self, axis: str = "all") -> None:
        log.info("farmbot: set_home axis=%s", axis)
        self._bot().set_home(axis=axis)

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
        return self.get_xyz()

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
        del refresh
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
        self._bot().e_stop()

    def unlock(self) -> None:
        log.info("farmbot: unlock")
        self._bot().unlock()

    def reboot(self) -> None:
        log.warning("farmbot: reboot unsupported on local OS")


backend: FarmBotBackend = FarmBotBackend()
