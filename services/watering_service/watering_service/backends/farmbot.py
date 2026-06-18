"""Real FarmBot backend.

Drives the FarmBot over WiFi via ``farmbot_gateway``. Exposes a flat
vocabulary that matches our ``Action.kind`` namespace — handlers stay
tiny (one method call each).

The ``pins`` map (loaded by ``watering_service``) tells us which
FarmBot peripheral pin controls each bed's valve. To add a new bed,
just add a row to ``configs/dev.yaml`` (or set ``FARMBOT_PIN_bN=...``).
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

log = logging.getLogger(__name__)


class FarmBotBackend:
    """Thin pass-through to ``farmbot_gateway.get_farmbot()`` plus our vocab."""

    def __init__(self) -> None:
        # Do not capture get_farmbot — resolve lazily so tests can monkeypatch
        # farmbot_gateway.get_farmbot and have it take effect immediately.
        self._images_cache: list[dict[str, Any]] = []
        self._images_cached_at = 0.0
        self._images_lock = Lock()

    @property
    def pins(self) -> dict[str, int]:
        # Read fresh each call so config edits / env overrides take effect.
        from watering_service import _pin_map

        return _pin_map()

    def _bot(self) -> Any:
        from farmbot_gateway import get_farmbot

        return get_farmbot()

    # -------- Watering ----------------------------------------------------

    def water(self, bed_id: str, seconds: float) -> None:
        pin = self.pins.get(bed_id)
        if pin is None:
            raise ValueError(
                f"no valve pin configured for bed {bed_id!r}; "
                f"known beds: {sorted(self.pins)}"
            )
        bot = self._bot()
        log.info("farmbot: open valve pin=%s bed=%s seconds=%s", pin, bed_id, seconds)
        bot.control_peripheral(peripheral_name=str(pin), value=1, mode="digital")
        try:
            time.sleep(seconds)
        finally:
            bot.control_peripheral(peripheral_name=str(pin), value=0, mode="digital")
            log.info("farmbot: closed valve pin=%s", pin)

    # The WaterSource protocol used by watering_service.water_bed:
    def open(self, bed_id: str) -> None:
        pin = self.pins[bed_id]
        self._bot().control_peripheral(peripheral_name=str(pin), value=1, mode="digital")

    def close(self, bed_id: str) -> None:
        pin = self.pins[bed_id]
        self._bot().control_peripheral(peripheral_name=str(pin), value=0, mode="digital")

    # -------- Movement ----------------------------------------------------

    def move(self, x: float, y: float, z: float, speed: float | None = None) -> None:
        kwargs: dict[str, Any] = {"x": x, "y": y, "z": z}
        if speed is not None:
            kwargs["speed"] = speed
        log.info("farmbot: move x=%s y=%s z=%s speed=%s", x, y, z, speed)
        self._bot().move(**kwargs)

    def find_home(self, axis: str = "all", speed: float = 100) -> None:
        log.info("farmbot: find_home axis=%s", axis)
        self._bot().find_home(axis=axis, speed=speed)

    def set_home(self, axis: str = "all") -> None:
        log.info("farmbot: set_home axis=%s", axis)
        self._bot().set_home(axis=axis)

    # -------- Pins / peripherals -----------------------------------------

    def read_pin(self, pin: int, mode: str = "digital") -> Any:
        log.info("farmbot: read_pin pin=%s mode=%s", pin, mode)
        return self._bot().read_pin(pin_number=pin, mode=mode)

    def write_pin(self, pin: int, value: int, mode: str = "digital") -> None:
        log.info("farmbot: write_pin pin=%s value=%s mode=%s", pin, value, mode)
        self._bot().write_pin(pin_number=pin, value=value, mode=mode)

    def control_peripheral(self, peripheral_name: str, value: int, mode: str | None = None) -> None:
        log.info("farmbot: control_peripheral %s=%s mode=%s", peripheral_name, value, mode)
        self._bot().control_peripheral(peripheral_name=peripheral_name, value=value, mode=mode)

    # -------- Tools -------------------------------------------------------

    def mount_tool(self, tool_name: str) -> None:
        log.info("farmbot: mount_tool %s", tool_name)
        self._bot().mount_tool(tool_name)

    def dismount_tool(self) -> None:
        log.info("farmbot: dismount_tool")
        self._bot().dismount_tool()

    # -------- Sensors / inspection ---------------------------------------

    def get_xyz(self) -> Any:
        bot = self._bot()
        cached_xyz = getattr(bot, "cached_xyz", None)
        return cached_xyz() if callable(cached_xyz) else bot.get_xyz()

    def refresh_xyz(self) -> Any:
        """Ask FarmBot for a fresh status tree, updating its local MQTT cache."""
        bot = self._bot()
        state = getattr(getattr(bot, "_fb", bot), "state", None)
        old_verbosity = getattr(state, "verbosity", None)
        old_json_printing = getattr(state, "json_printing", None)
        try:
            if state is not None:
                state.verbosity = 0
                state.json_printing = False
            return bot.get_xyz()
        finally:
            if state is not None:
                state.verbosity = old_verbosity
                state.json_printing = old_json_printing

    def get_last_messages(self) -> Any:
        """Return the FarmBot's ``state.last_messages`` (or ``None``)."""
        bot = self._bot()
        cached_messages = getattr(bot, "cached_last_messages", None)
        return cached_messages() if callable(cached_messages) else getattr(bot.state, "last_messages", None)

    def take_photo(self) -> None:
        log.info("farmbot: take_photo")
        self._bot().take_photo()

    def get_images(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return newest image records, caching the expensive cloud API call."""
        with self._images_lock:
            if self._images_cache and time.monotonic() - self._images_cached_at < 60:
                return self._images_cache[:limit]

            bot = self._bot()
            api_get = getattr(getattr(bot, "info", None), "api_get", None)
            images = (
                api_get("images", data_print=False)
                if callable(api_get)
                else bot.api_get("images")
            )
            if not isinstance(images, list):
                log.warning("farmbot image API unavailable; serving cached gallery")
                return self._images_cache[:limit]

            self._images_cache = sorted(
                (image for image in images if isinstance(image, dict)),
                key=lambda image: image.get("created_at", ""),
                reverse=True,
            )
            self._images_cached_at = time.monotonic()
            return self._images_cache[:limit]

    def measure_soil_height(self) -> Any:
        log.info("farmbot: measure_soil_height")
        return self._bot().measure_soil_height()

    # -------- Feedback / control ----------------------------------------

    def send_message(self, message: str, message_type: str = "info", channels: list[str] | None = None) -> None:
        log.info("farmbot: send_message %s: %s", message_type, message)
        self._bot().send_message(message_str=message, message_type=message_type, channels=channels)

    def toast(self, message: str, message_type: str = "info") -> None:
        log.info("farmbot: toast %s: %s", message_type, message)
        self._bot().toast(message_str=message, message_type=message_type)

    def e_stop(self) -> None:
        log.warning("farmbot: EMERGENCY STOP")
        self._bot().e_stop()

    def reboot(self) -> None:
        log.warning("farmbot: reboot")
        self._bot().reboot()


backend: FarmBotBackend = FarmBotBackend()
