"""watering_service — start the bed pump through safety + a backend.

Default backend is the Shelly plug (one pump, all sprayers). The plug's own
1-minute auto-off is the duration. ``WATERING_BACKEND=farmbot`` still drives
a GPIO pin for the old solenoid path.
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

from safety_service import UnsafeActionError, validate as safety_validate
from twfarmbot_core.config import load_yaml_config

if TYPE_CHECKING:
    from twfarmbot_core.domain.action import Action

log = logging.getLogger(__name__)


def _load_yaml_config() -> dict:
    return load_yaml_config()


def _watering_cfg() -> dict[str, Any]:
    return _load_yaml_config().get("watering", {}) or {}


def _pump_pin() -> int:
    """Return the pump pin from config, with an optional env override."""
    return int(os.getenv("FARMBOT_PUMP_PIN", _watering_cfg().get("pump_pin", 7)))


def _cycle_seconds() -> float:
    """Duration used only by the FarmBot pin backend."""
    return float(os.getenv("FARMBOT_WATER_SECONDS", _watering_cfg().get("cycle_seconds", 60)))


def _backend_name() -> str:
    return (os.getenv("WATERING_BACKEND") or str(_watering_cfg().get("backend") or "farmbot")).lower()


def _load_backend() -> Any:
    module = import_module(f"watering_service.backends.{_backend_name()}")
    return module.backend


_backend: Any | None = None


def get_backend() -> Any:
    global _backend
    if _backend is None:
        _backend = _load_backend()
        log.info("watering backend loaded: %s", type(_backend).__name__)
    return _backend


def water(seconds: float | None = None) -> Action:
    """Validate then start a watering cycle. Returns the executed Action.

    The Shelly plug already cuts the pump after 1 minute. ``seconds`` is only
    used by the FarmBot pin backend.
    """
    from twfarmbot_core.domain.action import Action

    params: dict[str, Any] = {}
    if seconds is not None:
        params["seconds"] = seconds
    action = Action(kind="water", params=params)
    try:
        safety_validate(action)
    except UnsafeActionError:
        log.warning("safety: rejected %s", action)
        raise

    backend = get_backend()
    duration = _cycle_seconds() if seconds is None else float(seconds)
    log.info("watering: backend=%s duration=%s", type(backend).__name__, duration)
    backend.water(duration)
    return action


def stop() -> None:
    """Cut the pump immediately (ESTOP). No-op if the backend has no stop."""
    off = getattr(get_backend(), "stop", None)
    if callable(off):
        off()


def pump_status() -> dict[str, Any]:
    backend = get_backend()
    out: dict[str, Any] = {"backend": type(backend).__name__}
    getter = getattr(backend, "status", None)
    if callable(getter):
        out.update(getter())
    return out


def set_pump(on: bool) -> dict[str, Any]:
    """Manual on/off for the I/O tab. On stays on until off or ESTOP."""
    backend = get_backend()
    if on:
        turn_on = getattr(backend, "turn_on", None)
        if callable(turn_on):
            turn_on()
        else:
            backend.water(_cycle_seconds())
    else:
        stop()
    return pump_status()


__all__ = [
    "_pump_pin",
    "get_backend",
    "pump_status",
    "set_pump",
    "stop",
    "water",
]
