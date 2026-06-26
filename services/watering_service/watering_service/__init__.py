"""watering_service — orchestrates a manual watering action.

Flow: build Action → safety_service.validate → backend.water.
The backend is ``services/watering_service/watering_service/backends/farmbot.py``
— the only place that translates our vocabulary into ``farmbot-py`` calls.

Config is loaded from ``configs/dev.yaml`` (path overridable via
``TWFB_CONFIG``). Env vars override file values so ops can override
without editing YAML — e.g. ``FARMBOT_PUMP_PIN=8``.
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


def _pump_pin() -> int:
    """Return the pump pin from config, with an optional env override."""
    cfg = _load_yaml_config().get("watering", {}) or {}
    return int(os.getenv("FARMBOT_PUMP_PIN", cfg.get("pump_pin", 7)))


def _load_backend() -> Any:
    name = os.getenv("WATERING_BACKEND", "farmbot")
    module = import_module(f"watering_service.backends.{name}")
    return module.backend


_backend: Any | None = None


def get_backend() -> Any:
    global _backend
    if _backend is None:
        _backend = _load_backend()
        log.info("watering backend loaded: %s", type(_backend).__name__)
    return _backend


def water(seconds: float) -> Action:
    """Validate then execute a watering action. Returns the executed Action."""
    from twfarmbot_core.domain.action import Action

    action = Action(kind="water", params={"seconds": seconds})
    try:
        safety_validate(action)
    except UnsafeActionError:
        log.warning("safety: rejected %s", action)
        raise

    backend = get_backend()
    log.info("watering: pump seconds=%s", seconds)
    backend.water(seconds)
    return action


__all__ = [
    "_pump_pin",
    "get_backend",
    "water",
]
