"""watering_service — orchestrates a manual watering action.

Flow: build Action → safety_service.validate → backend.open → sleep → backend.close.
The backend is ``services/watering_service/watering_service/backends/farmbot.py``
— the only place that translates our vocabulary into ``farmbot-py`` calls.

Config is loaded from ``configs/dev.yaml`` (path overridable via
``TWFB_CONFIG``). Env vars override file values so ops can override
without editing YAML — e.g. ``FARMBOT_PIN_b1=27``.
"""

from __future__ import annotations

import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Protocol

import yaml

from safety_service import UnsafeActionError, validate as safety_validate

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dev.yaml"


def _load_yaml_config() -> dict:
    path = Path(os.getenv("TWFB_CONFIG", _DEFAULT_CONFIG_PATH))
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _pin_map() -> dict[str, int]:
    """Return the bed_id → FarmBot pin mapping (yaml + env override)."""
    cfg = _load_yaml_config().get("watering", {}).get("pins", {}) or {}
    pins = {str(k): int(v) for k, v in cfg.items()}
    # env override: FARMBOT_PIN_b1=27
    for k, v in os.environ.items():
        if k.startswith("FARMBOT_PIN_"):
            bed = k[len("FARMBOT_PIN_"):]
            pins[bed] = int(v)
    return pins


class WaterSource(Protocol):
    def open(self, bed_id: str) -> None: ...
    def close(self, bed_id: str) -> None: ...


def _load_backend() -> WaterSource:
    name = os.getenv("WATERING_BACKEND", "farmbot")
    module = import_module(f"watering_service.backends.{name}")
    return module.backend  # type: ignore[attr-defined] 


_backend: WaterSource | None = None


def get_backend() -> WaterSource:
    global _backend
    if _backend is None:
        _backend = _load_backend()
        log.info("watering backend loaded: %s", type(_backend).__name__)
    return _backend


def water_bed(bed_id: str, seconds: float) -> object:
    """Validate then execute a watering action. Returns the executed Action."""
    from twfarmbot_core.domain.action import Action

    action = Action(kind="water", params={"bed_id": bed_id, "seconds": seconds})
    try:
        safety_validate(action)
    except UnsafeActionError:
        log.warning("safety: rejected %s", action)
        raise

    backend = get_backend()
    log.info("watering: opening valve for bed=%s seconds=%s", bed_id, seconds)
    backend.open(bed_id)
    try:
        time.sleep(seconds)
    finally:
        backend.close(bed_id)
        log.info("watering: closed valve for bed=%s", bed_id)

    return action


# Re-export for handlers / tests
__all__ = [
    "WaterSource",
    "_pin_map",
    "get_backend",
    "water_bed",
]
