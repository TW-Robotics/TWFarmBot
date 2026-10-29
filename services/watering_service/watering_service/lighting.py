"""Bed light — second Shelly plug. Stays on until turned off (no auto-off)."""

from __future__ import annotations

import logging
import os
from typing import Any

from twfarmbot_core.config import load_yaml_config

from .backends.shelly import ShellyPlug, _parse_host

log = logging.getLogger(__name__)


def _cfg() -> dict[str, Any]:
    lighting = load_yaml_config().get("lighting", {}) or {}
    return lighting.get("shelly", {}) or {} if isinstance(lighting, dict) else {}


def _host() -> str:
    raw = (os.getenv("SHELLY_LIGHT_HOST") or str(_cfg().get("host") or "")).strip()
    if not raw:
        raise ValueError(
            "Shelly light needs SHELLY_LIGHT_HOST or lighting.shelly.host in config"
        )
    return _parse_host(raw)


def _switch_id() -> int:
    return int(os.getenv("SHELLY_LIGHT_SWITCH_ID", _cfg().get("switch_id", 0)))


def _plug() -> ShellyPlug:
    return ShellyPlug(_host(), _switch_id())


def light_status() -> dict[str, Any]:
    return {"backend": "ShellyPlug", **_plug().status()}


def set_light(on: bool) -> dict[str, Any]:
    """Manual on/off. No timer — grow/work light stays until off."""
    _plug().set(bool(on))
    return light_status()
