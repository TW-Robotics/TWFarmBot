"""Handlers for pin read/write."""

from __future__ import annotations

from twfarmbot_core.config import load_yaml_config
from twfarmbot_core.domain import Action

import watering_service


def _pin_mode(pin: int, requested: str | None) -> str:
    """Return the requested pin mode, falling back to the configured mode."""
    if requested:
        return str(requested)
    try:
        pins = load_yaml_config().get("pins", []) or []
        for p in pins:
            if int(p.get("pin", -1)) == pin:
                return str(p.get("mode", "digital"))
    except Exception:  # noqa: BLE001
        pass
    return "digital"


def handle_read_pin(action: Action) -> Action:
    pin = int(action.params["pin"])
    mode = _pin_mode(pin, action.params.get("mode"))
    backend = watering_service.get_backend()
    value = backend.read_pin(pin, mode)
    return Action(kind=action.kind, params={"pin": pin, "mode": mode, "value": value})


def handle_write_pin(action: Action) -> Action:
    pin = int(action.params["pin"])
    value = int(action.params["value"])
    mode = _pin_mode(pin, action.params.get("mode"))
    seconds = action.params.get("seconds")
    duration = None
    if value == 1 and seconds is not None:
        try:
            duration = float(seconds)
        except (TypeError, ValueError):
            duration = 0.0
    backend = watering_service.get_backend()
    backend.write_pin(pin, value, mode, seconds=duration)
    return action
