"""Handlers for pin read/write."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_read_pin(action: Action) -> Action:
    pin = int(action.params["pin"])
    mode = str(action.params.get("mode", "digital"))
    value = farmbot.backend.read_pin(pin, mode)
    return Action(kind=action.kind, params={
        "pin": pin, "mode": mode, "value": value
    })


def handle_write_pin(action: Action) -> Action:
    pin = int(action.params["pin"])
    value = int(action.params["value"])
    mode = str(action.params.get("mode", "digital"))
    seconds = action.params.get("seconds")
    duration = None
    if value == 1 and seconds is not None:
        try:
            duration = float(seconds)
        except (TypeError, ValueError):
            duration = 0.0
    farmbot.backend.write_pin(pin, value, mode, seconds=duration)
    return action
