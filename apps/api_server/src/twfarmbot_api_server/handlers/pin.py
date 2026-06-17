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
    farmbot.backend.write_pin(pin, value, mode)
    return action
