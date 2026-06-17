"""Handler for Action(kind='move', params={'x','y','z','speed?'})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_move(action: Action) -> Action:
    x = float(action.params["x"])
    y = float(action.params["y"])
    z = float(action.params["z"])
    speed = action.params.get("speed")
    farmbot.backend.move(x, y, z, speed=float(speed) if speed is not None else None)
    return action
