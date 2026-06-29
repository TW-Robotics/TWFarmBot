"""Handler for Action(kind='move', params={'x','y','z','speed?'})."""

from __future__ import annotations

import watering_service
from twfarmbot_core.domain import Action


def handle_move(action: Action) -> Action:
    x = float(action.params["x"])
    y = float(action.params["y"])
    z = float(action.params["z"])
    speed = action.params.get("speed")
    backend = watering_service.get_backend()
    backend.move(x, y, z, speed=float(speed) if speed is not None else None)
    return action
