"""Handler for Action(kind='move_axis', params={'axis','distance','speed?'})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_move_axis(action: Action) -> Action:
    axis = str(action.params["axis"])
    distance = float(action.params["distance"])
    speed = action.params.get("speed")
    farmbot.backend.move_axis(
        axis,
        distance,
        speed=float(speed) if speed is not None else None,
    )
    return action
