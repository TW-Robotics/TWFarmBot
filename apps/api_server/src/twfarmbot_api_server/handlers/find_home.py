"""Handler for Action(kind='find_home', params={})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_find_home(action: Action) -> Action:
    axis = str(action.params.get("axis", "all"))
    speed = action.params.get("speed")
    farmbot.backend.find_home(
        axis=axis,
        speed=float(speed) if speed is not None else 100,
    )
    return action
