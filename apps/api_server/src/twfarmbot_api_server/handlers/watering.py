"""Handler for Action(kind='water')."""

from __future__ import annotations

from twfarmbot_core.domain import Action

import watering_service


def handle_water(action: Action) -> Action:
    seconds = action.params.get("seconds")
    watering_service.water(None if seconds is None else float(seconds))
    return action
