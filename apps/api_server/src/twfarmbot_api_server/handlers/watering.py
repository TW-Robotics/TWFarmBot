"""Handler for Action(kind='water', params={'seconds'})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

import watering_service


def handle_water(action: Action) -> Action:
    seconds = float(action.params["seconds"])
    watering_service.water(seconds)
    return action
