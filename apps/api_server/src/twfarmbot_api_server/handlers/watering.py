"""Handler for Action(kind='water', params={'seconds'})."""

from __future__ import annotations

import watering_service
from twfarmbot_core.domain import Action


def handle_water(action: Action) -> Action:
    seconds = float(action.params["seconds"])
    backend = watering_service.get_backend()
    backend.water(seconds)
    return action
