"""Handler for Action(kind='find_home', params={})."""

from __future__ import annotations

import watering_service
from twfarmbot_core.domain import Action


def handle_find_home(action: Action) -> Action:
    backend = watering_service.get_backend()
    backend.find_home()
    return action
