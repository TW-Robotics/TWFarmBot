"""Camera action handlers."""

from __future__ import annotations

import watering_service
from twfarmbot_core.domain import Action


def handle_take_photo(action: Action) -> Action:
    backend = watering_service.get_backend()
    backend.take_photo()
    return action
