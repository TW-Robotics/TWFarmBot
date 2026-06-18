"""Camera action handlers."""

from __future__ import annotations

from twfarmbot_core.domain import Action
from watering_service.backends import farmbot


def handle_take_photo(action: Action) -> Action:
    farmbot.backend.take_photo()
    return action
