"""Camera action handlers."""

from __future__ import annotations

from twfarmbot_core.domain import Action
from watering_service.backends import farmbot


def handle_take_photo(action: Action) -> Action:
    farmbot.backend.take_photo()
    wait_for_new_photo = getattr(farmbot.backend, "wait_for_new_photo", None)
    if callable(wait_for_new_photo):
        wait_for_new_photo()
    return action
