"""Handler for Action(kind='unlock', params={})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_unlock(action: Action) -> Action:
    farmbot.backend.unlock()
    return action
