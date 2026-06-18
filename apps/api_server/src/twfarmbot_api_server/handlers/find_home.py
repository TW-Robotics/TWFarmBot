"""Handler for Action(kind='find_home', params={})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_find_home(action: Action) -> Action:
    farmbot.backend.find_home()
    return action
