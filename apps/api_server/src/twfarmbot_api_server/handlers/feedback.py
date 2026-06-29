"""Handlers for feedback / control."""

from __future__ import annotations

import watering_service
from twfarmbot_core.domain import Action


def handle_send_message(action: Action) -> Action:
    backend = watering_service.get_backend()
    backend.send_message(
        str(action.params["message"]),
        message_type=str(action.params.get("type", "info")),
        channels=action.params.get("channels"),
    )
    return action


def handle_e_stop(action: Action) -> Action:
    backend = watering_service.get_backend()
    backend.e_stop()
    return action
