"""Handlers for feedback / control."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_send_message(action: Action) -> Action:
    farmbot.backend.send_message(
        str(action.params["message"]),
        message_type=str(action.params.get("type", "info")),
        channels=action.params.get("channels"),
    )
    return action


def handle_e_stop(action: Action) -> Action:
    farmbot.backend.e_stop()
    return action
