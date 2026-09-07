"""Handlers for feedback / control."""

from __future__ import annotations

import logging

from twfarmbot_core.domain import Action

from farmbot_gateway import farmbot

log = logging.getLogger(__name__)


def handle_send_message(action: Action) -> Action:
    farmbot.backend.send_message(
        str(action.params["message"]),
        message_type=str(action.params.get("type", "info")),
        channels=action.params.get("channels"),
    )
    return action


def handle_e_stop(action: Action) -> Action:
    farmbot.backend.e_stop()
    try:
        import watering_service

        watering_service.stop()
    except Exception:  # noqa: BLE001
        log.warning("watering stop after e_stop failed", exc_info=True)
    return action
