"""Handlers for tool mount/dismount."""

from __future__ import annotations

import watering_service
from twfarmbot_core.domain import Action


def handle_mount_tool(action: Action) -> Action:
    backend = watering_service.get_backend()
    backend.mount_tool(str(action.params["tool_name"]))
    return action


def handle_dismount_tool(action: Action) -> Action:
    backend = watering_service.get_backend()
    backend.dismount_tool()
    return action
