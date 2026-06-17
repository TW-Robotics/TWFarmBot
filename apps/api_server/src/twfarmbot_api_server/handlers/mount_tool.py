"""Handlers for tool mount/dismount."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_mount_tool(action: Action) -> Action:
    farmbot.backend.mount_tool(str(action.params["tool_name"]))
    return action


def handle_dismount_tool(action: Action) -> Action:
    farmbot.backend.dismount_tool()
    return action
