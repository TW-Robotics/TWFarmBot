"""Handler for Action(kind='move_path', params={'waypoints': [...], ...})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


def handle_move_path(action: Action) -> Action:
    waypoints = action.params.get("waypoints", [])
    speed = action.params.get("speed")
    photo = bool(action.params.get("photo_at_waypoints", False))

    for wp in waypoints:
        x = float(wp["x"])
        y = float(wp["y"])
        z = float(wp["z"])
        farmbot.backend.move(
            x, y, z, speed=float(speed) if speed is not None else None
        )
        if photo:
            farmbot.backend.take_photo()

    return action
