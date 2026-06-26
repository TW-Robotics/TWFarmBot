"""Handler for Action(kind='move_path', params={'waypoints': [...], ...})."""

from __future__ import annotations

import math
import time
from threading import Thread
from typing import Any

from twfarmbot_core.domain import Action

from watering_service.backends import farmbot


# Tolerance for considering the gantry to have reached the target waypoint.
_POSITION_TOLERANCE_MM = 5.0
# Maximum total time to wait for the path to finish (safety backstop).
_MAX_PATH_WAIT_S = 300.0
# How often to poll the current position while watering.
_POSITION_POLL_INTERVAL_S = 0.5


def _execute_moves(
    waypoints: list[dict[str, Any]], speed: float | None, photo: bool
) -> None:
    """Issue all waypoint moves (and optional photos) sequentially."""
    for wp in waypoints:
        x = float(wp["x"])
        y = float(wp["y"])
        z = float(wp["z"])
        farmbot.backend.move(x, y, z, speed=float(speed) if speed is not None else None)
        if photo:
            farmbot.backend.take_photo()


def _current_position() -> dict[str, float] | None:
    """Return the latest cached gantry position, or None if unknown."""
    try:
        pos = farmbot.backend.get_xyz()
        if isinstance(pos, dict):
            return {
                "x": float(pos.get("x", 0)),
                "y": float(pos.get("y", 0)),
                "z": float(pos.get("z", 0)),
            }
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
    except Exception:  # noqa: BLE001
        pass
    return None


def _distance_to_target(position: dict[str, float], target: dict[str, Any]) -> float:
    """2D distance from the current position to a waypoint."""
    return math.hypot(
        position["x"] - float(target["x"]),
        position["y"] - float(target["y"]),
    )


def _wait_for_path_completion(
    waypoints: list[dict[str, Any]],
    move_thread: Thread,
    timeout_s: float = _MAX_PATH_WAIT_S,
) -> bool:
    """Poll position until the move thread finishes and the final waypoint is reached."""
    if not waypoints:
        move_thread.join(timeout=timeout_s)
        return not move_thread.is_alive()

    final = waypoints[-1]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        move_thread.join(timeout=_POSITION_POLL_INTERVAL_S)
        pos = _current_position()
        if pos is None:
            continue
        reached = _distance_to_target(pos, final) <= _POSITION_TOLERANCE_MM
        if reached and not move_thread.is_alive():
            return True
    return False


def handle_move_path(action: Action) -> Action:
    waypoints = action.params.get("waypoints", [])
    speed = action.params.get("speed")
    photo = bool(action.params.get("photo_at_waypoints", False))
    water_pin = action.params.get("water_pin")

    if water_pin is None:
        _execute_moves(waypoints, speed, photo)
        return action

    # Watering mode: keep the pin HIGH for the duration of the path and turn
    # it OFF once the final waypoint is actually reached (position feedback).
    pin = int(water_pin)
    try:
        farmbot.backend.write_pin(pin, 1, "digital")
        move_thread = Thread(
            target=_execute_moves,
            args=(waypoints, speed, photo),
            name="move_path_executor",
        )
        move_thread.start()
        _wait_for_path_completion(waypoints, move_thread)
        move_thread.join(timeout=5.0)
    finally:
        farmbot.backend.write_pin(pin, 0, "digital")

    return action
