"""Composite garden jobs: inspect a bed, water a bed, go to a named target."""

from __future__ import annotations

from typing import Any

from twfarmbot_core.config import load_yaml_config
from twfarmbot_core.domain import Action
from watering_service.backends import farmbot

from .move import handle_move
from .path import handle_move_path
from .watering import handle_water


def handle_goto_named(action: Action) -> Action:
    from spatial_service import resolve_named

    target = resolve_named(str(action.params["name"]))
    z = action.params.get("z", target["z"])
    handle_move(
        Action(kind="move", params={"x": target["x"], "y": target["y"], "z": z})
    )
    return Action(kind="goto_named", params={**action.params, "resolved": target})


def handle_water_zone(action: Action) -> Action:
    from spatial_service import resolve_named

    target = resolve_named(str(action.params["zone_id"]))
    if target.get("kind") != "zone":
        raise ValueError(f"{action.params.get('zone_id')!r} is not a watering zone")
    z = action.params.get("z", target.get("z", 0.0))
    handle_move(
        Action(kind="move", params={"x": target["x"], "y": target["y"], "z": z})
    )
    handle_water(Action(kind="water", params={"seconds": action.params["seconds"]}))
    return Action(kind="water_zone", params={**action.params, "resolved": target})


def handle_inspect_zone(action: Action) -> Action:
    from planning_service.inspect import (
        score_image_urls,
        summarize_frames,
        summary_text,
    )
    from planning_service.path_planning import scan_zone
    from spatial_service import resolve_named

    target = resolve_named(str(action.params["zone_id"]))
    if target.get("kind") != "zone":
        raise ValueError(f"{action.params.get('zone_id')!r} is not a garden zone")
    step_mm = float(action.params.get("step_mm", 200))
    z = float(action.params.get("z", 0))
    classes = str(action.params.get("classes") or "plant, weed, soil")
    waypoints = scan_zone(str(target["id"]), step_mm=step_mm, z=z)
    handle_move_path(
        Action(
            kind="move_path",
            params={
                "waypoints": waypoints,
                "photo_at_waypoints": True,
                "wait_for_photos": True,
            },
        )
    )
    images = farmbot.backend.get_images(limit=max(len(waypoints), 1), refresh=True)
    urls = [
        str(image.get("attachment_url"))
        for image in images[: len(waypoints)]
        if isinstance(image, dict) and image.get("attachment_url")
    ]
    frames = score_image_urls(urls, classes=classes) if urls else []
    summary = summarize_frames(frames)
    scorecard = {
        "zone_id": target["id"],
        "zone_name": target["name"],
        "waypoint_count": len(waypoints),
        "frames": frames,
        "summary": summary,
        "summary_text": summary_text(str(target["name"]), summary),
        "sensors": _read_named_sensors(),
    }
    return Action(kind="inspect_zone", params={**action.params, "scorecard": scorecard})


def _read_named_sensors() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pin in load_yaml_config().get("pins") or []:
        if pin.get("kind") != "sensor":
            continue
        try:
            number = int(pin["pin"])
            mode = str(pin.get("mode") or "analog")
            out[str(pin.get("label") or number)] = farmbot.backend.read_pin(
                number, mode
            )
        except Exception:  # noqa: BLE001
            continue
    return out
