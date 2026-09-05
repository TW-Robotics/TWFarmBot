"""Composite garden jobs: inspect a bed, water a bed, go to a named target."""

from __future__ import annotations

from typing import Any

from twfarmbot_core.config import load_yaml_config
from twfarmbot_core.domain import Action
from watering_service.backends import farmbot
from planning_service.harness.cancel import is_cancelled

from .move import handle_move
from .ndre import handle_capture_ndre
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


_MAX_NDRE_STOPS = 12


def axis_stops(start: float, end: float, step: float) -> list[float]:
    """Inclusive millimetre stops from ``start`` to ``end``."""
    if step <= 0:
        raise ValueError("step_mm must be positive")
    step = abs(float(step))
    if end < start:
        step = -step
    stops: list[float] = []
    pos = float(start)
    for _ in range(_MAX_NDRE_STOPS):
        stops.append(round(pos, 3))
        if abs(pos - end) < 1e-6:
            return stops
        nxt = pos + step
        if (step > 0 and nxt > end + 1e-6) or (step < 0 and nxt < end - 1e-6):
            if abs(stops[-1] - end) > 1e-3:
                stops.append(round(float(end), 3))
            return stops
        pos = nxt
    raise ValueError(
        f"scan_ndre would take more than {_MAX_NDRE_STOPS} stops; increase step_mm"
    )


def handle_scan_ndre(action: Action) -> Action:
    """Move along one axis and run capture_ndre at every stop."""
    axis = str(action.params.get("axis") or "y").strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("scan_ndre axis must be 'x' or 'y'")
    xyz = farmbot.backend.get_xyz()
    origin = {
        "x": float(xyz.get("x") or 0),
        "y": float(xyz.get("y") or 0),
        "z": float(xyz.get("z") or 0),
    }
    start = action.params.get("start_mm")
    start_mm = origin[axis] if start is None else float(start)
    end_mm = float(action.params["end_mm"])
    step_mm = float(action.params.get("step_mm") or 100)
    z = origin["z"] if action.params.get("z") is None else float(action.params["z"])
    return_home = bool(action.params.get("return_to_start", True))
    stops = axis_stops(start_mm, end_mm, step_mm)
    samples: list[dict[str, Any]] = []
    cancelled = False
    for stop in stops:
        if is_cancelled():
            cancelled = True
            break
        target = {"x": origin["x"], "y": origin["y"], "z": z, axis: stop}
        handle_move(Action(kind="move", params=target))
        captured = handle_capture_ndre(
            Action(kind="capture_ndre", params={"return_to_start": False})
        )
        nir = captured.params.get("nir")
        artifact_id = (
            str(nir.get("artifact_id"))
            if isinstance(nir, dict) and nir.get("artifact_id")
            else None
        )
        preview = captured.params.get("ndre_preview")
        if not preview and artifact_id:
            preview = f"/captures/{artifact_id}/ndre"
        samples.append(
            {
                "x": target["x"],
                "y": target["y"],
                "z": target["z"],
                "ndre": captured.params.get("ndre"),
                "interpretation": captured.params.get("interpretation"),
                "summary": captured.params.get("summary"),
                "nir": nir,
                "rededge": captured.params.get("rededge"),
                "ndre_preview": preview,
            }
        )
    if return_home and not cancelled:
        handle_move(Action(kind="move", params=origin))
    return Action(
        kind="scan_ndre",
        params={
            **action.params,
            "stops": stops,
            "samples": samples,
            "count": len(samples),
            "cancelled": cancelled,
        },
    )


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
