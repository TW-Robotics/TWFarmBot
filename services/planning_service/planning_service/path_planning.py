"""Spatial path-planning helpers for the FarmBot assistant.

These functions are pure geometry: they take the configured garden world
(from ``spatial_service``) and produce waypoint lists. They do **not**
mutate the robot.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from spatial_service import load_world
from twfarmbot_core.domain import GardenWorld, GardenZone, Point3D, Rectangle


def _point(value: Any) -> Point3D:
    """Build a Point3D from a dict, list/tuple, or Point3D."""
    if isinstance(value, Point3D):
        return value
    if isinstance(value, Mapping):
        return Point3D(
            x=float(value.get("x", 0)),
            y=float(value.get("y", 0)),
            z=float(value.get("z", 0)),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return Point3D(
            x=float(value[0]),
            y=float(value[1]),
            z=float(value[2]) if len(value) > 2 else 0.0,
        )
    raise ValueError(f"cannot interpret {value!r} as a point")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _within_bounds(point: Point3D, bounds: Rectangle) -> bool:
    return (
        bounds.x <= point.x <= bounds.x + bounds.width
        and bounds.y <= point.y <= bounds.y + bounds.height
    )


def plan_path(
    start: Any,
    target: Any,
    step_mm: float = 100.0,
    z: float = 0.0,
    *,
    world: GardenWorld | None = None,
) -> list[dict[str, float]]:
    """Generate waypoints along a straight line from ``start`` to ``target``.

    ``step_mm`` is the maximum distance between consecutive waypoints.
    The start and target are always included. Waypoints are clamped to the
    garden bounds.
    """
    if world is None:
        world = load_world()
    start_pt = _point(start)
    target_pt = _point(target)

    # Override Z unless the caller explicitly provided it in start/target.
    start_pt = Point3D(start_pt.x, start_pt.y, z)
    target_pt = Point3D(target_pt.x, target_pt.y, z)

    dx = target_pt.x - start_pt.x
    dy = target_pt.y - start_pt.y
    distance = math.hypot(dx, dy)

    step = max(1.0, float(step_mm))
    if distance <= step:
        steps = 1
    else:
        steps = max(1, math.ceil(distance / step))

    bounds = world.bounds
    waypoints: list[dict[str, float]] = []
    for i in range(steps + 1):
        t = i / steps
        x = _clamp(start_pt.x + dx * t, bounds.x, bounds.x + bounds.width)
        y = _clamp(start_pt.y + dy * t, bounds.y, bounds.y + bounds.height)
        waypoints.append({"x": round(x, 2), "y": round(y, 2), "z": z})

    # Deduplicate start == target case.
    if len(waypoints) > 1 and waypoints[0] == waypoints[-1]:
        return [waypoints[0]]
    return waypoints


def scan_zone(
    zone_id: str,
    step_mm: float = 200.0,
    z: float = 0.0,
    *,
    world: GardenWorld | None = None,
) -> list[dict[str, float]]:
    """Generate a raster (boustrophedon) waypoint list covering a zone.

    The zone is scanned in rows along the X axis; each subsequent row is
    traversed in the opposite direction to minimise unnecessary travel.
    Waypoints are clamped to the garden bounds.
    """
    if world is None:
        world = load_world()

    zone: GardenZone | None = None
    for z_candidate in world.zones:
        if z_candidate.id == zone_id or z_candidate.name == zone_id:
            zone = z_candidate
            break
    if zone is None:
        raise ValueError(f"zone {zone_id!r} not found")

    step = max(1.0, float(step_mm))
    bounds = world.bounds
    b = zone.bounds

    # Snap the scan lines to be centred inside the zone.
    y_start = b.y + step / 2
    y_end = b.y + b.height - step / 2
    if y_start > y_end:
        y_start = b.y + b.height / 2
        y_end = y_start

    x_start = b.x + step / 2
    x_end = b.x + b.width - step / 2
    if x_start > x_end:
        x_start = b.x + b.width / 2
        x_end = x_start

    waypoints: list[dict[str, float]] = []
    reverse = False
    y = y_start
    while y <= y_end + 1e-6:
        y_clamped = _clamp(y, bounds.y, bounds.y + bounds.height)
        xs = list(_raster_x_line(x_start, x_end, step, reverse))
        for x in xs:
            x_clamped = _clamp(x, bounds.x, bounds.x + bounds.width)
            waypoints.append(
                {"x": round(x_clamped, 2), "y": round(y_clamped, 2), "z": z}
            )
        reverse = not reverse
        if y >= y_end - 1e-6:
            break
        y += step

    return waypoints


def _raster_x_line(
    x_start: float, x_end: float, step: float, reverse: bool
) -> Sequence[float]:
    """Return X coordinates for one raster row."""
    if x_start > x_end:
        return []
    if reverse:
        pts: list[float] = []
        x = x_end
        while x >= x_start - 1e-6:
            pts.append(x)
            if x <= x_start + 1e-6:
                break
            x -= step
        return pts
    pts = []
    x = x_start
    while x <= x_end + 1e-6:
        pts.append(x)
        if x >= x_end - 1e-6:
            break
        x += step
    return pts
