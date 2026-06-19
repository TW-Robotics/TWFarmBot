"""Garden coordinate and world-model service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from twfarmbot_core.domain import (
    CameraPose,
    GardenEntity,
    GardenWorld,
    GardenZone,
    Point3D,
    Rectangle,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "dev.yaml"


def _point(data: Any) -> Point3D:
    """Extract a ``Point3D`` from a dict, list/tuple, or ``Point3D``-like.

    The FarmBot gateway returns ``cached_xyz()`` as ``{x, y, z}`` while the
    raw ``farmbot-py`` ``bot.get_xyz()`` returns a ``(x, y, z)`` tuple. We
    accept either so the spatial service works against either backend.
    """
    if isinstance(data, Point3D):
        return data
    if isinstance(data, Mapping):
        return Point3D(
            x=float(data.get("x", 0)),
            y=float(data.get("y", 0)),
            z=float(data.get("z", 0)),
        )
    if isinstance(data, (list, tuple)) and len(data) >= 3:
        return Point3D(
            x=float(data[0] or 0),
            y=float(data[1] or 0),
            z=float(data[2] or 0),
        )
    return Point3D(x=0.0, y=0.0, z=0.0)


def load_world(path: str | Path = DEFAULT_CONFIG) -> GardenWorld:
    """Load the configured world model."""
    config = yaml.safe_load(Path(path).read_text()) or {}
    spatial = config.get("spatial", {})
    bounds = spatial.get("bounds", {})
    camera = spatial.get("camera", {})
    camera_offset = spatial.get("camera_offset", {})

    entities = tuple(
        GardenEntity(
            id=str(item["id"]),
            kind=str(item["kind"]),
            name=str(item.get("name", item["id"])),
            position=_point(item),
            radius_mm=float(item.get("radius_mm", 20)),
            metadata=dict(item.get("metadata", {})),
        )
        for item in spatial.get("entities", [])
    )
    zones = tuple(
        GardenZone(
            id=str(item["id"]),
            kind=str(item["kind"]),
            name=str(item.get("name", item["id"])),
            bounds=Rectangle(
                x=float(item.get("x", 0)),
                y=float(item.get("y", 0)),
                width=float(item["width"]),
                height=float(item["height"]),
            ),
            metadata=dict(item.get("metadata", {})),
        )
        for item in spatial.get("zones", [])
    )
    return GardenWorld(
        bounds=Rectangle(
            x=float(bounds.get("x", 0)),
            y=float(bounds.get("y", 0)),
            width=float(bounds["width"]),
            height=float(bounds["height"]),
        ),
        camera=CameraPose(
            position=_point(camera_offset),
            yaw_deg=float(camera.get("yaw_deg", 0)),
            pitch_deg=float(camera.get("pitch_deg", 90)),
            roll_deg=float(camera.get("roll_deg", 0)),
        ),
        camera_offset=_point(camera_offset),
        entities=entities,
        zones=zones,
    )


def get_snapshot(robot_position: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the configured world plus optional live robot position.
    
    If robot_position is provided and camera_offset is configured,
    the camera position is computed as robot_position + camera_offset.
    """
    world = load_world()
    snapshot = world.to_dict()

    robot = _point(robot_position) if robot_position else Point3D(0.0, 0.0, 0.0)
    snapshot["robot"] = robot.__dict__

    camera_pos = Point3D(
        x=robot.x + world.camera_offset.x,
        y=robot.y + world.camera_offset.y,
        z=robot.z + world.camera_offset.z,
    )
    snapshot["camera"]["position"] = camera_pos.__dict__

    return snapshot


__all__ = ["get_snapshot", "load_world"]
