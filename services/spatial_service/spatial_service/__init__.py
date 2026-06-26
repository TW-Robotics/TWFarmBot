"""Garden coordinate and world-model service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from twfarmbot_core.config import load_yaml_config
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
    config = load_yaml_config(path)
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


def format_world_context(world: GardenWorld | Mapping[str, Any] | None = None) -> str:
    """Render a rich, model-friendly summary of the world model.

    Includes zones with computed centres, entities with positions, garden
    bounds, camera offset, and the last known robot/camera positions if they
    are present in the snapshot.
    """
    if world is None:
        world = load_world()
    snapshot = world.to_dict() if hasattr(world, "to_dict") else dict(world)
    lines: list[str] = []

    bounds = snapshot.get("bounds", {})
    lines.append(
        f"Garden bounds: x={bounds.get('x', 0)}, y={bounds.get('y', 0)}, "
        f"width={bounds.get('width', 0)}, height={bounds.get('height', 0)}. "
        f"All coordinates are in millimetres."
    )

    camera = snapshot.get("camera", {})
    cam_pos = camera.get("position") or snapshot.get("camera_offset", {})
    lines.append(
        f"Camera offset from robot origin: "
        f"x={cam_pos.get('x', 0)}, y={cam_pos.get('y', 0)}, z={cam_pos.get('z', 0)}. "
        f"When the robot is at (x,y,z), the camera is at robot + offset."
    )

    robot = snapshot.get("robot")
    if robot:
        lines.append(
            f"Last known robot position: "
            f"x={robot.get('x', 0)}, y={robot.get('y', 0)}, z={robot.get('z', 0)}."
        )

    zones = snapshot.get("zones", [])
    if zones:
        lines.append("\nZones (move to the centre when asked to go to a zone):")
        for zone in zones:
            bounds = zone.get("bounds", {})
            x = float(bounds.get("x", 0))
            y = float(bounds.get("y", 0))
            w = float(bounds.get("width", 0))
            h = float(bounds.get("height", 0))
            cx = round(x + w / 2)
            cy = round(y + h / 2)
            name = zone.get("name", zone.get("id"))
            meta = zone.get("metadata", {})
            meta_text = ", ".join(f"{k}={v}" for k, v in meta.items())
            meta_part = f", metadata={meta_text}" if meta_text else ""
            lines.append(
                f"- {name!r} (kind={zone.get('kind')}, id={zone.get('id')}, "
                f"bounds x={x}..{x + w}, y={y}..{y + h}, "
                f"centre=({cx}, {cy}){meta_part})"
            )

    entities = snapshot.get("entities", [])
    if entities:
        lines.append("\nPlants / objects:")
        for entity in entities:
            pos = entity.get("position", {})
            meta = entity.get("metadata", {})
            meta_text = ", ".join(f"{k}={v}" for k, v in meta.items())
            meta_part = f", metadata={meta_text}" if meta_text else ""
            lines.append(
                f"- {entity.get('name', entity.get('id'))!r} "
                f"(kind={entity.get('kind')}, id={entity.get('id')}, "
                f"x={pos.get('x')}, y={pos.get('y')}, z={pos.get('z')}, "
                f"radius_mm={entity.get('radius_mm', 20)}{meta_part})"
            )

    if not zones and not entities:
        lines.append("(no zones or entities configured)")

    lines.append(
        "\nWhen the user refers to a zone or plant by name, move to its centre "
        "or position first, then perform the requested action (photo, cut, water, etc.)."
    )
    return "\n".join(lines)


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


__all__ = ["format_world_context", "get_snapshot", "load_world"]
