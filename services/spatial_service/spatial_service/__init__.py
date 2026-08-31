"""Garden coordinate and world-model service."""

from __future__ import annotations

import re
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
_STOP_WORDS = frozenset({"the", "a", "an", "to", "zone", "bed", "plant", "go", "named"})


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


def _name_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^a-z0-9]+", text.casefold())
        if token and token not in _STOP_WORDS
    ]


def _name_score(query_tokens: list[str], candidate: str) -> int:
    name_tokens = _name_tokens(candidate) or [candidate.casefold()]
    score = 0
    for query in query_tokens:
        for name in name_tokens:
            if query == name:
                score = max(score, 3)
            elif query.startswith(name) or name.startswith(query):
                score = max(score, 2)
            elif query in name or name in query:
                score = max(score, 1)
    return score


def resolve_named(
    name: str, world: GardenWorld | Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve a loose zone / plant / preset name to a gantry target."""
    query = (name or "").strip()
    if not query:
        raise ValueError("name is empty")
    query_tokens = _name_tokens(query) or [query.casefold()]
    if world is None:
        garden = load_world()
    elif isinstance(world, GardenWorld):
        garden = world
    else:
        garden = load_world()

    ranked: list[tuple[int, dict[str, Any]]] = []
    for zone in garden.zones:
        score = max(
            _name_score(query_tokens, zone.id),
            _name_score(query_tokens, zone.name),
            _name_score(query_tokens, zone.kind),
        )
        if not score:
            continue
        ranked.append(
            (
                score,
                {
                    "kind": "zone",
                    "id": zone.id,
                    "name": zone.name,
                    "x": float(round(zone.bounds.x + zone.bounds.width / 2)),
                    "y": float(round(zone.bounds.y + zone.bounds.height / 2)),
                    "z": 0.0,
                    "width": float(zone.bounds.width),
                    "height": float(zone.bounds.height),
                },
            )
        )
    for entity in garden.entities:
        score = max(
            _name_score(query_tokens, entity.id),
            _name_score(query_tokens, entity.name),
            _name_score(query_tokens, entity.kind),
        )
        if not score:
            continue
        ranked.append(
            (
                score,
                {
                    "kind": "entity",
                    "id": entity.id,
                    "name": entity.name,
                    "x": float(entity.position.x),
                    "y": float(entity.position.y),
                    "z": float(entity.position.z),
                },
            )
        )
    for item in load_yaml_config().get("positions") or []:
        label = str(item.get("label") or "")
        if not label:
            continue
        score = _name_score(query_tokens, label)
        if not score:
            continue
        ranked.append(
            (
                score,
                {
                    "kind": "position",
                    "id": label.casefold(),
                    "name": label,
                    "x": float(item.get("x", 0)),
                    "y": float(item.get("y", 0)),
                    "z": float(item.get("z", 0)),
                },
            )
        )
    if not ranked:
        raise ValueError(f"no zone, plant, or preset named {query!r}")
    ranked.sort(key=lambda item: (-item[0], item[1]["name"]))
    return ranked[0][1]


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


__all__ = ["format_world_context", "get_snapshot", "load_world", "resolve_named"]
