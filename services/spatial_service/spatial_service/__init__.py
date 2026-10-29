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

from .observations import (
    DEFAULT_CELL_MM,
    find_locations,
    format_cells,
    latest_cells,
    record_observation,
    try_record_ndre,
    try_record_segment,
)
from .positions import list_positions, save_positions
from .scripts import list_scripts, resolve_script, save_scripts

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "dev.yaml"
_STOP_WORDS = frozenset({"the", "a", "an", "to", "zone", "bed", "plant", "go", "named"})


def _point(data: Any) -> Point3D:
    """Extract a ``Point3D`` from a dict, list/tuple, or ``Point3D``-like.

    The FarmBot gateway returns ``cached_xyz()`` as ``{x, y, z}`` while the
    ``get_xyz()`` returns ``{x, y, z}``. We
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

    Includes entities with positions, camera offset, and the last known
    robot/camera positions if they are present in the snapshot.
    """
    if world is None:
        world = load_world()
    snapshot = world.to_dict() if hasattr(world, "to_dict") else dict(world)
    lines: list[str] = [
        "Coordinates are millimetres from home (0,0,0) after find_home."
    ]

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

    positions = snapshot.get("positions")
    if positions is None:
        try:
            positions = list_positions()
        except Exception:  # noqa: BLE001
            positions = []
    if positions:
        lines.append(
            "\nNamed gantry presets (goto_named; labels and millimetres "
            "are edited on the Motion page):"
        )
        for item in positions:
            group = item.get("group") or "Travel"
            if item.get("set") is False or any(
                item.get(axis) is None for axis in ("x", "y", "z")
            ):
                lines.append(
                    f"- {item.get('label')!r} ({group}): not set — "
                    "save a pose on the Motion page before going there"
                )
            else:
                lines.append(
                    f"- {item.get('label')!r} ({group}): "
                    f"x={item.get('x')}, y={item.get('y')}, z={item.get('z')}"
                )

    scripts = snapshot.get("scripts")
    if scripts is None:
        try:
            scripts = list_scripts()
        except Exception:  # noqa: BLE001
            scripts = []
    if scripts:
        lines.append(
            "\nNamed scripts (run_script; waypoint lists edited on the Scripts page). "
            "Use these for tool pickup — do not replay the millimetres by hand:"
        )
        for item in scripts:
            wps = item.get("waypoints") or []
            if not wps:
                lines.append(f"- {item.get('label')!r}: empty — record waypoints on Scripts")
                continue
            path = " → ".join(
                f"({wp.get('x')},{wp.get('y')},{wp.get('z')})" for wp in wps
            )
            lines.append(f"- {item.get('label')!r}: {path}")

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
        lines.append("(no named zones or plants configured)")

    cells = snapshot.get("cells")
    if cells is None:
        try:
            cells = latest_cells()
        except Exception:  # noqa: BLE001
            cells = []
    lines.append("")
    lines.append(format_cells(list(cells) if cells else []))

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
    for item in list_positions():
        label = str(item.get("label") or "")
        if not label:
            continue
        score = _name_score(query_tokens, label)
        if not score:
            continue
        if not item.get("set"):
            ranked.append(
                (
                    score,
                    {
                        "kind": "position",
                        "id": label.casefold(),
                        "name": label,
                        "unset": True,
                    },
                )
            )
            continue
        ranked.append(
            (
                score,
                {
                    "kind": "position",
                    "id": label.casefold(),
                    "name": label,
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "z": float(item["z"]),
                },
            )
        )
    if not ranked:
        raise ValueError(f"no zone, plant, or preset named {query!r}")
    ranked.sort(
        key=lambda item: (
            -item[0],
            1 if item[1].get("unset") else 0,
            item[1]["name"],
        )
    )
    best = ranked[0][1]
    if best.get("unset"):
        raise ValueError(
            f"{best.get('name')!r} has no saved gantry pose yet. "
            "Save it on the Motion page first."
        )
    return best


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
    snapshot["cell_mm"] = DEFAULT_CELL_MM
    try:
        snapshot["cells"] = latest_cells()
    except Exception:  # noqa: BLE001
        snapshot["cells"] = []

    return snapshot


__all__ = [
    "DEFAULT_CELL_MM",
    "find_locations",
    "format_cells",
    "format_world_context",
    "get_snapshot",
    "latest_cells",
    "list_positions",
    "list_scripts",
    "load_world",
    "record_observation",
    "resolve_named",
    "resolve_script",
    "save_positions",
    "save_scripts",
    "try_record_ndre",
    "try_record_segment",
]
