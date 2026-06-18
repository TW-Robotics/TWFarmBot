"""Shared spatial vocabulary for the garden world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class Rectangle:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class CameraPose:
    position: Point3D
    yaw_deg: float = 0.0
    pitch_deg: float = 90.0
    roll_deg: float = 0.0


@dataclass(frozen=True)
class GardenEntity:
    id: str
    kind: str
    name: str
    position: Point3D
    radius_mm: float = 20.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GardenZone:
    id: str
    kind: str
    name: str
    bounds: Rectangle
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GardenWorld:
    bounds: Rectangle
    camera: CameraPose
    entities: tuple[GardenEntity, ...] = ()
    zones: tuple[GardenZone, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
