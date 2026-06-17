from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bed:
    id: str
    name: str
    width_mm: float
    height_mm: float
