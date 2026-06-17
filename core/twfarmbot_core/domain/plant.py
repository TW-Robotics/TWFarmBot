from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plant:
    id: str
    species: str
    bed_id: str
