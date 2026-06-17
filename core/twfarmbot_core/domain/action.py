from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
