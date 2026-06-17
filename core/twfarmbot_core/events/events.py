from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Event:
    topic: str
    payload: dict[str, Any]
    occurred_at: datetime
