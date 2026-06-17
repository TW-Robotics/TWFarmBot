from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SensorReading:
    sensor_id: str
    metric: str
    value: float
    unit: str
    taken_at: datetime
