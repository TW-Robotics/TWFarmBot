from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from .events import Event

Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)

    def subscribe(self, topic: str, fn: Subscriber) -> None:
        self._subscribers[topic].append(fn)

    def publish(self, topic: str, payload: dict) -> None:
        event = Event(
            topic=topic, payload=payload, occurred_at=datetime.now(timezone.utc)
        )
        for fn in self._subscribers.get(topic, []):
            fn(event)


_default_bus: EventBus | None = None


def get_default_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus
