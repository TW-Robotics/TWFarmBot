"""Action dispatch — the shared vocabulary for "things the system can do".

Lives in ``core`` so any layer (api_server, worker, projects, experiments)
can dispatch an ``Action`` through the same registry without depending on
``apps/``. Every dispatch runs through ``safety_service.validate`` first;
no caller can bypass safety by going around this module.
"""

from __future__ import annotations

from typing import Callable, Protocol

from safety_service import UnsafeActionError, validate as safety_validate

from .domain import Action

ActionHandler = Callable[[Action], Action]


class UnknownActionError(KeyError):
    """Raised when no handler is registered for an Action's kind."""


class ActionRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, kind: str, handler: ActionHandler) -> None:
        if kind in self._handlers:
            raise ValueError(f"action kind {kind!r} already registered")
        self._handlers[kind] = handler

    def kinds(self) -> list[str]:
        return sorted(self._handlers)

    def dispatch(self, action: Action) -> Action:
        handler = self._handlers.get(action.kind)
        if handler is None:
            raise UnknownActionError(
                f"no handler registered for kind={action.kind!r}; "
                f"known kinds: {self.kinds()}"
            )
        safety_validate(action)  # every dispatch goes through safety
        return handler(action)
