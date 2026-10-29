"""Action dispatch — the shared vocabulary for "things the system can do".

Lives in ``core`` so any layer (api_server, worker, projects, experiments)
can dispatch an ``Action`` through the same registry without depending on
``apps/``.

Safety validation is **injected** at construction time — pass
``safety_service.validate`` (or any ``Action -> Action`` callable) as the
``validate`` argument.  When omitted, dispatch runs handlers directly,
which is useful for tests and benchmarks.
"""

from __future__ import annotations

from typing import Callable

from .domain import Action

ActionHandler = Callable[[Action], Action]

DEFAULT_ACTION_KINDS: tuple[str, ...] = (
    "water",
    "move",
    "move_axis",
    "move_path",
    "mount_tool",
    "dismount_tool",
    "read_pin",
    "write_pin",
    "e_stop",
    "unlock",
    "find_home",
    "take_photo",
    "capture",
    "capture_ndre",
    "inspect_zone",
    "water_zone",
    "goto_named",
    "run_script",
    "scan_ndre",
    "set_light",
)


class UnknownActionError(KeyError):
    """Raised when no handler is registered for an Action's kind."""


class ActionRegistry:
    def __init__(
        self,
        validate: Callable[[Action], Action] | None = None,
    ) -> None:
        self._handlers: dict[str, ActionHandler] = {}
        self._validate = validate

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
        if self._validate is not None:
            self._validate(action)
        return handler(action)
