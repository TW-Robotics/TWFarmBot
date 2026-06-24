"""Action dispatch — the shared vocabulary for "things the system can do".

Lives in ``core`` so any layer (api_server, worker, projects, experiments)
can dispatch an ``Action`` through the same registry without depending on
``apps/``. Every dispatch runs through ``safety_service.validate`` first;
no caller can bypass safety by going around this module.
"""

from __future__ import annotations

from typing import Any, Callable

from safety_service import UnsafeActionError, validate as safety_validate

from .domain import Action

ActionHandler = Callable[[Action], Action]


class UnknownActionError(KeyError):
    """Raised when no handler is registered for an Action's kind."""


def _num(value: Any) -> str:
    """Format a number for display, stripping trailing ``.0``."""
    if value is None:
        return "—"
    try:
        f = float(value)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return str(value)


def _summarize_move(params: dict[str, Any]) -> str:
    return (
        f"🛠️ **move** → "
        f"({_num(params.get('x'))}, {_num(params.get('y'))}, {_num(params.get('z'))})"
    )


def _summarize_move_path(params: dict[str, Any]) -> str:
    waypoints = params.get("waypoints", [])
    photo = params.get("photo_at_waypoints", False)
    water_pin = params.get("water_pin")
    extras = ""
    if water_pin is not None:
        extras += f" 💧 pin {water_pin}"
    if photo:
        extras += " 📷"
    return f"🛤️ **move_path** ({len(waypoints)} waypoints){extras}"


def _summarize_water(params: dict[str, Any]) -> str:
    return f"🌊 **water** for {_num(params.get('seconds'))} s"


def _summarize_find_home(params: dict[str, Any]) -> str:
    return f"🏠 **find_home** (axis={params.get('axis', 'all')}, speed={params.get('speed', '—')})"


def _summarize_take_photo(params: dict[str, Any]) -> str:
    return "📷 **take_photo**"


def _summarize_read_pin(params: dict[str, Any]) -> str:
    return f"📖 **read_pin** {params.get('pin', '—')} ({params.get('mode', 'digital')})"


def _summarize_write_pin(params: dict[str, Any]) -> str:
    return f"✏️ **write_pin** {params.get('pin', '—')} = {params.get('value', '—')}"


def _summarize_mount_tool(params: dict[str, Any]) -> str:
    return f"🔧 **mount_tool** {params.get('tool_name', '—')}"


def _summarize_dismount_tool(params: dict[str, Any]) -> str:
    return "🔧 **dismount_tool**"


def _summarize_e_stop(params: dict[str, Any]) -> str:
    return "🛑 **e_stop**"


ACTION_SUMMARIES: dict[str, Callable[[dict[str, Any]], str]] = {
    "move": _summarize_move,
    "move_path": _summarize_move_path,
    "water": _summarize_water,
    "find_home": _summarize_find_home,
    "take_photo": _summarize_take_photo,
    "read_pin": _summarize_read_pin,
    "write_pin": _summarize_write_pin,
    "mount_tool": _summarize_mount_tool,
    "dismount_tool": _summarize_dismount_tool,
    "e_stop": _summarize_e_stop,
}


def summarize_action(action: Action | dict[str, Any]) -> str:
    """Return a compact, human-readable summary of an action."""
    if isinstance(action, Action):
        kind = action.kind
        params = action.params
    else:
        kind = action.get("kind", "action")
        params = action.get("params") or {}
    fn = ACTION_SUMMARIES.get(kind)
    if fn is None:
        return f"🛠️ **{kind}**"
    return fn(params)


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
