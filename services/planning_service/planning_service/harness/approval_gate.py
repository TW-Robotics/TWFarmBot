"""Action gate: execute tool calls or collect them during planning.

Safety validation still runs before physical actions execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from safety_service import UnsafeActionError, validate as safety_validate
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

from .tool_policy import ToolDescriptor
from ..tools import normalize_params


@dataclass(frozen=True)
class ToolResult:
    """Result of an executed or collected tool call."""

    status: str  # "ok", "proposed", "error", "noop"
    kind: str
    params: dict[str, Any]
    note: str = ""
    error: str = ""


@dataclass(frozen=True)
class ProposedResult(ToolResult):
    """Collected action during planning (not executed)."""

    def __init__(self, kind: str, params: dict[str, Any]) -> None:
        super().__init__(
            status="proposed",
            kind=kind,
            params=params,
            note="Action collected for plan output.",
        )


class ApprovalGate:
    """Execute tool calls immediately when actions are allowed."""

    def __init__(
        self, registry: ActionRegistry, *, planning_mode: bool = False
    ) -> None:
        self._registry = registry
        self._planning_mode = planning_mode

    def resolve(
        self,
        descriptor: ToolDescriptor,
        params: dict[str, Any],
        *,
        propose_only: bool,
        allow_actions: bool,
    ) -> ToolResult:
        del propose_only
        kind = descriptor.name
        params = normalize_params(dict(params))

        if descriptor.is_introspection:
            return ToolResult(status="noop", kind=kind, params=params)

        if self._planning_mode:
            return ProposedResult(kind, params)

        if not allow_actions:
            return ToolResult(
                status="error",
                kind=kind,
                params=params,
                error="physical actions are disabled for this request",
            )

        try:
            action = Action(kind=kind, params=params)
            result = self._registry.dispatch(action)
            return ToolResult(status="ok", kind=kind, params=result.params)
        except UnsafeActionError as err:
            return ToolResult(
                status="error",
                kind=kind,
                params=params,
                error=f"unsafe: {err}",
            )
        except Exception as err:  # noqa: BLE001
            return ToolResult(
                status="error",
                kind=kind,
                params=params,
                error=f"{type(err).__name__}: {err}",
            )

    def check_safety(self, kind: str, params: dict[str, Any]) -> None:
        """Raise ``UnsafeActionError`` if the action would be rejected."""
        safety_validate(Action(kind=kind, params=normalize_params(dict(params))))
