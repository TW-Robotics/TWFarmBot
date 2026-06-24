"""Approval gate: decide whether a tool call executes or becomes a proposal.

This is the single place where "can the robot do this right now?" is
answered. It combines the tool's policy with the conversation mode
(propose_only / allow_actions) and the safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from safety_service import UnsafeActionError, validate as safety_validate
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

from .tool_policy import ToolDescriptor


@dataclass(frozen=True)
class ToolResult:
    """Result of an executed or proposed tool call."""

    status: str  # "ok", "proposed", "error", "noop"
    kind: str
    params: dict[str, Any]
    note: str = ""
    error: str = ""


@dataclass(frozen=True)
class ProposedResult(ToolResult):
    """Convenience marker for a proposed action."""

    def __init__(self, kind: str, params: dict[str, Any]) -> None:
        super().__init__(
            status="proposed",
            kind=kind,
            params=params,
            note="This action is proposed and requires user approval.",
        )


class ApprovalGate:
    """Execute or propose a tool call according to policy and mode."""

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
        policy = descriptor.policy
        kind = descriptor.name

        if descriptor.is_introspection:
            # Read-only / analysis tools never require approval.
            return ToolResult(status="noop", kind=kind, params=params)

        # In planning mode we never execute physical actions; we only collect
        # proposed actions so the planner can return them for later approval.
        if self._planning_mode:
            return ProposedResult(kind, params)

        # ACT tools that require approval become proposals in proposal mode
        # or when actions are not allowed at all.
        if policy.requires_approval:
            if propose_only or not allow_actions:
                return ProposedResult(kind, params)

        # Some tools are always immediate (e.g. e_stop), but still gated by
        # allow_actions unless explicitly allowed without user confirmation.
        if not policy.allow_without_user and not allow_actions:
            return ProposedResult(kind, params)

        # Execute through the registry. Safety validation runs inside dispatch.
        try:
            action = Action(kind=kind, params=dict(params))
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
        """Raise ``UnsafeActionError`` if the action would be rejected.

        Exposed so callers (e.g. the planner JSON fallback) can validate
        without dispatching.
        """
        safety_validate(Action(kind=kind, params=dict(params)))
