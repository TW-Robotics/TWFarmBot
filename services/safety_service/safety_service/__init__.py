"""safety_service — gates every action that affects the real world.

Per the README: *Any code path that ultimately moves the FarmBot (watering,
weeding, tool changes, …) must pass through safety_service before it reaches
farmbot_gateway.*
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from twfarmbot_core.domain.action import Action

log = logging.getLogger(__name__)


class UnsafeActionError(ValueError):
    """Raised when an action fails a safety rule."""


@dataclass(frozen=True)
class SafetyLimits:
    max_water_seconds: float = 300.0
    allowed_beds: frozenset[str] = frozenset()  # empty = allow all
    max_axis_mm: dict[str, float] = field(
        default_factory=lambda: {"x": 3000.0, "y": 1500.0, "z": 800.0}
    )


def load_limits() -> SafetyLimits:
    beds_raw = os.getenv("FARMBOT_ALLOWED_BEDS", "")
    allowed = frozenset(b for b in beds_raw.split(",") if b)
    return SafetyLimits(
        max_water_seconds=float(os.getenv("FARMBOT_MAX_WATER_SECONDS", "300")),
        allowed_beds=allowed,
        max_axis_mm={
            "x": float(os.getenv("FARMBOT_MAX_AXIS_X", "3000")),
            "y": float(os.getenv("FARMBOT_MAX_AXIS_Y", "1500")),
            "z": float(os.getenv("FARMBOT_MAX_AXIS_Z", "800")),
        },
    )


def _check_move(action: Action, limits: SafetyLimits) -> None:
    for axis in ("x", "y", "z"):
        if axis not in action.params:
            raise UnsafeActionError(f"move action needs {axis!r}")
        try:
            value = float(action.params[axis])
        except (TypeError, ValueError) as err:
            raise UnsafeActionError(
                f"move action {axis!r} must be numeric, got {action.params[axis]!r}"
            ) from err
        cap = limits.max_axis_mm.get(axis, float("inf"))
        if abs(value) > cap:
            raise UnsafeActionError(
                f"move action {axis}={value} exceeds |max| {cap} mm"
            )


def validate(action: Action, *, limits: SafetyLimits | None = None) -> Action:
    """Check an Action against the safety rules. Returns it unchanged on pass.

    Raises :class:`UnsafeActionError` if the action is rejected.
    """
    limits = limits or load_limits()

    if action.kind == "water":
        seconds = float(action.params.get("seconds", 0.0))
        bed_id = str(action.params.get("bed_id", ""))
        if seconds <= 0:
            raise UnsafeActionError(f"water action needs positive seconds, got {seconds}")
        if seconds > limits.max_water_seconds:
            raise UnsafeActionError(
                f"water action exceeds max {limits.max_water_seconds}s (got {seconds}s)"
            )
        if limits.allowed_beds and bed_id not in limits.allowed_beds:
            raise UnsafeActionError(f"bed {bed_id!r} not in allowed set")

    elif action.kind == "move":
        _check_move(action, limits)

    log.info("safety: approved %s", action)
    return action

