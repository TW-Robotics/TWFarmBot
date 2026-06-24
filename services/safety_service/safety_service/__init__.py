"""safety_service — gates every action that affects the real world.

Per the README: *Any code path that ultimately moves the FarmBot (watering,
weeding, tool changes, …) must pass through safety_service before it reaches
farmbot_gateway.*

Validators are registered by action kind. Adding a new safety rule is now a
single line: ``register("my_kind", my_validator)``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from twfarmbot_core.domain.action import Action

log = logging.getLogger(__name__)


class UnsafeActionError(ValueError):
    """Raised when an action fails a safety rule."""


@dataclass(frozen=True)
class SafetyLimits:
    max_water_seconds: float = 300.0
    max_axis_mm: dict[str, float] = field(
        default_factory=lambda: {"x": 3000.0, "y": 1500.0, "z": 800.0}
    )


def load_limits() -> SafetyLimits:
    return SafetyLimits(
        max_water_seconds=float(os.getenv("FARMBOT_MAX_WATER_SECONDS", "300")),
        max_axis_mm={
            "x": float(os.getenv("FARMBOT_MAX_AXIS_X", "3000")),
            "y": float(os.getenv("FARMBOT_MAX_AXIS_Y", "1500")),
            "z": float(os.getenv("FARMBOT_MAX_AXIS_Z", "800")),
        },
    )


Validator = Callable[[Action, SafetyLimits], None]

_VALIDATORS: dict[str, Validator] = {}


def register(kind: str, validator: Validator) -> None:
    """Register a safety validator for an action kind."""
    if kind in _VALIDATORS:
        raise ValueError(f"safety validator already registered for {kind!r}")
    _VALIDATORS[kind] = validator


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


def _check_water(action: Action, limits: SafetyLimits) -> None:
    seconds = float(action.params.get("seconds", 0.0))
    if seconds <= 0:
        raise UnsafeActionError(f"water action needs positive seconds, got {seconds}")
    if seconds > limits.max_water_seconds:
        raise UnsafeActionError(
            f"water action exceeds max {limits.max_water_seconds}s (got {seconds}s)"
        )


def _check_move_path(action: Action, limits: SafetyLimits) -> None:
    waypoints = action.params.get("waypoints")
    if not isinstance(waypoints, list):
        raise UnsafeActionError("move_path action needs a list of waypoints")
    for idx, wp in enumerate(waypoints):
        if not isinstance(wp, dict):
            raise UnsafeActionError(f"waypoint {idx} must be an object")
        for axis in ("x", "y", "z"):
            if axis not in wp:
                raise UnsafeActionError(f"waypoint {idx} needs {axis!r}")
            try:
                value = float(wp[axis])
            except (TypeError, ValueError) as err:
                raise UnsafeActionError(
                    f"waypoint {idx} {axis!r} must be numeric, got {wp[axis]!r}"
                ) from err
            cap = limits.max_axis_mm.get(axis, float("inf"))
            if abs(value) > cap:
                raise UnsafeActionError(
                    f"waypoint {idx} {axis}={value} exceeds |max| {cap} mm"
                )

    water_pin = action.params.get("water_pin")
    if water_pin is not None:
        try:
            int(water_pin)
        except (TypeError, ValueError) as err:
            raise UnsafeActionError(
                f"water_pin must be an integer, got {water_pin!r}"
            ) from err


register("move", _check_move)
register("move_path", _check_move_path)
register("water", _check_water)


def validate(action: Action, *, limits: SafetyLimits | None = None) -> Action:
    """Check an Action against the safety rules. Returns it unchanged on pass.

    Raises :class:`UnsafeActionError` if the action is rejected.
    """
    limits = limits or load_limits()
    validator = _VALIDATORS.get(action.kind)
    if validator is not None:
        validator(action, limits)
    log.info("safety: approved %s", action)
    return action
