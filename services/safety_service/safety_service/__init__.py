"""safety_service — gates every action that affects the real world.

Per the README: *Any code path that ultimately moves the FarmBot (watering,
weeding, tool changes, …) must pass through safety_service before it reaches
the local FarmBot HTTP client.

Validators are registered by action kind. Adding a new safety rule is now a
single line: ``register("my_kind", my_validator)``.
"""

from __future__ import annotations

import logging
import math
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
    # When False (default), axis magnitude is not gated — operator is present.
    enforce_workspace: bool = False
    max_axis_mm: dict[str, float] = field(
        default_factory=lambda: {"x": 650.0, "y": 1900.0, "z": 300.0}
    )


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def load_limits() -> SafetyLimits:
    return SafetyLimits(
        max_water_seconds=float(os.getenv("FARMBOT_MAX_WATER_SECONDS", "300")),
        enforce_workspace=_env_flag("FARMBOT_ENFORCE_WORKSPACE", "0"),
        max_axis_mm={
            "x": float(os.getenv("FARMBOT_MAX_AXIS_X", "650")),
            "y": float(os.getenv("FARMBOT_MAX_AXIS_Y", "1900")),
            "z": float(os.getenv("FARMBOT_MAX_AXIS_Z", "300")),
        },
    )


Validator = Callable[[Action, SafetyLimits], None]

_VALIDATORS: dict[str, Validator] = {}


def register(kind: str, validator: Validator) -> None:
    """Register a safety validator for an action kind."""
    if kind in _VALIDATORS:
        raise ValueError(f"safety validator already registered for {kind!r}")
    _VALIDATORS[kind] = validator


def _check_axis_value(axis: str, value: float, limits: SafetyLimits) -> None:
    if not limits.enforce_workspace:
        return
    cap = limits.max_axis_mm.get(axis, float("inf"))
    if abs(value) > cap:
        raise UnsafeActionError(
            f"move action {axis}={value} exceeds |max| {cap} mm"
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
        _check_axis_value(axis, value, limits)


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
            try:
                _check_axis_value(axis, value, limits)
            except UnsafeActionError as err:
                raise UnsafeActionError(f"waypoint {idx} {err}") from err

    water_pin = action.params.get("water_pin")
    if water_pin is not None:
        try:
            int(water_pin)
        except (TypeError, ValueError) as err:
            raise UnsafeActionError(
                f"water_pin must be an integer, got {water_pin!r}"
            ) from err


def _check_xyz(x: float, y: float, z: float, limits: SafetyLimits) -> None:
    _check_move(Action(kind="move", params={"x": x, "y": y, "z": z}), limits)


def _resolve_or_reject(name: str) -> dict:
    from spatial_service import resolve_named

    try:
        return resolve_named(name)
    except ValueError as err:
        raise UnsafeActionError(str(err)) from err


def _check_goto_named(action: Action, limits: SafetyLimits) -> None:
    name = action.params.get("name")
    if not name:
        raise UnsafeActionError("goto_named action needs a name")
    target = _resolve_or_reject(str(name))
    z = action.params.get("z", target["z"])
    try:
        z_value = float(z)
    except (TypeError, ValueError) as err:
        raise UnsafeActionError(f"goto_named z must be numeric, got {z!r}") from err
    _check_xyz(float(target["x"]), float(target["y"]), z_value, limits)


def _check_water_zone(action: Action, limits: SafetyLimits) -> None:
    zone_id = action.params.get("zone_id")
    if not zone_id:
        raise UnsafeActionError("water_zone action needs zone_id")
    target = _resolve_or_reject(str(zone_id))
    if target.get("kind") != "zone":
        raise UnsafeActionError(f"{zone_id!r} is not a watering zone")
    z = action.params.get("z", target.get("z", 0))
    try:
        z_value = float(z)
    except (TypeError, ValueError) as err:
        raise UnsafeActionError(f"water_zone z must be numeric, got {z!r}") from err
    _check_xyz(float(target["x"]), float(target["y"]), z_value, limits)
    _check_water(
        Action(kind="water", params={"seconds": action.params.get("seconds", 0)}),
        limits,
    )


_CAPTURE_BANDS = frozenset({"rgb", "nir", "rededge", "thermal", "swir"})
_UNPINNED_CAPTURE_BANDS = frozenset({"thermal", "swir"})


def _check_capture(action: Action, limits: SafetyLimits) -> None:
    del limits
    band = action.params.get("band")
    if not band:
        raise UnsafeActionError("capture action needs band")
    key = str(band).strip().lower()
    if key not in _CAPTURE_BANDS:
        raise UnsafeActionError(
            f"capture band must be one of {sorted(_CAPTURE_BANDS)}, got {band!r}"
        )
    if key in _UNPINNED_CAPTURE_BANDS:
        raise UnsafeActionError(f"{key} capture is unavailable: bus not pinned")


def _check_inspect_zone(action: Action, limits: SafetyLimits) -> None:
    zone_id = action.params.get("zone_id")
    if not zone_id:
        raise UnsafeActionError("inspect_zone action needs zone_id")
    target = _resolve_or_reject(str(zone_id))
    if target.get("kind") != "zone":
        raise UnsafeActionError(f"{zone_id!r} is not a garden zone")
    try:
        step = float(action.params.get("step_mm", 200))
        z_value = float(action.params.get("z", 0))
    except (TypeError, ValueError) as err:
        raise UnsafeActionError(
            f"inspect_zone step_mm/z must be numeric: {err}"
        ) from err
    if step < 50 or step > 800:
        raise UnsafeActionError(f"inspect_zone step_mm must be 50..800, got {step}")
    width = float(target.get("width") or 0)
    height = float(target.get("height") or 0)
    photos = max(1, math.ceil(width / step)) * max(1, math.ceil(height / step))
    if photos > 24:
        raise UnsafeActionError(
            f"inspect_zone would take about {photos} photos; increase step_mm"
        )
    _check_xyz(float(target["x"]), float(target["y"]), z_value, limits)


def _check_capture_ndre(action: Action, limits: SafetyLimits) -> None:
    """Validate calib + optional workspace when enforce_workspace is on."""
    from vision_service.spectral_analysis import (
        SpectralAnalysisError,
        band_separation_mm,
    )
    from watering_service.backends import farmbot

    try:
        separation = band_separation_mm()
        start = farmbot.backend.get_xyz()
    except SpectralAnalysisError as err:
        raise UnsafeActionError(f"capture_ndre calibration: {err}") from err
    except Exception as err:  # noqa: BLE001
        raise UnsafeActionError(f"capture_ndre cannot read position: {err}") from err

    if not limits.enforce_workspace:
        return

    if isinstance(start, dict):
        x0, y0, z0 = float(start.get("x", 0)), float(start.get("y", 0)), float(start.get("z", 0))
    elif isinstance(start, (list, tuple)) and len(start) >= 3:
        x0, y0, z0 = float(start[0]), float(start[1]), float(start[2])
    else:
        raise UnsafeActionError("capture_ndre: unexpected position shape")

    _check_xyz(
        x0 + float(separation.get("x", 0)),
        y0 + float(separation.get("y", 0)),
        z0 + float(separation.get("z", 0)),
        limits,
    )


register("move", _check_move)
register("move_path", _check_move_path)
register("water", _check_water)
register("goto_named", _check_goto_named)
register("water_zone", _check_water_zone)
register("inspect_zone", _check_inspect_zone)
register("capture", _check_capture)
register("capture_ndre", _check_capture_ndre)


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
