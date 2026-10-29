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


_INF = float("inf")
_DEFAULT_MIN = {"x": 0.0, "y": 0.0, "z": -_INF}
_DEFAULT_MAX = {"x": 1900.0, "y": 650.0, "z": _INF}


@dataclass(frozen=True)
class SafetyLimits:
    max_water_seconds: float = 60.0
    enforce_workspace: bool = True
    min_axis_mm: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_MIN))
    max_axis_mm: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_MAX))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _cap(raw: str | None, fallback: float) -> float:
    if raw is None or not str(raw).strip():
        return fallback
    text = str(raw).strip().lower()
    if text in {"inf", "+inf", "none", "unbounded", "-"}:
        return _INF
    if text in {"-inf"}:
        return -_INF
    return float(raw)


def _axis_pair(raw: object, lo: float, hi: float) -> tuple[float, float]:
    if raw is None:
        return lo, hi
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        a, b = raw
        return (
            -_INF if a is None else float(a),
            _INF if b is None else float(b),
        )
    return lo, float(raw)


def load_limits() -> SafetyLimits:
    cfg: dict = {}
    try:
        from twfarmbot_core.config import load_yaml_config

        cfg = load_yaml_config().get("safety") or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    ws = cfg.get("workspace") or {}
    lo = dict(_DEFAULT_MIN)
    hi = dict(_DEFAULT_MAX)
    for axis in ("x", "y", "z"):
        lo[axis], hi[axis] = _axis_pair(ws.get(axis), lo[axis], hi[axis])
    water = cfg.get("max_water_seconds", 60)
    enforce = bool(cfg.get("enforce_workspace", True))
    if "FARMBOT_ENFORCE_WORKSPACE" in os.environ:
        enforce = _env_flag("FARMBOT_ENFORCE_WORKSPACE")
    if "FARMBOT_MAX_WATER_SECONDS" in os.environ:
        water = os.environ["FARMBOT_MAX_WATER_SECONDS"]
    for axis, env_name in (
        ("x", "FARMBOT_MAX_AXIS_X"),
        ("y", "FARMBOT_MAX_AXIS_Y"),
        ("z", "FARMBOT_MAX_AXIS_Z"),
    ):
        if env_name in os.environ:
            hi[axis] = _cap(os.environ.get(env_name), hi[axis])
    for axis, env_name in (
        ("x", "FARMBOT_MIN_AXIS_X"),
        ("y", "FARMBOT_MIN_AXIS_Y"),
        ("z", "FARMBOT_MIN_AXIS_Z"),
    ):
        if env_name in os.environ:
            lo[axis] = _cap(os.environ.get(env_name), lo[axis])
    return SafetyLimits(
        max_water_seconds=float(water),
        enforce_workspace=enforce,
        min_axis_mm=lo,
        max_axis_mm=hi,
    )


def format_workspace_box(limits: SafetyLimits | None = None) -> str:
    """Prompt text: allowed millimetres from home (0,0,0)."""
    limits = limits or load_limits()

    def _fmt(axis: str) -> str:
        a = limits.min_axis_mm.get(axis, 0.0)
        b = limits.max_axis_mm.get(axis, _INF)
        if not math.isfinite(a) and not math.isfinite(b):
            return f"{axis} unconstrained"
        lo = f"{a:g}" if math.isfinite(a) else "open"
        hi = f"{b:g}" if math.isfinite(b) else "open"
        return f"{axis} {lo}..{hi}"

    box = ", ".join(_fmt(a) for a in ("x", "y", "z"))
    if not limits.enforce_workspace:
        return f"Home is (0,0,0) after find_home. Travel box (not enforced): {box} mm."
    return (
        f"Home is (0,0,0) after find_home. Hard travel box (mm): {box}. "
        "Do not request poses outside it; tools reject them."
    )


def workspace_payload(limits: SafetyLimits | None = None) -> dict[str, dict[str, float | None]]:
    """JSON limits for the UI map. ``null`` means that side is open."""
    limits = limits or load_limits()

    def _side(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    return {
        axis: {
            "min": _side(limits.min_axis_mm.get(axis, 0.0)),
            "max": _side(limits.max_axis_mm.get(axis, _INF)),
        }
        for axis in ("x", "y", "z")
    }


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
    lo = limits.min_axis_mm.get(axis, 0.0)
    hi = limits.max_axis_mm.get(axis, _INF)
    if value < lo or value > hi:
        raise UnsafeActionError(
            f"move action {axis}={value} outside [{lo:g}, {hi:g}] mm from home"
        )


def _read_xyz() -> tuple[float, float, float]:
    from farmbot_gateway import farmbot

    start = farmbot.backend.get_xyz()
    if isinstance(start, dict):
        return (
            float(start.get("x", 0)),
            float(start.get("y", 0)),
            float(start.get("z", 0)),
        )
    if isinstance(start, (list, tuple)) and len(start) >= 3:
        return float(start[0]), float(start[1]), float(start[2])
    raise UnsafeActionError("unexpected position shape")


def _check_move_axis(action: Action, limits: SafetyLimits) -> None:
    axis = str(action.params.get("axis") or "").strip().lower()
    if axis not in {"x", "y", "z"}:
        raise UnsafeActionError("move_axis needs axis x, y, or z")
    try:
        dist = float(action.params.get("distance"))
    except (TypeError, ValueError) as err:
        raise UnsafeActionError("move_axis distance must be numeric") from err
    if not limits.enforce_workspace:
        return
    try:
        x, y, z = _read_xyz()
    except UnsafeActionError:
        raise
    except Exception as err:  # noqa: BLE001
        raise UnsafeActionError(f"move_axis cannot read position: {err}") from err
    cur = {"x": x, "y": y, "z": z}[axis]
    _check_axis_value(axis, cur + dist, limits)


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
    if action.params.get("seconds") is None:
        return
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
    water_params: dict = {}
    if action.params.get("seconds") is not None:
        water_params["seconds"] = action.params["seconds"]
    _check_water(Action(kind="water", params=water_params), limits)


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

    try:
        separation = band_separation_mm()
        x0, y0, z0 = _read_xyz()
    except SpectralAnalysisError as err:
        raise UnsafeActionError(f"capture_ndre calibration: {err}") from err
    except UnsafeActionError:
        raise
    except Exception as err:  # noqa: BLE001
        raise UnsafeActionError(f"capture_ndre cannot read position: {err}") from err

    if not limits.enforce_workspace:
        return
    _check_xyz(
        x0 + float(separation.get("x", 0)),
        y0 + float(separation.get("y", 0)),
        z0 + float(separation.get("z", 0)),
        limits,
    )


def _check_scan_ndre(action: Action, limits: SafetyLimits) -> None:
    axis = str(action.params.get("axis") or "y").strip().lower()
    if axis not in {"x", "y"}:
        raise UnsafeActionError("scan_ndre axis must be 'x' or 'y'")
    if action.params.get("end_mm") is None:
        raise UnsafeActionError("scan_ndre action needs end_mm")
    try:
        step = float(action.params.get("step_mm") or 100)
        end = float(action.params["end_mm"])
        start = float(action.params["start_mm"]) if action.params.get("start_mm") is not None else 0.0
        z_param = action.params.get("z")
        z_value = float(z_param) if z_param is not None else None
    except (TypeError, ValueError) as err:
        raise UnsafeActionError(f"scan_ndre params must be numeric: {err}") from err
    if step < 50 or step > 800:
        raise UnsafeActionError(f"scan_ndre step_mm must be 50..800, got {step}")
    stops = abs(end - start) / step + 1
    if stops > 12:
        raise UnsafeActionError(
            "scan_ndre would take too many stops; increase step_mm"
        )
    x0, y0, z0 = 0.0, 0.0, 0.0
    if limits.enforce_workspace:
        try:
            x0, y0, z0 = _read_xyz()
        except Exception as err:  # noqa: BLE001
            raise UnsafeActionError(f"scan_ndre cannot read position: {err}") from err
    if z_value is None:
        z_value = z0
    if axis == "y":
        _check_xyz(x0, start, z_value, limits)
        _check_xyz(x0, end, z_value, limits)
    else:
        _check_xyz(start, y0, z_value, limits)
        _check_xyz(end, y0, z_value, limits)
    _check_capture_ndre(Action(kind="capture_ndre", params={}), limits)


def _check_run_script(action: Action, limits: SafetyLimits) -> None:
    name = action.params.get("name")
    if not name:
        raise UnsafeActionError("run_script action needs a name")
    try:
        from spatial_service import resolve_script

        script = resolve_script(str(name))
    except ValueError as err:
        raise UnsafeActionError(str(err)) from err
    waypoints = script.get("waypoints") or []
    if not waypoints:
        raise UnsafeActionError(
            f"{script['label']!r} has no waypoints yet. Record them on the Scripts page."
        )
    _check_move_path(
        Action(kind="move_path", params={"waypoints": waypoints}), limits
    )


register("move", _check_move)
register("move_axis", _check_move_axis)
register("move_path", _check_move_path)
register("water", _check_water)
register("goto_named", _check_goto_named)
register("run_script", _check_run_script)
register("water_zone", _check_water_zone)
register("inspect_zone", _check_inspect_zone)
register("capture", _check_capture)
register("capture_ndre", _check_capture_ndre)
register("scan_ndre", _check_scan_ndre)


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
