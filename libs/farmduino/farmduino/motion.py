"""FarmBot movement helpers (speed units, firmware parameter IDs)."""

from __future__ import annotations

# Firmware parameter numbers (see farmbot-arduino-firmware README).
PARAM_ACCEL_X = 41
PARAM_ACCEL_Y = 42
PARAM_ACCEL_Z = 43
PARAM_MIN_SPD_X = 61
PARAM_MIN_SPD_Y = 62
PARAM_MIN_SPD_Z = 63
PARAM_MAX_SPD_X = 71
PARAM_MAX_SPD_Y = 72
PARAM_MAX_SPD_Z = 73
PARAM_STEALTH_X = 165
PARAM_STEALTH_Y = 166
PARAM_STEALTH_Z = 167

DEFAULT_MIN_STEPS_S = 50
DEFAULT_MAX_STEPS_S = 400
DEFAULT_ACCEL_STEPS = 600


def resolve_axis_speed(
    speed: float,
    *,
    min_steps_s: float = DEFAULT_MIN_STEPS_S,
    max_steps_s: float = DEFAULT_MAX_STEPS_S,
) -> int:
    """Convert API speed to G00 axis speed in integer steps/s.

    Values in ``1..100`` are treated as percent of max axis speed (FarmBot UI
    convention). Values above ``100`` are passed through as raw steps/s.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if speed <= 100:
        pct = max(1.0, min(100.0, float(speed)))
        lo = min(min_steps_s, max_steps_s)
        hi = max(min_steps_s, max_steps_s)
        return int(round(lo + (hi - lo) * (pct - 1.0) / 99.0))
    return int(round(min(float(speed), max_steps_s)))


def resolve_axis_speeds(
    speed: float,
    *,
    min_steps_s: tuple[int, int, int] = (
        DEFAULT_MIN_STEPS_S,
        DEFAULT_MIN_STEPS_S,
        DEFAULT_MIN_STEPS_S,
    ),
    max_steps_s: tuple[int, int, int] = (
        DEFAULT_MAX_STEPS_S,
        DEFAULT_MAX_STEPS_S,
        DEFAULT_MAX_STEPS_S,
    ),
) -> tuple[int, int, int]:
    """Return integer (A, B, C) steps/s for X, Y, Z."""
    mins = (
        min_steps_s[0],
        min_steps_s[1],
        min_steps_s[2],
    )
    maxs = (
        max_steps_s[0],
        max_steps_s[1],
        max_steps_s[2],
    )
    return (
        resolve_axis_speed(speed, min_steps_s=mins[0], max_steps_s=maxs[0]),
        resolve_axis_speed(speed, min_steps_s=mins[1], max_steps_s=maxs[1]),
        resolve_axis_speed(speed, min_steps_s=mins[2], max_steps_s=maxs[2]),
    )
