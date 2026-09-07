"""Auto-store short facts after watering and NDRE — not rules."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..memory import get_memory_store

log = logging.getLogger(__name__)


def _payload(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    params = result.get("params")
    return params if isinstance(params, dict) else result


def _coords(
    xyz: list[float] | None, payload: dict[str, Any]
) -> tuple[float | None, float | None, float | None]:
    gantry = payload.get("gantry_mm")
    if isinstance(gantry, dict):
        try:
            return float(gantry["x"]), float(gantry["y"]), float(gantry.get("z") or 0)
        except (KeyError, TypeError, ValueError):
            pass
    if xyz and len(xyz) >= 2:
        z = float(xyz[2]) if len(xyz) > 2 else None
        return float(xyz[0]), float(xyz[1]), z
    return None, None, None


def record_tool_fact(
    name: str,
    args: dict[str, Any],
    result: Any,
    *,
    xyz: list[float] | None = None,
    store: Any | None = None,
) -> None:
    if not isinstance(result, dict):
        return
    if result.get("status") in {"error", "proposed", "skipped"} or result.get("error"):
        return
    payload = _payload(result)
    x, y, z = _coords(xyz, payload)
    pose = f"x={x:.0f}, y={y:.0f}" if x is not None and y is not None else ""
    where = f" at {pose}" if pose else ""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    try:
        memory = store or get_memory_store()
        if name == "water":
            memory.remember(
                content=f"{when} Pump ran for the whole bed (~60s).",
                category="treatment",
                tags=["auto", "water"],
                importance=0.7,
            )
            memory.collapse_auto_entries()
        elif name == "capture_ndre":
            interp = payload.get("interpretation") if isinstance(payload.get("interpretation"), dict) else {}
            ndre = payload.get("ndre") if isinstance(payload.get("ndre"), dict) else {}
            hint = interp.get("action_hint") or interp.get("advice") or payload.get("summary") or "NDRE captured"
            mean = ndre.get("mean")
            mean_text = f", mean={mean}" if mean is not None else ""
            memory.remember(
                content=f"{when} {hint}{mean_text}{where}.",
                category="agronomy",
                x=x,
                y=y,
                z=z,
                tags=["auto", "ndre"],
                importance=0.6,
            )
            memory.collapse_auto_entries()
    except Exception:  # noqa: BLE001
        log.warning("auto fact for %s failed", name, exc_info=True)
