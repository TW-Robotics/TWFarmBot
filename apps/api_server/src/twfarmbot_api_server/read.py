"""Read-only robot probes for the UI.

Read handlers do NOT go through ``ActionRegistry`` because there's no
``Action`` envelope and no safety rule to apply — they're pure GETs.
They use the active robot backend from ``watering_service``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from spatial_service import get_snapshot
from twfarmbot_core.config import load_yaml_config
from watering_service import get_backend

log = logging.getLogger("twfarmbot.api_server.read")


router = APIRouter(tags=["read"])


@router.get("/position")
def get_position() -> dict[str, Any]:
    try:
        xyz = get_backend().get_xyz()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"position read failed: {err}"
        ) from err
    return {"xyz": xyz}


@router.get("/status")
def get_status() -> dict[str, Any]:
    try:
        state = get_backend().get_status()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"status read failed: {err}"
        ) from err
    return {"state": state}


@router.get("/pin/{pin}")
def get_pin(pin: int, mode: str = "digital") -> dict[str, Any]:
    try:
        value = get_backend().read_pin(pin, mode)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"pin read failed: {err}"
        ) from err
    return {"pin": pin, "mode": mode, "value": value}


@router.get("/messages")
def get_messages() -> dict[str, Any]:
    # Local-only stack has no MQTT message queue.
    return {"last_messages": get_backend().get_last_messages()}


@router.get("/images")
def get_images(limit: int = 10, refresh: bool = False) -> dict[str, Any]:
    """Newest photos captured by the local camera."""
    limit = max(1, min(limit, 50))
    try:
        images = get_backend().get_images(limit, refresh=refresh)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"image read failed: {err}"
        ) from err
    return {"images": images}


@router.get("/pins")
def get_pins() -> dict[str, Any]:
    """Named pins from ``configs/dev.yaml`` for the UI pin grid."""
    return {"pins": load_yaml_config().get("pins", []) or []}


@router.get("/positions")
def get_positions() -> dict[str, Any]:
    """Preset gantry positions for the UI Move widget."""
    return {"positions": load_yaml_config().get("positions", []) or []}


@router.get("/garden")
def get_garden() -> dict[str, Any]:
    """Configured world model composed with the latest cached robot position."""
    try:
        return get_snapshot(get_backend().get_xyz())
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"garden read failed: {err}"
        ) from err
