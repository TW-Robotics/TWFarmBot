"""Read-only FarmBot probes for the UI.

Read handlers do NOT go through ``ActionRegistry`` because there's no
``Action`` envelope and no safety rule to apply — they're pure GETs.
They still use the same ``FarmBotBackend`` so the UI never imports
``farmbot-py`` directly.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from spatial_service import get_snapshot
from twfarmbot_core.config import load_yaml_config
from watering_service.backends import farmbot

log = logging.getLogger("twfarmbot.api_server.read")


router = APIRouter(tags=["read"])


@router.get("/position")
def get_position() -> dict[str, Any]:
    try:
        xyz = farmbot.backend.get_xyz()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"farmbot read failed: {err}"
        ) from err
    return {"xyz": xyz}


@router.get("/status")
def get_status(path: str | None = None) -> dict[str, Any]:
    try:
        state = farmbot.backend._bot().read_status(path=path)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"farmbot read failed: {err}"
        ) from err
    return {"path": path, "state": state}


@router.get("/pin/{pin}")
def get_pin(pin: int, mode: str = "digital") -> dict[str, Any]:
    try:
        value = farmbot.backend.read_pin(pin, mode)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"farmbot read failed: {err}"
        ) from err
    return {"pin": pin, "mode": mode, "value": value}


@router.get("/messages")
def get_messages() -> dict[str, Any]:
    try:
        last = farmbot.backend.get_last_messages()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"farmbot read failed: {err}"
        ) from err
    return {"last_messages": last}


@router.get("/images")
def get_images(limit: int = 10, refresh: bool = False) -> dict[str, Any]:
    """Newest photos uploaded by the FarmBot camera."""
    limit = max(1, min(limit, 50))
    try:
        images = farmbot.backend.get_images(limit, refresh=refresh)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"farmbot image read failed: {err}"
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
        return get_snapshot(farmbot.backend.get_xyz())
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"garden read failed: {err}"
        ) from err
