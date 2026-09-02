"""Read-only FarmBot probes for the UI.

Read handlers do NOT go through ``ActionRegistry`` because there's no
``Action`` envelope and no safety rule to apply — they're pure GETs.
They still use the same ``FarmBotBackend`` so the UI never talks to the Pi.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from farmbot_client.client import base_url

from spatial_service import get_snapshot
from twfarmbot_core.config import load_yaml_config
from vision_service import capture_path, list_captures
from watering_service.backends import farmbot

log = logging.getLogger("twfarmbot.api_server.read")


router = APIRouter(tags=["read"])


def _photo_attachment_url(photo_id: str) -> str:
    return f"/photos/{photo_id}"


def _capture_attachment_url(artifact_id: str, band: str) -> str:
    return f"/captures/{artifact_id}/{band}"


def _sort_ts(row: dict[str, Any]) -> float:
    raw = row.get("created_at")
    if raw is None:
        return 0.0
    try:
        return float(int(raw))
    except (TypeError, ValueError):
        return 0.0


def _with_attachment_urls(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        row = dict(image)
        if row.get("attachment_url"):
            out.append(row)
            continue
        if row.get("artifact_id") and row.get("band"):
            row["attachment_url"] = _capture_attachment_url(
                str(row["artifact_id"]), str(row["band"])
            )
        elif row.get("id"):
            row["attachment_url"] = _photo_attachment_url(str(row["id"]))
        out.append(row)
    return out


@router.get("/position")
def get_position(refresh: bool = False) -> dict[str, Any]:
    try:
        xyz = (
            farmbot.backend.refresh_xyz()
            if refresh
            else farmbot.backend.get_xyz()
        )
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"farmbot read failed: {err}"
        ) from err
    return {"xyz": xyz}


@router.get("/status")
def get_status(path: str | None = None) -> dict[str, Any]:
    try:
        bot = farmbot.backend._bot()
        state = (
            bot.read_status(path=path)
            if hasattr(bot, "read_status")
            else bot.get_state()
        )
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
    """FarmBot photos plus payload USB capture artifacts."""
    limit = max(1, min(limit, 50))
    images: list[dict[str, Any]] = []
    try:
        images.extend(farmbot.backend.get_images(limit, refresh=refresh))
    except Exception as err:  # noqa: BLE001
        log.warning("farmbot image read failed: %s", err)
    try:
        images.extend(list_captures(limit))
    except Exception as err:  # noqa: BLE001
        log.warning("capture list failed: %s", err)
    images.sort(key=_sort_ts, reverse=True)
    return {"images": _with_attachment_urls(images[:limit])}


@router.get("/photos/{photo_id}")
def get_photo(photo_id: str) -> Response:
    url = f"{base_url()}/photos/{photo_id}"
    try:
        image = httpx.get(url, timeout=30.0)
    except httpx.HTTPError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err
    if image.status_code == 404:
        raise HTTPException(status_code=404, detail="photo not found")
    if image.status_code >= 400:
        raise HTTPException(status_code=502, detail=image.text)
    return Response(content=image.content, media_type="image/jpeg")


@router.get("/captures")
def get_captures(limit: int = 10) -> dict[str, Any]:
    limit = max(1, min(limit, 50))
    try:
        captures = list_captures(limit)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(err)) from err
    return {"captures": _with_attachment_urls(captures)}


@router.get("/captures/{artifact_id}/{band}")
def get_capture_file(artifact_id: str, band: str) -> FileResponse:
    path = capture_path(artifact_id, band)
    if path is None:
        raise HTTPException(status_code=404, detail="capture not found")
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media)


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
