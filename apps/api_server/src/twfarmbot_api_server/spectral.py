"""NIR + red-edge paired capture and NDRE analysis."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from safety_service import UnsafeActionError, validate
from twfarmbot_core.domain import Action
from vision_service import capture, capture_path
from vision_service.spectral_analysis import (
    SpectralAnalysisError,
    analyze_spectral_pair,
    band_separation_mm,
)
from vision_service.spectral_calib import (
    DEFAULT_CONFIG_PATH,
    build_dot_calibration,
    load_image_align,
    normalize_image_align,
    save_calibration,
)
from watering_service.backends import farmbot

from .handlers.move import handle_move

log = logging.getLogger("twfarmbot.api_server.spectral")

router = APIRouter(tags=["spectral"])


class OffsetPayload(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class ImageAlignPayload(BaseModel):
    dx_px: float = 0.0
    dy_px: float = 0.0
    rotation_deg: float = 0.0


class CalibrationPayload(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    image_align: ImageAlignPayload | None = None


class SpectralPairPayload(BaseModel):
    return_to_start: bool = Field(
        default=True,
        description="Move back to the starting gantry pose after capture.",
    )
    offset_mm: OffsetPayload | None = None
    image_align: ImageAlignPayload | None = None


class LiveNdrePayload(BaseModel):
    nir_artifact_id: str = Field(..., min_length=1)
    image_align: ImageAlignPayload | None = None


class AnalyzePairPayload(BaseModel):
    nir_artifact_id: str = Field(..., min_length=1)
    rededge_artifact_id: str = Field(..., min_length=1)
    image_align: ImageAlignPayload | None = None


def _xyz_dict(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict):
        return {
            "x": float(raw.get("x", 0.0)),
            "y": float(raw.get("y", 0.0)),
            "z": float(raw.get("z", 0.0)),
        }
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return {"x": float(raw[0]), "y": float(raw[1]), "z": float(raw[2])}
    raise SpectralAnalysisError(f"unexpected position shape: {raw!r}")


def _align_dict(payload: ImageAlignPayload | None) -> dict[str, float] | None:
    if payload is None:
        return None
    return normalize_image_align(payload.model_dump())


def capture_spectral_pair(
    *,
    return_to_start: bool = True,
    offset_mm: dict[str, float] | None = None,
    image_align: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Capture NIR, offset gantry, capture red-edge, compute NDRE."""
    separation = offset_mm if offset_mm is not None else band_separation_mm()
    start = _xyz_dict(farmbot.backend.get_xyz())
    nir_pose = dict(start)
    rededge_pose = {
        "x": start["x"] + separation["x"],
        "y": start["y"] + separation["y"],
        "z": start["z"] + separation["z"],
    }

    nir_id = capture("nir")
    try:
        move = Action(kind="move", params=rededge_pose)
        validate(move)
        handle_move(move)
    except UnsafeActionError as err:
        raise SpectralAnalysisError(f"move blocked: {err}") from err

    rededge_id = capture("rededge")

    if return_to_start:
        try:
            validate(Action(kind="move", params=start))
            handle_move(Action(kind="move", params=start))
        except Exception as err:  # noqa: BLE001
            log.warning("return move after spectral pair failed: %s", err)

    nir_path = capture_path(nir_id, "nir")
    rededge_path = capture_path(rededge_id, "rededge")
    if nir_path is None or rededge_path is None:
        raise SpectralAnalysisError("capture files missing after paired capture")

    metrics = analyze_spectral_pair(
        nir_path,
        rededge_path,
        image_align=image_align,
        preview_artifact_id=nir_id,
    )
    return {
        "status": "ok",
        "calibration": {
            "band_separation_mm": separation,
            "image_align": metrics.get("image_align"),
        },
        "gantry_mm": {"nir": nir_pose, "rededge": rededge_pose},
        "nir": {
            "artifact_id": nir_id,
            "band": "nir",
            "attachment_url": f"/captures/{nir_id}/nir",
        },
        "rededge": {
            "artifact_id": rededge_id,
            "band": "rededge",
            "attachment_url": f"/captures/{rededge_id}/rededge",
        },
        "metrics": metrics,
    }


def live_ndre(
    nir_artifact_id: str,
    *,
    image_align: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Capture a fresh red-edge frame and compute NDRE against a frozen NIR."""
    nir_id = str(nir_artifact_id).strip()
    nir_path = capture_path(nir_id, "nir")
    if nir_path is None:
        raise SpectralAnalysisError(f"frozen NIR capture not found: {nir_id}")

    rededge_id = capture("rededge")
    rededge_path = capture_path(rededge_id, "rededge")
    if rededge_path is None:
        raise SpectralAnalysisError("red-edge capture file missing")

    metrics = analyze_spectral_pair(
        nir_path,
        rededge_path,
        image_align=image_align,
        preview_artifact_id=nir_id,
    )
    return {
        "status": "ok",
        "gantry_mm": _xyz_dict(farmbot.backend.get_xyz()),
        "nir": {
            "artifact_id": nir_id,
            "band": "nir",
            "attachment_url": f"/captures/{nir_id}/nir",
        },
        "rededge": {
            "artifact_id": rededge_id,
            "band": "rededge",
            "attachment_url": f"/captures/{rededge_id}/rededge",
        },
        "metrics": metrics,
    }


@router.get("/spectral/calibration")
def get_spectral_calibration() -> dict[str, Any]:
    """Return gantry band separation + image-space align params."""
    try:
        separation = band_separation_mm()
    except SpectralAnalysisError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return {
        "path": str(DEFAULT_CONFIG_PATH),
        "band_separation_mm": separation,
        "image_align": load_image_align(),
    }


@router.put("/spectral/calibration")
def put_spectral_calibration(payload: CalibrationPayload) -> dict[str, Any]:
    """Persist gantry offset + image align into spectral_calibration.yaml."""
    separation = {"x": payload.x, "y": payload.y, "z": payload.z}
    align = _align_dict(payload.image_align) or load_image_align()
    doc = build_dot_calibration(separation, image_align=align)
    path = save_calibration(doc, DEFAULT_CONFIG_PATH)
    return {
        "status": "ok",
        "path": str(path),
        "band_separation_mm": separation,
        "image_align": align,
    }


@router.post("/spectral/analyze-pair")
def post_spectral_analyze_pair(payload: AnalyzePairPayload) -> dict[str, Any]:
    """Recompute NDRE for existing captures with a live image_align override."""
    try:
        nir_path = capture_path(payload.nir_artifact_id, "nir")
        rededge_path = capture_path(payload.rededge_artifact_id, "rededge")
        if nir_path is None or rededge_path is None:
            raise SpectralAnalysisError("capture artifacts not found")
        metrics = analyze_spectral_pair(
            nir_path,
            rededge_path,
            image_align=_align_dict(payload.image_align),
            preview_artifact_id=payload.nir_artifact_id,
        )
        return {
            "status": "ok",
            "nir": {
                "artifact_id": payload.nir_artifact_id,
                "band": "nir",
                "attachment_url": f"/captures/{payload.nir_artifact_id}/nir",
            },
            "rededge": {
                "artifact_id": payload.rededge_artifact_id,
                "band": "rededge",
                "attachment_url": f"/captures/{payload.rededge_artifact_id}/rededge",
            },
            "metrics": metrics,
        }
    except SpectralAnalysisError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:  # noqa: BLE001
        log.exception("spectral analyze-pair failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(err).__name__}: {err}",
        ) from err


@router.post("/spectral/live-ndre")
def post_spectral_live_ndre(payload: LiveNdrePayload) -> dict[str, Any]:
    """Capture red-edge now and compute NDRE vs a previously frozen NIR frame."""
    try:
        return live_ndre(
            payload.nir_artifact_id,
            image_align=_align_dict(payload.image_align),
        )
    except SpectralAnalysisError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:  # noqa: BLE001
        log.exception("spectral live-ndre failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(err).__name__}: {err}",
        ) from err


@router.post("/spectral/capture-pair")
def post_spectral_capture_pair(payload: SpectralPairPayload) -> dict[str, Any]:
    """Capture NIR, move by offset, capture red-edge, return NDRE."""
    try:
        offset = (
            {
                "x": payload.offset_mm.x,
                "y": payload.offset_mm.y,
                "z": payload.offset_mm.z,
            }
            if payload.offset_mm is not None
            else None
        )
        return capture_spectral_pair(
            return_to_start=payload.return_to_start,
            offset_mm=offset,
            image_align=_align_dict(payload.image_align),
        )
    except SpectralAnalysisError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:  # noqa: BLE001
        log.exception("spectral capture-pair failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(err).__name__}: {err}",
        ) from err
