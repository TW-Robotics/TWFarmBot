"""NIR / red-edge pair analysis for zone-level NDRE (no georeferencing)."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .spectral_calib import (
    DEFAULT_CONFIG_PATH,
    load_calibration,
    load_image_align,
    normalize_image_align,
)

log = logging.getLogger(__name__)

_EPS = 1e-6


class SpectralAnalysisError(ValueError):
    """Raised when spectral analysis cannot run."""


def band_separation_mm(path: str | Path | None = None) -> dict[str, float]:
    """Load red-edge gantry delta relative to nir from calibration yaml."""
    doc = load_calibration(path or DEFAULT_CONFIG_PATH)
    bands = doc.get("bands") or {}
    rededge = bands.get("rededge") or {}
    separation = rededge.get("gantry_offset_mm")
    if not isinstance(separation, dict):
        computed = doc.get("computed") or {}
        separation = computed.get("band_separation_mm")
    if not isinstance(separation, dict):
        raise SpectralAnalysisError("spectral calibration missing band_separation_mm")
    return {
        "x": float(separation.get("x", 0.0)),
        "y": float(separation.get("y", 0.0)),
        "z": float(separation.get("z", 0.0)),
    }


def read_grayscale(path: str | Path) -> np.ndarray:
    """Load a capture as a single-channel float32 array."""
    file_path = Path(path)
    if not file_path.is_file():
        raise SpectralAnalysisError(f"capture not found: {file_path}")
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SpectralAnalysisError(f"cannot read image: {file_path}")
    return image.astype(np.float32)


def _align_pair(nir: np.ndarray, rededge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if nir.shape == rededge.shape:
        return nir, rededge
    rededge_resized = cv2.resize(
        rededge,
        (nir.shape[1], nir.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    log.warning(
        "resized red-edge %s -> %s for NDRE",
        rededge.shape,
        rededge_resized.shape,
    )
    return nir, rededge_resized


def apply_image_align(
    rededge: np.ndarray,
    *,
    dx_px: float = 0.0,
    dy_px: float = 0.0,
    rotation_deg: float = 0.0,
    out_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Warp red-edge onto NIR with rotation (about center) + pixel translation."""
    height, width = out_shape or rededge.shape[:2]
    if (
        abs(dx_px) < 1e-9
        and abs(dy_px) < 1e-9
        and abs(rotation_deg) < 1e-9
        and rededge.shape[:2] == (height, width)
    ):
        return rededge
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0)
    matrix[0, 2] += float(dx_px)
    matrix[1, 2] += float(dy_px)
    return cv2.warpAffine(
        rededge,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _to_u8(image: np.ndarray) -> np.ndarray:
    scaled = np.clip(image, 0, 255)
    if scaled.dtype != np.uint8:
        scaled = scaled.astype(np.uint8)
    return scaled


def align_overlay_preview(nir: np.ndarray, rededge: np.ndarray) -> str:
    """False-color overlay: NIR=red, red-edge=green (yellow where aligned)."""
    overlay = np.zeros((*nir.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = _to_u8(nir)
    overlay[:, :, 1] = _to_u8(rededge)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise SpectralAnalysisError("failed to encode align overlay")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def compute_ndre(
    nir: np.ndarray,
    rededge: np.ndarray,
    *,
    image_align: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Per-pixel NDRE after optional calibrated image-space warp of red-edge."""
    nir_aligned, rededge_aligned = _align_pair(nir, rededge)
    align = normalize_image_align(image_align)
    warped = apply_image_align(
        rededge_aligned,
        dx_px=align["dx_px"],
        dy_px=align["dy_px"],
        rotation_deg=align["rotation_deg"],
        out_shape=nir_aligned.shape[:2],
    )
    denom = nir_aligned + warped
    mask = denom > _EPS
    ndre = np.zeros_like(nir_aligned, dtype=np.float32)
    ndre[mask] = (nir_aligned[mask] - warped[mask]) / denom[mask]
    return np.clip(ndre, -1.0, 1.0), warped, align


def _band_stats(values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1)
    return {
        "mean": round(float(np.mean(flat)), 2),
        "std": round(float(np.std(flat)), 2),
        "min": round(float(np.min(flat)), 2),
        "max": round(float(np.max(flat)), 2),
        "p10": round(float(np.percentile(flat, 10)), 2),
        "p90": round(float(np.percentile(flat, 90)), 2),
    }


def _ndre_stats(ndre: np.ndarray) -> dict[str, float]:
    stats = _band_stats(ndre)
    valid = ndre.reshape(-1)
    stats["vegetation_fraction"] = round(float(np.mean(valid > 0.1)), 3)
    stats["stress_fraction"] = round(float(np.mean(valid < 0.05)), 3)
    return stats


def ndre_preview_png(ndre: np.ndarray) -> bytes:
    """False-color PNG bytes for UI preview."""
    normalized = ((ndre + 1.0) * 0.5 * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".png", colored)
    if not ok:
        raise SpectralAnalysisError("failed to encode NDRE preview")
    return encoded.tobytes()


def ndre_preview_data_url(ndre: np.ndarray) -> str:
    payload = base64.b64encode(ndre_preview_png(ndre)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def interpret_ndre(ndre_stats: dict[str, float]) -> dict[str, Any]:
    """Turn NDRE metrics into a decision-oriented interpretation for the LLM."""
    mean = float(ndre_stats.get("mean") or 0.0)
    veg = float(ndre_stats.get("vegetation_fraction") or 0.0)
    stress = float(ndre_stats.get("stress_fraction") or 0.0)

    if mean < 0.05 and veg < 0.2:
        label = "bare_or_non_canopy"
        advice = (
            "Little canopy signal — likely soil, empty bed, or wrong viewpoint. "
            "Do not water based on this alone; move to a planted zone and recheck."
        )
        action_hint = "reposition"
    elif stress >= 0.4 and mean < 0.15:
        label = "likely_stressed"
        advice = (
            "High low-vigor fraction with weak NDRE — canopy may be water-stressed "
            "or sparse. Prefer water_zone / closer inspect if this is a planted bed."
        )
        action_hint = "consider_water"
    elif mean >= 0.25 and veg >= 0.4:
        label = "healthy_canopy"
        advice = (
            "Healthy canopy signal — no watering needed from NDRE alone; "
            "continue monitoring."
        )
        action_hint = "ok"
    elif mean >= 0.15:
        label = "moderate_canopy"
        advice = (
            "Moderate vegetation — watch stress_fraction; water only if the bed "
            "is known planted and soil/sensor evidence also supports it."
        )
        action_hint = "monitor"
    else:
        label = "weak_canopy"
        advice = (
            "Weak NDRE — sparse cover or poor alignment. Confirm zone, then "
            "re-run capture_ndre or inspect with RGB segmentation."
        )
        action_hint = "recheck"

    return {
        "label": label,
        "action_hint": action_hint,
        "advice": advice,
        "mean": round(mean, 3),
        "vegetation_fraction": round(veg, 3),
        "stress_fraction": round(stress, 3),
    }


def analyze_spectral_pair(
    nir_path: str | Path,
    rededge_path: str | Path,
    *,
    image_align: dict[str, float] | None = None,
    preview_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Compute NDRE and intensity stats from paired captures.

    When ``preview_artifact_id`` is set, the NDRE map is written beside the
    captures as ``{id}-ndre.png`` and ``ndre_preview`` becomes a URL path
    instead of an inline base64 data URL (keeps chat SSE small).
    """
    nir = read_grayscale(nir_path)
    rededge = read_grayscale(rededge_path)
    align = (
        normalize_image_align(image_align)
        if image_align is not None
        else load_image_align()
    )
    ndre, warped, used_align = compute_ndre(nir, rededge, image_align=align)
    ndre_stats = _ndre_stats(ndre)
    interpretation = interpret_ndre(ndre_stats)
    if preview_artifact_id:
        from .capture import write_capture_file

        write_capture_file(preview_artifact_id, "ndre", ndre_preview_png(ndre))
        preview: str = f"/captures/{preview_artifact_id}/ndre"
    else:
        preview = ndre_preview_data_url(ndre)
    return {
        "nir": _band_stats(nir),
        "rededge": _band_stats(rededge),
        "ndre": ndre_stats,
        "interpretation": interpretation,
        "image_align": used_align,
        "summary": _summary_text(ndre_stats, used_align, interpretation),
        "ndre_preview": preview,
        "align_preview": align_overlay_preview(nir, warped),
        "shape_px": {"width": int(nir.shape[1]), "height": int(nir.shape[0])},
    }


def _summary_text(
    ndre_stats: dict[str, float],
    image_align: dict[str, float] | None = None,
    interpretation: dict[str, Any] | None = None,
) -> str:
    mean = ndre_stats.get("mean", 0.0)
    stress = ndre_stats.get("stress_fraction", 0.0)
    veg = ndre_stats.get("vegetation_fraction", 0.0)
    metrics = (
        f"NDRE mean {mean:.2f}; "
        f"vegetation-like {veg * 100:.0f}%; "
        f"low-vigor {stress * 100:.0f}%"
    )
    if interpretation:
        text = (
            f"{interpretation.get('label')} "
            f"({interpretation.get('action_hint')}): "
            f"{interpretation.get('advice')} "
            f"[{metrics}]"
        )
    else:
        text = metrics
    if image_align and (
        abs(image_align.get("dx_px", 0.0)) > 1e-6
        or abs(image_align.get("dy_px", 0.0)) > 1e-6
        or abs(image_align.get("rotation_deg", 0.0)) > 1e-6
    ):
        text += (
            f"; image align dx={image_align.get('dx_px', 0):+.1f}px "
            f"dy={image_align.get('dy_px', 0):+.1f}px "
            f"rot={image_align.get('rotation_deg', 0):+.2f}°"
        )
    return text
