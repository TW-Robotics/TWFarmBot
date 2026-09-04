"""Fast NIR / red-edge ground calibration for vegetation-index maps.

Both payload DMK cameras share the same sensor model, so one ArUco session
per band is enough: center the marker in each camera, record gantry pose,
and derive band offset + ground scale (mm/px).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

SPECTRAL_BANDS = frozenset({"nir", "rededge"})
DEFAULT_MARKER_SIZE_MM = 100.0
DEFAULT_MARKER_ID = 0
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "spectral_calibration.yaml"
)


class SpectralCalibrationError(ValueError):
    """Raised when calibration inputs are invalid."""


def _aruco_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary)


def detect_aruco_marker(
    image_path: str | Path,
    *,
    marker_id: int = DEFAULT_MARKER_ID,
) -> dict[str, Any]:
    """Detect one ArUco marker and return centroid + scale hints in pixels."""
    path = Path(image_path)
    if not path.is_file():
        raise SpectralCalibrationError(f"image not found: {path}")
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SpectralCalibrationError(f"cannot read image: {path}")

    corners, ids, _rejected = _aruco_detector().detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise SpectralCalibrationError(
            f"no ArUco marker found in {path}; use DICT_4X4_50 id={marker_id}"
        )

    matches = [
        corners[index]
        for index, value in enumerate(ids.flatten().tolist())
        if int(value) == marker_id
    ]
    if not matches:
        found = sorted({int(value) for value in ids.flatten().tolist()})
        raise SpectralCalibrationError(
            f"marker id {marker_id} not found in {path}; saw ids {found}"
        )
    if len(matches) > 1:
        log.warning("multiple markers with id %s; using the first", marker_id)

    points = matches[0].reshape(-1, 2)
    centroid = points.mean(axis=0)
    width_px = float(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    height_px = float(
        max(
            np.linalg.norm(points[1] - points[2]),
            np.linalg.norm(points[3] - points[0]),
        )
    )
    marker_span_px = (width_px + height_px) / 2.0
    if marker_span_px <= 0:
        raise SpectralCalibrationError(f"invalid marker size in pixels for {path}")

    h, w = gray.shape[:2]
    return {
        "path": str(path.resolve()),
        "marker_id": marker_id,
        "centroid_px": {"x": float(centroid[0]), "y": float(centroid[1])},
        "marker_span_px": marker_span_px,
        "image_size_px": {"width": int(w), "height": int(h)},
    }


def gsd_mm_per_px(marker_size_mm: float, marker_span_px: float) -> float:
    """Ground sample distance from a known marker width in the image."""
    if marker_size_mm <= 0 or marker_span_px <= 0:
        raise SpectralCalibrationError("marker size and span must be positive")
    return marker_size_mm / marker_span_px


def observe_band(
    band: str,
    image_path: str | Path,
    gantry_mm: dict[str, float],
    *,
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
    marker_id: int = DEFAULT_MARKER_ID,
) -> dict[str, Any]:
    """Build one band observation from a capture and gantry pose."""
    key = str(band).strip().lower()
    if key not in SPECTRAL_BANDS:
        raise SpectralCalibrationError(
            f"band must be one of {sorted(SPECTRAL_BANDS)}, got {band!r}"
        )
    detection = detect_aruco_marker(image_path, marker_id=marker_id)
    span_px = float(detection["marker_span_px"])
    scale = gsd_mm_per_px(marker_size_mm, span_px)
    return {
        "band": key,
        "image_path": detection["path"],
        "gantry_mm": {
            "x": float(gantry_mm["x"]),
            "y": float(gantry_mm["y"]),
            "z": float(gantry_mm.get("z", 0.0)),
        },
        "centroid_px": detection["centroid_px"],
        "image_size_px": detection["image_size_px"],
        "marker_size_mm": marker_size_mm,
        "marker_span_px": span_px,
        "gsd_mm_per_px": round(scale, 4),
    }


def build_dot_calibration(
    separation_mm: dict[str, float],
    *,
    reference_band: str = "nir",
    image_align: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a dot-centering calibration (gantry + optional image-space align)."""
    ref_key = str(reference_band).strip().lower()
    if ref_key not in SPECTRAL_BANDS:
        raise SpectralCalibrationError(
            f"reference band must be one of {sorted(SPECTRAL_BANDS)}"
        )
    other = "rededge" if ref_key == "nir" else "nir"
    sep = {
        "x": float(separation_mm.get("x", 0.0)),
        "y": float(separation_mm.get("y", 0.0)),
        "z": float(separation_mm.get("z", 0.0)),
    }
    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    bands = {
        ref_key: {"gantry_offset_mm": dict(zero)},
        other: {"gantry_offset_mm": dict(sep)},
    }
    align = normalize_image_align(image_align)
    return {
        "version": 1,
        "method": "dot_centering",
        "reference_band": ref_key,
        "bands": bands,
        "image_align": align,
        "computed": {
            "band_separation_mm": sep,
            "interpretation": (
                "Gantry delta (rededge minus nir) when both cameras are centered "
                "on the same bed point. image_align warps red-edge in pixel space "
                "onto NIR before NDRE."
            ),
        },
    }


def normalize_image_align(raw: dict[str, Any] | None = None) -> dict[str, float]:
    """Normalize image-space red-edge → NIR transform params."""
    data = raw if isinstance(raw, dict) else {}
    return {
        "dx_px": float(data.get("dx_px", 0.0)),
        "dy_px": float(data.get("dy_px", 0.0)),
        "rotation_deg": float(data.get("rotation_deg", 0.0)),
    }


def load_image_align(path: str | Path | None = None) -> dict[str, float]:
    """Load image_align from calibration yaml (zeros if missing)."""
    try:
        doc = load_calibration(path)
    except SpectralCalibrationError:
        return normalize_image_align(None)
    return normalize_image_align(doc.get("image_align"))


def build_calibration(
    observations: list[dict[str, Any]],
    *,
    reference_band: str = "nir",
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
) -> dict[str, Any]:
    """Merge per-band observations into a spectral calibration document."""
    if len(observations) < 2:
        raise SpectralCalibrationError("need observations for nir and rededge")

    by_band = {str(item["band"]): item for item in observations}
    missing = sorted(SPECTRAL_BANDS - set(by_band))
    if missing:
        raise SpectralCalibrationError(f"missing band observations: {missing}")

    ref_key = str(reference_band).strip().lower()
    if ref_key not in by_band:
        raise SpectralCalibrationError(f"reference band {reference_band!r} missing")

    ref = by_band[ref_key]
    ref_gantry = ref["gantry_mm"]
    gsd_values = [float(item["gsd_mm_per_px"]) for item in by_band.values()]
    gsd_mean = round(sum(gsd_values) / len(gsd_values), 4)

    bands: dict[str, Any] = {}
    for key, item in by_band.items():
        gantry = item["gantry_mm"]
        bands[key] = {
            "gantry_offset_mm": {
                "x": round(gantry["x"] - ref_gantry["x"], 3),
                "y": round(gantry["y"] - ref_gantry["y"], 3),
                "z": round(gantry.get("z", 0.0) - ref_gantry.get("z", 0.0), 3),
            },
            "centroid_px": item["centroid_px"],
            "image_size_px": item["image_size_px"],
            "calibration_capture": item["image_path"],
        }

    separation = bands["rededge"]["gantry_offset_mm"]
    return {
        "version": 1,
        "method": "aruco_bed_marker",
        "reference_band": ref_key,
        "marker_size_mm": marker_size_mm,
        "shared": {
            "gsd_mm_per_px": gsd_mean,
            "image_center_px": {
                "x": round(ref["image_size_px"]["width"] / 2.0, 1),
                "y": round(ref["image_size_px"]["height"] / 2.0, 1),
            },
        },
        "bands": bands,
        "computed": {
            "band_separation_mm": separation,
            "interpretation": (
                "Move gantry by band_separation_mm (relative to nir) so red-edge "
                "views the same bed point as nir."
            ),
        },
    }


def save_calibration(
    doc: dict[str, Any], path: str | Path = DEFAULT_CONFIG_PATH
) -> Path:
    """Write calibration YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out


def load_calibration(path: str | Path | None = None) -> dict[str, Any]:
    """Load calibration YAML if present."""
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        raise SpectralCalibrationError(f"calibration file not found: {cfg_path}")
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SpectralCalibrationError(f"invalid calibration yaml: {cfg_path}")
    return loaded


def gantry_for_band(
    target_mm: dict[str, float],
    band: str,
    calibration: dict[str, Any],
) -> dict[str, float]:
    """Gantry pose that places ``band`` over a bed target point."""
    key = str(band).strip().lower()
    bands = calibration.get("bands") or {}
    if key not in bands:
        raise SpectralCalibrationError(f"band {band!r} missing from calibration")
    ref_key = str(calibration.get("reference_band") or "nir")
    ref_offset = (bands.get(ref_key) or {}).get("gantry_offset_mm") or {}
    band_offset = bands[key].get("gantry_offset_mm") or {}
    # target is the gantry pose for the reference band; other bands add their offset delta.
    return {
        "x": float(target_mm["x"])
        + float(band_offset.get("x", 0))
        - float(ref_offset.get("x", 0)),
        "y": float(target_mm["y"])
        + float(band_offset.get("y", 0))
        - float(ref_offset.get("y", 0)),
        "z": float(target_mm.get("z", 0.0))
        + float(band_offset.get("z", 0))
        - float(ref_offset.get("z", 0)),
    }
