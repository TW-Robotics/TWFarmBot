"""Tests for spectral camera calibration helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_service.spectral_calib import (
    build_calibration,
    build_dot_calibration,
    detect_aruco_marker,
    gantry_for_band,
    gsd_mm_per_px,
    observe_band,
    save_calibration,
)


@pytest.fixture
def aruco_image(tmp_path: Path) -> Path:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 400)
    canvas = np.full((720, 1280), 255, dtype=np.uint8)
    y0 = (canvas.shape[0] - marker.shape[0]) // 2
    x0 = (canvas.shape[1] - marker.shape[1]) // 2
    canvas[y0 : y0 + marker.shape[0], x0 : x0 + marker.shape[1]] = marker
    path = tmp_path / "aruco.jpg"
    cv2.imwrite(str(path), canvas)
    return path


def test_detect_aruco_marker_centroid(aruco_image: Path) -> None:
    found = detect_aruco_marker(aruco_image)
    assert found["marker_id"] == 0
    assert found["marker_span_px"] > 100
    assert abs(found["centroid_px"]["x"] - 640) < 5
    assert abs(found["centroid_px"]["y"] - 360) < 5


def test_build_calibration_computes_band_separation(aruco_image: Path) -> None:
    nir = observe_band("nir", aruco_image, {"x": 100.0, "y": 200.0, "z": 0.0})
    rededge = observe_band(
        "rededge", aruco_image, {"x": 100.0, "y": 282.0, "z": 0.0}
    )
    doc = build_calibration([nir, rededge])
    assert doc["computed"]["band_separation_mm"]["y"] == pytest.approx(82.0)
    assert doc["shared"]["gsd_mm_per_px"] == pytest.approx(
        gsd_mm_per_px(100.0, nir["marker_span_px"]), rel=1e-3
    )


def test_gantry_for_band_applies_offset(aruco_image: Path) -> None:
    nir = observe_band("nir", aruco_image, {"x": 0.0, "y": 0.0, "z": 0.0})
    rededge = observe_band("rededge", aruco_image, {"x": 0.0, "y": 80.0, "z": 0.0})
    doc = build_calibration([nir, rededge])
    pose = gantry_for_band({"x": 150.0, "y": 300.0, "z": 50.0}, "rededge", doc)
    assert pose["x"] == pytest.approx(150.0)
    assert pose["y"] == pytest.approx(380.0)
    assert pose["z"] == pytest.approx(50.0)


def test_build_dot_calibration() -> None:
    doc = build_dot_calibration(
        {"x": 100.0, "y": 0.0, "z": 0.0},
        image_align={"dx_px": 5, "dy_px": -2, "rotation_deg": 1.5},
    )
    pose = gantry_for_band({"x": 200.0, "y": 50.0, "z": 10.0}, "rededge", doc)
    assert pose == {"x": 300.0, "y": 50.0, "z": 10.0}
    assert doc["image_align"]["dx_px"] == 5.0
    assert doc["image_align"]["rotation_deg"] == 1.5


def test_save_calibration_roundtrip(tmp_path: Path, aruco_image: Path) -> None:
    nir = observe_band("nir", aruco_image, {"x": 1.0, "y": 2.0, "z": 0.0})
    rededge = observe_band("rededge", aruco_image, {"x": 1.0, "y": 90.0, "z": 0.0})
    doc = build_calibration([nir, rededge])
    path = save_calibration(doc, tmp_path / "spectral_calibration.yaml")
    assert path.is_file()
    assert "band_separation_mm" in path.read_text(encoding="utf-8")
