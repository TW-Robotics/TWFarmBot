"""Tests for NIR / red-edge NDRE analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_service.spectral_analysis import (
    analyze_spectral_pair,
    apply_image_align,
    compute_ndre,
    interpret_ndre,
    read_grayscale,
)


@pytest.fixture
def gradient_pair(tmp_path: Path) -> tuple[Path, Path]:
    height, width = 120, 160
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    nir = np.tile(y, (1, width))
    rededge = (nir.astype(np.float32) * 0.6).astype(np.uint8)
    nir_path = tmp_path / "nir.jpg"
    rededge_path = tmp_path / "rededge.jpg"
    cv2.imwrite(str(nir_path), nir)
    cv2.imwrite(str(rededge_path), rededge)
    return nir_path, rededge_path


def test_interpret_ndre_bare_signal() -> None:
    out = interpret_ndre(
        {"mean": 0.01, "vegetation_fraction": 0.1, "stress_fraction": 0.8}
    )
    assert out["label"] == "bare_or_non_canopy"
    assert out["action_hint"] == "reposition"


def test_interpret_ndre_healthy() -> None:
    out = interpret_ndre(
        {"mean": 0.35, "vegetation_fraction": 0.7, "stress_fraction": 0.1}
    )
    assert out["label"] == "healthy_canopy"
    assert out["action_hint"] == "ok"


def test_compute_ndre_on_gradient(gradient_pair: tuple[Path, Path]) -> None:
    nir = read_grayscale(gradient_pair[0])
    rededge = read_grayscale(gradient_pair[1])
    ndre, _warped, align = compute_ndre(nir, rededge)
    assert float(np.mean(ndre)) > 0.1
    assert float(np.max(ndre)) <= 1.0
    assert align["dx_px"] == 0.0


def test_apply_image_align_translation() -> None:
    image = np.zeros((100, 120), dtype=np.float32)
    image[40:60, 50:70] = 200
    warped = apply_image_align(image, dx_px=10, dy_px=-5, rotation_deg=0)
    assert warped[35:55, 60:80].mean() > 100


def test_analyze_spectral_pair_returns_metrics(
    gradient_pair: tuple[Path, Path],
) -> None:
    result = analyze_spectral_pair(
        *gradient_pair,
        image_align={"dx_px": 0, "dy_px": 0, "rotation_deg": 0},
    )
    assert "ndre" in result
    assert "summary" in result
    assert result["ndre"]["mean"] > 0
    assert result["ndre_preview"].startswith("data:image/png;base64,")
    assert result["align_preview"].startswith("data:image/png;base64,")
    assert result["image_align"]["dx_px"] == 0.0
    assert "interpretation" in result
    assert result["interpretation"]["action_hint"] in {
        "ok",
        "monitor",
        "consider_water",
        "recheck",
        "reposition",
    }


def test_write_capture_file_roundtrip(tmp_path: Path, monkeypatch) -> None:
    import importlib

    # Package exports shadow the submodule name; load the real module file.
    mod = importlib.import_module("vision_service.capture")
    # If the package attribute leaked into sys.modules, reload from path.
    if not hasattr(mod, "write_capture_file"):
        from pathlib import Path as _P
        import vision_service as vs

        path = _P(vs.__file__).parent / "capture.py"
        spec = importlib.util.spec_from_file_location(
            "vision_service._capture_mod", path
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "_artifact_dir", lambda _cfg: tmp_path)
    out = mod.write_capture_file("abc", "ndre", b"png-bytes")
    assert out == tmp_path / "abc-ndre.png"
    assert out.read_bytes() == b"png-bytes"
    assert mod.capture_path("abc", "ndre") == out
