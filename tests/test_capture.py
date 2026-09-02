"""Sim-only tests for capture(band). No physical camera, no /dev/video0."""

from __future__ import annotations

from pathlib import Path

import pytest

from safety_service import UnsafeActionError, validate
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action
from vision_service import CaptureError, capture
from watering_service.backends import farmbot


def _sim_config(tmp_path: Path, *, devices: dict[str, str] | None = None) -> str:
    dev = tmp_path / "dev"
    dev.mkdir(exist_ok=True)
    mapping = devices if devices is not None else {}
    if devices is None:
        for band in ("rgb", "nir", "rededge"):
            node = dev / f"camera-{band}"
            node.write_bytes(b"")
            mapping[band] = str(node)
    lines = [
        "cameras:",
        "  dwell_s: 0",
        f"  artifact_dir: {tmp_path / 'artifacts'}",
        "  devices:",
    ]
    for band, path in mapping.items():
        lines.append(f"    {band}: {path}")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("\n".join(lines) + "\n")
    return str(cfg)


@pytest.fixture
def sim_cameras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TWFB_CONFIG", _sim_config(tmp_path))
    return tmp_path / "artifacts"


def test_missing_band_rejected(sim_cameras: Path) -> None:
    del sim_cameras
    with pytest.raises(UnsafeActionError, match="needs band"):
        validate(Action(kind="capture", params={}))


def test_unknown_band_rejected(sim_cameras: Path) -> None:
    del sim_cameras
    with pytest.raises(UnsafeActionError, match="must be one of"):
        validate(Action(kind="capture", params={"band": "ultraviolet"}))
    with pytest.raises(CaptureError, match="must be one of"):
        capture("ultraviolet")


def test_thermal_and_swir_rejected(sim_cameras: Path) -> None:
    del sim_cameras
    for band in ("thermal", "swir"):
        with pytest.raises(UnsafeActionError, match="bus not pinned"):
            validate(Action(kind="capture", params={"band": band}))
        with pytest.raises(CaptureError, match="bus not pinned"):
            capture(band)


def test_usb_bands_return_distinct_artifacts(sim_cameras: Path) -> None:
    ids = [capture(band) for band in ("rgb", "nir", "rededge")]
    assert len(set(ids)) == 3
    files = {p.name for p in sim_cameras.glob("*.jpg")}
    for artifact_id, band in zip(ids, ("rgb", "nir", "rededge")):
        assert f"{artifact_id}-{band}.jpg" in files


def test_missing_node_fails_without_video0_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "dev" / "camera-rgb"
    monkeypatch.setenv(
        "TWFB_CONFIG",
        _sim_config(tmp_path, devices={"rgb": str(missing)}),
    )
    opened: list[object] = []

    def _fake_capture(source: object, *args: object, **kwargs: object) -> None:
        opened.append(source)
        raise AssertionError("VideoCapture must not run in sim tests")

    monkeypatch.setattr("cv2.VideoCapture", _fake_capture, raising=False)
    with pytest.raises(CaptureError, match="missing camera node"):
        capture("rgb")
    assert opened == []


def test_refuses_raw_video_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "TWFB_CONFIG",
        _sim_config(tmp_path, devices={"rgb": "/dev/video0"}),
    )
    with pytest.raises(CaptureError, match="never open /dev/videoN"):
        capture("rgb")


def test_handler_capture_returns_artifact_id(
    sim_cameras: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del sim_cameras
    from twfarmbot_api_server.handlers import camera as h

    class _StubBackend:
        def take_photo(self) -> None:
            raise AssertionError("capture must not wrap take_photo")

    monkeypatch.setattr(farmbot, "backend", _StubBackend())
    out = h.handle_capture(Action(kind="capture", params={"band": "rgb"}))
    assert out.params["band"] == "rgb"
    assert out.params["artifact_id"]
    nir = h.handle_capture(Action(kind="capture", params={"band": "nir"}))
    assert nir.params["artifact_id"] != out.params["artifact_id"]


def test_take_photo_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from twfarmbot_api_server.handlers import camera as h

    called: list[str] = []

    class _StubBackend:
        def take_photo(self) -> None:
            called.append("take_photo")

        def wait_for_new_photo(self) -> bool:
            called.append("wait")
            return True

    def _no_capture(_band: str) -> str:
        raise AssertionError("take_photo must not call capture")

    monkeypatch.setattr(farmbot, "backend", _StubBackend())
    monkeypatch.setattr(
        "twfarmbot_api_server.handlers.camera.capture",
        _no_capture,
    )
    h.handle_take_photo(Action(kind="take_photo", params={}))
    assert called == ["take_photo", "wait"]


def test_dispatch_runs_safety_then_capture(sim_cameras: Path) -> None:
    del sim_cameras
    from twfarmbot_api_server.handlers.camera import handle_capture

    registry = ActionRegistry()
    registry.register("capture", handle_capture)
    with pytest.raises(UnsafeActionError, match="needs band"):
        registry.dispatch(Action(kind="capture", params={}))
    out = registry.dispatch(Action(kind="capture", params={"band": "rededge"}))
    assert out.params["artifact_id"]
    assert out.params["band"] == "rededge"


def test_capture_tool_policy_is_read() -> None:
    from planning_service.harness.tool_policy import ToolCategory
    from planning_service.harness.tool_registry import ToolRegistry

    registry = ActionRegistry()
    registry.register("capture", lambda action: action)
    policy = ToolRegistry(registry).by_name()["capture"].policy
    assert policy.category == ToolCategory.READ
    assert policy.requires_approval is False
