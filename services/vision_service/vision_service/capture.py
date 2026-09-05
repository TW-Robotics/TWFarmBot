"""capture(band) → artifact_id.

rgb / nir / rededge share one UVC backend and are opened by udev symlink
(``/dev/camera-{band}``), never by ``/dev/videoN``. thermal and swir fail
closed: those buses are not pinned.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from twfarmbot_core.config import load_yaml_config

log = logging.getLogger(__name__)

CAPTURE_BANDS = frozenset({"rgb", "nir", "rededge", "thermal", "swir"})
USB_CAPTURE_BANDS = frozenset({"rgb", "nir", "rededge"})
UNPINNED_CAPTURE_BANDS = frozenset({"thermal", "swir"})

# 1x1 JPEG used in sim/CI so we avoid importing cv2 and never open V4L2.
_STUB_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb0043000c08080809080c09"
    "090c11110f1111161414141416161c1818181818181c1e1c1c1c1c1c1c1e2b2323"
    "232323232b322c2c2c2c2c322c2c2c2c2c2c2c2cffc00011080001000103012200"
    "021101031101ffc40015000101000000000000000000000000000008ffc4001401"
    "01000000000000000000000000000000ffda000c0301000210031000003f00d2cf"
    "20ffd9"
)

_VIDEO_N = re.compile(r"^video\d+$")


class CaptureError(ValueError):
    """Raised when capture cannot produce a still."""


def capture(band: str) -> str:
    """Grab one still from the payload camera for ``band``. Return artifact_id."""
    key = str(band or "").strip().lower()
    if key not in CAPTURE_BANDS:
        raise CaptureError(
            f"capture band must be one of {sorted(CAPTURE_BANDS)}, got {band!r}"
        )
    if key in UNPINNED_CAPTURE_BANDS:
        raise CaptureError(f"{key} capture is unavailable: bus not pinned")
    if key not in USB_CAPTURE_BANDS:
        raise CaptureError(f"{key} capture has no USB backend")

    cfg = _cameras_config()
    node = _device_node(cfg, key)
    _refuse_raw_video_node(node)
    if not node.exists():
        fallback = (
            _v4l_node_for_band(key) if _is_canonical_camera_node(node, key) else None
        )
        if fallback is None:
            raise CaptureError(f"missing camera node {node}")
        log.warning("camera symlink %s missing; using %s", node, fallback)
        node = fallback

    artifact_id = uuid.uuid4().hex
    dest = _artifact_dir(cfg) / f"{artifact_id}-{key}.jpg"
    dwell_s = _dwell_s(cfg)
    _grab_still(node, dest, dwell_s=dwell_s, band=key)
    log.info("capture band=%s artifact_id=%s path=%s", key, artifact_id, dest)
    return artifact_id


def list_captures(limit: int = 10) -> list[dict[str, Any]]:
    """Newest capture artifacts on disk (newest first)."""
    limit = max(1, min(int(limit), 50))
    cfg = _cameras_config()
    directory = _artifact_dir(cfg)
    files = sorted(
        directory.glob("*-*.jpg"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        stem = path.stem
        artifact_id, _, band = stem.partition("-")
        if not artifact_id or not band:
            continue
        out.append(
            {
                "id": stem,
                "artifact_id": artifact_id,
                "band": band,
                "path": str(path.resolve()),
                "created_at": str(int(path.stat().st_mtime)),
            }
        )
    return out


def capture_path(artifact_id: str, band: str) -> Path | None:
    """Resolve a capture still (JPEG) or derived artifact (PNG) if it exists."""
    key = str(band or "").strip().lower()
    aid = str(artifact_id or "").strip()
    if not aid or not key:
        return None
    directory = _artifact_dir(_cameras_config())
    for suffix in (".jpg", ".png"):
        if key not in USB_CAPTURE_BANDS and suffix == ".jpg":
            continue
        path = directory / f"{aid}-{key}{suffix}"
        if path.is_file():
            return path
    return None


def write_capture_file(
    artifact_id: str,
    band: str,
    data: bytes,
    *,
    suffix: str = ".png",
) -> Path:
    """Write a derived capture artifact (e.g. NDRE preview PNG)."""
    key = str(band or "").strip().lower()
    aid = str(artifact_id or "").strip()
    if not aid or not key:
        raise CaptureError("write_capture_file needs artifact_id and band")
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    dest = _artifact_dir(_cameras_config()) / f"{aid}-{key}{suffix}"
    try:
        dest.write_bytes(data)
    except OSError as err:
        raise CaptureError(f"failed to write {dest}: {err}") from err
    return dest


def _cameras_config() -> dict[str, Any]:
    block = load_yaml_config().get("cameras") or {}
    return block if isinstance(block, dict) else {}


def _device_node(cfg: dict[str, Any], band: str) -> Path:
    devices = cfg.get("devices") or {}
    if not isinstance(devices, dict):
        devices = {}
    raw = devices.get(band)
    if not raw:
        # Locked mapping when YAML omits a USB band: /dev/camera-{band}.
        raw = f"/dev/camera-{band}"
    path = Path(str(raw))
    if not path.is_absolute():
        raise CaptureError(f"camera device path must be absolute, got {path}")
    return path


def _is_canonical_camera_node(path: Path, band: str) -> bool:
    return path == Path(f"/dev/camera-{band}")


# Keep in sync with configs/udev/99-twfarmbot-cameras.rules (index 0 = capture).
_BAND_USB = {
    "rgb": (
        {"ID_VENDOR_ID": "0408", "ID_MODEL_ID": "a061"},
        {"ID_VENDOR_ID": "1bcf", "ID_MODEL_ID": "2085"},
    ),
    "nir": ({"ID_SERIAL_SHORT": "47520551"},),
    "rededge": ({"ID_SERIAL_SHORT": "47520552"},),
}


def _v4l_props(device: Path) -> dict[str, str]:
    import subprocess

    try:
        raw = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={device}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: dict[str, str] = {}
    for line in raw.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def _v4l_node_for_band(band: str) -> Path | None:
    """Find the capture node by USB identity when /dev/camera-* is not installed."""
    wanted = _BAND_USB.get(band)
    if not wanted:
        return None
    for candidate in sorted(Path("/dev").glob("video*")):
        if not _VIDEO_N.match(candidate.name):
            continue
        index = Path(f"/sys/class/video4linux/{candidate.name}/index")
        try:
            if index.read_text(encoding="utf-8").strip() != "0":
                continue
        except OSError:
            continue
        props = _v4l_props(candidate)
        if not props:
            continue
        for match in wanted:
            if all(props.get(key) == value for key, value in match.items()):
                return candidate
    return None


def _refuse_raw_video_node(path: Path) -> None:
    if path.parent == Path("/dev") and _VIDEO_N.match(path.name):
        raise CaptureError(
            f"refusing {path}: bind udev SYMLINK+=camera-{{band}}, never open /dev/videoN"
        )


def _artifact_dir(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("artifact_dir") or os.getenv("TWFB_CAPTURE_DIR", "data/captures")
    directory = Path(str(raw))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _dwell_s(cfg: dict[str, Any]) -> float:
    raw = cfg.get("dwell_s", 0.25)
    try:
        value = float(raw)
    except (TypeError, ValueError) as err:
        raise CaptureError(f"cameras.dwell_s must be numeric, got {raw!r}") from err
    return max(0.0, value)


def _grab_still(device: Path, dest: Path, *, dwell_s: float, band: str) -> None:
    try:
        mode = device.stat().st_mode
    except OSError as err:
        raise CaptureError(f"cannot stat camera node {device}: {err}") from err
    if stat.S_ISCHR(mode):
        _grab_uvc(device, dest, dwell_s=dwell_s, band=band)
        return
    # Sim: the configured node exists but is not a V4L2 character device.
    if dwell_s:
        time.sleep(dwell_s)
    try:
        dest.write_bytes(_STUB_JPEG)
    except OSError as err:
        raise CaptureError(f"failed to write stub still {dest}: {err}") from err


def _grab_uvc(device: Path, dest: Path, *, dwell_s: float, band: str) -> None:
    from farmduino.uvc import grab_uvc_still

    try:
        grab_uvc_still(device, dest, dwell_s=dwell_s, band=band)
    except RuntimeError as err:
        raise CaptureError(str(err)) from err
