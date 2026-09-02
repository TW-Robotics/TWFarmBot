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

# 1x1 JPEG so sim / CI needs no camera library and never opens V4L2.
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
        raise CaptureError(f"missing camera node {node}")

    artifact_id = uuid.uuid4().hex
    dest = _artifact_dir(cfg) / f"{artifact_id}-{key}.jpg"
    dwell_s = _dwell_s(cfg)
    _grab_still(node, dest, dwell_s=dwell_s)
    log.info("capture band=%s artifact_id=%s path=%s", key, artifact_id, dest)
    return artifact_id


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


def _grab_still(device: Path, dest: Path, *, dwell_s: float) -> None:
    try:
        mode = device.stat().st_mode
    except OSError as err:
        raise CaptureError(f"cannot stat camera node {device}: {err}") from err
    if stat.S_ISCHR(mode):
        _grab_uvc(device, dest, dwell_s=dwell_s)
        return
    # Sim: the configured node exists but is not a V4L2 character device.
    if dwell_s:
        time.sleep(dwell_s)
    try:
        dest.write_bytes(_STUB_JPEG)
    except OSError as err:
        raise CaptureError(f"failed to write stub still {dest}: {err}") from err


def _grab_uvc(device: Path, dest: Path, *, dwell_s: float) -> None:
    # pylint: disable=import-outside-toplevel,no-member
    import cv2  # noqa: PLC0415  # lazy: tests never open a V4L2 node

    cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise CaptureError(f"cannot open UVC device {device}")
        if dwell_s:
            time.sleep(dwell_s)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise CaptureError(f"UVC device {device} returned no frame")
        if not cv2.imwrite(str(dest), frame):
            raise CaptureError(f"failed to write still {dest}")
    finally:
        cap.release()
