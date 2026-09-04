"""Shared UVC still capture for payload USB cameras."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def _v4l2_prepare(device: Path) -> None:
    """Best-effort exposure tuning for over-bright UVC webcams."""
    if os.getenv("FARMBOT_CAMERA_V4L2_TUNING", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    exposure = os.getenv("FARMBOT_CAMERA_EXPOSURE", "300")
    controls = [
        "auto_exposure=1",
        f"exposure_time_absolute={exposure}",
        "backlight_compensation=0",
    ]
    cmd = ["v4l2-ctl", "-d", str(device), "--set-ctrl", ",".join(controls)]
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return


def grab_uvc_still(device: Path, dest: Path, *, dwell_s: float = 0.5) -> None:
    """Grab one JPEG from a UVC device using MJPEG when available."""
    import cv2  # noqa: PLC0415

    _v4l2_prepare(device)
    cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open UVC device {device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        warmup = int(os.getenv("FARMBOT_CAMERA_WARMUP_FRAMES", "8"))
        if dwell_s:
            time.sleep(dwell_s)
        frame = None
        for _ in range(max(1, warmup)):
            ok, frame = cap.read()
            if not ok:
                frame = None
        if frame is None:
            raise RuntimeError(f"UVC device {device} returned no frame")
        if not cv2.imwrite(str(dest), frame):
            raise RuntimeError(f"failed to write still {dest}")
    finally:
        cap.release()
