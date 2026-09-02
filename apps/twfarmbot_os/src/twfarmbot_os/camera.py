"""Capture stills with rpicam-still / libcamera-still / fswebcam, or a stub JPEG."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

# 1x1 JPEG so simulation needs no camera library.
_STUB_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb0043000c08080809080c09"
    "090c11110f1111161414141416161c1818181818181c1e1c1c1c1c1c1c1e2b2323"
    "232323232b322c2c2c2c2c322c2c2c2c2c2c2c2cffc00011080001000103012200"
    "021101031101ffc40015000101000000000000000000000000000008ffc4001401"
    "01000000000000000000000000000000ffda000c0301000210031000003f00d2cf"
    "20ffd9"
)


class Camera:
    def __init__(self, directory: str | None = None) -> None:
        raw = directory or os.getenv("FARMBOT_PHOTO_DIR") or "data/farmbot_photos"
        self.directory = Path(raw)
        self.directory.mkdir(parents=True, exist_ok=True)

    def capture(self) -> dict[str, str]:
        photo_id = uuid.uuid4().hex
        path = self.directory / f"{photo_id}.jpg"
        cmd = os.getenv("FARMBOT_CAMERA_CMD")
        if not cmd:
            for binary in ("rpicam-still", "libcamera-still", "fswebcam"):
                found = shutil.which(binary)
                if found:
                    cmd = found
                    break
        if cmd and shutil.which(cmd.split()[0]):
            args = cmd.split()
            if os.path.basename(args[0]) == "fswebcam":
                args.extend(["-r", "1280x720", "--no-banner", str(path)])
            else:
                args.extend(["-n", "-o", str(path), "-t", "1"])
            subprocess.run(args, check=True, timeout=30)
        else:
            path.write_bytes(_STUB_JPEG)
        return {
            "id": photo_id,
            "path": str(path.resolve()),
            "created_at": str(int(time.time())),
        }

    def list_photos(self, limit: int = 10) -> list[dict[str, str]]:
        files = sorted(
            self.directory.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        out: list[dict[str, str]] = []
        for path in files[:limit]:
            out.append(
                {
                    "id": path.stem,
                    "path": str(path.resolve()),
                    "created_at": str(int(path.stat().st_mtime)),
                }
            )
        return out

    def path_for(self, photo_id: str) -> Path | None:
        path = self.directory / f"{photo_id}.jpg"
        return path if path.is_file() else None
