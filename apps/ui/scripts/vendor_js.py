"""Vendor the UI's external JS dependencies into ``static/vendor/``.

The Material 3 frontend currently loads three third-party libraries from
``esm.run`` (Material Web, Chart.js, marked, DOMPurify) and Google Fonts via
``fonts.googleapis.com``. Both fail completely when the on-farm controller
loses internet connectivity — and the ESTOP button is part of the same
SPA, so every load fails along with them.

This script fetches the exact-pinned files once and writes them into
``apps/ui/src/twfarmbot_ui/static/vendor/`` so the FastAPI server can serve
them as ordinary static assets. Subsequent runs verify the SHA256 of each
file against the manifest below; a hash mismatch triggers a re-download.
Commit the resulting ``vendor/`` tree to make the UI fully offline-capable.

Usage::

    # Initial setup (also re-runs after a manifest change):
    python apps/ui/scripts/vendor_js.py

    # Force re-download even if the local file already matches the hash:
    python apps/ui/scripts/vendor_js.py --force

The script intentionally uses only the Python stdlib (``urllib`` +
``hashlib``) so it can run in any environment — including a fresh
``uv`` checkout, a CI runner, or a hardened image without network
access to PyPI.

At runtime the UI is unaffected: the FastAPI server serves the vendored
files from ``static/vendor/`` just like any other static asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_USER_AGENT = "twfarmbot-vendor-js/1.0"
_TIMEOUT = 60.0

# Resolved relative to the repo root (apps/ui/...).  Keep the script
# importable from both the repo root and apps/ui/.
HERE = Path(__file__).resolve().parent
DEFAULT_STATIC = HERE.parent / "src" / "twfarmbot_ui" / "static"

log = logging.getLogger("vendor_js")


@dataclass(frozen=True)
class Asset:
    """One file to vendor: a remote URL, the local path under ``vendor/``,
    and the expected SHA256 (hex) of the bytes we want on disk."""

    url: str
    relpath: str
    sha256: str


# Pinned versions + SHA256 hashes.  Update both fields together; a mismatch
# is treated as tampering/bit-rot and triggers a re-download.
MANIFEST: tuple[Asset, ...] = (
    Asset(
        url="https://esm.run/@material/web@2.4.0/all.js",
        relpath="@material/web/all.js",
        sha256="07023ec48b3b495f34b485a625cc470ed07692594ccfc74d6b800f225d1a80fb",
    ),
    Asset(
        url="https://esm.run/@material/web@2.4.0/typography/md-typescale-styles.js",
        relpath="@material/web/typography/md-typescale-styles.js",
        sha256="07546a71476de34433a067774d3192900d28e88169fb7aec8831f95b0c97bc07",
    ),
    Asset(
        url="https://esm.run/chart.js@4.4.9/auto",
        relpath="chart.js/auto.js",
        sha256="e38916283b32321696a17c0acb4123f3889b4331e1859d471b195eecd4a9e497",
    ),
    Asset(
        url="https://esm.run/marked@15.0.12",
        relpath="marked.js",
        sha256="5b2f8940c0c4fd3f568aa4e08e169cbbcef8496a094eb02b579d342e5e9377e4",
    ),
    Asset(
        url="https://esm.run/dompurify@3.2.6",
        relpath="dompurify.js",
        sha256="c7cf8c441c3a0be7597d0b15c45df444e624b23a72101b68b5a96dfad8c5f2b5",
    ),
)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _needs_fetch(target: Path, asset: Asset) -> bool:
    if not target.exists():
        return True
    if _hash(target.read_bytes()) != asset.sha256:
        log.warning(
            "hash mismatch for %s — file is stale or corrupted", asset.relpath
        )
        return True
    return False


def _download(asset: Asset, target: Path) -> None:
    log.info("GET %s", asset.url)
    req = urllib.request.Request(asset.url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = resp.read()
    except urllib.error.HTTPError as err:
        raise SystemExit(f"HTTP {err.code} fetching {asset.url}") from err
    except urllib.error.URLError as err:
        raise SystemExit(f"failed to fetch {asset.url}: {err.reason}") from err
    digest = _hash(data)
    if digest != asset.sha256:
        raise SystemExit(
            f"SHA256 mismatch for {asset.url}\n"
            f"  expected: {asset.sha256}\n"
            f"  got:      {digest}\n"
            "Update MANIFEST deliberately, or pin a different upstream."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    log.info("wrote %s (%d bytes)", target, len(data))


def vendor(static_dir: Path, *, force: bool = False) -> list[Path]:
    vendor_dir = static_dir / "vendor"
    written: list[Path] = []
    for asset in MANIFEST:
        target = vendor_dir / asset.relpath
        if not force and not _needs_fetch(target, asset):
            log.info("up to date: %s", asset.relpath)
            continue
        _download(asset, target)
        written.append(target)
    manifest_path = vendor_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "url": a.url,
                    "path": a.relpath,
                    "sha256": a.sha256,
                    "size": (vendor_dir / a.relpath).stat().st_size,
                }
                for a in MANIFEST
            ],
            indent=2,
        )
        + "\n"
    )
    return written


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--static-dir",
        type=Path,
        default=DEFAULT_STATIC,
        help="path to the UI's static/ directory (default: %(default)s)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download every asset even if the local file is current",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable debug logging",
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.static_dir.is_dir():
        raise SystemExit(f"static dir not found: {args.static_dir}")
    written = vendor(args.static_dir, force=args.force)
    target = args.static_dir / "vendor"
    if written:
        log.info("updated %d file(s) under %s", len(written), target)
    else:
        log.info("vendor tree is up to date at %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
