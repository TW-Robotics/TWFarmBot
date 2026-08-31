#!/usr/bin/env -S uv run python
"""Ping the local FarmBot HTTP API and print position.

Usage:
    FARMBOT_LOCAL_URL=http://192.168.1.50:3001 uv run python scripts/test_farmbot_connect.py

Exits 0 on a successful ping + position read, non-zero otherwise.
"""

from __future__ import annotations

import sys
import time

from farmbot_client import FarmBotConnectionError, base_url, connect


def main() -> int:
    url = base_url()
    print(f"→ connecting to local FarmBot at {url}")
    t0 = time.monotonic()
    try:
        fb = connect()
    except FarmBotConnectionError as err:
        print(
            f"✗ connection failed after {time.monotonic() - t0:.1f}s: {err}",
            file=sys.stderr,
        )
        if err.__cause__ is not None:
            print(
                f"  cause: {type(err.__cause__).__name__}: {err.__cause__}",
                file=sys.stderr,
            )
        return 1

    print(f"✓ connected in {time.monotonic() - t0:.1f}s")

    try:
        xyz = fb.get_xyz()
    except Exception as err:  # noqa: BLE001
        print(f"  get_xyz() raised: {type(err).__name__}: {err}", file=sys.stderr)
        return 1

    print(f"  current position: {xyz!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
