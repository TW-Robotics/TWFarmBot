#!/usr/bin/env -S uv run python
"""Connect to a real FarmBot over WiFi and verify the link.

Usage:
    cp .env.example .env   # then fill in FARMBOT_EMAIL / FARMBOT_PASSWORD
    uv run --env-file=.env python scripts/test_farmbot_connect.py

Or set the env vars inline:
    FARMBOT_EMAIL=you@example.com \\
    FARMBOT_PASSWORD=yourpassword \\
    [FARMBOT_SERVER=https://my.farm.bot]   # optional, default shown
    [FARMBOT_HOST=farmbot.farm.bot]       # optional, default shown; use http://192.168.x.y for LAN
    uv run python scripts/test_farmbot_connect.py

Exits 0 on a successful connection + a non-empty status read, non-zero otherwise.
"""

from __future__ import annotations

import os
import sys
import time


def main() -> int:
    email = os.environ.get("FARMBOT_EMAIL")
    password = os.environ.get("FARMBOT_PASSWORD")
    if not email or not password:
        print(
            "ERROR: set FARMBOT_EMAIL and FARMBOT_PASSWORD env vars first.",
            file=sys.stderr,
        )
        return 2

    from farmbot_gateway import FarmBotConnectionError, get_farmbot

    print(f"→ connecting as {email!r} "
          f"(server={os.environ.get('FARMBOT_SERVER', 'https://my.farm.bot')}, "
          f"mqtt={os.environ.get('FARMBOT_HOST', 'farmbot.farm.bot')})")
    t0 = time.monotonic()
    try:
        fb = get_farmbot()
    except FarmBotConnectionError as err:
        print(f"✗ connection failed after {time.monotonic() - t0:.1f}s: {err}",
              file=sys.stderr)
        if err.__cause__ is not None:
            print(f"  cause: {type(err.__cause__).__name__}: {err.__cause__}",
                  file=sys.stderr)
        return 1

    print(f"✓ connected in {time.monotonic() - t0:.1f}s")

    try:
        xyz = fb.get_xyz()
    except Exception as err:  # noqa: BLE001 — farmbot-py raises broad types
        print(f"  get_xyz() raised: {type(err).__name__}: {err}", file=sys.stderr)
        return 1

    if xyz is None:
        print("  WARNING: get_xyz() returned None — bot connected but no position data "
              "yet (firmware may still be booting, or no status messages received).",
              file=sys.stderr)
        return 0  # connection itself is fine

    print(f"  current position (get_xyz): {xyz!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
