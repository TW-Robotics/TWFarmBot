"""Scheduled zone inspect: POST /actions, then write a history snapshot."""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from spatial_service import load_world
from twfarmbot_core.config import load_yaml_config

log = logging.getLogger("twfarmbot.worker")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_data_dir() -> Path:
    path = Path(os.getenv("TWFB_UI_DATA_DIR", Path.cwd() / "data" / "ui_sessions"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def inspect_config() -> dict[str, Any]:
    block = load_yaml_config().get("worker") or {}
    interval = os.getenv("TWFB_INSPECT_INTERVAL_S")
    return {
        "interval_s": float(
            interval if interval is not None else block.get("inspect_interval_s", 21600)
        ),
        "step_mm": float(
            os.getenv("TWFB_INSPECT_STEP_MM", block.get("inspect_step_mm", 250))
        ),
        "z": float(os.getenv("TWFB_INSPECT_Z", block.get("inspect_z", 0))),
        "classes": str(
            os.getenv(
                "TWFB_INSPECT_CLASSES",
                block.get("inspect_classes", "plant, weed, soil"),
            )
        ),
        "api_url": os.getenv("TWFB_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        "timeout_s": float(
            os.getenv("TWFB_INSPECT_TIMEOUT_S", block.get("inspect_timeout_s", 1800))
        ),
    }


def write_report(
    scorecard: dict[str, Any], *, zone_id: str, error: str | None = None
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    session_id = f"{stamp}-inspect-{zone_id}-{secrets.token_hex(3)}"
    summary = (
        error
        or scorecard.get("summary_text")
        or f"Inspect {scorecard.get('zone_name') or zone_id}"
    )
    snapshot = {
        "session_id": session_id,
        "label": f"Inspect · {scorecard.get('zone_name') or zone_id}",
        "kind": "inspect",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "assistant_messages": [
            {"role": "user", "content": f"scheduled inspect {zone_id}"},
            {"role": "assistant", "content": summary},
        ],
        "inspect": {**scorecard, "error": error}
        if scorecard or error
        else {"error": error},
    }
    path = session_data_dir() / f"{session_id}.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return path


def run_once(client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Inspect every configured zone and persist a history card per zone."""
    cfg = inspect_config()
    own_client = client is None
    client = client or httpx.Client(base_url=cfg["api_url"], timeout=cfg["timeout_s"])
    reports: list[dict[str, Any]] = []
    try:
        health = client.get("/health").json()
        if health.get("farmbot") != "connected":
            log.warning("skip inspect; farmbot=%s", health.get("farmbot"))
            return reports
        for zone in load_world().zones:
            payload = {
                "kind": "inspect_zone",
                "params": {
                    "zone_id": zone.id,
                    "step_mm": cfg["step_mm"],
                    "z": cfg["z"],
                    "classes": cfg["classes"],
                },
            }
            try:
                response = client.post("/actions", json=payload)
                response.raise_for_status()
                body = response.json()
                scorecard = (body.get("action") or {}).get("params", {}).get(
                    "scorecard"
                ) or {}
                path = write_report(scorecard, zone_id=zone.id)
                reports.append({"zone_id": zone.id, "path": str(path), "ok": True})
                log.info("inspect %s -> %s", zone.id, scorecard.get("summary_text"))
            except Exception as err:  # noqa: BLE001
                log.exception("inspect failed for %s", zone.id)
                path = write_report(
                    {}, zone_id=zone.id, error=f"{type(err).__name__}: {err}"
                )
                reports.append(
                    {
                        "zone_id": zone.id,
                        "path": str(path),
                        "ok": False,
                        "error": str(err),
                    }
                )
        return reports
    finally:
        if own_client:
            client.close()
