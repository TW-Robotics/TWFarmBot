"""What was planted where (rectangular areas), kept as plain JSON next to the other farm data."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

PLANTINGS_FILE = "plantings.json"


def _path() -> Path:
    return Path(os.getenv("TWFB_DATA_DIR", Path.cwd() / "data")) / PLANTINGS_FILE


def normalize_planting(item: Mapping[str, Any]) -> dict[str, Any]:
    crop = str(item.get("crop") or "").strip()
    if not crop:
        raise ValueError("planting needs a crop")
    out: dict[str, Any] = {
        "id": str(item.get("id") or "").strip(),
        "crop": crop,
        "x1": min(float(item["x1"]), float(item["x2"])),
        "y1": min(float(item["y1"]), float(item["y2"])),
        "x2": max(float(item["x1"]), float(item["x2"])),
        "y2": max(float(item["y1"]), float(item["y2"])),
        "planted": date.fromisoformat(str(item["planted"])).isoformat(),
        "note": str(item.get("note") or "").strip(),
    }
    if not out["id"]:
        raise ValueError("planting needs an id")
    return out


def normalize_plantings(raw: Sequence[Any] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw or []:
        if not isinstance(row, Mapping):
            raise ValueError("each planting must be a mapping")
        item = normalize_planting(row)
        if item["id"] in seen:
            raise ValueError(f"duplicate planting id {item['id']!r}")
        seen.add(item["id"])
        items.append(item)
    return items


def list_plantings() -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    return normalize_plantings(json.loads(path.read_text(encoding="utf-8")))


def save_plantings(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_plantings(items)
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    tmp.replace(path)
    return normalized
