"""Append-only trajectory log for agent runs.

One JSONL file per thread under ``data/trajectories/`` (override with
``TWFB_TRAJECTORY_DIR``, disable with ``TWFB_TRAJECTORY=0``). Every record
is ``{seq, ts, thread_id, type, ...}`` written as the run unfolds, so a
crash or restart still leaves a complete, replayable ledger: run_start,
thinking/delta, tool_start/tool_call (args, result, duration, pose),
position, goal, approval, followup, messages, run_end.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")
_MAX_STR = 20_000


def trajectory_dir(explicit: str | Path | None = None) -> Path | None:
    """Resolve the log directory, or ``None`` when logging is off.

    Off under pytest unless a directory is given explicitly (constructor
    argument or env), so unit tests never write into the repo's ``data/``.
    """
    if os.getenv("TWFB_TRAJECTORY") == "0":
        return None
    env = os.getenv("TWFB_TRAJECTORY_DIR")
    chosen = explicit or env
    if chosen is None and "pytest" in sys.modules:
        return None
    return Path(chosen) if chosen else Path.cwd() / "data" / "trajectories"


def _sanitize_id(thread_id: str) -> str:
    return _SAFE_ID.sub("_", str(thread_id))[:128] or "unknown"


def _safe(value: Any) -> Any:
    """JSON-friendly copy; image data URIs and huge strings are elided."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if value.startswith("data:image"):
            return f"<image omitted: {len(value)} chars>"
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "…<truncated>"
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


class TrajectoryLog:
    """Append-only writer for one thread's trajectory file."""

    def __init__(
        self, thread_id: str, directory: str | Path, *, model_name: str = ""
    ) -> None:
        self.thread_id = str(thread_id)
        self.model_name = model_name
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{_sanitize_id(self.thread_id)}.jsonl"
        self._lock = threading.Lock()
        self._seq = 0
        if self.path.exists():  # resumed run: continue the sequence
            try:
                self._seq = sum(1 for _ in self.path.open(encoding="utf-8"))
            except OSError:
                self._seq = 0

    def record(self, record_type: str, **fields: Any) -> None:
        entry: dict[str, Any] = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "thread_id": self.thread_id,
            "type": record_type,
        }
        if record_type == "run_start" and self.model_name:
            entry["model"] = self.model_name
        entry.update(_safe(fields))
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, default=str) + "\n")
                self._seq += 1
        except OSError as err:  # logging must never break a run
            _log.warning("trajectory write failed (%s): %s", self.path, err)


def _base_dir(directory: str | Path | None) -> Path:
    chosen = directory or os.getenv("TWFB_TRAJECTORY_DIR")
    return Path(chosen) if chosen else Path.cwd() / "data" / "trajectories"


def read_trajectory(
    thread_id: str, directory: str | Path | None = None
) -> list[dict[str, Any]]:
    """Return all records for ``thread_id`` (empty when none/invalid id)."""
    if _sanitize_id(thread_id) != str(thread_id):
        return []
    base = _base_dir(directory)
    path = Path(base) / f"{thread_id}.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def list_trajectories(
    directory: str | Path | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Summaries of stored trajectories, newest first."""
    base = _base_dir(directory)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(
        base.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        records = read_trajectory(path.stem, base)
        if not records:
            continue
        start = next((r for r in records if r.get("type") == "run_start"), {})
        end = next((r for r in reversed(records) if r.get("type") == "run_end"), {})
        out.append(
            {
                "thread_id": path.stem,
                "started_at": start.get("ts") or records[0].get("ts", ""),
                "prompt": str(start.get("prompt") or "")[:200],
                "model": start.get("model") or "",
                "stop_reason": end.get("stop_reason"),
                "records": len(records),
                "tool_calls": sum(1 for r in records if r.get("type") == "tool_call"),
                "duration_s": (end.get("metrics") or {}).get("total_latency_s"),
            }
        )
        if len(out) >= limit:
            break
    return out
