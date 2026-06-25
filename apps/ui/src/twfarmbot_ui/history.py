"""Persistence for Streamlit UI session state.

Chat history, plan previews, and executed plans are saved as JSON files so
they survive page reloads. Storage is local and intended for a single-user
research UI; concurrent writes to the same session file are last-write-wins.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def session_data_dir() -> Path:
    """Return the directory used to store session JSON files."""
    path = Path(
        os.getenv("TWFB_UI_DATA_DIR", Path.cwd() / "data" / "ui_sessions")
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str) -> Path:
    return session_data_dir() / f"{session_id}.json"


def new_session_id() -> str:
    """Generate a new session id based on an ISO timestamp plus a random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    suffix = secrets.token_hex(4)
    return f"{stamp}-{suffix}"


def save_session(snapshot: dict[str, Any]) -> Path:
    """Write a session snapshot to disk.

    ``snapshot`` must contain a ``session_id`` key. The ``updated_at`` field
    is refreshed automatically.
    """
    session_id = snapshot["session_id"]
    snapshot["updated_at"] = _utc_now()
    path = _session_path(session_id)
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return path


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load a session snapshot by id, or return None if it does not exist."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns True if it existed and was removed."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Return metadata for saved sessions, newest first.

    Each item contains ``session_id``, ``label``, ``created_at``,
    ``updated_at``, and a ``preview`` snippet of the latest user message.
    """
    sessions: list[dict[str, Any]] = []
    data_dir = session_data_dir()
    for path in data_dir.glob("*.json"):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        session_id = snapshot.get("session_id")
        if not session_id:
            continue
        messages = snapshot.get("assistant_messages") or []
        preview = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                preview = str(msg.get("content", ""))[:80]
                break
        sessions.append(
            {
                "session_id": session_id,
                "label": snapshot.get("label") or None,
                "created_at": snapshot.get("created_at", ""),
                "updated_at": snapshot.get("updated_at", ""),
                "preview": preview,
            }
        )
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions[:limit]


def empty_snapshot(session_id: str | None = None) -> dict[str, Any]:
    """Return a fresh, empty session snapshot."""
    now = _utc_now()
    return {
        "session_id": session_id or new_session_id(),
        "label": None,
        "created_at": now,
        "updated_at": now,
        "assistant_messages": [],
        "assistant_plan_request": "",
        "assistant_plan_response": None,
        "assistant_plan_status": None,
        "assistant_selected_model": None,
        "executed_plans": [],
    }
