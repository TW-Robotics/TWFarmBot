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


MAX_FEEDBACK_TEXT_CHARS = int(os.getenv("TWFB_FEEDBACK_MAX_TEXT_CHARS", "4000"))
MAX_FEEDBACK_CONTEXT_MESSAGES = int(
    os.getenv("TWFB_FEEDBACK_MAX_CONTEXT_MESSAGES", "20")
)
MAX_FEEDBACK_LIST_ITEMS = int(os.getenv("TWFB_FEEDBACK_MAX_LIST_ITEMS", "25"))


def session_data_dir() -> Path:
    """Return the directory used to store session JSON files."""
    path = Path(os.getenv("TWFB_UI_DATA_DIR", Path.cwd() / "data" / "ui_sessions"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def feedback_log_path(session_id: str | None = None) -> Path:
    """Return the append-only feedback/event log path.

    By default each assistant chat session gets its own JSONL dataset file.
    ``TWFB_FEEDBACK_LOG`` can still force a single explicit path for exports
    or deployments that want central collection.
    """
    override = os.getenv("TWFB_FEEDBACK_LOG")
    if override:
        path = Path(override)
    else:
        name = session_id or "feedback_events"
        path = session_data_dir() / "feedback" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def new_event_id() -> str:
    """Generate a compact event id for feedback/preference records."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{stamp}-{secrets.token_hex(6)}"


def sanitize_for_feedback(value: Any) -> Any:
    """Return a compact JSON-safe copy suitable for feedback datasets.

    This keeps tool-call structure and text, but strips bulky image payloads
    and caps large strings/lists so JSONL logs remain small.
    """
    if isinstance(value, dict):
        return {str(k): sanitize_for_feedback(v) for k, v in value.items()}
    if isinstance(value, list):
        out = [sanitize_for_feedback(v) for v in value[:MAX_FEEDBACK_LIST_ITEMS]]
        if len(value) > MAX_FEEDBACK_LIST_ITEMS:
            out.append({"_truncated_items": len(value) - MAX_FEEDBACK_LIST_ITEMS})
        return out
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return "[image data omitted]"
        if len(value) > MAX_FEEDBACK_TEXT_CHARS:
            return value[:MAX_FEEDBACK_TEXT_CHARS] + "...[truncated]"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def compact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the bounded, sanitized message context stored with feedback."""
    return sanitize_for_feedback(messages[-MAX_FEEDBACK_CONTEXT_MESSAGES:])


def append_feedback_event(event: dict[str, Any]) -> Path:
    """Append one compact feedback/preference event as JSONL."""
    payload = sanitize_for_feedback(
        {
            "event_id": event.get("event_id") or new_event_id(),
            "created_at": event.get("created_at") or _utc_now(),
            **event,
        }
    )
    path = feedback_log_path(event.get("session_id"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    return path


def load_feedback_events(session_id: str | None = None) -> list[dict[str, Any]]:
    """Load feedback events from JSONL. Intended for tests/export scripts."""
    path = feedback_log_path(session_id)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


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
        "assistant_metrics": {},
        "executed_plans": [],
    }
