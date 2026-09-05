"""Operator cancel flag and mid-run follow-up mailbox.

Aborting the browser fetch does not stop LangGraph or the gantry. The UI
POSTs ``/chat/cancel``; tools and the loop check ``is_cancelled()``.

``/chat/followup`` queues user text on the active thread. The tools node
drains it after the current tool — it does not e-stop.
"""

from __future__ import annotations

import threading

STOP_CANCELLED = "cancelled"

_lock = threading.Lock()
_cancelled: set[str] = set()
_followups: dict[str, list[str]] = {}
_active_thread: str | None = None


class RunCancelled(Exception):
    """Raised when the operator stops the current agent run."""


class FollowupRejected(Exception):
    """No matching in-flight run to attach a follow-up to."""


def begin_run(thread_id: str) -> None:
    global _active_thread
    with _lock:
        _cancelled.discard(thread_id)
        _followups.pop(thread_id, None)
        _active_thread = thread_id


def end_run(thread_id: str) -> None:
    global _active_thread
    with _lock:
        if _active_thread == thread_id:
            _active_thread = None
        _cancelled.discard(thread_id)
        _followups.pop(thread_id, None)


def cancel_run(thread_id: str | None = None) -> str | None:
    """Mark a thread (or the active run) cancelled. Returns the id."""
    with _lock:
        tid = thread_id or _active_thread
        if tid:
            _cancelled.add(tid)
            _followups.pop(tid, None)
        return tid


def is_cancelled(thread_id: str | None = None) -> bool:
    with _lock:
        tid = thread_id or _active_thread
        return bool(tid and tid in _cancelled)


def raise_if_cancelled(thread_id: str | None = None) -> None:
    if is_cancelled(thread_id):
        raise RunCancelled("stopped by operator")


def active_thread() -> str | None:
    with _lock:
        return _active_thread


def enqueue_followup(thread_id: str | None, text: str) -> str:
    """Queue user text for the active run. Returns the thread id."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("follow-up text is empty")
    with _lock:
        tid = thread_id or _active_thread
        if not tid or tid != _active_thread or tid in _cancelled:
            raise FollowupRejected("no active run")
        _followups.setdefault(tid, []).append(cleaned)
        return tid


def peek_followup(thread_id: str | None = None) -> bool:
    with _lock:
        tid = thread_id or _active_thread
        return bool(tid and _followups.get(tid))


def drain_followup(thread_id: str | None = None) -> str | None:
    """Return joined pending follow-ups, or ``None``."""
    with _lock:
        tid = thread_id or _active_thread
        if not tid:
            return None
        parts = _followups.pop(tid, [])
        return " ".join(parts) if parts else None
