"""Operator cancel flag for the in-flight agent run.

Aborting the browser fetch does not stop LangGraph or the gantry. The UI
POSTs ``/chat/cancel``; tools and the loop check ``is_cancelled()``.
"""

from __future__ import annotations

import threading

STOP_CANCELLED = "cancelled"

_lock = threading.Lock()
_cancelled: set[str] = set()
_active_thread: str | None = None


class RunCancelled(Exception):
    """Raised when the operator stops the current agent run."""


def begin_run(thread_id: str) -> None:
    global _active_thread
    with _lock:
        _cancelled.discard(thread_id)
        _active_thread = thread_id


def end_run(thread_id: str) -> None:
    global _active_thread
    with _lock:
        if _active_thread == thread_id:
            _active_thread = None
        _cancelled.discard(thread_id)


def cancel_run(thread_id: str | None = None) -> str | None:
    """Mark a thread (or the active run) cancelled. Returns the id."""
    global _active_thread
    with _lock:
        tid = thread_id or _active_thread
        if tid:
            _cancelled.add(tid)
        return tid


def is_cancelled(thread_id: str | None = None) -> bool:
    with _lock:
        tid = thread_id or _active_thread
        return bool(tid and tid in _cancelled)


def raise_if_cancelled(thread_id: str | None = None) -> None:
    if is_cancelled(thread_id):
        raise RunCancelled("stopped by operator")
