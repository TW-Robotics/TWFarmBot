"""Tests for the UI session persistence helper."""

from __future__ import annotations

from typing import Any

import pytest

from twfarmbot_ui import history


@pytest.fixture
def tmp_history_dir(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Redirect session storage to a temp directory for each test."""
    monkeypatch.setattr(history, "session_data_dir", lambda: tmp_path)
    return tmp_path


def test_empty_snapshot_has_required_keys() -> None:
    snap = history.empty_snapshot("test-id")
    assert snap["session_id"] == "test-id"
    assert snap["assistant_messages"] == []
    assert snap["executed_plans"] == []
    assert "created_at" in snap
    assert "updated_at" in snap


def test_save_and_load_session_round_trip(tmp_history_dir: Any) -> None:
    snap = history.empty_snapshot("sess-1")
    snap["label"] = "my session"
    snap["assistant_messages"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    snap["executed_plans"] = [
        {
            "request": "water bed",
            "actions": [{"kind": "water", "params": {"seconds": 10}}],
            "results": [{"kind": "water", "ok": True, "detail": None}],
            "queued_at": "2026-06-24T12:00:00",
            "status": "ok",
        }
    ]
    history.save_session(snap)

    loaded = history.load_session("sess-1")
    assert loaded is not None
    assert loaded["label"] == "my session"
    assert loaded["assistant_messages"][0]["content"] == "hello"
    assert len(loaded["executed_plans"]) == 1
    assert loaded["executed_plans"][0]["status"] == "ok"


def test_save_session_updates_updated_at(tmp_history_dir: Any) -> None:
    snap = history.empty_snapshot("sess-2")
    original_updated = snap["updated_at"]
    history.save_session(snap)
    loaded = history.load_session("sess-2")
    assert loaded is not None
    assert loaded["updated_at"] >= original_updated


def test_load_missing_session_returns_none(tmp_history_dir: Any) -> None:
    assert history.load_session("does-not-exist") is None


def test_delete_session(tmp_history_dir: Any) -> None:
    snap = history.empty_snapshot("sess-del")
    history.save_session(snap)
    assert history.delete_session("sess-del") is True
    assert history.load_session("sess-del") is None
    assert history.delete_session("sess-del") is False


def test_list_sessions_orders_by_updated_desc(tmp_history_dir: Any) -> None:
    old = history.empty_snapshot("old-sess")
    history.save_session(old)

    new = history.empty_snapshot("new-sess")
    new["assistant_messages"] = [{"role": "user", "content": "latest"}]
    history.save_session(new)

    sessions = history.list_sessions()
    ids = [s["session_id"] for s in sessions]
    assert ids == ["new-sess", "old-sess"]


def test_list_sessions_includes_preview(tmp_history_dir: Any) -> None:
    snap = history.empty_snapshot("preview-sess")
    snap["assistant_messages"] = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "water the tomatoes please"},
    ]
    history.save_session(snap)

    sessions = history.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["preview"] == "water the tomatoes please"


def test_list_sessions_respects_limit(tmp_history_dir: Any) -> None:
    for i in range(5):
        snap = history.empty_snapshot(f"sess-{i}")
        history.save_session(snap)
    assert len(history.list_sessions(limit=2)) == 2
    assert len(history.list_sessions(limit=10)) == 5
