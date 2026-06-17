"""Direct tests for the shared action dispatch in ``core.actions``."""

from __future__ import annotations

import pytest

from safety_service import UnsafeActionError
from twfarmbot_core.actions import ActionRegistry, UnknownActionError
from twfarmbot_core.domain import Action


def test_dispatch_runs_safety_then_handler() -> None:
    seen: list[Action] = []

    def handler(action: Action) -> Action:
        seen.append(action)
        return action

    r = ActionRegistry()
    r.register("noop", handler)
    out = r.dispatch(Action(kind="noop", params={}))
    assert out.kind == "noop"
    assert seen == [out]


def test_dispatch_rejects_unsafe_before_handler() -> None:
    called = False

    def handler(action: Action) -> Action:
        nonlocal called
        called = True
        return action

    r = ActionRegistry()
    r.register("water", handler)
    with pytest.raises(UnsafeActionError):
        r.dispatch(Action(kind="water", params={"bed_id": "b1", "seconds": 9999}))
    assert called is False, "handler must not run when safety rejects"


def test_dispatch_rejects_unknown_kind() -> None:
    r = ActionRegistry()
    with pytest.raises(UnknownActionError):
        r.dispatch(Action(kind="teleport", params={}))


def test_register_rejects_duplicate_kind() -> None:
    r = ActionRegistry()
    r.register("water", lambda a: a)
    with pytest.raises(ValueError, match="already registered"):
        r.register("water", lambda a: a)


def test_kinds_returns_sorted_list() -> None:
    r = ActionRegistry()
    r.register("z", lambda a: a)
    r.register("a", lambda a: a)
    assert r.kinds() == ["a", "z"]
