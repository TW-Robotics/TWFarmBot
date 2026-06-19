"""Tests for ``POST /plan`` — natural-language planning endpoint.

The planner is stubbed via ``monkeypatch`` on ``planning_service.plan``
so no LLM credentials or network are needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from twfarmbot_api_server.app import create_app
from twfarmbot_core.domain import Action
from planning_service import PlanResult


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_plan_preview_returns_actions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_plan(request: str, **_kwargs: Any) -> PlanResult:
        return PlanResult(
            request=request,
            actions=[
                Action(kind="water", params={"bed_id": "b1", "seconds": 60}),
                Action(kind="move", params={"x": 0, "y": 0, "z": 0}),
            ],
            rationale="water then home",
        )

    monkeypatch.setattr("planning_service.plan", fake_plan)
    r = client.post("/plan", json={"request": "water bed 1 then go home"})
    assert r.status_code == 200
    body = r.json()
    assert body["request"] == "water bed 1 then go home"
    assert body["actions"] == [
        {"kind": "water", "params": {"bed_id": "b1", "seconds": 60}},
        {"kind": "move", "params": {"x": 0, "y": 0, "z": 0}},
    ]
    assert body["rationale"] == "water then home"
    assert "results" not in body  # preview mode does not dispatch


def test_plan_execute_dispatches_and_returns_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: list[Action] = []

    def fake_plan(request: str, **_kwargs: Any) -> PlanResult:
        return PlanResult(
            request=request,
            actions=[Action(kind="take_photo", params={})],
        )

    monkeypatch.setattr("planning_service.plan", fake_plan)

    def recording_dispatch(action: Action) -> Action:
        dispatched.append(action)
        return action

    monkeypatch.setattr(client.app.state.registry, "dispatch", recording_dispatch)

    r = client.post("/plan?execute=true", json={"request": "snap a pic"})
    assert r.status_code == 200
    body = r.json()
    assert body["actions"] == [{"kind": "take_photo", "params": {}}]
    assert body["results"] == [
        {"kind": "take_photo", "status": "ok", "params": {}}
    ]
    assert len(dispatched) == 1


def test_plan_returns_400_on_planner_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from planning_service import PlanError

    def fake_plan(request: str, **_kwargs: Any) -> PlanResult:
        raise PlanError("planner emitted unknown action kind 'fly'")

    monkeypatch.setattr("planning_service.plan", fake_plan)
    r = client.post("/plan", json={"request": "do the thing"})
    assert r.status_code == 400
    assert "unknown action kind" in r.json()["detail"]


def test_plan_validates_request_field(client: TestClient) -> None:
    r = client.post("/plan", json={"request": ""})
    assert r.status_code == 422


def test_plan_returns_rationale_for_empty_plan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_plan(request: str, **_kwargs: Any) -> PlanResult:
        return PlanResult(
            request=request,
            actions=[],
            rationale="request is ambiguous",
        )

    monkeypatch.setattr("planning_service.plan", fake_plan)
    r = client.post("/plan", json={"request": "do something vague"})
    assert r.status_code == 200
    body = r.json()
    assert body["actions"] == []
    assert body["rationale"] == "request is ambiguous"
