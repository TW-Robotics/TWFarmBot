"""Tests for the Assistant tab in the Streamlit UI.

Streamlit's renderers are hard to unit-test directly. We exercise the
end-to-end flow by talking to the real ``POST /plan`` endpoint with
``planning_service.plan`` stubbed, then verifying the endpoint returns
the shapes the UI expects.
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


def test_assistant_preview_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """UI renders the plan by iterating response['actions']."""

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

    r = client.post("/plan", json={"request": "water b1 then home"})
    assert r.status_code == 200
    body = r.json()

    # Shape the UI's _render_assistant consumes.
    assert isinstance(body["actions"], list)
    for action in body["actions"]:
        assert "kind" in action
        assert "params" in action
    assert body["actions"][0]["kind"] == "water"
    assert body["actions"][1]["kind"] == "move"
    assert body["rationale"] == "water then home"


def test_assistant_execute_returns_per_action_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_plan(request: str, **_kwargs: Any) -> PlanResult:
        return PlanResult(
            request=request,
            actions=[Action(kind="take_photo", params={})],
        )

    monkeypatch.setattr("planning_service.plan", fake_plan)
    monkeypatch.setattr(
        client.app.state.registry, "dispatch", lambda a: a,
    )

    r = client.post("/plan?execute=true", json={"request": "snap a pic"})
    assert r.status_code == 200
    body = r.json()

    # Shape the UI iterates after Run.
    assert "results" in body
    assert len(body["results"]) == len(body["actions"])
    for item in body["results"]:
        assert "kind" in item
        assert "status" in item
        assert item["status"] == "ok"


def test_assistant_empty_plan_surfaces_warning_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UI shows a warning when actions is empty, plus the rationale."""

    def fake_plan(request: str, **_kwargs: Any) -> PlanResult:
        return PlanResult(
            request=request,
            actions=[],
            rationale="request is ambiguous",
        )

    monkeypatch.setattr("planning_service.plan", fake_plan)
    r = client.post("/plan", json={"request": "do nothing safe"})
    assert r.status_code == 200
    body = r.json()
    assert body["actions"] == []
    assert body["rationale"] == "request is ambiguous"
