"""Tests for safety_service, watering_service, and the /actions API endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from safety_service import SafetyLimits, UnsafeActionError, validate
from twfarmbot_core.domain import Action
from twfarmbot_api_server.app import app


class _StubWaterBackend:
    """Stand-in for the farmbot backend — never touches hardware."""

    def __init__(self) -> None:
        self.waters: int = 0

    def water(self, seconds: float) -> None:
        self.waters += 1


# Module-scoped stub so the API tests don't accidentally hit the real FarmBot.
@pytest.fixture(autouse=True)
def _stub_water_backend():
    import watering_service as ws
    stub = _StubWaterBackend()
    ws._backend = stub
    yield stub
    ws._backend = None


# ---------- safety_service --------------------------------------------------

def test_validate_water_passes_within_limits() -> None:
    action = Action(kind="water", params={"seconds": 10})
    out = validate(action, limits=SafetyLimits(max_water_seconds=300))
    assert out is action


def test_validate_water_rejects_zero_seconds() -> None:
    action = Action(kind="water", params={"seconds": 0})
    with pytest.raises(UnsafeActionError):
        validate(action, limits=SafetyLimits())


def test_validate_water_rejects_excessive_seconds() -> None:
    action = Action(kind="water", params={"seconds": 9999})
    with pytest.raises(UnsafeActionError, match="exceeds max"):
        validate(action, limits=SafetyLimits(max_water_seconds=300))


# ---------- watering_service -----------------------------------------------


def test_water_calls_safety_first(_stub_water_backend: _StubWaterBackend) -> None:
    from watering_service import water

    action = water(0.01)
    assert action.kind == "water"
    assert _stub_water_backend.waters == 1


def test_water_rejects_unsafe(_stub_water_backend: _StubWaterBackend) -> None:
    from watering_service import water

    with pytest.raises(UnsafeActionError):
        water(9999)
    # Safety must reject before touching the backend
    assert _stub_water_backend.waters == 0


# ---------- /actions endpoint ----------------------------------------------

def test_api_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "water" in body["actions"]


def test_api_actions_dispatches_water() -> None:
    client = TestClient(app)
    r = client.post("/actions", json={"kind": "water", "params": {"seconds": 0.01}})
    assert r.status_code == 200
    assert r.json()["action"]["kind"] == "water"


def test_api_actions_can_queue_without_waiting() -> None:
    client = TestClient(app)
    r = client.post(
        "/actions?wait=false",
        json={"kind": "water", "params": {"seconds": 0.01}},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"


def test_api_actions_rejects_unknown_kind() -> None:
    client = TestClient(app)
    r = client.post("/actions", json={"kind": "teleport", "params": {}})
    assert r.status_code == 404
    assert "teleport" in r.json()["detail"]


def test_api_actions_runs_through_safety() -> None:
    client = TestClient(app)
    r = client.post("/actions", json={"kind": "water", "params": {"seconds": 9999}})
    assert r.status_code == 400
    assert "exceeds max" in r.json()["detail"]


def test_api_actions_validates_payload_shape() -> None:
    client = TestClient(app)
    r = client.post("/actions", json={"params": {"seconds": 1}})  # no kind
    assert r.status_code == 422
