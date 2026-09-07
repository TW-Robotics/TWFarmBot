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
        self.on: bool = False

    def water(self, seconds: float) -> None:
        del seconds
        self.waters += 1
        self.on = True

    def turn_on(self) -> None:
        self.on = True

    def stop(self) -> None:
        self.on = False

    def status(self) -> dict:
        return {"on": self.on, "power_w": 6.0 if self.on else 0.0}


# Module-scoped stub so the API tests don't accidentally hit the real FarmBot.
@pytest.fixture(autouse=True)
def _stub_water_backend():
    import watering_service as ws

    stub = _StubWaterBackend()
    ws._backend = stub
    yield stub
    ws._backend = None


# ---------- safety_service --------------------------------------------------


def test_validate_water_allows_omitted_seconds() -> None:
    action = Action(kind="water", params={})
    assert validate(action, limits=SafetyLimits()) is action


def test_water_without_seconds(_stub_water_backend: _StubWaterBackend) -> None:
    from watering_service import water

    action = water()
    assert action.kind == "water"
    assert _stub_water_backend.waters == 1


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


def test_api_actions_dispatches_water_cycle() -> None:
    client = TestClient(app)
    r = client.post("/actions", json={"kind": "water", "params": {}})
    assert r.status_code == 200
    assert r.json()["action"]["kind"] == "water"


def test_shelly_water_pulses_switch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import watering_service.backends.shelly as shelly

    monkeypatch.setenv("SHELLY_HOST", "10.0.0.8")
    posts: list[tuple[str, dict]] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *args) -> None:
            del args

        def post(self, url: str, json: dict | None = None) -> _Resp:
            posts.append((url, json or {}))
            return _Resp()

    monkeypatch.setattr(shelly.httpx, "Client", _Client)
    shelly.backend.water(99)
    assert posts == [
        ("http://10.0.0.8/rpc/Switch.Set", {"id": 0, "on": True, "toggle_after": 60})
    ]
    posts.clear()
    shelly.backend.stop()
    assert posts == [("http://10.0.0.8/rpc/Switch.Set", {"id": 0, "on": False})]


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


def test_watering_io_toggle(_stub_water_backend: _StubWaterBackend) -> None:
    client = TestClient(app)
    r = client.get("/watering")
    assert r.status_code == 200
    assert r.json()["on"] is False
    r = client.post("/watering", json={"on": True})
    assert r.status_code == 200
    assert r.json()["on"] is True
    assert _stub_water_backend.on is True
    r = client.post("/watering", json={"on": False})
    assert r.json()["on"] is False
    assert _stub_water_backend.on is False
