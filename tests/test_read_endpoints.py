"""Tests for the read-only GET routes (position, status, pin, messages)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import twfarmbot_api_server.read as read_module
from twfarmbot_api_server.app import app


class _StubBot:
    def read_status(self, path=None):
        return {"connected": True, "path": path}


class _StubBackend:
    def get_xyz(self):
        return {"x": 100.0, "y": 200.0, "z": 50.0}

    def read_pin(self, pin, mode="digital"):
        return 42 if pin == 13 else 0

    def get_last_messages(self):
        return ["msg1", "msg2"]

    def get_images(self, limit=10, refresh=False):
        return [
            {
                "id": 1,
                "created_at": "2026-06-18T10:36:51.909Z",
                "attachment_url": "https://example.test/photo.jpg",
                "meta": {"x": 2.2, "y": 99.4, "z": 0.0},
            }
        ][:limit]

    def _bot(self):
        return _StubBot()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(read_module.farmbot, "backend", _StubBackend())
    return TestClient(app)


def test_get_position(client: TestClient) -> None:
    r = client.get("/position")
    assert r.status_code == 200
    assert r.json()["xyz"] == {"x": 100.0, "y": 200.0, "z": 50.0}
    assert r.json()["workspace"]["x"]["max"] == 2000
    assert r.json()["workspace"]["y"]["max"] == 650
    assert r.json()["workspace"]["z"]["max"] is None


def test_get_status_default_path(client: TestClient) -> None:
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["path"] is None
    assert body["state"]["connected"] is True


def test_get_status_with_path(client: TestClient) -> None:
    r = client.get("/status", params={"path": "position"})
    assert r.status_code == 200
    assert r.json()["path"] == "position"


def test_get_pin(client: TestClient) -> None:
    r = client.get("/pin/13")
    assert r.status_code == 200
    assert r.json() == {"pin": 13, "mode": "digital", "value": 42}


def test_get_pin_with_mode(client: TestClient) -> None:
    r = client.get("/pin/13", params={"mode": "analog"})
    assert r.status_code == 200
    assert r.json()["mode"] == "analog"


def test_get_messages(client: TestClient) -> None:
    r = client.get("/messages")
    assert r.status_code == 200
    assert r.json() == {"last_messages": ["msg1", "msg2"]}


def test_get_images(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(read_module, "list_captures", lambda limit=10: [])
    r = client.get("/images", params={"limit": 1, "refresh": "true"})
    assert r.status_code == 200
    assert r.json()["images"][0]["attachment_url"] == "https://example.test/photo.jpg"


def test_read_error_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenBackend:
        def get_xyz(self):
            raise RuntimeError("broker down")

    monkeypatch.setattr(read_module.farmbot, "backend", _BrokenBackend())
    r = client.get("/position")
    assert r.status_code == 502
    assert "broker down" in r.json()["detail"]


def test_get_pins(client: TestClient) -> None:
    r = client.get("/pins")
    assert r.status_code == 200
    body = r.json()
    assert "pins" in body
    # The dev.yaml in the repo has at least one named pin
    assert isinstance(body["pins"], list)
    if body["pins"]:
        first = body["pins"][0]
        assert "label" in first
        assert "pin" in first
        assert "mode" in first


def test_get_positions(client: TestClient) -> None:
    r = client.get("/positions")
    assert r.status_code == 200
    body = r.json()
    assert "positions" in body
    assert isinstance(body["positions"], list)
    labels = {item["label"] for item in body["positions"]}
    assert "Home" in labels
    assert "Cutting" in labels
    if body["positions"]:
        first = body["positions"][0]
        assert "label" in first
        assert "group" in first
        assert "set" in first


def test_put_positions(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "dev.yaml"
    path.write_text("safety: {}\npositions: []\n", encoding="utf-8")
    monkeypatch.setenv("TWFB_CONFIG", str(path))
    r = client.put(
        "/positions",
        json={
            "positions": [
                {"label": "Home", "group": "Travel", "x": 0, "y": 0, "z": 0},
                {"label": "Cutting", "group": "Tools"},
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()["positions"]
    assert body[0]["set"] is True
    assert body[1]["set"] is False
    listed = client.get("/positions").json()["positions"]
    assert listed[1]["label"] == "Cutting"


def test_get_scripts(client: TestClient) -> None:
    r = client.get("/scripts")
    assert r.status_code == 200
    labels = {item["label"] for item in r.json()["scripts"]}
    assert "Get tool servo" in labels


def test_put_scripts(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "dev.yaml"
    path.write_text("scripts: []\n", encoding="utf-8")
    monkeypatch.setenv("TWFB_CONFIG", str(path))
    r = client.put(
        "/scripts",
        json={
            "scripts": [
                {
                    "label": "Get tool servo",
                    "waypoints": [
                        {"x": 1927, "y": 186, "z": 0},
                        {"x": 1927, "y": 186, "z": -297},
                    ],
                }
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()["scripts"]
    assert body[0]["label"] == "Get tool servo"
    assert len(body[0]["waypoints"]) == 2


def test_get_garden(client: TestClient) -> None:
    r = client.get("/garden")
    assert r.status_code == 200
    body = r.json()
    assert body["bounds"]["width"] > 0
    assert body["robot"] == {"x": 100.0, "y": 200.0, "z": 50.0}
    assert isinstance(body["entities"], list)
    assert isinstance(body["zones"], list)
    assert isinstance(body["cells"], list)
    assert body["cell_mm"] == 250


def test_find_garden_locations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spatial_service.observations import ObservationStore

    store = ObservationStore(":memory:")
    store.record(
        x=1200, y=400, ndre_mean=0.02, stress_fraction=0.7, action_hint="consider_water"
    )
    store.record(x=200, y=100, ndre_mean=0.3, stress_fraction=0.05, action_hint="ok")
    monkeypatch.setattr(
        "spatial_service.observations.get_observation_store",
        lambda db_path=None: store,
    )
    r = client.get("/garden/find", params={"predicate": "stressed"})
    assert r.status_code == 200
    body = r.json()
    assert body["locations"][0]["x"] == 1200
    ripe = client.get("/garden/find", params={"predicate": "ripe"})
    assert ripe.status_code == 200
    assert ripe.json()["locations"] == []
    bad = client.get("/garden/find", params={"predicate": "moon"})
    assert bad.status_code == 400


def test_get_memory(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from planning_service import memory as memory_mod

    store = memory_mod.MemoryStore(tmp_path / "memory.db")
    stored = store.remember(
        "NDRE stress at bed A",
        category="agronomy",
        x=120.0,
        y=800.0,
        tags=["ndre"],
    )
    monkeypatch.setattr(memory_mod, "get_memory_store", lambda: store)
    r = client.get("/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["memories"][0]["id"] == stored["id"]
    assert body["memories"][0]["content"] == "NDRE stress at bed A"
    assert body["memories"][0]["x"] == 120.0


def test_get_meteorology(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from planning_service import meteorology as meteo_mod

    snapshot = {
        "timestamp": "2026-09-07T08:00:00+00:00",
        "location": {
            "latitude": 48.2394,
            "longitude": 16.3776,
            "elevation_m": 165.0,
            "timezone": "Europe/Vienna",
        },
        "sun": {
            "altitude_deg": 32.1,
            "azimuth_deg": 140.0,
            "sun_phase": "day",
            "is_daylight": True,
            "clear_sky_irradiance_w_m2": 612.0,
        },
        "weather": {"temperature_c": 18.4, "source": "open-meteo"},
        "agronomic_advice": {
            "camera_lighting_quality": "acceptable",
            "watering_suitability": "favorable",
        },
    }
    monkeypatch.setattr(meteo_mod, "get_meteorology", lambda: snapshot)
    r = client.get("/meteorology")
    assert r.status_code == 200
    assert r.json()["sun"]["sun_phase"] == "day"
    assert r.json()["weather"]["temperature_c"] == 18.4
