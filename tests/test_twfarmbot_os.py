from __future__ import annotations

from fastapi.testclient import TestClient

from twfarmbot_os.app import create_app
from twfarmbot_os.camera import Camera
from twfarmbot_os.controller import FarmduinoController
from twfarmbot_os.transport import SimulatedTransport


def _client(tmp_path) -> TestClient:
    bot = FarmduinoController(SimulatedTransport())
    cam = Camera(directory=str(tmp_path / "photos"))
    return TestClient(create_app(bot, cam))


def test_absolute_move_and_state(tmp_path) -> None:
    client = _client(tmp_path)
    res = client.post("/move", json={"x": 40, "y": 50, "z": -10})
    assert res.status_code == 200
    state = client.get("/state").json()["state"]
    assert state["x"] == 40
    assert state["y"] == 50
    assert state["z"] == -10


def test_move_speed_percent_is_encoded_as_steps_per_second(tmp_path) -> None:
    client = _client(tmp_path)
    res = client.post("/move", json={"x": 1, "y": 0, "z": 0, "speed": 50})
    assert res.status_code == 200
    joined = " ".join(res.json()["reports"])
    assert "A223" in joined


def test_relative_move(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/move", json={"x": 10, "y": 0, "z": 0})
    client.post("/move/x", json={"distance": 5})
    assert client.get("/state").json()["state"]["x"] == 15


def test_find_home_uses_firmware_not_software_zero(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/move", json={"x": 100, "y": 100, "z": 10})
    res = client.post("/find_home", json={"axis": "x"})
    assert res.status_code == 200
    assert any(line.startswith("R11") for line in res.json()["reports"])
    assert client.get("/state").json()["state"]["x"] == 0


def test_e_stop_blocks_then_unlock(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.post("/stop").status_code == 200
    blocked = client.post("/move", json={"x": 1, "y": 0, "z": 0})
    assert blocked.status_code == 409
    assert client.post("/unlock").status_code == 200
    assert client.post("/move", json={"x": 2, "y": 0, "z": 0}).status_code == 200


def test_raw_command_and_pin(tmp_path) -> None:
    client = _client(tmp_path)
    written = client.post("/command", json={"gcode": "F41 P8 V1 M0"})
    assert written.status_code == 200
    read = client.get("/pins/8", params={"mode": "digital"})
    assert read.json()["value"] == 1


def test_photo(tmp_path) -> None:
    client = _client(tmp_path)
    shot = client.post("/photo")
    assert shot.status_code == 200
    photo_id = shot.json()["photo"]["id"]
    listed = client.get("/photos").json()["images"]
    assert listed[0]["id"] == photo_id
    assert client.get(f"/photos/{photo_id}").status_code == 200
