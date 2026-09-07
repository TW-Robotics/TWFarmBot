from __future__ import annotations

from fastapi.testclient import TestClient

from farmduino import Report, parse_command
from twfarmbot_os.app import create_app
from twfarmbot_os.camera import Camera
from twfarmbot_os.controller import FarmduinoController
from twfarmbot_os.transport import BaseTransport


class _MockTransport(BaseTransport):
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.locked = False
        self.pins: dict[int, float] = {}

    def request(self, gcode: str, timeout_s: float = 120.0) -> list[Report]:
        cmd = parse_command(gcode)
        reports = [Report(number=8, raw=f"R08 {gcode}"), Report(number=1, raw="R01")]
        if self.locked and cmd.code != "F09" and cmd.kind != "E":
            reports.append(Report(number=87, raw="R87"))
            return reports

        if cmd.kind == "E":
            self.locked = True
            reports.append(Report(number=87, raw="R87"))
            return reports
        if cmd.code == "F09":
            self.locked = False
            reports.append(Report(number=2, raw="R02"))
            return reports
        if cmd.code == "G00":
            if "X" in cmd.params:
                self.x = cmd.params["X"]
            if "Y" in cmd.params:
                self.y = cmd.params["Y"]
            if "Z" in cmd.params:
                self.z = cmd.params["Z"]
            reports.append(
                Report(
                    number=82,
                    params={"X": self.x, "Y": self.y, "Z": self.z},
                    raw=f"R82 X{self.x} Y{self.y} Z{self.z}",
                )
            )
        elif cmd.code == "F11":
            self.x = 0.0
            reports.extend(
                [
                    Report(number=11, raw="R11"),
                    Report(
                        number=82,
                        params={"X": 0.0, "Y": self.y, "Z": self.z},
                        raw=f"R82 X0.0 Y{self.y} Z{self.z}",
                    ),
                ]
            )
        elif cmd.code == "G28":
            self.x = self.y = self.z = 0.0
            reports.extend(
                [
                    Report(number=11, raw="R11"),
                    Report(number=12, raw="R12"),
                    Report(number=13, raw="R13"),
                    Report(
                        number=82,
                        params={"X": 0.0, "Y": 0.0, "Z": 0.0},
                        raw="R82 X0.0 Y0.0 Z0.0",
                    ),
                ]
            )
        elif cmd.code == "F41":
            p = int(cmd.params.get("P", 0))
            v = cmd.params.get("V", 0.0)
            self.pins[p] = v
            reports.append(Report(number=41, params={"P": p, "V": v}, raw=f"R41 P{p} V{v}"))
        elif cmd.code == "F42":
            p = int(cmd.params.get("P", 0))
            v = self.pins.get(p, 0.0)
            reports.append(Report(number=41, params={"P": p, "V": v}, raw=f"R41 P{p} V{v}"))

        reports.append(Report(number=2, raw="R02"))
        return reports


def _client(tmp_path) -> TestClient:
    bot = FarmduinoController(_MockTransport())
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
