"""Local FarmBot OS: Farmduino G/E/F + R codes + camera over HTTP."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from twfarmbot_os.camera import Camera
from twfarmbot_os.controller import FarmduinoController
from twfarmbot_os.transport import TransportError, build_transport

log = logging.getLogger("twfarmbot.os")

_PUMP_PIN = int(os.getenv("FARMBOT_PUMP_PIN", "8"))


class MoveAxisBody(BaseModel):
    distance: float


class MoveBody(BaseModel):
    x: float | None = None
    y: float | None = None
    z: float | None = None
    speed: float | None = None


class AxisBody(BaseModel):
    axis: str = "all"


class CommandBody(BaseModel):
    gcode: str = Field(..., min_length=1)


class PinWriteBody(BaseModel):
    pin: int
    value: float
    mode: int | str = 0


class ServoBody(BaseModel):
    pin: int
    angle: float


class PulseBody(BaseModel):
    pin: int
    value: float = 1
    wait_ms: float
    then: float = 0
    mode: int = 0


def _mode_int(mode: int | str) -> int:
    if isinstance(mode, str):
        return 0 if mode.lower().startswith("d") else 1
    return int(mode)


def create_app(
    controller: FarmduinoController | None = None,
    camera: Camera | None = None,
) -> FastAPI:
    bot = controller or FarmduinoController(build_transport())
    cam = camera or Camera()
    app = FastAPI(title="TWFarmbotOS", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _ok(message: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "success": True,
            "message": message,
            "state": bot.snapshot(),
            "reports": bot.last_reports,
        }
        if extra:
            payload.update(extra)
        return payload

    def _run(fn, message: str):
        try:
            fn()
        except TransportError as err:
            raise HTTPException(
                status_code=409 if bot.locked else 400,
                detail={"success": False, "message": str(err), "state": bot.snapshot()},
            ) from err
        return _ok(message)

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"success": True, "message": "TWFarmbotOS", "state": bot.snapshot()}

    @app.get("/state")
    def state() -> dict[str, Any]:
        try:
            bot.refresh_position()
        except TransportError:
            log.debug("position refresh failed", exc_info=True)
        return {"success": True, "state": bot.snapshot()}

    @app.post("/command")
    def command(body: CommandBody) -> dict[str, Any]:
        return _run(lambda: bot.execute(body.gcode), body.gcode)

    @app.post("/move")
    def move(body: MoveBody) -> dict[str, Any]:
        return _run(
            lambda: bot.move_absolute(body.x, body.y, body.z, body.speed),
            "move",
        )

    @app.post("/move/{axis}")
    def move_axis(axis: str, body: MoveAxisBody) -> dict[str, Any]:
        if axis.lower() not in {"x", "y", "z"}:
            raise HTTPException(status_code=400, detail="axis must be x, y, or z")
        return _run(lambda: bot.move_axis(axis, body.distance), f"move {axis}")

    @app.post("/home")
    def home() -> dict[str, Any]:
        return _run(lambda: bot.find_home("all"), "home")

    @app.post("/find_home")
    def find_home(body: AxisBody | None = None) -> dict[str, Any]:
        axis = body.axis if body else "all"
        return _run(lambda: bot.find_home(axis), f"find_home {axis}")

    @app.post("/calibrate")
    def calibrate(body: AxisBody) -> dict[str, Any]:
        return _run(lambda: bot.calibrate(body.axis), f"calibrate {body.axis}")

    @app.post("/zero")
    def zero(body: AxisBody | None = None) -> dict[str, Any]:
        axis = body.axis if body else "all"
        return _run(lambda: bot.set_zero(axis), f"zero {axis}")

    @app.post("/stop")
    def stop() -> dict[str, Any]:
        return _run(lambda: bot.e_stop(), "emergency stop")

    @app.post("/unlock")
    def unlock() -> dict[str, Any]:
        return _run(lambda: bot.unlock(), "unlock")

    @app.post("/water/on")
    def water_on() -> dict[str, Any]:
        return _run(lambda: bot.write_pin(_PUMP_PIN, 1, 0), "water on")

    @app.post("/water/off")
    def water_off() -> dict[str, Any]:
        return _run(lambda: bot.write_pin(_PUMP_PIN, 0, 0), "water off")

    @app.post("/pins/write")
    def pins_write(body: PinWriteBody) -> dict[str, Any]:
        return _run(
            lambda: bot.write_pin(body.pin, body.value, _mode_int(body.mode)),
            "write_pin",
        )

    @app.get("/pins/{pin}")
    def pins_read(pin: int, mode: str = "digital") -> dict[str, Any]:
        try:
            value = bot.read_pin(pin, _mode_int(mode))
        except TransportError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return _ok("read_pin", {"pin": pin, "mode": mode, "value": value})

    @app.post("/pins/mode")
    def pins_mode(body: PinWriteBody) -> dict[str, Any]:
        return _run(
            lambda: bot.set_pin_mode(body.pin, _mode_int(body.mode)),
            "set_pin_mode",
        )

    @app.post("/pins/pulse")
    def pins_pulse(body: PulseBody) -> dict[str, Any]:
        return _run(
            lambda: bot.pulse_pin(
                body.pin, body.value, body.wait_ms, body.then, body.mode
            ),
            "pulse_pin",
        )

    @app.post("/servo")
    def servo(body: ServoBody) -> dict[str, Any]:
        return _run(lambda: bot.servo(body.pin, body.angle), "servo")

    @app.get("/endstops")
    def endstops() -> dict[str, Any]:
        return _run(lambda: bot.endstops(), "endstops")

    @app.post("/photo")
    def photo() -> dict[str, Any]:
        try:
            info = cam.capture()
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(err)) from err
        return _ok("photo", {"photo": info})

    @app.get("/photos")
    def photos(limit: int = 10) -> dict[str, Any]:
        return {"success": True, "images": cam.list_photos(limit)}

    @app.get("/photos/{photo_id}")
    def photo_file(photo_id: str) -> FileResponse:
        path = cam.path_for(photo_id)
        if path is None:
            raise HTTPException(status_code=404, detail="photo not found")
        return FileResponse(path, media_type="image/jpeg")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.getenv("FARMBOT_LOG_LEVEL", "INFO"))
    host = os.getenv("FARMBOT_OS_HOST", "0.0.0.0")
    port = int(os.getenv("FARMBOT_OS_PORT", "3001"))
    uvicorn.run("twfarmbot_os.app:app", host=host, port=port, reload=False)
