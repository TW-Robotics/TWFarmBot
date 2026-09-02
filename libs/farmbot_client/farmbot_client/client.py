"""HTTP client for the local TWFarmbotOS REST API on the Raspberry Pi."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:3001"


class FarmBotConnectionError(RuntimeError):
    """Raised when the local FarmBot HTTP API cannot be reached."""


def base_url() -> str:
    return os.getenv("FARMBOT_LOCAL_URL", DEFAULT_URL).rstrip("/")


class LocalFarmbot:
    def __init__(
        self,
        url: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.url = (url or base_url()).rstrip("/")
        self._http = client or httpx.Client(base_url=self.url, timeout=timeout_s)

    def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, json=json, params=params)
        except httpx.HTTPError as err:
            raise FarmBotConnectionError(
                f"local FarmBot {method} {self.url}{path} failed: {err}"
            ) from err
        if response.status_code >= 400:
            raise FarmBotConnectionError(
                f"local FarmBot {method} {path} -> {response.status_code}: {response.text}"
            )
        if not response.content:
            return {}
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    def ping(self) -> dict[str, Any]:
        return self.request("GET", "/")

    def get_state(self, *, refresh: bool = False) -> dict[str, Any]:
        params = {"refresh": "true"} if refresh else None
        payload = self.request("GET", "/state", params=params)
        state = payload.get("state")
        return state if isinstance(state, dict) else payload

    def get_xyz(self) -> dict[str, float]:
        state = self.get_state()
        return {axis: float(state.get(axis, 0) or 0) for axis in ("x", "y", "z")}

    def refresh_state(self) -> dict[str, Any]:
        return self.get_state(refresh=True)

    def command(self, gcode: str) -> dict[str, Any]:
        return self.request("POST", "/command", json={"gcode": gcode})

    def move_axis(
        self, axis: str, distance: float, speed: float | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"distance": float(distance)}
        if speed is not None:
            payload["speed"] = float(speed)
        return self.request(
            "POST",
            f"/move/{axis.lower()}",
            json=payload,
        )

    def move(self, x: float, y: float, z: float, speed: float | None = None) -> None:
        payload: dict[str, Any] = {"x": float(x), "y": float(y), "z": float(z)}
        if speed is not None:
            payload["speed"] = float(speed)
        self.request("POST", "/move", json=payload)

    def find_home(self, axis: str = "all", speed: float = 100) -> None:
        payload: dict[str, Any] = {"axis": axis}
        if speed != 100:
            payload["speed"] = float(speed)
        self.request("POST", "/find_home", json=payload)

    def set_home(self, axis: str = "all") -> None:
        self.request("POST", "/zero", json={"axis": axis})

    def calibrate(self, axis: str) -> dict[str, Any]:
        return self.request("POST", "/calibrate", json={"axis": axis})

    def e_stop(self) -> None:
        self.request("POST", "/stop")

    def unlock(self) -> None:
        self.request("POST", "/unlock")

    def water_on(self) -> None:
        self.request("POST", "/water/on")

    def water_off(self) -> None:
        self.request("POST", "/water/off")

    def write_pin(
        self, pin: int, value: float, mode: str = "digital"
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            "/pins/write",
            json={"pin": int(pin), "value": value, "mode": mode},
        )

    def read_pin(self, pin: int, mode: str = "digital") -> float:
        payload = self.request("GET", f"/pins/{int(pin)}", params={"mode": mode})
        return float(payload.get("value", 0) or 0)

    def set_servo_angle(self, pin: int, angle: float) -> dict[str, Any]:
        return self.request("POST", "/servo", json={"pin": pin, "angle": angle})

    def take_photo(self) -> dict[str, Any]:
        return self.request("POST", "/photo")

    def get_images(self, limit: int = 10) -> list[dict[str, Any]]:
        payload = self.request("GET", "/photos", params={"limit": int(limit)})
        images = payload.get("images")
        return images if isinstance(images, list) else []

    def read_status(self, path: str | None = None) -> dict[str, Any]:
        state = self.get_state()
        if path:
            return {"path": path, **state}
        return state


_client: LocalFarmbot | None = None


def get_client() -> LocalFarmbot:
    global _client
    if _client is None:
        _client = LocalFarmbot()
    return _client


def connect(
    url: str | None = None,
    *,
    timeout_s: float = 30.0,
    retries: int = 1,
    backoff_s: float = 0.0,
) -> LocalFarmbot:
    del retries, backoff_s
    global _client
    client = LocalFarmbot(url, timeout_s=timeout_s)
    client.ping()
    _client = client
    return client
