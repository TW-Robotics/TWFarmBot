"""Shelly plug RPC. Watering uses auto-off; lighting stays on until off."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx

from twfarmbot_core.config import load_yaml_config

log = logging.getLogger(__name__)


def _parse_host(raw: str) -> str:
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = parsed.netloc or parsed.path
    if not host:
        raise ValueError(f"invalid Shelly host: {raw!r}")
    return host.split("/")[0]


def _watering_cfg() -> dict[str, Any]:
    watering = load_yaml_config().get("watering", {}) or {}
    return watering.get("shelly", {}) or {} if isinstance(watering, dict) else {}


def _host() -> str:
    raw = (os.getenv("SHELLY_HOST") or str(_watering_cfg().get("host") or "")).strip()
    if not raw:
        raise ValueError(
            "Shelly watering needs SHELLY_HOST or watering.shelly.host in config"
        )
    return _parse_host(raw)


def _cycle_seconds() -> float:
    watering = load_yaml_config().get("watering", {}) or {}
    return float(os.getenv("FARMBOT_WATER_SECONDS", watering.get("cycle_seconds", 60)))


def _switch_id() -> int:
    return int(os.getenv("SHELLY_SWITCH_ID", _watering_cfg().get("switch_id", 0)))


def _auth() -> tuple[str, str] | None:
    user = os.getenv("SHELLY_USER") or str(_watering_cfg().get("user") or "")
    password = os.getenv("SHELLY_PASSWORD") or str(_watering_cfg().get("password") or "")
    if user or password:
        return (user, password)
    return None


class ShellyPlug:
    """Gen2/3 RPC switch, with Gen1 ``/relay`` fallback."""

    def __init__(
        self,
        host: str,
        switch_id: int = 0,
        auth: tuple[str, str] | None = None,
    ) -> None:
        self.host = _parse_host(host)
        self.switch_id = int(switch_id)
        self.auth = auth

    def set(self, on: bool, toggle_after: float | None = None) -> None:
        url = f"http://{self.host}/rpc/Switch.Set"
        payload: dict[str, Any] = {"id": self.switch_id, "on": on}
        if on and toggle_after and toggle_after > 0:
            payload["toggle_after"] = int(toggle_after)
        log.info("shelly: switch %s host=%s", payload, self.host)
        with httpx.Client(timeout=5.0, auth=self.auth) as client:
            response = client.post(url, json=payload)
            if response.status_code == 404:
                params: dict[str, Any] = {"turn": "on" if on else "off"}
                if on and toggle_after and toggle_after > 0:
                    params["timer"] = int(toggle_after)
                response = client.get(
                    f"http://{self.host}/relay/{self.switch_id}",
                    params=params,
                )
            response.raise_for_status()

    def status(self) -> dict[str, Any]:
        with httpx.Client(timeout=5.0, auth=self.auth) as client:
            response = client.get(
                f"http://{self.host}/rpc/Switch.GetStatus",
                params={"id": self.switch_id},
            )
            if response.status_code == 404:
                response = client.get(f"http://{self.host}/relay/{self.switch_id}")
            response.raise_for_status()
            body = response.json()
        output = body.get("output")
        if output is None:
            output = body.get("ison")
        return {
            "on": bool(output),
            "power_w": body.get("apower"),
            "host": self.host,
        }


class ShellyBackend:
    """Pulse the pump plug. Auto-off lives on the Shelly, not here."""

    def _plug(self) -> ShellyPlug:
        return ShellyPlug(_host(), _switch_id(), _auth())

    def water(self, seconds: float) -> None:
        del seconds
        self._plug().set(True, toggle_after=_cycle_seconds())

    def turn_on(self) -> None:
        self._plug().set(True)

    def stop(self) -> None:
        self._plug().set(False)

    def status(self) -> dict[str, Any]:
        return self._plug().status()


backend = ShellyBackend()
