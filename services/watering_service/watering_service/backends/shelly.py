"""Shelly plug backend: one pump for every bed sprayer.

The plug is already configured to cut power after 1 minute, so we only turn
it on (and off on ESTOP). We do not sleep for a software duration.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx

from twfarmbot_core.config import load_yaml_config

log = logging.getLogger(__name__)


def _cfg() -> dict[str, Any]:
    watering = load_yaml_config().get("watering", {}) or {}
    return watering.get("shelly", {}) or {} if isinstance(watering, dict) else {}


def _host() -> str:
    raw = (os.getenv("SHELLY_HOST") or str(_cfg().get("host") or "")).strip()
    if not raw:
        raise ValueError(
            "Shelly watering needs SHELLY_HOST or watering.shelly.host in config"
        )
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = parsed.netloc or parsed.path
    if not host:
        raise ValueError(f"invalid Shelly host: {raw!r}")
    return host.split("/")[0]


def _cycle_seconds() -> float:
    watering = load_yaml_config().get("watering", {}) or {}
    return float(os.getenv("FARMBOT_WATER_SECONDS", watering.get("cycle_seconds", 60)))


def _switch_id() -> int:
    return int(os.getenv("SHELLY_SWITCH_ID", _cfg().get("switch_id", 0)))


def _auth() -> tuple[str, str] | None:
    user = os.getenv("SHELLY_USER") or str(_cfg().get("user") or "")
    password = os.getenv("SHELLY_PASSWORD") or str(_cfg().get("password") or "")
    if user or password:
        return (user, password)
    return None


class ShellyBackend:
    """Pulse the pump plug. Auto-off lives on the Shelly, not here."""

    def water(self, seconds: float) -> None:
        del seconds
        self._set(True, toggle_after=_cycle_seconds())

    def turn_on(self) -> None:
        self._set(True)

    def stop(self) -> None:
        self._set(False)

    def status(self) -> dict[str, Any]:
        host = _host()
        switch_id = _switch_id()
        auth = _auth()
        with httpx.Client(timeout=5.0, auth=auth) as client:
            response = client.get(
                f"http://{host}/rpc/Switch.GetStatus",
                params={"id": switch_id},
            )
            if response.status_code == 404:
                response = client.get(f"http://{host}/relay/{switch_id}")
            response.raise_for_status()
            body = response.json()
        output = body.get("output")
        if output is None:
            output = body.get("ison")
        return {
            "on": bool(output),
            "power_w": body.get("apower"),
            "host": host,
        }

    def _set(self, on: bool, toggle_after: float | None = None) -> None:
        host = _host()
        switch_id = _switch_id()
        auth = _auth()
        url = f"http://{host}/rpc/Switch.Set"
        payload: dict[str, Any] = {"id": switch_id, "on": on}
        if on and toggle_after and toggle_after > 0:
            payload["toggle_after"] = int(toggle_after)
        log.info("shelly: switch %s host=%s", payload, host)
        with httpx.Client(timeout=5.0, auth=auth) as client:
            response = client.post(url, json=payload)
            if response.status_code == 404:
                params: dict[str, Any] = {"turn": "on" if on else "off"}
                if on and toggle_after and toggle_after > 0:
                    params["timer"] = int(toggle_after)
                response = client.get(
                    f"http://{host}/relay/{switch_id}",
                    params=params,
                )
            response.raise_for_status()


backend = ShellyBackend()
