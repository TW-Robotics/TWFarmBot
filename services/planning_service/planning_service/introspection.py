"""Read-only system-introspection tools for the planner.

These give the LLM the ability to look at the live system state and
the configured world model *during* a planning call, not just at the
prompt-time snapshot. The model can chain them: e.g. "where am I?"
→ ``get_position`` → "move to the tomatoes" → ``list_zones`` →
``move(x=center, ...)``.

All tools are read-only. The action tools in :mod:`tools` (move, water,
find_home, …) remain the only way to mutate the system.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ── Tool argument schemas ────────────────────────────────────────────────


class _NoArgs(BaseModel):
    pass


class ListEndpointsArgs(BaseModel):
    prefix: str = Field(
        default="",
        description=(
            "Optional filter: only include paths starting with this string. "
            "E.g. 'GET' to list only GET endpoints."
        ),
    )


# ── Provider protocol ────────────────────────────────────────────────────


class SystemStateProvider:
    """Pluggable source of live system data.

    The API server injects a provider that knows how to talk to the
    FarmBot gateway and the spatial/bed config. Tests inject a stub.
    Each method is allowed to raise — the tool wrapper turns the
    exception into a structured error string the model can reason about.
    """

    def list_endpoints(self) -> list[dict[str, str]]:
        raise NotImplementedError

    def get_health(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_position(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_status(self, path: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def get_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError

    def read_pin(self, pin: int, mode: str = "digital") -> dict[str, Any]:
        raise NotImplementedError

    def get_garden(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_pins(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_images(self, limit: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError


# ── Tool builder ────────────────────────────────────────────────────────


def build_introspection_tools(
    provider: SystemStateProvider,
) -> list[BaseTool]:
    """Build LangChain read-only tools that query the live system."""

    def _safe(name: str, fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(**kwargs)
        except Exception as err:  # noqa: BLE001
            log.warning("introspection tool %s failed: %s", name, err)
            return {"error": f"{type(err).__name__}: {err}", "tool": name}

    @tool(args_schema=ListEndpointsArgs)
    def list_endpoints(prefix: str = "") -> dict[str, Any]:
        """List every HTTP endpoint exposed by the FarmBot API.

        Returns a list of `{"method": "GET", "path": "/position", "summary": "..."}`.
        Useful when the planner wants to know what data it can pull or
        what actions it can call.
        """
        endpoints = provider.list_endpoints()
        if prefix:
            endpoints = [
                e for e in endpoints
                if e.get("method", "").startswith(prefix.upper())
                or e.get("path", "").startswith(prefix)
            ]
        return {"endpoints": endpoints, "count": len(endpoints)}

    @tool(args_schema=_NoArgs)
    def get_health() -> dict[str, Any]:
        """Return the API server's health snapshot.

        Includes `farmbot` connection status, registered action kinds,
        and overall status. Use this to check whether the FarmBot is
        connected before planning a move.
        """
        return _safe("get_health", provider.get_health)

    @tool(args_schema=_NoArgs)
    def get_position() -> dict[str, Any]:
        """Return the FarmBot's last known gantry position (x, y, z) in mm.

        Use this to know where the gantry currently is before issuing a
        move. Returns an empty dict if no position is known yet.
        """
        return _safe("get_position", provider.get_position)

    @tool(args_schema=_NoArgs)
    def get_status() -> dict[str, Any]:
        """Return the FarmBot's full status tree (location_data, pins, sensors, ...).

        Heavy call — use only when you need detailed state, not for
        simple position checks.
        """
        return _safe("get_status", lambda: provider.get_status())

    @tool(args_schema=_NoArgs)
    def get_messages(limit: int = 20) -> dict[str, Any]:
        """Return the most recent MQTT messages from the FarmBot.

        `limit` caps how many are returned (default 20). Useful to see
        what the robot has been doing lately.
        """
        return _safe("get_messages", lambda: provider.get_messages(limit=limit))

    @tool
    def read_pin(pin: int, mode: str = "digital") -> dict[str, Any]:
        """Read a GPIO pin value from the FarmBot.

        `pin` is the GPIO number. `mode` is 'digital' (0/1) or 'analog'
        (raw ADC value). Returns `{"value": ..., "mode": ...}`.
        """
        return _safe("read_pin", lambda: provider.read_pin(pin, mode))

    @tool(args_schema=_NoArgs)
    def get_garden() -> dict[str, Any]:
        """Return the configured world model (bounds, zones, entities, camera pose).

        Use this to look up zone names, bed_ids, and coordinates. This
        is the same data the planner sees in the world-model context,
        but queried live so it reflects the current config.
        """
        return _safe("get_garden", provider.get_garden)

    @tool(args_schema=_NoArgs)
    def list_beds() -> dict[str, Any]:
        """List all configured beds and the zones they water.

        Returns a mapping `bed_id -> {"pin": <int>, "zones": [<names>]}`
        so the planner can map zone names to watering actions.
        """
        garden = _safe("get_garden", provider.get_garden)
        if "error" in garden:
            return garden
        # Build bed_id -> zones via the spatial model + watering.pins.
        bed_to_zones: dict[str, dict[str, Any]] = {}
        pins_block = garden.get("watering", {}).get("pins", {})
        zones = garden.get("spatial", {}).get("zones", []) or []
        pin_to_zones: dict[int, list[str]] = {}
        for zone in zones:
            meta = zone.get("metadata") or {}
            pin = meta.get("valve_pin")
            name = zone.get("name") or zone.get("id")
            if pin is not None and name:
                pin_to_zones.setdefault(int(pin), []).append(str(name))
        for bed, pin in pins_block.items():
            bed_to_zones[bed] = {
                "pin": int(pin),
                "zones": pin_to_zones.get(int(pin), []),
            }
        return {"beds": bed_to_zones, "count": len(bed_to_zones)}

    @tool(args_schema=_NoArgs)
    def list_zones() -> dict[str, Any]:
        """List all configured zones with their bounds and centre coordinates.

        Each entry: `{"name", "id", "x", "y", "width", "height",
        "center": (cx, cy), "bed_id"}`. Use this to look up a zone by
        name and get its centre for a `move` action.
        """
        garden = _safe("get_garden", provider.get_garden)
        if "error" in garden:
            return garden
        bed_to_zones: dict[str, list[str]] = {}
        zones_block = garden.get("spatial", {}).get("zones", []) or []
        pin_to_bed: dict[int, str] = {
            int(pin): bed
            for bed, pin in (garden.get("watering", {}).get("pins", {}) or {}).items()
        }
        # Build name -> bed via valve_pin
        name_to_bed: dict[str, str] = {}
        for zone in zones_block:
            meta = zone.get("metadata") or {}
            pin = meta.get("valve_pin")
            name = zone.get("name") or zone.get("id")
            if pin is not None and name:
                bed = pin_to_bed.get(int(pin))
                if bed:
                    name_to_bed[str(name)] = bed
                    name_to_bed[str(zone.get("id"))] = bed
        out: list[dict[str, Any]] = []
        for zone in zones_block:
            b = zone.get("bounds", {})
            x = b.get("x", 0)
            y = b.get("y", 0)
            w = b.get("width", 0)
            h = b.get("height", 0)
            name = zone.get("name") or zone.get("id")
            out.append({
                "name": name,
                "id": zone.get("id"),
                "kind": zone.get("kind"),
                "x": x, "y": y, "width": w, "height": h,
                "center": (round(x + w / 2), round(y + h / 2)),
                "bed_id": name_to_bed.get(str(name)),
            })
        return {"zones": out, "count": len(out)}

    @tool(args_schema=_NoArgs)
    def list_endpoints_action() -> dict[str, Any]:
        """List all registered action kinds (move, water, find_home, ...).

        Returns `{"actions": [{"kind": ..., "params_schema": ...}, ...]}`.
        Use this to confirm what actions are available before planning.
        """
        # Kept as a separate tool from list_endpoints for clarity.
        return list_endpoints.invoke({})  # type: ignore[attr-defined]

    @tool(args_schema=_NoArgs)
    def get_pins() -> dict[str, Any]:
        """List all named GPIO pins (label, number, mode, kind, group)."""
        return _safe("get_pins", lambda: {"pins": provider.get_pins()})

    @tool(args_schema=_NoArgs)
    def get_positions() -> dict[str, Any]:
        """List the named gantry position presets (Home, Bed 1, ...)."""
        return _safe("get_positions", lambda: {"positions": provider.get_positions()})

    @tool(args_schema=_NoArgs)
    def get_images(limit: int = 5) -> dict[str, Any]:
        """Return the most recent camera images uploaded by the FarmBot."""
        return _safe("get_images", lambda: {"images": provider.get_images(limit=limit)})

    return [
        list_endpoints,
        get_health,
        get_position,
        get_status,
        get_messages,
        read_pin,
        get_garden,
        list_beds,
        list_zones,
        get_pins,
        get_positions,
        get_images,
    ]


# ── Concrete provider for the live API server ──────────────────────────


class HttpSystemStateProvider(SystemStateProvider):
    """Calls the api_server over HTTP. Used in production."""

    def __init__(self, base_url: str) -> None:
        import httpx
        self._client = httpx.Client(base_url=base_url, timeout=10.0)
        self._endpoints_cache: list[dict[str, str]] | None = None

    def list_endpoints(self) -> list[dict[str, str]]:
        if self._endpoints_cache is not None:
            return self._endpoints_cache
        # We don't have a /endpoints endpoint; enumerate from the
        # documented surface. This is best-effort and cached.
        self._endpoints_cache = [
            {"method": "GET", "path": "/health", "summary": "API + FarmBot health"},
            {"method": "GET", "path": "/position", "summary": "Last gantry (x,y,z)"},
            {"method": "GET", "path": "/status", "summary": "Full status tree"},
            {"method": "GET", "path": "/pin/{pin}", "summary": "Read a GPIO pin"},
            {"method": "GET", "path": "/messages", "summary": "Recent MQTT messages"},
            {"method": "GET", "path": "/images", "summary": "Recent camera images"},
            {"method": "GET", "path": "/pins", "summary": "Named GPIO pins"},
            {"method": "GET", "path": "/positions", "summary": "Named gantry presets"},
            {"method": "GET", "path": "/garden", "summary": "World model snapshot"},
            {"method": "POST", "path": "/actions", "summary": "Dispatch an Action"},
            {"method": "POST", "path": "/plan", "summary": "LLM plan (preview or execute)"},
        ]
        return self._endpoints_cache

    def _get(self, path: str) -> dict[str, Any]:
        r = self._client.get(path)
        r.raise_for_status()
        return r.json() if r.content else {}

    def get_health(self) -> dict[str, Any]:
        return self._get("/health")

    def get_position(self) -> dict[str, Any]:
        return self._get("/position")

    def get_status(self, path: str | None = None) -> dict[str, Any]:
        return self._get(f"/status?path={path}" if path else "/status")

    def get_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        body = self._get("/messages")
        raw = body.get("last_messages", []) if isinstance(body, dict) else []
        return list(raw[-limit:])

    def read_pin(self, pin: int, mode: str = "digital") -> dict[str, Any]:
        return self._get(f"/pin/{pin}?mode={mode}")

    def get_garden(self) -> dict[str, Any]:
        return self._get("/garden")

    def get_pins(self) -> list[dict[str, Any]]:
        body = self._get("/pins")
        return list(body.get("pins", [])) if isinstance(body, dict) else []

    def get_positions(self) -> list[dict[str, Any]]:
        body = self._get("/positions")
        return list(body.get("positions", [])) if isinstance(body, dict) else []

    def get_images(self, limit: int = 5) -> list[dict[str, Any]]:
        body = self._get(f"/images?limit={limit}")
        return list(body.get("images", [])) if isinstance(body, dict) else []


class InMemorySystemStateProvider(SystemStateProvider):
    """Test/dry-run provider. Inject whatever dicts you want."""

    def __init__(self, **data: Any) -> None:
        self._data = data

    def list_endpoints(self) -> list[dict[str, str]]:
        return list(self._data.get("endpoints", []))

    def get_health(self) -> dict[str, Any]:
        return dict(self._data.get("health", {"status": "ok"}))

    def get_position(self) -> dict[str, Any]:
        return dict(self._data.get("position", {"x": 0, "y": 0, "z": 0}))

    def get_status(self, path: str | None = None) -> dict[str, Any]:
        return dict(self._data.get("status", {}))

    def get_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self._data.get("messages", []))[:limit]

    def read_pin(self, pin: int, mode: str = "digital") -> dict[str, Any]:
        pins = dict(self._data.get("pins", {}))
        return pins.get(pin, {"value": None, "mode": mode})

    def get_garden(self) -> dict[str, Any]:
        return dict(self._data.get("garden", {}))

    def get_pins(self) -> list[dict[str, Any]]:
        return list(self._data.get("pins_list", []))

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self._data.get("positions", []))

    def get_images(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(self._data.get("images", []))[:limit]
