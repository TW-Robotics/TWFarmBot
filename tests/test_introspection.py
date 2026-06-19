"""Tests for the introspection tools and the in-memory provider."""

from __future__ import annotations

from planning_service.introspection import (
    InMemorySystemStateProvider,
    build_introspection_tools,
)


def test_inmemory_provider_returns_configured_data() -> None:
    p = InMemorySystemStateProvider(
        position={"x": 100, "y": 200, "z": 0},
        pins={13: {"value": 1, "mode": "digital"}},
        garden={"spatial": {"zones": []}, "watering": {"pins": {}}},
    )
    assert p.get_position() == {"x": 100, "y": 200, "z": 0}
    assert p.read_pin(13, "digital") == {"value": 1, "mode": "digital"}
    assert p.read_pin(99, "digital") == {"value": None, "mode": "digital"}
    assert p.get_garden() == {"spatial": {"zones": []}, "watering": {"pins": {}}}


def test_introspection_tools_are_built() -> None:
    p = InMemorySystemStateProvider(
        position={"x": 0, "y": 0, "z": 0},
        health={"status": "ok", "farmbot": "connected"},
        garden={
            "spatial": {
                "zones": [
                    {
                        "id": "tomato",
                        "name": "Tomato Zone",
                        "kind": "zone",
                        "bounds": {"x": 1100, "y": 100, "width": 800, "height": 300},
                    }
                ]
            },
        },
    )
    tools = build_introspection_tools(p)
    names = {t.name for t in tools}
    assert "get_position" in names
    assert "list_zones" in names
    assert "list_endpoints" in names
    assert "read_pin" in names
    assert "analyze_image" in names


def test_list_zones_computes_centre() -> None:
    p = InMemorySystemStateProvider(
        garden={
            "spatial": {
                "zones": [
                    {
                        "id": "tomato",
                        "name": "Tomato Zone",
                        "kind": "zone",
                        "bounds": {"x": 1100, "y": 100, "width": 800, "height": 300},
                    }
                ]
            },
        }
    )
    tools = {t.name: t for t in build_introspection_tools(p)}
    result = tools["list_zones"].invoke({})
    assert result["count"] == 1
    zone = result["zones"][0]
    assert zone["name"] == "Tomato Zone"
    assert zone["center"] == (1500, 250)  # x + w/2, y + h/2


def test_introspection_tool_swallows_provider_errors() -> None:
    class Broken(InMemorySystemStateProvider):
        def get_position(self):  # type: ignore[override]
            raise RuntimeError("nope")

    tools = {t.name: t for t in build_introspection_tools(Broken())}
    result = tools["get_position"].invoke({})
    assert "error" in result
    assert "nope" in result["error"]
