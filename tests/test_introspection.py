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
    assert "get_garden" in names
    assert "find_locations" in names
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
        def get_position(self):
            raise RuntimeError("nope")

    tools = {t.name: t for t in build_introspection_tools(Broken())}
    result = tools["get_position"].invoke({})
    assert "error" in result
    assert "nope" in result["error"]


def test_find_locations_resolves_saved_named_position() -> None:
    """'carrot' is a Motion-page preset, not a bed-map predicate."""
    from planning_service.introspection import InMemorySystemStateProvider, build_introspection_tools

    provider = InMemorySystemStateProvider(
        positions=[{"label": "Carrot", "group": "Travel", "x": 861, "y": 590, "z": -300}]
    )
    tools = {t.name: t for t in build_introspection_tools(provider)}
    out = tools["find_locations"].invoke({"predicate": "the carrot"})
    assert out["named_position"] == "Carrot"
    assert "goto_named" in out["next"]


def test_think_tool_is_a_noop_scratchpad() -> None:
    from planning_service.introspection import InMemorySystemStateProvider, build_introspection_tools

    tools = {t.name: t for t in build_introspection_tools(InMemorySystemStateProvider())}
    out = tools["think"].invoke({"thought": "carrot is a saved preset"})
    assert out["status"] == "ok" and "tool" in out["next"]
