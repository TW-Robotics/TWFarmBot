"""Tests for meteorology, solar position tracking, and agronomic advice."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from planning_service import (
    calculate_solar_position,
    get_meteorology,
    interpret_agronomic_conditions,
)
from planning_service.harness import ContextBuilder, ToolRegistry
from planning_service.introspection import InMemorySystemStateProvider, build_introspection_tools
from twfarmbot_api_server.app import create_app
from twfarmbot_core.actions import ActionRegistry


def test_solar_position_noon_vs_midnight() -> None:
    """Test solar altitude and day/night detection at Vienna coordinates."""
    lat, lon = 48.2394, 16.3776

    # Summer solstice solar noon (approx 11:00 UTC = 13:00 CEST)
    noon_dt = datetime(2026, 6, 21, 11, 0, 0, tzinfo=timezone.utc)
    solar_noon = calculate_solar_position(lat, lon, noon_dt)

    assert solar_noon["is_daylight"] is True
    assert solar_noon["sun_phase"] == "day"
    assert solar_noon["altitude_deg"] > 60.0
    assert 160.0 < solar_noon["azimuth_deg"] < 200.0  # South
    assert solar_noon["clear_sky_irradiance_w_m2"] > 700.0

    # Midnight (22:00 UTC)
    midnight_dt = datetime(2026, 6, 21, 22, 0, 0, tzinfo=timezone.utc)
    solar_midnight = calculate_solar_position(lat, lon, midnight_dt)

    assert solar_midnight["is_daylight"] is False
    assert solar_midnight["sun_phase"] == "night"
    assert solar_midnight["altitude_deg"] < -15.0
    assert solar_midnight["clear_sky_irradiance_w_m2"] == 0.0


def test_agronomic_advice_lighting_and_watering() -> None:
    """Test lighting quality and watering suitability rules."""
    # 1. Harsh direct sunlight
    harsh_solar = {"altitude_deg": 55.0, "clear_sky_irradiance_w_m2": 800.0}
    harsh_weather = {
        "cloud_cover_pct": 10.0,
        "total_radiation_w_m2": 780.0,
        "temperature_c": 30.0,
        "precipitation_mm": 0.0,
    }
    advice1 = interpret_agronomic_conditions(harsh_solar, harsh_weather)
    assert advice1["camera_lighting_quality"] == "harsh_direct_sun"
    assert "glare" in advice1["camera_lighting_advice"].lower()
    assert advice1["watering_suitability"] == "midday_heat_warning"
    assert advice1["sun_exposure_level"] == "intense"

    # 2. Overcast diffuse conditions (ideal for NDRE)
    overcast_solar = {"altitude_deg": 35.0, "clear_sky_irradiance_w_m2": 450.0}
    overcast_weather = {
        "cloud_cover_pct": 80.0,
        "total_radiation_w_m2": 250.0,
        "temperature_c": 21.0,
        "precipitation_mm": 0.0,
    }
    advice2 = interpret_agronomic_conditions(overcast_solar, overcast_weather)
    assert advice2["camera_lighting_quality"] == "optimal_diffuse"
    assert advice2["watering_suitability"] == "favorable"

    # 3. Rain forecasted
    rain_weather = {
        "cloud_cover_pct": 95.0,
        "total_radiation_w_m2": 100.0,
        "temperature_c": 16.0,
        "precipitation_mm": 4.5,
    }
    advice3 = interpret_agronomic_conditions(overcast_solar, rain_weather)
    assert advice3["watering_suitability"] == "rain_forecasted_or_detected"
    assert "skip" in advice3["watering_advice"].lower()


def test_get_meteorology_payload() -> None:
    """get_meteorology returns full schema with solar, weather, and advice."""
    met = get_meteorology()
    assert "location" in met
    assert "sun" in met
    assert "weather" in met
    assert "agronomic_advice" in met
    assert met["location"]["latitude"] == 48.2394
    assert "altitude_deg" in met["sun"]
    assert "temperature_c" in met["weather"]


def test_introspection_tool_get_meteorology() -> None:
    """The get_meteorology tool executes cleanly via SystemStateProvider."""
    provider = InMemorySystemStateProvider(
        meteorology={
            "location": {"latitude": 48.2394, "longitude": 16.3776},
            "sun": {"altitude_deg": 42.0, "is_daylight": True},
            "weather": {"temperature_c": 23.0, "total_radiation_w_m2": 520.0},
            "agronomic_advice": {"camera_lighting_quality": "acceptable"},
        }
    )
    tools = {t.name: t for t in build_introspection_tools(provider)}
    assert "get_meteorology" in tools
    res = tools["get_meteorology"].invoke({})
    assert res["sun"]["altitude_deg"] == 42.0
    assert res["weather"]["temperature_c"] == 23.0


def test_context_builder_does_not_inject_meteorology_context() -> None:
    """Weather is fetched via get_meteorology/get_forecast, never put in the prompt."""
    reg = ActionRegistry()
    tool_reg = ToolRegistry(reg, system_state=None)
    cb = ContextBuilder(tool_reg)
    prompt = cb.chat_system_prompt()
    assert "Environment & Solar Context" not in prompt


def test_api_server_get_meteorology() -> None:
    """GET /meteorology returns 200 with complete telemetry."""
    reg = ActionRegistry()
    app = create_app(registry=reg)
    client = TestClient(app)

    res = client.get("/meteorology")
    assert res.status_code == 200
    body = res.json()
    assert "sun" in body
    assert "weather" in body
    assert "agronomic_advice" in body


def _hourly_points(start: float, rain_hour: int) -> list[dict]:
    return [
        {
            "t": start + h * 3600,
            "temperature_c": 10.0 + h,
            "relative_humidity_pct": 80,
            "cloud_cover_pct": 50,
            "precipitation_mm": 1.5 if h == rain_hour else 0.0,
            "total_radiation_w_m2": 100.0,
        }
        for h in range(12)
    ]


def test_get_forecast_blocks_keep_rain(monkeypatch) -> None:
    """3-hour blocks sum rain (not sample it) and report when rain starts."""
    import time

    from planning_service import meteorology

    start = time.time() + 3600
    monkeypatch.setattr(
        meteorology,
        "fetch_meteorology_history",
        lambda *args, **kwargs: {
            "timezone": "Europe/Vienna",
            "hourly": _hourly_points(start, rain_hour=4),
            "daily": [],
        },
    )
    forecast = meteorology.get_forecast(hours=12, step_hours=3)
    assert len(forecast["hourly"]) == 4
    # Hour 4 falls in the second block; a plain every-3rd-hour sample would miss it.
    assert [b["precipitation_mm"] for b in forecast["hourly"]] == [0.0, 1.5, 0.0, 0.0]
    assert forecast["rain_total_mm"] == 1.5
    assert forecast["next_rain_at"] == start + 4 * 3600
    assert sum(day["rain_mm"] for day in forecast["daily"]) == 1.5


def test_introspection_tool_get_forecast_display_flag() -> None:
    """get_forecast passes the display flag through so the chat can skip the chart."""
    provider = InMemorySystemStateProvider(forecast={"hourly": [], "daily": []})
    tools = {t.name: t for t in build_introspection_tools(provider)}
    assert tools["get_forecast"].invoke({})["display"] is True
    assert tools["get_forecast"].invoke({"display": False})["display"] is False
