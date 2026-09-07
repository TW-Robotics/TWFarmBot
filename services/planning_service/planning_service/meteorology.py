"""Meteorology and solar position service for TWFarmBot.

Provides GPS-grounded solar geometry (altitude, azimuth, day/night phase,
clear-sky irradiance) and live meteorological data (cloud cover, temperature,
solar radiation, rain forecast via Open-Meteo).

Includes agronomic interpretations for camera capture lighting quality
and irrigation suitability.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from twfarmbot_core.config import load_yaml_config

log = logging.getLogger(__name__)

_WEATHER_CACHE: dict[str, Any] = {}
_WEATHER_CACHE_TIME: float = 0.0
_WEATHER_CACHE_TTL_S: float = 900.0  # 15 minutes


@dataclass(frozen=True)
class LocationConfig:
    latitude: float = 48.2394
    longitude: float = 16.3776
    elevation_m: float = 165.0
    timezone: str = "Europe/Vienna"


def load_location_config() -> LocationConfig:
    """Read geographic location from dev.yaml or environment variables."""
    cfg = load_yaml_config().get("location", {}) or {}
    lat = float(os.getenv("FARMBOT_LATITUDE", cfg.get("latitude", 48.2394)))
    lon = float(os.getenv("FARMBOT_LONGITUDE", cfg.get("longitude", 16.3776)))
    elev = float(os.getenv("FARMBOT_ELEVATION_M", cfg.get("elevation_m", 165.0)))
    tz = str(os.getenv("FARMBOT_TIMEZONE", cfg.get("timezone", "Europe/Vienna")))
    return LocationConfig(latitude=lat, longitude=lon, elevation_m=elev, timezone=tz)


def calculate_solar_position(
    lat_deg: float, lon_deg: float, dt: datetime | None = None
) -> dict[str, Any]:
    """Calculate solar altitude, azimuth, day/night status, and clear-sky irradiance.

    Uses the NOAA Solar Position Algorithm. Requires zero external network.
    """
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    time_offset = eqtime + 4.0 * lon_deg
    tst = hour * 60.0 + time_offset
    solar_time_fix = (tst / 4.0) - 180.0
    ha = math.radians(solar_time_fix)

    lat = math.radians(lat_deg)

    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    altitude = math.pi / 2.0 - zenith
    alt_deg = math.degrees(altitude)

    sin_zenith = math.sin(zenith)
    if sin_zenith < 1e-6:
        azimuth = 180.0
    else:
        cos_az = (math.sin(decl) - math.sin(lat) * cos_zenith) / (
            math.cos(lat) * sin_zenith
        )
        cos_az = max(-1.0, min(1.0, cos_az))
        az = math.degrees(math.acos(cos_az))
        azimuth = (360.0 - az) if ha > 0 else az

    if alt_deg > 0:
        air_mass = 1.0 / (
            math.sin(math.radians(alt_deg)) + 0.50572 * (alt_deg + 6.07995) ** -1.6364
        )
        irradiance = max(
            0.0, 1361.0 * (0.7 ** (air_mass**0.678)) * math.sin(math.radians(alt_deg))
        )
    else:
        irradiance = 0.0

    if alt_deg < -6.0:
        phase = "night"
    elif alt_deg <= 0.0:
        phase = "twilight"
    elif alt_deg <= 15.0:
        phase = "golden_hour"
    else:
        phase = "day"

    return {
        "altitude_deg": round(alt_deg, 2),
        "azimuth_deg": round(azimuth, 2),
        "zenith_deg": round(math.degrees(zenith), 2),
        "is_daylight": alt_deg > 0.0,
        "sun_phase": phase,
        "clear_sky_irradiance_w_m2": round(irradiance, 1),
    }


def fetch_open_meteo_weather(
    lat: float, lon: float, tz: str = "Europe/Vienna", timeout_s: float = 4.0
) -> dict[str, Any]:
    """Fetch live meteorological forecast data from Open-Meteo with local caching."""
    global _WEATHER_CACHE, _WEATHER_CACHE_TIME  # noqa: PLW0603

    now = time.time()
    if _WEATHER_CACHE and (now - _WEATHER_CACHE_TIME) < _WEATHER_CACHE_TTL_S:
        return _WEATHER_CACHE

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
            "direct_radiation",
            "diffuse_radiation",
            "uv_index",
        ],
        "daily": ["sunrise", "sunset", "uv_index_max"],
        "timezone": tz,
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            curr = data.get("current") or {}
            daily = data.get("daily") or {}

            direct_rad = float(curr.get("direct_radiation") or 0.0)
            diffuse_rad = float(curr.get("diffuse_radiation") or 0.0)
            total_rad = direct_rad + diffuse_rad

            sunrise_list = daily.get("sunrise") or []
            sunset_list = daily.get("sunset") or []
            uv_max_list = daily.get("uv_index_max") or []

            weather = {
                "temperature_c": float(curr.get("temperature_2m") or 20.0),
                "relative_humidity_pct": float(curr.get("relative_humidity_2m") or 50.0),
                "precipitation_mm": float(curr.get("precipitation") or 0.0),
                "cloud_cover_pct": float(curr.get("cloud_cover") or 0.0),
                "direct_radiation_w_m2": round(direct_rad, 1),
                "diffuse_radiation_w_m2": round(diffuse_rad, 1),
                "total_radiation_w_m2": round(total_rad, 1),
                "uv_index": float(curr.get("uv_index") or 0.0),
                "max_uv_index_today": float(uv_max_list[0]) if uv_max_list else 0.0,
                "sunrise": str(sunrise_list[0]) if sunrise_list else None,
                "sunset": str(sunset_list[0]) if sunset_list else None,
                "source": "open-meteo",
            }
            _WEATHER_CACHE = weather
            _WEATHER_CACHE_TIME = now
            return weather
    except Exception as err:  # noqa: BLE001
        log.warning("Open-Meteo weather fetch failed (%s); using offline estimates", err)
        return {
            "temperature_c": 20.0,
            "relative_humidity_pct": 50.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 20.0,
            "direct_radiation_w_m2": 0.0,
            "diffuse_radiation_w_m2": 0.0,
            "total_radiation_w_m2": 0.0,
            "uv_index": 0.0,
            "max_uv_index_today": 5.0,
            "sunrise": None,
            "sunset": None,
            "source": "fallback_offline",
        }


def interpret_agronomic_conditions(
    solar: dict[str, Any], weather: dict[str, Any]
) -> dict[str, Any]:
    """Derive lighting quality and watering suitability advice for the agent."""
    alt = float(solar.get("altitude_deg") or 0.0)
    clouds = float(weather.get("cloud_cover_pct") or 0.0)
    rad = float(weather.get("total_radiation_w_m2") or solar.get("clear_sky_irradiance_w_m2") or 0.0)
    temp = float(weather.get("temperature_c") or 20.0)
    rain = float(weather.get("precipitation_mm") or 0.0)

    # 1. Camera / Spectral Lighting Quality
    if alt <= 0.0:
        camera_quality = "dark"
        camera_advice = "Night / twilight — multi-spectral camera requires artificial illumination."
    elif clouds >= 60.0 or alt <= 25.0:
        camera_quality = "optimal_diffuse"
        camera_advice = "Overcast or low-angle diffuse light — optimal for multi-spectral NDRE capture; low specular glare."
    elif alt > 45.0 and clouds < 30.0:
        camera_quality = "harsh_direct_sun"
        camera_advice = "Direct overhead sunlight — warning: leaf wax may produce specular reflections and glare distorting NDRE. Prefer early morning, dusk, or shaded viewpoint."
    else:
        camera_quality = "acceptable"
        camera_advice = "Moderate sunlight — suitable for imaging; check image histograms for blown highlights."

    # 2. Watering Suitability
    if rain >= 1.0:
        watering_suitability = "rain_forecasted_or_detected"
        watering_advice = f"Rain detected/forecasted ({rain}mm) — skip irrigation to prevent root over-saturation."
    elif temp < 3.0:
        watering_suitability = "frost_risk"
        watering_advice = f"Low temperature ({temp}°C) — do not water; risk of soil frost."
    elif (rad > 650.0 or temp > 28.0) and alt > 35.0:
        watering_suitability = "midday_heat_warning"
        watering_advice = f"High solar radiation ({rad} W/m²) and heat ({temp}°C) — rapid evaporative loss. Postpone watering to late afternoon or morning."
    else:
        watering_suitability = "favorable"
        watering_advice = "Favorable environmental conditions for irrigation."

    # 3. Sun Exposure Level
    if alt <= 0.0:
        exposure = "none"
    elif rad > 750.0:
        exposure = "intense"
    elif rad > 450.0:
        exposure = "high"
    elif rad > 150.0:
        exposure = "moderate"
    else:
        exposure = "low"

    return {
        "camera_lighting_quality": camera_quality,
        "camera_lighting_advice": camera_advice,
        "watering_suitability": watering_suitability,
        "watering_advice": watering_advice,
        "sun_exposure_level": exposure,
    }


def get_meteorology(
    dt: datetime | None = None,
    location: LocationConfig | None = None,
) -> dict[str, Any]:
    """Retrieve full solar position, meteorological readings, and agronomic advice."""
    loc = location or load_location_config()
    solar = calculate_solar_position(loc.latitude, loc.longitude, dt=dt)
    weather = fetch_open_meteo_weather(loc.latitude, loc.longitude, tz=loc.timezone)
    advice = interpret_agronomic_conditions(solar, weather)

    return {
        "timestamp": (dt or datetime.now(timezone.utc)).isoformat(),
        "location": {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "elevation_m": loc.elevation_m,
            "timezone": loc.timezone,
        },
        "sun": solar,
        "weather": weather,
        "agronomic_advice": advice,
    }
