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
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

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



_HISTORY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

_HOURLY_VARS = {
    "temperature_c": "temperature_2m",
    "relative_humidity_pct": "relative_humidity_2m",
    "cloud_cover_pct": "cloud_cover",
    "precipitation_mm": "precipitation",
    "total_radiation_w_m2": "shortwave_radiation",
    "uv_index": "uv_index",
    "wind_speed_kmh": "wind_speed_10m",
    "wind_gusts_kmh": "wind_gusts_10m",
    "pressure_hpa": "surface_pressure",
    "dew_point_c": "dew_point_2m",
    "vpd_kpa": "vapour_pressure_deficit",
    "et0_mm": "et0_fao_evapotranspiration",
    "soil_temperature_c": "soil_temperature_0cm",
    "soil_moisture_m3_m3": "soil_moisture_0_to_1cm",
}
_DAILY_VARS = {
    "temperature_c": "temperature_2m_mean",
    "relative_humidity_pct": "relative_humidity_2m_mean",
    "cloud_cover_pct": "cloud_cover_mean",
    "precipitation_mm": "precipitation_sum",
    "total_radiation_w_m2": "shortwave_radiation_sum",  # MJ/m²/day, converted below
    "uv_index": "uv_index_max",
    "wind_speed_kmh": "wind_speed_10m_max",
    "wind_gusts_kmh": "wind_gusts_10m_max",
    "pressure_hpa": "surface_pressure_mean",
    "dew_point_c": "dew_point_2m_mean",
    "vpd_kpa": "vapour_pressure_deficit_max",
    "et0_mm": "et0_fao_evapotranspiration",
    "soil_temperature_c": "soil_temperature_0_to_7cm_mean",
    "soil_moisture_m3_m3": "soil_moisture_0_to_7cm_mean",
}
_ARCHIVE_DAILY_VARS = {k: v for k, v in _DAILY_VARS.items() if k != "uv_index"}
HISTORY_RANGES = ("day", "month", "year")


def _open_meteo(url: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_meteorology_history(
    range_: str = "day",
    location: LocationConfig | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Weather + sun series for the Overview charts.

    day   → hourly, yesterday through tomorrow (forecast API)
    month → daily, last 31 days + 2-day forecast (forecast API)
    year  → daily, last 365 days (archive API, ~5 days behind)

    ``forecast`` (internal, used by ``get_forecast``) is hourly from today, 3 days ahead.
    """
    if range_ not in (*HISTORY_RANGES, "forecast"):
        raise ValueError(f"range must be one of {HISTORY_RANGES}")
    hourly = range_ in ("day", "forecast")
    cached = _HISTORY_CACHE.get(range_)
    now = time.time()
    if cached and (now - cached[0]) < _WEATHER_CACHE_TTL_S:
        return cached[1]

    loc = location or load_location_config()
    base = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "timezone": loc.timezone,
        "timeformat": "unixtime",
    }
    if range_ == "year":
        today = datetime.now(timezone.utc).date()
        variables = _ARCHIVE_DAILY_VARS
        data = _open_meteo(
            "https://archive-api.open-meteo.com/v1/archive",
            {
                **base,
                "start_date": (today - timedelta(days=365)).isoformat(),
                "end_date": today.isoformat(),
                "daily": [*variables.values(), "sunrise", "sunset"],
            },
            timeout_s,
        )
    else:
        variables = _HOURLY_VARS if hourly else _DAILY_VARS
        params = {
            **base,
            "past_days": {"day": 1, "forecast": 0}.get(range_, 31),
            "forecast_days": 3 if range_ == "forecast" else 2,
        }
        if hourly:
            params["hourly"] = list(variables.values())
            params["daily"] = ["sunrise", "sunset"]
        else:
            params["daily"] = [*variables.values(), "sunrise", "sunset"]
        data = _open_meteo("https://api.open-meteo.com/v1/forecast", params, timeout_s)

    daily = data.get("daily") or {}
    days = [
        {"t": day, "sunrise": rise, "sunset": set_}
        for day, rise, set_ in zip(
            daily.get("time") or [], daily.get("sunrise") or [], daily.get("sunset") or []
        )
        if rise is not None and set_ is not None
    ]

    if hourly:
        series = data.get("hourly") or {}
        points = []
        for i, ts in enumerate(series.get("time") or []):
            sun = calculate_solar_position(
                loc.latitude, loc.longitude, datetime.fromtimestamp(ts, timezone.utc)
            )
            point = {key: series[var][i] for key, var in variables.items()}
            points.append(
                {
                    "t": ts,
                    **point,
                    "altitude_deg": sun["altitude_deg"],
                    "azimuth_deg": sun["azimuth_deg"],
                    "clear_sky_irradiance_w_m2": sun["clear_sky_irradiance_w_m2"],
                }
            )
    else:
        points = []
        for i, ts in enumerate(daily.get("time") or []):
            point = {key: daily[var][i] for key, var in variables.items()}
            if point.get("total_radiation_w_m2") is not None:
                point["total_radiation_w_m2"] = round(
                    point["total_radiation_w_m2"] * 1e6 / 86400, 1
                )
            point.setdefault("uv_index", None)
            # Sun geometry per day: peak altitude, mean clear-sky irradiance,
            # and the azimuth the sun rises at (it swings over the year).
            hours = [
                calculate_solar_position(
                    loc.latitude, loc.longitude, datetime.fromtimestamp(ts + h * 3600, timezone.utc)
                )
                for h in range(24)
            ]
            sunrise = daily["sunrise"][i]
            rise = (
                calculate_solar_position(
                    loc.latitude, loc.longitude, datetime.fromtimestamp(sunrise, timezone.utc)
                )
                if sunrise is not None
                else None
            )
            points.append(
                {
                    "t": ts,
                    **point,
                    "altitude_deg": max(h["altitude_deg"] for h in hours),
                    "azimuth_deg": rise["azimuth_deg"] if rise else None,
                    "clear_sky_irradiance_w_m2": round(
                        sum(h["clear_sky_irradiance_w_m2"] for h in hours) / 24, 1
                    ),
                }
            )

    result = {
        "range": range_,
        "resolution": "hourly" if hourly else "daily",
        "timezone": loc.timezone,
        "hourly": points,
        "daily": days,
    }
    _HISTORY_CACHE[range_] = (now, result)
    return result


def get_forecast(
    hours: int = 48,
    step_hours: int = 3,
    location: LocationConfig | None = None,
) -> dict[str, Any]:
    """Upcoming weather for the assistant and the inline chat forecast card.

    ``hourly`` is downsampled to ``step_hours`` so the model gets a compact table;
    ``daily`` summarises each calendar day from the full hourly series.
    """
    data = fetch_meteorology_history("forecast", location=location)
    now = time.time()
    upcoming = [p for p in data["hourly"] if p["t"] >= now - 3600][: max(1, hours)]
    step = max(1, step_hours)

    def block(points: list[dict[str, Any]]) -> dict[str, Any]:
        """One step_hours block: rain is summed, everything else averaged."""

        def avg(key: str) -> float | None:
            values = [p[key] for p in points if p.get(key) is not None]
            return round(sum(values) / len(values), 1) if values else None

        rain = [p["precipitation_mm"] for p in points if p.get("precipitation_mm") is not None]
        return {
            "t": points[0]["t"],
            "temperature_c": avg("temperature_c"),
            "relative_humidity_pct": avg("relative_humidity_pct"),
            "cloud_cover_pct": avg("cloud_cover_pct"),
            "precipitation_mm": round(sum(rain), 1) if rain else None,
            "total_radiation_w_m2": avg("total_radiation_w_m2"),
        }

    blocks = [block(upcoming[i : i + step]) for i in range(0, len(upcoming), step)]

    by_day: dict[str, list[dict[str, Any]]] = {}
    for point in upcoming:
        day = datetime.fromtimestamp(point["t"], ZoneInfo(data["timezone"])).date().isoformat()
        by_day.setdefault(day, []).append(point)

    def stat(points: list[dict[str, Any]], key: str, fn: Any) -> float | None:
        values = [p[key] for p in points if p.get(key) is not None]
        return round(fn(values), 1) if values else None

    daily = [
        {
            "date": day,
            "temp_min_c": stat(points, "temperature_c", min),
            "temp_max_c": stat(points, "temperature_c", max),
            "rain_mm": stat(points, "precipitation_mm", sum),
            "cloud_mean_pct": stat(points, "cloud_cover_pct", lambda v: sum(v) / len(v)),
            "solar_peak_w_m2": stat(points, "total_radiation_w_m2", max),
        }
        for day, points in by_day.items()
    ]
    wet = next((p for p in upcoming if (p.get("precipitation_mm") or 0) >= 0.2), None)
    return {
        "timezone": data["timezone"],
        "generated_at": int(now),
        "step_hours": step_hours,
        "hourly": blocks,
        "daily": daily,
        "next_rain_at": wet["t"] if wet else None,
        "rain_total_mm": stat(upcoming, "precipitation_mm", sum),
        "sun_times": data["daily"],
    }


_HISTORY_TITLES = {
    "day": "Weather history · yesterday to tomorrow (hourly)",
    "month": "Weather history · last month (daily)",
    "year": "Weather history · last year (daily)",
}


def shape_weather_history(data: dict[str, Any], range_: str) -> dict[str, Any]:
    """Shape ``fetch_meteorology_history`` output for the chat chart card.

    Reuses the forecast card shape (`hourly` points); daily ranges plot on a
    day/month axis. No `next_rain_at` — this is the past, not a forecast.
    """
    daily = data.get("resolution") == "daily"
    return {
        "timezone": data.get("timezone"),
        "range": range_,
        "resolution": data.get("resolution"),
        "title": _HISTORY_TITLES.get(range_, "Weather history"),
        "axis": ("month" if range_ == "year" else "day") if daily else "hour",
        "hourly": data.get("hourly", []),
        "daily": [],
    }


def get_weather_history(
    range_: str = "month", location: LocationConfig | None = None
) -> dict[str, Any]:
    """Past weather (day=hourly, month/year=daily) for assistant + chat card."""
    return shape_weather_history(
        fetch_meteorology_history(range_, location=location), range_
    )
