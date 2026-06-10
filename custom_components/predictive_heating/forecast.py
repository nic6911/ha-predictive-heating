"""Forecast ingestion: weather (outdoor temp + solar proxy) and energy price.

All inputs are mapped onto the MPC horizon grid (``n`` steps of ``step_minutes``).
Everything stays inside Home Assistant -- the weather entity is location-bound and
already configured by the user, and the optional price entity is whatever the user
already runs (Nord Pool, Energi Data Service, Tibber, ...).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


def solar_proxy(cloud_coverage, uv_index) -> float:
    """Map cloud cover (%) and UV index to a 0..1 solar-gain proxy.

    The RC model's ``b_sol`` learns the true scaling, so this only needs to be
    monotonic in real irradiance. UV index encodes sun angle + season; cloud cover
    attenuates it.
    """
    uv = 0.0 if uv_index is None else max(0.0, float(uv_index))
    cloud = 0.0 if cloud_coverage is None else max(0.0, min(100.0, float(cloud_coverage)))
    clear = min(1.0, uv / 8.0)
    return float(clear * (1.0 - 0.7 * cloud / 100.0))


def _grid(now: datetime, n: int, step_minutes: float) -> list[datetime]:
    return [now + timedelta(minutes=step_minutes * k) for k in range(n)]


def _interp_series(
    times: list[datetime],
    values: list[float],
    grid: list[datetime],
    fill: float,
) -> np.ndarray:
    """Piecewise-linear interpolation of a timestamped series onto ``grid``."""
    if not times or not values:
        return np.full(len(grid), fill, dtype=float)
    base = grid[0]
    xs = np.array([(t - base).total_seconds() for t in times], dtype=float)
    ys = np.array(values, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    gx = np.array([(t - base).total_seconds() for t in grid], dtype=float)
    return np.interp(gx, xs, ys, left=ys[0], right=ys[-1]).astype(float)


async def async_get_weather_forecast(
    hass: HomeAssistant,
    weather_entity: str,
    n: int,
    step_minutes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_out, sol)`` arrays of length ``n`` over the horizon."""
    now = dt_util.now()
    grid = _grid(now, n, step_minutes)
    cur = hass.states.get(weather_entity)
    fallback_temp = 10.0
    if cur is not None:
        try:
            fallback_temp = float(cur.attributes.get("temperature", fallback_temp))
        except (TypeError, ValueError):
            pass

    forecast: list[dict] = []
    try:
        response = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": weather_entity, "type": "hourly"},
            blocking=True,
            return_response=True,
        )
        forecast = (response or {}).get(weather_entity, {}).get("forecast", []) or []
    except Exception as err:  # noqa: BLE001 - degrade gracefully
        _LOGGER.warning("Weather forecast unavailable for %s: %s", weather_entity, err)

    times: list[datetime] = []
    temps: list[float] = []
    sols: list[float] = []
    for point in forecast:
        ts = point.get("datetime")
        if ts is None:
            continue
        parsed = dt_util.parse_datetime(ts) if isinstance(ts, str) else ts
        if parsed is None:
            continue
        times.append(dt_util.as_local(parsed))
        temps.append(float(point.get("temperature", fallback_temp)))
        sols.append(
            solar_proxy(point.get("cloud_coverage"), point.get("uv_index"))
        )

    t_out = _interp_series(times, temps, grid, fallback_temp)
    sol = _interp_series(times, sols, grid, 0.0)
    return t_out, sol


def _extract_price_points(state) -> tuple[list[datetime], list[float]]:
    """Best-effort extraction of timestamped prices from common price integrations."""
    if state is None:
        return [], []
    attrs = state.attributes
    times: list[datetime] = []
    values: list[float] = []
    for key in ("raw_today", "raw_tomorrow", "forecast"):
        series = attrs.get(key)
        if not isinstance(series, list):
            continue
        for item in series:
            if not isinstance(item, dict):
                continue
            start = item.get("start") or item.get("hour") or item.get("time")
            value = item.get("value")
            if value is None:
                value = item.get("price")
            if start is None or value is None:
                continue
            parsed = dt_util.parse_datetime(start) if isinstance(start, str) else start
            if parsed is None:
                continue
            times.append(dt_util.as_local(parsed))
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                times.pop()
    return times, values


async def async_get_price_forecast(
    hass: HomeAssistant,
    price_entity: str | None,
    n: int,
    step_minutes: float,
) -> np.ndarray:
    """Return a price array of length ``n``. Flat 1.0 if no price entity configured."""
    grid = _grid(dt_util.now(), n, step_minutes)
    if not price_entity:
        return np.ones(n, dtype=float)
    state = hass.states.get(price_entity)
    flat = 1.0
    if state is not None:
        try:
            flat = float(state.state)
        except (TypeError, ValueError):
            flat = 1.0
    times, values = _extract_price_points(state)
    if not times:
        return np.full(n, flat, dtype=float)
    series = _interp_series(times, values, grid, flat)
    # Guard against zero/negative scaling collapsing the objective.
    return np.clip(series, 0.01, None)
