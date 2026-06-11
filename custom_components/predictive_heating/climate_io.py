"""Read/write helpers for the underlying climate (thermostat) entities.

Encapsulates:
* reading the current indoor temperature and setpoint,
* writing a new setpoint via ``climate.set_temperature`` (with a deadband),
* detecting a *manual* setpoint change (user turned the dial) versus our own
  last write, so autonomous control can politely back off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging

from homeassistant.components.climate import ATTR_TEMPERATURE
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    MANUAL_OVERRIDE_HOLD,
    MANUAL_OVERRIDE_TOLERANCE,
    PLAUSIBLE_TEMP_MAX,
    PLAUSIBLE_TEMP_MIN,
    SETPOINT_DEADBAND,
)

_LOGGER = logging.getLogger(__name__)


def _plausible(value: float | None) -> float | None:
    """Return the value if it is a physically plausible temperature, else None.

    Filters out sensor-fault sentinels (e.g. 327.67 C) so a single bad reading can
    never enter the learning/control pipeline.
    """
    if value is None:
        return None
    if value < PLAUSIBLE_TEMP_MIN or value > PLAUSIBLE_TEMP_MAX:
        _LOGGER.debug("Ignoring implausible temperature reading %.2f", value)
        return None
    return value


@dataclass
class ZoneRuntime:
    """Mutable per-zone runtime state held in memory by the coordinator."""

    zone_id: str
    last_written_setpoint: float | None = None
    override_until: datetime | None = None
    min_temp: float = 5.0
    max_temp: float = 30.0
    step: float = 0.1
    extra: dict = field(default_factory=dict)


def read_indoor(hass: HomeAssistant, climate_entity: str, temp_sensor: str | None) -> float | None:
    """Return the indoor temperature, preferring an explicit sensor if mapped."""
    if temp_sensor:
        state = hass.states.get(temp_sensor)
        if state is not None:
            try:
                return _plausible(float(state.state))
            except (TypeError, ValueError):
                pass
    state = hass.states.get(climate_entity)
    if state is None:
        return None
    val = state.attributes.get("current_temperature")
    try:
        return None if val is None else _plausible(float(val))
    except (TypeError, ValueError):
        return None


def read_setpoint(hass: HomeAssistant, climate_entity: str) -> float | None:
    state = hass.states.get(climate_entity)
    if state is None:
        return None
    val = state.attributes.get(ATTR_TEMPERATURE)
    try:
        return None if val is None else _plausible(float(val))
    except (TypeError, ValueError):
        return None


def read_limits(hass: HomeAssistant, climate_entity: str, runtime: ZoneRuntime) -> None:
    """Refresh actuator limits from the climate entity attributes."""
    state = hass.states.get(climate_entity)
    if state is None:
        return
    runtime.min_temp = float(state.attributes.get("min_temp", runtime.min_temp))
    runtime.max_temp = float(state.attributes.get("max_temp", runtime.max_temp))
    runtime.step = float(state.attributes.get("target_temp_step", runtime.step))


def detect_manual_override(
    hass: HomeAssistant, climate_entity: str, runtime: ZoneRuntime
) -> bool:
    """Return True if the current setpoint differs from what we last wrote.

    Engages a hold window so the user's manual choice is respected for a while.
    """
    current = read_setpoint(hass, climate_entity)
    now = dt_util.now()
    if (
        current is not None
        and runtime.last_written_setpoint is not None
        and abs(current - runtime.last_written_setpoint) > MANUAL_OVERRIDE_TOLERANCE
    ):
        runtime.override_until = now + MANUAL_OVERRIDE_HOLD
    return runtime.override_until is not None and now < runtime.override_until


def quantise(value: float, runtime: ZoneRuntime) -> float:
    """Clamp to actuator limits and round to the entity's step."""
    value = max(runtime.min_temp, min(runtime.max_temp, value))
    step = runtime.step or 0.1
    return round(round(value / step) * step, 3)


async def async_write_setpoint(
    hass: HomeAssistant,
    climate_entity: str,
    runtime: ZoneRuntime,
    setpoint: float,
) -> bool:
    """Write a new setpoint if it differs from the last write beyond the deadband."""
    target = quantise(setpoint, runtime)
    last = runtime.last_written_setpoint
    if last is not None and abs(target - last) < SETPOINT_DEADBAND:
        return False
    await hass.services.async_call(
        "climate",
        "set_temperature",
        {ATTR_ENTITY_ID: climate_entity, ATTR_TEMPERATURE: target},
        blocking=True,
    )
    runtime.last_written_setpoint = target
    _LOGGER.debug("Wrote setpoint %.2f to %s", target, climate_entity)
    return True
