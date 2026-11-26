"""Tests for climate_io reads, setpoint writes and manual-override detection."""

import math
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from homeassistant.util import dt as dt_util

from custom_components.predictive_heating import climate_io
from custom_components.predictive_heating.climate_io import ZoneRuntime


class _FakeStates:
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, entity_id):
        return self._mapping.get(entity_id)


class _FakeServices:
    """Records set_temperature calls and emulates a 0.5 C actuator."""

    def __init__(self, hass):
        self.hass = hass
        self.calls = []

    async def async_call(self, domain, service, data, blocking=True):
        self.calls.append((domain, service, data))
        # The real Sentio floors a setpoint onto its 0.5 C grid.
        snapped = math.floor(data["temperature"] * 2.0) / 2.0
        self.hass.states._mapping["climate.room"].attributes["temperature"] = round(snapped, 3)


class _FakeHass:
    def __init__(self, mapping):
        self.states = _FakeStates(mapping)
        self.services = _FakeServices(self)


def _state(state=None, attrs=None):
    return SimpleNamespace(state=state, attributes=attrs or {})


def _climate_hass(setpoint: float) -> _FakeHass:
    return _FakeHass(
        {
            "climate.room": _state(
                state="off",
                attrs={
                    "temperature": setpoint,
                    "current_temperature": 22.0,
                    "min_temp": 5.0,
                    "max_temp": 30.0,
                    "target_temp_step": 0.1,
                },
            )
        }
    )


def test_read_indoor_rejects_sensor_fault_sentinel():
    """The 327.67 C sentinel must be filtered to None, not learned from."""
    hass = _FakeHass({"sensor.room": _state(state="327.67")})
    assert climate_io.read_indoor(hass, "climate.room", "sensor.room") is None


def test_read_indoor_accepts_plausible_value():
    hass = _FakeHass({"sensor.room": _state(state="21.4")})
    assert climate_io.read_indoor(hass, "climate.room", "sensor.room") == 21.4


def test_read_setpoint_rejects_out_of_range():
    hass = _FakeHass(
        {"climate.room": _state(attrs={"temperature": 327.67})}
    )
    assert climate_io.read_setpoint(hass, "climate.room") is None


def test_read_indoor_falls_back_to_climate_attribute():
    hass = _FakeHass(
        {"climate.room": _state(attrs={"current_temperature": 22.1})}
    )
    assert climate_io.read_indoor(hass, "climate.room", None) == 22.1


async def test_write_adopts_device_rounding_and_learns_step():
    """A 0.5 C actuator must not look like a manual override."""
    hass = _climate_hass(21.0)
    runtime = ZoneRuntime(zone_id="room", step=0.1)

    assert await climate_io.async_write_setpoint(hass, "climate.room", runtime, 22.2)
    assert runtime.effective_step == 0.5
    assert runtime.last_written_setpoint == 22.0
    assert climate_io.detect_manual_override(hass, "climate.room", runtime) is False


async def test_flooring_with_large_offset_is_adopted():
    """22.4 floored to 22.0 (0.4 C) must still count as our own write."""
    hass = _climate_hass(21.0)
    runtime = ZoneRuntime(zone_id="room", step=0.1)

    assert await climate_io.async_write_setpoint(hass, "climate.room", runtime, 22.4)
    assert runtime.effective_step == 0.5
    assert runtime.last_written_setpoint == 22.0
    assert climate_io.detect_manual_override(hass, "climate.room", runtime) is False


def test_user_change_after_grid_is_learned_is_detected():
    hass = _climate_hass(23.0)
    now = dt_util.now()
    runtime = ZoneRuntime(
        zone_id="room",
        step=0.1,
        effective_step=0.5,
        last_written_setpoint=22.5,
        last_written_at=now - timedelta(minutes=30),
        last_seen_setpoint=22.5,
    )
    assert climate_io.detect_manual_override(hass, "climate.room", runtime) is True


async def test_second_write_quantises_to_learned_step():
    hass = _climate_hass(21.0)
    runtime = ZoneRuntime(zone_id="room", step=0.1, effective_step=0.5, last_written_setpoint=21.0)

    assert await climate_io.async_write_setpoint(hass, "climate.room", runtime, 22.7)
    assert hass.services.calls[-1][2]["temperature"] == 22.5
    assert runtime.last_written_setpoint == 22.5


async def test_write_is_deadbanded():
    hass = _climate_hass(21.0)
    runtime = ZoneRuntime(zone_id="room", last_written_setpoint=21.0)
    assert await climate_io.async_write_setpoint(hass, "climate.room", runtime, 21.1) is False
    assert hass.services.calls == []


def test_device_rounding_is_not_manual_override():
    hass = _climate_hass(22.5)
    runtime = ZoneRuntime(
        zone_id="room",
        last_written_setpoint=22.7,
        last_written_at=dt_util.now(),
        last_seen_setpoint=22.7,
    )
    assert climate_io.detect_manual_override(hass, "climate.room", runtime) is False
    assert runtime.last_written_setpoint == 22.5
    assert runtime.effective_step == 0.5


def test_first_cycle_does_not_arm_override():
    hass = _climate_hass(22.5)
    runtime = ZoneRuntime(zone_id="room")
    assert climate_io.detect_manual_override(hass, "climate.room", runtime) is False
    assert runtime.override_until is None


def test_real_change_arms_once_and_can_expire():
    hass = _climate_hass(23.5)
    now = dt_util.now()
    runtime = ZoneRuntime(
        zone_id="room",
        last_written_setpoint=22.5,
        last_written_at=now - timedelta(minutes=30),
        last_seen_setpoint=22.5,
    )

    with patch.object(climate_io.dt_util, "now", return_value=now):
        assert climate_io.detect_manual_override(hass, "climate.room", runtime) is True
        armed_until = runtime.override_until
        assert armed_until is not None
        # The same change on the next cycle must not extend the hold.
        assert climate_io.detect_manual_override(hass, "climate.room", runtime) is True
        assert runtime.override_until == armed_until

    with patch.object(
        climate_io.dt_util, "now", return_value=now + timedelta(hours=3)
    ):
        assert climate_io.detect_manual_override(hass, "climate.room", runtime) is False
