"""Tests for sensor-fault filtering in climate_io reads."""

from types import SimpleNamespace

from custom_components.predictive_heating import climate_io


class _FakeStates:
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, entity_id):
        return self._mapping.get(entity_id)


class _FakeHass:
    def __init__(self, mapping):
        self.states = _FakeStates(mapping)


def _state(state=None, attrs=None):
    return SimpleNamespace(state=state, attributes=attrs or {})


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
