"""Runtime overrides must persist across restarts, global comfort must push to
every zone, and single-sample temperature spikes must be ignored."""

from pytest_homeassistant_custom_component.common import MockConfigEntry

from homeassistant.core import HomeAssistant

from custom_components.predictive_heating.const import (
    CONF_CLIMATE_ENTITY,
    CONF_COMFORT_MAX,
    CONF_COMFORT_MIN,
    CONF_COMFORT_TARGET,
    CONF_WEATHER_ENTITY,
    CONF_ZONE_ENABLED,
    CONF_ZONE_ID,
    CONF_ZONES,
    DOMAIN,
)


def _make_entry(*zones) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_WEATHER_ENTITY: "weather.home"},
        options={CONF_ZONES: list(zones)},
    )
    return entry


def _zone(zone_id: str) -> dict:
    return {
        CONF_ZONE_ID: zone_id,
        CONF_CLIMATE_ENTITY: f"climate.{zone_id}",
        "name": zone_id.title(),
        CONF_COMFORT_MIN: 19.0,
        CONF_COMFORT_TARGET: 21.0,
        CONF_COMFORT_MAX: 23.0,
        CONF_ZONE_ENABLED: True,
    }


def _set_climate(hass: HomeAssistant, zone_id: str, current: float) -> None:
    hass.states.async_set(
        f"climate.{zone_id}",
        "off",
        {
            "current_temperature": current,
            "temperature": 21.0,
            "min_temp": 5,
            "max_temp": 30,
            "target_temp_step": 0.1,
            "supported_features": 1,
        },
    )


async def test_overrides_persist_across_reload(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    _set_climate(hass, "stue", 20.5)
    hass.states.async_set(
        "weather.home", "cloudy", {"temperature": 4.0, "cloud_coverage": 80, "uv_index": 1}
    )
    entry = _make_entry(_zone("stue"))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    coordinator.set_zone_override("stue", CONF_COMFORT_TARGET, 22.5)
    coordinator.set_zone_override("stue", CONF_COMFORT_MAX, 25.0)
    await hass.async_block_till_done()

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    reloaded = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    assert reloaded.zone_param("stue", CONF_COMFORT_TARGET, 21.0) == 22.5
    assert reloaded.zone_param("stue", CONF_COMFORT_MAX, 23.0) == 25.0


async def test_global_comfort_pushes_to_all_zones(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    _set_climate(hass, "stue", 20.5)
    _set_climate(hass, "kjokken", 21.0)
    hass.states.async_set(
        "weather.home", "cloudy", {"temperature": 4.0, "cloud_coverage": 80, "uv_index": 1}
    )
    entry = _make_entry(_zone("stue"), _zone("kjokken"))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    coordinator.set_global_comfort(CONF_COMFORT_TARGET, 22.0)
    assert coordinator.zone_param("stue", CONF_COMFORT_TARGET, 21.0) == 22.0
    assert coordinator.zone_param("kjokken", CONF_COMFORT_TARGET, 21.0) == 22.0

    # Individual adjustments afterwards must still win.
    coordinator.set_zone_override("stue", CONF_COMFORT_TARGET, 19.5)
    assert coordinator.zone_param("stue", CONF_COMFORT_TARGET, 21.0) == 19.5
    assert coordinator.zone_param("kjokken", CONF_COMFORT_TARGET, 21.0) == 22.0

    # A later global change re-syncs everyone again.
    coordinator.set_global_comfort(CONF_COMFORT_TARGET, 23.0)
    assert coordinator.zone_param("stue", CONF_COMFORT_TARGET, 21.0) == 23.0


async def test_temperature_spike_is_ignored(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    _set_climate(hass, "stue", 20.5)
    hass.states.async_set(
        "weather.home", "cloudy", {"temperature": 4.0, "cloud_coverage": 80, "uv_index": 1}
    )
    entry = _make_entry(_zone("stue"))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    await coordinator.async_refresh()
    core = coordinator._zones["stue"]
    assert core.last_obs is not None
    good = core.last_obs["indoor"]

    # Sensor glitches to 0 C for one sample.
    _set_climate(hass, "stue", 0.0)
    await coordinator.async_refresh()
    assert coordinator.data["stue"].indoor is None
    assert coordinator.data["stue"].prediction_error is None
    assert core.last_obs["indoor"] == good  # last good value preserved

    # Recovery: the next sane reading is accepted again.
    _set_climate(hass, "stue", 20.6)
    await coordinator.async_refresh()
    assert coordinator.data["stue"].indoor == 20.6
