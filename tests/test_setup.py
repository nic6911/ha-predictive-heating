"""End-to-end setup test: mock a climate + weather entity and verify the
coordinator sets up, runs a control cycle, and produces a recommendation."""

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


async def test_setup_entry_runs_cycle(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    hass.states.async_set(
        "climate.test_room",
        "off",
        {
            "current_temperature": 20.5,
            "temperature": 21.0,
            "min_temp": 5,
            "max_temp": 30,
            "target_temp_step": 0.1,
            "supported_features": 1,
        },
    )
    hass.states.async_set(
        "weather.home",
        "cloudy",
        {"temperature": 4.0, "cloud_coverage": 80, "uv_index": 1},
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_WEATHER_ENTITY: "weather.home"},
        options={
            CONF_ZONES: [
                {
                    CONF_ZONE_ID: "test_room",
                    CONF_CLIMATE_ENTITY: "climate.test_room",
                    "name": "Test Room",
                    CONF_COMFORT_MIN: 19.0,
                    CONF_COMFORT_TARGET: 21.0,
                    CONF_COMFORT_MAX: 23.0,
                    CONF_ZONE_ENABLED: True,
                }
            ]
        },
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    assert "test_room" in coordinator.data
    result = coordinator.data["test_room"]
    assert result.indoor == 20.5
    assert result.recommended_setpoint is not None
    # Recommendation must respect the comfort band.
    assert 19.0 <= result.recommended_setpoint <= 23.0

    # At least the recommended-setpoint sensor should be registered.
    states = [
        s for s in hass.states.async_all("sensor") if "recommended" in s.entity_id
    ]
    assert states, "recommended setpoint sensor not created"

    # Train/reset buttons should be available so users don't need Dev Tools.
    buttons = hass.states.async_all("button")
    assert any("train" in s.entity_id for s in buttons), "train button not created"


async def test_disabled_zone_still_predicts(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    """Disabling predictive control must NOT blank the advisory sensors: the zone
    should keep computing a recommendation but never apply it."""
    hass.states.async_set(
        "climate.test_room",
        "off",
        {
            "current_temperature": 20.5,
            "temperature": 21.0,
            "min_temp": 5,
            "max_temp": 30,
            "target_temp_step": 0.1,
            "supported_features": 1,
        },
    )
    hass.states.async_set(
        "weather.home",
        "cloudy",
        {"temperature": 4.0, "cloud_coverage": 80, "uv_index": 1},
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_WEATHER_ENTITY: "weather.home"},
        options={
            CONF_ZONES: [
                {
                    CONF_ZONE_ID: "test_room",
                    CONF_CLIMATE_ENTITY: "climate.test_room",
                    "name": "Test Room",
                    CONF_COMFORT_MIN: 19.0,
                    CONF_COMFORT_TARGET: 21.0,
                    CONF_COMFORT_MAX: 23.0,
                    CONF_ZONE_ENABLED: False,
                }
            ]
        },
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = hass.data[DOMAIN][entry.entry_id]["coordinator"].data["test_room"]
    assert result.enabled is False
    assert result.recommended_setpoint is not None  # still advisory, not "unknown"
    assert result.advisory is True
    assert result.applied is False
