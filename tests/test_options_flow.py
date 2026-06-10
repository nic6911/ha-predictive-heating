"""Reproduce the options flow: open the menu and add a zone."""

from pathlib import Path

from pytest_homeassistant_custom_component.common import MockConfigEntry

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.predictive_heating.const import (
    CONF_CLIMATE_ENTITY,
    CONF_WEATHER_ENTITY,
    CONF_ZONES,
    DOMAIN,
)


async def test_options_menu_and_add_zone(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    hass.states.async_set(
        "climate.stue_ka",
        "off",
        {
            "current_temperature": 21.0,
            "temperature": 21.5,
            "min_temp": 5,
            "max_temp": 30,
            "target_temp_step": 0.1,
            "supported_features": 1,
        },
    )
    hass.states.async_set("weather.home", "cloudy", {"temperature": 5.0})

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_WEATHER_ENTITY: "weather.home"},
        options={CONF_ZONES: []},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Open the options flow -> should present a MENU with add_zone.
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.MENU, result
    assert "add_zone" in result["menu_options"]

    # Choose add_zone -> should show the zone form.
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "add_zone"}
    )
    assert result["type"] == FlowResultType.FORM, result
    assert result["step_id"] == "add_zone"

    # Submit a zone.
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_CLIMATE_ENTITY: "climate.stue_ka", "name": "Stue"}
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY, result
    zones = result["data"][CONF_ZONES]
    assert len(zones) == 1
    assert zones[0][CONF_CLIMATE_ENTITY] == "climate.stue_ka"


def test_options_flow_never_sets_reserved_config_entry():
    """Guard: assigning ``self.config_entry`` in an OptionsFlow hits a deprecated
    setter that breaks in newer HA (-> 500 when opening the dialog). We must use a
    private attribute instead."""
    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "predictive_heating"
        / "config_flow.py"
    ).read_text(encoding="utf-8")
    assert "self.config_entry =" not in src, (
        "Do not assign self.config_entry in OptionsFlow; use self._entry instead."
    )
