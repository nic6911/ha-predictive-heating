"""Tests for the config flow."""

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.predictive_heating.const import (
    CONF_WEATHER_ENTITY,
    CONF_ZONES,
    DOMAIN,
)


async def test_user_flow_creates_entry(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_WEATHER_ENTITY: "weather.home"},
    )
    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert result2["data"][CONF_WEATHER_ENTITY] == "weather.home"
    assert result2["options"][CONF_ZONES] == []


async def test_single_instance(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    first = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    await hass.config_entries.flow.async_configure(
        first["flow_id"], {CONF_WEATHER_ENTITY: "weather.home"}
    )
    second = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert second["type"] == FlowResultType.ABORT
    assert second["reason"] == "single_instance_allowed"
