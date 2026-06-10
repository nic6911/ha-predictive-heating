"""Config and options flow for Predictive Floor Heating."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_CLIMATE_ENTITY,
    CONF_CO2_ENTITY,
    CONF_COMFORT_MAX,
    CONF_COMFORT_MIN,
    CONF_COMFORT_TARGET,
    CONF_HORIZON_HOURS,
    CONF_IRRADIANCE_SENSOR,
    CONF_MODE,
    CONF_OUTDOOR_SENSOR,
    CONF_PRICE_ENTITY,
    CONF_PRICE_OPTIMIZE,
    CONF_STEP_MINUTES,
    CONF_TEMP_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_WEATHER_ENTITY,
    CONF_ZONE_ENABLED,
    CONF_ZONE_ID,
    CONF_ZONE_MODE,
    CONF_ZONES,
    DEFAULT_COMFORT_MAX,
    DEFAULT_COMFORT_MIN,
    DEFAULT_COMFORT_TARGET,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_MODE,
    DEFAULT_PRICE_OPTIMIZE,
    DEFAULT_STEP_MINUTES,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
    MAX_ZONES,
    MODES,
    ZONE_MODE_AUTO,
    ZONE_MODES,
)


def _opt(key: str, value: Any) -> vol.Marker:
    """Optional key that only carries a default when a real value exists.

    Avoids feeding ``None`` into selectors (e.g. EntitySelector), which would
    fail validation when the user leaves an optional entity field empty.
    """
    if value is None:
        return vol.Optional(key)
    return vol.Optional(key, default=value)


def _req(key: str, value: Any) -> vol.Marker:
    if value is None:
        return vol.Required(key)
    return vol.Required(key, default=value)


def _global_schema(defaults: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            _req(
                CONF_WEATHER_ENTITY, defaults.get(CONF_WEATHER_ENTITY)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="weather")
            ),
            _opt(
                CONF_PRICE_ENTITY, defaults.get(CONF_PRICE_ENTITY)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor")
            ),
            _opt(
                CONF_CO2_ENTITY, defaults.get(CONF_CO2_ENTITY)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor")
            ),
            vol.Optional(
                CONF_PRICE_OPTIMIZE,
                default=defaults.get(CONF_PRICE_OPTIMIZE, DEFAULT_PRICE_OPTIMIZE),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_MODE, default=defaults.get(CONF_MODE, DEFAULT_MODE)
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(options=MODES, translation_key="mode")
            ),
            vol.Optional(
                CONF_HORIZON_HOURS,
                default=defaults.get(CONF_HORIZON_HOURS, DEFAULT_HORIZON_HOURS),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=4, max=48, step=1, unit_of_measurement="h")
            ),
            vol.Optional(
                CONF_STEP_MINUTES,
                default=defaults.get(CONF_STEP_MINUTES, DEFAULT_STEP_MINUTES),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=15, max=60, step=15, unit_of_measurement="min")
            ),
            vol.Optional(
                CONF_UPDATE_INTERVAL,
                default=defaults.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=5, max=60, step=5, unit_of_measurement="min")
            ),
        }
    )


def _zone_schema(defaults: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            _req(
                CONF_CLIMATE_ENTITY, defaults.get(CONF_CLIMATE_ENTITY)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="climate")
            ),
            vol.Optional("name", default=defaults.get("name", "")): selector.TextSelector(),
            _opt(
                CONF_TEMP_SENSOR, defaults.get(CONF_TEMP_SENSOR)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor")
            ),
            _opt(
                CONF_OUTDOOR_SENSOR, defaults.get(CONF_OUTDOOR_SENSOR)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor")
            ),
            _opt(
                CONF_IRRADIANCE_SENSOR, defaults.get(CONF_IRRADIANCE_SENSOR)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor")
            ),
            vol.Optional(
                CONF_COMFORT_MIN,
                default=defaults.get(CONF_COMFORT_MIN, DEFAULT_COMFORT_MIN),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=5, max=30, step=0.5, unit_of_measurement="°C")
            ),
            vol.Optional(
                CONF_COMFORT_TARGET,
                default=defaults.get(CONF_COMFORT_TARGET, DEFAULT_COMFORT_TARGET),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=5, max=30, step=0.5, unit_of_measurement="°C")
            ),
            vol.Optional(
                CONF_COMFORT_MAX,
                default=defaults.get(CONF_COMFORT_MAX, DEFAULT_COMFORT_MAX),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=5, max=30, step=0.5, unit_of_measurement="°C")
            ),
            vol.Optional(
                CONF_ZONE_MODE, default=defaults.get(CONF_ZONE_MODE, ZONE_MODE_AUTO)
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=ZONE_MODES, translation_key="zone_mode"
                )
            ),
            vol.Optional(
                CONF_ZONE_ENABLED, default=defaults.get(CONF_ZONE_ENABLED, True)
            ): selector.BooleanSelector(),
        }
    )


class PredictiveHeatingConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial configuration."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> Any:
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")
        if user_input is not None:
            return self.async_create_entry(
                title="Predictive Floor Heating",
                data=user_input,
                options={CONF_ZONES: []},
            )
        return self.async_show_form(
            step_id="user", data_schema=_global_schema({})
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return PredictiveHeatingOptionsFlow(config_entry)


class PredictiveHeatingOptionsFlow(OptionsFlow):
    """Manage global settings and zones after setup."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        self.config_entry = config_entry
        self._options = dict(config_entry.options)
        self._options.setdefault(CONF_ZONES, list(config_entry.options.get(CONF_ZONES, [])))

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> Any:
        return self.async_show_menu(
            step_id="init",
            menu_options=["global", "add_zone", "remove_zone"],
        )

    async def async_step_global(self, user_input: dict[str, Any] | None = None) -> Any:
        if user_input is not None:
            merged = {**self.config_entry.data, **self._options, **user_input}
            merged[CONF_ZONES] = self._options.get(CONF_ZONES, [])
            return self.async_create_entry(title="", data=merged)
        defaults = {**self.config_entry.data, **self._options}
        return self.async_show_form(
            step_id="global", data_schema=_global_schema(defaults)
        )

    async def async_step_add_zone(self, user_input: dict[str, Any] | None = None) -> Any:
        zones = self._options.get(CONF_ZONES, [])
        if len(zones) >= MAX_ZONES:
            return self.async_abort(reason="max_zones_reached")
        if user_input is not None:
            zone = dict(user_input)
            climate_entity = zone[CONF_CLIMATE_ENTITY]
            zone[CONF_ZONE_ID] = climate_entity.split(".", 1)[-1]
            if not zone.get("name"):
                zone["name"] = zone[CONF_ZONE_ID].replace("_", " ").title()
            zones = [z for z in zones if z.get(CONF_ZONE_ID) != zone[CONF_ZONE_ID]]
            zones.append(zone)
            self._options[CONF_ZONES] = zones
            merged = {**self.config_entry.data, **self._options}
            return self.async_create_entry(title="", data=merged)
        return self.async_show_form(
            step_id="add_zone", data_schema=_zone_schema({})
        )

    async def async_step_remove_zone(
        self, user_input: dict[str, Any] | None = None
    ) -> Any:
        zones = self._options.get(CONF_ZONES, [])
        if not zones:
            return self.async_abort(reason="no_zones")
        if user_input is not None:
            keep = set(user_input.get("zones", []))
            self._options[CONF_ZONES] = [
                z for z in zones if z[CONF_ZONE_ID] in keep
            ]
            merged = {**self.config_entry.data, **self._options}
            return self.async_create_entry(title="", data=merged)
        options = [
            selector.SelectOptionDict(value=z[CONF_ZONE_ID], label=z.get("name", z[CONF_ZONE_ID]))
            for z in zones
        ]
        schema = vol.Schema(
            {
                vol.Optional(
                    "zones", default=[z[CONF_ZONE_ID] for z in zones]
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=options, multiple=True)
                )
            }
        )
        return self.async_show_form(step_id="remove_zone", data_schema=schema)
