"""Number entities for per-zone comfort bounds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.components.number import NumberEntity, NumberEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_COMFORT_MAX,
    CONF_COMFORT_MIN,
    CONF_COMFORT_TARGET,
    CONF_ZONE_ID,
    DEFAULT_COMFORT_MAX,
    DEFAULT_COMFORT_MIN,
    DEFAULT_COMFORT_TARGET,
    DOMAIN,
)
from .coordinator import PredictiveHeatingCoordinator
from .entity import PredictiveZoneEntity


@dataclass(frozen=True, kw_only=True)
class ZoneNumberDescription(NumberEntityDescription):
    param_key: str
    default: float


NUMBERS: tuple[ZoneNumberDescription, ...] = (
    ZoneNumberDescription(
        key="comfort_min",
        translation_key="comfort_min",
        param_key=CONF_COMFORT_MIN,
        default=DEFAULT_COMFORT_MIN,
        native_min_value=5,
        native_max_value=30,
        native_step=0.5,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
    ),
    ZoneNumberDescription(
        key="comfort_target",
        translation_key="comfort_target",
        param_key=CONF_COMFORT_TARGET,
        default=DEFAULT_COMFORT_TARGET,
        native_min_value=5,
        native_max_value=30,
        native_step=0.5,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
    ),
    ZoneNumberDescription(
        key="comfort_max",
        translation_key="comfort_max",
        param_key=CONF_COMFORT_MAX,
        default=DEFAULT_COMFORT_MAX,
        native_min_value=5,
        native_max_value=30,
        native_step=0.5,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: PredictiveHeatingCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    entities: list[NumberEntity] = []
    for cfg in {**entry.data, **entry.options}.get("zones", []):
        zone_id = cfg[CONF_ZONE_ID]
        name = cfg.get("name", zone_id)
        for desc in NUMBERS:
            entities.append(ZoneNumber(coordinator, zone_id, name, desc))
    async_add_entities(entities)


class ZoneNumber(PredictiveZoneEntity, NumberEntity):
    entity_description: ZoneNumberDescription

    def __init__(self, coordinator, zone_id, name, description) -> None:
        super().__init__(coordinator, zone_id, name)
        self.entity_description = description
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{zone_id}_{description.key}"

    @property
    def native_value(self) -> float:
        return float(
            self.coordinator.zone_param(
                self._zone_id,
                self.entity_description.param_key,
                self.entity_description.default,
            )
        )

    async def async_set_native_value(self, value: float) -> None:
        self.coordinator.set_zone_override(
            self._zone_id, self.entity_description.param_key, float(value)
        )
        self.async_write_ha_state()
        await self.coordinator.async_request_refresh()
