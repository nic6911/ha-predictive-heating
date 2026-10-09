"""Number entities for per-zone and controller-wide comfort bounds."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.components.number import NumberEntity, NumberEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
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

GLOBAL_NUMBERS: tuple[ZoneNumberDescription, ...] = (
    ZoneNumberDescription(
        key="global_comfort_min",
        translation_key="global_comfort_min",
        param_key=CONF_COMFORT_MIN,
        default=DEFAULT_COMFORT_MIN,
        native_min_value=5,
        native_max_value=30,
        native_step=0.5,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
    ),
    ZoneNumberDescription(
        key="global_comfort_target",
        translation_key="global_comfort_target",
        param_key=CONF_COMFORT_TARGET,
        default=DEFAULT_COMFORT_TARGET,
        native_min_value=5,
        native_max_value=30,
        native_step=0.5,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
    ),
    ZoneNumberDescription(
        key="global_comfort_max",
        translation_key="global_comfort_max",
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
    entities: list[NumberEntity] = [
        GlobalComfortNumber(coordinator, desc) for desc in GLOBAL_NUMBERS
    ]
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


class GlobalComfortNumber(NumberEntity):
    """Controller-wide comfort bound that pushes to every zone when changed."""

    entity_description: ZoneNumberDescription
    _attr_has_entity_name = True

    def __init__(
        self, coordinator: PredictiveHeatingCoordinator, description: ZoneNumberDescription
    ) -> None:
        self.coordinator = coordinator
        self.entity_description = description
        self._attr_unique_id = (
            f"{coordinator.entry.entry_id}_{description.key}"
        )
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="HeatPilot Controller",
            manufacturer="HeatPilot",
            model="Predictive controller",
        )

    @property
    def native_value(self) -> float:
        return float(
            self.coordinator.global_comfort_value(self.entity_description.param_key)
        )

    async def async_set_native_value(self, value: float) -> None:
        self.coordinator.set_global_comfort(
            self.entity_description.param_key, float(value)
        )
        self.async_write_ha_state()
        await self.coordinator.async_request_refresh()
