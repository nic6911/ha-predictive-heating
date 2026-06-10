"""Sensor entities for Predictive Floor Heating."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_ZONE_ID, DOMAIN
from .coordinator import PredictiveHeatingCoordinator, ZoneResult
from .entity import PredictiveZoneEntity


@dataclass(frozen=True, kw_only=True)
class ZoneSensorDescription(SensorEntityDescription):
    """Describes a per-zone sensor."""

    value_fn: Callable[[ZoneResult], float | None]


SENSORS: tuple[ZoneSensorDescription, ...] = (
    ZoneSensorDescription(
        key="recommended_setpoint",
        translation_key="recommended_setpoint",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda z: z.recommended_setpoint,
    ),
    ZoneSensorDescription(
        key="predicted_temperature",
        translation_key="predicted_temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda z: z.predicted_temperature,
    ),
    ZoneSensorDescription(
        key="estimated_savings",
        translation_key="estimated_savings",
        native_unit_of_measurement=UnitOfTemperature.KELVIN,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda z: z.estimated_savings,
    ),
    ZoneSensorDescription(
        key="fit_rmse",
        translation_key="fit_rmse",
        native_unit_of_measurement=UnitOfTemperature.KELVIN,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z: z.fit_rmse,
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
    entities: list[SensorEntity] = []
    for cfg in {**entry.data, **entry.options}.get("zones", []):
        zone_id = cfg[CONF_ZONE_ID]
        name = cfg.get("name", zone_id)
        for desc in SENSORS:
            entities.append(ZoneSensor(coordinator, zone_id, name, desc))
    async_add_entities(entities)


class ZoneSensor(PredictiveZoneEntity, SensorEntity):
    """A per-zone predictive sensor."""

    entity_description: ZoneSensorDescription

    def __init__(self, coordinator, zone_id, name, description) -> None:
        super().__init__(coordinator, zone_id, name)
        self.entity_description = description
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{zone_id}_{description.key}"

    @property
    def native_value(self) -> float | None:
        zone = self.zone
        if zone is None:
            return None
        return self.entity_description.value_fn(zone)
