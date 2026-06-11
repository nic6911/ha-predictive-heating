"""Binary sensor entities for Predictive Floor Heating."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_ZONE_ID, DOMAIN
from .coordinator import PredictiveHeatingCoordinator, ZoneResult
from .entity import PredictiveZoneEntity


@dataclass(frozen=True, kw_only=True)
class ZoneBinaryDescription(BinarySensorEntityDescription):
    value_fn: Callable[[ZoneResult], bool]


BINARY_SENSORS: tuple[ZoneBinaryDescription, ...] = (
    ZoneBinaryDescription(
        key="manual_override",
        translation_key="manual_override",
        value_fn=lambda z: z.manual_override,
    ),
    ZoneBinaryDescription(
        key="free_heat",
        translation_key="free_heat",
        value_fn=lambda z: not z.has_authority,
    ),
    ZoneBinaryDescription(
        key="disturbance_detected",
        translation_key="disturbance_detected",
        value_fn=lambda z: z.disturbance,
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
    entities: list[BinarySensorEntity] = []
    for cfg in {**entry.data, **entry.options}.get("zones", []):
        zone_id = cfg[CONF_ZONE_ID]
        name = cfg.get("name", zone_id)
        for desc in BINARY_SENSORS:
            entities.append(ZoneBinarySensor(coordinator, zone_id, name, desc))
    async_add_entities(entities)


class ZoneBinarySensor(PredictiveZoneEntity, BinarySensorEntity):
    entity_description: ZoneBinaryDescription

    def __init__(self, coordinator, zone_id, name, description) -> None:
        super().__init__(coordinator, zone_id, name)
        self.entity_description = description
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{zone_id}_{description.key}"

    @property
    def is_on(self) -> bool | None:
        zone = self.zone
        if zone is None:
            return None
        return self.entity_description.value_fn(zone)
