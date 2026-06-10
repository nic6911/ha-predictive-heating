"""Button entities: train-from-history and reset-model shortcuts.

These wrap the integration's services so users can trigger learning from the
dashboard without going through Developer Tools.
"""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_ZONE_ID, DOMAIN
from .coordinator import PredictiveHeatingCoordinator
from .entity import PredictiveZoneEntity
from .services import SERVICE_RESET_MODEL, SERVICE_TRAIN_NOW


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: PredictiveHeatingCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    entities: list[ButtonEntity] = [TrainAllButton(coordinator)]
    for cfg in {**entry.data, **entry.options}.get("zones", []):
        zone_id = cfg[CONF_ZONE_ID]
        name = cfg.get("name", zone_id)
        entities.append(TrainZoneButton(coordinator, zone_id, name))
        entities.append(ResetZoneButton(coordinator, zone_id, name))
    async_add_entities(entities)


class TrainAllButton(ButtonEntity):
    """Train every configured zone from recorder history."""

    _attr_has_entity_name = True
    _attr_translation_key = "train_all"
    _attr_icon = "mdi:school"

    def __init__(self, coordinator: PredictiveHeatingCoordinator) -> None:
        self.coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_train_all"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="HeatPilot Controller",
            manufacturer="HeatPilot",
            model="Predictive controller",
        )

    async def async_press(self) -> None:
        await self.coordinator.hass.services.async_call(
            DOMAIN, SERVICE_TRAIN_NOW, {}, blocking=True
        )


class TrainZoneButton(PredictiveZoneEntity, ButtonEntity):
    """Train one zone from recorder history."""

    _attr_translation_key = "train_zone"
    _attr_icon = "mdi:school-outline"

    def __init__(self, coordinator, zone_id, name) -> None:
        super().__init__(coordinator, zone_id, name)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{zone_id}_train"

    async def async_press(self) -> None:
        await self.coordinator.hass.services.async_call(
            DOMAIN, SERVICE_TRAIN_NOW, {"zone_id": self._zone_id}, blocking=True
        )


class ResetZoneButton(PredictiveZoneEntity, ButtonEntity):
    """Forget the learned model for one zone."""

    _attr_translation_key = "reset_zone"
    _attr_icon = "mdi:restore"

    def __init__(self, coordinator, zone_id, name) -> None:
        super().__init__(coordinator, zone_id, name)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{zone_id}_reset"

    async def async_press(self) -> None:
        await self.coordinator.hass.services.async_call(
            DOMAIN, SERVICE_RESET_MODEL, {"zone_id": self._zone_id}, blocking=True
        )
