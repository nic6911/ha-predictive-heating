"""Switch entities: master enable and per-zone predictive enable."""

from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_ZONE_ENABLED, CONF_ZONE_ID, DOMAIN
from .coordinator import PredictiveHeatingCoordinator
from .entity import PredictiveZoneEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: PredictiveHeatingCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    entities: list[SwitchEntity] = [MasterSwitch(coordinator)]
    for cfg in {**entry.data, **entry.options}.get("zones", []):
        zone_id = cfg[CONF_ZONE_ID]
        name = cfg.get("name", zone_id)
        entities.append(ZoneEnableSwitch(coordinator, zone_id, name))
    async_add_entities(entities)


class MasterSwitch(SwitchEntity):
    """Global kill-switch for autonomous setpoint writing."""

    _attr_has_entity_name = True
    _attr_translation_key = "master_enable"
    _attr_icon = "mdi:home-thermometer"

    def __init__(self, coordinator: PredictiveHeatingCoordinator) -> None:
        self.coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_master_enable"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="HeatPilot Controller",
            manufacturer="HeatPilot",
            model="Predictive controller",
        )

    @property
    def is_on(self) -> bool:
        return bool(
            self.coordinator.hass.data.get(DOMAIN, {}).get("master_enabled", True)
        )

    async def async_turn_on(self, **kwargs) -> None:
        self.coordinator.hass.data[DOMAIN]["master_enabled"] = True
        self.async_write_ha_state()
        await self.coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        self.coordinator.hass.data[DOMAIN]["master_enabled"] = False
        self.async_write_ha_state()


class ZoneEnableSwitch(PredictiveZoneEntity, SwitchEntity):
    """Enable/disable predictive control for one zone."""

    _attr_translation_key = "zone_enable"

    def __init__(self, coordinator, zone_id, name) -> None:
        super().__init__(coordinator, zone_id, name)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{zone_id}_enable"

    @property
    def is_on(self) -> bool:
        return bool(
            self.coordinator.zone_param(self._zone_id, CONF_ZONE_ENABLED, True)
        )

    async def async_turn_on(self, **kwargs) -> None:
        self.coordinator.set_zone_override(self._zone_id, CONF_ZONE_ENABLED, True)
        self.async_write_ha_state()
        await self.coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        self.coordinator.set_zone_override(self._zone_id, CONF_ZONE_ENABLED, False)
        self.async_write_ha_state()
        await self.coordinator.async_request_refresh()
