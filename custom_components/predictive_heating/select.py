"""Select entity for the global optimization mode."""

from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, MODES
from .coordinator import PredictiveHeatingCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: PredictiveHeatingCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    async_add_entities([ModeSelect(coordinator)])


class ModeSelect(SelectEntity):
    """Switch the global comfort/eco/price optimization profile."""

    _attr_has_entity_name = True
    _attr_translation_key = "mode"
    _attr_options = MODES
    _attr_icon = "mdi:tune"

    def __init__(self, coordinator: PredictiveHeatingCoordinator) -> None:
        self.coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_mode"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="HeatPilot Controller",
            manufacturer="HeatPilot",
            model="Predictive controller",
        )

    @property
    def current_option(self) -> str:
        return self.coordinator.active_mode

    async def async_select_option(self, option: str) -> None:
        self.coordinator.mode_override = option
        self.async_write_ha_state()
        await self.coordinator.async_request_refresh()
