"""Shared entity helpers."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import PredictiveHeatingCoordinator, ZoneResult


class PredictiveZoneEntity(CoordinatorEntity[PredictiveHeatingCoordinator]):
    """Base class for per-zone entities."""

    _attr_has_entity_name = True

    def __init__(
        self, coordinator: PredictiveHeatingCoordinator, zone_id: str, name: str
    ) -> None:
        super().__init__(coordinator)
        self._zone_id = zone_id
        self._zone_name = name
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{coordinator.entry.entry_id}_{zone_id}")},
            name=f"HeatPilot {name}",
            manufacturer="HeatPilot",
            model="Predictive zone",
        )

    @property
    def zone(self) -> ZoneResult | None:
        data = self.coordinator.data or {}
        return data.get(self._zone_id)
