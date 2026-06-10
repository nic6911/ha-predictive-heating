"""The Predictive Floor Heating (HeatPilot) integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, PLATFORMS
from .coordinator import PredictiveHeatingCoordinator
from .services import async_register_services, async_unregister_services
from .storage import ModelStore

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Predictive Floor Heating from a config entry."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    domain_data.setdefault("master_enabled", True)

    store = ModelStore(hass, entry.entry_id)
    await store.async_load()

    coordinator = PredictiveHeatingCoordinator(hass, entry, store)
    coordinator.init_zones()
    await coordinator.async_config_entry_first_refresh()

    domain_data[entry.entry_id] = {"coordinator": coordinator, "store": store}

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    async_register_services(hass)

    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))
    return True


async def _async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        data = hass.data[DOMAIN].pop(entry.entry_id, None)
        if data is not None:
            await data["store"].async_save()
        # Drop services only when no entries remain.
        if not any(k for k in hass.data[DOMAIN] if k != "master_enabled"):
            async_unregister_services(hass)
    return unload_ok
