"""Persistence of per-zone models, training buffers and runtime overrides."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import STORAGE_KEY, STORAGE_VERSION
from .models.rc_model_3r2c import RCModel3R2C


class ModelStore:
    """Thin wrapper around HA's Store for model parameters and sample buffers."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry_id}")
        self._data: dict = {"zones": {}, "meta": {}}

    async def async_load(self) -> None:
        loaded = await self._store.async_load()
        if loaded:
            self._data = loaded
        self._data.setdefault("zones", {})
        self._data.setdefault("meta", {})

    # ---------------------------------------------------------------- models
    def get_model(self, zone_id: str) -> RCModel3R2C | None:
        raw = self._data["zones"].get(zone_id, {}).get("model")
        if not raw:
            return None
        return RCModel3R2C.from_dict(raw)

    def get_buffer(self, zone_id: str) -> list[list[float]]:
        return list(self._data["zones"].get(zone_id, {}).get("buffer", []))

    def set_model(self, zone_id: str, model: RCModel3R2C) -> None:
        self._data["zones"].setdefault(zone_id, {})["model"] = model.as_dict()

    def set_buffer(self, zone_id: str, buffer: list[list[float]]) -> None:
        self._data["zones"].setdefault(zone_id, {})["buffer"] = buffer

    def clear_zone(self, zone_id: str) -> None:
        self._data["zones"].pop(zone_id, None)

    # ------------------------------------------------------------------ meta
    def _meta(self) -> dict:
        return self._data.setdefault("meta", {})

    def get_overrides(self) -> dict:
        return dict(self._meta().get("overrides", {}))

    def set_overrides(self, overrides: dict) -> None:
        self._meta()["overrides"] = overrides

    def get_global_comfort(self) -> dict:
        return dict(self._meta().get("global_comfort", {}))

    def set_global_comfort(self, comfort: dict) -> None:
        self._meta()["global_comfort"] = comfort

    def get_mode_override(self) -> str | None:
        return self._meta().get("mode_override")

    def set_mode_override(self, mode: str | None) -> None:
        self._meta()["mode_override"] = mode

    def get_master_enabled(self) -> bool | None:
        return self._meta().get("master_enabled")

    def set_master_enabled(self, enabled: bool) -> None:
        self._meta()["master_enabled"] = bool(enabled)

    async def async_save(self) -> None:
        await self._store.async_save(self._data)
