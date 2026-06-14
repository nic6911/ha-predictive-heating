"""Persistence of per-zone RC models and training buffers."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import MODEL_3R2C, MODEL_AUTO, STORAGE_KEY, STORAGE_VERSION
from .models.rc_model import RCModel
from .models.rc_model_3r2c import RCModel3R2C


class ModelStore:
    """Thin wrapper around HA's Store for model parameters and sample buffers."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry_id}")
        self._data: dict = {"zones": {}}

    async def async_load(self) -> None:
        loaded = await self._store.async_load()
        if loaded:
            self._data = loaded
        self._data.setdefault("zones", {})

    def get_model(self, zone_id: str) -> RCModel | RCModel3R2C | None:
        raw = self._data["zones"].get(zone_id, {}).get("model")
        if not raw:
            return None
        if raw.get("model_type") in (MODEL_3R2C, MODEL_AUTO):
            return RCModel3R2C.from_dict(raw)
        return RCModel.from_dict(raw)

    def get_buffer(self, zone_id: str) -> list[list[float]]:
        return list(self._data["zones"].get(zone_id, {}).get("buffer", []))

    def set_model(self, zone_id: str, model: RCModel) -> None:
        self._data["zones"].setdefault(zone_id, {})["model"] = model.as_dict()

    def set_buffer(self, zone_id: str, buffer: list[list[float]]) -> None:
        self._data["zones"].setdefault(zone_id, {})["buffer"] = buffer

    def clear_zone(self, zone_id: str) -> None:
        self._data["zones"].pop(zone_id, None)

    async def async_save(self) -> None:
        await self._store.async_save(self._data)
