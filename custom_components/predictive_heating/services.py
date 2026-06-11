"""Services: bootstrap training, model reset, and comfort-profile updates."""

from __future__ import annotations

from datetime import timedelta
import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall
import homeassistant.helpers.config_validation as cv
from homeassistant.util import dt as dt_util

from .const import (
    CONF_CLIMATE_ENTITY,
    CONF_COMFORT_MAX,
    CONF_COMFORT_MIN,
    CONF_COMFORT_TARGET,
    CONF_OUTDOOR_SENSOR,
    CONF_STEP_MINUTES,
    CONF_TEMP_SENSOR,
    CONF_ZONE_ID,
    DEFAULT_STEP_MINUTES,
    DOMAIN,
    OUTLIER_ABS_CAP,
    OUTLIER_SIGMA,
    PLAUSIBLE_TEMP_MAX,
    PLAUSIBLE_TEMP_MIN,
)
from .models.identification import RecursiveLeastSquares, batch_fit
from .models.rc_model import RCModel

_LOGGER = logging.getLogger(__name__)

SERVICE_TRAIN_NOW = "train_now"
SERVICE_RESET_MODEL = "reset_model"
SERVICE_SET_COMFORT_PROFILE = "set_comfort_profile"

TRAIN_SCHEMA = vol.Schema(
    {
        vol.Optional("zone_id"): cv.string,
        vol.Optional("days", default=7): vol.All(int, vol.Range(min=1, max=60)),
    }
)
RESET_SCHEMA = vol.Schema({vol.Optional("zone_id"): cv.string})
COMFORT_SCHEMA = vol.Schema(
    {
        vol.Required("zone_id"): cv.string,
        vol.Optional(CONF_COMFORT_MIN): vol.Coerce(float),
        vol.Optional(CONF_COMFORT_TARGET): vol.Coerce(float),
        vol.Optional(CONF_COMFORT_MAX): vol.Coerce(float),
    }
)


def _iter_coordinators(hass: HomeAssistant):
    for key, value in hass.data.get(DOMAIN, {}).items():
        if key == "master_enabled":
            continue
        yield value["coordinator"]


def _plausible(value: float) -> bool:
    """Reject sensor-fault readings (e.g. the 327.67 C sentinel) from training."""
    return PLAUSIBLE_TEMP_MIN <= value <= PLAUSIBLE_TEMP_MAX


def _resample(pairs, grid):
    """Zero-order-hold a list of (time, value) pairs onto grid times."""
    out = []
    pairs = sorted(pairs, key=lambda p: p[0])
    idx = 0
    last = None
    for t in grid:
        while idx < len(pairs) and pairs[idx][0] <= t:
            last = pairs[idx][1]
            idx += 1
        out.append(last)
    return out


async def _bootstrap_zone(hass: HomeAssistant, coordinator, cfg: dict, days: int) -> None:
    from homeassistant.components.recorder import get_instance, history

    climate_entity = cfg[CONF_CLIMATE_ENTITY]
    temp_sensor = cfg.get(CONF_TEMP_SENSOR)
    outdoor_sensor = cfg.get(CONF_OUTDOOR_SENSOR)
    step_min = coordinator._global(CONF_STEP_MINUTES, DEFAULT_STEP_MINUTES)

    end = dt_util.utcnow()
    start = end - timedelta(days=days)
    entity_ids = [climate_entity]
    if temp_sensor:
        entity_ids.append(temp_sensor)
    if outdoor_sensor:
        entity_ids.append(outdoor_sensor)

    raw = await get_instance(hass).async_add_executor_job(
        history.get_significant_states,
        hass,
        start,
        end,
        entity_ids,
        None,
        True,
    )

    indoor_pairs, setpoint_pairs, outdoor_pairs = [], [], []
    for state in raw.get(climate_entity, []):
        ts = state.last_changed
        try:
            if temp_sensor is None:
                cur = state.attributes.get("current_temperature")
                if cur is not None and _plausible(float(cur)):
                    indoor_pairs.append((ts, float(cur)))
            sp = state.attributes.get("temperature")
            if sp is not None and _plausible(float(sp)):
                setpoint_pairs.append((ts, float(sp)))
        except (TypeError, ValueError):
            continue
    if temp_sensor:
        for state in raw.get(temp_sensor, []):
            try:
                if _plausible(float(state.state)):
                    indoor_pairs.append((state.last_changed, float(state.state)))
            except (TypeError, ValueError):
                continue
    if outdoor_sensor:
        for state in raw.get(outdoor_sensor, []):
            try:
                if _plausible(float(state.state)):
                    outdoor_pairs.append((state.last_changed, float(state.state)))
            except (TypeError, ValueError):
                continue

    if len(indoor_pairs) < 10:
        _LOGGER.warning(
            "Not enough history to bootstrap zone %s", cfg.get(CONF_ZONE_ID)
        )
        return

    n = int(days * 24 * 60 / step_min)
    grid = [start + timedelta(minutes=step_min * k) for k in range(n)]
    indoor = _resample(indoor_pairs, grid)
    setpoint = _resample(setpoint_pairs, grid)
    outdoor = _resample(outdoor_pairs, grid) if outdoor_pairs else [None] * n

    samples = []
    for k in range(n - 1):
        if (
            indoor[k] is None
            or indoor[k + 1] is None
            or setpoint[k] is None
        ):
            continue
        t_out = outdoor[k] if outdoor[k] is not None else 8.0
        u = RCModel.heat_demand(setpoint[k], indoor[k])
        samples.append((indoor[k], t_out, 0.0, u, indoor[k + 1]))

    model = batch_fit(samples, step_minutes=step_min)
    zone_id = cfg[CONF_ZONE_ID]
    core = coordinator._zones.get(zone_id)
    if core is not None:
        core.rls = RecursiveLeastSquares(
            params=model.params,
            outlier_sigma=OUTLIER_SIGMA,
            outlier_abs_cap=OUTLIER_ABS_CAP,
        )
        core.last_obs = None
        core.disturbance_until = None
        core.hold_setpoint = None
    coordinator.store.set_model(zone_id, model)
    _LOGGER.info(
        "Bootstrapped zone %s from %d samples (rmse=%s)",
        zone_id,
        len(samples),
        model.rmse,
    )


def async_register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_TRAIN_NOW):
        return

    async def _train(call: ServiceCall) -> None:
        zone_id = call.data.get("zone_id")
        days = call.data.get("days", 7)
        for coordinator in _iter_coordinators(hass):
            for cfg in coordinator._zone_configs():
                if zone_id and cfg[CONF_ZONE_ID] != zone_id:
                    continue
                await _bootstrap_zone(hass, coordinator, cfg, days)
            await coordinator.store.async_save()
            await coordinator.async_request_refresh()

    async def _reset(call: ServiceCall) -> None:
        zone_id = call.data.get("zone_id")
        for coordinator in _iter_coordinators(hass):
            for cfg in coordinator._zone_configs():
                if zone_id and cfg[CONF_ZONE_ID] != zone_id:
                    continue
                coordinator.reset_zone(cfg[CONF_ZONE_ID])
            await coordinator.store.async_save()
            await coordinator.async_request_refresh()

    async def _set_comfort(call: ServiceCall) -> None:
        zone_id = call.data["zone_id"]
        for coordinator in _iter_coordinators(hass):
            for key in (CONF_COMFORT_MIN, CONF_COMFORT_TARGET, CONF_COMFORT_MAX):
                if key in call.data:
                    coordinator.set_zone_override(zone_id, key, float(call.data[key]))
            await coordinator.async_request_refresh()

    hass.services.async_register(DOMAIN, SERVICE_TRAIN_NOW, _train, schema=TRAIN_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_RESET_MODEL, _reset, schema=RESET_SCHEMA)
    hass.services.async_register(
        DOMAIN, SERVICE_SET_COMFORT_PROFILE, _set_comfort, schema=COMFORT_SCHEMA
    )


def async_unregister_services(hass: HomeAssistant) -> None:
    for service in (SERVICE_TRAIN_NOW, SERVICE_RESET_MODEL, SERVICE_SET_COMFORT_PROFILE):
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
