"""DataUpdateCoordinator: the predictive control loop.

Each cycle, for every enabled zone:

1. Read indoor temperature, setpoint and actuator limits.
2. Turn the previous observation into a training sample and update the online RLS
   estimator (and periodically persist the model).
3. Pull the weather + price forecast over the MPC horizon.
4. Solve the economic MPC to get the optimal near-term heat demand ``u0``.
5. Apply guardrails (comfort clamp, fit quality, manual-override hold, advisory mode)
   and -- when allowed -- write the resulting setpoint to the thermostat.
6. Publish per-zone results for the entities to display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import logging

import numpy as np

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from . import climate_io
from .const import (
    CONF_CLIMATE_ENTITY,
    CONF_CO2_ENTITY,
    CONF_COMFORT_MAX,
    CONF_COMFORT_MIN,
    CONF_COMFORT_TARGET,
    CONF_HORIZON_HOURS,
    CONF_IRRADIANCE_SENSOR,
    CONF_MODE,
    CONF_OUTDOOR_SENSOR,
    CONF_PRICE_ENTITY,
    CONF_PRICE_OPTIMIZE,
    CONF_STEP_MINUTES,
    CONF_TEMP_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_WEATHER_ENTITY,
    CONF_ZONE_ENABLED,
    CONF_ZONE_ID,
    CONF_ZONE_MODE,
    CONF_ZONES,
    DEFAULT_COMFORT_MAX,
    DEFAULT_COMFORT_MIN,
    DEFAULT_COMFORT_TARGET,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_MODE,
    DEFAULT_STEP_MINUTES,
    DEFAULT_UPDATE_INTERVAL,
    DISTURBANCE_DROP_MIN,
    DISTURBANCE_DROP_SIGMA,
    DISTURBANCE_HOLD,
    DOMAIN,
    FIT_RMSE_AUTONOMY_THRESHOLD,
    MODE_WEIGHTS,
    OUTLIER_ABS_CAP,
    OUTLIER_SIGMA,
    ZONE_MODE_ADVISORY,
)
from .control import mpc
from .forecast import (
    async_get_price_forecast,
    async_get_weather_forecast,
    solar_proxy,
)
from .models.identification import RecursiveLeastSquares
from .models.rc_model import RCModel
from .storage import ModelStore

_LOGGER = logging.getLogger(__name__)

# Persist learned models roughly every this many cycles.
SAVE_EVERY = 8


@dataclass
class ZoneResult:
    """Published per-zone state consumed by entities."""

    zone_id: str
    name: str
    enabled: bool
    indoor: float | None = None
    current_setpoint: float | None = None
    recommended_setpoint: float | None = None
    predicted_temperature: float | None = None
    has_authority: bool = True
    manual_override: bool = False
    applied: bool = False
    advisory: bool = False
    fit_rmse: float | None = None
    estimated_savings: float | None = None
    # True while a window/door-type disturbance is active (learning frozen,
    # last good setpoint held).
    disturbance: bool = False
    # Last one-step prediction error (measured now minus previously predicted), deg C.
    prediction_error: float | None = None
    # Predicted indoor-temperature trajectory over the MPC horizon, as a list of
    # {"datetime", "predicted", "free_float", "outdoor"} points (step 1..n).
    forecast: list[dict] = field(default_factory=list)
    horizon_hours: float | None = None
    step_minutes: int | None = None


@dataclass
class _ZoneCore:
    """Internal learning/runtime state for one zone."""

    config: dict
    rls: RecursiveLeastSquares
    runtime: climate_io.ZoneRuntime
    last_obs: dict | None = None  # {indoor, t_out, sol, u, predicted}
    extra: dict = field(default_factory=dict)
    # Disturbance state: timestamp until which learning is frozen and the last good
    # setpoint is held, plus the setpoint to hold.
    disturbance_until: object | None = None
    hold_setpoint: float | None = None


class PredictiveHeatingCoordinator(DataUpdateCoordinator):
    """Coordinates forecasting, learning and predictive setpoint control."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, store: ModelStore) -> None:
        self.entry = entry
        self.store = store
        self._zones: dict[str, _ZoneCore] = {}
        # Live, in-memory per-zone overrides set via entities (enable + comfort).
        self.overrides: dict[str, dict] = {}
        # Live global optimization-mode override set via the select entity.
        self.mode_override: str | None = None
        self._cycle = 0
        interval = self._global(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=interval),
        )

    # ----------------------------------------------------------------- config
    def _global(self, key: str, default=None):
        return {**self.entry.data, **self.entry.options}.get(key, default)

    def _zone_configs(self) -> list[dict]:
        merged = {**self.entry.data, **self.entry.options}
        return list(merged.get(CONF_ZONES, []))

    @property
    def horizon_steps(self) -> int:
        hours = self._global(CONF_HORIZON_HOURS, DEFAULT_HORIZON_HOURS)
        step = self._global(CONF_STEP_MINUTES, DEFAULT_STEP_MINUTES)
        return max(1, int(round(hours * 60 / step)))

    # ------------------------------------------------------------- bootstrap
    def init_zones(self) -> None:
        """Initialise learning state, loading any persisted models."""
        self._zones = {}
        for cfg in self._zone_configs():
            zone_id = cfg[CONF_ZONE_ID]
            model = self.store.get_model(zone_id)
            params = model.params if model else None
            rls = RecursiveLeastSquares(
                params=params,
                outlier_sigma=OUTLIER_SIGMA,
                outlier_abs_cap=OUTLIER_ABS_CAP,
            )
            self._zones[zone_id] = _ZoneCore(
                config=cfg,
                rls=rls,
                runtime=climate_io.ZoneRuntime(zone_id=zone_id),
            )

    def zone_param(self, zone_id: str, key: str, default):
        """Return a per-zone setting, preferring a live override over config."""
        override = self.overrides.get(zone_id, {})
        if key in override:
            return override[key]
        for cfg in self._zone_configs():
            if cfg.get(CONF_ZONE_ID) == zone_id:
                return cfg.get(key, default)
        return default

    def set_zone_override(self, zone_id: str, key: str, value) -> None:
        self.overrides.setdefault(zone_id, {})[key] = value

    # ---------------------------------------------------------------- inputs
    def _current_outdoor_solar(self, cfg: dict) -> tuple[float, float]:
        weather = self._global(CONF_WEATHER_ENTITY)
        t_out = 10.0
        sol = 0.0
        wstate = self.hass.states.get(weather) if weather else None
        if wstate is not None:
            try:
                t_out = float(wstate.attributes.get("temperature", t_out))
            except (TypeError, ValueError):
                pass
            sol = solar_proxy(
                wstate.attributes.get("cloud_coverage"),
                wstate.attributes.get("uv_index"),
            )
        outdoor_sensor = cfg.get(CONF_OUTDOOR_SENSOR)
        if outdoor_sensor:
            ostate = self.hass.states.get(outdoor_sensor)
            if ostate is not None:
                try:
                    t_out = float(ostate.state)
                except (TypeError, ValueError):
                    pass
        irr_sensor = cfg.get(CONF_IRRADIANCE_SENSOR)
        if irr_sensor:
            istate = self.hass.states.get(irr_sensor)
            if istate is not None:
                try:
                    sol = max(0.0, min(1.0, float(istate.state) / 1000.0))
                except (TypeError, ValueError):
                    pass
        return t_out, sol

    @property
    def active_mode(self) -> str:
        return self.mode_override or self._global(CONF_MODE, DEFAULT_MODE)

    def _weights(self) -> tuple[float, float]:
        mode = self.active_mode
        w_comfort, w_energy = MODE_WEIGHTS.get(mode, MODE_WEIGHTS[DEFAULT_MODE])
        if not self._global(CONF_PRICE_OPTIMIZE, False):
            w_energy = min(w_energy, 0.05)  # efficiency only, ignore price shape
        return w_comfort, w_energy

    # ----------------------------------------------------------------- cycle
    async def _async_update_data(self) -> dict[str, ZoneResult]:
        self._cycle += 1
        n = self.horizon_steps
        step_min = self._global(CONF_STEP_MINUTES, DEFAULT_STEP_MINUTES)
        weather = self._global(CONF_WEATHER_ENTITY)
        price_entity = self._global(CONF_PRICE_ENTITY)
        w_comfort, w_energy = self._weights()

        t_out_fc, sol_fc = (np.array([]), np.array([]))
        price_fc = np.array([])
        if weather:
            t_out_fc, sol_fc = await async_get_weather_forecast(
                self.hass, weather, n, step_min
            )
        price_fc = await async_get_price_forecast(
            self.hass, price_entity, n, step_min
        )

        results: dict[str, ZoneResult] = {}
        for zone_id, core in self._zones.items():
            results[zone_id] = await self._update_zone(
                core, t_out_fc, sol_fc, price_fc, w_comfort, w_energy
            )

        if self._cycle % SAVE_EVERY == 0:
            await self._persist()
        return results

    async def _update_zone(
        self,
        core: _ZoneCore,
        t_out_fc: np.ndarray,
        sol_fc: np.ndarray,
        price_fc: np.ndarray,
        w_comfort: float,
        w_energy: float,
    ) -> ZoneResult:
        cfg = core.config
        zone_id = cfg[CONF_ZONE_ID]
        climate_entity = cfg[CONF_CLIMATE_ENTITY]
        name = cfg.get("name", zone_id)
        enabled = bool(self.zone_param(zone_id, CONF_ZONE_ENABLED, True))

        climate_io.read_limits(self.hass, climate_entity, core.runtime)
        indoor = climate_io.read_indoor(
            self.hass, climate_entity, cfg.get(CONF_TEMP_SENSOR)
        )
        current_setpoint = climate_io.read_setpoint(self.hass, climate_entity)
        t_out_now, sol_now = self._current_outdoor_solar(cfg)
        now = dt_util.utcnow()

        result = ZoneResult(
            zone_id=zone_id,
            name=name,
            enabled=enabled,
            indoor=indoor,
            current_setpoint=current_setpoint,
        )

        # 1) Learn from the previous observation, with disturbance rejection.
        #
        # We work in temperature-difference form: the model predicts the change
        # ``indoor - prev_indoor`` from the regressor ``[t_out-indoor, sol, u, 1]``.
        # A window/door disturbance shows up as the room cooling far faster than the
        # model expects (a large *negative* residual). When that happens we freeze
        # learning and hold the last good setpoint for a recovery window, so a
        # transient never corrupts the learned dynamics.
        in_hold = core.disturbance_until is not None and now < core.disturbance_until
        disturbance = in_hold
        if core.last_obs is not None and indoor is not None:
            prev = core.last_obs
            phi_d = np.array(
                [
                    prev["t_out"] - prev["indoor"],
                    prev["sol"],
                    prev["u"],
                    1.0,
                ]
            )
            predicted_delta = core.rls.predict_delta(phi_d)
            measured_delta = indoor - prev["indoor"]
            residual = measured_delta - predicted_delta
            result.prediction_error = round(residual, 3)

            scale = core.rls._scale or 0.0
            drop_threshold = max(
                DISTURBANCE_DROP_MIN, DISTURBANCE_DROP_SIGMA * scale
            )
            if residual < -drop_threshold:
                # New (or continuing) disturbance: freeze learning, hold setpoint.
                disturbance = True
                core.disturbance_until = now + DISTURBANCE_HOLD
            elif in_hold and residual > -DISTURBANCE_DROP_MIN:
                # Temperature has recovered toward the prediction -> release hold.
                disturbance = False
                core.disturbance_until = None

            if not disturbance:
                # Innovation-gated RLS update (also rejects lone outliers internally).
                core.rls.update(phi_d, measured_delta)

        result.disturbance = disturbance

        # Record this observation for next cycle's learning / detection.
        if indoor is not None and current_setpoint is not None:
            core.last_obs = {
                "indoor": indoor,
                "t_out": t_out_now,
                "sol": sol_now,
                "u": RCModel.heat_demand(current_setpoint, indoor),
            }

        model = core.rls.to_model(step_minutes=self._global(CONF_STEP_MINUTES, DEFAULT_STEP_MINUTES))
        result.fit_rmse = model.rmse

        # We still compute predictions/recommendations when the zone is disabled
        # so the sensors stay meaningful; we simply never write the setpoint then
        # (handled by the advisory guardrail below).
        if indoor is None or len(t_out_fc) == 0:
            return result

        comfort_target = float(
            self.zone_param(zone_id, CONF_COMFORT_TARGET, DEFAULT_COMFORT_TARGET)
        )
        comfort_min = float(
            self.zone_param(zone_id, CONF_COMFORT_MIN, DEFAULT_COMFORT_MIN)
        )
        comfort_max = float(
            self.zone_param(zone_id, CONF_COMFORT_MAX, DEFAULT_COMFORT_MAX)
        )

        plan = mpc.solve(
            model=model,
            t0=indoor,
            t_out=t_out_fc,
            sol=sol_fc,
            price=price_fc,
            comfort_target=comfort_target,
            comfort_min=comfort_min,
            comfort_max=comfort_max,
            w_comfort=w_comfort,
            w_energy=w_energy,
        )

        # Honest summer / no-authority behaviour: if the room free-floats above the
        # comfort ceiling for the whole horizon, the heating has no downward control
        # authority -- never recommend heating an already-too-warm room.
        free_float = np.asarray(plan.free_float, dtype=float)
        summer_no_authority = len(free_float) > 0 and bool(
            np.all(free_float >= comfort_max)
        )
        has_authority = plan.has_authority and not summer_no_authority

        # Map the near-term heat demand back to a thermostat setpoint.
        recommended = indoor + plan.u0
        recommended = max(comfort_min, min(comfort_max, recommended))
        if not has_authority:
            recommended = comfort_min

        # During an active disturbance (e.g. an open window) freeze control and hold
        # the last good setpoint rather than chasing the transient.
        if disturbance and core.hold_setpoint is not None:
            recommended = core.hold_setpoint

        result.recommended_setpoint = climate_io.quantise(recommended, core.runtime)
        result.predicted_temperature = (
            float(plan.temperature[0]) if len(plan.temperature) else None
        )
        result.has_authority = has_authority

        # Publish the full predicted trajectory so it can be graphed over the horizon.
        step_min = int(self._global(CONF_STEP_MINUTES, DEFAULT_STEP_MINUTES))
        forecast: list[dict] = []
        for k in range(len(plan.temperature)):
            point = {
                "datetime": (now + timedelta(minutes=step_min * (k + 1))).isoformat(),
                "predicted": round(float(plan.temperature[k]), 2),
            }
            if k < len(plan.free_float):
                point["free_float"] = round(float(plan.free_float[k]), 2)
            if k < len(t_out_fc):
                point["outdoor"] = round(float(t_out_fc[k]), 2)
            forecast.append(point)
        result.forecast = forecast
        result.horizon_hours = self._global(CONF_HORIZON_HOURS, DEFAULT_HORIZON_HOURS)
        result.step_minutes = step_min

        # Estimated savings proxy: heat avoided versus holding the comfort target.
        baseline_u = max(0.0, comfort_target - indoor)
        result.estimated_savings = round(max(0.0, baseline_u - plan.u0), 2)

        # 2) Guardrails.
        manual = climate_io.detect_manual_override(
            self.hass, climate_entity, core.runtime
        )
        result.manual_override = manual
        zone_mode = cfg.get(CONF_ZONE_MODE)
        fit_ok = model.rmse is not None and model.rmse <= FIT_RMSE_AUTONOMY_THRESHOLD
        advisory = (
            not enabled
            or zone_mode == ZONE_MODE_ADVISORY
            or manual
            or not fit_ok
            or not self._master_enabled()
        )
        result.advisory = advisory

        if not advisory:
            result.applied = await climate_io.async_write_setpoint(
                self.hass, climate_entity, core.runtime, result.recommended_setpoint
            )
            # Remember the last good (undisturbed) setpoint so we can hold it if a
            # disturbance starts on a later cycle.
            if not disturbance:
                core.hold_setpoint = result.recommended_setpoint
        return result

    # --------------------------------------------------------------- helpers
    def _master_enabled(self) -> bool:
        return bool(self.hass.data.get(DOMAIN, {}).get("master_enabled", True))

    async def _persist(self) -> None:
        for zone_id, core in self._zones.items():
            self.store.set_model(zone_id, core.rls.to_model())
        await self.store.async_save()

    async def async_train_zone_from_history(self, zone_id: str) -> None:
        """Hook for batch bootstrap from recorder history (see services)."""
        # Implemented in services.py via recorder; kept here for API symmetry.
        return None

    def reset_zone(self, zone_id: str) -> None:
        core = self._zones.get(zone_id)
        if core is not None:
            core.rls = RecursiveLeastSquares(
                outlier_sigma=OUTLIER_SIGMA, outlier_abs_cap=OUTLIER_ABS_CAP
            )
            core.last_obs = None
            core.disturbance_until = None
            core.hold_setpoint = None
            self.store.clear_zone(zone_id)
