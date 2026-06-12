"""Tests for the stable batch-refit + offset-free bias forecast path.

The integration forecasts and controls from a periodically re-identified robust
batch model (seeded/grown from a rolling buffer) plus an online offset-free bias --
NOT from the fast online RLS, whose one-step-optimal parameters drift into
combinations that explode in multi-step open-loop rollout.
"""

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
from pytest_homeassistant_custom_component.common import MockConfigEntry

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.predictive_heating.const import (
    CONF_CLIMATE_ENTITY,
    CONF_COMFORT_MAX,
    CONF_COMFORT_MIN,
    CONF_COMFORT_TARGET,
    CONF_UPDATE_INTERVAL,
    CONF_WEATHER_ENTITY,
    CONF_ZONE_ENABLED,
    CONF_ZONE_ID,
    CONF_ZONES,
    DOMAIN,
)
from custom_components.predictive_heating.models.rc_model import RCModel


def _seed_buffer(true_params, n=200, seed=1):
    """A clean, well-excited synthetic history for one zone."""
    rng = np.random.default_rng(seed)
    model = RCModel(params=np.array(true_params, dtype=float))
    rows = []
    indoor = 21.0
    for _ in range(n):
        t_out = rng.uniform(-5, 10)
        sol = rng.uniform(0, 1)
        u = rng.uniform(0, 4)
        nxt = model.step(indoor, t_out, sol, u) + rng.normal(0, 0.01)
        rows.append([indoor, t_out, sol, u, nxt])
        indoor = nxt
        if indoor < 12 or indoor > 30:
            indoor = 21.0
    return rows


async def _setup_coordinator(hass):
    # Capture autonomous setpoint writes so the control loop doesn't raise.
    hass.services.async_register("climate", "set_temperature", lambda call: None)
    hass.states.async_set(
        "climate.test_room",
        "off",
        {
            "current_temperature": 21.0,
            "temperature": 21.5,
            "min_temp": 5,
            "max_temp": 30,
            "target_temp_step": 0.1,
            "supported_features": 1,
        },
    )
    hass.states.async_set(
        "weather.home",
        "cloudy",
        {"temperature": 2.0, "cloud_coverage": 50, "uv_index": 1},
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_WEATHER_ENTITY: "weather.home"},
        options={
            # 6 h update interval -> refit cadence collapses to every cycle, so the
            # test triggers a refit deterministically on the next manual refresh.
            CONF_UPDATE_INTERVAL: 360,
            CONF_ZONES: [
                {
                    CONF_ZONE_ID: "test_room",
                    CONF_CLIMATE_ENTITY: "climate.test_room",
                    "name": "Test Room",
                    CONF_COMFORT_MIN: 19.0,
                    CONF_COMFORT_TARGET: 21.0,
                    CONF_COMFORT_MAX: 23.0,
                    CONF_ZONE_ENABLED: True,
                }
            ],
        },
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return hass.data[DOMAIN][entry.entry_id]["coordinator"]


async def test_refit_builds_stable_model_and_forecasts_from_it(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    coordinator = await _setup_coordinator(hass)
    core = coordinator._zones["test_room"]

    # Seed a clean buffer and force the next cycle to re-identify from it.
    true = [0.08, 0.30, 0.25, 0.25]
    core.buffer = _seed_buffer(true)
    core.model = None
    core.bias = 0.0

    await coordinator.async_request_refresh()
    await hass.async_block_till_done()

    # A stable batch model must now drive the forecast (not the online RLS).
    assert core.model is not None
    assert np.allclose(core.model.params, true, atol=0.05)
    # The published forecast must reflect that model + its bias.
    result = coordinator.data["test_room"]
    assert result.forecast, "no forecast published"
    assert result.fit_rmse is not None and result.fit_rmse < 0.2


async def test_bias_is_applied_to_the_rollout(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    """A non-zero offset-free bias must shift every simulated step."""
    coordinator = await _setup_coordinator(hass)
    core = coordinator._zones["test_room"]
    core.model = RCModel(params=np.array([0.08, 0.30, 0.25, 0.25]))
    core.buffer = _seed_buffer([0.08, 0.30, 0.25, 0.25], n=60)

    base = core.model.simulate(21.0, np.full(6, 2.0), np.zeros(6), np.zeros(6))
    biased = core.model.__class__(
        params=core.model.params, bias=0.2
    ).simulate(21.0, np.full(6, 2.0), np.zeros(6), np.zeros(6))
    # Each step accumulates the per-step bias, so the gap grows monotonically.
    diffs = np.diff(biased - base)
    assert np.all(diffs > 0)


async def test_bootstrap_sources_outdoor_from_weather(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant, monkeypatch
):
    """With no per-zone outdoor sensor, the bootstrap must reconstruct outdoor
    from the weather entity's recorded ``temperature`` -- NOT a constant
    fallback, which would make ka/kg unidentifiable and the horizon run away."""
    from custom_components.predictive_heating import services

    coordinator = await _setup_coordinator(hass)
    cfg = coordinator._global(CONF_ZONES)[0]

    end = dt_util.utcnow()
    start = end - timedelta(days=7)
    n = 7 * 24 * 4  # 15-min cadence over the window

    def _states():
        clim, wx = [], []
        indoor = 21.0
        for k in range(n):
            ts = start + timedelta(minutes=15 * k)
            # Varying outdoor drives a varying indoor so the fit is identifiable.
            t_out = 5.0 + 8.0 * np.sin(k / 20.0)
            indoor += 0.02 * (t_out - indoor) + 0.25 * max(0.0, 21.5 - indoor)
            clim.append(SimpleNamespace(
                last_changed=ts, state="off",
                attributes={"current_temperature": round(indoor, 3),
                            "temperature": 21.5}))
            wx.append(SimpleNamespace(
                last_changed=ts, state="cloudy",
                attributes={"temperature": round(t_out, 3)}))
        return {"climate.test_room": clim, "weather.home": wx}

    captured = {}

    def fake_get_significant_states(hass_, s, e, entity_ids, *a, **kw):
        captured["entity_ids"] = list(entity_ids)
        st = _states()
        return {eid: st.get(eid, []) for eid in entity_ids}

    from homeassistant.components.recorder import history as recorder_history

    monkeypatch.setattr(
        recorder_history, "get_significant_states", fake_get_significant_states
    )

    core = coordinator._zones["test_room"]
    core.model = None
    core.buffer = []
    await services._bootstrap_zone(hass, coordinator, cfg, days=7)

    # The weather entity must have been queried for outdoor history.
    assert "weather.home" in captured["entity_ids"]
    assert core.model is not None
    # Outdoor came from the weather entity (varies), not the constant fallback.
    t_out_col = np.array([s[1] for s in core.buffer])
    assert t_out_col.std() > 1.0
    assert not np.allclose(t_out_col, 8.0)
    # ka stays identifiable/low and the open-loop steady state stays physical
    # (no runaway toward 30 C) -- the regression this fix targets.
    ka, _ks, _kh, kg = core.model.params
    bias = float(np.clip(core.model.bias, -0.5, 0.5))
    steady_state = 14.0 + (kg + bias) / ka
    assert steady_state < 28.0


async def test_buffer_is_capped(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    from custom_components.predictive_heating.const import (
        BUFFER_DAYS,
        DEFAULT_STEP_MINUTES,
    )

    coordinator = await _setup_coordinator(hass)
    core = coordinator._zones["test_room"]
    cap = int(BUFFER_DAYS * 24 * 60 / DEFAULT_STEP_MINUTES)
    core.buffer = _seed_buffer([0.08, 0.30, 0.25, 0.25], n=cap + 500)

    await coordinator.async_request_refresh()
    await hass.async_block_till_done()

    assert len(core.buffer) <= cap
