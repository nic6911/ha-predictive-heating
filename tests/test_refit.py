"""Tests for the stable batch-refit + offset-free bias forecast path."""

from types import SimpleNamespace
from datetime import timedelta

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
from custom_components.predictive_heating.models.rc_model_3r2c import RCModel3R2C


def _seed_buffer(true_params, n=200, seed=1):
    """A clean, well-excited synthetic history for one zone."""
    rng = np.random.default_rng(seed)
    model = RCModel3R2C(params=np.array(true_params, dtype=float))
    rows = []
    indoor = 21.0
    tw = indoor
    for _ in range(n):
        t_out = rng.uniform(-5, 10)
        sol = rng.uniform(0, 1)
        u = rng.uniform(0, 4)
        delta = (
            model.ka * (t_out - indoor)
            + model.k_aw * (tw - indoor)
            + model.ks * sol + model.kh * u + model.kg
        )
        nxt = indoor + delta + rng.normal(0, 0.01)
        tw = tw + model.k_wa * (indoor - tw)
        rows.append([indoor, t_out, sol, u, nxt])
        indoor = nxt
        if indoor < 12 or indoor > 30:
            indoor = 21.0
            tw = indoor
    return rows


async def _setup_coordinator(hass):
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

    true = [0.08, 0.30, 0.25, 0.20, 0.08, 0.02]
    core.buffer = _seed_buffer(true)
    core.model = None

    await coordinator.async_request_refresh()
    await hass.async_block_till_done()

    assert core.model is not None
    assert core.model.rmse is not None and core.model.rmse < 0.2
    result = coordinator.data["test_room"]
    assert result.forecast, "no forecast published"
    assert result.fit_rmse is not None and result.fit_rmse < 0.2


async def test_bias_is_applied_to_the_rollout(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant
):
    coordinator = await _setup_coordinator(hass)
    core = coordinator._zones["test_room"]
    core.model = RCModel3R2C(params=np.array([0.08, 0.30, 0.25, 0.20, 0.08, 0.02]))
    core.buffer = _seed_buffer([0.08, 0.30, 0.25, 0.20, 0.08, 0.02], n=60)

    base = core.model.simulate(21.0, np.full(6, 2.0), np.zeros(6), np.zeros(6))
    biased = RCModel3R2C(
        params=core.model.params, hourly_bias=np.full(24, 0.2)
    ).simulate(21.0, np.full(6, 2.0), np.zeros(6), np.zeros(6))
    diffs = np.diff(biased - base)
    assert np.all(diffs > 0)


async def test_bootstrap_sources_outdoor_from_weather(
    recorder_mock, enable_custom_integrations, hass: HomeAssistant, monkeypatch
):
    from custom_components.predictive_heating import services

    coordinator = await _setup_coordinator(hass)
    cfg = coordinator._global(CONF_ZONES)[0]

    end = dt_util.utcnow()
    start = end - timedelta(days=7)
    n = 7 * 24 * 4

    def _states():
        clim, wx = [], []
        indoor = 21.0
        for k in range(n):
            ts = start + timedelta(minutes=15 * k)
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

    assert "weather.home" in captured["entity_ids"]
    assert core.model is not None
    t_out_col = np.array([s[1] for s in core.buffer])
    assert t_out_col.std() > 1.0
    assert not np.allclose(t_out_col, 8.0)


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
    core.buffer = _seed_buffer([0.08, 0.30, 0.25, 0.20, 0.08, 0.02], n=cap + 500)

    await coordinator.async_request_refresh()
    await hass.async_block_till_done()

    assert len(core.buffer) <= cap
