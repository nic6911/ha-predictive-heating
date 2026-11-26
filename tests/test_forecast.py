"""Tests for forecast helpers (clear-sky solar and outdoor anchoring)."""

from datetime import datetime, timezone

import numpy as np

from custom_components.predictive_heating.const import ANCHOR_HOURS
from custom_components.predictive_heating.forecast import (
    anchor_outdoor,
    clear_sky_index,
)


def _utc(month, day, hour):
    return datetime(2026, month, day, hour, 0, tzinfo=timezone.utc)


def test_clear_sky_zero_at_night():
    # Local midnight (lon ~10E => UTC ~22:00) the sun is well below the horizon.
    assert clear_sky_index(_utc(6, 21, 22), latitude=56.0, longitude=10.0) == 0.0


def test_clear_sky_positive_at_midday():
    assert clear_sky_index(_utc(6, 21, 11), latitude=56.0, longitude=10.0) > 0.3


def test_clear_sky_summer_higher_than_winter():
    summer = clear_sky_index(_utc(6, 21, 11), latitude=56.0, longitude=10.0)
    winter = clear_sky_index(_utc(12, 21, 11), latitude=56.0, longitude=10.0)
    assert summer > winter > 0.0


def test_clear_sky_never_negative():
    for hour in range(24):
        assert clear_sky_index(_utc(12, 21, hour), 70.0, 10.0) >= 0.0


def test_anchor_starts_at_measurement_and_decays():
    forecast = np.full(24, 2.0)
    anchored = anchor_outdoor(forecast, anchor_temp=7.0, step_minutes=60)
    assert anchored[0] == 7.0
    # Offset is gone once ANCHOR_HOURS have elapsed.
    assert anchored[int(ANCHOR_HOURS)] == 2.0
    assert np.all(np.diff(anchored[: int(ANCHOR_HOURS) + 1]) <= 1e-9)


def test_anchor_keeps_forecast_shape_after_decay():
    forecast = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    anchored = anchor_outdoor(forecast, anchor_temp=2.0, step_minutes=30)
    assert anchored[0] == 2.0
    assert np.allclose(anchored[2:], forecast[2:])


def test_anchor_none_is_unchanged():
    forecast = np.array([2.0, 1.5, 1.0])
    assert np.allclose(anchor_outdoor(forecast, None, 30), forecast)
