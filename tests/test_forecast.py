"""Tests for the clear-sky solar reconstruction used to bootstrap ``ks``."""

from datetime import datetime, timezone

from custom_components.predictive_heating.forecast import clear_sky_index


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
