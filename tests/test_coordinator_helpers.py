"""Tests for coordinator helpers that do not need a running integration."""

from custom_components.predictive_heating.const import COAST_HYSTERESIS
from custom_components.predictive_heating.coordinator import (
    recommended_setpoint,
    update_coasting,
)


def test_coasting_enters_only_above_max_plus_margin():
    assert update_coasting(False, [24.0 + COAST_HYSTERESIS, 24.6], 24.0) is True
    assert update_coasting(False, [24.0, 24.6], 24.0) is False
    assert update_coasting(False, [24.4, 24.6, 23.9], 24.0) is False


def test_coasting_stays_until_below_max_minus_margin():
    # Still above the exit threshold: stay coasting.
    assert update_coasting(True, [24.4, 24.0], 24.0) is True
    # Drops below the comfort maximum by the margin: resume control.
    assert update_coasting(True, [23.6, 24.0], 24.0) is False


def test_empty_horizon_is_not_coasting():
    assert update_coasting(False, [], 24.0) is False


def test_recommendation_never_holds_above_the_current_temperature():
    # Room above target with no heat demand: let it cool to the target.
    assert recommended_setpoint(23.2, 21.0, 22.5, 24.0, 0.0, False) == 22.5
    # Room below target: add the MPC heat demand.
    assert recommended_setpoint(22.0, 21.0, 22.5, 24.0, 0.3, False) == 22.3
    # Never above the comfort maximum or below the minimum.
    assert recommended_setpoint(22.0, 21.0, 22.5, 24.0, 5.0, False) == 24.0
    assert recommended_setpoint(23.0, 21.0, 22.5, 24.0, 0.0, True) == 21.0
