"""Tests for the economic MPC."""

import numpy as np

from custom_components.predictive_heating.control import mpc
from custom_components.predictive_heating.models.rc_model import RCModel


def _model():
    # Stable room with meaningful heating gain.
    return RCModel(params=np.array([0.92, 0.06, 0.4, 0.35, 0.0]))


def test_box_constraints_respected():
    model = _model()
    n = 24
    res = mpc.solve(
        model,
        t0=18.0,
        t_out=np.full(n, 0.0),
        sol=np.zeros(n),
        price=np.ones(n),
        comfort_target=21.0,
        comfort_min=20.0,
        comfort_max=23.0,
        w_comfort=3.0,
        w_energy=1.0,
        u_max=6.0,
    )
    assert np.all(res.u >= -1e-9)
    assert np.all(res.u <= 6.0 + 1e-6)


def test_cold_room_gets_heating_and_authority():
    model = _model()
    n = 24
    res = mpc.solve(
        model,
        t0=16.0,
        t_out=np.full(n, -2.0),
        sol=np.zeros(n),
        price=np.ones(n),
        comfort_target=21.0,
        comfort_min=20.0,
        comfort_max=23.0,
        w_comfort=5.0,
        w_energy=0.1,
    )
    assert res.u0 > 0.0
    assert res.has_authority
    # Heating should pull temperature up over the horizon.
    assert res.temperature[-1] > 18.0


def test_free_heat_no_authority():
    """If the room free-floats above target, the optimiser should coast."""
    model = _model()
    n = 24
    res = mpc.solve(
        model,
        t0=24.0,
        t_out=np.full(n, 26.0),  # warm outside -> room stays warm with no heat
        sol=np.ones(n),
        price=np.ones(n),
        comfort_target=21.0,
        comfort_min=20.0,
        comfort_max=23.0,
        w_comfort=3.0,
        w_energy=1.0,
    )
    assert res.u0 < 0.05
    assert not res.has_authority


def test_price_shifts_heating_to_cheap_hours():
    model = _model()
    n = 12
    price = np.ones(n)
    price[:6] = 5.0  # expensive early
    price[6:] = 0.2  # cheap later
    res = mpc.solve(
        model,
        t0=20.5,
        t_out=np.full(n, 5.0),
        sol=np.zeros(n),
        price=price,
        comfort_target=21.0,
        comfort_min=19.0,
        comfort_max=23.0,
        w_comfort=1.0,
        w_energy=4.0,
    )
    # More heat should be used in the cheap window than the expensive one.
    assert res.u[6:].sum() >= res.u[:6].sum()
