"""Tests for the 3R2C thermal model (RCModel3R2C)."""

import numpy as np

from custom_components.predictive_heating.models.rc_model_3r2c import RCModel3R2C


def test_prediction_matrices_match_simulation():
    """T_free + G @ u must equal a full simulation with that u."""
    model = RCModel3R2C(params=np.array([0.08, 0.20, 0.25, 0.20, 0.08, 0.02]))
    n = 12
    t_out = np.linspace(0, 5, n)
    sol = np.linspace(0, 0.5, n)
    u = np.abs(np.sin(np.linspace(0, 3, n)))

    t_free, g = model.prediction_matrices(20.0, t_out, sol)
    affine = t_free + g @ u

    sim = model.simulate(20.0, t_out, sol, u)[1:]
    assert np.allclose(affine, sim, atol=1e-9)


def test_heat_demand_is_nonnegative():
    assert RCModel3R2C.heat_demand(21.0, 22.0) == 0.0
    assert RCModel3R2C.heat_demand(22.0, 20.0) == 2.0


def test_roundtrip_serialisation():
    params = np.array([0.06, 0.15, 0.20, 0.18, 0.06, 0.015])
    model = RCModel3R2C(params=params, rmse=0.5, n_samples=42, tw=22.5)
    restored = RCModel3R2C.from_dict(model.as_dict())
    assert np.allclose(restored.params, model.params)
    assert restored.rmse == 0.5
    assert restored.n_samples == 42
    assert restored.tw == 22.5
    assert restored.model_type == "auto"


def test_from_dict_preserves_hourly_bias():
    params = np.array([0.06, 0.15, 0.20, 0.18, 0.06, 0.015])
    hb = np.arange(24, dtype=float) * 0.01
    model = RCModel3R2C(params=params, hourly_bias=hb)
    restored = RCModel3R2C.from_dict(model.as_dict())
    assert np.allclose(restored.hourly_bias, hb)


def test_predict_delta():
    """predict_delta must produce the same value as step delta (without bias)."""
    model = RCModel3R2C(params=np.array([0.08, 0.20, 0.25, 0.20, 0.08, 0.02]))
    predicted = model.predict_delta(20.0, 5.0, 0.3, 1.0, tw=20.5)
    expected = (
        0.08 * (5.0 - 20.0)
        + 0.08 * (20.5 - 20.0)
        + 0.20 * 0.3 + 0.25 * 1.0 + 0.20
    )
    assert np.isclose(predicted, expected)


def test_internal_gains_float_above_outdoor():
    """With heating off, the room must settle above outdoor temperature."""
    model = RCModel3R2C(params=np.array([0.08, 0.0, 0.25, 0.20, 0.08, 0.02]))
    t_out = np.full(400, 10.0)
    sol = np.zeros(400)
    traj = model.free_float(15.0, t_out, sol)
    assert traj[-1] > 12.0


def test_model_type_default_auto():
    model = RCModel3R2C()
    assert model.model_type == "auto"
