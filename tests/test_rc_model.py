"""Tests for the RC thermal model."""

import numpy as np

from custom_components.predictive_heating.models.rc_model import RCModel


def test_prediction_matrices_match_simulation():
    """T_free + G @ u must equal a full nonlinear-free simulation with that u."""
    model = RCModel(params=np.array([0.9, 0.05, 0.3, 0.2, 0.1]))
    n = 12
    t_out = np.linspace(0, 5, n)
    sol = np.linspace(0, 0.5, n)
    u = np.abs(np.sin(np.linspace(0, 3, n)))

    t_free, g = model.prediction_matrices(20.0, t_out, sol)
    affine = t_free + g @ u

    sim = model.simulate(20.0, t_out, sol, u)[1:]
    assert np.allclose(affine, sim, atol=1e-9)


def test_heat_demand_is_nonnegative():
    assert RCModel.heat_demand(21.0, 22.0) == 0.0
    assert RCModel.heat_demand(22.0, 20.0) == 2.0


def test_roundtrip_serialisation():
    model = RCModel(params=np.array([0.8, 0.1, 0.2, 0.3, -0.1]), rmse=0.5, n_samples=42)
    restored = RCModel.from_dict(model.as_dict())
    assert np.allclose(restored.params, model.params)
    assert restored.rmse == 0.5
    assert restored.n_samples == 42
