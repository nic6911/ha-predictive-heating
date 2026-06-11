"""Tests for the RC thermal model."""

import numpy as np

from custom_components.predictive_heating.models.rc_model import RCModel


def test_prediction_matrices_match_simulation():
    """T_free + G @ u must equal a full nonlinear-free simulation with that u."""
    # params = [ka, ks, kh, kg]
    model = RCModel(params=np.array([0.08, 0.30, 0.25, 0.20]))
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
    model = RCModel(params=np.array([0.08, 0.30, 0.25, 0.20]), rmse=0.5, n_samples=42)
    restored = RCModel.from_dict(model.as_dict())
    assert np.allclose(restored.params, model.params)
    assert restored.rmse == 0.5
    assert restored.n_samples == 42


def test_legacy_5param_model_resets_to_prior():
    """A persisted 5-parameter (legacy) model must be reset to the new prior."""
    from custom_components.predictive_heating.models.rc_model import DEFAULT_PARAMS

    legacy = {"params": [0.9, 0.05, 0.3, 0.2, 0.0], "rmse": 8.0, "n_samples": 10}
    restored = RCModel.from_dict(legacy)
    assert restored.params.shape == (4,)
    assert np.allclose(restored.params, DEFAULT_PARAMS)


def test_internal_gains_float_above_outdoor():
    """With heating off, the room must settle several degrees above outdoor."""
    model = RCModel(params=np.array([0.08, 0.0, 0.25, 0.25]))
    t_out = np.full(400, 10.0)
    sol = np.zeros(400)
    traj = model.free_float(15.0, t_out, sol)
    # Steady state ~ T_out + kg/ka = 10 + 0.25/0.08 ~ 13.1 C, clearly above outdoor.
    assert traj[-1] > 12.0
