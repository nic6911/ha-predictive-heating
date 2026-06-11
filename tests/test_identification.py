"""Tests for RC parameter identification (difference form)."""

import numpy as np

from custom_components.predictive_heating.models.identification import (
    RecursiveLeastSquares,
    batch_fit,
)
from custom_components.predictive_heating.models.rc_model import RCModel


def _make_samples(true_params, n=500, seed=0, u_low=0.0, u_high=4.0):
    rng = np.random.default_rng(seed)
    model = RCModel(params=np.array(true_params))
    samples = []
    indoor = 20.0
    for _ in range(n):
        t_out = rng.uniform(-5, 12)
        sol = rng.uniform(0, 1)
        u = rng.uniform(u_low, u_high)
        nxt = model.step(indoor, t_out, sol, u) + rng.normal(0, 0.01)
        samples.append((indoor, t_out, sol, u, nxt))
        indoor = nxt
        if indoor < 10 or indoor > 30:
            indoor = 20.0
    return samples


def _phi_d(indoor, t_out, sol, u):
    return np.array([t_out - indoor, sol, u, 1.0])


def test_batch_fit_recovers_parameters():
    true = [0.08, 0.30, 0.25, 0.25]  # ka, ks, kh, kg
    samples = _make_samples(true)
    model = batch_fit(samples)
    assert np.allclose(model.params, true, atol=0.05)
    assert model.rmse is not None and model.rmse < 0.05


def test_rls_converges():
    true = [0.08, 0.30, 0.25, 0.25]
    samples = _make_samples(true, n=1500)
    rls = RecursiveLeastSquares()
    for (indoor, t_out, sol, u, nxt) in samples:
        rls.update(_phi_d(indoor, t_out, sol, u), nxt - indoor)
    assert np.allclose(rls.theta, true, atol=0.05)


def test_batch_fit_handles_insufficient_data():
    model = batch_fit([(20.0, 5.0, 0.0, 1.0, 20.1)])
    # Falls back to the prior (4 params) without raising.
    assert model.params.shape == (4,)


def test_excitation_guard_holds_heating_gain_without_excitation():
    """With essentially no heating in the data, kh must stay at the prior."""
    from custom_components.predictive_heating.models.rc_model import DEFAULT_PARAMS

    # u is always ~0 (summer): kh is unidentifiable and must not be learned.
    samples = _make_samples([0.08, 0.30, 0.25, 0.25], u_low=0.0, u_high=0.0)
    model = batch_fit(samples)
    assert np.isclose(model.params[2], DEFAULT_PARAMS[2])


def test_rls_rejects_outlier_sample():
    """A gross one-step outlier must be rejected and leave parameters intact."""
    true = [0.08, 0.30, 0.25, 0.25]
    samples = _make_samples(true, n=400)
    rls = RecursiveLeastSquares()
    for (indoor, t_out, sol, u, nxt) in samples:
        rls.update(_phi_d(indoor, t_out, sol, u), nxt - indoor)
    theta_before = rls.theta.copy()
    # Inject a huge negative jump (a window flung open): residual far beyond cap.
    accepted = rls.update(_phi_d(20.0, 5.0, 0.0, 1.0), -6.0)
    assert accepted is False
    assert np.allclose(rls.theta, theta_before)
