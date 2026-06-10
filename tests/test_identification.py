"""Tests for RC parameter identification."""

import numpy as np

from custom_components.predictive_heating.models.identification import (
    RecursiveLeastSquares,
    batch_fit,
)
from custom_components.predictive_heating.models.rc_model import RCModel


def _make_samples(true_params, n=500, seed=0):
    rng = np.random.default_rng(seed)
    model = RCModel(params=np.array(true_params))
    samples = []
    indoor = 20.0
    for _ in range(n):
        t_out = rng.uniform(-5, 12)
        sol = rng.uniform(0, 1)
        u = rng.uniform(0, 4)
        nxt = model.step(indoor, t_out, sol, u) + rng.normal(0, 0.01)
        samples.append((indoor, t_out, sol, u, nxt))
        indoor = nxt
        if indoor < 10 or indoor > 30:
            indoor = 20.0
    return samples


def test_batch_fit_recovers_parameters():
    true = [0.85, 0.06, 0.25, 0.30, 0.05]
    samples = _make_samples(true)
    model = batch_fit(samples)
    assert np.allclose(model.params, true, atol=0.05)
    assert model.rmse is not None and model.rmse < 0.05


def test_rls_converges():
    true = [0.85, 0.06, 0.25, 0.30, 0.05]
    samples = _make_samples(true, n=800)
    rls = RecursiveLeastSquares()
    for (indoor, t_out, sol, u, nxt) in samples:
        rls.update(np.array([indoor, t_out, sol, u, 1.0]), nxt)
    assert np.allclose(rls.theta, true, atol=0.1)


def test_batch_fit_handles_insufficient_data():
    model = batch_fit([(20.0, 5.0, 0.0, 1.0, 20.1)])
    # Falls back to the prior without raising.
    assert model.params.shape == (5,)
