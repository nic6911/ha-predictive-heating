"""Tests for auto-adaptive parameter identification."""

import numpy as np

from custom_components.predictive_heating.models.identification import (
    _compute_effective_ka,
    _analyze_excitation,
    batch_fit_auto,
)
from custom_components.predictive_heating.models.rc_model_3r2c import RCModel3R2C


def _make_samples_well_excited(n=500, seed=0):
    """Synthetic data from a well-excited room with full 3R2C dynamics."""
    rng = np.random.default_rng(seed)
    model = RCModel3R2C(params=np.array([0.08, 0.20, 0.25, 0.20, 0.08, 0.02]))
    samples = []
    indoor = 20.0
    tw = indoor
    for _ in range(n):
        t_out = rng.uniform(-5, 12)
        sol = rng.uniform(0, 1)
        u = rng.uniform(0.0, 4.0)
        # Ground-truth 3R2C step.
        delta = (
            model.ka * (t_out - indoor)
            + model.k_aw * (tw - indoor)
            + model.ks * sol + model.kh * u + model.kg
        )
        nxt = indoor + delta + rng.normal(0, 0.01)
        tw = tw + model.k_wa * (indoor - tw)
        samples.append((indoor, t_out, sol, u, nxt))
        indoor = nxt
        if indoor < 10 or indoor > 30:
            indoor = 20.0
            tw = indoor
    return samples


def _make_samples_with_vent(n=700, seed=3, k_vent=0.06):
    """Synthetic data whose true model includes a ventilation disturbance."""
    rng = np.random.default_rng(seed)
    model = RCModel3R2C(params=np.array([0.08, 0.20, 0.25, 0.20, 0.08, 0.02, k_vent]))
    samples = []
    indoor = 21.5
    tw = indoor
    for i in range(n):
        t_out = rng.uniform(-5, 10)
        sol = rng.uniform(0, 1)
        u = rng.uniform(0.0, 3.0)
        vent = 17.0 + 2.5 * np.sin(i / 27.0) + rng.normal(0, 0.4)
        delta = (
            model.ka * (t_out - indoor)
            + model.k_aw * (tw - indoor)
            + model.ks * sol + model.kh * u + model.kg
            + k_vent * (vent - indoor)
        )
        nxt = indoor + delta + rng.normal(0, 0.01)
        tw = tw + model.k_wa * (indoor - tw)
        samples.append((indoor, t_out, sol, u, nxt, i % 24, vent))
        indoor = nxt
        if indoor < 12 or indoor > 30:
            indoor = 21.5
            tw = indoor
    return samples


def test_batch_fit_auto_recovers_vent_coefficient():
    """A varying ventilation input should be identified with the right sign."""
    fitted = batch_fit_auto(_make_samples_with_vent())
    assert 0.02 < fitted.k_vent < 0.12
    assert fitted.rmse is not None and fitted.rmse < 0.15


def test_batch_fit_auto_drops_unexcited_vent_input():
    """A constant ventilation effect must not be fitted (it is not identifiable)."""
    samples = [
        (row[0], row[1], row[2], row[3], row[4], 0, row[0] + 4.0)
        for row in _make_samples_well_excited()
    ]
    fitted = batch_fit_auto(samples)
    assert fitted.k_vent == 0.0


def test_batch_fit_auto_recovers_params_well_excited():
    """Well-excited data should recover ka, ks, kh, kg near ground truth."""
    samples = _make_samples_well_excited()
    model = batch_fit_auto(samples)
    assert model.params.shape == (7,)
    assert model.model_type == "auto"
    assert model.k_vent == 0.0
    # ka should be within reasonable range of the true 0.08.
    assert 0.04 < model.params[0] < 0.20
    assert model.rmse is not None and model.rmse < 0.10


def test_batch_fit_auto_handles_insufficient_data():
    """With only 1 sample, must return a default model without raising."""
    model = batch_fit_auto([(20.0, 5.0, 0.0, 1.0, 20.1)])
    assert model.params.shape == (7,)
    assert model.model_type == "auto"


def test_compute_effective_ka_low_variance():
    """Stable indoor (well-insulated) must yield ka near lower bound."""
    rng = np.random.default_rng(42)
    indoor = 25.0 + rng.normal(0, 0.1, 500)  # ±0.1°C
    t_out = 15.0 + 5.0 * np.sin(np.linspace(0, 24 * np.pi, 500))
    ka = _compute_effective_ka(indoor, t_out, step_minutes=30)
    assert ka < 0.03, f"Expected ka near 0, got {ka}"


def test_compute_effective_ka_high_variance():
    """Well-coupled room (indoor follows outdoor) must yield larger ka."""
    rng = np.random.default_rng(42)
    indoor = 15.0 + 4.0 * np.sin(np.linspace(0, 24 * np.pi, 500))
    indoor += rng.normal(0, 0.1, 500)
    t_out = 15.0 + 5.0 * np.sin(np.linspace(0, 24 * np.pi, 500))
    ka = _compute_effective_ka(indoor, t_out, step_minutes=30)
    assert ka > 0.05


def test_analyze_excitation_stable():
    indoor = np.full(100, 25.0) + np.random.default_rng(0).normal(0, 0.05, 100)
    t_out = 10.0 + 5.0 * np.sin(np.linspace(0, 24 * np.pi, 100))
    sol = np.zeros(100)
    u = np.zeros(100)
    analysis = _analyze_excitation(indoor, t_out, sol, u)
    assert analysis["coupling_ratio"] < 0.10


def test_batch_fit_auto_stable_room_returns_reasonable_model():
    """A very stable room (indoor range < 0.5°C) must return a model
    without fitting unidentifiable params -- and not blow up."""
    rng = np.random.default_rng(7)
    model = RCModel3R2C(params=np.array([0.006, 0.15, 0.20, 0.04, 0.02, 0.005]))
    samples = []
    indoor = 25.0
    tw = indoor
    for _ in range(300):
        t_out = rng.uniform(11, 19) + 0.5 * np.sin(_ / 4.0)
        sol = rng.uniform(0, 0.3)
        u = rng.uniform(0, 0.5)
        delta = (
            model.ka * (t_out - indoor)
            + model.k_aw * (tw - indoor)
            + model.ks * sol + model.kh * u + model.kg
        )
        nxt = indoor + delta + rng.normal(0, 0.01)
        tw = tw + model.k_wa * (indoor - tw)
        samples.append((indoor, t_out, sol, u, nxt))
        indoor = nxt
        if indoor < 24 or indoor > 26:
            indoor = 25.0
            tw = indoor
    fitted = batch_fit_auto(samples)
    # ka should stay very low (stable room).
    assert fitted.params[0] < 0.03
    assert fitted.rmse is not None and fitted.rmse < 0.10


def test_batch_fit_auto_returns_hourly_bias():
    """Hourly bias must be returned with the correct shape."""
    samples = _make_samples_well_excited(n=500)
    model = batch_fit_auto(samples)
    assert hasattr(model, 'hourly_bias')
    assert model.hourly_bias.shape == (24,)


def test_batch_fit_auto_returns_tw():
    """Final wall temperature must be returned."""
    samples = _make_samples_well_excited(n=500)
    model = batch_fit_auto(samples)
    assert hasattr(model, 'tw')
    assert isinstance(model.tw, float)
