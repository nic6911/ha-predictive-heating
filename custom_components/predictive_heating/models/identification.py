"""Parameter identification for the 3R2C thermal model (difference form).

The single entry point is :func:`batch_fit_auto`, which adaptively identifies
a two-node model using data-informed priors and excitation-aware column
selection.
"""

from __future__ import annotations

import numpy as np

from ..const import N_HOURS, N_PARAMS_3R2C
from .rc_model_3r2c import RCModel3R2C

# Below this standard deviation of the heating proxy ``u`` (deg C) we consider the
# data un-excited for heating and refuse to identify ``kh`` (hold it at the prior).
EXCITATION_U_STD = 0.15

# Below this standard deviation of the solar proxy (0..1) the data carries no usable
# solar variation, so ``ks`` is not identifiable. In that case we hold ``ks`` at the
# **prior** (not zero), so the model still has a reasonable solar estimate on
# low-variance training windows (e.g. an overcast summer week).
EXCITATION_SOL_STD = 0.05

# Huber threshold (deg C) for robust IRLS residual weighting.
HUBER_DELTA = 0.4


def _weighted_ridge(
    phi: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float, prior: np.ndarray
) -> np.ndarray:
    """Solve a weighted ridge regression that shrinks toward ``prior``."""
    wphi = phi * w[:, None]
    reg = ridge * np.eye(phi.shape[1])
    amat = phi.T @ wphi + reg
    bvec = phi.T @ (w * y) + ridge * prior
    try:
        return np.linalg.solve(amat, bvec)
    except np.linalg.LinAlgError:
        theta, *_ = np.linalg.lstsq(amat, bvec, rcond=None)
        return theta


# ---------------------------------------------------------------------------
# 3R2C (two-node) identification
# ---------------------------------------------------------------------------
# Bounds for the 6-parameter 3R2C model.
PARAM_LOWER_3R2C = np.array([0.005, 0.0, 0.0, -2.0, 0.001, 0.001], dtype=float)
PARAM_UPPER_3R2C = np.array([0.50, 2.0, 1.0, 4.0, 0.50, 0.50], dtype=float)

# Default wall-to-air heat-capacity ratio (C_w / C_a).  k_wa = k_aw / C_RATIO.
# Heavier slabs -> larger ratio; 4.0 is reasonable for a concrete floor.
C_RATIO_3R2C = 4.0


def _forward_filter_tw(
    samples: list[tuple], k_wa: float = 0.02
) -> np.ndarray:
    """Forward-filter an estimate of the wall temperature from data.

    Returns an array of T_w values aligned with the start of each sample
    transition.  T_w is updated as::

        T_w[k+1] = T_w[k] + k_wa * (T_a[k] - T_w[k])
    """
    n = len(samples)
    tw = np.empty(n + 1, dtype=float)
    if n == 0:
        return tw
    tw[0] = samples[0][0]
    for i in range(n):
        ta = samples[i][0]
        tw[i + 1] = tw[i] + k_wa * (ta - tw[i])
    return tw


# ---------------------------------------------------------------------------
# Auto (adaptive) identification
# ---------------------------------------------------------------------------

def _compute_effective_ka(
    indoor: np.ndarray, t_out: np.ndarray, step_minutes: float = 30.0
) -> float:
    """Compute effective outdoor coupling ka from the data's attenuation ratio.

    For a first-order thermal system driven by diurnal outdoor temperature
    variation, the indoor temperature attenuates the outdoor signal by::

        A = indoor_std / outdoor_std

    where τ is the building time constant and T = 24 h is the diurnal period.
    Solving for τ gives::

        τ = T / (2π) * sqrt(1/A² - 1)   (hours)

    then ka = step_hours / τ.
    """
    indoor_std = float(np.std(indoor))
    t_out_std = float(np.std(t_out))

    if t_out_std < 0.1 or indoor_std < 0.01:
        return 0.02

    A = indoor_std / t_out_std
    if A >= 0.99:
        return 0.30

    T_hours = 24.0
    tau_hours = T_hours / (2.0 * np.pi) * np.sqrt(max(0.0, 1.0 / (A * A) - 1.0))
    ka = step_minutes / (60.0 * tau_hours)
    return float(np.clip(ka, 0.005, 0.50))


def _analyze_excitation(
    indoor: np.ndarray, t_out: np.ndarray, sol: np.ndarray, u: np.ndarray
) -> dict:
    """Analyse buffer data and return excitation metrics."""
    indoor_range = float(np.ptp(indoor))
    indoor_std = float(np.std(indoor))
    t_out_std = float(np.std(t_out))
    coupling_ratio = indoor_std / max(t_out_std, 0.1)
    return {
        "indoor_range": indoor_range,
        "indoor_std": indoor_std,
        "t_out_std": t_out_std,
        "coupling_ratio": coupling_ratio,
    }


_EXCITATION_INDOOR_RANGE_STABLE = 0.5
_EXCITATION_INDOOR_RANGE_MODERATE = 1.5
_EXCITATION_COUPLING_STABLE = 0.10
_EXCITATION_COUPLING_MODERATE = 0.25


def batch_fit_auto(
    samples: list[tuple],
    step_minutes: float = 30.0,
    ridge: float = 1e-2,
    irls_iters: int = 5,
) -> RCModel3R2C:
    """Adaptive 3R2C identification with data-informed priors and column selection.

    The routine analyses the buffer data and picks an identification strategy
    that matches the room's excitation level:

    * **Stable** (indoor range < 0.5 °C or coupling ratio < 0.10):
      Only ``kg`` and the hourly bias are fitted.  ``ka``, ``k_aw`` and
      ``k_wa`` are set from the data-derived attenuation estimate.
    * **Moderate** (indoor range < 1.5 °C or coupling ratio < 0.25):
      ``ka`` and ``kg`` are fitted with data-informed priors.  Wall params
      (k_aw, k_wa) are held at priors derived from ka.
    * **Well-excited** (otherwise): Full 6-param fit with excitation guards
      for ks, kh.

    Returns an :class:`RCModel3R2C` with ``model_type="auto"``.
    """
    from .rc_model_3r2c import DEFAULT_PARAMS_3R2C

    prior = DEFAULT_PARAMS_3R2C.copy()
    n_params = N_PARAMS_3R2C

    if len(samples) < n_params + 1:
        return RCModel3R2C(
            params=prior.copy(), rmse=None, n_samples=len(samples),
            step_minutes=step_minutes, model_type="auto",
        )

    # ---- Step 1: data analysis ----
    indoor = np.array([s[0] for s in samples], dtype=float)
    t_out = np.array([s[1] for s in samples], dtype=float)
    sol = np.array([s[2] for s in samples], dtype=float)
    u = np.array([s[3] for s in samples], dtype=float)
    y = np.array([samples[i][4] - indoor[i] for i in range(len(samples))], dtype=float)

    analysis = _analyze_excitation(indoor, t_out, sol, u)
    ka_eff = _compute_effective_ka(indoor, t_out, step_minutes)
    kg_eff = ka_eff * float(np.mean(indoor - t_out))
    kg_eff = float(np.clip(kg_eff, -2.0, 4.0))

    # Adaptive priors from data characteristics.
    adaptive_prior = np.array([
        ka_eff,              # ka: from attenuation
        0.15,                # ks: moderate solar gain
        0.20,                # kh: heating effectiveness
        kg_eff,              # kg: steady-state offset
        ka_eff * 3.0,        # k_aw: wall coupling proportional to ka
        ka_eff * 0.75,       # k_wa: k_aw / C_ratio
    ], dtype=float)
    adaptive_prior = np.clip(adaptive_prior, PARAM_LOWER_3R2C, PARAM_UPPER_3R2C)

    # Determine excitation regime and select columns to fit.
    is_stable = (
        analysis["indoor_range"] < _EXCITATION_INDOOR_RANGE_STABLE
        or analysis["coupling_ratio"] < _EXCITATION_COUPLING_STABLE
    )
    is_moderate = not is_stable and (
        analysis["indoor_range"] < _EXCITATION_INDOOR_RANGE_MODERATE
        or analysis["coupling_ratio"] < _EXCITATION_COUPLING_MODERATE
    )

    if is_stable:
        fit_cols = [3]
        ridge_actual = ridge * 10.0
        n_iterations = 1
    elif is_moderate:
        fit_cols = [0, 3]
        ridge_actual = ridge * 3.0
        n_iterations = 2
    else:
        fit_cols = [0, 3, 4]
        if float(np.std(sol)) >= EXCITATION_SOL_STD:
            fit_cols.append(1)
        if float(np.std(u)) >= EXCITATION_U_STD:
            fit_cols.append(2)
        else:
            y = y - adaptive_prior[2] * u
        ridge_actual = ridge
        n_iterations = 3

    fit_cols = sorted(fit_cols)
    theta = adaptive_prior.copy()

    # ---- Iterative two-step identification ----
    tw = _forward_filter_tw(samples, k_wa=theta[5])

    for iteration in range(n_iterations):
        phi = np.zeros((len(samples), 5), dtype=float)
        phi[:, 0] = t_out - indoor
        phi[:, 1] = sol
        phi[:, 2] = u
        phi[:, 3] = 1.0
        phi[:, 4] = tw[:len(samples)] - indoor

        # Adjust target for params NOT being fitted.
        y_adj = y.copy()
        for c in range(5):
            if c not in fit_cols and c < len(adaptive_prior):
                y_adj -= adaptive_prior[c] * phi[:, c]

        phi_sub = phi[:, fit_cols]
        prior_sub = adaptive_prior[fit_cols].copy()

        w = np.ones(len(y_adj), dtype=float)
        theta_sub = prior_sub.copy()
        for _ in range(max(1, irls_iters)):
            theta_sub = _weighted_ridge(phi_sub, y_adj, w, ridge_actual, prior_sub)
            resid = y_adj - phi_sub @ theta_sub
            scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
            delta = max(HUBER_DELTA, scale)
            a = np.abs(resid)
            w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))

        for i, c in enumerate(fit_cols):
            theta[c] = theta_sub[i]

        theta[5] = float(np.clip(theta[4] / C_RATIO_3R2C, 0.001, 0.50))

        if iteration < n_iterations - 1:
            tw = _forward_filter_tw(samples, k_wa=theta[5])

    theta = np.clip(theta, PARAM_LOWER_3R2C, PARAM_UPPER_3R2C)
    tw_final = _forward_filter_tw(samples, k_wa=theta[5])
    tw_last = float(tw_final[-1])

    # Residuals on the full 5-col regressor.
    phi_final = np.zeros((len(samples), 5), dtype=float)
    phi_final[:, 0] = t_out - indoor
    phi_final[:, 1] = sol
    phi_final[:, 2] = u
    phi_final[:, 3] = 1.0
    phi_final[:, 4] = tw_final[:len(samples)] - indoor
    resids = y - (phi_final @ theta[:5])
    rmse = float(np.sqrt(np.mean(resids ** 2))) if len(resids) else None

    # Hourly bias from per-hour median residuals.
    hourly_residuals: list[list[float]] = [[] for _ in range(N_HOURS)]
    for i in range(len(samples)):
        hour = int(samples[i][5]) if len(samples[i]) >= 6 else 0
        if 0 <= hour < N_HOURS:
            hourly_residuals[hour].append(float(resids[i]))
    hourly_bias = np.zeros(N_HOURS, dtype=float)
    for h in range(N_HOURS):
        if hourly_residuals[h]:
            hourly_bias[h] = float(np.median(hourly_residuals[h]))

    return RCModel3R2C(
        params=theta,
        rmse=rmse,
        n_samples=len(samples),
        step_minutes=step_minutes,
        model_type="auto",
        hourly_bias=hourly_bias,
        tw=tw_last,
    )
