"""Parameter identification for the RC thermal model (difference form).

Two complementary routines:

* :func:`batch_fit` -- **robust** (Huber/IRLS) ridge regression over a buffer of
  history samples, used to bootstrap a zone from the recorder on first setup.
  Robust weighting stops sporadic disturbances (window opening, a brief sensor
  glitch) from biasing the fit. An *excitation guard* holds the heating gain at a
  physical prior when the history contains essentially no heating (e.g. summer),
  so ``kh`` is never identified from noise.
* :class:`RecursiveLeastSquares` -- online RLS with forgetting **and graduated
  innovation gating**: small residuals update fully, medium residuals are
  discounted, large residuals are only accepted if sustained across multiple
  samples (distinguishing transient disturbances from regime changes).

A training sample is a tuple ``(indoor, t_out, sol, u, indoor_next)``. The model
predicts the *temperature change* ``indoor_next - indoor`` from the difference-form
regressor ``[t_out - indoor, sol, u, 1.0]`` (standard) or
``[t_out - indoor, sol, u, 1.0, prev_delta]`` (enhanced).
"""

from __future__ import annotations

import numpy as np

from ..const import (
    GRADUATED_SUSTAINED_COUNT,
    GRADUATED_THRESHOLD_HIGH,
    GRADUATED_THRESHOLD_LOW,
    GRADUATED_WINDOW,
    N_HOURS,
    N_PARAMS_ENHANCED,
    N_PARAMS_STANDARD,
    N_PARAMS_3R2C,
)
from .rc_model import DEFAULT_PARAMS, DEFAULT_PARAMS_ENHANCED, N_PARAMS, RCModel
from .rc_model_3r2c import DEFAULT_PARAMS_3R2C, RCModel3R2C

# Bounds keep identified parameters physically sane and the model stable.
# Order -- standard: [ka, ks, kh, kg]
# Enhanced: [ka, ks, kh, kg, k_mem]
# ka's lower bound corresponds to the slowest envelope time constant we allow
# (step/ka): at a 30-min step ka=0.005 -> ~100 h, which comfortably covers very
# heavy, well-insulated slab systems. (Heavier than this and the open-loop steady
# state ``T_out + (...)/ka`` becomes hyper-sensitive; the offset-free bias term
# keeps the forecast honest regardless.)
PARAM_LOWER = np.array([0.005, 0.0, 0.0, -2.0, 0.0], dtype=float)
PARAM_UPPER = np.array([0.50, 2.0, 1.0, 4.0, 0.9], dtype=float)

# Below this standard deviation of the heating proxy ``u`` (deg C) we consider the
# data un-excited for heating and refuse to identify ``kh`` (hold it at the prior).
EXCITATION_U_STD = 0.15

# Below this standard deviation of the solar proxy (0..1) the data carries no usable
# solar variation, so ``ks`` is not identifiable. In that case we hold ``ks`` at the
# **prior** (not zero), so the model still has a reasonable solar estimate on
# low-variance training windows (e.g. an overcast summer week). Once sunny day/night
# variation enters the buffer, ks is re-identified automatically.
EXCITATION_SOL_STD = 0.05

# Huber threshold (deg C) for robust IRLS residual weighting.
HUBER_DELTA = 0.4

# Exponential-forgetting rate for the online output-bias estimate ``d``. Small ->
# slow, stable drift tracking (~1-2 days of memory at typical update intervals).
# Chosen by an offline forgetting-factor sweep on a year of data; it is a learning
# rate, not a building-specific constant.
BIAS_ALPHA = 0.02


def _clip_params(params: np.ndarray, n_params: int = N_PARAMS_STANDARD) -> np.ndarray:
    """Clip parameters to physically sane bounds."""
    return np.clip(params[:n_params], PARAM_LOWER[:n_params], PARAM_UPPER[:n_params])


def _build_regressor(indoor, t_out, sol, u, model_type="standard", prev_delta=0.0):
    """Build a difference-form regressor row."""
    if model_type == "enhanced":
        return np.array([t_out - indoor, sol, u, 1.0, prev_delta], dtype=float)
    if model_type == "3r2c":
        raise ValueError("_build_regressor does not support 3r2c (use batch_fit_3r2c)")
    return np.array([t_out - indoor, sol, u, 1.0], dtype=float)


def _build_matrices(
    samples: list[tuple], model_type: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    """Build regression matrix ``Phi`` and target vector ``y``.

    ``Phi`` rows are the appropriate regressor for the model type, and ``y``
    is the temperature change ``indoor_next - indoor``.
    Samples may have 5 elements ``(indoor, t_out, sol, u, nxt)`` or 6 elements
    (with trailing hour-of-day, which is ignored for matrix building).
    """
    phi_rows = []
    y_vals = []
    for i, sample in enumerate(samples):
        indoor, t_out, sol, u, nxt = sample[:5]
        prev_delta = 0.0
        if model_type == "enhanced" and i > 0:
            prev_delta = samples[i - 1][4] - samples[i - 1][0]
        phi_rows.append(_build_regressor(indoor, t_out, sol, u, model_type, prev_delta))
        y_vals.append(nxt - indoor)
    return np.array(phi_rows, dtype=float), np.array(y_vals, dtype=float)


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


def _compute_hourly_bias(
    samples: list[tuple], theta: np.ndarray, model_type: str = "standard"
) -> np.ndarray:
    """Compute initial 24-element hourly bias from per-hour median residuals.

    Each sample is ``(indoor, t_out, sol, u, nxt[, hour])``. The residual
    ``(nxt - indoor) - phi @ theta`` is grouped by hour of day, and the median
    of each hour's residuals becomes the initial bias for that hour.

    .. note::
       This function does **not** support ``model_type="3r2c"``; the 3R2C
       fit computes its hourly bias inline in :func:`batch_fit_3r2c`.
    """
    if model_type == "3r2c":
        raise ValueError("_compute_hourly_bias: use inline bias in batch_fit_3r2c")
    hourly_residuals: list[list[float]] = [[] for _ in range(N_HOURS)]
    for i, sample in enumerate(samples):
        indoor, t_out, sol, u, nxt = sample[:5]
        prev_delta = 0.0
        if model_type == "enhanced" and i > 0:
            prev_delta = samples[i - 1][4] - samples[i - 1][0]
        phi = _build_regressor(indoor, t_out, sol, u, model_type, prev_delta)
        pred_delta = float(phi @ theta)
        residual = (nxt - indoor) - pred_delta
        hour = int(sample[5]) if len(sample) >= 6 else 0
        if 0 <= hour < N_HOURS:
            hourly_residuals[hour].append(residual)

    hourly_bias = np.zeros(N_HOURS, dtype=float)
    for h in range(N_HOURS):
        if hourly_residuals[h]:
            hourly_bias[h] = float(np.median(hourly_residuals[h]))
    return hourly_bias


def batch_fit(
    samples: list[tuple],
    step_minutes: float = 30.0,
    ridge: float = 1e-2,
    irls_iters: int = 5,
    model_type: str = "standard",
) -> RCModel:
    """Fit an :class:`RCModel` from samples via robust (Huber/IRLS) ridge regression.

    ``model_type`` can be ``"standard"`` (4 params) or ``"enhanced"`` (5 params).
    """
    n_params = N_PARAMS_STANDARD if model_type == "standard" else N_PARAMS_ENHANCED
    prior = DEFAULT_PARAMS.copy() if model_type == "standard" else DEFAULT_PARAMS_ENHANCED.copy()

    if len(samples) < n_params + 1:
        return RCModel(
            params=prior.copy(),
            rmse=None,
            n_samples=len(samples),
            step_minutes=step_minutes,
            model_type=model_type,
        )

    phi, y = _build_matrices(samples, model_type)

    # Excitation guards: only identify a gain when its regressor is actually excited.
    #   * heating gain kh (col 2): if the heating proxy barely moves, hold kh at the
    #     prior (substitute prior*u into the target) -- a bounded, controlled input.
    #   * solar gain ks (col 1): if the solar proxy barely moves, hold ks at the
    #     PRIOR (not zero), so the model retains a plausible solar estimate on
    #     low-variance windows. ks is re-identified as soon as excitation returns.
    #   * k_mem (col 4, enhanced): always identified when model_type is enhanced.
    # ka (col 0) and kg (col 3) are always identified.
    sol_col = phi[:, 1]
    u_col = phi[:, 2]
    fit_sol = float(np.std(sol_col)) >= EXCITATION_SOL_STD
    fit_heat = float(np.std(u_col)) >= EXCITATION_U_STD

    cols = [0, 3]
    if fit_sol:
        cols.append(1)
    if fit_heat:
        cols.append(2)
    else:
        y = y - prior[2] * u_col
    if model_type == "enhanced":
        cols.append(4)  # k_mem always fitted for enhanced model
    cols = sorted(cols)

    phi_sub = phi[:, cols]
    # Shrink toward the prior. When ks is not excited we exclude it from cols,
    # so the prior value is used. When it IS excited, shrink toward 0 to avoid
    # the prior dominating on weakly-excited windows.
    prior_sub = prior[cols].copy()
    if fit_sol:
        sol_idx = cols.index(1)
        prior_sub[sol_idx] = 0.0

    # Iteratively re-weighted least squares with Huber weights for robustness.
    w = np.ones(len(y), dtype=float)
    theta_sub = prior_sub.copy()
    for _ in range(max(1, irls_iters)):
        theta_sub = _weighted_ridge(phi_sub, y, w, ridge, prior_sub)
        resid = y - phi_sub @ theta_sub
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        delta = max(HUBER_DELTA, scale)
        a = np.abs(resid)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))

    # Reassemble the full parameter vector.
    # ks defaults to its PRIOR when un-excited (not zero) so the model retains
    # a reasonable solar estimate on low-variance training windows.
    theta = prior.copy()
    for i, c in enumerate(cols):
        theta[c] = theta_sub[i]
    theta = _clip_params(theta, n_params)

    # One-step RMSE and per-hour bias initialisation.
    phi_full, y_full = _build_matrices(samples, model_type)
    residuals = y_full - phi_full @ theta
    rmse = (
        float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None
    )
    hourly_bias = _compute_hourly_bias(samples, theta, model_type)
    return RCModel(
        params=theta,
        rmse=rmse,
        n_samples=len(samples),
        step_minutes=step_minutes,
        model_type=model_type,
        hourly_bias=hourly_bias,
    )


class RecursiveLeastSquares:
    """Online RLS estimator with exponential forgetting and graduated innovation gating.

    ``forgetting`` < 1 lets the model track slow changes (e.g. seasons).

    **Graduated innovation gating** replaces the old binary accept/reject:
      - |residual| <= GRADUATED_THRESHOLD_LOW: full update (gain=1.0)
      - GRADUATED_THRESHOLD_LOW < |residual| <= GRADUATED_THRESHOLD_HIGH:
        discounted update (gain decreases linearly 1.0 -> 0.3)
      - |residual| > GRADUATED_THRESHOLD_HIGH: rejected UNLESS sustained
        (GRADUATED_SUSTAINED_COUNT of last GRADUATED_WINDOW samples exceed the
        threshold) -- distinguishes genuine regime change from transient disturbances.
    """

    def __init__(
        self,
        params: np.ndarray | None = None,
        forgetting: float = 0.999,
        p0: float = 1e3,
        bias: float = 0.0,
    ) -> None:
        if params is not None:
            params = np.asarray(params, dtype=float)
            if params.shape != (N_PARAMS,):
                params = None
        if params is None:
            params = DEFAULT_PARAMS.copy()
        self.theta = params
        self.forgetting = float(forgetting)
        self.P = np.eye(N_PARAMS) * p0
        # Offset-free output-disturbance estimate ``d`` (deg C / step).
        self.bias = float(bias)
        self._scale: float | None = None
        self._sq_err = 0.0
        self._count = 0
        self.last_residual: float | None = None
        self.last_rejected: bool = False
        # Rolling window for sustained-error detection (graduated gating).
        self._error_window: list[float] = []

    def predict_delta(self, phi: np.ndarray) -> float:
        """Predicted temperature change for regressor ``phi``."""
        phi = np.asarray(phi, dtype=float).reshape(-1)
        return float(phi @ self.theta[: len(phi)])

    def _gain_from_residual(self, residual: float) -> float:
        """Compute the update gain (0..1) from a graduated innovation gate."""
        abs_r = abs(residual)

        # Small residuals: full update.
        if abs_r <= GRADUATED_THRESHOLD_LOW:
            return 1.0

        # Medium residuals: discounted update (linear taper 1.0 -> 0.3).
        if abs_r <= GRADUATED_THRESHOLD_HIGH:
            frac = (abs_r - GRADUATED_THRESHOLD_LOW) / (
                GRADUATED_THRESHOLD_HIGH - GRADUATED_THRESHOLD_LOW
            )
            return 1.0 - 0.7 * frac

        # Large residuals: check if sustained (regime change) or transient.
        self._error_window.append(abs_r)
        if len(self._error_window) > GRADUATED_WINDOW:
            self._error_window.pop(0)

        sustained = (
            len(self._error_window) >= GRADUATED_WINDOW
            and sum(1 for e in self._error_window if e > GRADUATED_THRESHOLD_HIGH)
            >= GRADUATED_SUSTAINED_COUNT
        )
        if sustained:
            return 0.2  # regime change: heavily discounted update
        return 0.0  # transient disturbance: reject

    def update(self, phi: np.ndarray, target: float) -> bool:
        """Incorporate one sample ``(phi, target)`` with graduated gating.

        Returns ``True`` if the sample was used to update the model (even partially),
        ``False`` if fully rejected.
        """
        phi = np.asarray(phi, dtype=float).reshape(-1)
        err = float(target) - float(phi @ self.theta[: len(phi)])
        self.last_residual = err

        gain = self._gain_from_residual(err)

        # Always update the robust scale (for disturbance detection).
        if self._scale is None:
            self._scale = abs(err) + 1e-3
        else:
            self._scale = 0.97 * self._scale + 0.03 * abs(err)

        if gain <= 0.0:
            self.last_rejected = True
            return False

        # Apply (possibly discounted) RLS update.
        lam = self.forgetting
        p_phi = self.P @ phi
        denom = lam + float(phi @ p_phi)
        gain_vec = p_phi / denom
        self.theta = _clip_params(
            self.theta + gain * gain_vec * err, N_PARAMS
        )
        self.P = (self.P - gain * np.outer(gain_vec, p_phi)) / lam

        # Update offset-free bias (EW mean of accepted residuals, discounted).
        self.bias = (1.0 - BIAS_ALPHA * gain) * self.bias + BIAS_ALPHA * gain * err
        self.bias = float(np.clip(self.bias, -0.5, 0.5))
        self._sq_err += err * err
        self._count += 1
        self.last_rejected = False
        return True

    @property
    def rmse(self) -> float | None:
        if self._count == 0:
            return None
        return float(np.sqrt(self._sq_err / self._count))

    def to_model(self, step_minutes: float = 30.0, n_samples: int = 0) -> RCModel:
        return RCModel(
            params=self.theta.copy(),
            rmse=self.rmse,
            n_samples=n_samples or self._count,
            step_minutes=step_minutes,
            hourly_bias=np.full(N_HOURS, self.bias),
        )


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

    where k_wa is an initial guess (refined in the second step).
    """
    n = len(samples)
    tw = np.empty(n + 1, dtype=float)
    if n == 0:
        return tw
    # Initialise T_w from the first indoor temperature (equilibrium assumption).
    tw[0] = samples[0][0]
    for i in range(n):
        ta = samples[i][0]
        tw[i + 1] = tw[i] + k_wa * (ta - tw[i])
    return tw


def batch_fit_3r2c(
    samples: list[tuple],
    step_minutes: float = 30.0,
    ridge: float = 1e-2,
    irls_iters: int = 5,
) -> RCModel3R2C:
    """Fit a 3R2C two-node model from samples.

    Two-step procedure (cf. Lin et al. 2024):

    1. Forward-filter T_w from the data using an initial k_wa guess.
    2. Fit [ka, ks, kh, kg, k_aw] via Huber/IRLS ridge regression (T_a delta).
    3. Re-estimate k_wa from the wall-update equation using the fitted T_w.

    Returns an :class:`RCModel3R2C` with all 6 params + hourly bias.
    """
    prior = DEFAULT_PARAMS_3R2C.copy()
    n_params = N_PARAMS_3R2C

    if len(samples) < n_params + 1:
        return RCModel3R2C(
            params=prior.copy(), rmse=None, n_samples=len(samples),
            step_minutes=step_minutes,
        )

    # ---- Step 1: initial T_w estimation ----
    k_wa_guess = prior[5]  # default k_wa = 0.02
    tw = _forward_filter_tw(samples, k_wa=k_wa_guess)

    # ---- Step 2: fit air-node params [ka, ks, kh, kg, k_aw] ----
    # Regressor for air delta: [t_out - indoor, sol, u, 1.0, tw - indoor]
    phi_rows = []
    y_vals = []
    for i, sample in enumerate(samples):
        indoor, t_out, sol, u, nxt = sample[:5]
        phi_rows.append(np.array([
            t_out - indoor, sol, u, 1.0, tw[i] - indoor,
        ], dtype=float))
        y_vals.append(nxt - indoor)
    phi = np.array(phi_rows, dtype=float)
    y = np.array(y_vals, dtype=float)

    # Excitation guards (cols: 0=ka, 1=ks, 2=kh, 3=kg, 4=k_aw).
    sol_col = phi[:, 1]
    u_col = phi[:, 2]
    fit_sol = float(np.std(sol_col)) >= EXCITATION_SOL_STD
    fit_heat = float(np.std(u_col)) >= EXCITATION_U_STD

    cols = [0, 3, 4]  # ka, kg, k_aw are always fitted
    if fit_sol:
        cols.append(1)  # ks
    if fit_heat:
        cols.append(2)  # kh
    else:
        y = y - prior[2] * u_col
    cols = sorted(cols)

    phi_sub = phi[:, cols]
    prior_sub = prior[cols].copy()
    if fit_sol:
        sol_idx = cols.index(1)
        prior_sub[sol_idx] = 0.0

    w = np.ones(len(y), dtype=float)
    theta_sub = prior_sub.copy()
    for _ in range(max(1, irls_iters)):
        theta_sub = _weighted_ridge(phi_sub, y, w, ridge, prior_sub)
        resid = y - phi_sub @ theta_sub
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        delta = max(HUBER_DELTA, scale)
        a = np.abs(resid)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))

    theta = prior.copy()
    for i, c in enumerate(cols):
        theta[c] = theta_sub[i]

    # ---- Step 3: estimate k_wa from wall dynamics ----
    # Re-filter T_w with the newly fitted k_aw (use k_wa = k_aw / C_RATIO)
    k_aw_hat = theta[4]
    k_wa_val = float(np.clip(k_aw_hat / C_RATIO_3R2C, 0.001, 0.50))
    theta[5] = k_wa_val

    # Clip all params.
    theta = np.clip(theta, PARAM_LOWER_3R2C, PARAM_UPPER_3R2C)

    # Re-filter T_w with the final k_wa for consistent model initialisation.
    tw_final = _forward_filter_tw(samples, k_wa=theta[5])
    tw_last = float(tw_final[-1])

    # One-step RMSE and per-hour bias initialisation.
    residuals = y - (phi[:, :5] @ theta[:5])
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None

    # Hourly bias from per-hour median residuals (inline to avoid regressor mismatch).
    hourly_residuals: list[list[float]] = [[] for _ in range(N_HOURS)]
    for i in range(len(samples)):
        hour = int(samples[i][5]) if len(samples[i]) >= 6 else 0
        if 0 <= hour < N_HOURS:
            hourly_residuals[hour].append(float(residuals[i]))
    hourly_bias = np.zeros(N_HOURS, dtype=float)
    for h in range(N_HOURS):
        if hourly_residuals[h]:
            hourly_bias[h] = float(np.median(hourly_residuals[h]))

    return RCModel3R2C(
        params=theta,
        rmse=rmse,
        n_samples=len(samples),
        step_minutes=step_minutes,
        hourly_bias=hourly_bias,
        tw=tw_last,
    )


# ---------------------------------------------------------------------------
# Auto (adaptive) identification
# ---------------------------------------------------------------------------

def _compute_effective_ka(
    indoor: np.ndarray, t_out: np.ndarray, step_minutes: float = 30.0
) -> float:
    """Compute effective outdoor coupling ka from the data's attenuation ratio.

    For a first-order thermal system driven by diurnal outdoor temperature
    variation, the indoor temperature attenuates the outdoor signal by::

        A = indoor_std / outdoor_std ≈ 1 / sqrt(1 + (2π τ / T)²)

    where τ is the building time constant and T = 24 h is the diurnal period.
    Solving for τ gives::

        τ = T / (2π) * sqrt(1/A² - 1)   (hours)

    then ka = step_hours / τ.

    When the indoor is extremely stable (A → 0) the time constant is very
    large and ka approaches its lower bound.  When indoor tracks outdoor
    closely (A → 1) the time constant is very short and ka caps at an upper
    bound.

    Returns ka clipped to ``[0.005, 0.50]``.
    """
    indoor_std = float(np.std(indoor))
    t_out_std = float(np.std(t_out))

    if t_out_std < 0.1 or indoor_std < 0.01:
        return 0.02  # fallback for nearly constant data

    A = indoor_std / t_out_std
    if A >= 0.99:
        return 0.30

    # Diurnal period (hours).
    T_hours = 24.0
    # Time constant from attenuation formula.
    tau_hours = T_hours / (2.0 * np.pi) * np.sqrt(max(0.0, 1.0 / (A * A) - 1.0))
    # ka = step_size / time_constant.
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
      ``k_wa`` are set from the data-derived attenuation estimate.  This
      prevents the model from overestimating outdoor coupling in well-insulated
      rooms, which is the #1 cause of wrong forecasts.

    * **Moderate** (indoor range < 1.5 °C or coupling ratio < 0.25):
      ``ka`` and ``kg`` are fitted with data-informed priors.  Wall params
      (k_aw, k_wa) are held at priors derived from ka.

    * **Well-excited** (otherwise): Full 6-param fit with the standard
      two-step procedure, retaining the existing excitation guards for ks, kh.
      This path is identical to the original 3R2C identification but uses the
      attenuation-based ka as the prior.

    Returns an :class:`RCModel3R2C` with ``model_type="auto"``.
    """
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
        # Very stable room: fit only kg + hourly bias.
        # All thermal structure params are held at adaptive priors.
        fit_cols = [3]
        ridge_actual = ridge * 10.0
        n_iterations = 1
    elif is_moderate:
        # Moderately stable: fit ka and kg; hold wall params at priors.
        fit_cols = [0, 3]
        ridge_actual = ridge * 3.0
        n_iterations = 2
    else:
        # Well-excited: standard fit with excitation guards for ks, kh.
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
    # Forward-filter tw with current k_wa → fit air params → re-estimate k_wa.
    tw = _forward_filter_tw(samples, k_wa=theta[5])

    for iteration in range(n_iterations):
        # Build 5-col regressor: [t_out-indoor, sol, u, 1.0, tw-indoor].
        phi = np.zeros((len(samples), 5), dtype=float)
        phi[:, 0] = t_out - indoor
        phi[:, 1] = sol
        phi[:, 2] = u
        phi[:, 3] = 1.0
        phi[:, 4] = tw[:len(samples)] - indoor  # tw[i] is wall temp at step i start

        # Adjust the target for params NOT being fitted in this iteration.
        # Unfitted params are held at their adaptive priors, so we subtract
        # their contribution from y (same as the heating-excitation guard).
        y_adj = y.copy()
        for c in range(5):
            if c not in fit_cols and c < len(adaptive_prior):
                y_adj -= adaptive_prior[c] * phi[:, c]

        # Subset regressor to the columns actually being fitted.
        phi_sub = phi[:, fit_cols]
        prior_sub = adaptive_prior[fit_cols].copy()

        # IRLS ridge regression.
        w = np.ones(len(y_adj), dtype=float)
        theta_sub = prior_sub.copy()
        for _ in range(max(1, irls_iters)):
            theta_sub = _weighted_ridge(phi_sub, y_adj, w, ridge_actual, prior_sub)
            resid = y_adj - phi_sub @ theta_sub
            scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
            delta = max(HUBER_DELTA, scale)
            a = np.abs(resid)
            w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))

        # Write fitted values back into theta.
        for i, c in enumerate(fit_cols):
            theta[c] = theta_sub[i]

        # Derive k_wa from k_aw (heat-capacity ratio).
        theta[5] = float(np.clip(theta[4] / C_RATIO_3R2C, 0.001, 0.50))

        # Re-filter tw for the next iteration (unless this is the last).
        if iteration < n_iterations - 1:
            tw = _forward_filter_tw(samples, k_wa=theta[5])

    # Final clip and tw.
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
