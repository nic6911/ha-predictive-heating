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
    N_PARAMS_ENHANCED,
    N_PARAMS_STANDARD,
)
from .rc_model import DEFAULT_PARAMS, DEFAULT_PARAMS_ENHANCED, N_PARAMS, RCModel

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
    return np.array([t_out - indoor, sol, u, 1.0], dtype=float)


def _build_matrices(
    samples: list[tuple], model_type: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    """Build regression matrix ``Phi`` and target vector ``y``.

    ``Phi`` rows are the appropriate regressor for the model type, and ``y``
    is the temperature change ``indoor_next - indoor``.
    """
    phi_rows = []
    y_vals = []
    for i, (indoor, t_out, sol, u, nxt) in enumerate(samples):
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

    # One-step RMSE on the original targets, plus a robust seed for the bias.
    phi_full, y_full = _build_matrices(samples, model_type)
    residuals = y_full - phi_full @ theta
    bias = float(np.median(residuals)) if len(residuals) else 0.0
    bias = float(np.clip(bias, -0.5, 0.5))
    rmse = (
        float(np.sqrt(np.mean((residuals - bias) ** 2))) if len(residuals) else None
    )
    return RCModel(
        params=theta,
        rmse=rmse,
        n_samples=len(samples),
        step_minutes=step_minutes,
        model_type=model_type,
        bias=bias,
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
            bias=self.bias,
        )
