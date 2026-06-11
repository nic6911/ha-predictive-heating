"""Parameter identification for the RC thermal model (difference form).

Two complementary routines:

* :func:`batch_fit` -- **robust** (Huber/IRLS) ridge regression over a buffer of
  history samples, used to bootstrap a zone from the recorder on first setup.
  Robust weighting stops sporadic disturbances (window opening, a brief sensor
  glitch) from biasing the fit. An *excitation guard* holds the heating gain at a
  physical prior when the history contains essentially no heating (e.g. summer),
  so ``kh`` is never identified from noise.
* :class:`RecursiveLeastSquares` -- online RLS with forgetting **and innovation
  gating**: a sample whose one-step residual is a strong outlier is not allowed to
  update the parameters/covariance, so transient disturbances cannot corrupt the
  learned model.

A training sample is a tuple ``(indoor, t_out, sol, u, indoor_next)``. The model
predicts the *temperature change* ``indoor_next - indoor`` from the difference-form
regressor ``[t_out - indoor, sol, u, 1.0]``.
"""

from __future__ import annotations

import numpy as np

from .rc_model import DEFAULT_PARAMS, N_PARAMS, RCModel

# Bounds keep identified parameters physically sane and the model stable.
# Order: [ka, ks, kh, kg]
PARAM_LOWER = np.array([0.01, 0.0, 0.0, -2.0], dtype=float)
PARAM_UPPER = np.array([0.50, 2.0, 1.0, 4.0], dtype=float)

# Below this standard deviation of the heating proxy ``u`` (deg C) we consider the
# data un-excited for heating and refuse to identify ``kh`` (hold it at the prior).
EXCITATION_U_STD = 0.15

# Huber threshold (deg C) for robust IRLS residual weighting.
HUBER_DELTA = 0.4


def _clip_params(params: np.ndarray) -> np.ndarray:
    return np.clip(params, PARAM_LOWER, PARAM_UPPER)


def _build_matrices(samples: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Build difference-form regression matrix ``Phi`` and target vector ``y``.

    ``Phi`` rows are ``[t_out - indoor, sol, u, 1]`` and ``y`` is the temperature
    change ``indoor_next - indoor``.
    """
    phi = np.array(
        [[t_out - indoor, sol, u, 1.0] for (indoor, t_out, sol, u, _) in samples],
        dtype=float,
    )
    y = np.array(
        [nxt - indoor for (indoor, _t, _s, _u, nxt) in samples], dtype=float
    )
    return phi, y


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
) -> RCModel:
    """Fit an :class:`RCModel` from samples via robust (Huber/IRLS) ridge regression."""
    if len(samples) < N_PARAMS + 1:
        # Not enough data: return the prior model.
        return RCModel(
            params=DEFAULT_PARAMS.copy(),
            rmse=None,
            n_samples=len(samples),
            step_minutes=step_minutes,
        )

    phi, y = _build_matrices(samples)
    prior = DEFAULT_PARAMS.copy()

    # Excitation guard: if there is essentially no heating in the data, do not try
    # to identify the heating gain kh -- hold it at the prior and fit the rest.
    u_col = phi[:, 2]
    fit_heat = float(np.std(u_col)) >= EXCITATION_U_STD
    if fit_heat:
        cols = [0, 1, 2, 3]
    else:
        cols = [0, 1, 3]  # drop kh column; substitute prior*u into the target
        y = y - prior[2] * u_col

    phi_sub = phi[:, cols]
    prior_sub = prior[cols]

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
    theta = prior.copy()
    for i, c in enumerate(cols):
        theta[c] = theta_sub[i]
    theta = _clip_params(theta)

    # One-step RMSE on the original (un-substituted) targets.
    phi_full, y_full = _build_matrices(samples)
    residuals = y_full - phi_full @ theta
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None
    return RCModel(
        params=theta,
        rmse=rmse,
        n_samples=len(samples),
        step_minutes=step_minutes,
    )


class RecursiveLeastSquares:
    """Online RLS estimator with exponential forgetting and innovation gating.

    ``forgetting`` < 1 lets the model track slow changes (e.g. seasons). A sample
    whose a-priori residual is a strong outlier (relative to a robust running scale,
    or above an absolute cap) is rejected: the parameters and covariance are left
    unchanged so a window-opening transient cannot corrupt the model.
    """

    def __init__(
        self,
        params: np.ndarray | None = None,
        forgetting: float = 0.999,
        p0: float = 1e3,
        outlier_sigma: float = 4.0,
        outlier_abs_cap: float = 1.5,
    ) -> None:
        self.theta = (
            DEFAULT_PARAMS.copy() if params is None else np.asarray(params, float)
        )
        if self.theta.shape != (N_PARAMS,):
            self.theta = DEFAULT_PARAMS.copy()
        self.forgetting = float(forgetting)
        self.P = np.eye(N_PARAMS) * p0
        self.outlier_sigma = float(outlier_sigma)
        self.outlier_abs_cap = float(outlier_abs_cap)
        # Robust running residual scale (EW mean of |err|).
        self._scale: float | None = None
        self._sq_err = 0.0
        self._count = 0
        self.last_residual: float | None = None
        self.last_rejected: bool = False

    def predict_delta(self, phi: np.ndarray) -> float:
        """Predicted temperature change for regressor ``phi``."""
        phi = np.asarray(phi, dtype=float).reshape(-1)
        return float(phi @ self.theta)

    def is_outlier(self, residual: float) -> bool:
        """Decide whether a one-step residual should be rejected from learning."""
        if abs(residual) > self.outlier_abs_cap:
            return True
        if self._scale is not None and self._scale > 1e-6:
            return abs(residual) > self.outlier_sigma * self._scale
        return False

    def update(self, phi: np.ndarray, target: float) -> bool:
        """Incorporate one sample ``(phi, target)`` unless it is an outlier.

        ``target`` is the observed temperature change. Returns ``True`` if the
        sample was used to update the model, ``False`` if it was rejected.
        """
        phi = np.asarray(phi, dtype=float).reshape(-1)
        err = float(target) - float(phi @ self.theta)
        self.last_residual = err

        if self.is_outlier(err):
            # Reject: don't touch theta/P. Let the scale drift very slightly so a
            # genuine regime change eventually widens the acceptance band.
            if self._scale is not None:
                self._scale = 0.999 * self._scale + 0.001 * abs(err)
            self.last_rejected = True
            return False

        lam = self.forgetting
        p_phi = self.P @ phi
        denom = lam + float(phi @ p_phi)
        gain = p_phi / denom
        self.theta = _clip_params(self.theta + gain * err)
        self.P = (self.P - np.outer(gain, p_phi)) / lam

        # Update robust scale and running RMSE with the accepted residual.
        if self._scale is None:
            self._scale = abs(err) + 1e-3
        else:
            self._scale = 0.97 * self._scale + 0.03 * abs(err)
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
        )
