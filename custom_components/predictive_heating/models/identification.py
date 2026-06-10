"""Parameter identification for the RC thermal model.

Two complementary routines:

* :func:`batch_fit` -- ordinary least squares over a buffer of history samples,
  used to bootstrap a zone from the recorder on first setup.
* :class:`RecursiveLeastSquares` -- online RLS with forgetting factor, used by the
  coordinator to keep the model adapting as conditions change.

A training sample is a tuple ``(indoor, t_out, sol, u, indoor_next)``.
"""

from __future__ import annotations

import numpy as np

from .rc_model import DEFAULT_PARAMS, N_PARAMS, RCModel

# Bounds keep identified parameters physically sane and the model stable.
PARAM_LOWER = np.array([0.50, 0.0, 0.0, 0.0, -2.0], dtype=float)
PARAM_UPPER = np.array([0.999, 0.50, 2.0, 1.0, 2.0], dtype=float)


def _clip_params(params: np.ndarray) -> np.ndarray:
    return np.clip(params, PARAM_LOWER, PARAM_UPPER)


def _build_matrices(samples: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Build regression matrix ``Phi`` and target vector ``y``."""
    phi = np.array(
        [[indoor, t_out, sol, u, 1.0] for (indoor, t_out, sol, u, _) in samples],
        dtype=float,
    )
    y = np.array([nxt for (*_, nxt) in samples], dtype=float)
    return phi, y


def batch_fit(
    samples: list[tuple], step_minutes: float = 30.0, ridge: float = 1e-3
) -> RCModel:
    """Fit an :class:`RCModel` from a list of samples via ridge-regularised OLS."""
    if len(samples) < N_PARAMS + 1:
        # Not enough data: return the prior model.
        return RCModel(
            params=DEFAULT_PARAMS.copy(),
            rmse=None,
            n_samples=len(samples),
            step_minutes=step_minutes,
        )

    phi, y = _build_matrices(samples)
    # Ridge solution: (Phi^T Phi + ridge I)^-1 Phi^T y
    reg = ridge * np.eye(N_PARAMS)
    try:
        theta = np.linalg.solve(phi.T @ phi + reg, phi.T @ y)
    except np.linalg.LinAlgError:
        theta, *_ = np.linalg.lstsq(phi, y, rcond=None)
    theta = _clip_params(theta)

    residuals = y - phi @ theta
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None
    return RCModel(
        params=theta,
        rmse=rmse,
        n_samples=len(samples),
        step_minutes=step_minutes,
    )


class RecursiveLeastSquares:
    """Online RLS estimator with exponential forgetting.

    ``forgetting`` < 1 lets the model track slow changes (e.g. seasons). The
    covariance ``P`` is initialised large to trust early data quickly.
    """

    def __init__(
        self,
        params: np.ndarray | None = None,
        forgetting: float = 0.999,
        p0: float = 1e3,
    ) -> None:
        self.theta = (
            DEFAULT_PARAMS.copy() if params is None else np.asarray(params, float)
        )
        self.forgetting = float(forgetting)
        self.P = np.eye(N_PARAMS) * p0
        self._sq_err = 0.0
        self._count = 0

    def update(self, phi: np.ndarray, target: float) -> None:
        """Incorporate one sample ``(phi, target)``."""
        phi = np.asarray(phi, dtype=float).reshape(-1)
        lam = self.forgetting
        p_phi = self.P @ phi
        denom = lam + float(phi @ p_phi)
        gain = p_phi / denom
        err = float(target) - float(phi @ self.theta)
        self.theta = _clip_params(self.theta + gain * err)
        self.P = (self.P - np.outer(gain, p_phi)) / lam
        # Track a running RMSE estimate (a priori error).
        self._sq_err += err * err
        self._count += 1

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
