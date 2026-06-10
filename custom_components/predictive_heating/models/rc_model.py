"""Grey-box RC thermal model for a single heating zone.

Discrete 1R1C (single-state) model in regression form::

    T[k+1] = a * T[k] + b_out * T_out[k] + b_sol * Sol[k] + b_heat * u[k] + c

where

* ``T``      is the indoor temperature (deg C),
* ``T_out``  is the outdoor temperature (deg C),
* ``Sol``    is a solar-gain proxy (0..1, derived from cloud cover / UV / sun elevation),
* ``u``      is the heating-demand proxy ``max(0, setpoint - T)`` (deg C),
* ``a``      is the thermal-storage / inertia coefficient (0 < a < 1),
* ``b_*``    are the input gains and ``c`` an offset.

The model is intentionally linear in its parameters so it can be identified with
ordinary / recursive least squares, and linear in ``u`` so the controller (MPC) can
build a convex QP. A richer 2R2C variant can be layered on later behind the same
``predict`` interface.

References:
    Bacher & Madsen (2011), "Identifying suitable models for the heat dynamics of
    buildings", Energy and Buildings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Parameter vector order: [a, b_out, b_sol, b_heat, c]
N_PARAMS = 5
PARAM_NAMES = ["a", "b_out", "b_sol", "b_heat", "c"]

# Reasonable physical defaults for a well-damped underfloor-heated room at a
# 30-minute step. Used as a prior before any data is fitted.
DEFAULT_PARAMS = np.array([0.90, 0.05, 0.30, 0.20, 0.0], dtype=float)


@dataclass
class RCModel:
    """A single-zone RC thermal model with identifiable parameters."""

    params: np.ndarray = field(
        default_factory=lambda: DEFAULT_PARAMS.copy()
    )
    rmse: float | None = None  # last-known fit quality, deg C
    n_samples: int = 0
    step_minutes: float = 30.0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def heat_demand(setpoint: float, indoor: float) -> float:
        """Heating-demand proxy ``u`` for a proportional floor thermostat."""
        return max(0.0, float(setpoint) - float(indoor))

    @property
    def a(self) -> float:
        return float(self.params[0])

    @property
    def b_heat(self) -> float:
        return float(self.params[3])

    def regressor(
        self, indoor: float, t_out: float, sol: float, u: float
    ) -> np.ndarray:
        """Build the regression row ``phi`` such that ``T_next = phi @ params``."""
        return np.array([indoor, t_out, sol, u, 1.0], dtype=float)

    # ------------------------------------------------------------------ predict
    def step(self, indoor: float, t_out: float, sol: float, u: float) -> float:
        """Advance one step and return the predicted next indoor temperature."""
        return float(self.regressor(indoor, t_out, sol, u) @ self.params)

    def simulate(
        self,
        t0: float,
        t_out: np.ndarray,
        sol: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Roll the model forward over a horizon.

        Returns an array of length ``len(u) + 1`` starting with ``t0``.
        """
        t_out = np.asarray(t_out, dtype=float)
        sol = np.asarray(sol, dtype=float)
        u = np.asarray(u, dtype=float)
        n = len(u)
        out = np.empty(n + 1, dtype=float)
        out[0] = t0
        for k in range(n):
            out[k + 1] = self.step(out[k], t_out[k], sol[k], u[k])
        return out

    def free_float(
        self, t0: float, t_out: np.ndarray, sol: np.ndarray
    ) -> np.ndarray:
        """Trajectory with zero heating (used to detect 'no control authority')."""
        n = len(np.asarray(t_out))
        return self.simulate(t0, t_out, sol, np.zeros(n))

    # ------------------------------------------ linear prediction for the MPC
    def prediction_matrices(
        self,
        t0: float,
        t_out: np.ndarray,
        sol: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(T_free, G)`` so predicted ``T = T_free + G @ u``.

        ``T_free`` is the free-float trajectory (heating off) over steps 1..n and
        ``G`` is the lower-triangular step-response (heating) matrix. This makes the
        temperature an affine function of the heat-demand vector ``u`` -- exactly the
        form an MPC QP needs.
        """
        t_out = np.asarray(t_out, dtype=float)
        sol = np.asarray(sol, dtype=float)
        n = len(t_out)
        a = self.a
        b = self.b_heat

        t_free_full = self.free_float(t0, t_out, sol)  # length n+1
        t_free = t_free_full[1:]  # predictions for steps 1..n

        # Impulse response of a unit u at step j on temperature at step k:
        #   contribution = b * a**(k-1-j) for k > j, else 0
        g = np.zeros((n, n), dtype=float)
        for k in range(n):
            for j in range(k + 1):
                g[k, j] = b * (a ** (k - j))
        return t_free, g

    # ------------------------------------------------------------- (de)serialise
    def as_dict(self) -> dict:
        return {
            "params": self.params.tolist(),
            "rmse": self.rmse,
            "n_samples": self.n_samples,
            "step_minutes": self.step_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RCModel":
        params = np.array(
            data.get("params", DEFAULT_PARAMS.tolist()), dtype=float
        )
        if params.shape != (N_PARAMS,):
            params = DEFAULT_PARAMS.copy()
        return cls(
            params=params,
            rmse=data.get("rmse"),
            n_samples=int(data.get("n_samples", 0)),
            step_minutes=float(data.get("step_minutes", 30.0)),
        )
