"""Grey-box RC thermal model for a single heating zone.

Discrete single-state model written in **heat-balance / temperature-difference**
form so it is well conditioned and physically meaningful::

    T[k+1] - T[k] = ka * (T_out[k] - T[k]) + ks * Sol[k] + kh * u[k] + kg

equivalently::

    T[k+1] = (1 - ka) * T[k] + ka * T_out[k] + ks * Sol[k] + kh * u[k] + kg

where

* ``T``      is the indoor temperature (deg C),
* ``T_out``  is the outdoor temperature (deg C),
* ``Sol``    is a solar-gain proxy (0..1, derived from cloud cover / UV / sun elevation),
* ``u``      is the heating-demand proxy ``max(0, setpoint - T)`` (deg C),
* ``ka``     is the envelope coupling to outdoor (0 < ka < 1; small => slow / heavy mass),
* ``ks``     is the solar-gain gain,
* ``kh``     is the heating gain,
* ``kg``     is a persistent **internal-gains offset** (deg C / step).

Why this form (vs an unconstrained ``T[k+1] = a T + b_out T_out + ... + c``):
the difference form *ties* the outdoor coupling to the inertia (``a = 1 - ka``),
so the steady state is ``T_ss = T_out + (ks*Sol + kh*u + kg) / ka``. The
internal-gains term ``kg`` lets a room float several degrees above outdoor even
with no heating -- which is exactly the summer regime. The previous unconstrained
model could (and did) collapse predictions toward the outdoor temperature.

The model stays linear in its parameters (identifiable with OLS/RLS) and linear in
``u`` (so the controller builds a convex QP).

References:
    Bacher & Madsen (2011), "Identifying suitable models for the heat dynamics of
    buildings", Energy and Buildings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Parameter vector order: [ka, ks, kh, kg]
N_PARAMS = 4
PARAM_NAMES = ["ka", "ks", "kh", "kg"]

# Physically reasonable defaults for a heavy, well-damped underfloor-heated room
# at a 30-minute step (prior before any data is fitted):
#   ka=0.08  -> outdoor time constant ~ step/ka ~ 6 h
#   ks=0.30  -> moderate solar gain
#   kh=0.25  -> heating effectiveness
#   kg=0.25  -> floats ~ kg/ka ~ 3 C above outdoor on internal gains alone
DEFAULT_PARAMS = np.array([0.08, 0.30, 0.25, 0.25], dtype=float)


@dataclass
class RCModel:
    """A single-zone RC thermal model with identifiable parameters."""

    params: np.ndarray = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    rmse: float | None = None  # last-known one-step fit quality, deg C
    n_samples: int = 0
    step_minutes: float = 30.0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def heat_demand(setpoint: float, indoor: float) -> float:
        """Heating-demand proxy ``u`` for a proportional floor thermostat."""
        return max(0.0, float(setpoint) - float(indoor))

    @property
    def ka(self) -> float:
        return float(self.params[0])

    @property
    def kh(self) -> float:
        return float(self.params[2])

    @property
    def kg(self) -> float:
        return float(self.params[3])

    @property
    def a(self) -> float:
        """Effective inertia coefficient ``a = 1 - ka`` (for the MPC matrices)."""
        return float(1.0 - self.params[0])

    @property
    def b_heat(self) -> float:
        return float(self.params[2])

    def regressor(
        self, indoor: float, t_out: float, sol: float, u: float
    ) -> np.ndarray:
        """Difference-form regression row ``phi`` so ``T_next - T = phi @ params``."""
        return np.array([t_out - indoor, sol, u, 1.0], dtype=float)

    # ------------------------------------------------------------------ predict
    def step(self, indoor: float, t_out: float, sol: float, u: float) -> float:
        """Advance one step and return the predicted next indoor temperature."""
        delta = float(self.regressor(indoor, t_out, sol, u) @ self.params)
        return float(indoor) + delta

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
        #   contribution = b * a**(k-j) for k >= j, else 0
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
        # Reset legacy (5-param) or malformed models to the new prior.
        if params.shape != (N_PARAMS,):
            params = DEFAULT_PARAMS.copy()
        return cls(
            params=params,
            rmse=data.get("rmse"),
            n_samples=int(data.get("n_samples", 0)),
            step_minutes=float(data.get("step_minutes", 30.0)),
        )
