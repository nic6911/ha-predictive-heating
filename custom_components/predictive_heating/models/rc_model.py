"""Grey-box RC thermal model for a single heating zone.

Two model variants:

**Standard (1R1C)** -- discrete single-state model in heat-balance form::

    T[k+1] - T[k] = ka * (T_out[k] - T[k]) + ks * Sol[k] + kh * u[k] + kg

**Enhanced (with thermal inertia)** -- adds a memory term that captures the
slab/floor thermal-mass effect::

    T[k+1] - T[k] = ka * (T_out[k] - T[k]) + ks * Sol[k] + kh * u[k] + kg
                    + k_mem * (T[k] - T[k-1])

``k_mem > 0`` means the room tends to continue its previous trajectory -- the
floor slab stores heat and releases it slowly.

Parameters for both variants:

* ``T``      indoor temperature (deg C),
* ``T_out``  outdoor temperature (deg C),
* ``Sol``    solar-gain proxy (0..1),
* ``u``      heating-demand proxy ``max(0, setpoint - T)`` (deg C),
* ``ka``     envelope coupling to outdoor (0 < ka < 1),
* ``ks``     solar-gain coefficient,
* ``kh``     heating gain,
* ``kg``     internal-gains offset (deg C / step),
* ``k_mem``  thermal-inertia coefficient [0, 0.9] (enhanced model only).

Why the difference form: it *ties* outdoor coupling to inertia (a = 1-ka), so
steady state is ``T_ss = T_out + (ks*Sol + kh*u + kg) / ka``. The internal-gains
term ``kg`` lets a room float above outdoor even with no heating -- the summer
regime.

Both variants stay linear in parameters and linear in ``u`` (convex QP).

References:
    Bacher & Madsen (2011), "Identifying suitable models for the heat dynamics of
    buildings", Energy and Buildings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..const import DEFAULT_K_MEM, N_HOURS, N_PARAMS_ENHANCED, N_PARAMS_STANDARD, PARAM_MEM_LOWER, PARAM_MEM_UPPER

# Parameter vector order -- standard: [ka, ks, kh, kg]
# Enhanced: [ka, ks, kh, kg, k_mem]
N_PARAMS = N_PARAMS_STANDARD
PARAM_NAMES = ["ka", "ks", "kh", "kg"]

# Physically reasonable defaults for a heavy, well-damped underfloor-heated room
# at a 30-minute step (prior before any data is fitted):
#   ka=0.08  -> outdoor time constant ~ step/ka ~ 6 h
#   ks=0.30  -> moderate solar gain
#   kh=0.25  -> heating effectiveness
#   kg=0.25  -> floats ~ kg/ka ~ 3 C above outdoor on internal gains alone
DEFAULT_PARAMS = np.array([0.08, 0.30, 0.25, 0.25], dtype=float)
DEFAULT_PARAMS_ENHANCED = np.array([0.08, 0.30, 0.25, 0.25, DEFAULT_K_MEM], dtype=float)


@dataclass
class RCModel:
    """A single-zone RC thermal model with identifiable parameters.

    Supports two model types:
    - ``standard`` (4 params: ka, ks, kh, kg)
    - ``enhanced`` (5 params: ka, ks, kh, kg, k_mem) adds thermal inertia.
    """

    params: np.ndarray = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    rmse: float | None = None
    n_samples: int = 0
    step_minutes: float = 30.0
    model_type: str = "standard"
    # Time-varying output-disturbance correction (deg C / step), one entry per
    # hour of day (0..23). This captures diurnal internal-gain patterns from
    # occupancy (person + equipment) that a single scalar bias cannot represent.
    # Updated online from accepted one-step residuals via per-hour EWMA.
    # When all entries are equal it degenerates to the classic scalar offset-free
    # bias (Pannocchia & Rawlings, 2003).
    hourly_bias: np.ndarray = field(default_factory=lambda: np.zeros(N_HOURS))

    @property
    def bias(self) -> float:
        """Scalar equivalent = mean hourly bias (backward compat)."""
        return float(np.mean(self.hourly_bias))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def heat_demand(setpoint: float, indoor: float) -> float:
        """Heating-demand proxy ``u`` for a proportional floor thermostat."""
        return max(0.0, float(setpoint) - float(indoor))

    @staticmethod
    def _hour_at(start_hour: int, step_idx: int, step_minutes: float) -> int:
        """Hour of day (0..23) for step ``step_idx`` of the simulation."""
        total_minutes = int(start_hour * 60 + step_idx * step_minutes)
        return int(total_minutes / 60) % 24

    @property
    def n_params(self) -> int:
        return N_PARAMS_STANDARD if self.model_type == "standard" else N_PARAMS_ENHANCED

    @property
    def is_enhanced(self) -> bool:
        return self.model_type == "enhanced"

    @property
    def ka(self) -> float:
        return float(self.params[0])

    @property
    def ks(self) -> float:
        return float(self.params[1])

    @property
    def kh(self) -> float:
        return float(self.params[2])

    @property
    def kg(self) -> float:
        return float(self.params[3])

    @property
    def k_mem(self) -> float:
        """Thermal-inertia coefficient (0 for standard model)."""
        if self.is_enhanced and len(self.params) > 4:
            return float(self.params[4])
        return 0.0

    @property
    def a(self) -> float:
        """Effective inertia coefficient ``a = 1 - ka`` (for the MPC matrices)."""
        return float(1.0 - self.params[0])

    @property
    def b_heat(self) -> float:
        return float(self.params[2])

    def regressor(
        self, indoor: float, t_out: float, sol: float, u: float, prev_delta: float = 0.0
    ) -> np.ndarray:
        """Difference-form regression row ``phi`` so ``T_next - T = phi @ params``.

        Standard model: ``phi = [t_out - indoor, sol, u, 1.0]``
        Enhanced model: ``phi = [t_out - indoor, sol, u, 1.0, prev_delta]``
        """
        if self.is_enhanced:
            return np.array([t_out - indoor, sol, u, 1.0, prev_delta], dtype=float)
        return np.array([t_out - indoor, sol, u, 1.0], dtype=float)

    # ------------------------------------------------------------------ predict
    def step(
        self, indoor: float, t_out: float, sol: float, u: float,
        prev_delta: float = 0.0, hour: int | None = None
    ) -> float:
        """Advance one step and return the predicted next indoor temperature.

        ``prev_delta`` is ``T[k] - T[k-1]`` (0 at start, only used by enhanced model).
        ``hour`` is the hour of day (0..23) for time-varying bias lookup; when
        ``None`` the mean hourly bias is used (scalar fallback).
        """
        delta = float(self.regressor(indoor, t_out, sol, u, prev_delta) @ self.params)
        if hour is not None and 0 <= hour < N_HOURS:
            delta += self.hourly_bias[hour]
        else:
            delta += self.bias
        return float(indoor) + delta

    def simulate(
        self,
        t0: float,
        t_out: np.ndarray,
        sol: np.ndarray,
        u: np.ndarray,
        start_hour: int = 0,
    ) -> np.ndarray:
        """Roll the model forward over a horizon.

        ``start_hour`` is the UTC hour of day for the first simulation step.
        Returns an array of length ``len(u) + 1`` starting with ``t0``.
        For the enhanced model, tracks prev_delta internally.
        """
        t_out = np.asarray(t_out, dtype=float)
        sol = np.asarray(sol, dtype=float)
        u = np.asarray(u, dtype=float)
        n = len(u)
        out = np.empty(n + 1, dtype=float)
        out[0] = t0
        prev_delta = 0.0
        for k in range(n):
            hour = self._hour_at(start_hour, k, self.step_minutes)
            out[k + 1] = self.step(out[k], t_out[k], sol[k], u[k], prev_delta, hour=hour)
            prev_delta = out[k + 1] - out[k]
        return out

    def free_float(
        self, t0: float, t_out: np.ndarray, sol: np.ndarray, start_hour: int = 0
    ) -> np.ndarray:
        """Trajectory with zero heating (used to detect 'no control authority')."""
        n = len(np.asarray(t_out))
        return self.simulate(t0, t_out, sol, np.zeros(n), start_hour=start_hour)

    # ------------------------------------------ linear prediction for the MPC
    def prediction_matrices(
        self,
        t0: float,
        t_out: np.ndarray,
        sol: np.ndarray,
        start_hour: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(T_free, G)`` so predicted ``T = T_free + G @ u``.

        ``start_hour`` is the UTC hour of day for the first forecast step.
        For the **standard** model this is exact (analytical step response).
        For the **enhanced** model the thermal-inertia term makes G depend on
        past u, so we build G numerically by perturbing each input channel.
        The hourly bias enters T_free and cancels out in the numerical G.
        """
        t_out = np.asarray(t_out, dtype=float)
        sol = np.asarray(sol, dtype=float)
        n = len(t_out)

        if not self.is_enhanced:
            # Standard model: analytical lower-triangular step response.
            a = self.a
            b = self.b_heat
            t_free_full = self.free_float(t0, t_out, sol, start_hour=start_hour)
            t_free = t_free_full[1:]
            g = np.zeros((n, n), dtype=float)
            for k in range(n):
                for j in range(k + 1):
                    g[k, j] = b * (a ** (k - j))
            return t_free, g

        # Enhanced model: build G numerically by finite differences.
        eps = 1e-4
        t_free_full = self.free_float(t0, t_out, sol, start_hour=start_hour)
        t_free = t_free_full[1:]
        g = np.zeros((n, n), dtype=float)
        for j in range(n):
            u_pert = np.zeros(n, dtype=float)
            u_pert[j] = eps
            t_pert_full = self.simulate(t0, t_out, sol, u_pert, start_hour=start_hour)
            t_pert = t_pert_full[1:]
            g[:, j] = (t_pert - t_free) / eps
        return t_free, g

    # ------------------------------------------------------------- (de)serialise
    def as_dict(self) -> dict:
        return {
            "params": self.params.tolist(),
            "rmse": self.rmse,
            "n_samples": self.n_samples,
            "step_minutes": self.step_minutes,
            "model_type": self.model_type,
            "bias": self.bias,
            "hourly_bias": self.hourly_bias.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RCModel":
        params = np.array(
            data.get("params", DEFAULT_PARAMS.tolist()), dtype=float
        )
        model_type = data.get("model_type", "standard")
        expected_n = N_PARAMS_STANDARD if model_type == "standard" else N_PARAMS_ENHANCED
        if params.shape != (expected_n,):
            params = DEFAULT_PARAMS.copy() if model_type == "standard" else DEFAULT_PARAMS_ENHANCED.copy()
        # Load hourly_bias (backward compat: from scalar bias).
        hb = data.get("hourly_bias")
        if hb is not None:
            hourly_bias = np.array(hb, dtype=float)
            if hourly_bias.shape != (N_HOURS,):
                hourly_bias = np.full(N_HOURS, data.get("bias", 0.0))
        else:
            hourly_bias = np.full(N_HOURS, data.get("bias", 0.0))
        return cls(
            params=params,
            rmse=data.get("rmse"),
            n_samples=int(data.get("n_samples", 0)),
            step_minutes=float(data.get("step_minutes", 30.0)),
            model_type=model_type,
            hourly_bias=hourly_bias,
        )
