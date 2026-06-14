"""3R2C two-node thermal model for a single heating zone.

The model has two temperature states:

* **T_a** -- indoor air temperature (fast node, directly measured),
* **T_w** -- wall / slab temperature (slow node, unmeasured).

Discrete-time difference form at 30-minute step::

    T_a[k+1] - T_a[k] = ka * (T_out[k] - T_a[k])
                       + k_aw * (T_w[k] - T_a[k])
                       + ks * Sol[k] + kh * u[k] + kg
    T_w[k+1] - T_w[k] = k_wa * (T_a[k] - T_w[k])

Parameters (6): [ka, ks, kh, kg, k_aw, k_wa]

References:
    Chen et al. (2026), "3R2C and 3C-4C grey-box models for building
    thermal dynamics", Energy & Buildings.
    Bacher & Madsen (2011), "Identifying suitable models for the heat
    dynamics of buildings", Energy and Buildings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..const import N_HOURS, N_PARAMS_3R2C

DEFAULT_PARAMS_3R2C = np.array([0.06, 0.20, 0.20, 0.20, 0.08, 0.02], dtype=float)


@dataclass
class RCModel3R2C:
    """Two-node 3R2C thermal model with air (fast) and wall (slow) nodes."""

    params: np.ndarray = field(default_factory=lambda: DEFAULT_PARAMS_3R2C.copy())
    rmse: float | None = None
    n_samples: int = 0
    step_minutes: float = 30.0
    model_type: str = "auto"
    tw: float = 0.0
    hourly_bias: np.ndarray = field(default_factory=lambda: np.zeros(N_HOURS))

    @property
    def bias(self) -> float:
        return float(np.mean(self.hourly_bias))

    @staticmethod
    def heat_demand(setpoint: float, indoor: float) -> float:
        return max(0.0, float(setpoint) - float(indoor))

    @staticmethod
    def _hour_at(start_hour: int, step_idx: int, step_minutes: float) -> int:
        total_minutes = int(start_hour * 60 + step_idx * step_minutes)
        return int(total_minutes / 60) % 24

    @property
    def ka(self) -> float: return float(self.params[0])
    @property
    def ks(self) -> float: return float(self.params[1])
    @property
    def kh(self) -> float: return float(self.params[2])
    @property
    def kg(self) -> float: return float(self.params[3])
    @property
    def k_aw(self) -> float: return float(self.params[4])
    @property
    def k_wa(self) -> float: return float(self.params[5])

    def predict_delta(
        self, indoor: float, t_out: float, sol: float, u: float, tw: float | None = None
    ) -> float:
        """One-step temperature change prediction for disturbance detection."""
        if tw is None:
            tw = self.tw
        return float(self.regressor(indoor, t_out, sol, u, tw)[:5] @ self.params[:5])

    def regressor(
        self, indoor: float, t_out: float, sol: float, u: float, tw: float | None = None
    ) -> np.ndarray:
        """Difference-form regressor for the air-temperature change."""
        if tw is None:
            tw = self.tw
        return np.array([t_out - indoor, sol, u, 1.0, tw - indoor, 0.0], dtype=float)

    def step(
        self, indoor: float, t_out: float, sol: float, u: float,
        prev_delta: float = 0.0, hour: int | None = None, tw: float | None = None
    ) -> float:
        if tw is None:
            tw = self.tw
        delta = (
            self.ka * (t_out - indoor)
            + self.k_aw * (tw - indoor)
            + self.ks * sol + self.kh * u + self.kg
        )
        if hour is not None and 0 <= hour < N_HOURS:
            delta += self.hourly_bias[hour]
        else:
            delta += self.bias
        next_indoor = float(indoor) + delta
        next_tw = tw + self.k_wa * (indoor - tw)
        self.tw = next_tw
        return next_indoor

    def simulate(
        self,
        t0: float,
        t_out: np.ndarray,
        sol: np.ndarray,
        u: np.ndarray,
        start_hour: int = 0,
        tw0: float | None = None,
    ) -> np.ndarray:
        t_out = np.asarray(t_out, dtype=float)
        sol = np.asarray(sol, dtype=float)
        u = np.asarray(u, dtype=float)
        n = len(u)
        out = np.empty(n + 1, dtype=float)
        out[0] = t0
        tw = t0 if tw0 is None else tw0
        self.tw = tw
        for k in range(n):
            hour = self._hour_at(start_hour, k, self.step_minutes)
            out[k + 1] = self.step(out[k], t_out[k], sol[k], u[k], hour=hour)
            tw = self.tw
        return out

    def free_float(
        self, t0: float, t_out: np.ndarray, sol: np.ndarray,
        start_hour: int = 0, tw0: float | None = None,
    ) -> np.ndarray:
        n = len(np.asarray(t_out))
        return self.simulate(t0, t_out, sol, np.zeros(n), start_hour=start_hour, tw0=tw0)

    def prediction_matrices(
        self,
        t0: float,
        t_out: np.ndarray,
        sol: np.ndarray,
        start_hour: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_out = np.asarray(t_out, dtype=float)
        sol = np.asarray(sol, dtype=float)
        n = len(t_out)
        eps = 1e-4
        tw0 = self.tw if abs(self.tw) > 0.01 else t0
        t_free_full = self.free_float(t0, t_out, sol, start_hour=start_hour, tw0=tw0)
        t_free = t_free_full[1:]
        g = np.zeros((n, n), dtype=float)
        for j in range(n):
            u_pert = np.zeros(n, dtype=float)
            u_pert[j] = eps
            t_pert_full = self.simulate(
                t0, t_out, sol, u_pert, start_hour=start_hour, tw0=tw0,
            )
            t_pert = t_pert_full[1:]
            g[:, j] = (t_pert - t_free) / eps
        return t_free, g

    def as_dict(self) -> dict:
        return {
            "params": self.params.tolist(),
            "rmse": self.rmse,
            "n_samples": self.n_samples,
            "step_minutes": self.step_minutes,
            "model_type": self.model_type,
            "tw": self.tw,
            "bias": self.bias,
            "hourly_bias": self.hourly_bias.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RCModel3R2C":
        params = np.array(data.get("params", DEFAULT_PARAMS_3R2C.tolist()), dtype=float)
        if params.shape != (N_PARAMS_3R2C,):
            params = DEFAULT_PARAMS_3R2C.copy()
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
            model_type="auto",
            tw=float(data.get("tw", 0.0)),
            hourly_bias=hourly_bias,
        )
