"""Economic Model Predictive Control for a single heating zone.

The plant is the linear RC model, so the indoor-temperature trajectory is an affine
function of the heat-demand vector ``u``::

    T = T_free + G @ u                      (see RCModel.prediction_matrices)

We minimise, over the horizon, a soft-constrained economic objective::

    J(u) =  w_comfort * sum (T_k - target)^2
          + w_cold    * sum relu(comfort_min - T_k)^2     (never let it get cold)
          + w_hot     * sum relu(T_k - comfort_max)^2      (avoid overshoot)
          + w_energy  * sum price_k * u_k                  (cheap / efficient)

subject to box constraints ``0 <= u_k <= u_max``. ``u_k = 0`` means "no heating", so
when free solar / outdoor gains already keep the room warm the optimiser naturally
drives ``u`` to zero -- this is the "no control authority / coasting" situation.

The QP is solved with projected-gradient descent in pure NumPy (no hard third-party
solver dependency). If ``osqp``/``scipy`` are installed they could be slotted in here
later for speed, but for <=48 steps this is already fast and robust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.rc_model import RCModel


@dataclass
class MPCResult:
    """Outcome of one MPC solve."""

    u: np.ndarray  # optimal heat-demand trajectory
    temperature: np.ndarray  # predicted indoor temperature (steps 1..n)
    free_float: np.ndarray  # predicted temperature with heating off
    u0: float  # near-term heat demand to apply now
    has_authority: bool  # False => coasting on free heat
    objective: float


def solve(
    model: RCModel,
    t0: float,
    t_out: np.ndarray,
    sol: np.ndarray,
    price: np.ndarray,
    comfort_target: float,
    comfort_min: float,
    comfort_max: float,
    w_comfort: float,
    w_energy: float,
    u_max: float = 6.0,
    iterations: int = 400,
) -> MPCResult:
    """Solve the economic MPC and return the optimal plan."""
    t_out = np.asarray(t_out, dtype=float)
    sol = np.asarray(sol, dtype=float)
    price = np.asarray(price, dtype=float)
    n = len(t_out)
    if n == 0:
        return MPCResult(
            u=np.zeros(0),
            temperature=np.zeros(0),
            free_float=np.zeros(0),
            u0=0.0,
            has_authority=False,
            objective=0.0,
        )

    t_free, g = model.prediction_matrices(t0, t_out, sol)

    # One-sided penalties weighted relative to the comfort weight.
    w_cold = max(w_comfort * 8.0, 5.0)
    w_hot = max(w_comfort * 4.0, 2.0)

    # Lipschitz constant of the smooth (quadratic) part for a safe step size.
    gtg_norm = float(np.linalg.norm(g, 2)) ** 2
    lipschitz = 2.0 * (w_comfort + w_cold + w_hot) * gtg_norm + 1e-6
    lr = 1.0 / lipschitz

    u = np.zeros(n, dtype=float)

    def objective(u_vec: np.ndarray) -> float:
        t = t_free + g @ u_vec
        cold = np.maximum(comfort_min - t, 0.0)
        hot = np.maximum(t - comfort_max, 0.0)
        return float(
            w_comfort * np.sum((t - comfort_target) ** 2)
            + w_cold * np.sum(cold**2)
            + w_hot * np.sum(hot**2)
            + w_energy * np.sum(price * u_vec)
        )

    for _ in range(iterations):
        t = t_free + g @ u
        grad_track = 2.0 * w_comfort * (g.T @ (t - comfort_target))
        cold = np.maximum(comfort_min - t, 0.0)
        grad_cold = -2.0 * w_cold * (g.T @ cold)
        hot = np.maximum(t - comfort_max, 0.0)
        grad_hot = 2.0 * w_hot * (g.T @ hot)
        grad_energy = w_energy * price
        grad = grad_track + grad_cold + grad_hot + grad_energy
        u = np.clip(u - lr * grad, 0.0, u_max)

    temperature = t_free + g @ u
    free_float = t_free

    # Authority: does any meaningful heating help, or is the room already warm
    # enough on its own across the near horizon?
    near = slice(0, min(2, n))
    has_authority = bool(u[near].max() > 0.05) or bool(
        np.any(free_float[near] < comfort_target - 0.1)
    )

    return MPCResult(
        u=u,
        temperature=temperature,
        free_float=free_float,
        u0=float(u[0]),
        has_authority=has_authority,
        objective=objective(u),
    )
