"""Running residual scale tracker for disturbance detection.

Replaces the old RLS-based disturbance detection with a simpler,
model-agnostic EWMA of the absolute one-step residual.
"""

from __future__ import annotations


class ResidualTracker:
    """Tracks a running EWMA of |residual| for disturbance detection.

    ``init_scale`` is the initial robust residual scale (deg C) used before
    enough observations accumulate.
    """

    def __init__(self, init_scale: float = 0.1) -> None:
        self._scale = float(init_scale)

    @property
    def scale(self) -> float:
        return self._scale

    def update(self, residual: float) -> None:
        """Update the running scale estimate from the latest residual."""
        abs_r = abs(residual) + 1e-6
        self._scale = 0.97 * self._scale + 0.03 * abs_r
