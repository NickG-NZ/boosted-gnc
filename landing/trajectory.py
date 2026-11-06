"""Discrete guidance solution container with continuous-time sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class Trajectory:
    """Guidance output in SI units. Arrays are node-major: r[k] is a 3-vector."""

    t: np.ndarray  # (N,)
    r: np.ndarray  # (N,3)
    v: np.ndarray  # (N,3)
    m: np.ndarray  # (N,)
    u: np.ndarray  # (N,3) thrust acceleration T/m
    sigma: np.ndarray  # (N,) slack, ||u|| <= sigma
    hold: Literal["zoh", "foh"]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def tf(self) -> float:
        return float(self.t[-1])

    @property
    def T(self) -> np.ndarray:
        return self.u * self.m[:, None]

    def u_at(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self.t[0], self.t[-1]))
        if self.hold == "zoh":
            k = min(int(np.searchsorted(self.t, t, side="right")) - 1, len(self.t) - 2)
            return self.u[k].copy()
        return np.array([np.interp(t, self.t, self.u[:, i]) for i in range(3)])

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Reference position/velocity. Integrates the (drag-free part of the) node model
        within an interval would be more exact; cubic Hermite on (r, v) is plenty for tracking."""
        t = float(np.clip(t, self.t[0], self.t[-1]))
        k = min(int(np.searchsorted(self.t, t, side="right")) - 1, len(self.t) - 2)
        h = self.t[k + 1] - self.t[k]
        s = (t - self.t[k]) / h
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        r = h00 * self.r[k] + h10 * h * self.v[k] + h01 * self.r[k + 1] + h11 * h * self.v[k + 1]
        # velocity: linear interpolation (consistent enough with the Hermite position)
        v = (1 - s) * self.v[k] + s * self.v[k + 1]
        return r, v

    def to_dict(self, rho_max: float) -> dict[str, Any]:
        thrust = np.linalg.norm(self.T, axis=1)
        return {
            "t": self.t.tolist(),
            "r": self.r.tolist(),
            "v": self.v.tolist(),
            "m": self.m.tolist(),
            "T": self.T.tolist(),
            "throttle": (thrust / rho_max).tolist(),
            "hold": self.hold,
            "info": self.info,
        }
