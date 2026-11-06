"""Truth-model point-mass dynamics (spec §2), used by the simulator and for re-sim checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vehicle import Environment, Rocket


@dataclass(frozen=True)
class TruthModel:
    rocket: Rocket
    env: Environment
    drag: bool = True
    cd_scale: float = 1.0
    isp_scale: float = 1.0
    wind: np.ndarray | None = None  # overrides env.wind if given

    @property
    def alpha(self) -> float:
        return 1.0 / (self.rocket.isp * self.isp_scale * self.env.g0)

    def drag_force(self, r: np.ndarray, v: np.ndarray, thrusting: bool) -> np.ndarray:
        if not self.drag:
            return np.zeros(3)
        w = self.env.wind_vec if self.wind is None else np.asarray(self.wind, float)
        vr = v - w
        cd = self.rocket.C_D * self.cd_scale * (self.rocket.kappa if thrusting else 1.0)
        rho = self.env.rho_atm(max(r[0], 0.0))
        return -0.5 * rho * self.rocket.S_ref * cd * np.linalg.norm(vr) * vr

    def deriv(self, s: np.ndarray, T: np.ndarray) -> np.ndarray:
        """s = [r(3), v(3), m]."""
        r, v, m = s[0:3], s[3:6], s[6]
        Tn = np.linalg.norm(T)
        D = self.drag_force(r, v, Tn > 0)
        return np.concatenate([v, T / m + self.env.g + D / m, [-self.alpha * Tn]])

    def step(self, s: np.ndarray, T: np.ndarray, dt: float) -> np.ndarray:
        """RK4 with thrust held constant over the step."""
        k1 = self.deriv(s, T)
        k2 = self.deriv(s + 0.5 * dt * k1, T)
        k3 = self.deriv(s + 0.5 * dt * k2, T)
        k4 = self.deriv(s + dt * k3, T)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def open_loop_resim(model: TruthModel, traj, substeps: int = 200) -> np.ndarray:
    """Re-simulate the plan open-loop, holding thrust *acceleration* u per the trajectory's hold
    (pitfall 2: T(t) = u(t) * m(t)). Returns final state [r, v, m]."""
    s = np.concatenate([traj.r[0], traj.v[0], [traj.m[0]]])
    for k in range(len(traj.t) - 1):
        h = (traj.t[k + 1] - traj.t[k]) / substeps
        for j in range(substeps):
            tm = traj.t[k] + (j + 0.5) * h
            u = traj.u_at(tm)
            # integrate with u held -> thrust scales with current mass
            def f(x, u=u):
                return model.deriv(x, u * x[6])
            k1 = f(s)
            k2 = f(s + 0.5 * h * k1)
            k3 = f(s + 0.5 * h * k2)
            k4 = f(s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s
