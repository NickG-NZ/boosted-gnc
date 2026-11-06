"""Rocket, environment and scenario definitions, loaded from JSON configs.

Frame convention (spec §1): landing-site-fixed, index 0 is "up", indices 1-2 horizontal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

E1 = np.array([1.0, 0.0, 0.0])


def clip_to_cone(d: np.ndarray, theta: float) -> np.ndarray:
    """Unit vector closest to d inside the cone of half-angle theta about E1 (up)."""
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    d = E1.copy() if n < 1e-9 else d / n
    if d[0] >= np.cos(theta):
        return d
    hor = d.copy()
    hor[0] = 0.0
    hn = np.linalg.norm(hor)
    hor = hor / hn if hn > 1e-12 else np.array([0.0, 1.0, 0.0])
    return np.cos(theta) * E1 + np.sin(theta) * hor


def rotate_toward(a: np.ndarray, b: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotate unit vector a toward unit vector b by at most max_angle (slerp)."""
    cosang = float(np.clip(a @ b, -1.0, 1.0))
    ang = np.arccos(cosang)
    if ang <= max_angle or ang < 1e-9:
        return b.copy()
    axis_perp = b - cosang * a
    axis_perp /= np.linalg.norm(axis_perp)
    return np.cos(max_angle) * a + np.sin(max_angle) * axis_perp


@dataclass(frozen=True)
class Rocket:
    name: str
    m_dry: float  # kg
    isp: float  # s
    T_max: float  # N
    thr_min: float  # fraction of T_max (hard engine floor)
    thr_max: float  # fraction of T_max (planning derate, leaves margin for tracker)
    theta_max_deg: float  # thrust pointing cone half-angle from vertical
    S_ref: float = 0.0  # m^2
    C_D: float = 0.0
    kappa: float = 1.0  # C_D multiplier while thrusting (plume shielding)
    thr_min_plan: float | None = None  # planning floor (> thr_min leaves throttle-down margin)
    attitude: dict[str, Any] | None = None  # gimbal/TVC params, see landing/attitude.py
    geometry: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> Rocket:
        return cls(**json.loads(Path(path).read_text()))

    def alpha(self, g0: float) -> float:
        """Mass flow per unit thrust (kg/s/N)."""
        return 1.0 / (self.isp * g0)

    @property
    def rho1(self) -> float:
        """Planning thrust floor (N). Controllers use the engine floor T_min."""
        return (self.thr_min if self.thr_min_plan is None else self.thr_min_plan) * self.T_max

    @property
    def T_min(self) -> float:
        """Hard engine thrust floor (N)."""
        return self.thr_min * self.T_max

    @property
    def rho2(self) -> float:
        return self.thr_max * self.T_max

    @property
    def theta_max(self) -> float:
        return np.deg2rad(self.theta_max_deg)

    @property
    def CdS_thrusting(self) -> float:
        """Effective drag area while the engine is lit (always, during powered descent)."""
        return self.C_D * self.kappa * self.S_ref

    def with_overrides(self, **kw: Any) -> Rocket:
        return replace(self, **kw)


@dataclass(frozen=True)
class Environment:
    g0: float = 9.80665
    rho0: float = 1.225
    H: float = 8500.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def g(self) -> np.ndarray:
        return np.array([-self.g0, 0.0, 0.0])

    @property
    def wind_vec(self) -> np.ndarray:
        return np.asarray(self.wind, dtype=float)

    def rho_atm(self, h: float | np.ndarray) -> float | np.ndarray:
        return self.rho0 * np.exp(-np.asarray(h) / self.H)


@dataclass(frozen=True)
class Scenario:
    name: str
    rocket: Rocket
    env: Environment
    m0: float
    r0: np.ndarray
    v0: np.ndarray
    glide_slope_deg: float
    gfold: dict[str, Any]
    scvx: dict[str, Any]
    tracker: dict[str, Any]
    sim_cases: list[dict[str, Any]]
    coast: dict[str, Any] = field(default_factory=dict)  # empty -> engine lit at t=0
    constraints: dict[str, Any] = field(default_factory=dict)  # optional attitude constraints
    att0: np.ndarray | None = None  # current body/thrust axis (overrides the default below)
    mpc: dict[str, Any] = field(default_factory=dict)  # inner-loop MPC params, see landing/mpc.py

    @property
    def gamma_gs(self) -> float:
        return np.deg2rad(self.glide_slope_deg)

    # ---- optional attitude constraints (absent -> original spec problem) ----
    @property
    def initial_attitude_tol(self) -> float:
        """Allowed angle between the first planned thrust and the current body axis (rad).
        The gimbal can point thrust a few degrees off the body, so an exact equality is too rigid."""
        return float(np.deg2rad(self.constraints.get("initial_attitude_tol_deg", 0.0)))

    def initial_thrust_dir(self) -> np.ndarray | None:
        """Direction the first thrust must point (the vehicle's attitude at ignition/re-plan).
        Default when `constrain_initial_attitude` is set: falling engine-first, i.e. along the
        wind-relative velocity reversed (aerodynamically trimmed coast), clipped into the cone."""
        if not self.constraints.get("constrain_initial_attitude", False):
            return None
        d = self.att0 if self.att0 is not None else -(self.v0 - self.env.wind_vec)
        return clip_to_cone(d, self.rocket.theta_max)

    @classmethod
    def from_json(cls, path: str | Path) -> Scenario:
        path = Path(path)
        cfg = json.loads(path.read_text())
        rocket_ref = cfg["rocket"]
        rocket = (
            Rocket(**rocket_ref)
            if isinstance(rocket_ref, dict)
            else Rocket.from_json((path.parent / rocket_ref).resolve())
        )
        env_cfg = cfg.get("environment", {})
        env = Environment(**{**env_cfg, "wind": tuple(env_cfg.get("wind", (0, 0, 0)))})
        ic = cfg["initial_state"]
        return cls(
            name=cfg.get("name", path.stem),
            rocket=rocket,
            env=env,
            m0=float(ic["m0"]),
            r0=np.asarray(ic["r0"], dtype=float),
            v0=np.asarray(ic["v0"], dtype=float),
            glide_slope_deg=cfg["constraints"]["glide_slope_deg"],
            gfold=cfg.get("gfold", {}),
            scvx=cfg.get("scvx", {}),
            tracker=cfg.get("tracker", {}),
            sim_cases=cfg.get("sim_cases", []),
            coast=cfg.get("coast", {}),
            constraints=cfg["constraints"],
            mpc=cfg.get("mpc", {}),
        )

    def with_rocket(self, **kw: Any) -> Scenario:
        return replace(self, rocket=self.rocket.with_overrides(**kw))
