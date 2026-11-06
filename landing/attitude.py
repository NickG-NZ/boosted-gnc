"""Pitch/yaw attitude control with a gimbaled engine (thrust vector control).

The dynamics themselves (gimbal actuator, torques, rotation) are simulated by the Rust plant
(sim/crates/sim-vehicle); this module is the flight-software side: the attitude controller and
the pointing reference model.

State: body axis b (unit), body angular velocity w (perpendicular to b), gimbal deflection g.
The gimbal is stored as a vector perpendicular to b with |g| = sin(delta): the thrust direction is
    t_hat = sqrt(1 - |g|^2) b + g
The engine sits a distance `arm` below the CG, so lateral thrust produces a torque
    tau = r x F = (-arm b) x (T t_hat) = -arm T (b x g)
which rotates the body. To swing the nose toward direction e, the nozzle pushes the tail away:
the thrust first tilts *away* from e (non-minimum-phase, as on real vehicles).

Controller (cascaded): angle error -> desired body rate (capped) -> desired angular acceleration
-> required gimbal deflection, limited in angle and rate. With the engine off, grid fins provide a
limited angular acceleration that keeps the body aligned with the airflow (engine-first).
Transverse inertia and CG arm are held constant (simplification).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AttitudeParams:
    gimbal_max_deg: float = 8.0
    gimbal_rate_deg_s: float = 20.0
    inertia: float = 3.5e6  # transverse moment of inertia about CG, kg m^2
    arm: float = 15.0  # engine gimbal point to CG distance, m
    body_rate_max_deg_s: float = 12.0
    k_theta: float = 2.0  # 1/s, angle error -> rate command
    k_omega: float = 4.0  # 1/s, rate error -> angular accel command
    fin_alpha_max_deg_s2: float = 15.0  # grid-fin authority during the unpowered coast
    brake_frac: float = 0.5  # fraction of angular-accel authority budgeted for stopping rotations

    @classmethod
    def from_dict(cls, d: dict | None) -> AttitudeParams:
        return cls(**{k: v for k, v in (d or {}).items() if not k.startswith("_")})


@dataclass
class AttitudeState:
    b: np.ndarray  # body axis (unit, points nose-up along the vehicle)
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))  # angular velocity, perp. to b
    g: np.ndarray = field(default_factory=lambda: np.zeros(3))  # gimbal deflection, perp. to b

    @property
    def thrust_dir(self) -> np.ndarray:
        s2 = float(self.g @ self.g)
        d = math.sqrt(max(1.0 - s2, 0.0)) * self.b + self.g
        return d / np.linalg.norm(d)

    @property
    def gimbal_deg(self) -> float:
        return math.degrees(math.asin(min(1.0, float(np.linalg.norm(self.g)))))


def _perp(v: np.ndarray, b: np.ndarray) -> np.ndarray:
    return v - (v @ b) * b


def _clip_norm(v: np.ndarray, n: float) -> np.ndarray:
    vn = np.linalg.norm(v)
    return v if vn <= n else v * (n / vn)


def _rotate(b: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Rotate unit vector b by angular velocity w over dt (Rodrigues)."""
    ang = np.linalg.norm(w) * dt
    if ang < 1e-12:
        return b
    k = w / np.linalg.norm(w)
    out = b * math.cos(ang) + np.cross(k, b) * math.sin(ang) + k * (k @ b) * (1 - math.cos(ang))
    return out / np.linalg.norm(out)


class AttitudeModel:
    def __init__(self, p: AttitudeParams):
        self.p = p
        self.g_max = math.sin(math.radians(p.gimbal_max_deg))
        self.dg_max = math.radians(p.gimbal_rate_deg_s)  # per second (small-angle: d|g| ~ d delta)
        self.w_max = math.radians(p.body_rate_max_deg_s)
        self.fin_alpha = math.radians(p.fin_alpha_max_deg_s2)
        self.last: dict = {}  # controller signals of the last command (for probes/plots)

    def alpha_avail(self, thrust: float) -> float:
        """Max angular acceleration (rad/s^2): gimbal torque with the engine on, else grid fins."""
        if thrust > 0:
            return self.p.arm * thrust * self.g_max / self.p.inertia
        return self.fin_alpha

    def _alpha_cmd(self, st: AttitudeState, b_des: np.ndarray, w_ff: np.ndarray | None,
                   thrust: float) -> np.ndarray:
        b = st.b
        axis = np.cross(b, b_des)
        s, c = np.linalg.norm(axis), float(np.clip(b @ b_des, -1, 1))
        ang = math.atan2(s, c)
        # rate command: proportional near the target, but never faster than can still be stopped
        # within the remaining angle using a fraction of the available angular acceleration
        a_brake = self.p.brake_frac * self.alpha_avail(thrust)
        w_mag = min(self.p.k_theta * ang, math.sqrt(2.0 * a_brake * ang))
        w_des = np.zeros(3) if s < 1e-12 else (axis / s) * w_mag
        if w_ff is not None:  # feedforward: rotation rate of the reference itself
            w_des = w_des + _perp(w_ff, b)
        w_des = _clip_norm(w_des, self.w_max)
        return self.p.k_omega * (w_des - st.w)

    def command(self, st: AttitudeState, b_des: np.ndarray, thrust: float,
                w_ff: np.ndarray | None = None, a_ff: np.ndarray | None = None
                ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Actuator commands that steer the body axis toward `b_des`: `(gimbal, None)` with the
        engine lit (`thrust` > 0, N), else `(None, fin angular accel)`. `w_ff` / `a_ff` optional
        feedforward body rate (rad/s) and angular accel (rad/s^2). The plant applies the limits."""
        p, b = self.p, st.b
        alpha_des = self._alpha_cmd(st, b_des, w_ff, thrust)
        if a_ff is not None:
            alpha_des = alpha_des + _perp(a_ff, b)
        self.last = {"alpha_cmd": alpha_des}
        if thrust > 0:
            # tau = I alpha = -arm T (b x g)  ->  b x g = -I alpha / (arm T)  ->  g = (b x g) x b
            c = -p.inertia * alpha_des / (p.arm * thrust)
            return _clip_norm(_perp(np.cross(c, b), b), self.g_max), None
        return None, alpha_des  # engine off: nozzle neutral, grid fins steer


class ReferenceModel:
    """Second-order pointing reference: turns a (possibly jumpy) desired direction into a smooth
    one with bounded rate and angular acceleration. Its rate and acceleration are fed forward to
    the attitude loop, so the gimbal mostly delivers planned torque and feedback only trims."""

    def __init__(self, b0: np.ndarray, omega: float = 1.5, rate_max: float = math.radians(8.0)):
        self.b = b0 / np.linalg.norm(b0)
        self.w = np.zeros(3)
        self.a = np.zeros(3)
        self.k1, self.k2 = omega**2, 2.0 * omega  # critically damped
        self.rate_max = rate_max

    def reset(self, b: np.ndarray, w: np.ndarray | None = None):
        self.b = b / np.linalg.norm(b)
        self.w = np.zeros(3) if w is None else _perp(w, self.b)
        self.a = np.zeros(3)

    def step(self, b_des: np.ndarray, dt: float, a_max: float) -> None:
        b = self.b
        axis = np.cross(b, b_des)
        s, c = np.linalg.norm(axis), float(np.clip(b @ b_des, -1, 1))
        err = np.zeros(3) if s < 1e-12 else (axis / s) * math.atan2(s, c)
        a = _clip_norm(self.k1 * err - self.k2 * self.w, a_max)
        w = _clip_norm(_perp(self.w + a * dt, b), self.rate_max)
        a = (w - self.w) / dt  # acceleration actually applied after the rate limit
        b_new = _rotate(b, w, dt)
        self.b, self.w, self.a = b_new, _perp(w, b_new), a
