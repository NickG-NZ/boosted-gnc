"""Inner-loop tracking MPC: attitude-aware, fixed final time, linearized about the guidance plan.

The outer loop (G-FOLD / SCvx) fixes t_f, ignition, mass and the trajectory shape. This loop
re-solves a convex QP over the remaining time (shrinking horizon) at ~20 Hz and directly
commands throttle and gimbal, trading position error against tilt, body rate and gimbal effort.

Model (pad frame [up, x, y]; body axis within ~20 deg of vertical, so attitude is described by
the *horizontal* components of the body axis b_h, its rate s = d(b_h)/dt and the gimbal g_h):

    r' = v
    v' = (T/m) t_hat + g0 + D/m,        t_hat ~ b + g
    b_h'' = -(l T / I) g_h              (from I w' = -l T (b x g), small body rates)
    g_h' = u_g                          (gimbal rate is the control)

Linearized about the plan (r_bar, v_bar, thrust T_bar = m |u_bar|, direction b_bar = u_bar/|u_bar|)
with error state x = [dr(3), dv(3), db_h(2), s(2), g_h(2)] and control u = [dT/T_max, u_g(2)]:

    dr'   = dv
    dv_up' = (T_max bu / m) dT - (a / bu) b_bar_h . (db_h + g_h)
    dv_h'  = (T_max b_bar_h / m) dT + a (db_h + g_h)              a = T_bar / m,  bu = b_bar_up
    db_h' = s - b_bar_h'
    s'    = -(l T_bar / I) g_h
    g_h'  = u_g

plus the defect c = f(x_bar, u_bar) - x_bar' (gravity, drag at the nominal state, plan rotation).
A has no feedback from later to earlier states (g -> s -> b -> v -> r), so it is nilpotent
(A^5 = 0) and the zero-order-hold discretization is an exact finite series. Coefficients are
frozen at each interval's midpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from gnc.control.attitude import AttitudeParams, AttitudeState
from gnc.dynamics import TruthModel
from gnc.guidance.trajectory import Trajectory
from gnc.vehicle import Scenario

NX, NU = 12, 3
IR, IV, IB, IS, IG = slice(0, 3), slice(3, 6), slice(6, 8), slice(8, 10), slice(10, 12)
_FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0]


@dataclass(frozen=True)
class MPCParams:
    N: int = 30  # horizon intervals (shrinking horizon to the plan's t_f)
    dt0: float = 0.05  # first interval = re-solve period (s); later intervals grow geometrically
    stop_tgo: float = 0.15  # stop re-solving this close to t_f (hold the last solution)
    # terminal weights (absolute errors at t_f)
    w_r_lat: float = 20.0  # per m^2
    w_r_up: float = 500.0  # per m^2 (engine cuts at t_f: any height left is a free fall)
    w_v: float = 20.0  # per (m/s)^2
    w_tilt: float = 3.0e4  # per rad^2 (1 deg ~ 9)
    w_rate: float = 3.0e4  # per (rad/s)^2 of d(b_h)/dt
    w_gimbal_term: float = 1.0e3  # per rad^2
    # running weights (per second)
    w_dr: float = 0.02  # deviation from the plan position
    w_dv: float = 0.05
    w_db: float = 300.0  # attitude deviation from the plan's thrust direction, per rad^2
    w_cone: float = 1.0e6  # exact (L1) penalty per rad of tilt beyond the pointing cone
    huber_r: float = 5.0  # terminal position/velocity costs turn linear beyond this (m, m/s),
    huber_v: float = 5.0  # so a hopeless miss cannot outweigh staying upright
    w_ug: float = 20.0  # gimbal rate effort, per (rad/s)^2
    w_dT: float = 1.0  # throttle deviation, per (fraction of T_max)^2

    @classmethod
    def from_dict(cls, d: dict | None) -> MPCParams:
        return cls(**{k: v for k, v in (d or {}).items() if not k.startswith("_")})


def nominal_u(plan: Trajectory, t: float | np.ndarray) -> np.ndarray:
    """Smooth nominal thrust acceleration. ZOH plans jump at nodes, which would demand
    instantaneous rotations; interpolate their segment midpoints linearly instead."""
    t = np.atleast_1d(np.asarray(t, float))
    if plan.hold == "zoh":
        tm = 0.5 * (plan.t[:-1] + plan.t[1:])
        um = plan.u[:-1]
    else:
        tm, um = plan.t, plan.u
    return np.stack([np.interp(t, tm, um[:, i]) for i in range(3)], axis=1)


def time_grid(t_go: float, N: int, dt0: float) -> np.ndarray:
    """N interval lengths summing to t_go: the first is dt0, then geometric growth."""
    if t_go <= N * dt0:
        return np.full(N, t_go / N)
    lo, hi = 1.0, 2.0
    for _ in range(60):  # bisect the growth ratio q: dt0 (q^N - 1)/(q - 1) = t_go
        q = 0.5 * (lo + hi)
        (lo, hi) = (q, hi) if dt0 * (q**N - 1) / (q - 1) < t_go else (lo, q)
    d = dt0 * q ** np.arange(N)
    return d * (t_go / d.sum())


def continuous_model(T: float, m: float, T_max: float, bh: np.ndarray, k_att: float):
    """Analytic Jacobians A (12x12), B (12x3) at thrust T, mass m, nominal direction b_bar."""
    bu = math.sqrt(max(1.0 - float(bh @ bh), 1e-6))
    a = T / m
    A = np.zeros((NX, NX))
    B = np.zeros((NX, NU))
    A[IR, IV] = np.eye(3)
    A[3, IB] = A[3, IG] = -a * bh / bu  # vertical loss from tilting
    A[4:6, IB] = A[4:6, IG] = a * np.eye(2)  # lateral force from body tilt and gimbal
    A[IB, IS] = np.eye(2)
    A[IS, IG] = -k_att * np.eye(2)  # gimbal torque: pushes the tail, nose swings the other way
    B[3, 0] = T_max * bu / m
    B[4:6, 0] = T_max * bh / m
    B[IG, 1:3] = np.eye(2)
    return A, B


def discretize(A: np.ndarray, B: np.ndarray, c: np.ndarray, dt: float):
    """Exact ZOH for nilpotent A: exp(A dt) = sum_{k<5} (A dt)^k / k!."""
    Ad = np.eye(NX)
    G = np.eye(NX) * dt  # integral of exp(A s) ds over [0, dt]
    Ak = np.eye(NX)
    for k in range(1, 5):
        Ak = Ak @ A
        Ad = Ad + Ak * dt**k / _FACT[k]
        G = G + Ak * dt ** (k + 1) / _FACT[k + 1]
    return Ad, G @ B, G @ c


@dataclass
class MPCSolution:
    t: np.ndarray  # node times relative to the solve (N+1)
    T: np.ndarray  # thrust magnitude per interval (N)
    g_h: np.ndarray  # absolute horizontal gimbal per node (N+1, 2)
    x: np.ndarray  # predicted error states (N+1, 12)
    info: dict = field(default_factory=dict)

    def command(self, ts: float) -> tuple[float, np.ndarray]:
        """Thrust (held per interval) and gimbal target at time ts after the solve."""
        k = int(np.clip(np.searchsorted(self.t, ts, side="right") - 1, 0, len(self.T) - 1))
        gh = np.array([np.interp(ts, self.t, self.g_h[:, i]) for i in range(2)])
        return float(self.T[k]), gh


class TrackingMPC:
    def __init__(self, sc: Scenario, drag: bool = True, params: MPCParams | None = None):
        self.sc, rk = sc, sc.rocket
        self.p = p = params or MPCParams.from_dict(getattr(sc, "mpc", None))
        ap = AttitudeParams.from_dict(rk.attitude)
        self.k_per_T = ap.arm / ap.inertia  # angular accel per unit thrust per unit gimbal
        self.g_max = math.sin(math.radians(ap.gimbal_max_deg))
        self.dg_max = math.radians(ap.gimbal_rate_deg_s)
        self.model = TruthModel(rk, sc.env, drag=drag)  # nominal drag model (no perturbations)
        self.alpha = rk.alpha(sc.env.g0)
        self._build()

    # ---------------- problem (DPP: compiled once, parameters rewritten every solve) --------
    def _build(self):
        p, rk, N = self.p, self.sc.rocket, self.p.N
        X = cp.Variable((N + 1, NX))
        U = cp.Variable((N, NU))
        self.Ad = [cp.Parameter((NX, NX)) for _ in range(N)]
        self.Bd = [cp.Parameter((NX, NU)) for _ in range(N)]
        self.cd = [cp.Parameter(NX) for _ in range(N)]
        self.x0 = cp.Parameter(NX)
        self.Tbar = cp.Parameter(N)  # nominal thrust, fraction of T_max
        self.gcap = cp.Parameter(N, nonneg=True)  # horizontal gimbal bound per node
        self.sqdt = cp.Parameter(N, nonneg=True)  # sqrt(interval length): running-cost weights
        self.r_end = cp.Parameter(3)  # plan values at t_f (errors are relative to them)
        self.v_end = cp.Parameter(3)
        self.b_end = cp.Parameter(2)
        self.b_nom = cp.Parameter((N, 2))  # plan's horizontal thrust direction at nodes 1..N
        sin_cone = math.sin(rk.theta_max)

        cons = [X[0] == self.x0]
        cost = 0
        for k in range(N):
            cons += [
                X[k + 1] == self.Ad[k] @ X[k] + self.Bd[k] @ U[k] + self.cd[k],
                self.Tbar[k] + U[k, 0] >= rk.thr_min,
                self.Tbar[k] + U[k, 0] <= 1.0,
                cp.norm(U[k, 1:3]) <= self.dg_max,
                cp.norm(X[k + 1, IG]) <= self.gcap[k],
            ]
            w = self.sqdt[k]
            # soft pointing cone keeps the vehicle where the small-tilt model holds
            cost += p.w_cone * cp.pos(cp.norm(self.b_nom[k] + X[k + 1, IB]) - sin_cone)
            cost += (p.w_dr * cp.sum_squares(w * X[k + 1, IR]) + p.w_dv * cp.sum_squares(w * X[k + 1, IV])
                     + p.w_db * cp.sum_squares(w * X[k + 1, IB])
                     + p.w_ug * cp.sum_squares(w * U[k, 1:3]) + p.w_dT * cp.sum_squares(w * U[k, 0]))
        rN, vN, bN = self.r_end + X[N, IR], self.v_end + X[N, IV], self.b_end + X[N, IB]
        cost += (p.w_r_lat * cp.sum(cp.huber(rN[1:3], p.huber_r)) + p.w_r_up * cp.huber(rN[0], p.huber_r)
                 + p.w_v * cp.sum(cp.huber(vN, p.huber_v)) + p.w_tilt * cp.sum_squares(bN)
                 + p.w_rate * cp.sum_squares(X[N, IS]) + p.w_gimbal_term * cp.sum_squares(X[N, IG]))
        self.X, self.U = X, U
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        assert self.prob.is_dpp()

    # ---------------- linearization about the plan ----------------
    def linearize(self, plan: Trajectory, tau: float, m_now: float) -> dict:
        p, rk, N = self.p, self.sc.rocket, self.p.N
        dts = time_grid(plan.tf - tau, N, p.dt0)
        tk = tau + np.r_[0.0, np.cumsum(dts)]
        tm = tk[:-1] + 0.5 * dts
        rb, vb = map(np.array, zip(*(plan.state_at(t) for t in tk)))
        rm, vm = map(np.array, zip(*(plan.state_at(t) for t in tm)))
        un, um = nominal_u(plan, tk), nominal_u(plan, tm)
        bh_n = (un / np.linalg.norm(un, axis=1)[:, None])[:, 1:3]
        g0 = self.sc.env.g
        Ad, Bd, cd, Tbar, gcap = [], [], [], np.zeros(N), np.zeros(N)
        m = m_now
        for k in range(N):
            h = dts[k]
            ua = float(np.linalg.norm(um[k]))
            m_mid = m / (1.0 + 0.5 * self.alpha * ua * h)  # thrust ~ m |u|  ->  mass decays
            T = m_mid * ua
            bh = um[k, 1:3] / ua
            A, B = continuous_model(T, m_mid, rk.T_max, bh, self.k_per_T * T)
            D = self.model.drag_force(rm[k], vm[k], True)
            c = np.zeros(NX)
            c[IR] = vm[k] - (rb[k + 1] - rb[k]) / h
            c[IV] = um[k] + g0 + D / m_mid - (vb[k + 1] - vb[k]) / h
            c[IB] = -(bh_n[k + 1] - bh_n[k]) / h  # the plan rotates: keep up or fall behind
            a_d, b_d, c_d = discretize(A, B, c, h)
            Ad.append(a_d), Bd.append(b_d), cd.append(c_d)
            Tbar[k] = T / rk.T_max
            gcap[k] = self.g_max * math.sqrt(max(1.0 - float(bh @ bh), 0.0))
            m -= self.alpha * T * h
        return dict(dts=dts, tk=tk, rb=rb, vb=vb, bh=bh_n, Ad=Ad, Bd=Bd, cd=cd, Tbar=Tbar, gcap=gcap)

    def solve(self, plan: Trajectory, tau: float, r: np.ndarray, v: np.ndarray, m: float,
              ast: AttitudeState) -> MPCSolution | None:
        lin = self.linearize(plan, tau, m)
        b, w = ast.b, ast.w
        gh0 = ast.g[1:3]
        gn = np.linalg.norm(gh0)
        if gn > lin["gcap"][0]:
            gh0 = gh0 * (lin["gcap"][0] / gn)
        x0 = np.concatenate([r - lin["rb"][0], v - lin["vb"][0], b[1:3] - lin["bh"][0],
                             np.cross(w, b)[1:3], gh0])
        for k in range(self.p.N):
            self.Ad[k].value, self.Bd[k].value, self.cd[k].value = lin["Ad"][k], lin["Bd"][k], lin["cd"][k]
        self.x0.value = x0
        self.Tbar.value = lin["Tbar"]
        self.gcap.value = lin["gcap"]
        self.sqdt.value = np.sqrt(lin["dts"])
        self.r_end.value, self.v_end.value, self.b_end.value = lin["rb"][-1], lin["vb"][-1], lin["bh"][-1]
        self.b_nom.value = lin["bh"][1:]
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return None
        if self.prob.status not in ("optimal", "optimal_inaccurate") or self.X.value is None:
            return None
        X, U = self.X.value, self.U.value
        T = np.clip((lin["Tbar"] + U[:, 0]) * self.sc.rocket.T_max, self.sc.rocket.T_min, self.sc.rocket.T_max)
        return MPCSolution(t=lin["tk"] - tau, T=T, g_h=X[:, IG], x=X,
                           info={"cost": float(self.prob.value), "solve_s": self.prob.solver_stats.solve_time})


def embed_gimbal(g_h: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3D gimbal vector perpendicular to b with the given horizontal components."""
    return np.array([-(g_h @ b[1:3]) / max(b[0], 1e-3), g_h[0], g_h[1]])
