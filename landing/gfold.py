"""Phase 1: drag-free fuel-optimal powered descent (G-FOLD, lossless convexification).

One SOCP per fixed final time t_f, then a 1-D search over t_f (spec §3-4).
"""

from __future__ import annotations

import math
import warnings

import cvxpy as cp
import numpy as np
from scipy.linalg import expm

from .trajectory import Trajectory
from .vehicle import Scenario

SOLVER = cp.CLARABEL


class InfeasibleError(RuntimeError):
    pass


def discretize_zoh(alpha: float, g0: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact ZOH discretization of x=[r;v;z], w=[u;sigma] via one matrix exponential (§4.1)."""
    A = np.zeros((7, 7))
    A[0:3, 3:6] = np.eye(3)
    B = np.zeros((7, 4))
    B[3:6, 0:3] = np.eye(3)
    B[6, 3] = -alpha
    c = np.zeros(7)
    c[3] = -g0
    M = np.zeros((12, 12))
    M[:7, :7] = A
    M[:7, 7:11] = B
    M[:7, 11] = c
    Md = expm(M * dt)
    return Md[:7, :7], Md[:7, 7:11], Md[:7, 11]


def solve_fixed_tf(sc: Scenario, tf: float, N: int | None = None) -> Trajectory | None:
    """Solve the convexified fixed-final-time problem. Returns None if infeasible."""
    rk, env = sc.rocket, sc.env
    N = N or sc.gfold.get("N", 40)
    alpha, rho1, rho2 = rk.alpha(env.g0), rk.rho1, rk.rho2
    if sc.m0 - alpha * rho2 * tf <= 0:  # pitfall 7: z0(t) undefined
        return None

    dt = tf / (N - 1)
    t = np.arange(N) * dt
    Ad, Bd, cd = discretize_zoh(alpha, env.g0, dt)

    # Expansion about the lightest-possible mass trajectory (§3.3)
    z0 = np.log(sc.m0 - alpha * rho2 * t)
    z1 = np.log(sc.m0 - alpha * rho1 * t)
    mu1 = rho1 * np.exp(-z0)
    mu2 = rho2 * np.exp(-z0)

    x = cp.Variable((7, N))
    w = cp.Variable((4, N))
    r, v, z = x[0:3], x[3:6], x[6]
    u, sigma = w[0:3], w[3]
    d = z - z0

    cons = [
        x[:, 0] == np.concatenate([sc.r0, sc.v0, [math.log(sc.m0)]]),
        r[:, -1] == 0,
        v[:, -1] == 0,
        z[-1] >= math.log(rk.m_dry),
        x[:, 1:] == Ad @ x[:, :-1] + Bd @ w[:, :-1] + cd[:, None],
        cp.norm(u, axis=0) <= sigma,
        u[0] >= sigma * math.cos(rk.theta_max),
        cp.multiply(mu1, 1 - d + cp.square(d) / 2) <= sigma,
        sigma <= cp.multiply(mu2, 1 - d),
        z >= z0,
        z <= z1,
        cp.norm(r[1:3], axis=0) * math.tan(sc.gamma_gs) <= r[0],
    ]
    cons += attitude_constraints(sc, u, sigma, N)
    pen = attitude_penalties(sc, u, N, sig_nom=0.5 * (rho1 + rho2) / sc.m0, inv_dt=1.0 / dt, n_ctrl=N - 1, sigma=sigma)
    prob = cp.Problem(cp.Maximize(z[-1] - pen), cons)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve(solver=SOLVER)
    except cp.SolverError:
        return None
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        return None

    X, W = x.value, w.value
    return Trajectory(
        t=t,
        r=X[0:3].T,
        v=X[3:6].T,
        m=np.exp(X[6]),
        u=W[0:3].T,
        sigma=W[3],
        hold="zoh",
        info={"solver_status": prob.status, "N": N},
    )


def attitude_constraints(sc: Scenario, u, sigma, N: int, d0=None) -> list:
    """Hard attitude constraint (optional): the first thrust must point within
    `initial_attitude_tol_deg` of the vehicle's current body axis d0 (the gimbal can only deflect
    thrust a few degrees off the body). Equality if the tolerance is 0."""
    if d0 is None:
        d0 = sc.initial_thrust_dir()
    if d0 is None:
        return []
    tol = sc.initial_attitude_tol
    if tol > 0:
        return [d0 @ u[:, 0] >= sigma[0] * math.cos(tol)]
    return [u[:, 0] == sigma[0] * d0]


def attitude_penalties(sc: Scenario, u, N: int, sig_nom: float, inv_dt, n_ctrl: int | None = None, sigma=None):
    """Soft attitude preferences in fuel-equivalent kg (penalties never cause infeasibility):
      - final pose (L1, exact):  w_end * sum_{final nodes} ||E u_k||,   ||E u|| ~ sigma sin(tilt)
      - rotation rate (quadratic): w_rate * sum_k rate_k^2 * dt,  rate_k ~ ||E du_k|| / (sigma dt)
        Quadratic so rotations spread out smoothly (an L1 rotation penalty is a total-variation
        penalty: it limits total rotation but produces sudden jumps).
    `sig_nom` = nominal thrust accel in the units of u; `inv_dt` = 1/node spacing (a constant, or a
    nonnegative cp.Parameter in SCvx where the spacing follows the reference final time).
    Objective mass unit is ln(m/m0) ~ dm/m0. `n_ctrl` = controls actually applied (ZOH: N-1)."""
    n_ctrl = N if n_ctrl is None else n_ctrl
    deg = math.radians(1.0)
    pen = 0.0
    kg_end = sc.constraints.get("end_tilt_penalty_kg_per_deg")
    if kg_end:
        n = int(sc.constraints.get("end_tilt_nodes", 3))
        pen = pen + (kg_end / sc.m0) / (sig_nom * deg) * cp.sum(cp.norm(u[1:3, n_ctrl - n : n_ctrl], axis=0))
    kg_rate = sc.constraints.get("rotation_rate_penalty")  # kg per (deg/s)^2 per second
    if kg_rate:
        du = u[1:3, 1:n_ctrl] - u[1:3, 0 : n_ctrl - 1]
        # sum (|du|/(sig dt deg))^2 dt = sum |du|^2 / (sig deg)^2 * (1/dt)
        pen = pen + (kg_rate / sc.m0) / (sig_nom * deg) ** 2 * inv_dt * cp.sum_squares(du)
    # Penalties on u reward shrinking it below sigma (thrust paid for but not delivered, which
    # also dodges the min-throttle bound), breaking lossless convexification. sigma - e1.u >= 0 is
    # zero only for a tight, vertical thrust; a linear penalty on it removes that incentive.
    kg_slack = sc.constraints.get("slack_penalty_kg_per_pct")
    if kg_slack and sigma is not None and (kg_end or kg_rate):
        pen = pen + (kg_slack / sc.m0) / (sig_nom * 0.01) * cp.sum(sigma[:n_ctrl] - u[0, :n_ctrl])
    return pen


def tf_bracket(sc: Scenario) -> tuple[float, float]:
    """Açıkmeşe & Ploen 2007 final-time bounds (§4.3)."""
    rk = sc.rocket
    alpha = rk.alpha(sc.env.g0)
    t_min = rk.m_dry * np.linalg.norm(sc.v0) / rk.rho2
    t_max = (sc.m0 - rk.m_dry) / (alpha * rk.rho1)
    # keep z0(t) well-defined
    t_max = min(t_max, 0.999 * sc.m0 / (alpha * rk.rho2))
    return float(t_min), float(t_max)


def solve(
    sc: Scenario,
    verbose: bool = False,
    bracket: tuple[float, float] | None = None,
    n_scan: int | None = None,
    n_golden: int | None = None,
) -> Trajectory:
    """Coarse scan over t_f, then golden-section refine around the best grid point.
    `bracket` narrows the search (used for warm re-planning); it is clipped to the theoretical one."""
    n_scan = n_scan or sc.gfold.get("n_scan", 25)
    n_golden = n_golden or sc.gfold.get("n_golden", 20)
    t_lo, t_hi = tf_bracket(sc)
    if bracket is not None:
        t_lo, t_hi = max(t_lo, bracket[0]), min(t_hi, bracket[1])

    cache: dict[float, Trajectory | None] = {}

    def cost(tf: float) -> float:
        if tf not in cache:
            cache[tf] = solve_fixed_tf(sc, tf)
        traj = cache[tf]
        return math.inf if traj is None else -traj.m[-1]

    grid = np.linspace(t_lo, t_hi, n_scan)
    costs = np.array([cost(tf) for tf in grid])
    if not np.isfinite(costs).any():
        raise InfeasibleError("G-FOLD: no feasible final time in bracket")
    i = int(np.argmin(costs))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n_scan - 1)]

    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = cost(c), cost(d)
    for _ in range(n_golden):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = cost(d)

    best_tf = min((tf for tf in cache if cache[tf] is not None), key=cost)
    traj = cache[best_tf]
    assert traj is not None
    traj.info["max_dir_rate_deg_s"] = math.degrees(max_dir_rate(traj, skip_last=True))
    traj.info.update(
        tf_bracket=[t_lo, t_hi],
        scan_tf=grid.tolist(),
        scan_mf=[None if not np.isfinite(c_) else -c_ for c_ in costs],
        n_solves=len(cache),
    )
    traj.info.update(check_solution(sc, traj))
    if verbose:
        print(f"[gfold] t_f*={traj.tf:.3f}s m_f={traj.m[-1]:.1f}kg ({len(cache)} SOCPs)")
    return traj


def max_dir_rate(traj: Trajectory, skip_last: bool = False) -> float:
    """Max thrust-direction rate between nodes (rad/s)."""
    u = traj.u[:-1] if skip_last else traj.u
    d = u / np.maximum(np.linalg.norm(u, axis=1), 1e-9)[:, None]
    ang = np.arccos(np.clip(np.sum(d[1:] * d[:-1], axis=1), -1, 1))
    dts = np.diff(traj.t[: len(u)])
    return float(np.max(ang / dts)) if len(ang) else 0.0


def check_solution(sc: Scenario, traj: Trajectory, skip_last: bool = True) -> dict[str, float]:
    """Lossless-convexification tightness and exact thrust-bound checks (§8, pitfall 1)."""
    rk = sc.rocket
    sl = slice(0, len(traj.t) - 1) if skip_last else slice(None)
    un = np.linalg.norm(traj.u[sl], axis=1)
    sig = traj.sigma[sl]
    tight = float(np.max(np.abs(un - sig) / np.maximum(sig, 1e-12)))
    thrust = sig * traj.m[sl]
    viol = float(max(np.max(rk.rho1 - thrust), np.max(thrust - rk.rho2), 0.0))
    return {"tightness_max": tight, "thrust_bound_violation_N": viol}
