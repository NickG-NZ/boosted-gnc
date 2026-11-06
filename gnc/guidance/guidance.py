"""Guidance laws as pluggable objects: an initial plan + optional closed-loop re-planning.

Each guidance object exposes:
    plan()                                  -> Trajectory from the scenario's initial state
    plan_from(r, v, m, att=None) -> Trajectory | None  cold-start plan from an arbitrary state
                                            (None = infeasible); `att` = current body axis
    replan(r, v, m, prev, tau, att=None) -> Trajectory | None   from the current state,
                                            warm-started from `prev` at plan time `tau`
    models_drag                             whether the guidance's internal model includes drag
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gnc.guidance import gfold
from gnc.guidance.scvx import SCvxSolver
from gnc.guidance.trajectory import Trajectory
from gnc.vehicle import Scenario


def shift_trajectory(prev: Trajectory, tau: float, n: int | None = None) -> Trajectory:
    """Remaining portion of `prev` from plan time `tau`, re-timed to start at 0 (warm start)."""
    n = n or len(prev.t)
    t_new = np.linspace(tau, prev.tf, n)
    interp = lambda a: np.stack([np.interp(t_new, prev.t, a[:, i]) for i in range(a.shape[1])], 1)
    u = np.array([prev.u_at(t) for t in t_new])
    return Trajectory(
        t=t_new - tau,
        r=interp(prev.r),
        v=interp(prev.v),
        m=np.interp(t_new, prev.t, prev.m),
        u=u,
        sigma=np.linalg.norm(u, axis=1),
        hold="foh",
    )


def is_flyable(traj: Trajectory, tol: float = 1e-3) -> bool:
    """Lossless convexification must be tight (||u|| = sigma) for the plan to be physical; slack
    means the plan 'pays for' thrust it does not deliver, effectively throttling below the
    engine minimum (typically when igniting too early to stop at the pad with min thrust > weight)."""
    return traj.info.get("tightness_max", 0.0) <= tol


def straight_line_guess(sc: Scenario, N: int = 30) -> Trajectory:
    """Classic SCvx initialization: states interpolate linearly to the pad, constant decel."""
    rk, g0 = sc.rocket, sc.env.g0
    a_brake = 0.7 * rk.rho2 / sc.m0 - g0
    tf = float(np.linalg.norm(sc.v0) / max(a_brake, 1.0)) * 1.3
    s = np.linspace(0, 1, N)[:, None]
    u = np.tile(-sc.v0 / tf - sc.env.g, (N, 1))
    alpha = rk.alpha(g0)
    m = sc.m0 - alpha * np.linalg.norm(u[0]) * sc.m0 * s[:, 0] * tf
    return Trajectory(
        t=s[:, 0] * tf, r=(1 - s) * sc.r0, v=(1 - s) * sc.v0, m=np.maximum(m, rk.m_dry), u=u,
        sigma=np.linalg.norm(u, axis=1), hold="foh",
    )


class GFoldGuidance:
    """Drag-free lossless-convexification guidance (Phase 1)."""

    name = "gfold"
    models_drag = False

    def __init__(self, sc: Scenario):
        self.sc = sc

    def plan(self, verbose: bool = False) -> Trajectory:
        return gfold.solve(self.sc, verbose=verbose)

    def plan_from(self, r, v, m, att=None) -> Trajectory | None:
        sc = replace(self.sc, r0=np.asarray(r, float), v0=np.asarray(v, float), m0=float(m), att0=att)
        try:
            traj = gfold.solve(sc)
        except gfold.InfeasibleError:
            return None
        return traj if is_flyable(traj) else None

    def replan(self, r, v, m, prev: Trajectory, tau: float, att=None) -> Trajectory | None:
        sc = replace(self.sc, r0=np.asarray(r), v0=np.asarray(v), m0=float(m), att0=att)
        t_go = prev.tf - tau
        # fallback: if the plan must start from the current attitude and none exists, drop that
        # requirement (the attitude dynamics in the sim still apply) rather than give up
        relaxed = replace(sc, constraints={**sc.constraints, "constrain_initial_attitude": False})
        # warm: narrow t_f bracket around the previous time-to-go; then the full theoretical
        # bracket (attitude limits can lengthen the burn well beyond the old t_go)
        attempts = [
            (sc, (0.6 * t_go, 1.6 * t_go), False),
            (sc, None, False),
            (relaxed, None, True),
        ]
        for s_try, bracket, is_relaxed in attempts:
            try:
                traj = gfold.solve(s_try, bracket=bracket, n_scan=9 if bracket else 15, n_golden=10)
                traj.info["relaxed_attitude"] = is_relaxed
                return traj
            except gfold.InfeasibleError:
                continue
        return None


class SCvxGuidance:
    """Drag-aware successive-convexification guidance (Phase 2), warm-started from G-FOLD."""

    name = "scvx"
    models_drag = True

    def __init__(self, sc: Scenario, replan_iters: int = 8):
        self.sc = sc
        self.solver = SCvxSolver(sc)
        self.replan_iters = replan_iters

    def plan(self, verbose: bool = False, init: Trajectory | None = None) -> Trajectory:
        init = init or gfold.solve(self.sc)
        return self.solver.solve(init, verbose=verbose)

    def plan_from(self, r, v, m, att=None) -> Trajectory | None:
        """Initialize from G-FOLD at the same state when it is feasible (drag-free), otherwise from
        a straight-line guess (drag can make states feasible that are infeasible in vacuum)."""
        sc = replace(self.sc, r0=np.asarray(r, float), v0=np.asarray(v, float), m0=float(m), att0=att)
        try:
            init = gfold.solve(sc)
        except gfold.InfeasibleError:
            init = straight_line_guess(sc)
        try:
            traj = self.solver.solve(init, r0=sc.r0, v0=sc.v0, m0=sc.m0, att0=att, max_iters=20)
        except RuntimeError:
            return None
        ok = traj.info["converged"] or traj.info["history"][-1].get("nu_l1", 1.0) < 1e-5
        return traj if ok and traj.info["defect_l1"] < 1e-4 and is_flyable(traj) else None

    def replan(self, r, v, m, prev: Trajectory, tau: float, att=None) -> Trajectory | None:
        guess = shift_trajectory(prev, tau)
        try:
            new = self.solver.solve(guess, r0=r, v0=v, m0=m, att0=att, max_iters=self.replan_iters)
        except RuntimeError:
            return None
        # virtual control still large -> plan not dynamically consistent; reject it
        if new.info["history"] and new.info["history"][-1].get("nu_l1", 0.0) > 1e-3:
            return None
        return new


GUIDANCE = {"gfold": GFoldGuidance, "scvx": SCvxGuidance}
