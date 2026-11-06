"""Phase 2: successive convexification (SCvx) with aerodynamic drag and free final time (spec §5).

Everything inside this module runs in nondimensional units (§5.1); inputs/outputs are SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np

from .trajectory import Trajectory
from .vehicle import Scenario

SOLVER = cp.CLARABEL


# --------------------------------------------------------------------------- scaling / model


@dataclass(frozen=True)
class Scaling:
    L: float  # length (m)
    M: float  # mass (kg)
    Tt: float  # time (s)
    g0: float

    @property
    def V(self) -> float:
        return self.L / self.Tt

    @classmethod
    def for_scenario(cls, sc: Scenario) -> Scaling:
        L = float(np.linalg.norm(sc.r0)) or 1000.0
        return cls(L=L, M=sc.m0, Tt=math.sqrt(L / sc.env.g0), g0=sc.env.g0)


@dataclass(frozen=True)
class ScaledModel:
    """Nondimensional dynamics f(x, w); x=[r,v,z], w=[u,sigma], accel unit = g0."""

    alpha: float
    kS: float  # 0.5 * rho0 * S_ref * C_D_eff * L / M
    Hs: float  # scale height / L
    wind: np.ndarray
    eps: float = 1e-6

    @classmethod
    def build(cls, sc: Scenario, s: Scaling) -> ScaledModel:
        rk, env = sc.rocket, sc.env
        return cls(
            alpha=rk.alpha(env.g0) * env.g0 * s.Tt,  # = Tt / Isp
            kS=0.5 * env.rho0 * rk.CdS_thrusting * s.L / s.M,
            Hs=env.H / s.L,
            wind=env.wind_vec / s.V,
        )

    def _drag_terms(self, x: np.ndarray):
        r, v, z = x[:, 0:3], x[:, 3:6], x[:, 6]
        vr = v - self.wind
        nv = np.linalg.norm(vr, axis=1) + self.eps
        k = self.kS * np.exp(-r[:, 0] / self.Hs)
        ez = np.exp(-z)
        D = -(k * nv)[:, None] * vr
        return vr, nv, k, ez, D

    def f(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Batched f: x (K,7), w (K,4) -> (K,7)."""
        _, _, _, ez, D = self._drag_terms(x)
        out = np.empty_like(x)
        out[:, 0:3] = x[:, 3:6]
        out[:, 3:6] = w[:, 0:3] + ez[:, None] * D
        out[:, 3] -= 1.0  # gravity
        out[:, 6] = -self.alpha * w[:, 3]
        return out

    def A(self, x: np.ndarray) -> np.ndarray:
        """Batched df/dx (K,7,7) (§5.3)."""
        vr, nv, k, ez, D = self._drag_terms(x)
        K = x.shape[0]
        A = np.zeros((K, 7, 7))
        A[:, 0:3, 3:6] = np.eye(3)
        A[:, 3:6, 0] = (ez * k / self.Hs * nv)[:, None] * vr  # -k'(h) = k/H
        outer = vr[:, :, None] * vr[:, None, :] / nv[:, None, None]
        A[:, 3:6, 3:6] = -(k * ez)[:, None, None] * (nv[:, None, None] * np.eye(3) + outer)
        A[:, 3:6, 6] = -ez[:, None] * D
        return A

    @property
    def B(self) -> np.ndarray:
        B = np.zeros((7, 4))
        B[3:6, 0:3] = np.eye(3)
        B[6, 3] = -self.alpha
        return B


# --------------------------------------------------------------------------- FOH discretization


def discretize_foh(model: ScaledModel, X: np.ndarray, W: np.ndarray, p: float, n_sub: int = 10):
    """Linearize about (X, W, p) and discretize exactly with first-order hold (§5.4).

    X: (N,7), W: (N,4). Integrates all N-1 intervals in parallel (multiple shooting) with RK4.
    Returns A (K,7,7), Bm (K,7,4), Bp (K,7,4), S (K,7), c (K,7), x_end (K,7) where x_end is the
    nonlinear flow from X[k] (so defects are X[1:] - x_end).
    """
    K = X.shape[0] - 1
    dtau = 1.0 / K
    B = model.B
    wk, wk1 = W[:-1], W[1:]
    I7 = np.broadcast_to(np.eye(7), (K, 7, 7))

    def rhs(s, st):
        xb, Phi, Bm, Bp, S, c = st
        lm, lp = 1.0 - s, s
        w = lm * wk + lp * wk1
        f = model.f(xb, w)
        Ac = p * model.A(xb)
        pB = p * B
        dxb = p * f
        dPhi = Ac @ Phi
        dBm = Ac @ Bm + lm * pB
        dBp = Ac @ Bp + lp * pB
        dS = np.einsum("kij,kj->ki", Ac, S) + f
        # residual F - A xb - B w - S p, with F = S p
        dc = np.einsum("kij,kj->ki", Ac, c - xb) - w @ pB.T
        # derivatives w.r.t. tau; s = (tau - tau_k)/dtau so d/ds = dtau * d/dtau
        return [dtau * a for a in (dxb, dPhi, dBm, dBp, dS, dc)]

    st = [X[:-1].copy(), I7.copy(), np.zeros((K, 7, 4)), np.zeros((K, 7, 4)), np.zeros((K, 7)), np.zeros((K, 7))]
    h = 1.0 / n_sub
    for i in range(n_sub):
        s0 = i * h
        k1 = rhs(s0, st)
        k2 = rhs(s0 + h / 2, [a + h / 2 * b for a, b in zip(st, k1)])
        k3 = rhs(s0 + h / 2, [a + h / 2 * b for a, b in zip(st, k2)])
        k4 = rhs(s0 + h, [a + h * b for a, b in zip(st, k3)])
        st = [a + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(st, k1, k2, k3, k4)]
    xb, Phi, Bm, Bp, S, c = st
    return Phi, Bm, Bp, S, c, xb


# --------------------------------------------------------------------------- convex subproblem


class Subproblem:
    """DPP-compliant CVXPY problem built once; only parameters change between iterations."""

    def __init__(self, sc: Scenario, s: Scaling, model: ScaledModel, N: int, w_nu: float):
        rk = sc.rocket
        K = N - 1
        self.N = N
        x = self.x = cp.Variable((7, N))
        w = self.w = cp.Variable((4, N))
        p = self.p = cp.Variable()
        nu = self.nu = cp.Variable((7, K))
        dz = cp.Variable(N)

        self.Ak = [cp.Parameter((7, 7)) for _ in range(K)]
        self.Bmk = [cp.Parameter((7, 4)) for _ in range(K)]
        self.Bpk = [cp.Parameter((7, 4)) for _ in range(K)]
        self.Sk = cp.Parameter((7, K))
        self.ck = cp.Parameter((7, K))
        self.xbar = cp.Parameter((7, N))
        self.wbar = cp.Parameter((4, N))
        self.pbar = cp.Parameter()
        self.zbar = cp.Parameter(N)
        self.mu1 = cp.Parameter(N, nonneg=True)
        self.mu2 = cp.Parameter(N, nonneg=True)
        self.eta = cp.Parameter(nonneg=True)
        self.x0 = cp.Parameter(7)
        self.use_d0 = sc.initial_thrust_dir() is not None
        self.d0 = cp.Parameter(3)

        r, v, z = x[0:3], x[3:6], x[6]
        u, sigma = w[0:3], w[3]
        cons = [
            x[:, 0] == self.x0,
            r[:, -1] == 0,
            v[:, -1] == 0,
            z >= math.log(rk.m_dry / s.M),
            p >= 1e-2,
            cp.norm(u, axis=0) <= sigma,
            u[0] >= sigma * math.cos(rk.theta_max),
            dz == z - self.zbar,
            cp.multiply(self.mu1, 1 - dz + cp.square(dz) / 2) <= sigma,
            sigma <= cp.multiply(self.mu2, 1 - dz),
            cp.norm(r[1:3], axis=0) * math.tan(sc.gamma_gs) <= r[0],
            # trust region (§5.5)
            cp.max(cp.abs(x - self.xbar), axis=0) + cp.max(cp.abs(w - self.wbar), axis=0) <= self.eta,
            cp.abs(p - self.pbar) <= self.eta,
        ]
        for k in range(K):
            cons.append(
                x[:, k + 1]
                == self.Ak[k] @ x[:, k]
                + self.Bmk[k] @ w[:, k]
                + self.Bpk[k] @ w[:, k + 1]
                + self.Sk[:, k] * p
                + self.ck[:, k]
                + nu[:, k]
            )
        from .gfold import attitude_penalties

        if self.use_d0:  # first thrust within a small cone of the current body axis
            tol = sc.initial_attitude_tol
            if tol > 0:
                cons.append(self.d0 @ u[:, 0] >= sigma[0] * math.cos(tol))
            else:
                cons.append(u[:, 0] == sigma[0] * self.d0)
        sig_nom = 0.5 * (rk.rho1 + rk.rho2) / sc.m0 / s.g0
        self.inv_dt = cp.Parameter(nonneg=True)  # 1 / node spacing (scaled), from the reference t_f
        self.objective = -z[-1] + w_nu * cp.sum(cp.abs(nu)) + attitude_penalties(sc, u, N, sig_nom, self.inv_dt, sigma=sigma)
        self.prob = cp.Problem(cp.Minimize(self.objective), cons)
        self.rho1 = rk.rho1 / (s.M * s.g0)
        self.rho2 = rk.rho2 / (s.M * s.g0)
        assert self.prob.is_dpp()

    def set(self, X, W, p, lin, eta, x0, d0=None):
        self.x0.value = x0
        self.d0.value = d0 if d0 is not None else np.array([1.0, 0.0, 0.0])
        A, Bm, Bp, S, c, _ = lin
        for k in range(self.N - 1):
            self.Ak[k].value = A[k]
            self.Bmk[k].value = Bm[k]
            self.Bpk[k].value = Bp[k]
        self.Sk.value = S.T
        self.ck.value = c.T
        self.xbar.value = X.T
        self.wbar.value = W.T
        self.pbar.value = p
        self.zbar.value = X[:, 6]
        self.mu1.value = self.rho1 * np.exp(-X[:, 6])
        self.mu2.value = self.rho2 * np.exp(-X[:, 6])
        self.eta.value = eta
        self.inv_dt.value = (self.N - 1) / p

    def solve(self):
        self.prob.solve(solver=SOLVER, warm_start=True)
        if self.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        return self.x.value.T, self.w.value.T, float(self.p.value), self.nu.value.T, float(self.prob.value)


# --------------------------------------------------------------------------- SCvx driver


def _merit(X, defects, w_nu):
    return -X[-1, 6] + w_nu * np.abs(defects).sum()


def initial_guess_from(traj: Trajectory, s: Scaling, N: int):
    tau = np.linspace(0, 1, N)
    t = tau * traj.tf
    X = np.zeros((N, 7))
    for i in range(3):
        X[:, i] = np.interp(t, traj.t, traj.r[:, i]) / s.L
        X[:, 3 + i] = np.interp(t, traj.t, traj.v[:, i]) / s.V
    X[:, 6] = np.log(np.interp(t, traj.t, traj.m) / s.M)
    W = np.zeros((N, 4))
    for j, tj in enumerate(t):
        W[j, 0:3] = traj.u_at(tj) / s.g0
    W[:, 3] = np.linalg.norm(W[:, 0:3], axis=1)
    return X, W, traj.tf / s.Tt


class SCvxSolver:
    """Reusable SCvx solver: the CVXPY problem is compiled once (scaling fixed by the scenario),
    so it can be called repeatedly from new initial states for closed-loop re-planning."""

    DEFAULTS = {
        "N": 30, "max_iters": 30, "w_nu": 1e4, "eta0": 1.0, "eta_min": 1e-3, "eta_max": 10.0,
        "r0": 0.0, "r1": 0.25, "r2": 0.7, "a": 2.0, "b": 3.2, "eps_nu": 1e-6, "eps_x": 1e-4,
    }

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.cfg = {**self.DEFAULTS, **sc.scvx}
        self.s = Scaling.for_scenario(sc)
        self.model = ScaledModel.build(sc, self.s)
        self.sub = Subproblem(sc, self.s, self.model, self.cfg["N"], self.cfg["w_nu"])

    def solve(
        self,
        init: Trajectory,
        r0: np.ndarray | None = None,
        v0: np.ndarray | None = None,
        m0: float | None = None,
        att0: np.ndarray | None = None,
        max_iters: int | None = None,
        verbose: bool = False,
    ) -> Trajectory:
        sc, s, model, sub, cfg = self.sc, self.s, self.model, self.sub, self.cfg
        r0 = sc.r0 if r0 is None else r0
        v0 = sc.v0 if v0 is None else v0
        m0 = sc.m0 if m0 is None else m0
        x0 = np.concatenate([r0 / s.L, v0 / s.V, [math.log(m0 / s.M)]])
        d0 = replace(sc, v0=np.asarray(v0, float), att0=att0).initial_thrust_dir()
        N, w_nu = cfg["N"], cfg["w_nu"]

        X, W, p = initial_guess_from(init, s, N)
        lin = discretize_foh(model, X, W, p)
        J_ref = _merit(X, X[1:] - lin[5], w_nu)
        eta = cfg["eta0"]
        history = []
        converged = False
        accepted_any = False

        for it in range(max_iters or cfg["max_iters"]):
            sub.set(X, W, p, lin, eta, x0, d0)
            sol = sub.solve()
            if sol is None:
                eta = max(eta / cfg["a"], cfg["eta_min"])
                history.append({"iter": it, "status": "subproblem_failed", "eta": eta})
                continue
            Xs, Ws, ps, nus, L_pred = sol
            lin_s = discretize_foh(model, Xs, Ws, ps)
            defects = Xs[1:] - lin_s[5]
            J_s = _merit(Xs, defects, w_nu)
            denom = J_ref - L_pred
            ratio = (J_ref - J_s) / denom if abs(denom) > 1e-12 else 1.0
            nu_norm = float(np.abs(nus).sum())
            dx = float(np.max(np.abs(Xs - X)))

            rec = {
                "iter": it, "eta": eta, "ratio": float(ratio), "J": float(J_s), "L": L_pred,
                "nu_l1": nu_norm, "defect_l1": float(np.abs(defects).sum()), "dx_inf": dx,
                "tf": ps * s.Tt, "m_f": float(math.exp(Xs[-1, 6]) * s.M),
            }
            # the very first iterate from a re-plan guess must be taken (old guess violates x0)
            if ratio < cfg["r0"] and accepted_any:
                rec["status"] = "rejected"
                eta /= cfg["a"]
            else:
                rec["status"] = "accepted"
                accepted_any = True
                X, W, p, lin, J_ref = Xs, Ws, ps, lin_s, J_s
                if ratio < cfg["r1"]:
                    eta /= cfg["a"]
                elif ratio >= cfg["r2"]:
                    eta *= cfg["b"]
            eta = float(np.clip(eta, cfg["eta_min"], cfg["eta_max"]))
            history.append(rec)
            if verbose:
                print(
                    f"[scvx] it={it:2d} {rec['status']:8s} ratio={ratio:+.3f} eta={rec['eta']:.3g} "
                    f"|nu|={nu_norm:.2e} |defect|={rec['defect_l1']:.2e} dx={dx:.2e} "
                    f"tf={rec['tf']:.3f} mf={rec['m_f']:.1f}"
                )
            if rec["status"] == "accepted" and nu_norm <= cfg["eps_nu"] and dx <= cfg["eps_x"]:
                converged = True
                break

        if not accepted_any:
            raise RuntimeError("SCvx: no subproblem solved")
        tf = p * s.Tt
        traj = Trajectory(
            t=np.linspace(0, 1, N) * tf,
            r=X[:, 0:3] * s.L,
            v=X[:, 3:6] * s.V,
            m=np.exp(X[:, 6]) * s.M,
            u=W[:, 0:3] * s.g0,
            sigma=W[:, 3] * s.g0,
            hold="foh",
            info={
                "converged": converged,
                "iterations": len(history),
                "history": history,
                "N": N,
                "defect_l1": float(np.abs(X[1:] - lin[5]).sum()),
            },
        )
        from .gfold import check_solution

        traj.info.update(check_solution(sc, traj, skip_last=False))
        return traj


def solve(sc: Scenario, init: Trajectory, verbose: bool = False) -> Trajectory:
    """One-shot SCvx from the scenario's initial state, initialized from a G-FOLD solution."""
    return SCvxSolver(sc).solve(init, verbose=verbose)
