//! Phase 2: successive convexification (SCvx) with aerodynamic drag and free final time.
//!
//! Everything inside runs in nondimensional units (length L = |r0|, mass M = m0, time
//! Tt = sqrt(L / g0)); inputs/outputs are SI. Each iteration linearizes the drag dynamics about
//! the current reference, discretizes it exactly with a first-order hold (multiple shooting,
//! RK4), solves a convex SOCP with virtual control and a trust region, and accepts / rejects the
//! step from the ratio of actual to predicted merit decrease.

use nalgebra::{SMatrix, SVector};
use serde_json::{Map, Value, json};
use sim_core::{Vec3, math::interp};
use sim_opt::{Expr, Problem, Var, lin};
use sim_scenario::{Scenario, ScvxCfg};

use super::gfold::{attitude_constraint, attitude_penalties, check_solution};
use crate::trajectory::{Hold, Trajectory};

type V7 = SVector<f64, 7>;
type V4 = SVector<f64, 4>;
type M7 = SMatrix<f64, 7, 7>;
type M74 = SMatrix<f64, 7, 4>;

#[derive(Debug, Clone, Copy)]
pub struct Scaling {
    pub l: f64,
    pub m: f64,
    pub tt: f64,
    pub g0: f64,
}

impl Scaling {
    pub fn v(&self) -> f64 {
        self.l / self.tt
    }
    pub fn for_scenario(sc: &Scenario) -> Self {
        let n = sc.r0.norm();
        let l = if n != 0.0 { n } else { 1000.0 };
        Self { l, m: sc.m0, tt: (l / sc.env.g0).sqrt(), g0: sc.env.g0 }
    }
}

/// Nondimensional dynamics f(x, w); x = [r, v, z], w = [u, sigma], accel unit = g0.
#[derive(Debug, Clone)]
pub struct ScaledModel {
    pub alpha: f64,
    /// 0.5 rho0 S_ref C_D_eff L / M
    pub ks: f64,
    /// scale height / L
    pub hs: f64,
    pub wind: Vec3,
    pub eps: f64,
}

impl ScaledModel {
    pub fn build(sc: &Scenario, s: &Scaling) -> Self {
        let (rk, env) = (&sc.rocket, &sc.env);
        Self {
            alpha: rk.alpha(env.g0) * env.g0 * s.tt,
            ks: 0.5 * env.rho0 * rk.cds_thrusting() * s.l / s.m,
            hs: env.h / s.l,
            wind: env.wind_vec() / s.v(),
            eps: 1e-6,
        }
    }

    fn drag_terms(&self, x: &V7) -> (Vec3, f64, f64, f64, Vec3) {
        let vr = Vec3::new(x[3], x[4], x[5]) - self.wind;
        let nv = vr.norm() + self.eps;
        let k = self.ks * (-x[0] / self.hs).exp();
        let ez = (-x[6]).exp();
        let d = vr * -(k * nv);
        (vr, nv, k, ez, d)
    }

    pub fn f(&self, x: &V7, w: &V4) -> V7 {
        let (_, _, _, ez, d) = self.drag_terms(x);
        let mut out = V7::zeros();
        for i in 0..3 {
            out[i] = x[3 + i];
            out[3 + i] = w[i] + ez * d[i];
        }
        out[3] -= 1.0; // gravity
        out[6] = -self.alpha * w[3];
        out
    }

    /// df/dx
    pub fn a(&self, x: &V7) -> M7 {
        let (vr, nv, k, ez, d) = self.drag_terms(x);
        let mut a = M7::zeros();
        for i in 0..3 {
            a[(i, 3 + i)] = 1.0;
            a[(3 + i, 0)] = ez * k / self.hs * nv * vr[i];
            for j in 0..3 {
                let outer = vr[i] * vr[j] / nv;
                a[(3 + i, 3 + j)] = -(k * ez) * (if i == j { nv } else { 0.0 } + outer);
            }
            a[(3 + i, 6)] = -ez * d[i];
        }
        a
    }

    pub fn b(&self) -> M74 {
        let mut b = M74::zeros();
        for i in 0..3 {
            b[(3 + i, i)] = 1.0;
        }
        b[(6, 3)] = -self.alpha;
        b
    }
}

/// Linearized FOH discretization of one interval: x_{k+1} ~ A x_k + Bm w_k + Bp w_{k+1} + S p + c.
#[derive(Debug, Clone)]
pub struct Interval {
    pub a: M7,
    pub bm: M74,
    pub bp: M74,
    pub s: V7,
    pub c: V7,
    /// nonlinear flow from x_k (defect = x_{k+1} - x_end)
    pub x_end: V7,
}

#[derive(Clone)]
struct St {
    xb: V7,
    phi: M7,
    bm: M74,
    bp: M74,
    s: V7,
    c: V7,
}

impl St {
    fn axpy(&self, h: f64, d: &St) -> St {
        St {
            xb: self.xb + d.xb * h,
            phi: self.phi + d.phi * h,
            bm: self.bm + d.bm * h,
            bp: self.bp + d.bp * h,
            s: self.s + d.s * h,
            c: self.c + d.c * h,
        }
    }
}

/// Linearize about (X, W, p) and discretize exactly with first-order hold (RK4, `n_sub` steps).
pub fn discretize_foh(model: &ScaledModel, xs: &[V7], ws: &[V4], p: f64, n_sub: usize) -> Vec<Interval> {
    let k_int = xs.len() - 1;
    let dtau = 1.0 / k_int as f64;
    let b = model.b();
    let pb = b * p;
    (0..k_int)
        .map(|k| {
            let (wk, wk1) = (ws[k], ws[k + 1]);
            let rhs = |s: f64, st: &St| -> St {
                let (lm, lp) = (1.0 - s, s);
                let w = wk * lm + wk1 * lp;
                let f = model.f(&st.xb, &w);
                let ac = model.a(&st.xb) * p;
                St {
                    xb: f * (p * dtau),
                    phi: ac * st.phi * dtau,
                    bm: (ac * st.bm + pb * lm) * dtau,
                    bp: (ac * st.bp + pb * lp) * dtau,
                    s: (ac * st.s + f) * dtau,
                    c: (ac * (st.c - st.xb) - pb * w) * dtau,
                }
            };
            let mut st = St {
                xb: xs[k],
                phi: M7::identity(),
                bm: M74::zeros(),
                bp: M74::zeros(),
                s: V7::zeros(),
                c: V7::zeros(),
            };
            let h = 1.0 / n_sub as f64;
            for i in 0..n_sub {
                let s0 = i as f64 * h;
                let k1 = rhs(s0, &st);
                let k2 = rhs(s0 + h / 2.0, &st.axpy(h / 2.0, &k1));
                let k3 = rhs(s0 + h / 2.0, &st.axpy(h / 2.0, &k2));
                let k4 = rhs(s0 + h, &st.axpy(h, &k3));
                let sum = k1.axpy(2.0, &k2).axpy(2.0, &k3).axpy(1.0, &k4);
                st = st.axpy(h / 6.0, &sum);
            }
            Interval { a: st.phi, bm: st.bm, bp: st.bp, s: st.s, c: st.c, x_end: st.xb }
        })
        .collect()
}

fn defects(xs: &[V7], lin: &[Interval]) -> Vec<V7> {
    lin.iter().enumerate().map(|(k, iv)| xs[k + 1] - iv.x_end).collect()
}

fn l1(vs: &[V7]) -> f64 {
    vs.iter().map(|v| v.abs().sum()).sum()
}

fn merit(xs: &[V7], defects: &[V7], w_nu: f64) -> f64 {
    -xs[xs.len() - 1][6] + w_nu * l1(defects)
}

struct SubSolution {
    xs: Vec<V7>,
    ws: Vec<V4>,
    p: f64,
    nu_l1: f64,
    obj: f64,
}

/// Initial guess (scaled) from a trajectory: nodes uniformly spaced over its duration.
fn initial_guess(traj: &Trajectory, s: &Scaling, n: usize) -> (Vec<V7>, Vec<V4>, f64) {
    let tf = traj.tf();
    let mut xs = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    for j in 0..n {
        let t = j as f64 / (n - 1) as f64 * tf;
        let col = |vs: &[Vec3], i: usize| interp(t, &traj.t, &vs.iter().map(|v| v[i]).collect::<Vec<_>>());
        let mut x = V7::zeros();
        for i in 0..3 {
            x[i] = col(&traj.r, i) / s.l;
            x[3 + i] = col(&traj.v, i) / s.v();
        }
        x[6] = (interp(t, &traj.t, &traj.m) / s.m).ln();
        let u = traj.u_at(t) / s.g0;
        xs.push(x);
        ws.push(V4::new(u[0], u[1], u[2], u.norm()));
    }
    (xs, ws, tf / s.tt)
}

pub struct SCvxSolver {
    pub sc: Scenario,
    pub cfg: ScvxCfg,
    pub s: Scaling,
    pub model: ScaledModel,
}

#[derive(Debug, Clone, Default)]
pub struct SolveFrom {
    pub r0: Option<Vec3>,
    pub v0: Option<Vec3>,
    pub m0: Option<f64>,
    pub att0: Option<Vec3>,
    pub max_iters: Option<usize>,
}

#[derive(Debug)]
pub struct NoSubproblemSolved;

impl SCvxSolver {
    pub fn new(sc: &Scenario) -> Self {
        let s = Scaling::for_scenario(sc);
        Self { sc: sc.clone(), cfg: sc.scvx.clone(), model: ScaledModel::build(sc, &s), s }
    }

    #[allow(clippy::too_many_arguments)]
    fn subproblem(
        &self,
        xbar: &[V7],
        wbar: &[V4],
        pbar: f64,
        lin_: &[Interval],
        eta: f64,
        x0: &V7,
        d0: Option<Vec3>,
    ) -> Option<SubSolution> {
        let (sc, s, cfg) = (&self.sc, &self.s, &self.cfg);
        let rk = &sc.rocket;
        let n = cfg.n;
        let k_int = n - 1;
        let rho1 = rk.rho1() / (s.m * s.g0);
        let rho2 = rk.rho2() / (s.m * s.g0);

        let mut p = Problem::new();
        let x: Vec<[Var; 7]> = (0..n).map(|_| std::array::from_fn(|_| p.var())).collect();
        let w: Vec<[Var; 4]> = (0..n).map(|_| std::array::from_fn(|_| p.var())).collect();
        let pv = p.var();
        let nu: Vec<[Var; 7]> = (0..k_int).map(|_| std::array::from_fn(|_| p.var())).collect();
        let u: Vec<[Var; 3]> = w.iter().map(|wk| [wk[0], wk[1], wk[2]]).collect();
        let sigma: Vec<Var> = w.iter().map(|wk| wk[3]).collect();

        for i in 0..7 {
            p.eq(Expr::var(x[0][i]).plus_const(-x0[i]));
        }
        for i in 0..6 {
            p.eq(Expr::var(x[n - 1][i]));
        }
        p.ge0(Expr::var(pv).plus_const(-1e-2));
        let (cos_th, tan_gs, z_dry) = (rk.theta_max().cos(), sc.gamma_gs().tan(), (rk.m_dry / s.m).ln());
        for k in 0..n {
            let z = x[k][6];
            p.ge0(Expr::var(z).plus_const(-z_dry));
            p.soc(sigma[k].into(), u[k].iter().map(|&v| v.into()).collect());
            p.ge0(lin(&[(1.0, u[k][0]), (-cos_th, sigma[k])]));
            let zbar = xbar[k][6];
            let (mu1, mu2) = (rho1 * (-zbar).exp(), rho2 * (-zbar).exp());
            let dz = Expr::var(z).plus_const(-zbar);
            let sq = Expr::var(sigma[k]).scale(2.0 / mu1).plus_const(-2.0).plus(2.0, &dz);
            p.square_le(&dz, &sq);
            p.ge0(Expr::constant(mu2).plus(-mu2, &dz).plus_var(-1.0, sigma[k]));
            p.soc(x[k][0].into(), vec![Expr::var(x[k][1]).scale(tan_gs), Expr::var(x[k][2]).scale(tan_gs)]);
            // trust region: max|x - xbar| + max|w - wbar| <= eta  (per node)
            let (ta, tb) = (p.var(), p.var());
            for i in 0..7 {
                let e = Expr::var(x[k][i]).plus_const(-xbar[k][i]);
                p.ge0(Expr::var(ta).plus(-1.0, &e));
                p.ge0(Expr::var(ta).plus(1.0, &e));
            }
            for i in 0..4 {
                let e = Expr::var(w[k][i]).plus_const(-wbar[k][i]);
                p.ge0(Expr::var(tb).plus(-1.0, &e));
                p.ge0(Expr::var(tb).plus(1.0, &e));
            }
            p.ge0(Expr::constant(eta).plus_var(-1.0, ta).plus_var(-1.0, tb));
        }
        let dp = Expr::var(pv).plus_const(-pbar);
        p.ge0(Expr::constant(eta).plus(-1.0, &dp));
        p.ge0(Expr::constant(eta).plus(1.0, &dp));
        let mut nu_abs = Expr::default();
        for (k, iv) in lin_.iter().enumerate() {
            for i in 0..7 {
                let mut e = Expr::var(x[k + 1][i]).plus_const(-iv.c[i]).plus_var(-iv.s[i], pv).plus_var(-1.0, nu[k][i]);
                for j in 0..7 {
                    e = e.plus_var(-iv.a[(i, j)], x[k][j]);
                }
                for j in 0..4 {
                    e = e.plus_var(-iv.bm[(i, j)], w[k][j]).plus_var(-iv.bp[(i, j)], w[k + 1][j]);
                }
                p.eq(e);
                let t = p.abs(&Expr::var(nu[k][i]));
                nu_abs = nu_abs.plus_var(1.0, t);
            }
        }
        if let Some(d0) = d0 {
            attitude_constraint(&mut p, sc, &u[0], sigma[0], &d0);
        }
        let sig_nom = 0.5 * (rk.rho1() + rk.rho2()) / sc.m0 / s.g0;
        let pen = attitude_penalties(&mut p, sc, &u, &sigma, sig_nom, k_int as f64 / pbar, n);
        p.add_linear(&pen.plus(cfg.w_nu, &nu_abs).plus_var(-1.0, x[n - 1][6]));

        let sol = p.solve();
        if !sol.status.ok() {
            return None;
        }
        let g = |v: Var| sol.get(v);
        Some(SubSolution {
            xs: x.iter().map(|xk| V7::from_fn(|i, _| g(xk[i]))).collect(),
            ws: w.iter().map(|wk| V4::from_fn(|i, _| g(wk[i]))).collect(),
            p: g(pv),
            nu_l1: nu.iter().flat_map(|nk| nk.iter()).map(|&v| g(v).abs()).sum(),
            obj: sol.obj,
        })
    }

    pub fn solve(&self, init: &Trajectory, from: &SolveFrom) -> Result<Trajectory, NoSubproblemSolved> {
        let (sc, s, cfg) = (&self.sc, &self.s, &self.cfg);
        let r0 = from.r0.unwrap_or(sc.r0);
        let v0 = from.v0.unwrap_or(sc.v0);
        let m0 = from.m0.unwrap_or(sc.m0);
        let x0 = V7::from_column_slice(&[
            r0[0] / s.l,
            r0[1] / s.l,
            r0[2] / s.l,
            v0[0] / s.v(),
            v0[1] / s.v(),
            v0[2] / s.v(),
            (m0 / s.m).ln(),
        ]);
        let d0 = Scenario { v0, att0: from.att0, ..sc.clone() }.initial_thrust_dir();
        let (n, w_nu) = (cfg.n, cfg.w_nu);

        let (mut xs, mut ws, mut p) = initial_guess(init, s, n);
        let mut lin_ = discretize_foh(&self.model, &xs, &ws, p, 10);
        let mut j_ref = merit(&xs, &defects(&xs, &lin_), w_nu);
        let mut eta = cfg.eta0;
        let mut history: Vec<Value> = Vec::new();
        let mut converged = false;
        let mut accepted_any = false;

        for it in 0..from.max_iters.unwrap_or(cfg.max_iters) {
            let Some(sol) = self.subproblem(&xs, &ws, p, &lin_, eta, &x0, d0) else {
                eta = (eta / cfg.a).max(cfg.eta_min);
                history.push(json!({"iter": it, "status": "subproblem_failed", "eta": eta}));
                continue;
            };
            let lin_s = discretize_foh(&self.model, &sol.xs, &sol.ws, sol.p, 10);
            let def = defects(&sol.xs, &lin_s);
            let j_s = merit(&sol.xs, &def, w_nu);
            let denom = j_ref - sol.obj;
            let ratio = if denom.abs() > 1e-12 { (j_ref - j_s) / denom } else { 1.0 };
            let dx = sol.xs.iter().zip(&xs).map(|(a, b)| (a - b).abs().max()).fold(0.0, f64::max);
            let mut rec = Map::new();
            rec.insert("iter".into(), json!(it));
            rec.insert("eta".into(), json!(eta));
            rec.insert("ratio".into(), json!(ratio));
            rec.insert("J".into(), json!(j_s));
            rec.insert("L".into(), json!(sol.obj));
            rec.insert("nu_l1".into(), json!(sol.nu_l1));
            rec.insert("defect_l1".into(), json!(l1(&def)));
            rec.insert("dx_inf".into(), json!(dx));
            rec.insert("tf".into(), json!(sol.p * s.tt));
            rec.insert("m_f".into(), json!(sol.xs[n - 1][6].exp() * s.m));
            // the first iterate from a re-plan guess must be taken (the old guess violates x0)
            let accepted = !(ratio < cfg.r0 && accepted_any);
            if accepted {
                accepted_any = true;
                let nu_l1 = sol.nu_l1;
                (xs, ws, p, lin_, j_ref) = (sol.xs, sol.ws, sol.p, lin_s, j_s);
                if ratio < cfg.r1 {
                    eta /= cfg.a;
                } else if ratio >= cfg.r2 {
                    eta *= cfg.b;
                }
                eta = eta.clamp(cfg.eta_min, cfg.eta_max);
                rec.insert("status".into(), json!("accepted"));
                history.push(Value::Object(rec));
                if nu_l1 <= cfg.eps_nu && dx <= cfg.eps_x {
                    converged = true;
                    break;
                }
            } else {
                eta = (eta / cfg.a).clamp(cfg.eta_min, cfg.eta_max);
                rec.insert("status".into(), json!("rejected"));
                history.push(Value::Object(rec));
            }
        }
        if !accepted_any {
            return Err(NoSubproblemSolved);
        }
        let tf = p * s.tt;
        let mut info = Map::new();
        info.insert("converged".into(), json!(converged));
        info.insert("iterations".into(), json!(history.len()));
        info.insert("history".into(), Value::Array(history));
        info.insert("N".into(), json!(n));
        info.insert("defect_l1".into(), json!(l1(&defects(&xs, &lin_))));
        let mut traj = Trajectory {
            t: (0..n).map(|k| k as f64 / (n - 1) as f64 * tf).collect(),
            r: xs.iter().map(|x| Vec3::new(x[0], x[1], x[2]) * s.l).collect(),
            v: xs.iter().map(|x| Vec3::new(x[3], x[4], x[5]) * s.v()).collect(),
            m: xs.iter().map(|x| x[6].exp() * s.m).collect(),
            u: ws.iter().map(|w| Vec3::new(w[0], w[1], w[2]) * s.g0).collect(),
            sigma: ws.iter().map(|w| w[3] * s.g0).collect(),
            hold: Hold::Foh,
            info,
        };
        for (k, v) in check_solution(sc, &traj, false) {
            traj.info.insert(k.into(), json!(v));
        }
        Ok(traj)
    }
}

/// History entry field of the last iteration (e.g. "nu_l1").
pub fn last_history(traj: &Trajectory, key: &str) -> Option<f64> {
    traj.info.get("history")?.as_array()?.last()?.get(key)?.as_f64()
}
