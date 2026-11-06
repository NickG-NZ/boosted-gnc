//! Inner-loop tracking MPC: attitude-aware, fixed final time, linearized about the guidance plan.
//!
//! The outer loop (G-FOLD / SCvx) fixes t_f, ignition, mass and the trajectory shape. This loop
//! re-solves a convex SOCP over the remaining time (shrinking horizon) at ~20 Hz and directly
//! commands throttle and gimbal, trading position error against tilt, body rate and gimbal effort.
//!
//! Model (body axis within ~20 deg of vertical, so attitude is described by the horizontal
//! components of the body axis b_h, its rate s = d(b_h)/dt and the gimbal g_h):
//!
//! ```text
//! r' = v
//! v' = (T/m) t_hat + g0 + D/m,        t_hat ~ b + g
//! b_h'' = -(l T / I) g_h              (from I w' = -l T (b x g), small body rates)
//! g_h' = u_g                          (gimbal rate is the control)
//! ```
//!
//! Linearized about the plan with error state x = [dr(3), dv(3), db_h(2), s(2), g_h(2)] and
//! control u = [dT/T_max, u_g(2)], plus the defect c = f(x_bar, u_bar) - x_bar' (gravity, drag
//! at the nominal state, plan rotation). A is nilpotent (A^5 = 0), so the zero-order-hold
//! discretization is an exact finite series. Coefficients are frozen at interval midpoints.

use nalgebra::{SMatrix, SVector};
use sim_core::{Vec3, math::interp};
use sim_opt::{Expr, Problem, Var};
use sim_scenario::{MpcCfg, Scenario};

use crate::{NavState, model::NominalModel, trajectory::Trajectory};

pub const NX: usize = 12;
pub const NU: usize = 3;
const IR: usize = 0;
const IV: usize = 3;
const IB: usize = 6;
const IS: usize = 8;
const IG: usize = 10;
const FACT: [f64; 6] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0];

pub type MatX = SMatrix<f64, NX, NX>;
pub type MatXU = SMatrix<f64, NX, NU>;
pub type VecX = SVector<f64, NX>;

/// Smooth nominal thrust acceleration. ZOH plans jump at nodes (instantaneous rotations), so
/// their segment midpoints are interpolated linearly instead.
pub fn nominal_u(plan: &Trajectory, t: f64) -> Vec3 {
    let (tm, um): (Vec<f64>, Vec<Vec3>) = match plan.hold {
        crate::Hold::Zoh => (
            (0..plan.len() - 1).map(|k| 0.5 * (plan.t[k] + plan.t[k + 1])).collect(),
            plan.u[..plan.len() - 1].to_vec(),
        ),
        crate::Hold::Foh => (plan.t.clone(), plan.u.clone()),
    };
    Vec3::from_fn(|i, _| interp(t, &tm, &um.iter().map(|u| u[i]).collect::<Vec<_>>()))
}

/// N interval lengths summing to t_go: the first is dt0, then geometric growth.
pub fn time_grid(t_go: f64, n: usize, dt0: f64) -> Vec<f64> {
    if t_go <= n as f64 * dt0 {
        return vec![t_go / n as f64; n];
    }
    let (mut lo, mut hi) = (1.0f64, 2.0f64);
    let mut q = 1.5;
    for _ in 0..60 {
        q = 0.5 * (lo + hi);
        if dt0 * (q.powi(n as i32) - 1.0) / (q - 1.0) < t_go {
            lo = q;
        } else {
            hi = q;
        }
    }
    let d: Vec<f64> = (0..n).map(|i| dt0 * q.powi(i as i32)).collect();
    let sum: f64 = d.iter().sum();
    d.iter().map(|x| x * (t_go / sum)).collect()
}

/// Analytic Jacobians A, B at thrust T, mass m, nominal horizontal thrust direction bh.
pub fn continuous_model(t: f64, m: f64, t_max: f64, bh: [f64; 2], k_att: f64) -> (MatX, MatXU) {
    let bu = (1.0 - (bh[0] * bh[0] + bh[1] * bh[1])).max(1e-6).sqrt();
    let a = t / m;
    let mut am = MatX::zeros();
    let mut bm = MatXU::zeros();
    for i in 0..3 {
        am[(IR + i, IV + i)] = 1.0;
    }
    for j in 0..2 {
        am[(3, IB + j)] = -a * bh[j] / bu; // vertical loss from tilting
        am[(3, IG + j)] = -a * bh[j] / bu;
        am[(4 + j, IB + j)] = a; // lateral force from body tilt and gimbal
        am[(4 + j, IG + j)] = a;
        am[(IB + j, IS + j)] = 1.0;
        am[(IS + j, IG + j)] = -k_att; // gimbal torque swings the nose the other way
        bm[(4 + j, 0)] = t_max * bh[j] / m;
        bm[(IG + j, 1 + j)] = 1.0;
    }
    bm[(3, 0)] = t_max * bu / m;
    (am, bm)
}

/// Exact ZOH for nilpotent A: exp(A dt) = sum_{k<5} (A dt)^k / k!.
pub fn discretize(a: &MatX, b: &MatXU, c: &VecX, dt: f64) -> (MatX, MatXU, VecX) {
    let mut ad = MatX::identity();
    let mut g = MatX::identity() * dt;
    let mut ak = MatX::identity();
    for k in 1..5 {
        ak *= a;
        ad += ak * (dt.powi(k as i32) / FACT[k]);
        g += ak * (dt.powi(k as i32 + 1) / FACT[k + 1]);
    }
    (ad, g * b, g * c)
}

#[derive(Debug, Clone)]
pub struct MpcSolution {
    /// node times relative to the solve (N + 1)
    pub t: Vec<f64>,
    /// thrust magnitude per interval (N)
    pub thrust: Vec<f64>,
    /// absolute horizontal gimbal per node (N + 1)
    pub g_h: Vec<[f64; 2]>,
    pub cost: f64,
    pub solve_s: f64,
}

impl MpcSolution {
    /// Thrust (held per interval) and gimbal target at time `ts` after the solve.
    pub fn command(&self, ts: f64) -> (f64, [f64; 2]) {
        let k = (self.t.partition_point(|&x| x <= ts) as isize - 1).clamp(0, self.thrust.len() as isize - 1) as usize;
        let col = |i: usize| self.g_h.iter().map(|g| g[i]).collect::<Vec<_>>();
        (self.thrust[k], [interp(ts, &self.t, &col(0)), interp(ts, &self.t, &col(1))])
    }
}

struct Linearization {
    dts: Vec<f64>,
    tk: Vec<f64>,
    rb: Vec<Vec3>,
    vb: Vec<Vec3>,
    bh: Vec<[f64; 2]>,
    ad: Vec<MatX>,
    bd: Vec<MatXU>,
    cd: Vec<VecX>,
    tbar: Vec<f64>,
    gcap: Vec<f64>,
}

pub struct TrackingMpc {
    sc: Scenario,
    pub p: MpcCfg,
    k_per_t: f64,
    g_max: f64,
    dg_max: f64,
    model: NominalModel,
    alpha: f64,
}

impl TrackingMpc {
    pub fn new(sc: &Scenario, drag: bool) -> Self {
        let ap = &sc.rocket.attitude;
        Self {
            sc: sc.clone(),
            p: sc.mpc.clone(),
            k_per_t: ap.arm / ap.inertia,
            g_max: ap.gimbal_max_deg.to_radians().sin(),
            dg_max: ap.gimbal_rate_deg_s.to_radians(),
            model: NominalModel::new(&sc.rocket, &sc.env, drag),
            alpha: sc.rocket.alpha(sc.env.g0),
        }
    }

    fn linearize(&self, plan: &Trajectory, tau: f64, m_now: f64) -> Linearization {
        let (p, rk, n) = (&self.p, &self.sc.rocket, self.p.n);
        let dts = time_grid(plan.tf() - tau, n, p.dt0);
        let mut tk = vec![tau];
        for d in &dts {
            tk.push(tk.last().unwrap() + d);
        }
        let tm: Vec<f64> = (0..n).map(|k| tk[k] + 0.5 * dts[k]).collect();
        let (rb, vb): (Vec<Vec3>, Vec<Vec3>) = tk.iter().map(|&t| plan.state_at(t)).unzip();
        let bh: Vec<[f64; 2]> = tk
            .iter()
            .map(|&t| {
                let u = nominal_u(plan, t);
                let d = u / u.norm();
                [d[1], d[2]]
            })
            .collect();
        let g0 = self.sc.env.g();
        let mut m = m_now;
        let mut out = Linearization {
            ad: Vec::with_capacity(n),
            bd: Vec::with_capacity(n),
            cd: Vec::with_capacity(n),
            tbar: vec![0.0; n],
            gcap: vec![0.0; n],
            dts: dts.clone(),
            tk: tk.clone(),
            rb: rb.clone(),
            vb: vb.clone(),
            bh: bh.clone(),
        };
        for k in 0..n {
            let h = dts[k];
            let (rm, vm) = plan.state_at(tm[k]);
            let um = nominal_u(plan, tm[k]);
            let ua = um.norm();
            let m_mid = m / (1.0 + 0.5 * self.alpha * ua * h); // thrust ~ m |u| -> mass decays
            let t = m_mid * ua;
            let bhk = [um[1] / ua, um[2] / ua];
            let (a, b) = continuous_model(t, m_mid, rk.t_max, bhk, self.k_per_t * t);
            let d = self.model.drag_force(&rm, &vm, true);
            let mut c = VecX::zeros();
            let acc = um + g0 + d / m_mid - (vb[k + 1] - vb[k]) / h;
            for i in 0..3 {
                c[IR + i] = vm[i] - (rb[k + 1][i] - rb[k][i]) / h;
                c[IV + i] = acc[i];
            }
            for j in 0..2 {
                c[IB + j] = -(bh[k + 1][j] - bh[k][j]) / h; // the plan rotates: keep up or fall behind
            }
            let (ad, bd, cd) = discretize(&a, &b, &c, h);
            out.ad.push(ad);
            out.bd.push(bd);
            out.cd.push(cd);
            out.tbar[k] = t / rk.t_max;
            out.gcap[k] = self.g_max * (1.0 - (bhk[0] * bhk[0] + bhk[1] * bhk[1])).max(0.0).sqrt();
            m -= self.alpha * t * h;
        }
        out
    }

    pub fn solve(&self, plan: &Trajectory, tau: f64, nav: &NavState) -> Option<MpcSolution> {
        let (p, rk, n) = (&self.p, &self.sc.rocket, self.p.n);
        let lin = self.linearize(plan, tau, nav.m);
        let (b, w) = (nav.b, nav.w);
        let mut gh0 = [nav.g[1], nav.g[2]];
        let gn = (gh0[0] * gh0[0] + gh0[1] * gh0[1]).sqrt();
        if gn > lin.gcap[0] {
            gh0 = [gh0[0] * lin.gcap[0] / gn, gh0[1] * lin.gcap[0] / gn];
        }
        let wb = w.cross(&b);
        let x0: [f64; NX] = {
            let dr = nav.r - lin.rb[0];
            let dv = nav.v - lin.vb[0];
            [
                dr[0],
                dr[1],
                dr[2],
                dv[0],
                dv[1],
                dv[2],
                b[1] - lin.bh[0][0],
                b[2] - lin.bh[0][1],
                wb[1],
                wb[2],
                gh0[0],
                gh0[1],
            ]
        };

        let mut pr = Problem::new();
        let x: Vec<[Var; NX]> = (0..=n).map(|_| std::array::from_fn(|_| pr.var())).collect();
        let u: Vec<[Var; NU]> = (0..n).map(|_| std::array::from_fn(|_| pr.var())).collect();
        for i in 0..NX {
            pr.eq(Expr::var(x[0][i]).plus_const(-x0[i]));
        }
        let sin_cone = rk.theta_max().sin();
        for k in 0..n {
            for i in 0..NX {
                let mut e = Expr::var(x[k + 1][i]).plus_const(-lin.cd[k][i]);
                for j in 0..NX {
                    e = e.plus_var(-lin.ad[k][(i, j)], x[k][j]);
                }
                for j in 0..NU {
                    e = e.plus_var(-lin.bd[k][(i, j)], u[k][j]);
                }
                pr.eq(e);
            }
            pr.ge0(Expr::var(u[k][0]).plus_const(lin.tbar[k] - rk.thr_min));
            pr.ge0(Expr::var(u[k][0]).scale(-1.0).plus_const(1.0 - lin.tbar[k]));
            pr.soc(Expr::constant(self.dg_max), vec![u[k][1].into(), u[k][2].into()]);
            pr.soc(Expr::constant(lin.gcap[k]), vec![x[k + 1][IG].into(), x[k + 1][IG + 1].into()]);
            let w2 = lin.dts[k]; // (sqrt(dt))^2: running costs are per second
            // soft pointing cone keeps the vehicle where the small-tilt model holds
            let bn = lin.bh[k + 1];
            let tilt =
                pr.norm(vec![Expr::var(x[k + 1][IB]).plus_const(bn[0]), Expr::var(x[k + 1][IB + 1]).plus_const(bn[1])]);
            let over = pr.pos(&Expr::var(tilt).plus_const(-sin_cone));
            pr.add_linear(&Expr::var(over).scale(p.w_cone));
            for i in 0..3 {
                pr.add_square(p.w_dr * w2, &x[k + 1][IR + i].into());
                pr.add_square(p.w_dv * w2, &x[k + 1][IV + i].into());
            }
            for j in 0..2 {
                pr.add_square(p.w_db * w2, &x[k + 1][IB + j].into());
                pr.add_square(p.w_ug * w2, &u[k][1 + j].into());
            }
            pr.add_square(p.w_dt * w2, &u[k][0].into());
        }
        // terminal (absolute errors at t_f); position/velocity costs turn linear beyond huber_*
        let xn = &x[n];
        let (r_end, v_end, b_end) = (lin.rb[n], lin.vb[n], lin.bh[n]);
        pr.add_huber(p.w_r_up, &Expr::var(xn[IR]).plus_const(r_end[0]), p.huber_r);
        for i in 1..3 {
            pr.add_huber(p.w_r_lat, &Expr::var(xn[IR + i]).plus_const(r_end[i]), p.huber_r);
        }
        for i in 0..3 {
            pr.add_huber(p.w_v, &Expr::var(xn[IV + i]).plus_const(v_end[i]), p.huber_v);
        }
        for j in 0..2 {
            pr.add_square(p.w_tilt, &Expr::var(xn[IB + j]).plus_const(b_end[j]));
            pr.add_square(p.w_rate, &xn[IS + j].into());
            pr.add_square(p.w_gimbal_term, &xn[IG + j].into());
        }

        let sol = pr.solve();
        if !sol.status.ok() {
            return None;
        }
        let thrust =
            (0..n).map(|k| ((lin.tbar[k] + sol.get(u[k][0])) * rk.t_max).clamp(rk.t_min(), rk.t_max)).collect();
        Some(MpcSolution {
            t: lin.tk.iter().map(|t| t - tau).collect(),
            thrust,
            g_h: x.iter().map(|xk| [sol.get(xk[IG]), sol.get(xk[IG + 1])]).collect(),
            cost: sol.obj,
            solve_s: sol.solve_time,
        })
    }
}

/// 3D gimbal vector perpendicular to `b` with the given horizontal components.
pub fn embed_gimbal(g_h: [f64; 2], b: &Vec3) -> Vec3 {
    Vec3::new(-(g_h[0] * b[1] + g_h[1] * b[2]) / b[0].max(1e-3), g_h[0], g_h[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discretization_is_exact() {
        let (a, b) = continuous_model(4.0e5, 2.4e4, 8.45e5, [0.1, -0.05], 1.5);
        assert!((a.pow(5)).abs().max() < 1e-12, "nilpotent");
        let c = VecX::from_fn(|i, _| -1.0 + 2.0 * i as f64 / (NX - 1) as f64);
        let dt = 0.7;
        let (ad, bd, cd) = discretize(&a, &b, &c, dt);
        // compare with the exponential of the augmented matrix
        let mut m = SMatrix::<f64, 16, 16>::zeros();
        m.fixed_view_mut::<NX, NX>(0, 0).copy_from(&a);
        m.fixed_view_mut::<NX, NU>(0, NX).copy_from(&b);
        m.fixed_view_mut::<NX, 1>(0, 15).copy_from(&c);
        let e = (m * dt).exp();
        assert!((ad - e.fixed_view::<NX, NX>(0, 0)).abs().max() < 1e-9);
        assert!((bd - e.fixed_view::<NX, NU>(0, NX)).abs().max() < 1e-9);
        assert!((cd - e.fixed_view::<NX, 1>(0, 15)).abs().max() < 1e-9);
    }

    #[test]
    fn grid_sums_to_tgo() {
        let d = time_grid(10.0, 30, 0.05);
        assert!((d.iter().sum::<f64>() - 10.0).abs() < 1e-9);
        assert!((d[0] - 0.05).abs() < 1e-6 && d[29] > d[0]);
        assert_eq!(time_grid(1.0, 30, 0.05), vec![1.0 / 30.0; 30]);
    }
}
