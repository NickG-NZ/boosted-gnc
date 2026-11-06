//! Phase 1: drag-free fuel-optimal powered descent (G-FOLD, lossless convexification).
//!
//! One SOCP per fixed final time t_f (Açıkmeşe & Ploen 2007, log-mass change of variables
//! z = ln m), then a 1-D search over t_f: coarse scan + golden-section refinement.

use nalgebra::SMatrix;
use serde_json::{Value, json};
use sim_core::Vec3;
use sim_opt::{Expr, Problem, Status, Var, lin};
use sim_scenario::{Scenario, nonzero};

use crate::trajectory::{Hold, Trajectory};

#[derive(Debug, Clone)]
pub struct Infeasible(pub String);

impl std::fmt::Display for Infeasible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Infeasible {}

/// `numpy.linspace(a, b, n)`
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step + a).collect();
    v[n - 1] = b;
    v
}

/// Exact ZOH discretization of x = [r, v, z], w = [u, sigma] (one matrix exponential).
pub fn discretize_zoh(alpha: f64, g0: f64, dt: f64) -> (SMatrix<f64, 7, 7>, SMatrix<f64, 7, 4>, [f64; 7]) {
    let mut m = SMatrix::<f64, 12, 12>::zeros();
    for i in 0..3 {
        m[(i, 3 + i)] = 1.0; // r' = v
        m[(3 + i, 7 + i)] = 1.0; // v' = u
    }
    m[(6, 10)] = -alpha; // z' = -alpha sigma
    m[(3, 11)] = -g0; // gravity
    let md = (m * dt).exp();
    let ad = md.fixed_view::<7, 7>(0, 0).into_owned();
    let bd = md.fixed_view::<7, 4>(0, 7).into_owned();
    let cd = std::array::from_fn(|i| md[(i, 11)]);
    (ad, bd, cd)
}

/// Soft attitude preferences in fuel-equivalent kg (never cause infeasibility):
///   - final pose (L1):        w_end  * sum_{last n controls} ||E u_k||
///   - rotation rate (quad):   w_rate * sum_k ||E du_k||^2 / dt
///   - slack (linear):         w_slack * sum_k (sigma_k - u_k,up), keeps the convexification tight
///
/// `sig_nom` = nominal thrust accel in the units of u, `inv_dt` = 1 / node spacing,
/// `n_ctrl` = controls actually applied (ZOH: N - 1). Returns the penalty expression.
pub fn attitude_penalties(
    p: &mut Problem,
    sc: &Scenario,
    u: &[[Var; 3]],
    sigma: &[Var],
    sig_nom: f64,
    inv_dt: f64,
    n_ctrl: usize,
) -> Expr {
    let c = &sc.constraints;
    let deg = 1f64.to_radians();
    let mut pen = Expr::default();
    let kg_end = nonzero(c.end_tilt_penalty_kg_per_deg);
    if let Some(kg) = kg_end {
        let coef = (kg / sc.m0) / (sig_nom * deg);
        let n = c.end_tilt_nodes;
        for uk in &u[n_ctrl - n..n_ctrl] {
            let t = p.norm(vec![uk[1].into(), uk[2].into()]);
            pen = pen.plus_var(coef, t);
        }
    }
    let kg_rate = nonzero(c.rotation_rate_penalty);
    if let Some(kg) = kg_rate {
        let coef = (kg / sc.m0) / (sig_nom * deg).powi(2) * inv_dt;
        for k in 1..n_ctrl {
            for i in 1..3 {
                p.add_square(coef, &lin(&[(1.0, u[k][i]), (-1.0, u[k - 1][i])]));
            }
        }
    }
    if let Some(kg) = nonzero(c.slack_penalty_kg_per_pct)
        && (kg_end.is_some() || kg_rate.is_some())
    {
        let coef = (kg / sc.m0) / (sig_nom * 0.01);
        for k in 0..n_ctrl {
            pen = pen.plus(coef, &lin(&[(1.0, sigma[k]), (-1.0, u[k][0])]));
        }
    }
    pen
}

/// Hard initial-attitude constraint: first thrust within the tolerance cone of `d0`
/// (equality if the tolerance is 0).
pub fn attitude_constraint(p: &mut Problem, sc: &Scenario, u0: &[Var; 3], sigma0: Var, d0: &Vec3) {
    let tol = sc.initial_attitude_tol();
    if tol > 0.0 {
        p.ge0(lin(&[(d0[0], u0[0]), (d0[1], u0[1]), (d0[2], u0[2]), (-tol.cos(), sigma0)]));
    } else {
        for i in 0..3 {
            p.eq(lin(&[(1.0, u0[i]), (-d0[i], sigma0)]));
        }
    }
}

/// Solve the convexified fixed-final-time problem; `None` if infeasible.
pub fn solve_fixed_tf(sc: &Scenario, tf: f64) -> Option<Trajectory> {
    let (rk, env) = (&sc.rocket, &sc.env);
    let n = sc.gfold.n;
    let (alpha, rho1, rho2) = (rk.alpha(env.g0), rk.rho1(), rk.rho2());
    if sc.m0 - alpha * rho2 * tf <= 0.0 {
        return None; // z0(t) undefined
    }
    let dt = tf / (n - 1) as f64;
    let t: Vec<f64> = (0..n).map(|k| k as f64 * dt).collect();
    let (ad, bd, cd) = discretize_zoh(alpha, env.g0, dt);
    // expansion about the lightest-possible mass trajectory
    let z0: Vec<f64> = t.iter().map(|t| (sc.m0 - alpha * rho2 * t).ln()).collect();
    let z1: Vec<f64> = t.iter().map(|t| (sc.m0 - alpha * rho1 * t).ln()).collect();
    let mu1: Vec<f64> = z0.iter().map(|z| rho1 * (-z).exp()).collect();
    let mu2: Vec<f64> = z0.iter().map(|z| rho2 * (-z).exp()).collect();

    let mut p = Problem::new();
    let x: Vec<[Var; 7]> = (0..n).map(|_| std::array::from_fn(|_| p.var())).collect();
    let w: Vec<[Var; 4]> = (0..n).map(|_| std::array::from_fn(|_| p.var())).collect();
    let u: Vec<[Var; 3]> = w.iter().map(|wk| [wk[0], wk[1], wk[2]]).collect();
    let sigma: Vec<Var> = w.iter().map(|wk| wk[3]).collect();

    let x0 = [sc.r0[0], sc.r0[1], sc.r0[2], sc.v0[0], sc.v0[1], sc.v0[2], sc.m0.ln()];
    for i in 0..7 {
        p.eq(Expr::var(x[0][i]).plus_const(-x0[i]));
    }
    for i in 0..6 {
        p.eq(Expr::var(x[n - 1][i]));
    }
    p.ge0(Expr::var(x[n - 1][6]).plus_const(-rk.m_dry.ln()));
    for k in 0..n - 1 {
        for i in 0..7 {
            let mut e = Expr::var(x[k + 1][i]).plus_const(-cd[i]);
            for j in 0..7 {
                e = e.plus_var(-ad[(i, j)], x[k][j]);
            }
            for j in 0..4 {
                e = e.plus_var(-bd[(i, j)], w[k][j]);
            }
            p.eq(e);
        }
    }
    let (cos_th, tan_gs) = (rk.theta_max().cos(), sc.gamma_gs().tan());
    for k in 0..n {
        let z = x[k][6];
        let d = Expr::var(z).plus_const(-z0[k]);
        p.soc(sigma[k].into(), u[k].iter().map(|&v| v.into()).collect());
        p.ge0(lin(&[(1.0, u[k][0]), (-cos_th, sigma[k])]));
        // mu1 (1 - d + d^2/2) <= sigma  <=>  d^2 <= (2/mu1) sigma - 2 + 2 d
        let s = Expr::var(sigma[k]).scale(2.0 / mu1[k]).plus_const(-2.0).plus(2.0, &d);
        p.square_le(&d, &s);
        // sigma <= mu2 (1 - d)
        p.ge0(Expr::constant(mu2[k]).plus(-mu2[k], &d).plus_var(-1.0, sigma[k]));
        p.ge0(d.clone());
        p.ge0(Expr::constant(z1[k]).plus_var(-1.0, z));
        // glide slope: ||r_h|| tan(gamma) <= r_up
        p.soc(x[k][0].into(), vec![Expr::var(x[k][1]).scale(tan_gs), Expr::var(x[k][2]).scale(tan_gs)]);
    }
    if let Some(d0) = sc.initial_thrust_dir() {
        attitude_constraint(&mut p, sc, &u[0], sigma[0], &d0);
    }
    let sig_nom = 0.5 * (rho1 + rho2) / sc.m0;
    let pen = attitude_penalties(&mut p, sc, &u, &sigma, sig_nom, 1.0 / dt, n - 1);
    // maximize z_N - pen
    p.add_linear(&pen.plus_var(-1.0, x[n - 1][6]));

    let sol = p.solve();
    if !sol.status.ok() {
        return None;
    }
    let g = |v: Var| sol.get(v);
    let mut info = serde_json::Map::new();
    info.insert("solver_status".into(), json!(status_str(sol.status)));
    info.insert("N".into(), json!(n));
    Some(Trajectory {
        t,
        r: x.iter().map(|xk| Vec3::new(g(xk[0]), g(xk[1]), g(xk[2]))).collect(),
        v: x.iter().map(|xk| Vec3::new(g(xk[3]), g(xk[4]), g(xk[5]))).collect(),
        m: x.iter().map(|xk| g(xk[6]).exp()).collect(),
        u: u.iter().map(|uk| Vec3::new(g(uk[0]), g(uk[1]), g(uk[2]))).collect(),
        sigma: sigma.iter().map(|&s| g(s)).collect(),
        hold: Hold::Zoh,
        info,
    })
}

pub fn status_str(s: Status) -> &'static str {
    match s {
        Status::Solved => "optimal",
        Status::AlmostSolved => "optimal_inaccurate",
        Status::Failed => "failed",
    }
}

/// Açıkmeşe & Ploen final-time bounds.
pub fn tf_bracket(sc: &Scenario) -> (f64, f64) {
    let rk = &sc.rocket;
    let alpha = rk.alpha(sc.env.g0);
    let t_min = rk.m_dry * sc.v0.norm() / rk.rho2();
    let t_max = ((sc.m0 - rk.m_dry) / (alpha * rk.rho1())).min(0.999 * sc.m0 / (alpha * rk.rho2()));
    (t_min, t_max)
}

#[derive(Debug, Clone, Default)]
pub struct SearchOpts {
    pub bracket: Option<(f64, f64)>,
    pub n_scan: Option<usize>,
    pub n_golden: Option<usize>,
}

/// Coarse scan over t_f, then golden-section refinement around the best grid point.
/// `bracket` narrows the search (warm re-planning); it is clipped to the theoretical one.
pub fn solve(sc: &Scenario, opts: &SearchOpts) -> Result<Trajectory, Infeasible> {
    let n_scan = opts.n_scan.unwrap_or(sc.gfold.n_scan);
    let n_golden = opts.n_golden.unwrap_or(sc.gfold.n_golden);
    let (mut t_lo, mut t_hi) = tf_bracket(sc);
    if let Some((a, b)) = opts.bracket {
        t_lo = t_lo.max(a);
        t_hi = t_hi.min(b);
    }
    // insertion-ordered cache of solves
    let mut cache: Vec<(f64, Option<Trajectory>)> = Vec::new();
    let cost = |tf: f64, cache: &mut Vec<(f64, Option<Trajectory>)>| -> f64 {
        let i = match cache.iter().position(|(t, _)| *t == tf) {
            Some(i) => i,
            None => {
                cache.push((tf, solve_fixed_tf(sc, tf)));
                cache.len() - 1
            }
        };
        cache[i].1.as_ref().map_or(f64::INFINITY, |tr| -tr.m[tr.len() - 1])
    };

    let grid = linspace(t_lo, t_hi, n_scan);
    let costs: Vec<f64> = grid.iter().map(|&tf| cost(tf, &mut cache)).collect();
    if !costs.iter().any(|c| c.is_finite()) {
        return Err(Infeasible("G-FOLD: no feasible final time in bracket".into()));
    }
    let i = argmin(&costs);
    let (mut a, mut b) = (grid[i.saturating_sub(1)], grid[(i + 1).min(n_scan - 1)]);
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - phi * (b - a), a + phi * (b - a));
    let (mut fc, mut fd) = (cost(c, &mut cache), cost(d, &mut cache));
    for _ in 0..n_golden {
        if fc < fd {
            (b, d, fd) = (d, c, fc);
            c = b - phi * (b - a);
            fc = cost(c, &mut cache);
        } else {
            (a, c, fc) = (c, d, fd);
            d = a + phi * (b - a);
            fd = cost(d, &mut cache);
        }
    }
    let n_solves = cache.len();
    let best = cache
        .iter()
        .enumerate()
        .filter_map(|(i, (_, tr))| tr.as_ref().map(|tr| (i, -tr.m[tr.len() - 1])))
        .fold(None, |acc: Option<(usize, f64)>, (i, c)| match acc {
            Some((_, bc)) if bc <= c => acc,
            _ => Some((i, c)),
        })
        .map(|(i, _)| i)
        .expect("a feasible solve exists");
    let mut traj = cache.swap_remove(best).1.expect("feasible");
    traj.info.insert("max_dir_rate_deg_s".into(), json!(max_dir_rate(&traj, true).to_degrees()));
    traj.info.insert("tf_bracket".into(), json!([t_lo, t_hi]));
    traj.info.insert("scan_tf".into(), json!(grid));
    traj.info.insert(
        "scan_mf".into(),
        Value::Array(costs.iter().map(|c| if c.is_finite() { json!(-c) } else { Value::Null }).collect()),
    );
    traj.info.insert("n_solves".into(), json!(n_solves));
    for (k, v) in check_solution(sc, &traj, true) {
        traj.info.insert(k.into(), json!(v));
    }
    Ok(traj)
}

fn argmin(v: &[f64]) -> usize {
    v.iter().enumerate().fold(0, |bi, (i, &x)| if x < v[bi] { i } else { bi })
}

/// Max thrust-direction rate between nodes (rad/s).
pub fn max_dir_rate(traj: &Trajectory, skip_last: bool) -> f64 {
    let n = if skip_last { traj.len() - 1 } else { traj.len() };
    let d: Vec<Vec3> = traj.u[..n].iter().map(|u| u / u.norm().max(1e-9)).collect();
    (1..n).map(|k| d[k].dot(&d[k - 1]).clamp(-1.0, 1.0).acos() / (traj.t[k] - traj.t[k - 1])).fold(0.0, f64::max)
}

/// Lossless-convexification tightness and exact thrust-bound checks.
pub fn check_solution(sc: &Scenario, traj: &Trajectory, skip_last: bool) -> [(&'static str, f64); 2] {
    let rk = &sc.rocket;
    let n = if skip_last { traj.len() - 1 } else { traj.len() };
    let mut tight = 0f64;
    let mut viol = 0f64;
    for k in 0..n {
        let (un, sig) = (traj.u[k].norm(), traj.sigma[k]);
        tight = tight.max((un - sig).abs() / sig.max(1e-12));
        let thrust = sig * traj.m[k];
        viol = viol.max(rk.rho1() - thrust).max(thrust - rk.rho2());
    }
    [("tightness_max", tight), ("thrust_bound_violation_N", viol.max(0.0))]
}
