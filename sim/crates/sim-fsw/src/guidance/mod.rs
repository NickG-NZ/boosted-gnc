//! Guidance: trajectory planning, closed-loop re-planning and ignition timing.
//!
//! Each guidance law implements `Guidance`:
//!   - `plan()`            plan from the scenario's initial state
//!   - `plan_from(..)`     cold-start plan from an arbitrary state (None = infeasible / unflyable)
//!   - `replan(..)`        plan from the current state, warm-started from the previous plan
//!   - `models_drag()`     whether its internal model includes drag

pub mod gfold;
pub mod ignition;
pub mod scvx;

use serde_json::json;
use sim_core::Vec3;
use sim_scenario::{GuidanceKind, Scenario};

use crate::trajectory::{Hold, Trajectory};
use gfold::SearchOpts;
use scvx::{SCvxSolver, SolveFrom, last_history};

pub trait Guidance {
    fn name(&self) -> &'static str;
    fn models_drag(&self) -> bool;
    fn plan(&mut self) -> Option<Trajectory>;
    fn plan_from(&mut self, r: Vec3, v: Vec3, m: f64, att: Option<Vec3>) -> Option<Trajectory>;
    fn replan(
        &mut self,
        r: Vec3,
        v: Vec3,
        m: f64,
        prev: &Trajectory,
        tau: f64,
        att: Option<Vec3>,
    ) -> Option<Trajectory>;
}

pub fn make(kind: GuidanceKind, sc: &Scenario) -> Box<dyn Guidance> {
    match kind {
        GuidanceKind::Gfold => Box::new(GFoldGuidance::new(sc)),
        GuidanceKind::Scvx => Box::new(SCvxGuidance::new(sc)),
    }
}

/// Lossless convexification must be tight (||u|| = sigma) for a plan to be physical; slack means
/// the plan pays for thrust it does not deliver (effectively throttling below the engine floor).
pub fn is_flyable(traj: &Trajectory, tol: f64) -> bool {
    traj.info.get("tightness_max").and_then(|v| v.as_f64()).unwrap_or(0.0) <= tol
}

/// Classic SCvx initialization: states interpolate linearly to the pad, constant deceleration.
pub fn straight_line_guess(sc: &Scenario, n: usize) -> Trajectory {
    let (rk, g0) = (&sc.rocket, sc.env.g0);
    let a_brake = 0.7 * rk.rho2() / sc.m0 - g0;
    let tf = sc.v0.norm() / a_brake.max(1.0) * 1.3;
    let s: Vec<f64> = gfold::linspace(0.0, 1.0, n);
    let u = -sc.v0 / tf - sc.env.g();
    let alpha = rk.alpha(g0);
    Trajectory {
        t: s.iter().map(|s| s * tf).collect(),
        r: s.iter().map(|s| sc.r0 * (1.0 - s)).collect(),
        v: s.iter().map(|s| sc.v0 * (1.0 - s)).collect(),
        m: s.iter().map(|s| (sc.m0 - alpha * u.norm() * sc.m0 * s * tf).max(rk.m_dry)).collect(),
        u: vec![u; n],
        sigma: vec![u.norm(); n],
        hold: Hold::Foh,
        info: Default::default(),
    }
}

/// Drag-free lossless-convexification guidance (Phase 1).
pub struct GFoldGuidance {
    sc: Scenario,
}

impl GFoldGuidance {
    pub fn new(sc: &Scenario) -> Self {
        Self { sc: sc.clone() }
    }
}

impl Guidance for GFoldGuidance {
    fn name(&self) -> &'static str {
        "gfold"
    }
    fn models_drag(&self) -> bool {
        false
    }
    fn plan(&mut self) -> Option<Trajectory> {
        gfold::solve(&self.sc, &SearchOpts::default()).ok()
    }
    fn plan_from(&mut self, r: Vec3, v: Vec3, m: f64, att: Option<Vec3>) -> Option<Trajectory> {
        let sc = self.sc.from_state(r, v, m, att);
        gfold::solve(&sc, &SearchOpts::default()).ok().filter(|t| is_flyable(t, 1e-3))
    }
    fn replan(
        &mut self,
        r: Vec3,
        v: Vec3,
        m: f64,
        prev: &Trajectory,
        tau: f64,
        att: Option<Vec3>,
    ) -> Option<Trajectory> {
        let sc = self.sc.from_state(r, v, m, att);
        let t_go = prev.tf() - tau;
        // if the plan must start from the current attitude and none exists, drop that requirement
        // (the attitude dynamics in the sim still apply) rather than give up
        let mut relaxed = sc.clone();
        relaxed.constraints.constrain_initial_attitude = false;
        // warm: narrow t_f bracket around the previous time-to-go; then the full bracket
        let attempts = [(&sc, Some((0.6 * t_go, 1.6 * t_go)), false), (&sc, None, false), (&relaxed, None, true)];
        for (s_try, bracket, is_relaxed) in attempts {
            let opts = SearchOpts { bracket, n_scan: Some(if bracket.is_some() { 9 } else { 15 }), n_golden: Some(10) };
            if let Ok(mut traj) = gfold::solve(s_try, &opts) {
                traj.info.insert("relaxed_attitude".into(), json!(is_relaxed));
                return Some(traj);
            }
        }
        None
    }
}

/// Drag-aware successive-convexification guidance (Phase 2), warm-started from G-FOLD.
pub struct SCvxGuidance {
    sc: Scenario,
    solver: SCvxSolver,
    replan_iters: usize,
}

impl SCvxGuidance {
    pub fn new(sc: &Scenario) -> Self {
        Self { sc: sc.clone(), solver: SCvxSolver::new(sc), replan_iters: 8 }
    }

    pub fn plan_with_init(&self, init: &Trajectory) -> Option<Trajectory> {
        self.solver.solve(init, &SolveFrom::default()).ok()
    }
}

impl Guidance for SCvxGuidance {
    fn name(&self) -> &'static str {
        "scvx"
    }
    fn models_drag(&self) -> bool {
        true
    }
    fn plan(&mut self) -> Option<Trajectory> {
        let init = gfold::solve(&self.sc, &SearchOpts::default()).ok()?;
        self.plan_with_init(&init)
    }
    /// Initialized from G-FOLD at the same state when that is feasible (drag-free), otherwise from
    /// a straight-line guess (drag can make states feasible that are infeasible in vacuum).
    fn plan_from(&mut self, r: Vec3, v: Vec3, m: f64, att: Option<Vec3>) -> Option<Trajectory> {
        let sc = self.sc.from_state(r, v, m, att);
        let init = gfold::solve(&sc, &SearchOpts::default()).unwrap_or_else(|_| straight_line_guess(&sc, 30));
        let from = SolveFrom { r0: Some(r), v0: Some(v), m0: Some(m), att0: att, max_iters: Some(20) };
        let traj = self.solver.solve(&init, &from).ok()?;
        let converged = traj.info.get("converged").and_then(|v| v.as_bool()).unwrap_or(false);
        let ok = converged || last_history(&traj, "nu_l1").unwrap_or(1.0) < 1e-5;
        let defect = traj.info.get("defect_l1").and_then(|v| v.as_f64()).unwrap_or(1.0);
        (ok && defect < 1e-4 && is_flyable(&traj, 1e-3)).then_some(traj)
    }
    fn replan(
        &mut self,
        r: Vec3,
        v: Vec3,
        m: f64,
        prev: &Trajectory,
        tau: f64,
        att: Option<Vec3>,
    ) -> Option<Trajectory> {
        let guess = prev.shifted(tau, prev.len());
        let from = SolveFrom { r0: Some(r), v0: Some(v), m0: Some(m), att0: att, max_iters: Some(self.replan_iters) };
        let new = self.solver.solve(&guess, &from).ok()?;
        // virtual control still large -> plan not dynamically consistent; reject it
        if last_history(&new, "nu_l1").unwrap_or(0.0) > 1e-3 {
            return None;
        }
        Some(new)
    }
}
