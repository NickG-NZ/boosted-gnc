//! Closed-loop acceptance: plans flown through the truth plant.

use std::path::{Path, PathBuf};

use sim_fsw::{
    guidance::{self, SCvxGuidance, gfold, ignition::plan_with_ignition},
    trajectory::Trajectory,
};
use sim_runner::closed_loop::{CaseSpec, SimResult, simulate};
use sim_scenario::{ControllerKind, GuidanceKind, Scenario, TruthCase, WindSpec};

fn mission(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../../missions/{name}/scenario.json"))
}

fn spec() -> Scenario {
    let mut sc = Scenario::from_json(mission("spec_test_case")).unwrap();
    sc.rocket.thr_min_plan = None;
    sc
}

fn perturbed() -> TruthCase {
    TruthCase { drag: true, cd_scale: 1.2, isp_scale: 0.99, wind: Some(WindSpec::Constant([0.0, 0.0, 8.0])) }
}

fn fly(
    sc: &Scenario,
    plan: &Trajectory,
    truth: TruthCase,
    kind: GuidanceKind,
    hz: f64,
    ctrl: ControllerKind,
) -> SimResult {
    let mut g = guidance::make(kind, sc);
    let spec = CaseSpec { name: "test", plan, truth: &truth, controller: ctrl, replan_hz: hz, telemetry_path: None };
    simulate(sc, &spec, Some(g.as_mut())).unwrap()
}

fn f(r: &SimResult, k: &str) -> f64 {
    r.metrics[k].as_f64().unwrap_or_else(|| panic!("{k}"))
}

fn spec_scvx_plan(sc: &Scenario) -> Trajectory {
    let g = gfold::solve(sc, &Default::default()).unwrap();
    SCvxGuidance::new(sc).plan_with_init(&g).unwrap()
}

#[test]
fn closed_loop_nominal() {
    let sc = spec();
    let plan = spec_scvx_plan(&sc);
    let r = fly(&sc, &plan, TruthCase::default(), GuidanceKind::Scvx, 0.0, ControllerKind::Mpc);
    assert!(r.metrics["landed"].as_bool().unwrap());
    assert!(f(&r, "touchdown_pos_err_m") < 0.5 && f(&r, "touchdown_vert_speed") < 1.5, "{:?}", r.metrics);
}

#[test]
fn closed_loop_replan_handles_perturbation() {
    let sc = spec();
    let plan = spec_scvx_plan(&sc);
    let r = fly(&sc, &plan, perturbed(), GuidanceKind::Scvx, 2.0, ControllerKind::Mpc);
    // a rejected re-plan keeps the previous plan
    assert!(f(&r, "n_replan_failures") <= 2.0, "{:?}", r.metrics);
    assert!(f(&r, "touchdown_vert_speed") < 2.0 && f(&r, "touchdown_pos_err_m") < 1.0, "{:?}", r.metrics);
}

#[test]
fn coast_ignition_lands_with_engine_off_during_coast() {
    let sc = Scenario::from_json(mission("coast_landing")).unwrap();
    let mut g = SCvxGuidance::new(&sc);
    let plan = plan_with_ignition(&sc, &mut g, false).unwrap();
    let t_ig = plan.info["t_ignition"].as_f64().unwrap();
    let r = fly(&sc, &plan, TruthCase::default(), GuidanceKind::Scvx, 0.0, ControllerKind::Mpc);
    assert!(r.metrics["landed"].as_bool().unwrap());
    assert!(f(&r, "touchdown_vert_speed") < 1.5 && f(&r, "touchdown_pos_err_m") < 0.5);
    let coast: Vec<usize> = (0..r.t.len()).filter(|&k| r.t[k] < t_ig - 1e-6).collect();
    assert!(coast.len() > 100 && coast.iter().all(|&k| r.thrust[k].norm() == 0.0));
}

#[test]
fn gfold_plan_starts_on_attitude_and_ends_upright() {
    let sc = Scenario::from_json(mission("coast_landing")).unwrap();
    let mut g = guidance::make(GuidanceKind::Gfold, &sc);
    let plan = plan_with_ignition(&sc, g.as_mut(), false).unwrap();
    let d0 = plan.u[0] / plan.u[0].norm();
    let fall = -plan.v[0] / plan.v[0].norm();
    assert!(d0.dot(&fall).clamp(-1.0, 1.0).acos().to_degrees() <= sc.constraints.initial_attitude_tol_deg + 0.05);
    let n = plan.len();
    assert!((plan.u[n - 2][0] / plan.u[n - 2].norm()).acos().to_degrees() < 1.0);
}

#[test]
fn mpc_lands_precisely() {
    let sc = Scenario::from_json(mission("coast_landing")).unwrap();
    let plans: Vec<(GuidanceKind, Trajectory)> = [GuidanceKind::Gfold, GuidanceKind::Scvx]
        .into_iter()
        .map(|k| {
            let mut g = guidance::make(k, &sc);
            (k, plan_with_ignition(&sc, g.as_mut(), false).unwrap())
        })
        .collect();
    let plan = |k| &plans.iter().find(|(kk, _)| *kk == k).unwrap().1;
    for (kind, truth, hz) in [
        (GuidanceKind::Scvx, TruthCase::default(), 0.0),
        (GuidanceKind::Scvx, perturbed(), 0.0),
        (GuidanceKind::Gfold, TruthCase::default(), 2.0),
    ] {
        let r = fly(&sc, plan(kind), truth, kind, hz, ControllerKind::Mpc);
        let m = &r.metrics;
        assert!(m["landed"].as_bool().unwrap() && f(&r, "mpc_failures") <= 5.0, "{m:?}");
        assert!(f(&r, "touchdown_pos_err_m") < 0.5, "{m:?}");
        assert!(f(&r, "touchdown_vert_speed") < 0.5 && f(&r, "touchdown_horiz_speed") < 0.3, "{m:?}");
        assert!(f(&r, "touchdown_tilt_deg") < 0.5 && f(&r, "touchdown_body_rate_deg_s") < 0.5, "{m:?}");
    }
}
