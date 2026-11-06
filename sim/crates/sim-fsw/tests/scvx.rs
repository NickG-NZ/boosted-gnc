//! SCvx acceptance (spec test case, planning down to the engine floor).

use std::path::Path;

use sim_fsw::{
    guidance::{
        SCvxGuidance, gfold,
        scvx::{SCvxSolver, SolveFrom, last_history},
    },
    model::NominalModel,
};
use sim_scenario::Scenario;

fn spec() -> Scenario {
    let mut sc = Scenario::from_json(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../missions/spec_test_case/scenario.json"),
    )
    .unwrap();
    sc.rocket.thr_min_plan = None;
    sc
}

fn norm(s: &[f64]) -> f64 {
    s.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn scvx_converges_lands_and_saves_fuel() {
    let sc = spec();
    let g = gfold::solve(&sc, &Default::default()).unwrap();
    let s = SCvxSolver::new(&sc).solve(&g, &SolveFrom::default()).unwrap();
    assert!(s.info["converged"].as_bool().unwrap(), "{:?}", s.info["history"]);
    assert!(s.info["iterations"].as_u64().unwrap() < 15);
    assert!(last_history(&s, "nu_l1").unwrap() < 1e-6);
    let end = NominalModel::new(&sc.rocket, &sc.env, true).open_loop_resim(&s, 200);
    assert!(norm(&end[0..3]) < 0.1 && norm(&end[3..6]) < 0.1, "{end:?}");
    assert!(s.m[s.len() - 1] >= g.m[g.len() - 1]);
    assert!(s.info["tightness_max"].as_f64().unwrap() < 1e-4);
    assert!(s.info["thrust_bound_violation_N"].as_f64().unwrap() < 1.0);
}

#[test]
fn scvx_without_drag_matches_gfold() {
    let mut sc = spec();
    let g = gfold::solve(&sc, &Default::default()).unwrap();
    sc.rocket.c_d = 0.0;
    let s = SCvxSolver::new(&sc).solve(&g, &SolveFrom::default()).unwrap();
    assert!((s.tf() / g.tf() - 1.0).abs() < 2e-3, "{} {}", s.tf(), g.tf());
    assert!((s.m[s.len() - 1] / g.m[g.len() - 1] - 1.0).abs() < 1e-3);
}

#[test]
fn coast_ignition_scvx() {
    let sc = Scenario::from_json(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../missions/coast_landing/scenario.json"),
    )
    .unwrap();
    let mut g = SCvxGuidance::new(&sc);
    let plan = sim_fsw::guidance::ignition::plan_with_ignition(&sc, &mut g, false).unwrap();
    let t_ig = plan.info["t_ignition"].as_f64().unwrap();
    assert!(t_ig > 0.0 && t_ig < plan.info["t_latest_ignition"].as_f64().unwrap());
    // first thrust within the initial-attitude tolerance of falling engine-first
    let d0 = plan.u[0] / plan.u[0].norm();
    let fall = -plan.v[0] / plan.v[0].norm();
    let tol = sc.constraints.initial_attitude_tol_deg;
    assert!(d0.dot(&fall).clamp(-1.0, 1.0).acos().to_degrees() <= tol + 0.05);
    // end pose: final thrust (near) vertical
    let n = plan.len();
    let d_end = plan.u[n - 2] / plan.u[n - 2].norm();
    assert!(d_end[0].acos().to_degrees() < 1.0);
}
