//! G-FOLD acceptance numbers (spec test case; planning down to the engine floor).

use std::path::Path;

use sim_fsw::guidance::gfold;
use sim_scenario::Scenario;

fn spec() -> Scenario {
    let mut sc = Scenario::from_json(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../missions/spec_test_case/scenario.json"),
    )
    .unwrap();
    sc.rocket.thr_min_plan = None;
    sc
}

#[test]
fn bracket() {
    let (lo, hi) = gfold::tf_bracket(&spec());
    assert!((lo - 5.79).abs() < 0.01 && (hi - 32.73).abs() < 0.01, "{lo} {hi}");
}

#[test]
fn optimum_and_bang_bang() {
    let sc = spec();
    let tr = gfold::solve(&sc, &Default::default()).unwrap();
    assert!((tr.tf() - 12.19).abs() < 0.05, "tf {}", tr.tf());
    assert!((tr.m[tr.len() - 1] - 23150.0).abs() < 10.0, "mf {}", tr.m[tr.len() - 1]);
    let rk = &sc.rocket;
    let thr: Vec<f64> = (0..tr.len() - 1).map(|k| tr.sigma[k] * tr.m[k] / rk.t_max).collect();
    assert!((thr[0] - rk.thr_min).abs() < 1e-3, "starts at min: {}", thr[0]);
    assert!((thr[thr.len() - 1] - rk.thr_max).abs() < 1e-3, "ends at max");
    let mid = thr.iter().filter(|&&x| x > rk.thr_min + 1e-3 && x < rk.thr_max - 1e-3).count();
    assert!(mid <= 1, "{mid} transition intervals");
    assert!(tr.info["tightness_max"].as_f64().unwrap() < 1e-3);
}

#[test]
fn lossless_and_resim_lands() {
    let sc = spec();
    let tr = gfold::solve(&sc, &Default::default()).unwrap();
    assert!(tr.info["tightness_max"].as_f64().unwrap() < 1e-4);
    assert_eq!(tr.info["thrust_bound_violation_N"].as_f64().unwrap(), 0.0);
    let s = sim_fsw::model::NominalModel::new(&sc.rocket, &sc.env, false).open_loop_resim(&tr, 200);
    let (r, v) = (s[0..3].iter().map(|x| x * x).sum::<f64>().sqrt(), s[3..6].iter().map(|x| x * x).sum::<f64>().sqrt());
    assert!(r < 1e-2 && v < 1e-2, "{r} {v}");
}
