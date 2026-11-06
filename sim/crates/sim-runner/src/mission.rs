//! Plan (G-FOLD / SCvx, with ignition timing for coast scenarios), simulate a scenario's cases
//! and build the data the three.js viewer reads.

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use serde_json::{Map, Value, json};
use sim_fsw::{
    guidance::{self, SCvxGuidance, ignition::plan_with_ignition},
    model::NominalModel,
    trajectory::Trajectory,
};
use sim_scenario::{GuidanceKind, Scenario};

use crate::{
    RunError,
    closed_loop::{CaseSpec, simulate},
};

/// `progress(stage, done, total, label)`: called before each plan / sim, and once with "done".
/// Returning `false` cancels the run.
pub type Progress<'a> = dyn FnMut(&str, usize, usize, &str) -> bool + 'a;

#[derive(Default)]
pub struct SolveOpts<'a> {
    /// solve only these cases (by name); None = all
    pub only: Option<Vec<String>>,
    pub verbose: bool,
    /// write each sim's plant telemetry to `<artifacts>/<mission>/<case>.telemetry.json`
    pub artifacts: Option<PathBuf>,
    pub progress: Option<Box<Progress<'a>>>,
}

pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn plan_label(k: GuidanceKind) -> &'static str {
    match k {
        GuidanceKind::Gfold => "G-FOLD plan",
        GuidanceKind::Scvx => "SCvx plan",
    }
}

/// Plan the guidance laws the selected cases need.
fn make_plans(
    sc: &Scenario,
    keys: &[GuidanceKind],
    verbose: bool,
    step: &mut dyn FnMut(&str, &str) -> Result<(), RunError>,
) -> Result<Vec<(GuidanceKind, Trajectory)>, RunError> {
    let mut plans: Vec<(GuidanceKind, Trajectory)> = Vec::new();
    for &key in keys {
        step("plan", plan_label(key))?;
        let t0 = Instant::now();
        let err = |w: &str| RunError::Guidance(format!("{}: {w}", key.key()));
        let mut p = if sc.coast.is_some() {
            // engine-off start: choose the ignition time, then plan from the predicted state
            let mut g = guidance::make(key, sc);
            plan_with_ignition(sc, g.as_mut(), verbose).map_err(|e| err(&e.0))?
        } else {
            match key {
                GuidanceKind::Gfold => guidance::make(key, sc).plan().ok_or_else(|| err("infeasible"))?,
                GuidanceKind::Scvx => {
                    let init = &plans.iter().find(|(k, _)| *k == GuidanceKind::Gfold).expect("gfold planned first").1;
                    SCvxGuidance::new(sc).plan_with_init(init).ok_or_else(|| err("no subproblem solved"))?
                }
            }
        };
        p.info.insert("solve_time_s".into(), json!(t0.elapsed().as_secs_f64()));
        let end = NominalModel::new(&sc.rocket, &sc.env, key == GuidanceKind::Scvx).open_loop_resim(&p, 200);
        let n = |s: &[f64]| s.iter().map(|x| x * x).sum::<f64>().sqrt();
        p.info.insert("open_loop_terminal_err".into(), json!([n(&end[0..3]), n(&end[3..6])]));
        plans.push((key, p));
    }
    Ok(plans)
}

/// Plan and simulate the scenario's cases (all, or those in `opts.only`). Returns the viewer data
/// `{scenario, rocket, plans, sims}`; with `only`, `sims` holds just the re-run cases (the caller
/// merges them into its stored results).
pub fn solve(scenario_path: &Path, mut opts: SolveOpts) -> Result<Value, RunError> {
    let sc = Scenario::from_json(scenario_path)?;
    let rk = &sc.rocket;
    let verbose = opts.verbose;
    if verbose {
        println!("Scenario: {} | rocket: {}", sc.name, rk.name);
    }
    let mut cases = sc.sim_cases.clone();
    if let Some(only) = &opts.only {
        let missing: Vec<String> = only.iter().filter(|n| !cases.iter().any(|c| &c.name == *n)).cloned().collect();
        if !missing.is_empty() {
            return Err(RunError::UnknownCases(missing));
        }
        cases.retain(|c| only.contains(&c.name));
    }
    let mut keys: Vec<GuidanceKind> = Vec::new();
    for k in [GuidanceKind::Gfold, GuidanceKind::Scvx] {
        let needed = cases.iter().any(|c| c.plan == k)
            // SCvx is warm-started from G-FOLD without a coast phase
            || (k == GuidanceKind::Gfold && sc.coast.is_none() && cases.iter().any(|c| c.plan == GuidanceKind::Scvx));
        if needed {
            keys.push(k);
        }
    }
    let total = keys.len() + cases.len();
    let mut done = 0usize;
    let mut progress = opts.progress.take();
    let mut step = |stage: &str, label: &str| -> Result<(), RunError> {
        let go = progress.as_mut().is_none_or(|p| p(stage, done, total, label));
        done += 1;
        if go { Ok(()) } else { Err(RunError::Cancelled) }
    };

    let plans = make_plans(&sc, &keys, verbose, &mut step)?;
    if verbose {
        println!("\nPlans");
        println!(
            "  {:8} {:>7} {:>7} {:>7} {:>10} {:>10} {:>9} {:>16}",
            "", "ignite", "h_ig", "burn", "m_f [kg]", "fuel [kg]", "tight", "resim r/v err"
        );
        for (k, p) in &plans {
            let e = &p.info["open_loop_terminal_err"];
            println!(
                "  {:8} {:6.2}s {:6.0}m {:6.2}s {:10.1} {:10.1} {:9.1e} {:7.4}m {:6.4}m/s",
                k.key(),
                p.info.get("t_ignition").and_then(|v| v.as_f64()).unwrap_or(0.0),
                p.r[0][0],
                p.tf(),
                p.m[p.len() - 1],
                sc.m0 - p.m[p.len() - 1],
                p.info["tightness_max"].as_f64().unwrap_or(f64::NAN),
                e[0].as_f64().unwrap_or(f64::NAN),
                e[1].as_f64().unwrap_or(f64::NAN),
            );
        }
    }
    let plan_json: Map<String, Value> = plans.iter().map(|(k, p)| (k.key().to_string(), p.to_json(rk.t_max))).collect();

    if verbose {
        println!("\nClosed-loop simulations");
        println!(
            "  {:62} {:>8} {:>7} {:>6} {:>5} {:>7} {:>5} {:>7}",
            "case", "pos err", "v_vert", "v_hor", "tilt", "fuel", "gsat%", "replans"
        );
    }
    let mut sims = Vec::new();
    for case in &cases {
        step("sim", &case.name)?;
        let t0 = Instant::now();
        let plan = &plans.iter().find(|(k, _)| *k == case.plan).expect("planned").1;
        let mut g = guidance::make(case.plan, &sc);
        let tel = opts.artifacts.as_ref().map(|dir| {
            let mission = scenario_path.parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().into_owned());
            dir.join(mission.unwrap_or_else(|| sc.name.clone())).join(format!("{}.telemetry.json", slug(&case.name)))
        });
        let spec = CaseSpec {
            name: &case.name,
            plan,
            truth: &case.truth,
            controller: case.controller,
            replan_hz: case.replan_hz,
            telemetry_path: tel.as_deref(),
        };
        let mut res = simulate(&sc, &spec, Some(g.as_mut()))?;
        res.metrics.insert("wall_time_s".into(), json!(t0.elapsed().as_secs_f64()));
        res.metrics.insert("plan".into(), json!(case.plan.key()));
        res.metrics.insert("truth".into(), serde_json::to_value(&case.truth).unwrap_or(Value::Null));
        if verbose {
            let m = &res.metrics;
            let f = |k: &str| m[k].as_f64().unwrap_or(f64::NAN);
            println!(
                "  {:62} {:7.2}m {:6.2} {:6.2} {:5.1} {:7.0} {:5.1} {:4}/{}",
                case.name,
                f("touchdown_pos_err_m"),
                f("touchdown_vert_speed"),
                f("touchdown_horiz_speed"),
                f("touchdown_tilt_deg"),
                f("fuel_used_kg"),
                100.0 * f("gimbal_sat_frac"),
                m["n_replans"],
                m["n_replan_failures"],
            );
        }
        let mut sim = res.to_json(rk.t_max, 2);
        sim["plan"] = plan_json[case.plan.key()].clone();
        sims.push(sim);
    }
    step("done", "")?;

    let attitude = serde_json::to_value(&rk.attitude).unwrap_or(Value::Null);
    Ok(json!({
        "scenario": {
            "name": sc.name,
            "path": scenario_path.to_string_lossy(),
            "m0": sc.m0,
            "r0": [sc.r0[0], sc.r0[1], sc.r0[2]],
            "v0": [sc.v0[0], sc.v0[1], sc.v0[2]],
            "glide_slope_deg": sc.constraints.glide_slope_deg,
            "wind": sc.env.wind,
        },
        "rocket": {
            "name": rk.name, "m_dry": rk.m_dry, "T_max": rk.t_max, "thr_min": rk.thr_min,
            "thr_max": rk.thr_max, "theta_max_deg": rk.theta_max_deg, "isp": rk.isp,
            "S_ref": rk.s_ref, "C_D": rk.c_d, "geometry": rk.geometry, "attitude": attitude,
        },
        "plans": plan_json,
        "sims": sims,
    }))
}
