//! Engine-off coast + ignition-time selection ("light at the latest feasible time").
//!
//! Using the guidance's own model (vacuum for G-FOLD, drag for SCvx) the ballistic coast is
//! predicted and the latest ignition time with a feasible landing plan found, then backed off by
//! `margin_s` so the plan keeps a min-throttle arc (throttle authority for the tracker).
//!
//! Feasible ignition times form a window (too late: max thrust cannot stop in time; too early:
//! min thrust > weight stops the vehicle high above the pad), so we scan backward from the
//! latest coast time until a feasible point is found, then bisect the late edge.

use std::collections::{HashMap, hash_map::Entry};

use serde_json::json;
use sim_core::{Vec3, math::interp};
use sim_scenario::{CoastCfg, Scenario};

use super::Guidance;
use crate::{model::NominalModel, trajectory::Trajectory};

#[derive(Debug)]
pub struct NoIgnitionTime(pub String);

/// Coast propagation (dt 0.05 s) until `t_max` or altitude `h_floor`.
pub fn coast_propagate(model: &NominalModel, s0: [f64; 7], t_max: f64) -> (Vec<f64>, Vec<[f64; 7]>) {
    let (dt, h_floor) = (0.05, 50.0);
    let (mut ts, mut ss) = (vec![0.0], vec![s0]);
    let mut s = s0;
    while *ts.last().unwrap() < t_max && s[0] > h_floor {
        s = model.step(&s, &Vec3::zeros(), dt);
        ts.push(ts.last().unwrap() + dt);
        ss.push(s);
    }
    (ts, ss)
}

pub fn plan_with_ignition(
    sc: &Scenario,
    guidance: &mut dyn Guidance,
    verbose: bool,
) -> Result<Trajectory, NoIgnitionTime> {
    let cfg =
        sc.coast.clone().unwrap_or(CoastCfg { max_coast_s: 60.0, scan_step_s: 1.0, resolution_s: 0.1, margin_s: 1.0 });
    let model = NominalModel::new(&sc.rocket, &sc.env, guidance.models_drag());
    let s0 = [sc.r0[0], sc.r0[1], sc.r0[2], sc.v0[0], sc.v0[1], sc.v0[2], sc.m0];
    let (ts, ss) = coast_propagate(&model, s0, cfg.max_coast_s);
    let cols: Vec<Vec<f64>> = (0..7).map(|i| ss.iter().map(|s| s[i]).collect()).collect();
    let state = |t: f64| -> [f64; 7] { std::array::from_fn(|i| interp(t, &ts, &cols[i])) };

    let mut cache: HashMap<i64, Option<Trajectory>> = HashMap::new();
    let mut try_plan = |t: f64, guidance: &mut dyn Guidance| -> bool {
        let key = (t * 1000.0).round() as i64;
        if let Entry::Vacant(slot) = cache.entry(key) {
            let st = state(key as f64 / 1000.0);
            let plan = guidance.plan_from(Vec3::new(st[0], st[1], st[2]), Vec3::new(st[3], st[4], st[5]), st[6], None);
            if verbose {
                let ok = if plan.is_some() { "feasible" } else { "infeasible" };
                let speed = Vec3::new(st[3], st[4], st[5]).norm();
                println!("[ignition:{}] t_ig={:6.2}s h={:7.1}m |v|={:6.1} -> {ok}", guidance.name(), t, st[0], speed);
            }
            slot.insert(plan);
        }
        cache[&key].is_some()
    };

    let t_end = *ts.last().unwrap();
    let step = cfg.scan_step_s;
    let grid: Vec<f64> = (0..).map(|i| t_end - i as f64 * step).take_while(|&t| t > -1e-9).collect();
    let mut bracket = None;
    for (i, &tg) in grid.iter().enumerate() {
        if try_plan(tg, guidance) {
            bracket = Some((tg, if i > 0 { grid[i - 1] } else { tg }));
            break;
        }
    }
    let Some((mut lo, mut hi)) = bracket else {
        return Err(NoIgnitionTime(format!("{}: no feasible ignition time during the coast", guidance.name())));
    };
    while hi - lo > cfg.resolution_s {
        let mid = 0.5 * (lo + hi);
        if try_plan(mid, guidance) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t_latest = lo;
    let mut t_ig = (t_latest - cfg.margin_s).max(0.0);
    if !try_plan(t_ig, guidance) {
        t_ig = t_latest; // non-monotone corner case: fall back to the latest feasible time
    }
    let n_solves = cache.len();
    let mut plan = cache.remove(&((t_ig * 1000.0).round() as i64)).flatten().expect("feasible plan");

    let k = ts.partition_point(|&t| t < t_ig);
    let st = state(t_ig);
    let mut coast_t: Vec<f64> = ts[..k].iter().map(|t| round(*t, 3)).collect();
    coast_t.push(round(t_ig, 3));
    let mut coast_r: Vec<[f64; 3]> = ss[..k].iter().map(|s| [round(s[0], 3), round(s[1], 3), round(s[2], 3)]).collect();
    coast_r.push([round(st[0], 3), round(st[1], 3), round(st[2], 3)]);
    plan.info.insert("t_ignition".into(), json!(t_ig));
    plan.info.insert("t_latest_ignition".into(), json!(t_latest));
    plan.info.insert("ignition_solves".into(), json!(n_solves));
    plan.info.insert("coast".into(), json!({"t": coast_t, "r": coast_r}));
    Ok(plan)
}

pub fn round(x: f64, d: i32) -> f64 {
    let p = 10f64.powi(d);
    (x * p).round() / p
}
