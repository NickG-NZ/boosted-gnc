//! One simulated flight: flight software (guidance re-planning + cascade or MPC control) flying a
//! plan through the truth plant.
//!
//! If the plan carries `info.t_ignition` the vehicle first coasts engine-off (grid fins keep it
//! aligned with the airflow). With a guidance law and `replan_hz` > 0 it re-plans from the
//! current (perfectly known) state at ignition and then at that rate until t_go < 2 s. After the
//! plan's t_f the engine is cut (hoverslam: min thrust > weight) and it falls to contact.

use std::path::Path;

use serde_json::{Map, Value, json};
use sim_core::Vec3;
use sim_fsw::{
    ActuatorCmd, NavState,
    control::{
        cascade::CascadeController,
        mpc::{MpcSolution, TrackingMpc, embed_gimbal},
    },
    guidance::{Guidance, ignition::round},
    trajectory::Trajectory,
};
use sim_scenario::{ControllerKind, E1, Scenario, TruthCase};
use sim_vehicle::{Command, Plant, StepInfo};

use crate::{RunError, truth::plant_config};

const FREEZE_TGO: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct SimResult {
    pub name: String,
    pub t: Vec<f64>,
    pub r: Vec<Vec3>,
    pub v: Vec<Vec3>,
    pub m: Vec<f64>,
    pub thrust: Vec<Vec3>,
    pub r_ref: Vec<Vec3>,
    /// body axis
    pub att: Vec<Vec3>,
    /// gimbal vector (perpendicular to att, |g| = sin(deflection)), zero with the engine off
    pub gimbal: Vec<Vec3>,
    /// truth wind at the vehicle
    pub wind: Vec<Vec3>,
    pub metrics: Map<String, Value>,
    pub replans: Vec<Value>,
}

impl SimResult {
    /// Viewer export: every `decimate`-th sample plus the last.
    pub fn to_json(&self, t_max: f64, decimate: usize) -> Value {
        let n = self.t.len();
        let mut idx: Vec<usize> = (0..n - 1).step_by(decimate).collect();
        idx.push(n - 1);
        let vs = |a: &[Vec3], d: i32| -> Vec<[f64; 3]> {
            idx.iter().map(|&i| [round(a[i][0], d), round(a[i][1], d), round(a[i][2], d)]).collect()
        };
        let ss = |a: &[f64], d: i32| -> Vec<f64> { idx.iter().map(|&i| round(a[i], d)).collect() };
        let throttle: Vec<f64> = self.thrust.iter().map(|t| t.norm() / t_max).collect();
        json!({
            "name": self.name,
            "t": ss(&self.t, 4),
            "r": vs(&self.r, 4),
            "v": vs(&self.v, 4),
            "m": ss(&self.m, 2),
            "T": vs(&self.thrust, 1),
            "throttle": ss(&throttle, 4),
            "r_ref": vs(&self.r_ref, 4),
            "att": vs(&self.att, 5),
            "gimbal": vs(&self.gimbal, 5),
            "wind": vs(&self.wind, 3),
            "metrics": self.metrics,
            "replans": self.replans,
        })
    }
}

fn nav(p: &Plant) -> NavState {
    let s = p.state();
    NavState { t: s.t, r: s.r, v: s.v, m: s.m, b: s.b, w: s.w, g: s.g }
}

fn step(p: &mut Plant, cmd: &ActuatorCmd, dt: f64) -> Result<StepInfo, RunError> {
    Ok(p.step(&Command { thrust: cmd.thrust, gimbal: cmd.gimbal, fins: cmd.fins }, dt)?)
}

/// Max angular rate of a unit-vector history (rad/s).
fn max_rate(t: &[f64], att: &[Vec3]) -> f64 {
    (1..t.len())
        .filter(|&k| t[k] - t[k - 1] > 1e-6)
        .map(|k| att[k].dot(&att[k - 1]).clamp(-1.0, 1.0).acos() / (t[k] - t[k - 1]))
        .fold(0.0, f64::max)
}

pub struct CaseSpec<'a> {
    pub name: &'a str,
    pub plan: &'a Trajectory,
    pub truth: &'a TruthCase,
    pub controller: ControllerKind,
    pub replan_hz: f64,
    pub telemetry_path: Option<&'a Path>,
}

pub fn simulate(
    sc: &Scenario,
    case: &CaseSpec,
    mut guidance: Option<&mut dyn Guidance>,
) -> Result<SimResult, RunError> {
    let rk = &sc.rocket;
    let dt = sc.tracker.dt;
    let plan = case.plan;
    let t_ig = plan.info.get("t_ignition").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let mut active = plan.clone();
    let mut plan_t0 = t_ig;
    let replan_dt = if guidance.is_some() && case.replan_hz > 0.0 { 1.0 / case.replan_hz } else { f64::INFINITY };
    let mut next_replan = if replan_dt.is_finite() { if t_ig > 0.0 { t_ig } else { replan_dt } } else { f64::INFINITY };
    let mut replans: Vec<Value> = Vec::new();
    let mut n_replan_fail = 0usize;

    // truth model; with the engine off the grid fins keep the body aligned with the airflow
    let probe = Plant::new(plant_config(sc, case.truth, &E1))?;
    let airflow_axis = |plant: &Plant, r: &Vec3, v: &Vec3| -> Vec3 {
        let vr = v - plant.wind_at(r);
        let n = vr.norm();
        if n > 1e-6 { -vr / n } else { E1 }
    };
    let b0 = match sc.att0 {
        Some(a) => a / a.norm(),
        None => airflow_axis(&probe, &sc.r0, &sc.v0),
    };
    let mut plant = Plant::new(plant_config(sc, case.truth, &b0))?;

    let mut cascade = CascadeController::new(sc, &b0);
    let use_mpc = case.controller == ControllerKind::Mpc;
    let mpc = use_mpc.then(|| TrackingMpc::new(sc, case.truth.drag));
    let (mut sol, mut sol_t0, mut next_mpc): (Option<MpcSolution>, f64, f64) = (None, 0.0, f64::NEG_INFINITY);
    let (mut mpc_solves, mut mpc_fail, mut mpc_time) = (0usize, 0usize, 0.0f64);

    let mut log_ref: Vec<Vec3> = Vec::new();
    let (mut max_gimbal, mut steps, mut max_track_err, mut gimbal_sat_steps) = (0.0f64, 0usize, 0.0f64, 0usize);

    loop {
        let nv = nav(&plant);
        let t = nv.t;
        let mut tau = t - plan_t0;
        let mut t_go = active.tf() - tau;

        if t < t_ig - 1e-9 {
            // engine-off coast before ignition
            let axis = airflow_axis(&plant, &nv.r, &nv.v);
            let cmd = cascade.fins_toward(&nv, &axis);
            let info = step(&mut plant, &cmd, dt.min(t_ig - t))?;
            let s = plant.state();
            cascade.reference.reset(&s.b, &s.w);
            log_ref.push(nv.r);
            if info.contact {
                break;
            }
            continue;
        }

        if t >= next_replan - 1e-9 {
            next_replan += replan_dt;
            if t_go > FREEZE_TGO
                && let Some(g) = guidance.as_deref_mut()
            {
                match g.replan(nv.r, nv.v, nv.m, &active, tau, Some(nv.b)) {
                    None => n_replan_fail += 1,
                    Some(new) => {
                        active = new;
                        (plan_t0, tau) = (t, 0.0);
                        t_go = active.tf();
                        next_mpc = f64::NEG_INFINITY;
                        let relaxed = active.info.get("relaxed_attitude").and_then(|v| v.as_bool()).unwrap_or(false);
                        let r: Vec<[f64; 3]> =
                            active.r.iter().map(|r| [round(r[0], 3), round(r[1], 3), round(r[2], 3)]).collect();
                        replans.push(json!({"t0": t, "tf": active.tf(), "r": r, "relaxed": relaxed}));
                    }
                }
            }
        }

        let (r_ref, info) = if t_go > 1e-9 && use_mpc {
            let mpc = mpc.as_ref().expect("mpc");
            let (r_ref, _) = active.state_at(tau);
            let h = dt.min(t_go);
            if t >= next_mpc - 1e-9 && (sol.is_none() || t_go > mpc.p.stop_tgo) {
                next_mpc = t + mpc.p.dt0;
                mpc_solves += 1;
                match mpc.solve(&active, tau, &nv) {
                    None => mpc_fail += 1,
                    Some(s) => {
                        mpc_time += s.solve_s;
                        sol = Some(s);
                        sol_t0 = t;
                    }
                }
            }
            let s = sol.as_ref().ok_or_else(|| RunError::Control("MPC failed on its first solve".into()))?;
            let tn = s.command(t + 0.5 * h - sol_t0).0; // thrust held over the interval
            let g_h = s.command(t + h - sol_t0).1; // gimbal target at the end of this step
            let tn = if nv.m > rk.m_dry { tn.clamp(rk.t_min(), rk.t_max) } else { 0.0 };
            let cmd = ActuatorCmd { thrust: tn, gimbal: Some(embed_gimbal(g_h, &nv.b)), fins: None };
            max_track_err = max_track_err.max((r_ref - nv.r).norm());
            (r_ref, step(&mut plant, &cmd, h)?)
        } else if t_go > 1e-9 {
            let (r_ref, _) = active.state_at(tau);
            let h = dt.min(t_go);
            let cmd = cascade.command(&active, tau, t_go, &nv, h);
            max_track_err = max_track_err.max((r_ref - nv.r).norm());
            (r_ref, step(&mut plant, &cmd, h)?)
        } else {
            // engine cut at the end of the plan: ballistic until contact (fins hold attitude)
            let cmd = cascade.fins_toward(&nv, &nv.b);
            (Vec3::zeros(), step(&mut plant, &cmd, dt)?)
        };
        log_ref.push(r_ref);
        if info.thrust > 0.0 {
            let gd = plant.state().gimbal_deg();
            max_gimbal = max_gimbal.max(gd);
            if gd >= 0.98 * rk.attitude.gimbal_max_deg {
                gimbal_sat_steps += 1;
            }
        }
        steps += 1;
        if info.contact || plant.state().t > plan_t0 + active.tf() + 30.0 {
            break;
        }
    }

    plant.finish();
    log_ref.push(*log_ref.last().unwrap_or(&Vec3::zeros()));
    let rec = plant.recorder().expect("recording enabled");
    let col1 = |name: &str| rec.get(name).expect("channel").data.clone();
    let col3 = |name: &str| -> Vec<Vec3> {
        let c = rec.get(name).expect("channel");
        (0..c.rows()).map(|i| Vec3::from_column_slice(c.row(i))).collect()
    };
    let s = plant.state();
    let mut res = SimResult {
        name: case.name.into(),
        t: col1("t"),
        r: col3("r"),
        v: col3("v"),
        m: col1("m"),
        thrust: col3("thrust"),
        r_ref: log_ref,
        att: col3("b"),
        gimbal: col3("g"),
        wind: col3("wind"),
        metrics: Map::new(),
        replans,
    };
    let (r_end, v_end, m_end) = (*res.r.last().unwrap(), *res.v.last().unwrap(), *res.m.last().unwrap());
    let relaxed = res.replans.iter().filter(|r| r["relaxed"].as_bool() == Some(true)).count();
    let m = &mut res.metrics;
    m.insert("landed".into(), json!(r_end[0] <= 1e-6));
    m.insert("touchdown_time".into(), json!(res.t.last().unwrap()));
    m.insert("plan_tf".into(), json!(plan.tf()));
    m.insert("t_ignition".into(), json!(t_ig));
    m.insert("touchdown_pos_err_m".into(), json!(r_end[1].hypot(r_end[2])));
    m.insert("touchdown_vert_speed".into(), json!(-v_end[0]));
    m.insert("touchdown_horiz_speed".into(), json!(v_end[1].hypot(v_end[2])));
    m.insert("touchdown_tilt_deg".into(), json!(s.b[0].clamp(-1.0, 1.0).acos().to_degrees()));
    m.insert("touchdown_body_rate_deg_s".into(), json!(s.w.norm().to_degrees()));
    m.insert("max_slew_rate_deg_s".into(), json!(max_rate(&res.t, &res.att).to_degrees()));
    m.insert("max_gimbal_deg".into(), json!(max_gimbal));
    m.insert("fuel_used_kg".into(), json!(sc.m0 - m_end));
    m.insert("fuel_margin_kg".into(), json!(m_end - rk.m_dry));
    m.insert("max_tracking_err_m".into(), json!(max_track_err));
    m.insert("gimbal_sat_frac".into(), json!(gimbal_sat_steps as f64 / steps.max(1) as f64));
    m.insert("controller".into(), json!(case.controller.key()));
    m.insert("replan_hz".into(), json!(case.replan_hz));
    m.insert("n_replans".into(), json!(res.replans.len()));
    m.insert("n_replan_failures".into(), json!(n_replan_fail));
    m.insert("n_replans_relaxed".into(), json!(relaxed));
    if use_mpc {
        m.insert("mpc_solves".into(), json!(mpc_solves));
        m.insert("mpc_failures".into(), json!(mpc_fail));
        m.insert("mpc_mean_solve_ms".into(), json!(1e3 * mpc_time / (mpc_solves - mpc_fail).max(1) as f64));
    }
    if let Some(path) = case.telemetry_path {
        let meta = json!({"name": case.name, "scenario": sc.name, "metrics": res.metrics});
        sim_telemetry::artifact::write_json(path, rec, meta)
            .map_err(|e| RunError::Control(format!("writing telemetry: {e}")))?;
    }
    Ok(res)
}
