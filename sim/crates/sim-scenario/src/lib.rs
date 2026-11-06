//! Mission configuration, loaded from the JSON files in `missions/` and `vehicles/`.
//!
//! Frame: landing-site fixed, index 0 is "up", indices 1-2 horizontal. Units SI.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sim_core::Vec3;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("parsing {0}: {1}")]
    Parse(PathBuf, serde_json::Error),
}

// ------------------------------------------------------------------------------------ vehicle

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttitudeParams {
    #[serde(default = "d_gimbal_max")]
    pub gimbal_max_deg: f64,
    #[serde(default = "d_gimbal_rate")]
    pub gimbal_rate_deg_s: f64,
    /// transverse moment of inertia about the CG (kg m^2)
    #[serde(default = "d_inertia")]
    pub inertia: f64,
    /// engine gimbal point to CG (m)
    #[serde(default = "d_arm")]
    pub arm: f64,
    #[serde(default = "d_body_rate")]
    pub body_rate_max_deg_s: f64,
    /// angle error -> rate command (1/s)
    #[serde(default = "d_k_theta")]
    pub k_theta: f64,
    /// rate error -> angular accel command (1/s)
    #[serde(default = "d_k_omega")]
    pub k_omega: f64,
    #[serde(default = "d_fin_alpha")]
    pub fin_alpha_max_deg_s2: f64,
    /// fraction of angular-accel authority budgeted for stopping rotations
    #[serde(default = "d_half")]
    pub brake_frac: f64,
}

impl Default for AttitudeParams {
    fn default() -> Self {
        serde_json::from_value(Value::Object(Default::default())).expect("defaults")
    }
}

fn d_gimbal_max() -> f64 {
    8.0
}
fn d_gimbal_rate() -> f64 {
    20.0
}
fn d_inertia() -> f64 {
    3.5e6
}
fn d_arm() -> f64 {
    15.0
}
fn d_body_rate() -> f64 {
    12.0
}
fn d_k_theta() -> f64 {
    2.0
}
fn d_k_omega() -> f64 {
    4.0
}
fn d_fin_alpha() -> f64 {
    15.0
}
fn d_half() -> f64 {
    0.5
}
fn d_one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rocket {
    pub name: String,
    pub m_dry: f64,
    pub isp: f64,
    #[serde(rename = "T_max")]
    pub t_max: f64,
    /// hard engine floor, fraction of T_max
    pub thr_min: f64,
    /// planning derate, fraction of T_max
    pub thr_max: f64,
    pub theta_max_deg: f64,
    #[serde(rename = "S_ref", default)]
    pub s_ref: f64,
    #[serde(rename = "C_D", default)]
    pub c_d: f64,
    /// C_D multiplier while thrusting (plume shielding)
    #[serde(default = "d_one")]
    pub kappa: f64,
    /// planning floor (> thr_min leaves throttle-down margin)
    #[serde(default)]
    pub thr_min_plan: Option<f64>,
    #[serde(default)]
    pub attitude: AttitudeParams,
    #[serde(default)]
    pub geometry: Value,
}

impl Rocket {
    /// Mass flow per unit thrust (kg/s/N).
    pub fn alpha(&self, g0: f64) -> f64 {
        1.0 / (self.isp * g0)
    }
    /// Planning thrust floor (N).
    pub fn rho1(&self) -> f64 {
        self.thr_min_plan.unwrap_or(self.thr_min) * self.t_max
    }
    /// Planning thrust ceiling (N).
    pub fn rho2(&self) -> f64 {
        self.thr_max * self.t_max
    }
    /// Hard engine floor (N).
    pub fn t_min(&self) -> f64 {
        self.thr_min * self.t_max
    }
    pub fn theta_max(&self) -> f64 {
        self.theta_max_deg.to_radians()
    }
    pub fn cds_thrusting(&self) -> f64 {
        self.c_d * self.kappa * self.s_ref
    }
}

// ------------------------------------------------------------------------------------ environment

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    #[serde(default = "d_g0")]
    pub g0: f64,
    #[serde(default = "d_rho0")]
    pub rho0: f64,
    #[serde(rename = "H", default = "d_h")]
    pub h: f64,
    #[serde(default)]
    pub wind: [f64; 3],
}

fn d_g0() -> f64 {
    9.80665
}
fn d_rho0() -> f64 {
    1.225
}
fn d_h() -> f64 {
    8500.0
}

impl Environment {
    pub fn g(&self) -> Vec3 {
        Vec3::new(-self.g0, 0.0, 0.0)
    }
    pub fn wind_vec(&self) -> Vec3 {
        Vec3::from(self.wind)
    }
    pub fn rho_atm(&self, h: f64) -> f64 {
        self.rho0 * (-h / self.h).exp()
    }
}

// ------------------------------------------------------------------------------------ settings

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraints {
    pub glide_slope_deg: f64,
    #[serde(default)]
    pub constrain_initial_attitude: bool,
    #[serde(default)]
    pub initial_attitude_tol_deg: f64,
    #[serde(default)]
    pub end_tilt_penalty_kg_per_deg: Option<f64>,
    #[serde(default = "d_end_nodes")]
    pub end_tilt_nodes: usize,
    /// kg per (deg/s)^2 per second
    #[serde(default)]
    pub rotation_rate_penalty: Option<f64>,
    #[serde(default)]
    pub slack_penalty_kg_per_pct: Option<f64>,
}

fn d_end_nodes() -> usize {
    3
}

/// `Some(x)` with x != 0 (Python truthiness of an optional number).
pub fn nonzero(x: Option<f64>) -> Option<f64> {
    x.filter(|v| *v != 0.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GFoldCfg {
    #[serde(rename = "N", default = "d_gfold_n")]
    pub n: usize,
    #[serde(default = "d_n_scan")]
    pub n_scan: usize,
    #[serde(default = "d_n_golden")]
    pub n_golden: usize,
}

impl Default for GFoldCfg {
    fn default() -> Self {
        Self { n: 40, n_scan: 25, n_golden: 20 }
    }
}
fn d_gfold_n() -> usize {
    40
}
fn d_n_scan() -> usize {
    25
}
fn d_n_golden() -> usize {
    20
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScvxCfg {
    #[serde(rename = "N")]
    pub n: usize,
    pub max_iters: usize,
    pub w_nu: f64,
    pub eta0: f64,
    pub eta_min: f64,
    pub eta_max: f64,
    pub r0: f64,
    pub r1: f64,
    pub r2: f64,
    pub a: f64,
    pub b: f64,
    pub eps_nu: f64,
    pub eps_x: f64,
}

impl Default for ScvxCfg {
    fn default() -> Self {
        Self {
            n: 30,
            max_iters: 30,
            w_nu: 1e4,
            eta0: 1.0,
            eta_min: 1e-3,
            eta_max: 10.0,
            r0: 0.0,
            r1: 0.25,
            r2: 0.7,
            a: 2.0,
            b: 3.2,
            eps_nu: 1e-6,
            eps_x: 1e-4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackerCfg {
    pub omega_n: f64,
    pub zeta: f64,
    pub omega_n_lat: Option<f64>,
    pub zeta_lat: Option<f64>,
    pub lateral_fade_s: f64,
    pub dt: f64,
    pub ref_omega: f64,
    pub cmd_rate_deg_s: f64,
    pub ref_accel_frac: f64,
}

impl Default for TrackerCfg {
    fn default() -> Self {
        Self {
            omega_n: 1.0,
            zeta: 0.7,
            omega_n_lat: None,
            zeta_lat: None,
            lateral_fade_s: 0.0,
            dt: 0.01,
            ref_omega: 1.5,
            cmd_rate_deg_s: 8.0,
            ref_accel_frac: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MpcCfg {
    /// horizon intervals (shrinking horizon to the plan's t_f)
    #[serde(rename = "N")]
    pub n: usize,
    /// first interval = re-solve period (s); later intervals grow geometrically
    pub dt0: f64,
    /// stop re-solving this close to t_f
    pub stop_tgo: f64,
    pub w_r_lat: f64,
    pub w_r_up: f64,
    pub w_v: f64,
    pub w_tilt: f64,
    pub w_rate: f64,
    pub w_gimbal_term: f64,
    pub w_dr: f64,
    pub w_dv: f64,
    pub w_db: f64,
    pub w_cone: f64,
    pub huber_r: f64,
    pub huber_v: f64,
    pub w_ug: f64,
    #[serde(rename = "w_dT")]
    pub w_dt: f64,
}

impl Default for MpcCfg {
    fn default() -> Self {
        Self {
            n: 30,
            dt0: 0.05,
            stop_tgo: 0.15,
            w_r_lat: 20.0,
            w_r_up: 500.0,
            w_v: 20.0,
            w_tilt: 3.0e4,
            w_rate: 3.0e4,
            w_gimbal_term: 1.0e3,
            w_dr: 0.02,
            w_dv: 0.05,
            w_db: 300.0,
            w_cone: 1.0e6,
            huber_r: 5.0,
            huber_v: 5.0,
            w_ug: 20.0,
            w_dt: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoastCfg {
    #[serde(default = "d_max_coast")]
    pub max_coast_s: f64,
    #[serde(default = "d_one")]
    pub scan_step_s: f64,
    #[serde(default = "d_res")]
    pub resolution_s: f64,
    #[serde(default = "d_one")]
    pub margin_s: f64,
}

fn d_max_coast() -> f64 {
    60.0
}
fn d_res() -> f64 {
    0.1
}

// ------------------------------------------------------------------------------------ sim cases

/// Truth wind: a constant `[up, x, y]` vector or a shear + gust profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WindSpec {
    Constant([f64; 3]),
    Profile(WindProfile),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindProfile {
    #[serde(default)]
    pub mean: [f64; 3],
    #[serde(default = "d_ref_alt")]
    pub ref_alt: f64,
    #[serde(default)]
    pub shear: f64,
    #[serde(default)]
    pub gust_sigma: f64,
    #[serde(default = "d_gust_len")]
    pub gust_length: f64,
    #[serde(default)]
    pub seed: u64,
}

fn d_ref_alt() -> f64 {
    10.0
}
fn d_gust_len() -> f64 {
    200.0
}

/// Truth-model deviations a case flies against (guidance never sees them).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthCase {
    #[serde(default = "d_true")]
    pub drag: bool,
    #[serde(default = "d_one")]
    pub cd_scale: f64,
    #[serde(default = "d_one")]
    pub isp_scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wind: Option<WindSpec>,
}

impl Default for TruthCase {
    fn default() -> Self {
        Self { drag: true, cd_scale: 1.0, isp_scale: 1.0, wind: None }
    }
}

fn d_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidanceKind {
    Gfold,
    Scvx,
}

impl GuidanceKind {
    pub fn key(self) -> &'static str {
        match self {
            GuidanceKind::Gfold => "gfold",
            GuidanceKind::Scvx => "scvx",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ControllerKind {
    #[default]
    Cascade,
    Mpc,
}

impl ControllerKind {
    pub fn key(self) -> &'static str {
        match self {
            ControllerKind::Cascade => "cascade",
            ControllerKind::Mpc => "mpc",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimCase {
    pub name: String,
    pub plan: GuidanceKind,
    #[serde(default)]
    pub truth: TruthCase,
    #[serde(default)]
    pub replan_hz: f64,
    #[serde(default)]
    pub controller: ControllerKind,
}

// ------------------------------------------------------------------------------------ scenario

#[derive(Debug, Clone)]
pub struct Scenario {
    pub name: String,
    pub path: PathBuf,
    pub rocket: Rocket,
    pub env: Environment,
    pub m0: f64,
    pub r0: Vec3,
    pub v0: Vec3,
    pub constraints: Constraints,
    pub gfold: GFoldCfg,
    pub scvx: ScvxCfg,
    pub tracker: TrackerCfg,
    pub mpc: MpcCfg,
    /// `None` -> engine lit at t = 0
    pub coast: Option<CoastCfg>,
    pub sim_cases: Vec<SimCase>,
    /// current body axis (overrides the default initial thrust direction)
    pub att0: Option<Vec3>,
}

#[derive(Deserialize)]
struct RawScenario {
    name: Option<String>,
    rocket: Value,
    #[serde(default)]
    environment: Option<Environment>,
    initial_state: RawInitial,
    constraints: Constraints,
    #[serde(default)]
    gfold: Option<GFoldCfg>,
    #[serde(default)]
    scvx: Option<Value>,
    #[serde(default)]
    tracker: TrackerCfg,
    #[serde(default)]
    mpc: MpcCfg,
    #[serde(default)]
    coast: Option<CoastCfg>,
    #[serde(default)]
    sim_cases: Vec<SimCase>,
}

#[derive(Deserialize)]
struct RawInitial {
    m0: f64,
    r0: [f64; 3],
    v0: [f64; 3],
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ConfigError> {
    let text = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(path.into(), e))?;
    serde_json::from_str(&text).map_err(|e| ConfigError::Parse(path.into(), e))
}

/// Merge `over` onto the defaults of `T` (partial config objects, like `{**DEFAULTS, **cfg}`).
fn with_defaults<T: Serialize + for<'de> Deserialize<'de> + Default>(over: Option<Value>) -> serde_json::Result<T> {
    let mut base = serde_json::to_value(T::default())?;
    if let (Value::Object(b), Some(Value::Object(o))) = (&mut base, over) {
        for (k, v) in o {
            if !k.starts_with('_') {
                b.insert(k, v);
            }
        }
    }
    serde_json::from_value(base)
}

impl Scenario {
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let raw: RawScenario = read_json(path)?;
        let rocket: Rocket = match raw.rocket {
            Value::String(rel) => {
                let rp = path.parent().unwrap_or(Path::new(".")).join(rel);
                read_json(&rp)?
            }
            v => serde_json::from_value(v).map_err(|e| ConfigError::Parse(path.into(), e))?,
        };
        let scvx = with_defaults::<ScvxCfg>(raw.scvx).map_err(|e| ConfigError::Parse(path.into(), e))?;
        let name = raw.name.unwrap_or_else(|| path.file_stem().unwrap_or_default().to_string_lossy().into_owned());
        Ok(Self {
            name,
            path: path.to_path_buf(),
            rocket,
            env: raw.environment.unwrap_or_else(|| serde_json::from_str("{}").expect("defaults")),
            m0: raw.initial_state.m0,
            r0: Vec3::from(raw.initial_state.r0),
            v0: Vec3::from(raw.initial_state.v0),
            constraints: raw.constraints,
            gfold: raw.gfold.unwrap_or_default(),
            scvx,
            tracker: raw.tracker,
            mpc: raw.mpc,
            coast: raw.coast,
            sim_cases: raw.sim_cases,
            att0: None,
        })
    }

    pub fn gamma_gs(&self) -> f64 {
        self.constraints.glide_slope_deg.to_radians()
    }

    /// Allowed angle between the first planned thrust and the current body axis (rad).
    pub fn initial_attitude_tol(&self) -> f64 {
        self.constraints.initial_attitude_tol_deg.to_radians()
    }

    /// Direction the first thrust must point, if `constrain_initial_attitude`: the current body
    /// axis, by default falling engine-first (wind-relative velocity reversed), clipped into the
    /// pointing cone.
    pub fn initial_thrust_dir(&self) -> Option<Vec3> {
        if !self.constraints.constrain_initial_attitude {
            return None;
        }
        let d = self.att0.unwrap_or_else(|| -(self.v0 - self.env.wind_vec()));
        Some(clip_to_cone(&d, self.rocket.theta_max()))
    }

    /// Same scenario from another state (re-planning, ignition search).
    pub fn from_state(&self, r: Vec3, v: Vec3, m: f64, att: Option<Vec3>) -> Self {
        Self { r0: r, v0: v, m0: m, att0: att, ..self.clone() }
    }
}

pub const E1: Vec3 = Vec3::new(1.0, 0.0, 0.0);

/// Unit vector closest to `d` inside the cone of half-angle `theta` about up.
pub fn clip_to_cone(d: &Vec3, theta: f64) -> Vec3 {
    let n = d.norm();
    let d = if n < 1e-9 { E1 } else { d / n };
    if d[0] >= theta.cos() {
        return d;
    }
    let mut hor = d;
    hor[0] = 0.0;
    let hn = hor.norm();
    let hor = if hn > 1e-12 { hor / hn } else { Vec3::new(0.0, 1.0, 0.0) };
    E1 * theta.cos() + hor * theta.sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn missions() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../missions")
    }

    #[test]
    fn loads_both_missions() {
        let sc = Scenario::from_json(missions().join("coast_landing/scenario.json")).unwrap();
        assert_eq!(sc.rocket.m_dry, 22000.0);
        assert!(sc.coast.is_some() && sc.constraints.constrain_initial_attitude);
        assert_eq!(sc.mpc.n, 30);
        assert!(sc.sim_cases.iter().any(|c| matches!(c.truth.wind, Some(WindSpec::Profile(_)))));
        assert_eq!(sc.rocket.attitude.arm, 14.0);
        let sp = Scenario::from_json(missions().join("spec_test_case/scenario.json")).unwrap();
        assert!(sp.coast.is_none() && sp.initial_thrust_dir().is_none());
        assert_eq!(sp.scvx.eps_x, 1e-4);
        assert_eq!(sp.tracker.zeta, 0.8);
        assert_eq!(sp.tracker.ref_omega, 1.5);
        assert!(matches!(sp.sim_cases[4].truth.wind, Some(WindSpec::Constant(w)) if w[2] == 8.0));
    }
}
