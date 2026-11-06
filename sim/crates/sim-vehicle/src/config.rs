//! Plant configuration (serde). This is the contract between whatever builds a scenario (today
//! the Python `gnc.plant` module, later a Rust scenario loader) and the sim.

use serde::Deserialize;
use sim_env::EnvironmentConfig;

use crate::{aero::AeroParams, body::BodyParams, engine::EngineParams, fins::FinParams, tvc::TvcParams};

#[derive(Debug, Clone, Deserialize)]
pub struct PlantConfig {
    pub vehicle: VehicleParams,
    pub environment: EnvironmentConfig,
    #[serde(default)]
    pub dispersions: Dispersions,
    pub initial: InitialState,
    /// Record telemetry rows each step (default on).
    #[serde(default = "yes")]
    pub record: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VehicleParams {
    pub m_dry: f64,
    pub engine: EngineParams,
    pub aero: AeroParams,
    pub tvc: TvcParams,
    pub body: BodyParams,
    pub fins: FinParams,
}

/// Truth-model deviations from the nominal vehicle the flight software believes in.
#[derive(Debug, Clone, Deserialize)]
pub struct Dispersions {
    #[serde(default = "yes")]
    pub drag: bool,
    #[serde(default = "one")]
    pub cd_scale: f64,
    #[serde(default = "one")]
    pub isp_scale: f64,
}

impl Default for Dispersions {
    fn default() -> Self {
        Self { drag: true, cd_scale: 1.0, isp_scale: 1.0 }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct InitialState {
    #[serde(default)]
    pub t: f64,
    pub r: [f64; 3],
    pub v: [f64; 3],
    pub m: f64,
    /// body axis (normalised on load)
    pub b: [f64; 3],
    #[serde(default)]
    pub w: [f64; 3],
    #[serde(default)]
    pub g: [f64; 3],
}

fn yes() -> bool {
    true
}
fn one() -> f64 {
    1.0
}
