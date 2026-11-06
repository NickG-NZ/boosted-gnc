//! Mission runner: wires flight software (sim-fsw) to the truth plant (sim-vehicle).
//!
//! - `truth`: scenario + truth case -> plant configuration.
//! - `closed_loop`: one simulated flight (coast, re-planning, cascade or MPC control).
//! - `mission`: plan every guidance law a scenario needs, fly its cases, build the viewer data.

pub mod closed_loop;
pub mod mission;
pub mod truth;

pub use closed_loop::{SimResult, simulate};
pub use mission::{Progress, SolveOpts, solve};

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error(transparent)]
    Config(#[from] sim_scenario::ConfigError),
    #[error(transparent)]
    Plant(#[from] sim_vehicle::PlantError),
    #[error("{0}")]
    Guidance(String),
    #[error("{0}")]
    Control(String),
    #[error("cancelled")]
    Cancelled,
    #[error("cases not in the scenario: {0:?}")]
    UnknownCases(Vec<String>),
}
