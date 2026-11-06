//! Environment models. Each concern is a trait with interchangeable implementations, selected
//! from config (`EnvironmentConfig`), so new models (e.g. a standard atmosphere, a spherical
//! gravity field, a time-varying wind) slot in without touching the vehicle code.
//!
//! Frame: landing-site fixed, index 0 is "up", indices 1-2 horizontal.

pub mod atmosphere;
pub mod config;
pub mod gravity;
pub mod wind;

pub use atmosphere::{Atmosphere, ExponentialAtmosphere};
pub use config::{EnvironmentConfig, build};
pub use gravity::{Gravity, UniformGravity};
pub use wind::{ConstantWind, GustTable, ProfileWind, Wind, gust_profile};

/// The assembled environment the vehicle flies through.
pub struct Environment {
    pub atmosphere: Box<dyn Atmosphere>,
    pub gravity: Box<dyn Gravity>,
    pub wind: Box<dyn Wind>,
}

#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    #[error("invalid environment config: {0}")]
    Config(String),
}
