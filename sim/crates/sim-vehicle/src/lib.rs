//! Vehicle models for the powered-descent sim.
//!
//! Subsystems are small, separately testable structs (`engine`, `tvc`, `fins`, `aero`, `body`).
//! `Plant` composes them with an `Environment` into the truth model the flight software flies:
//! it takes actuator commands (`Command`), advances the true state by one step, and records
//! telemetry. Flight software (guidance, control, estimation) lives outside this crate.

pub mod aero;
pub mod body;
pub mod config;
pub mod engine;
pub mod fins;
pub mod plant;
pub mod state;
pub mod translational;
pub mod tvc;

pub use config::{PlantConfig, VehicleParams};
pub use plant::{Command, Plant, PlantError, StepInfo};
pub use state::VehicleState;
