//! Flight software: everything the vehicle computes on board.
//!
//! - `guidance`: trajectory planning (G-FOLD, SCvx), re-planning and ignition timing.
//! - `control`: tracking controllers (cascade PD + attitude loop, tracking MPC).
//! - `model`: the flight software's own nominal dynamics model (never the truth model).
//!
//! The flight software sees the vehicle only through `NavState` (perfect navigation for now) and
//! acts only through `ActuatorCmd`.

// matrix assembly reads best with explicit indices
#![allow(clippy::needless_range_loop)]

pub mod control;
pub mod guidance;
pub mod model;
pub mod trajectory;

use sim_core::Vec3;

pub use trajectory::{Hold, Trajectory};

/// Navigation solution handed to the flight software each control step.
#[derive(Debug, Clone)]
pub struct NavState {
    pub t: f64,
    pub r: Vec3,
    pub v: Vec3,
    pub m: f64,
    /// body axis (unit), body rate, gimbal deflection (|g| = sin(angle))
    pub b: Vec3,
    pub w: Vec3,
    pub g: Vec3,
}

impl NavState {
    pub fn thrust_dir(&self) -> Vec3 {
        let s2 = self.g.dot(&self.g);
        let d = self.b * (1.0 - s2).max(0.0).sqrt() + self.g;
        d / d.norm()
    }
}

/// Actuator commands, held over one control step.
#[derive(Debug, Clone, Default)]
pub struct ActuatorCmd {
    /// thrust magnitude (N); 0 = engine off
    pub thrust: f64,
    /// gimbal target (perpendicular to the body axis); None = stowed
    pub gimbal: Option<Vec3>,
    /// grid-fin angular acceleration request (rad/s^2); None = neutral
    pub fins: Option<Vec3>,
}
