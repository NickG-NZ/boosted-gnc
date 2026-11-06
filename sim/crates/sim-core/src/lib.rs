//! Domain-free simulation engine.
//!
//! Nothing in this crate knows about rockets: it provides the vector math, numerical
//! integrators, ODE model trait and event helpers that physics crates (`sim-env`,
//! `sim-vehicle`, ...) are built on.

pub mod event;
pub mod integrate;
pub mod math;
pub mod model;

pub use integrate::{Integrator, Rk4, StateVector};
pub use math::Vec3;
pub use model::Model;
