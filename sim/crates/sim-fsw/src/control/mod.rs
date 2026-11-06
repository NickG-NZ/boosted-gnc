//! Control: tracking the guidance plan.
//!
//! - `attitude`: cascaded attitude loop (angle -> rate -> angular accel -> gimbal / fins).
//! - `reference`: smooth pointing reference with bounded rate and angular acceleration.
//! - `cascade`: PD translational tracker feeding the reference + attitude loop.
//! - `mpc`: attitude-aware tracking MPC commanding throttle and gimbal directly.

pub mod attitude;
pub mod cascade;
pub mod mpc;
pub mod reference;
