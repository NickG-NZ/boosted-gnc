use serde::Deserialize;
use sim_core::Vec3;

pub trait Gravity: Send + Sync {
    /// Gravitational acceleration (m/s^2) at position `r`.
    fn accel(&self, r: &Vec3) -> Vec3;
    /// Standard gravity used for specific impulse (m/s^2).
    fn g0(&self) -> f64;
}

/// Flat-earth constant gravity pointing down (-index 0).
#[derive(Debug, Clone, Deserialize)]
pub struct UniformGravity {
    pub g0: f64,
}

impl Gravity for UniformGravity {
    fn accel(&self, _r: &Vec3) -> Vec3 {
        Vec3::new(-self.g0, 0.0, 0.0)
    }
    fn g0(&self) -> f64 {
        self.g0
    }
}
