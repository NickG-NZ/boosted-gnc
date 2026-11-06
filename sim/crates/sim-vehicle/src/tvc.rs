use serde::Deserialize;
use sim_core::{
    Vec3,
    math::{clip_norm, perp},
};

#[derive(Debug, Clone, Deserialize)]
pub struct TvcParams {
    pub gimbal_max_deg: f64,
    pub gimbal_rate_deg_s: f64,
}

/// Engine gimbal actuator: deflection limited in angle and slew rate.
#[derive(Debug, Clone)]
pub struct Tvc {
    /// max |g| (= sin of the max deflection)
    pub g_max: f64,
    /// max d|g|/dt (rad/s, small-angle)
    pub rate_max: f64,
}

impl Tvc {
    pub fn new(p: &TvcParams) -> Self {
        Self { g_max: p.gimbal_max_deg.to_radians().sin(), rate_max: p.gimbal_rate_deg_s.to_radians() }
    }

    /// Slew from `g` toward `cmd` over `dt`. Returns `(limited command, new deflection)`.
    pub fn slew(&self, g: &Vec3, cmd: &Vec3, b: &Vec3, dt: f64) -> (Vec3, Vec3) {
        let cmd = clip_norm(&perp(cmd, b), self.g_max);
        let g = perp(g, b);
        let g = g + clip_norm(&(cmd - g), self.rate_max * dt);
        (cmd, clip_norm(&g, self.g_max))
    }
}
