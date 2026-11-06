use serde::Deserialize;
use sim_core::{Vec3, math::clip_norm};

#[derive(Debug, Clone, Deserialize)]
pub struct FinParams {
    pub alpha_max_deg_s2: f64,
}

/// Grid fins, modelled as a bounded angular-acceleration source (engine-off attitude control).
#[derive(Debug, Clone)]
pub struct GridFins {
    pub alpha_max: f64,
}

impl GridFins {
    pub fn new(p: &FinParams) -> Self {
        Self { alpha_max: p.alpha_max_deg_s2.to_radians() }
    }

    pub fn alpha(&self, cmd: &Vec3) -> Vec3 {
        clip_norm(cmd, self.alpha_max)
    }
}
