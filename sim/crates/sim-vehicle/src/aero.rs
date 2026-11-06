use serde::Deserialize;
use sim_core::Vec3;

#[derive(Debug, Clone, Deserialize)]
pub struct AeroParams {
    pub s_ref: f64,
    pub c_d: f64,
    /// C_D multiplier while the engine is lit (plume shielding)
    #[serde(default = "one")]
    pub kappa: f64,
}

fn one() -> f64 {
    1.0
}

/// Quadratic drag on the wind-relative velocity.
#[derive(Debug, Clone)]
pub struct Aero {
    pub enabled: bool,
    pub s_ref: f64,
    /// C_D including dispersion
    pub c_d: f64,
    pub kappa: f64,
}

impl Aero {
    pub fn new(p: &AeroParams, enabled: bool, cd_scale: f64) -> Self {
        Self { enabled, s_ref: p.s_ref, c_d: p.c_d * cd_scale, kappa: p.kappa }
    }

    pub fn drag(&self, rho: f64, v_rel: &Vec3, thrusting: bool) -> Vec3 {
        if !self.enabled {
            return Vec3::zeros();
        }
        let cd = if thrusting { self.c_d * self.kappa } else { self.c_d };
        v_rel * (-0.5 * rho * self.s_ref * cd * v_rel.norm())
    }
}
