use serde::Deserialize;

pub trait Atmosphere: Send + Sync {
    /// Air density (kg/m^3) at altitude `h` (m). Below ground is clamped to ground level.
    fn density(&self, h: f64) -> f64;
}

/// rho(h) = rho0 * exp(-h / H)
#[derive(Debug, Clone, Deserialize)]
pub struct ExponentialAtmosphere {
    pub rho0: f64,
    pub scale_height: f64,
}

impl Atmosphere for ExponentialAtmosphere {
    fn density(&self, h: f64) -> f64 {
        self.rho0 * (-h.max(0.0) / self.scale_height).exp()
    }
}
