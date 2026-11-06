use serde::Deserialize;

/// Throttleable engine: thrust magnitude limited to `[thr_min, 1] * t_max` while lit, no thrust
/// once the propellant is gone. Mass flow `T / (isp * g0)`.
#[derive(Debug, Clone, Deserialize)]
pub struct EngineParams {
    pub t_max: f64,
    pub thr_min: f64,
    pub isp: f64,
}

#[derive(Debug, Clone)]
pub struct Engine {
    pub t_min: f64,
    pub t_max: f64,
    /// Mass flow per unit thrust (kg/s/N), includes any Isp dispersion.
    pub mdot_per_n: f64,
}

impl Engine {
    pub fn new(p: &EngineParams, isp_scale: f64, g0: f64) -> Self {
        Self { t_min: p.thr_min * p.t_max, t_max: p.t_max, mdot_per_n: 1.0 / (p.isp * isp_scale * g0) }
    }

    /// Delivered thrust (N) for a commanded magnitude. `cmd <= 0` means engine off.
    pub fn thrust(&self, cmd: f64, m: f64, m_dry: f64) -> f64 {
        if cmd <= 0.0 || m <= m_dry { 0.0 } else { cmd.clamp(self.t_min, self.t_max) }
    }
}
