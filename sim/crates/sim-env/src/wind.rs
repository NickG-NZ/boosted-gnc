use serde::Deserialize;
use sim_core::{Vec3, math::interp};

pub trait Wind: Send + Sync {
    /// Wind velocity (m/s) at position `r` and time `t`.
    fn velocity(&self, r: &Vec3, t: f64) -> Vec3;
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConstantWind {
    pub velocity: [f64; 3],
}

impl Wind for ConstantWind {
    fn velocity(&self, _r: &Vec3, _t: f64) -> Vec3 {
        Vec3::from(self.velocity)
    }
}

/// Altitude-only wind (frozen turbulence): power-law shear on a mean wind plus a tabulated gust
/// profile, linearly interpolated in altitude.
///
/// ```text
/// w(h) = mean * (max(h, h_floor) / ref_alt)^shear + gust(clamp(h, alt[0], alt[-1]))
/// ```
///
/// The gust table comes from `gust_profile` (seeded), so a run is exactly repeatable.
#[derive(Debug, Clone, Deserialize)]
pub struct ProfileWind {
    pub mean: [f64; 3],
    #[serde(default = "default_ref_alt")]
    pub ref_alt: f64,
    #[serde(default)]
    pub shear: f64,
    #[serde(default = "default_h_floor")]
    pub h_floor: f64,
    #[serde(default)]
    pub gust: Option<GustTable>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GustTable {
    /// Altitude grid (m), increasing.
    pub alt: Vec<f64>,
    /// Gust velocity per grid point (m/s), one `[up, x, y]` row per altitude.
    pub vel: Vec<[f64; 3]>,
    #[serde(skip)]
    cols: Option<[Vec<f64>; 3]>,
}

impl GustTable {
    pub fn new(alt: Vec<f64>, vel: Vec<[f64; 3]>) -> Self {
        let mut t = Self { alt, vel, cols: None };
        t.prepare();
        t
    }

    fn prepare(&mut self) {
        self.cols = Some(std::array::from_fn(|i| self.vel.iter().map(|v| v[i]).collect()));
    }

    fn at(&self, h: f64) -> Vec3 {
        let cols = self.cols.as_ref().expect("GustTable::prepare not called");
        Vec3::from_fn(|i, _| interp(h, &self.alt, &cols[i]))
    }
}

/// Top of the generated gust profile (held constant above), m.
pub const GUST_H_MAX: f64 = 12_000.0;
/// Vertical gusts are this fraction of the horizontal standard deviation.
pub const VERTICAL_GUST_FRAC: f64 = 0.5;

/// Frozen-turbulence gust profile: a first-order Gauss-Markov process in altitude with
/// correlation length `length` (grid spacing length / 10), standard deviation `sigma`
/// horizontally and `VERTICAL_GUST_FRAC * sigma` vertically. Deterministic per `seed`.
pub fn gust_profile(sigma: f64, length: f64, seed: u64) -> GustTable {
    let dh = length / 10.0;
    let n = (GUST_H_MAX / dh).ceil() as usize + 1;
    let alt: Vec<f64> = (0..n).map(|k| (k as f64 * dh).min(GUST_H_MAX)).collect();
    let sig = [VERTICAL_GUST_FRAC * sigma, sigma, sigma];
    let a = (-dh / length).exp();
    let mut rng = Normal::new(seed);
    let mut vel = Vec::with_capacity(n);
    let mut g: [f64; 3] = std::array::from_fn(|i| sig[i] * rng.sample());
    vel.push(g);
    for _ in 1..n {
        g = std::array::from_fn(|i| a * g[i] + sig[i] * (1.0 - a * a).sqrt() * rng.sample());
        vel.push(g);
    }
    GustTable::new(alt, vel)
}

/// Standard normal samples: SplitMix64 + Box-Muller. Self-contained so a seed gives the same
/// profile forever (no dependence on an external RNG crate's versioned algorithms).
struct Normal {
    state: u64,
    spare: Option<f64>,
}

impl Normal {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0x9E37_79B9_7F4A_7C15, spare: None }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in (0, 1]
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }
    fn sample(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}

fn default_ref_alt() -> f64 {
    10.0
}
fn default_h_floor() -> f64 {
    1.0
}

impl ProfileWind {
    /// Must be called after deserialisation (builds the column-major gust table).
    pub fn prepared(mut self) -> Self {
        if let Some(g) = self.gust.as_mut() {
            g.prepare();
        }
        self
    }
}

impl Wind for ProfileWind {
    fn velocity(&self, r: &Vec3, _t: f64) -> Vec3 {
        let h = r[0];
        let mut w = Vec3::from(self.mean);
        if self.shear != 0.0 {
            w *= (h.max(self.h_floor) / self.ref_alt).powf(self.shear);
        }
        if let Some(g) = &self.gust {
            w += g.at(h);
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shear_power_law() {
        let w = ProfileWind { mean: [0.0, 0.0, 8.0], ref_alt: 10.0, shear: 0.14, h_floor: 1.0, gust: None };
        assert_relative_eq!(w.velocity(&Vec3::new(10.0, 0.0, 0.0), 0.0)[2], 8.0);
        assert_relative_eq!(w.velocity(&Vec3::new(1000.0, 0.0, 0.0), 0.0)[2], 8.0 * 100f64.powf(0.14));
    }

    #[test]
    fn gusts_are_seeded_continuous_and_sized() {
        let (a, b, other) = (gust_profile(2.0, 200.0, 3), gust_profile(2.0, 200.0, 3), gust_profile(2.0, 200.0, 4));
        assert_eq!(a.vel, b.vel);
        assert_ne!(a.vel, other.vel);
        assert_eq!(*a.alt.last().unwrap(), GUST_H_MAX);
        let w = ProfileWind { mean: [0.0; 3], ref_alt: 10.0, shear: 0.0, h_floor: 1.0, gust: Some(a) };
        let hs: Vec<f64> = (0..=4000).map(|i| i as f64 * 2.5).collect();
        let g: Vec<Vec3> = hs.iter().map(|&h| w.velocity(&Vec3::new(h, 0.0, 0.0), 0.0)).collect();
        let max_jump = g.windows(2).map(|p| (p[1] - p[0]).abs().max()).fold(0.0, f64::max);
        assert!(max_jump < 1.0, "continuous: {max_jump}");
        let std = |i: usize| {
            let m = g.iter().map(|v| v[i]).sum::<f64>() / g.len() as f64;
            (g.iter().map(|v| (v[i] - m).powi(2)).sum::<f64>() / g.len() as f64).sqrt()
        };
        assert!(std(1) > 1.0 && std(1) < 3.0 && std(2) > 1.0 && std(2) < 3.0, "{} {}", std(1), std(2));
        assert!(std(0) < std(1));
    }

    #[test]
    fn gust_table_interpolates() {
        let g = GustTable::new(vec![0.0, 10.0], vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
        let w = ProfileWind { mean: [0.0; 3], ref_alt: 10.0, shear: 0.0, h_floor: 1.0, gust: Some(g) };
        assert_relative_eq!(w.velocity(&Vec3::new(5.0, 0.0, 0.0), 0.0), Vec3::new(0.5, 1.0, 1.5));
        assert_relative_eq!(w.velocity(&Vec3::new(50.0, 0.0, 0.0), 0.0), Vec3::new(1.0, 2.0, 3.0));
    }
}
