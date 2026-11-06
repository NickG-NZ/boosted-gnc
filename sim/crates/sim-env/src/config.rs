//! Serde config -> boxed environment models.

use serde::Deserialize;

use crate::{ConstantWind, EnvError, Environment, ExponentialAtmosphere, ProfileWind, UniformGravity};

#[derive(Debug, Clone, Deserialize)]
pub struct EnvironmentConfig {
    pub atmosphere: AtmosphereConfig,
    pub gravity: GravityConfig,
    #[serde(default)]
    pub wind: WindConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AtmosphereConfig {
    Exponential(ExponentialAtmosphere),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GravityConfig {
    Uniform(UniformGravity),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WindConfig {
    Constant(ConstantWind),
    Profile(ProfileWind),
}

impl Default for WindConfig {
    fn default() -> Self {
        WindConfig::Constant(ConstantWind { velocity: [0.0; 3] })
    }
}

pub fn build(cfg: EnvironmentConfig) -> Result<Environment, EnvError> {
    let atmosphere: Box<dyn crate::Atmosphere> = match cfg.atmosphere {
        AtmosphereConfig::Exponential(a) => {
            if a.scale_height <= 0.0 {
                return Err(EnvError::Config("atmosphere.scale_height must be > 0".into()));
            }
            Box::new(a)
        }
    };
    let gravity: Box<dyn crate::Gravity> = match cfg.gravity {
        GravityConfig::Uniform(g) => Box::new(g),
    };
    let wind: Box<dyn crate::Wind> = match cfg.wind {
        WindConfig::Constant(w) => Box::new(w),
        WindConfig::Profile(w) => {
            if let Some(g) = &w.gust
                && (g.alt.len() != g.vel.len() || g.alt.windows(2).any(|p| p[1] <= p[0]))
            {
                return Err(EnvError::Config("wind.gust: alt must be increasing and match vel".into()));
            }
            Box::new(w.prepared())
        }
    };
    Ok(Environment { atmosphere, gravity, wind })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_core::Vec3;

    #[test]
    fn parses_profile_wind() {
        let cfg: EnvironmentConfig = serde_json::from_str(
            r#"{"atmosphere": {"type": "exponential", "rho0": 1.225, "scale_height": 8500},
                "gravity": {"type": "uniform", "g0": 9.80665},
                "wind": {"type": "profile", "mean": [0, 0, 8], "shear": 0.14,
                         "gust": {"alt": [0, 100], "vel": [[0, 1, 0], [0, 3, 0]]}}}"#,
        )
        .unwrap();
        let env = build(cfg).unwrap();
        let w = env.wind.velocity(&Vec3::new(50.0, 0.0, 0.0), 0.0);
        assert!((w[1] - 2.0).abs() < 1e-12);
    }
}
