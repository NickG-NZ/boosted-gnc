//! Scenario + truth case -> plant (truth model) configuration.

use sim_core::Vec3;
use sim_env::{
    EnvironmentConfig, ExponentialAtmosphere, ProfileWind, UniformGravity,
    config::{AtmosphereConfig, GravityConfig, WindConfig},
    gust_profile,
};
use sim_scenario::{Scenario, TruthCase, WindSpec};
use sim_vehicle::{
    PlantConfig, VehicleParams,
    aero::AeroParams,
    body::BodyParams,
    config::{Dispersions, InitialState},
    engine::EngineParams,
    fins::FinParams,
    tvc::TvcParams,
};

pub fn wind_config(sc: &Scenario, truth: &TruthCase) -> WindConfig {
    match &truth.wind {
        None => WindConfig::Constant(sim_env::ConstantWind { velocity: sc.env.wind }),
        Some(WindSpec::Constant(v)) => WindConfig::Constant(sim_env::ConstantWind { velocity: *v }),
        Some(WindSpec::Profile(p)) => WindConfig::Profile(ProfileWind {
            mean: p.mean,
            ref_alt: p.ref_alt,
            shear: p.shear,
            h_floor: 1.0,
            gust: (p.gust_sigma != 0.0).then(|| gust_profile(p.gust_sigma, p.gust_length, p.seed)),
        }),
    }
}

pub fn plant_config(sc: &Scenario, truth: &TruthCase, b0: &Vec3) -> PlantConfig {
    let (rk, env, ap) = (&sc.rocket, &sc.env, &sc.rocket.attitude);
    PlantConfig {
        vehicle: VehicleParams {
            m_dry: rk.m_dry,
            engine: EngineParams { t_max: rk.t_max, thr_min: rk.thr_min, isp: rk.isp },
            aero: AeroParams { s_ref: rk.s_ref, c_d: rk.c_d, kappa: rk.kappa },
            tvc: TvcParams { gimbal_max_deg: ap.gimbal_max_deg, gimbal_rate_deg_s: ap.gimbal_rate_deg_s },
            body: BodyParams { inertia: ap.inertia, arm: ap.arm },
            fins: FinParams { alpha_max_deg_s2: ap.fin_alpha_max_deg_s2 },
        },
        environment: EnvironmentConfig {
            atmosphere: AtmosphereConfig::Exponential(ExponentialAtmosphere { rho0: env.rho0, scale_height: env.h }),
            gravity: GravityConfig::Uniform(UniformGravity { g0: env.g0 }),
            wind: wind_config(sc, truth),
        },
        dispersions: Dispersions { drag: truth.drag, cd_scale: truth.cd_scale, isp_scale: truth.isp_scale },
        initial: InitialState {
            t: 0.0,
            r: sc.r0.into(),
            v: sc.v0.into(),
            m: sc.m0,
            b: (*b0).into(),
            w: [0.0; 3],
            g: [0.0; 3],
        },
        record: true,
    }
}
