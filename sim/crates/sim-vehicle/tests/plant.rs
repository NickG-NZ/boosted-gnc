//! Truth plant behaviour.

use sim_core::Vec3;
use sim_env::{
    EnvironmentConfig, ExponentialAtmosphere, UniformGravity,
    config::{AtmosphereConfig, GravityConfig, WindConfig},
};
use sim_vehicle::{
    Command, Plant, PlantConfig, VehicleParams,
    aero::AeroParams,
    body::BodyParams,
    config::{Dispersions, InitialState},
    engine::EngineParams,
    fins::FinParams,
    tvc::TvcParams,
};

const G0: f64 = 9.80665;
const T_MAX: f64 = 845_000.0;

fn config(drag: bool, isp_scale: f64) -> PlantConfig {
    PlantConfig {
        vehicle: VehicleParams {
            m_dry: 22_000.0,
            engine: EngineParams { t_max: T_MAX, thr_min: 0.4, isp: 282.0 },
            aero: AeroParams { s_ref: 10.5, c_d: 1.0, kappa: 1.0 },
            tvc: TvcParams { gimbal_max_deg: 8.0, gimbal_rate_deg_s: 30.0 },
            body: BodyParams { inertia: 4.0e6, arm: 14.0 },
            fins: FinParams { alpha_max_deg_s2: 15.0 },
        },
        environment: EnvironmentConfig {
            atmosphere: AtmosphereConfig::Exponential(ExponentialAtmosphere { rho0: 1.225, scale_height: 8500.0 }),
            gravity: GravityConfig::Uniform(UniformGravity { g0: G0 }),
            wind: WindConfig::default(),
        },
        dispersions: Dispersions { drag, cd_scale: 1.0, isp_scale },
        initial: InitialState {
            t: 0.0,
            r: [4000.0, 300.0, -160.0],
            v: [-200.0, -12.0, 6.0],
            m: 26_500.0,
            b: [1.0, 0.0, 0.0],
            w: [0.0; 3],
            g: [0.0; 3],
        },
        record: true,
    }
}

#[test]
fn vacuum_free_fall_is_exact_and_lands() {
    let mut p = Plant::new(config(false, 1.0)).unwrap();
    let (r0, v0, g) = (Vec3::new(4000.0, 300.0, -160.0), Vec3::new(-200.0, -12.0, 6.0), Vec3::new(-G0, 0.0, 0.0));
    let coast = Command { thrust: 0.0, gimbal: None, fins: Some(Vec3::zeros()) };
    loop {
        let info = p.step(&coast, 0.01).unwrap();
        let t = p.state().t;
        let exact = r0 + v0 * t + g * (0.5 * t * t);
        assert!((p.state().r - exact).norm() < 1e-6 * exact.norm().max(1.0), "t={t}");
        if info.contact {
            break;
        }
    }
    assert!(p.state().r[0].abs() < 1e-9, "interpolated to contact");
    assert_eq!(p.state().m, 26_500.0);
    assert!(p.step(&coast, 0.01).is_err(), "no steps on the ground");
    p.finish();
    let rec = p.recorder().unwrap();
    let rows = rec.get("t").unwrap().rows();
    assert!(rows > 100 && rec.channels().iter().all(|c| c.rows() == rows));
}

#[test]
fn mass_flow_follows_isp() {
    let mut p = Plant::new(config(false, 0.9)).unwrap();
    let t = 0.6 * T_MAX;
    for _ in 0..100 {
        p.step(&Command { thrust: t, gimbal: Some(Vec3::zeros()), fins: None }, 0.01).unwrap();
    }
    let expect = 26_500.0 - t * 1.0 / (282.0 * 0.9 * G0);
    assert!((p.state().m - expect).abs() < 1e-8 * expect);
}

#[test]
fn engine_and_gimbal_limits() {
    let mut p = Plant::new(config(false, 1.0)).unwrap();
    let info =
        p.step(&Command { thrust: 10.0 * T_MAX, gimbal: Some(Vec3::new(0.0, 1.0, 0.0)), fins: None }, 0.01).unwrap();
    assert_eq!(info.thrust, T_MAX);
    assert!((info.g_cmd.norm() - 8f64.to_radians().sin()).abs() < 1e-12);
    assert!(p.state().g.norm() <= 30f64.to_radians() * 0.01 + 1e-12, "slew-rate limited");
    let low = p.step(&Command { thrust: 0.1 * T_MAX, gimbal: Some(Vec3::zeros()), fins: None }, 0.01).unwrap();
    assert_eq!(low.thrust, 0.4 * T_MAX);
    assert!(p.step(&Command::default(), -0.01).is_err());
}
