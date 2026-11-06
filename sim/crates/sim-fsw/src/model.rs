//! The flight software's nominal point-mass model: what guidance and control *believe* the
//! dynamics are (nominal drag coefficient and Isp, the scenario's constant wind). The truth model
//! the vehicle actually flies is in sim-vehicle.

use sim_core::Vec3;
use sim_scenario::{Environment, Rocket};

use crate::trajectory::Trajectory;

#[derive(Debug, Clone)]
pub struct NominalModel {
    pub rocket: Rocket,
    pub env: Environment,
    pub drag: bool,
}

impl NominalModel {
    pub fn new(rocket: &Rocket, env: &Environment, drag: bool) -> Self {
        Self { rocket: rocket.clone(), env: env.clone(), drag }
    }

    pub fn alpha(&self) -> f64 {
        self.rocket.alpha(self.env.g0)
    }

    pub fn drag_force(&self, r: &Vec3, v: &Vec3, thrusting: bool) -> Vec3 {
        if !self.drag {
            return Vec3::zeros();
        }
        let vr = v - self.env.wind_vec();
        let cd = self.rocket.c_d * if thrusting { self.rocket.kappa } else { 1.0 };
        let rho = self.env.rho_atm(r[0].max(0.0));
        vr * (-0.5 * rho * self.rocket.s_ref * cd * vr.norm())
    }

    /// s = [r(3), v(3), m]
    pub fn deriv(&self, s: &[f64; 7], thrust: &Vec3) -> [f64; 7] {
        let r = Vec3::new(s[0], s[1], s[2]);
        let v = Vec3::new(s[3], s[4], s[5]);
        let m = s[6];
        let tn = thrust.norm();
        let d = self.drag_force(&r, &v, tn > 0.0);
        let a = thrust / m + self.env.g() + d / m;
        [v[0], v[1], v[2], a[0], a[1], a[2], -self.alpha() * tn]
    }

    /// RK4 step with `thrust_of(state)` evaluated at each stage.
    pub fn step_with(&self, s: &[f64; 7], dt: f64, thrust_of: impl Fn(&[f64; 7]) -> Vec3) -> [f64; 7] {
        let f = |x: &[f64; 7]| self.deriv(x, &thrust_of(x));
        let ax = |x: &[f64; 7], a: f64, k: &[f64; 7]| -> [f64; 7] { std::array::from_fn(|i| x[i] + a * k[i]) };
        let k1 = f(s);
        let k2 = f(&ax(s, 0.5 * dt, &k1));
        let k3 = f(&ax(s, 0.5 * dt, &k2));
        let k4 = f(&ax(s, dt, &k3));
        std::array::from_fn(|i| s[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    }

    pub fn step(&self, s: &[f64; 7], thrust: &Vec3, dt: f64) -> [f64; 7] {
        self.step_with(s, dt, |_| *thrust)
    }

    /// Re-simulate a plan open loop, holding thrust *acceleration* u (T = u m). Final [r, v, m].
    pub fn open_loop_resim(&self, traj: &Trajectory, substeps: usize) -> [f64; 7] {
        let (r, v) = (traj.r[0], traj.v[0]);
        let mut s = [r[0], r[1], r[2], v[0], v[1], v[2], traj.m[0]];
        for k in 0..traj.len() - 1 {
            let h = (traj.t[k + 1] - traj.t[k]) / substeps as f64;
            for j in 0..substeps {
                let u = traj.u_at(traj.t[k] + (j as f64 + 0.5) * h);
                s = self.step_with(&s, h, |x| u * x[6]);
            }
        }
        s
    }
}
