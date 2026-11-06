//! Second-order pointing reference: turns a (possibly jumpy) desired direction into a smooth one
//! with bounded rate and angular acceleration. Its rate and acceleration are fed forward to the
//! attitude loop, so the gimbal mostly delivers planned torque and feedback only trims.

use sim_core::{
    Vec3,
    math::{clip_norm, perp, rotate},
};

#[derive(Debug, Clone)]
pub struct ReferenceModel {
    pub b: Vec3,
    pub w: Vec3,
    pub a: Vec3,
    k1: f64,
    k2: f64,
    rate_max: f64,
}

impl ReferenceModel {
    pub fn new(b0: &Vec3, omega: f64, rate_max: f64) -> Self {
        Self { b: b0 / b0.norm(), w: Vec3::zeros(), a: Vec3::zeros(), k1: omega * omega, k2: 2.0 * omega, rate_max }
    }

    pub fn reset(&mut self, b: &Vec3, w: &Vec3) {
        self.b = b / b.norm();
        self.w = perp(w, &self.b);
        self.a = Vec3::zeros();
    }

    pub fn step(&mut self, b_des: &Vec3, dt: f64, a_max: f64) {
        let b = self.b;
        let axis = b.cross(b_des);
        let (s, c) = (axis.norm(), b.dot(b_des).clamp(-1.0, 1.0));
        let err = if s < 1e-12 { Vec3::zeros() } else { axis / s * s.atan2(c) };
        let a = clip_norm(&(err * self.k1 - self.w * self.k2), a_max);
        let w = clip_norm(&perp(&(self.w + a * dt), &b), self.rate_max);
        let a = (w - self.w) / dt; // acceleration actually applied after the rate limit
        let b_new = rotate(&b, &w, dt);
        self.b = b_new;
        self.w = perp(&w, &b_new);
        self.a = a;
    }
}
