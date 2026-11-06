use serde::Deserialize;
use sim_core::{
    Vec3,
    math::{perp, rotate},
};

#[derive(Debug, Clone, Deserialize)]
pub struct BodyParams {
    /// transverse moment of inertia about the CG (kg m^2), held constant
    pub inertia: f64,
    /// gimbal point to CG distance (m), held constant
    pub arm: f64,
}

/// Pitch/yaw rigid-body rotation driven by the gimbaled engine and other torque sources.
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub inertia: f64,
    pub arm: f64,
}

impl RigidBody {
    pub fn new(p: &BodyParams) -> Self {
        Self { inertia: p.inertia, arm: p.arm }
    }

    /// Angular acceleration from engine thrust `thrust` deflected by gimbal `g`:
    /// tau = (-arm b) x (T t_hat) = -arm T (b x g).
    pub fn gimbal_alpha(&self, thrust: f64, b: &Vec3, g: &Vec3) -> Vec3 {
        b.cross(g) * (-self.arm * thrust) / self.inertia
    }

    /// Semi-implicit step: w += alpha dt (kept perpendicular to b), then rotate b by w.
    /// Returns the new `(b, w)`.
    pub fn propagate(&self, b: &Vec3, w: &Vec3, alpha: &Vec3, dt: f64) -> (Vec3, Vec3) {
        let w = perp(&(w + alpha * dt), b);
        let b_new = rotate(b, &w, dt);
        (b_new, perp(&w, &b_new))
    }
}
