use sim_core::{Model, Vec3};
use sim_env::Environment;

use crate::{aero::Aero, engine::Engine};

/// 3-DoF point-mass translational dynamics with variable mass.
/// State `[r(3), v(3), m]`, input: thrust vector (N), held over the step.
pub struct PointMass<'a> {
    pub env: &'a Environment,
    pub aero: &'a Aero,
    pub engine: &'a Engine,
    pub t: f64,
}

impl Model for PointMass<'_> {
    type State = [f64; 7];
    type Input = Vec3;

    fn derivative(&self, x: &[f64; 7], thrust: &Vec3) -> [f64; 7] {
        let r = Vec3::new(x[0], x[1], x[2]);
        let v = Vec3::new(x[3], x[4], x[5]);
        let m = x[6];
        let tn = thrust.norm();
        let v_rel = v - self.env.wind.velocity(&r, self.t);
        let drag = self.aero.drag(self.env.atmosphere.density(r[0]), &v_rel, tn > 0.0);
        let a = thrust / m + self.env.gravity.accel(&r) + drag / m;
        [v[0], v[1], v[2], a[0], a[1], a[2], -self.engine.mdot_per_n * tn]
    }
}
