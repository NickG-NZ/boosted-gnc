use sim_core::Vec3;

/// True vehicle state. Attitude is pitch/yaw only: the body axis `b` (unit, nose up), its
/// angular velocity `w` (perpendicular to `b`) and the engine gimbal `g` (perpendicular to `b`,
/// `|g| = sin(deflection)`).
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleState {
    pub t: f64,
    pub r: Vec3,
    pub v: Vec3,
    pub m: f64,
    pub b: Vec3,
    pub w: Vec3,
    pub g: Vec3,
}

impl VehicleState {
    /// Direction of the thrust vector: body axis deflected by the gimbal.
    pub fn thrust_dir(&self) -> Vec3 {
        let s2 = self.g.dot(&self.g);
        let d = self.b * (1.0 - s2).max(0.0).sqrt() + self.g;
        d / d.norm()
    }

    pub fn gimbal_deg(&self) -> f64 {
        self.g.norm().min(1.0).asin().to_degrees()
    }

    /// Translational state `[r, v, m]` as the integrator vector.
    pub fn translational(&self) -> [f64; 7] {
        [self.r[0], self.r[1], self.r[2], self.v[0], self.v[1], self.v[2], self.m]
    }

    pub fn set_translational(&mut self, x: &[f64; 7]) {
        self.r = Vec3::new(x[0], x[1], x[2]);
        self.v = Vec3::new(x[3], x[4], x[5]);
        self.m = x[6];
    }
}
