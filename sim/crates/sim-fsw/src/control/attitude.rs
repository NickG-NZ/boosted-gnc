//! Pitch/yaw attitude controller for a gimbaled engine (thrust vector control).
//!
//! The engine sits `arm` below the CG, so a gimbal deflection g (perpendicular to the body axis
//! b, |g| = sin(delta)) produces the torque tau = -arm T (b x g): to swing the nose toward e the
//! nozzle pushes the tail away (non-minimum phase).
//!
//! Cascade: angle error -> desired body rate (capped, and never faster than can still be stopped
//! within the remaining angle) -> desired angular acceleration -> gimbal deflection. With the
//! engine off the request goes to the grid fins instead.

use sim_core::{
    Vec3,
    math::{clip_norm, perp},
};
use sim_scenario::AttitudeParams;

#[derive(Debug, Clone)]
pub struct AttitudeController {
    pub p: AttitudeParams,
    pub g_max: f64,
    pub w_max: f64,
    pub fin_alpha: f64,
    /// angular acceleration demanded by the last command (for probes)
    pub last_alpha_cmd: Vec3,
}

impl AttitudeController {
    pub fn new(p: &AttitudeParams) -> Self {
        Self {
            p: p.clone(),
            g_max: p.gimbal_max_deg.to_radians().sin(),
            w_max: p.body_rate_max_deg_s.to_radians(),
            fin_alpha: p.fin_alpha_max_deg_s2.to_radians(),
            last_alpha_cmd: Vec3::zeros(),
        }
    }

    /// Max angular acceleration (rad/s^2): gimbal torque with the engine on, else grid fins.
    pub fn alpha_avail(&self, thrust: f64) -> f64 {
        if thrust > 0.0 { self.p.arm * thrust * self.g_max / self.p.inertia } else { self.fin_alpha }
    }

    fn alpha_cmd(&self, b: &Vec3, w: &Vec3, b_des: &Vec3, w_ff: Option<&Vec3>, thrust: f64) -> Vec3 {
        let axis = b.cross(b_des);
        let (s, c) = (axis.norm(), b.dot(b_des).clamp(-1.0, 1.0));
        let ang = s.atan2(c);
        let a_brake = self.p.brake_frac * self.alpha_avail(thrust);
        let w_mag = (self.p.k_theta * ang).min((2.0 * a_brake * ang).sqrt());
        let mut w_des = if s < 1e-12 { Vec3::zeros() } else { axis / s * w_mag };
        if let Some(wf) = w_ff {
            w_des += perp(wf, b); // feedforward: rotation rate of the reference itself
        }
        let w_des = clip_norm(&w_des, self.w_max);
        (w_des - w) * self.p.k_omega
    }

    /// Commands steering the body axis toward `b_des`: `(Some(gimbal), None)` with the engine
    /// lit (`thrust` > 0, N), else `(None, Some(fin angular accel))`. The plant applies limits.
    pub fn command(
        &mut self,
        b: &Vec3,
        w: &Vec3,
        b_des: &Vec3,
        thrust: f64,
        w_ff: Option<&Vec3>,
        a_ff: Option<&Vec3>,
    ) -> (Option<Vec3>, Option<Vec3>) {
        let mut alpha_des = self.alpha_cmd(b, w, b_des, w_ff, thrust);
        if let Some(af) = a_ff {
            alpha_des += perp(af, b);
        }
        self.last_alpha_cmd = alpha_des;
        if thrust > 0.0 {
            // tau = I alpha = -arm T (b x g)  ->  b x g = -I alpha / (arm T)  ->  g = (b x g) x b
            let c = alpha_des * (-self.p.inertia) / (self.p.arm * thrust);
            return (Some(clip_norm(&perp(&c.cross(b), b), self.g_max)), None);
        }
        (None, Some(alpha_des))
    }
}
