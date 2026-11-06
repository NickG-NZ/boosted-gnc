//! PD translational tracker (~100 Hz) feeding the pointing reference and attitude loop:
//!
//!   a_cmd = u_ref + Kr (r_ref - r) + Kv (v_ref - v),   T_cmd = m a_cmd
//!
//! projected onto the physical thrust set (magnitude in [T_min, T_max], inside the pointing
//! cone). The vertical channel acts through throttle (fast), the lateral one through body
//! rotation (slow), so they get separate gains.

use sim_core::Vec3;
use sim_scenario::{Scenario, TrackerCfg};

use super::{attitude::AttitudeController, reference::ReferenceModel};
use crate::{ActuatorCmd, NavState, trajectory::Trajectory};

const E1: Vec3 = Vec3::new(1.0, 0.0, 0.0);

/// Euclidean projection of a thrust command onto the pointing cone, then magnitude clipping.
/// Downward commands (projection = apex) map to minimum vertical thrust.
pub fn project_thrust(t: &Vec3, t_min: f64, t_max: f64, theta_max: f64) -> Vec3 {
    let n = t.norm();
    if n < 1e-9 {
        return E1 * t_min;
    }
    let mut d = t / n;
    let mut mag = n;
    if d[0] < theta_max.cos() {
        let mut hor = d;
        hor[0] = 0.0;
        let hn = hor.norm();
        let hor = if hn > 1e-12 { hor / hn } else { Vec3::new(0.0, 1.0, 0.0) };
        d = E1 * theta_max.cos() + hor * theta_max.sin();
        mag = t.dot(&d);
        if mag <= 0.0 {
            return E1 * t_min;
        }
    }
    d * mag.clamp(t_min, t_max)
}

pub struct CascadeController {
    sc: Scenario,
    kr: Vec3,
    kv: Vec3,
    lat_fade: f64,
    ref_brake: f64,
    pub reference: ReferenceModel,
    pub attitude: AttitudeController,
}

impl CascadeController {
    pub fn new(sc: &Scenario, b0: &Vec3) -> Self {
        let c: &TrackerCfg = &sc.tracker;
        let (wl, zl) = (c.omega_n_lat.unwrap_or(c.omega_n), c.zeta_lat.unwrap_or(c.zeta));
        Self {
            sc: sc.clone(),
            kr: Vec3::new(c.omega_n.powi(2), wl * wl, wl * wl),
            kv: Vec3::new(2.0 * c.zeta * c.omega_n, 2.0 * zl * wl, 2.0 * zl * wl),
            lat_fade: c.lateral_fade_s,
            ref_brake: c.ref_accel_frac,
            reference: ReferenceModel::new(b0, c.ref_omega, c.cmd_rate_deg_s.to_radians()),
            attitude: AttitudeController::new(&sc.rocket.attitude),
        }
    }

    /// Tracking command for plan time `tau` over a step of `h` seconds.
    pub fn command(&mut self, plan: &Trajectory, tau: f64, t_go: f64, nav: &NavState, h: f64) -> ActuatorCmd {
        let rk = &self.sc.rocket;
        let (r_ref, v_ref) = plan.state_at(tau);
        let u_ref = plan.u_at(tau);
        // lateral corrections fade out over the last `lateral_fade_s`: near touchdown the vehicle
        // follows the (upright) plan instead of tilting to chase small position errors
        let f = if self.lat_fade > 0.0 { (t_go / self.lat_fade).min(1.0) } else { 1.0 };
        let fade = Vec3::new(1.0, f, f);
        let a_cmd = u_ref
            + fade.component_mul(&(self.kr.component_mul(&(r_ref - nav.r)) + self.kv.component_mul(&(v_ref - nav.v))));
        let t = project_thrust(&(a_cmd * nav.m), rk.t_min(), rk.t_max, rk.theta_max());
        // magnitude from the *current* nozzle direction: vertical priority, so a lagging body
        // tilt is compensated by throttle (clipped to engine limits)
        let td = nav.thrust_dir();
        let mut tn = t.norm();
        if td[0] > 0.2 {
            tn = (nav.m * a_cmd[0] / td[0]).clamp(rk.t_min(), rk.t_max);
        }
        if nav.m <= rk.m_dry {
            tn = 0.0;
        }
        // attitude: smooth the commanded direction, then the attitude loop commands the gimbal
        let a_max = self.ref_brake * self.attitude.alpha_avail(tn.max(rk.t_min()));
        self.reference.step(&(t / t.norm()), h, a_max);
        let (rb, rw, ra) = (self.reference.b, self.reference.w, self.reference.a);
        let (gimbal, fins) = self.attitude.command(&nav.b, &nav.w, &rb, tn, Some(&rw), Some(&ra));
        ActuatorCmd { thrust: tn, gimbal, fins }
    }

    /// Engine-off: grid fins steer the body toward `b_des`.
    pub fn fins_toward(&mut self, nav: &NavState, b_des: &Vec3) -> ActuatorCmd {
        let (_, fins) = self.attitude.command(&nav.b, &nav.w, b_des, 0.0, None, None);
        ActuatorCmd { thrust: 0.0, gimbal: None, fins }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downward_command_gives_min_thrust() {
        let t = project_thrust(&Vec3::new(-5.0, 1.0, 0.0), 1.0, 10.0, 20f64.to_radians());
        assert!((t - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
    }
}
