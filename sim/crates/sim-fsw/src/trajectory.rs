//! Discrete guidance solution with continuous-time sampling.

use serde_json::{Map, Value, json};
use sim_core::{Vec3, math::interp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    /// zero-order hold on the control between nodes (G-FOLD)
    Zoh,
    /// first-order hold (SCvx)
    Foh,
}

/// Guidance output in SI units, one entry per node.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub t: Vec<f64>,
    pub r: Vec<Vec3>,
    pub v: Vec<Vec3>,
    pub m: Vec<f64>,
    /// thrust acceleration T/m (m/s^2)
    pub u: Vec<Vec3>,
    /// slack, ||u|| <= sigma
    pub sigma: Vec<f64>,
    pub hold: Hold,
    /// solver diagnostics (exported with the plan)
    pub info: Map<String, Value>,
}

/// Index k of the interval [t_k, t_k+1) containing `t` (numpy searchsorted(side="right") - 1),
/// clamped to the last interval.
pub fn interval(ts: &[f64], t: f64) -> usize {
    let k = ts.partition_point(|&x| x <= t).saturating_sub(1);
    k.min(ts.len() - 2)
}

pub fn interp_vec(t: f64, ts: &[f64], vs: &[Vec3]) -> Vec3 {
    let k = if t <= ts[0] {
        return vs[0];
    } else if t >= ts[ts.len() - 1] {
        return vs[vs.len() - 1];
    } else {
        ts.partition_point(|&x| x <= t) - 1
    };
    let s = (t - ts[k]) / (ts[k + 1] - ts[k]);
    Vec3::from_fn(|i, _| (vs[k + 1][i] - vs[k][i]) * s + vs[k][i])
}

impl Trajectory {
    pub fn len(&self) -> usize {
        self.t.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
    pub fn tf(&self) -> f64 {
        *self.t.last().expect("non-empty trajectory")
    }

    pub fn thrust(&self) -> Vec<Vec3> {
        self.u.iter().zip(&self.m).map(|(u, m)| u * *m).collect()
    }

    pub fn u_at(&self, t: f64) -> Vec3 {
        let t = t.clamp(self.t[0], self.tf());
        match self.hold {
            Hold::Zoh => self.u[interval(&self.t, t)],
            Hold::Foh => Vec3::from_fn(|i, _| {
                let col: Vec<f64> = self.u.iter().map(|u| u[i]).collect();
                interp(t, &self.t, &col)
            }),
        }
    }

    /// Reference position (cubic Hermite on r, v) and velocity (linear).
    pub fn state_at(&self, t: f64) -> (Vec3, Vec3) {
        let t = t.clamp(self.t[0], self.tf());
        let k = interval(&self.t, t);
        let h = self.t[k + 1] - self.t[k];
        let s = (t - self.t[k]) / h;
        let (s2, s3) = (s * s, s * s * s);
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        let r = self.r[k] * h00 + self.v[k] * (h10 * h) + self.r[k + 1] * h01 + self.v[k + 1] * (h11 * h);
        let v = self.v[k] * (1.0 - s) + self.v[k + 1] * s;
        (r, v)
    }

    /// Viewer / results export.
    pub fn to_json(&self, rho_max: f64) -> Value {
        let thrust = self.thrust();
        json!({
            "t": self.t,
            "r": self.r.iter().map(v3).collect::<Vec<_>>(),
            "v": self.v.iter().map(v3).collect::<Vec<_>>(),
            "m": self.m,
            "T": thrust.iter().map(v3).collect::<Vec<_>>(),
            "throttle": thrust.iter().map(|t| t.norm() / rho_max).collect::<Vec<_>>(),
            "hold": match self.hold { Hold::Zoh => "zoh", Hold::Foh => "foh" },
            "info": self.info,
        })
    }

    /// Remaining portion from plan time `tau`, re-timed to start at 0, resampled to `n` nodes
    /// (warm start for re-planning).
    pub fn shifted(&self, tau: f64, n: usize) -> Trajectory {
        let tf = self.tf();
        let t_new: Vec<f64> = (0..n).map(|i| tau + (tf - tau) * i as f64 / (n - 1) as f64).collect();
        let u: Vec<Vec3> = t_new.iter().map(|&t| self.u_at(t)).collect();
        Trajectory {
            t: t_new.iter().map(|t| t - tau).collect(),
            r: t_new.iter().map(|&t| interp_vec(t, &self.t, &self.r)).collect(),
            v: t_new.iter().map(|&t| interp_vec(t, &self.t, &self.v)).collect(),
            m: t_new.iter().map(|&t| interp(t, &self.t, &self.m)).collect(),
            sigma: u.iter().map(|u| u.norm()).collect(),
            u,
            hold: Hold::Foh,
            info: Map::new(),
        }
    }
}

pub fn v3(v: &Vec3) -> [f64; 3] {
    [v[0], v[1], v[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line() -> Trajectory {
        Trajectory {
            t: vec![0.0, 1.0, 2.0],
            r: vec![Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::zeros()],
            v: vec![Vec3::new(-1.0, 0.0, 0.0); 3],
            m: vec![3.0, 2.0, 1.0],
            u: vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0)],
            sigma: vec![1.0, 3.0, 5.0],
            hold: Hold::Zoh,
            info: Map::new(),
        }
    }

    #[test]
    fn sampling() {
        let mut tr = line();
        assert_eq!(tr.u_at(0.5)[0], 1.0);
        assert_eq!(tr.u_at(1.0)[0], 3.0);
        assert_eq!(tr.u_at(2.0)[0], 3.0); // last interval
        tr.hold = Hold::Foh;
        assert_eq!(tr.u_at(0.5)[0], 2.0);
        let (r, v) = tr.state_at(0.5);
        assert!((r[0] - 1.5).abs() < 1e-12 && (v[0] + 1.0).abs() < 1e-12);
        let s = tr.shifted(1.0, 3);
        assert_eq!(s.t, vec![0.0, 0.5, 1.0]);
        assert!((s.m[1] - 1.5).abs() < 1e-12);
    }
}
