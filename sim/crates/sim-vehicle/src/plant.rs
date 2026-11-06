use sim_core::{Integrator, Rk4, Vec3, event::zero_crossing};
use sim_env::{EnvError, Environment};
use sim_telemetry::{ChannelId, Recorder};

use crate::{
    aero::Aero, body::RigidBody, config::PlantConfig, engine::Engine, fins::GridFins, state::VehicleState,
    translational::PointMass, tvc::Tvc,
};

/// Actuator commands from flight software, held over one step.
#[derive(Debug, Clone, Default)]
pub struct Command {
    /// Commanded thrust magnitude (N); `<= 0` = engine off.
    pub thrust: f64,
    /// Gimbal deflection target (perpendicular to the body axis, |g| = sin(angle)).
    /// `None` = nozzle stowed at neutral.
    pub gimbal: Option<Vec3>,
    /// Grid-fin angular acceleration request (rad/s^2). `None` = fins neutral.
    pub fins: Option<Vec3>,
}

/// What happened during a step (for probes / controller bookkeeping).
#[derive(Debug, Clone)]
pub struct StepInfo {
    pub dt: f64,
    /// Ground contact during the step; the state was interpolated to the contact instant.
    pub contact: bool,
    pub thrust: f64,
    pub thrust_vec: Vec3,
    pub g_cmd: Vec3,
    pub torque: Vec3,
    pub alpha: Vec3,
}

#[derive(Debug, thiserror::Error)]
pub enum PlantError {
    #[error(transparent)]
    Env(#[from] EnvError),
    #[error("invalid plant config: {0}")]
    Config(String),
    #[error("invalid step: {0}")]
    Step(String),
}

struct Channels {
    t: ChannelId,
    r: ChannelId,
    v: ChannelId,
    m: ChannelId,
    b: ChannelId,
    w: ChannelId,
    g: ChannelId,
    thrust: ChannelId,
    wind: ChannelId,
}

/// The truth model: environment + vehicle subsystems + true state.
///
/// Step order (one control period, commands held):
///   1. TVC slews the gimbal toward its command; fins produce their (bounded) angular accel.
///   2. Attitude propagates (semi-implicit Euler on w, exact rotation of b).
///   3. Thrust acts along the *new* nozzle direction; translational state advances with RK4.
///   4. Ground contact is detected and the translational state interpolated to touchdown.
pub struct Plant {
    pub env: Environment,
    pub m_dry: f64,
    pub engine: Engine,
    pub tvc: Tvc,
    pub fins: GridFins,
    pub aero: Aero,
    pub body: RigidBody,
    state: VehicleState,
    landed: bool,
    last_thrust: Vec3,
    recorder: Option<(Recorder, Channels)>,
}

impl Plant {
    pub fn from_json(json: &str) -> Result<Self, PlantError> {
        let cfg: PlantConfig = serde_json::from_str(json).map_err(|e| PlantError::Config(e.to_string()))?;
        Self::new(cfg)
    }

    pub fn new(cfg: PlantConfig) -> Result<Self, PlantError> {
        let env = sim_env::build(cfg.environment)?;
        let vp = &cfg.vehicle;
        let d = &cfg.dispersions;
        let engine = Engine::new(&vp.engine, d.isp_scale, env.gravity.g0());
        let ic = &cfg.initial;
        let b = Vec3::from(ic.b);
        if b.norm() < 1e-9 {
            return Err(PlantError::Config("initial.b must be non-zero".into()));
        }
        let state = VehicleState {
            t: ic.t,
            r: Vec3::from(ic.r),
            v: Vec3::from(ic.v),
            m: ic.m,
            b: b / b.norm(),
            w: Vec3::from(ic.w),
            g: Vec3::from(ic.g),
        };
        let recorder = cfg.record.then(|| {
            let mut rec = Recorder::new();
            let ch = Channels {
                t: rec.channel("t", 1, "s"),
                r: rec.channel("r", 3, "m"),
                v: rec.channel("v", 3, "m/s"),
                m: rec.channel("m", 1, "kg"),
                b: rec.channel("b", 3, "-"),
                w: rec.channel("w", 3, "rad/s"),
                g: rec.channel("g", 3, "-"),
                thrust: rec.channel("thrust", 3, "N"),
                wind: rec.channel("wind", 3, "m/s"),
            };
            (rec, ch)
        });
        Ok(Self {
            m_dry: vp.m_dry,
            engine,
            tvc: Tvc::new(&vp.tvc),
            fins: GridFins::new(&vp.fins),
            aero: Aero::new(&vp.aero, d.drag, d.cd_scale),
            body: RigidBody::new(&vp.body),
            env,
            state,
            landed: false,
            last_thrust: Vec3::zeros(),
            recorder,
        })
    }

    pub fn state(&self) -> &VehicleState {
        &self.state
    }

    pub fn landed(&self) -> bool {
        self.landed
    }

    pub fn wind_at(&self, r: &Vec3) -> Vec3 {
        self.env.wind.velocity(r, self.state.t)
    }

    pub fn step(&mut self, cmd: &Command, dt: f64) -> Result<StepInfo, PlantError> {
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(PlantError::Step(format!("dt must be positive and finite, got {dt}")));
        }
        if self.landed {
            return Err(PlantError::Step("vehicle is on the ground".into()));
        }
        let s0 = self.state.clone();
        let b = s0.b;

        // 1. actuators
        let thrust = self.engine.thrust(cmd.thrust, s0.m, self.m_dry);
        let (g_cmd, g) = match &cmd.gimbal {
            Some(gc) => self.tvc.slew(&s0.g, gc, &b, dt),
            None => (Vec3::zeros(), Vec3::zeros()),
        };
        let mut alpha = if cmd.gimbal.is_some() { self.body.gimbal_alpha(thrust, &b, &g) } else { Vec3::zeros() };
        if let Some(fa) = &cmd.fins {
            alpha += self.fins.alpha(fa);
        }

        // 2. attitude
        let (b_new, w_new) = self.body.propagate(&b, &s0.w, &alpha, dt);
        let mut s1 = s0.clone();
        s1.b = b_new;
        s1.w = w_new;
        s1.g = sim_core::math::perp(&g, &b_new);

        // 3. translation along the new nozzle direction
        let thrust_vec = if thrust > 0.0 { s1.thrust_dir() * thrust } else { Vec3::zeros() };
        let x0 = s0.translational();
        let dyn_ = PointMass { env: &self.env, aero: &self.aero, engine: &self.engine, t: s0.t };
        let mut x1 = Rk4.step(&dyn_, &x0, &thrust_vec, dt);
        let mut h = dt;

        // 4. ground contact
        let contact = zero_crossing(x0[0], x1[0]);
        if let Some(frac) = contact {
            x1 = std::array::from_fn(|i| x0[i] + frac * (x1[i] - x0[i]));
            h = frac * dt;
            self.landed = true;
        }
        s1.set_translational(&x1);
        s1.t = s0.t + h;

        self.record(&s0, &thrust_vec);
        self.state = s1;
        self.last_thrust = thrust_vec;
        Ok(StepInfo {
            dt: h,
            contact: contact.is_some(),
            thrust,
            thrust_vec,
            g_cmd,
            torque: alpha * self.body.inertia,
            alpha,
        })
    }

    /// Record the current state as the final telemetry row (thrust held from the last step).
    pub fn finish(&mut self) {
        let s = self.state.clone();
        let tv = self.last_thrust;
        self.record(&s, &tv);
    }

    /// One telemetry row: state at the start of a step + the thrust applied during it.
    fn record(&mut self, s: &VehicleState, thrust_vec: &Vec3) {
        let wind = self.env.wind.velocity(&s.r, s.t);
        if let Some((rec, ch)) = self.recorder.as_mut() {
            let g = if thrust_vec.norm() > 0.0 { s.g } else { Vec3::zeros() };
            rec.push(ch.t, &[s.t]);
            rec.push(ch.r, s.r.as_slice());
            rec.push(ch.v, s.v.as_slice());
            rec.push(ch.m, &[s.m]);
            rec.push(ch.b, s.b.as_slice());
            rec.push(ch.w, s.w.as_slice());
            rec.push(ch.g, g.as_slice());
            rec.push(ch.thrust, thrust_vec.as_slice());
            rec.push(ch.wind, wind.as_slice());
        }
    }

    pub fn recorder(&self) -> Option<&Recorder> {
        self.recorder.as_ref().map(|(r, _)| r)
    }
}
