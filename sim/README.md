# sim: Rust flight software and simulation

Everything that computes runs here: guidance, control, the truth model, and the closed-loop
runner. Python (`run.py`) only starts runs, merges results and serves the three.js viewer in `web/`.

## Crates

| Crate | Role | Depends on |
|---|---|---|
| `sim-core` | Domain-free engine: `Vec3` helpers, `Model` trait, `Rk4`/`Euler` integrators, zero-crossing events | nalgebra |
| `sim-opt` | Conic problem builder on top of [Clarabel](https://clarabel.org): affine expressions, zero / nonnegative / second-order cones, quadratic objective, epigraph helpers (abs, norm, pos, Huber) | clarabel |
| `sim-scenario` | Mission config from `missions/*/scenario.json` + `vehicles/*.json`: vehicle, environment, constraints, solver settings, sim cases | core |
| `sim-env` | Environment traits + impls: `Atmosphere` (exponential), `Gravity` (uniform), `Wind` (constant, shear + seeded gusts) | core |
| `sim-vehicle` | Truth model: engine, TVC, grid fins, aero, rigid body, `PointMass` dynamics, assembled `Plant` | core, env, telemetry |
| `sim-fsw` | Flight software. **Guidance:** G-FOLD, SCvx, ignition timing, re-planning. **Control:** cascade PD + attitude loop, tracking MPC. Has its own nominal model and never sees the truth | core, opt, scenario |
| `sim-telemetry` | `Recorder` (named fixed-width channels) and telemetry artifact files | — |
| `sim-runner` | Wires flight software to the plant. `closed_loop` runs one flight, `mission` plans and flies a scenario's cases and builds the viewer data | all of the above |
| `sim-py` | PyO3 module `boosted_sim`: `solve(scenario, only, verbose, artifacts, progress) -> JSON` | runner |

Dependencies only point down. Flight software and truth model are independent: they meet only in
`sim-runner`, through `NavState` (perfect navigation for now) and `ActuatorCmd`.

## Optimisation

All three problems are second-order cone programs, solved with Clarabel (the solver CVXPY used
before the port), assembled with `sim-opt`:

- **G-FOLD**: lossless convexification (thrust slack `||u|| <= sigma`, log-mass), one SOCP per
  final time, with a coarse scan + golden-section search over t_f.
- **SCvx**: drag dynamics linearized and FOH-discretized by RK4 (multiple shooting), with virtual
  control, an L-inf trust region per node, and a ratio-test acceptance loop.
- **MPC**: shrinking-horizon tracking about the plan at 20 Hz. Linearized attitude-aware model,
  exact ZOH (nilpotent), soft pointing cone, Huber terminal costs.

Each problem is rebuilt per solve; that's cheap next to the factorization. Typical times (M-series
laptop, release): G-FOLD search ~0.4 s, SCvx plan with ignition search a few seconds, MPC solve ~6 ms.

## Plant step

Commands are held over a step (`Command { thrust, gimbal, fins }`):

1. The engine clips thrust to `[thr_min, 1] * T_max`, or 0 if commanded off or out of propellant.
   The TVC slews the gimbal toward its command (angle and rate limited). Grid fins add bounded
   angular acceleration.
2. Attitude (pitch/yaw: body axis `b`, rate `w`) advances: semi-implicit Euler on `w`, exact rotation of `b`.
3. Thrust acts along the new nozzle direction. `[r, v, m]` advances with RK4: gravity, drag on the
   wind-relative velocity, variable mass.
4. Ground contact is found by zero crossing, and the state is interpolated to touchdown.

Telemetry row `k` is the state at the start of step `k` plus the thrust applied during it.

## Develop

```bash
cargo test                         # all crates (closed-loop acceptance tests take ~40 s)
cargo test -p sim-fsw              # guidance / control only
cargo clippy --all-targets
cargo fmt
uv sync                            # (repo root) rebuild + install the Python module
```

`.cargo/config.toml` sets macOS dynamic-lookup link flags so plain `cargo build` links the
extension crate as well.

## Next steps

- A `FlightSoftware` trait with a multi-rate scheduler (guidance / control / attitude at their own rates).
- Sensors and navigation (IMU/GNSS/altimeter models, an estimator) between plant and flight software.
- Solver latency in the loop. Embedded code generation for the convex problems (e.g. CVXPYgen) if needed.
- Full 6-DoF: quaternion attitude, roll, CG/inertia varying with propellant, aero moments.
