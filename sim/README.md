# sim: Rust simulation engine

The truth model of the closed-loop sim. Flight software (guidance and control, currently Python in
`gnc/`) drives it through actuator commands. The controller moves into Rust later. The
three.js viewer in `web/` stays as is and reads the outputs.

## Crates

| Crate | Role | Depends on |
|---|---|---|
| `sim-core` | Domain-free engine: `Vec3` helpers, `Model` trait, `Rk4`/`Euler` integrators, zero-crossing events | nalgebra |
| `sim-env` | Environment traits + impls: `Atmosphere` (exponential), `Gravity` (uniform), `Wind` (constant, shear + tabulated gusts) | core |
| `sim-vehicle` | Vehicle subsystems (`Engine`, `Tvc`, `GridFins`, `Aero`, `RigidBody`), `PointMass` dynamics, and the assembled `Plant` | core, env, telemetry |
| `sim-telemetry` | `Recorder` (named fixed-width channels) and artifact writers | — |
| `sim-py` | PyO3 bindings, module `boosted_sim`: conversion and error mapping only | vehicle, telemetry |

The dependency arrows only point down. Physics never depends on bindings or outputs. New front ends,
such as a Rust CLI runner, a batch or Monte-Carlo driver, or a live telemetry stream, sit beside `sim-py`.

## Plant step

Commands are held over a step (`Command { thrust, gimbal, fins }`):

1. The engine clips thrust to `[thr_min, 1] * T_max`, or 0 if commanded off or the propellant is gone. The TVC
   slews the gimbal toward its command, limited in angle and rate. Grid fins add bounded angular acceleration.
2. Attitude (pitch/yaw: body axis `b`, rate `w`) advances: semi-implicit Euler on `w`, then an exact rotation of `b`.
3. Thrust acts along the new nozzle direction. `[r, v, m]` advances with RK4: gravity, drag on the
   wind-relative velocity, variable mass.
4. Ground contact is found by zero crossing and the state interpolated to touchdown.

Telemetry row `k` is the state at the start of step `k` plus the thrust applied during it.
`finish()` appends the final state.

## Config

`PlantConfig` (JSON, `crates/sim-vehicle/src/config.rs`) is the contract with whatever builds a
scenario: `vehicle`, `environment` (tagged `type` per model), `dispersions` (truth vs. nominal:
`drag`, `cd_scale`, `isp_scale`), `initial`. Today `gnc/plant.py:plant_config` builds it from
the JSON scenarios. Gust profiles are passed as tables, so runs are repeatable and match the Python
`WindModel` exactly.

## Develop

```bash
cargo test                        # unit tests (all crates)
cargo clippy --all-targets
cargo fmt
uv sync                           # (from the repo root) rebuild + install the Python module
```

`.cargo/config.toml` sets macOS dynamic-lookup link flags so plain `cargo build` links the
extension crate as well.

## Next steps

- Move the controllers into Rust behind a `FlightSoftware` trait (`step(&Measurements) -> Command`),
  with a Rust sim loop and multi-rate scheduler. Python then only configures and analyses runs.
- Add sensors and navigation (IMU/GNSS/altimeter models, an estimator) between plant and flight software.
- Full 6-DoF: quaternion attitude, roll, CG/inertia varying with propellant, aero moments.
