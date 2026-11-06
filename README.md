<p align="center">
  <img src="docs/hero.png" alt="boosted-gnc" width="100%">
</p>

<h1 align="center">boosted-gnc</h1>

<p align="center">
  Guidance, navigation and control simulation for rockets and aircraft.
</p>

---

## Overview

boosted-gnc is two things:

1. **A simulation environment.** A fast, modular flight simulator written in Rust, covering vehicle dynamics,
   environment models and telemetry.
2. **A library of modern flight control.** Minimal, readable implementations of state-of-the-art
   optimization-based guidance and control for aircraft and rocket landing.

The goal is to make advanced GNC methods easy to read, run and compare against a realistic model of the world.

## Highlights

- 🦀 Fast, modular simulation engine written in Rust, with Python bindings
- 🚀 High-fidelity 6-DOF vehicle dynamics with thrust vector control, grid fins and variable mass
- 🌬️ Realistic environments: atmosphere, drag, wind shear and repeatable gusts
- 📐 Convex optimization trajectory planning with G-FOLD and SCvx
- 🎯 Model predictive control for closed-loop tracking and in-flight re-planning
- 🧩 Minimal implementations: each algorithm lives in a single, self-contained module
- 🌐 3D browser visualization for replaying and comparing flights
- ⚙️ Vehicles and scenarios defined in plain JSON

## Quick start

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/) and a Rust toolchain with `cargo` on `PATH`. `uv sync`
builds the Rust guidance, control and sim in `sim/` (see [sim/README.md](sim/README.md)) and rebuilds it when its
sources change.

```bash
uv sync
uv run python run.py --serve     # open the viewer at http://127.0.0.1:8000
```

From the viewer, **Run selected** or **Run all** re-solves and re-simulates in the background and reloads with the
new results.

### Command line

```bash
uv run python run.py                          # solve and simulate every case
uv run python run.py --only "CASE NAME"       # re-run a single case
uv run python run.py missions/spec_test_case/scenario.json    # use a different mission
cd sim && cargo test                          # run the test suite (Rust)
uv run pytest                                 # Python bindings smoke test
```

Results are written to `web/data/results.json`, and each run's truth telemetry to
`artifacts/<mission>/<case>.telemetry.json` (`--artifacts ''` to skip).

## Configuration

Vehicles and scenarios are plain JSON.

| Path | Contents |
|---|---|
| `vehicles/` | Mass, Isp, max thrust, throttle limits, gimbal limits, reference area, C_D, geometry |
| `missions/<name>/scenario.json` | Initial state, environment, constraints, solver settings, and the list of sim cases |

Each sim case chooses a guidance law, a controller and the truth conditions it flies against:

```json
{
  "name": "SCvx · gusty wind · re-plan 2 Hz · MPC",
  "plan": "scvx",
  "controller": "mpc",
  "replan_hz": 2.0,
  "truth": {
    "drag": true,
    "cd_scale": 1.2,
    "isp_scale": 0.99,
    "wind": {"mean": [0, 0, 8], "shear": 0.14, "gust_sigma": 1.5, "gust_length": 200, "seed": 1}
  }
}
```

| Field | Options |
|---|---|
| `plan` | `gfold`, `scvx` |
| `controller` | `cascade`, `mpc` |
| `replan_hz` | Re-planning rate in flight (omit for a fixed plan) |
| `truth.wind` | Constant vector `[up, x, y]` in m/s, or a shear + gust model |

Guidance never sees the truth model; it plans with the scenario's nominal environment.

### Attitude constraints

Optional keys in a scenario's `constraints` block:

| Key | Effect |
|---|---|
| `constrain_initial_attitude` | Initial thrust direction matches the current body axis |
| `terminal_cone_deg`, `terminal_frac` | Keeps the final part of the burn near vertical |
| `max_slew_deg_s` | Rate limit on the planned thrust direction |

## License

Released under the [MIT License](LICENSE).
