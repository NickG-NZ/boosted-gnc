# boosted-gnc


3-DOF rocket landing simulator implementing the spec in `powered_descent_guidance_spec.md`:

- **Phase 1 — G-FOLD** (drag-free): lossless convexification, one SOCP per `t_f` + scan/golden-section search.
- **Phase 2 — SCvx** (with drag, free `t_f`): successive convexification, FOH discretization, trust region, virtual control.
- **Engine-off coast + ignition timing**: each guidance law predicts the fall with its own model and lights at the
  latest feasible time (minus a margin).
- **Closed-loop sim**: truth dynamics (drag, wind, C_D/Isp perturbations) + PD reference tracker, optional re-planning.
- **3D browser viewer** (three.js): replay, chase cam, plans vs flown paths, re-plans, telemetry, charts.

## Quick start

```bash
uv sync
uv run python run.py --serve        # open the viewer on the stored results (no solve) at http://127.0.0.1:8000
uv run python run.py                # solve all cases from the command line
uv run python run.py --only "SCvx drag · atmosphere · MPC"   # re-solve one case, merged into the results
uv run pytest                       # acceptance tests
```

In the viewer, **Run selected** re-plans and re-simulates the selected run and **Run all** does every case.
Both re-read the scenario config, run the solve in a separate process with live progress, save to
`web/data/results.json` and reload the page on the same run. Re-opening the app shows the last stored
results. Each run stores the plan it flew, so partial re-runs stay consistent. Solving needs `--serve`;
serving `web/` any other way gives a read-only viewer (ES modules don't load over `file://`).

Scenarios: `coast_landing.json` (default: engine-off fall from 4 km, then hoverslam) and `spec_test_case.json`
(the spec §7 case: engine already lit at 1.5 km, used by the acceptance tests). Pass a scenario path to `run.py` to use another one.
Click a run (or ↑/↓) to select it. **3D view** flies it; **Details** (key D) shows its configuration,
touchdown state, guidance plan, control effort, time histories and every metric. "Compare all" overlays the other runs.
Deep links: `index.html?case=6&t=9.5&chase=0&compare=1&paused=1&tab=details`.

## Layout

```
configs/rockets/*.json      vehicle params: mass, Isp, T_max, throttle limits, gimbal cone, S_ref, C_D, geometry
configs/scenarios/*.json    initial state, environment, glide slope, solver settings, list of sim cases
landing/
  vehicle.py     Rocket / Environment / Scenario dataclasses (loaded from JSON)
  trajectory.py  Trajectory container (ZOH or FOH controls) with continuous-time sampling
  gfold.py       Phase 1: convex SOCP per t_f, t_f search, tightness checks
  scvx.py        Phase 2: scaled model + Jacobians, FOH discretization, DPP subproblem, SCvx loop
  guidance.py    pluggable guidance objects (plan / plan_from / replan) for G-FOLD and SCvx
  ignition.py    coast prediction + latest-feasible ignition search
  dynamics.py    truth model (RK4), open-loop re-sim
  wind.py        truth wind: steady wind with altitude shear + seeded gusts
  sim.py         closed-loop simulator: tracker, thrust projection, optional re-planning
web/             three.js viewer: scene.js, rocket3d.js (model built from rocket geometry), charts.js, main.js
```

Sim cases are defined in the scenario JSON:

```json
{"name": "...", "plan": "gfold" | "scvx",
 "truth": {"drag": true, "cd_scale": 1.2, "wind": [0, 0, 8], "isp_scale": 0.99},
 "replan_hz": 2.0}
```

`truth.wind` is either a constant vector (m/s, frame `[up, x, y]`) or the wind model in `landing/wind.py`:

```json
"wind": {"mean": [0, 0, 8], "ref_alt": 10, "shear": 0.14, "gust_sigma": 1.5, "gust_length": 200, "seed": 1}
```

The steady part is `mean` at `ref_alt`, scaled with altitude by the power law `(h / ref_alt)^shear`. Gusts are a
seeded random profile over altitude (correlation length `gust_length`, std `gust_sigma` horizontally, half that
vertically), so the rocket meets them as it descends and each seed is exactly repeatable. Guidance never sees the
truth wind; it plans with `environment.wind`.

## Attitude constraints (`constraints` block of a scenario; all optional)

| key | effect |
|---|---|
| `constrain_initial_attitude` | first thrust direction = current body axis (falling engine-first at ignition, actual attitude on re-plans) |
| `terminal_cone_deg`, `terminal_frac` | final fraction of the burn stays within a narrow vertical cone, so the final descent and touchdown are vertical |
| `max_slew_deg_s` | planning rate limit on the thrust direction. G-FOLD: exact convex cone *ramps* away from the initial attitude and into the terminal cone. SCvx: linearized rate constraint about each iterate |

The rocket's `max_slew_deg_s` is the actuator limit in the simulator. The body axis is a logged state that slews at
most this fast, and the viewer draws it. Plan below it (8 vs 12 deg/s) so the tracker has margin.

## Results (`coast_landing.json`)

| Plan | ignites | at altitude | burn | fuel |
|---|---|---|---|---|
| G-FOLD (vacuum model) | T+7.4 s | 2265 m | 14.6 s | 3726 kg |
| SCvx (drag model) | T+12.6 s | 1301 m | 13.3 s | 2840 kg |

G-FOLD thinks the rocket keeps accelerating in free fall, so it lights early. SCvx knows drag holds the rocket near
terminal velocity, so it waits longer and saves ~25% fuel. Feasible ignition times form a *window*. Too late, and
max thrust can't stop in time. Too early, and min thrust (> weight, plus drag) stops the rocket high up, where it can't hover.

| Closed-loop case | result |
|---|---|
| G-FOLD · vacuum | landed 0.98 m/s, 0.5° tilt |
| G-FOLD · atmosphere, plan fixed in advance | crash: coast and burn both mispredicted |
| G-FOLD · atmosphere · re-plan 2 Hz | crash (24 m/s). The drag-unaware plans overshoot sideways, and the slew-limited vehicle can't recover |
| SCvx · atmosphere | landed 0.86 m/s, 0.5° tilt |
| SCvx · +20% C_D, 8 m/s wind, plan fixed in advance | crash: the extra coast drag puts it off-plan at ignition |
| SCvx · +20% C_D, 8 m/s wind · re-plan 2 Hz | landed 1.58 m/s, 0.5° tilt |

The spec §7 case (`spec_test_case.json`, no attitude constraints) reproduces the spec §8 references exactly:
G-FOLD gives t_f 12.19 s and m_f 23150 kg.

## Simplifications

- Point mass (3-DOF): attitude = thrust direction with a slew-rate limit; no rotational dynamics or torques.
- Perfect navigation; re-planning is instantaneous (no compute latency modeled).
- At end of plan the engine cuts. Residual sub-m/s touchdown speed comes from tracker lag.
