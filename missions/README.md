# Missions

One folder per mission, each with a `scenario.json` (initial state, environment, constraints, solver
settings, sim cases). Vehicles are shared from `vehicles/`; the Rust flight software and sim
engine (`sim/`) are shared by all missions.

| Mission | |
|---|---|
| `coast_landing/` | Engine-off fall from 4 km, then hoverslam (default for `run.py`) |
| `spec_test_case/` | Spec section 7 powered-descent test case (acceptance tests) |
