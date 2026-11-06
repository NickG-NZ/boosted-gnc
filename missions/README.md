# Missions

One folder per scenario family. A mission holds its vehicle and scenario configs (and later its own
run script and sim-loop wiring); the GNC algorithms (`landing/`) and the Rust sim engine (`sim/`) are
shared.

| Mission | Contents |
|---|---|
| `powered_landing/` | Booster engine-off fall and hoverslam landing: `configs/rockets/`, `configs/scenarios/` |
