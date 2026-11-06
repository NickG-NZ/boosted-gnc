"""Truth plant: the Rust sim engine (`sim/`, Python module `boosted_sim`) that the flight software
(guidance + control, in `sim.py`) flies.

The plant owns the true state and the physics (environment, engine, TVC, grid fins, aero, rigid
body). Each control step the flight software sends actuator commands, held over the step:

    info = plant.step(thrust, dt, gimbal=g_cmd)     # engine lit, TVC steering
    info = plant.step(0.0, dt, fins=alpha_cmd)      # engine off, grid fins steering

`plant_config` translates a scenario + truth case into the Rust `PlantConfig` JSON
(sim/crates/sim-vehicle/src/config.rs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boosted_sim
import numpy as np

from gnc.control.attitude import AttitudeParams, AttitudeState
from gnc.dynamics import TruthModel
from gnc.vehicle import Scenario
from gnc.wind import H_FLOOR


@dataclass(frozen=True)
class StepInfo:
    dt: float  # actual step length (shorter on ground contact)
    contact: bool  # touched down during the step (state interpolated to contact)
    thrust: float  # delivered thrust magnitude (N)
    thrust_vec: np.ndarray  # thrust vector applied during the step (N)
    g_cmd: np.ndarray  # gimbal command after angle limiting
    torque: np.ndarray  # applied torque (N m)
    alpha: np.ndarray  # applied angular acceleration (rad/s^2)


def plant_config(sc: Scenario, truth: TruthModel, b0: np.ndarray, record: bool = True) -> dict[str, Any]:
    rk, env = sc.rocket, sc.env
    ap = AttitudeParams.from_dict(rk.attitude)
    if truth.wind is None:
        wind = {"type": "constant", "velocity": env.wind_vec.tolist()}
    else:
        w = truth.wind
        wind = {"type": "profile", "mean": list(w.mean), "ref_alt": w.ref_alt, "shear": w.shear,
                "h_floor": H_FLOOR}
        table = w.gust_table()
        if table is not None:
            wind["gust"] = {"alt": table[0].tolist(), "vel": table[1].tolist()}
    return {
        "vehicle": {
            "m_dry": rk.m_dry,
            "engine": {"t_max": rk.T_max, "thr_min": rk.thr_min, "isp": rk.isp},
            "aero": {"s_ref": rk.S_ref, "c_d": rk.C_D, "kappa": rk.kappa},
            "tvc": {"gimbal_max_deg": ap.gimbal_max_deg, "gimbal_rate_deg_s": ap.gimbal_rate_deg_s},
            "body": {"inertia": ap.inertia, "arm": ap.arm},
            "fins": {"alpha_max_deg_s2": ap.fin_alpha_max_deg_s2},
        },
        "environment": {
            "atmosphere": {"type": "exponential", "rho0": env.rho0, "scale_height": env.H},
            "gravity": {"type": "uniform", "g0": env.g0},
            "wind": wind,
        },
        "dispersions": {"drag": truth.drag, "cd_scale": truth.cd_scale, "isp_scale": truth.isp_scale},
        "initial": {"r": sc.r0.tolist(), "v": sc.v0.tolist(), "m": sc.m0, "b": np.asarray(b0, float).tolist()},
        "record": record,
    }


class Plant:
    """Python view of `boosted_sim.Plant`: numpy in/out, typed step results."""

    def __init__(self, config: dict[str, Any]):
        self._p = boosted_sim.Plant(json.dumps(config))

    t = property(lambda self: self._p.t)
    r = property(lambda self: np.array(self._p.r))
    v = property(lambda self: np.array(self._p.v))
    m = property(lambda self: self._p.m)

    @property
    def att(self) -> AttitudeState:
        return AttitudeState(b=np.array(self._p.b), w=np.array(self._p.w), g=np.array(self._p.g))

    def step(self, thrust: float, dt: float, gimbal: np.ndarray | None = None,
             fins: np.ndarray | None = None) -> StepInfo:
        d = self._p.step(float(thrust), float(dt),
                         None if gimbal is None else [float(x) for x in gimbal],
                         None if fins is None else [float(x) for x in fins])
        return StepInfo(dt=d["dt"], contact=d["contact"], thrust=d["thrust"],
                        thrust_vec=np.array(d["thrust_vec"]), g_cmd=np.array(d["g_cmd"]),
                        torque=np.array(d["torque"]), alpha=np.array(d["alpha"]))

    def wind_at(self, r: np.ndarray) -> np.ndarray:
        return np.array(self._p.wind_at([float(x) for x in r]))

    def finish(self) -> None:
        """Record the final telemetry row (after the last step)."""
        self._p.finish()

    def telemetry(self) -> dict[str, np.ndarray]:
        """Channels (rows = samples): t, r, v, m, b, w, g, thrust, wind. Row k is the state at the
        start of step k plus the thrust applied during it; the last row is the final state."""
        return {k: np.asarray(data, float).reshape(-1, w) if w > 1 else np.asarray(data, float)
                for k, (w, _unit, data) in self._p.telemetry().items()}

    def write_telemetry(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        """Write the run's telemetry artifact (JSON, see sim-telemetry/src/artifact.rs)."""
        self._p.write_telemetry(str(path), json.dumps(meta or {}, default=float))


def make_plant(sc: Scenario, truth: TruthModel, b0: np.ndarray) -> Plant:
    return Plant(plant_config(sc, truth, b0))
