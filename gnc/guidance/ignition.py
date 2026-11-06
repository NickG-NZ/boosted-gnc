"""Engine-off coast + ignition-time selection (spec §6: "light at latest feasible time").

The vehicle starts falling with the engine off. Using the guidance's *own* model (vacuum for
G-FOLD, drag for SCvx) we predict the ballistic coast and find the latest ignition time from which
a feasible landing plan exists, then back off by `margin_s` so the plan keeps a min-throttle arc
(throttle authority for the tracker).

Feasible ignition times form a *window*: too late and max thrust cannot stop in time; too early
(and min thrust > weight, plus drag) the vehicle stops high above the pad and cannot hover. So we
scan backward from the latest coast time on a coarse grid until a feasible point is found, then
bisect the late edge of the window.
"""

from __future__ import annotations

import numpy as np

from gnc.dynamics import TruthModel
from gnc.guidance.trajectory import Trajectory
from gnc.vehicle import Scenario


def coast_propagate(model: TruthModel, s0: np.ndarray, t_max: float, dt: float = 0.05,
                    h_floor: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    ts, S = [0.0], [s0.copy()]
    s = s0.copy()
    while ts[-1] < t_max and s[0] > h_floor:
        s = model.step(s, np.zeros(3), dt)
        ts.append(ts[-1] + dt)
        S.append(s.copy())
    return np.array(ts), np.array(S)


def plan_with_ignition(sc: Scenario, guidance, verbose: bool = False) -> Trajectory:
    cfg = {"max_coast_s": 60.0, "margin_s": 1.0, "resolution_s": 0.1, **sc.coast}
    model = TruthModel(sc.rocket, sc.env, drag=guidance.models_drag)
    s0 = np.concatenate([sc.r0, sc.v0, [sc.m0]])
    ts, S = coast_propagate(model, s0, cfg["max_coast_s"])

    def state(t: float) -> np.ndarray:
        return np.array([np.interp(t, ts, S[:, i]) for i in range(7)])

    cache: dict[float, Trajectory | None] = {}

    def try_plan(t: float) -> Trajectory | None:
        t = round(t, 3)
        if t not in cache:
            st = state(t)
            cache[t] = guidance.plan_from(st[0:3], st[3:6], st[6])
            if verbose:
                ok = "feasible" if cache[t] is not None else "infeasible"
                print(f"[ignition:{guidance.name}] t_ig={t:6.2f}s h={st[0]:7.1f}m |v|={np.linalg.norm(st[3:6]):6.1f} -> {ok}")
        return cache[t]

    grid = np.arange(float(ts[-1]), -1e-9, -cfg.get("scan_step_s", 1.0))
    lo = hi = None
    for i, tg in enumerate(grid):
        if try_plan(tg) is not None:
            lo, hi = tg, (grid[i - 1] if i > 0 else tg)
            break
    if lo is None:
        raise RuntimeError(f"{guidance.name}: no feasible ignition time during the coast")
    while hi - lo > cfg["resolution_s"]:
        mid = 0.5 * (lo + hi)
        if try_plan(mid) is not None:
            lo = mid
        else:
            hi = mid
    t_latest = lo
    t_ig = max(0.0, t_latest - cfg["margin_s"])
    plan = try_plan(t_ig)
    if plan is None:  # non-monotone corner case: fall back to the latest feasible time
        t_ig, plan = t_latest, try_plan(t_latest)
    assert plan is not None

    k = int(np.searchsorted(ts, t_ig))
    coast_t = np.r_[ts[:k], t_ig]
    coast_r = np.vstack([S[:k, 0:3], state(t_ig)[0:3]])
    plan.info.update(
        t_ignition=t_ig,
        t_latest_ignition=t_latest,
        ignition_solves=len(cache),
        coast={"t": coast_t.round(3).tolist(), "r": coast_r.round(3).tolist()},
    )
    return plan
