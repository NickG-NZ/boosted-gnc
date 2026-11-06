"""Acceptance tests from the spec §8 (Phase 1 & 2) plus closed-loop sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from landing import gfold, scvx
from landing.dynamics import TruthModel, open_loop_resim
from landing.guidance import GUIDANCE, SCvxGuidance
from landing.ignition import plan_with_ignition
from landing.mpc import NU, NX, continuous_model, discretize
from landing.sim import project_thrust, simulate, truth_from_case
from landing.vehicle import Scenario

CFG = Path(__file__).resolve().parents[1] / "configs/scenarios/spec_test_case.json"


@pytest.fixture(scope="module")
def sc() -> Scenario:
    # spec numbers assume planning down to the engine floor (no throttle-down margin)
    return Scenario.from_json(CFG).with_rocket(thr_min_plan=None)


@pytest.fixture(scope="module")
def plan_g(sc):
    return gfold.solve(sc)


@pytest.fixture(scope="module")
def plan_s(sc, plan_g):
    return scvx.solve(sc, plan_g)


# ---------------------------------------------------------------- Phase 1


def test_gfold_bracket(sc):
    lo, hi = gfold.tf_bracket(sc)
    assert lo == pytest.approx(5.79, abs=0.01)
    assert hi == pytest.approx(32.73, abs=0.01)


def test_gfold_optimum(sc, plan_g):
    assert plan_g.tf == pytest.approx(12.19, abs=0.05)
    assert plan_g.m[-1] == pytest.approx(23150, abs=10)


def test_gfold_bang_bang(sc, plan_g):
    thr = np.linalg.norm(plan_g.T, axis=1)[:-1] / sc.rocket.T_max
    assert thr[0] == pytest.approx(sc.rocket.thr_min, abs=1e-3)  # starts at min
    assert thr[-1] == pytest.approx(sc.rocket.thr_max, abs=1e-3)  # ends at max
    # at most one interval in transition
    mid = (thr > sc.rocket.thr_min + 1e-3) & (thr < sc.rocket.thr_max - 1e-3)
    assert mid.sum() <= 1


def test_gfold_lossless(plan_g):
    assert plan_g.info["tightness_max"] < 1e-4
    assert plan_g.info["thrust_bound_violation_N"] == 0.0


def test_gfold_resim_lands(sc, plan_g):
    s = open_loop_resim(TruthModel(sc.rocket, sc.env, drag=False), plan_g)
    assert np.linalg.norm(s[0:3]) < 1e-2
    assert np.linalg.norm(s[3:6]) < 1e-2


# ---------------------------------------------------------------- Phase 2


def test_scvx_regression_no_drag(sc, plan_g):
    p0 = scvx.solve(sc.with_rocket(C_D=0.0), plan_g)
    assert p0.tf == pytest.approx(plan_g.tf, rel=2e-3)
    assert p0.m[-1] == pytest.approx(plan_g.m[-1], rel=1e-3)


def test_scvx_converges(plan_s):
    assert plan_s.info["converged"]
    assert plan_s.info["iterations"] < 15
    assert plan_s.info["history"][-1]["nu_l1"] < 1e-6


def test_scvx_resim_lands(sc, plan_s):
    s = open_loop_resim(TruthModel(sc.rocket, sc.env, drag=True), plan_s)
    assert np.linalg.norm(s[0:3]) < 0.1
    assert np.linalg.norm(s[3:6]) < 0.1


def test_drag_saves_fuel(sc, plan_g, plan_s):
    assert plan_s.m[-1] >= plan_g.m[-1]


def test_scvx_lossless(plan_s):
    assert plan_s.info["tightness_max"] < 1e-4
    assert plan_s.info["thrust_bound_violation_N"] < 1.0  # N, O(d^3) Taylor residual


# ---------------------------------------------------------------- closed loop


def test_projection_downward_command_gives_min_thrust():
    T = project_thrust(np.array([-5.0, 1.0, 0.0]), 1.0, 10.0, np.deg2rad(20))
    assert np.allclose(T, [1.0, 0, 0])


def test_closed_loop_nominal(sc, plan_s):
    res = simulate(sc, plan_s, truth_from_case(sc, {"drag": True}), controller="mpc")
    assert res.metrics["landed"]
    assert res.metrics["touchdown_pos_err_m"] < 0.5
    assert res.metrics["touchdown_vert_speed"] < 1.5


def test_closed_loop_replan_handles_perturbation(sc, plan_s):
    truth = truth_from_case(sc, {"drag": True, "cd_scale": 1.2, "wind": [0, 0, 8], "isp_scale": 0.99})
    res = simulate(sc, plan_s, truth, guidance=SCvxGuidance(sc), replan_hz=2.0, controller="mpc")
    assert res.metrics["n_replan_failures"] <= 2  # a rejected re-plan keeps the previous plan
    assert res.metrics["touchdown_vert_speed"] < 2.0
    assert res.metrics["touchdown_pos_err_m"] < 1.0


# ---------------------------------------------------------------- coast + ignition


def test_coast_ignition_scvx_lands():
    sc2 = Scenario.from_json(CFG.parent / "coast_landing.json")
    g = SCvxGuidance(sc2)
    plan = plan_with_ignition(sc2, g)
    assert 0 < plan.info["t_ignition"] < plan.info["t_latest_ignition"]
    res = simulate(sc2, plan, truth_from_case(sc2, {"drag": True}), controller="mpc")
    m = res.metrics
    assert m["landed"] and m["touchdown_vert_speed"] < 1.5 and m["touchdown_pos_err_m"] < 0.5
    # engine is off during the coast
    coast = res.t < plan.info["t_ignition"] - 1e-6
    assert coast.sum() > 100 and np.all(np.linalg.norm(res.T[coast], axis=1) == 0)


# ---------------------------------------------------------------- attitude penalties + MPC


@pytest.fixture(scope="module")
def coast_sc():
    return Scenario.from_json(CFG.parent / "coast_landing.json")


@pytest.fixture(scope="module")
def coast_plans(coast_sc):
    return {k: plan_with_ignition(coast_sc, GUIDANCE[k](coast_sc)) for k in ("gfold", "scvx")}


@pytest.mark.parametrize("guidance_name", ["gfold", "scvx"])
def test_plan_starts_on_attitude_and_ends_upright(coast_sc, coast_plans, guidance_name):
    plan = coast_plans[guidance_name]
    # first thrust direction within the initial-attitude tolerance of falling engine-first
    d0 = plan.u[0] / np.linalg.norm(plan.u[0])
    tol = coast_sc.constraints["initial_attitude_tol_deg"]
    assert np.degrees(np.arccos(d0 @ (-plan.v[0] / np.linalg.norm(plan.v[0])))) <= tol + 0.05
    # end-pose penalty: final thrust (near) vertical
    d_end = plan.u[-2] / np.linalg.norm(plan.u[-2])
    assert np.degrees(np.arccos(d_end[0])) < 1.0


def test_mpc_discretization_is_exact():
    A, B = continuous_model(4.0e5, 2.4e4, 8.45e5, np.array([0.1, -0.05]), 1.5)
    c = np.linspace(-1, 1, NX)
    dt = 0.7
    assert np.allclose(np.linalg.matrix_power(A, 5), 0)  # nilpotent
    Ad, Bd, cd = discretize(A, B, c, dt)
    M = np.zeros((NX + NU + 1, NX + NU + 1))
    M[:NX, :NX], M[:NX, NX:NX + NU], M[:NX, -1] = A, B, c
    E = expm(M * dt)
    assert np.allclose(Ad, E[:NX, :NX]) and np.allclose(Bd, E[:NX, NX:NX + NU]) and np.allclose(cd, E[:NX, -1])


@pytest.mark.parametrize("case", [
    ("scvx", {"drag": True}, 0.0),
    ("scvx", {"drag": True, "cd_scale": 1.2, "wind": [0, 0, 8], "isp_scale": 0.99}, 0.0),
    ("gfold", {"drag": True}, 2.0),
    ("scvx", {"drag": True, "cd_scale": 1.2, "isp_scale": 0.99,
              "wind": {"mean": [0, 0, 8], "shear": 0.14, "gust_sigma": 1.5, "seed": 1}}, 0.0),
])
def test_mpc_lands_precisely(coast_sc, coast_plans, case):
    key, truth, hz = case
    res = simulate(coast_sc, coast_plans[key], truth_from_case(coast_sc, truth),
                   guidance=GUIDANCE[key](coast_sc), replan_hz=hz, controller="mpc")
    m = res.metrics
    assert m["landed"] and m["mpc_failures"] <= 5
    assert m["touchdown_pos_err_m"] < 0.5
    assert m["touchdown_vert_speed"] < 0.5 and m["touchdown_horiz_speed"] < 0.3
    assert m["touchdown_tilt_deg"] < 0.5 and m["touchdown_body_rate_deg_s"] < 0.5
