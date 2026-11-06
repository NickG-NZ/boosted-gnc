"""Rust truth plant (sim/) driven through the Python interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnc.plant import make_plant
from gnc.sim import truth_from_case
from gnc.vehicle import Scenario

CFG = Path(__file__).resolve().parents[1] / "missions/coast_landing/scenario.json"
CHANNELS = {"t", "r", "v", "m", "b", "w", "g", "thrust", "wind"}


@pytest.fixture(scope="module")
def sc() -> Scenario:
    return Scenario.from_json(CFG)


def test_vacuum_free_fall_is_exact_and_lands(sc):
    plant = make_plant(sc, truth_from_case(sc, {"drag": False}), np.array([1.0, 0.0, 0.0]))
    t, dt = 0.0, 0.01
    while not (info := plant.step(0.0, dt, fins=np.zeros(3))).contact:
        t += info.dt
        g = np.array([-sc.env.g0, 0.0, 0.0])
        assert np.allclose(plant.r, sc.r0 + sc.v0 * t + 0.5 * g * t**2, rtol=1e-9, atol=1e-6)
    assert plant.r[0] == pytest.approx(0.0, abs=1e-9)  # interpolated to ground contact
    assert plant.m == sc.m0
    plant.finish()
    tel = plant.telemetry()
    assert set(tel) == CHANNELS and len(tel["t"]) > 100
    assert all(len(v) == len(tel["t"]) for v in tel.values())
    with pytest.raises(ValueError):  # on the ground: no more steps
        plant.step(0.0, dt)


def test_mass_flow_follows_isp(sc):
    truth = truth_from_case(sc, {"drag": False, "isp_scale": 0.9})
    plant = make_plant(sc, truth, np.array([1.0, 0.0, 0.0]))
    T = 0.6 * sc.rocket.T_max
    for _ in range(100):
        plant.step(T, 0.01, gimbal=np.zeros(3))
    assert plant.m == pytest.approx(sc.m0 - T * 1.0 / (sc.rocket.isp * 0.9 * sc.env.g0), rel=1e-12)


def test_gusty_wind_matches_python_wind_model(sc):
    case = {"wind": {"mean": [0.0, 1.0, 8.0], "shear": 0.14, "gust_sigma": 1.5, "gust_length": 200.0, "seed": 1}}
    truth = truth_from_case(sc, case)
    plant = make_plant(sc, truth, np.array([1.0, 0.0, 0.0]))
    for h in np.linspace(-10.0, 13_000.0, 2001):
        assert np.allclose(plant.wind_at(np.array([h, 0.0, 0.0])), truth.wind_at(h), rtol=0, atol=1e-12)


def test_engine_and_gimbal_limits_enforced(sc):
    plant = make_plant(sc, truth_from_case(sc, {"drag": False}), np.array([1.0, 0.0, 0.0]))
    info = plant.step(10 * sc.rocket.T_max, 0.01, gimbal=np.array([0.0, 1.0, 0.0]))
    assert info.thrust == pytest.approx(sc.rocket.T_max)
    assert np.linalg.norm(info.g_cmd) == pytest.approx(np.sin(np.radians(sc.rocket.attitude["gimbal_max_deg"])))
    # one 10 ms step: the gimbal moves at most rate * dt
    assert np.linalg.norm(plant.att.g) <= np.radians(sc.rocket.attitude["gimbal_rate_deg_s"]) * 0.01 + 1e-12
    assert plant.step(0.1 * sc.rocket.T_max, 0.01).thrust == pytest.approx(sc.rocket.T_min)


def test_bad_step_is_rejected(sc):
    plant = make_plant(sc, truth_from_case(sc, {}), np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        plant.step(0.0, -0.01)
