"""Truth wind model (landing/wind.py)."""

from __future__ import annotations

import numpy as np

from landing.wind import WindModel


def test_constant_config_is_unchanged():
    w = WindModel.from_config([0.0, 1.0, 8.0])
    for h in (0.0, 50.0, 3000.0):
        assert np.array_equal(w.at(h), [0.0, 1.0, 8.0])


def test_shear_power_law():
    w = WindModel(mean=(0, 0, 8), ref_alt=10, shear=0.14)
    assert np.allclose(w.at(10.0), [0, 0, 8])
    assert np.isclose(w.at(1000.0)[2], 8 * 100**0.14)
    assert np.isfinite(w.at(0.0)).all() and w.at(0.0)[2] < 8


def test_gusts_are_seeded_continuous_and_sized():
    cfg = {"mean": [0, 0, 0], "gust_sigma": 2.0, "gust_length": 200.0, "seed": 3}
    a, b = WindModel.from_config(cfg), WindModel.from_config(cfg)
    other = WindModel.from_config({**cfg, "seed": 4})
    h = np.linspace(0, 10_000, 4001)
    ga = np.array([a.at(x) for x in h])
    assert np.array_equal(ga, [b.at(x) for x in h])  # repeatable per seed
    assert not np.allclose(ga, [other.at(x) for x in h])
    assert np.max(np.abs(np.diff(ga, axis=0))) < 1.0  # continuous: no jumps between 2.5 m steps
    std = ga.std(axis=0)
    assert 1.0 < std[1] < 3.0 and 1.0 < std[2] < 3.0  # horizontal ~ sigma
    assert std[0] < std[1]  # vertical gusts weaker
