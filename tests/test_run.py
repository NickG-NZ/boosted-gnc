"""Python entry point: run a mission case through the Rust flight software + sim."""

from __future__ import annotations

import json
from pathlib import Path

import boosted_sim
import pytest

SPEC = Path(__file__).resolve().parents[1] / "missions/spec_test_case/scenario.json"


def test_solve_one_case():
    calls = []
    data = json.loads(boosted_sim.solve(str(SPEC), only=["G-FOLD · vacuum"], verbose=False,
                                        progress=lambda *a: calls.append(a)))
    assert set(data) >= {"scenario", "rocket", "plans", "sims"}
    (sim,) = data["sims"]
    assert sim["name"] == "G-FOLD · vacuum" and sim["metrics"]["landed"]
    assert len(sim["t"]) == len(sim["r"]) == len(sim["att"]) > 100
    assert [c[0] for c in calls] == ["plan", "sim", "done"]


def test_errors_are_raised():
    with pytest.raises(RuntimeError, match="not in the scenario"):
        boosted_sim.solve(str(SPEC), only=["no such case"], verbose=False)
    with pytest.raises(ZeroDivisionError):  # a failing progress callback cancels the run
        boosted_sim.solve(str(SPEC), only=["G-FOLD · vacuum"], verbose=False, progress=lambda *a: 1 / 0)
