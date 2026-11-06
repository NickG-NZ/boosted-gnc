"""Simple truth wind model: steady wind with altitude shear plus seeded turbulence.

The wind depends on altitude only (frozen turbulence: the vehicle descends through a fixed random
profile), so it plugs into the time-invariant truth dynamics and is exactly repeatable per seed.

    w(h) = mean * (max(h, h_floor) / ref_alt) ** shear  +  gust(h)

`gust` is a first-order Gauss-Markov process in altitude (correlation length `gust_length`), with
standard deviation `gust_sigma` on the horizontal axes and half that vertically, linearly
interpolated between grid points so it is continuous.

Config (a case's `truth.wind`): a 3-vector gives a constant wind (the previous behaviour), or
    {"mean": [0, 0, 8], "ref_alt": 10, "shear": 0.14, "gust_sigma": 1.5, "gust_length": 200, "seed": 1}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

H_FLOOR = 1.0  # m, keeps the shear factor finite at the ground
H_MAX = 12_000.0  # m, top of the gust profile (held constant above)
VERTICAL_GUST_FRAC = 0.5


@dataclass(frozen=True)
class WindModel:
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)  # steady wind at ref_alt (m/s, guidance frame)
    ref_alt: float = 10.0  # m
    shear: float = 0.0  # power-law exponent (~0.14 over open ground); 0 = no shear
    gust_sigma: float = 0.0  # m/s, horizontal turbulence std
    gust_length: float = 200.0  # m, turbulence correlation length in altitude
    seed: int = 0
    _h: np.ndarray = field(init=False, repr=False, compare=False)
    _gust: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dh = self.gust_length / 10.0
        h = np.arange(0.0, H_MAX + dh, dh)
        gust = np.zeros((len(h), 3))
        if self.gust_sigma > 0:
            rng = np.random.default_rng(self.seed)
            sig = self.gust_sigma * np.array([VERTICAL_GUST_FRAC, 1.0, 1.0])
            a = math.exp(-dh / self.gust_length)
            gust[0] = sig * rng.standard_normal(3)
            noise = rng.standard_normal((len(h) - 1, 3)) * sig * math.sqrt(1 - a * a)
            for k in range(1, len(h)):
                gust[k] = a * gust[k - 1] + noise[k - 1]
        object.__setattr__(self, "_h", h)
        object.__setattr__(self, "_gust", gust)

    @classmethod
    def from_config(cls, cfg: Any) -> WindModel:
        if isinstance(cfg, dict):
            return cls(**{**cfg, "mean": tuple(float(x) for x in cfg.get("mean", (0, 0, 0)))})
        return cls(mean=tuple(float(x) for x in cfg))

    def at(self, h: float) -> np.ndarray:
        """Wind velocity (m/s, guidance frame [up, x, y]) at altitude h (m)."""
        mean = np.asarray(self.mean, float)
        if self.shear:
            mean = mean * (max(h, H_FLOOR) / self.ref_alt) ** self.shear
        if not self.gust_sigma:
            return mean
        hc = min(max(h, 0.0), H_MAX)
        gust = np.array([np.interp(hc, self._h, self._gust[:, i]) for i in range(3)])
        return mean + gust
