"""Pseudo-data with known time variation in the GEV parameters (power study, Stage A).

A path describes how one GEV parameter changes over the sample. The time index is u_t = t/(T-1),
running from 0 to 1. Each path is

    theta_t = theta_0 + amount * g(u_t)

where g is a shape with g(1) - g(0) = 1 for monotone shapes (peak height 1 for the hump), centred
so that its average over the sample is zero. Centring keeps the time-averaged parameter at the
baseline, so tail shape and scale are comparable across shapes. Amount units:

* "mu":    in units of the baseline sigma (amount = 1 shifts the location by one sigma)
* "sigma": in log sigma (amount = 0.2 raises the scale by about 22% over the sample)
* "xi":    in xi itself

The k largest of each period are drawn from GEV_k(mu_t, sigma_t, xi_t) through the exponential
representation of the paper's eq. (2): if Gamma_j are cumulative sums of standard exponentials,
X_j = mu + sigma (Gamma_j^-xi - 1)/xi. Feeding the same exponential draws to every path gives
common random numbers across alternatives, which makes power differences much less noisy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import likelihood as lk

SHAPES = ("linear", "accel", "step50", "step80", "hump", "randomwalk")
PARAMS = ("mu", "sigma", "xi")


def time_index(T):
    return np.linspace(0.0, 1.0, T) if T > 1 else np.zeros(1)


def shape_values(shape, T, brownian=None):
    """g(u_t) for t = 0..T-1, centred to mean zero.

    brownian: (T,) standard Brownian motion at u_t (W(0) = 0), needed for "randomwalk".
    """
    u = time_index(T)
    if shape == "linear":
        g = u
    elif shape == "accel":                 # quadratic: slow then fast, like recent warming
        g = u**2
    elif shape == "step50":
        g = (u >= 0.5).astype(float)
    elif shape == "step80":
        g = (u >= 0.8).astype(float)
    elif shape == "hump":                  # rise then fall; a linear-trend test should miss it
        g = np.sin(np.pi * u)
    elif shape == "randomwalk":
        if brownian is None:
            raise ValueError("randomwalk needs a Brownian path")
        g = np.asarray(brownian, float)
    else:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    return g - g.mean()


def brownian_paths(N, T, rng):
    """N standard Brownian motions on [0, 1] observed at u_t, shape (N, T), with W(0) = 0."""
    du = np.diff(time_index(T))
    inc = rng.standard_normal((N, T - 1)) * np.sqrt(du)
    return np.concatenate([np.zeros((N, 1)), np.cumsum(inc, axis=1)], axis=1)


@dataclass(frozen=True)
class Alternative:
    """One kind of time variation: which parameter moves, along which shape, by how much."""
    param: str
    shape: str
    amount: float

    def __post_init__(self):
        if self.param not in PARAMS:
            raise ValueError(f"unknown parameter {self.param!r}")
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}")

    @property
    def label(self):
        return f"{self.param}:{self.shape}:{self.amount:g}"


NULL = Alternative("mu", "linear", 0.0)


def parameter_paths(alt: Alternative, T, base=(0.0, 1.0, 0.0), brownian=None):
    """(mu_t, sigma_t, xi_t), each of shape (T,), for an alternative around base (mu, sigma, xi)."""
    mu0, s0, xi0 = base
    mu = np.full(T, float(mu0))
    sigma = np.full(T, float(s0))
    xi = np.full(T, float(xi0))
    if alt.amount == 0:
        return mu, sigma, xi
    g = alt.amount * shape_values(alt.shape, T, brownian)
    if alt.param == "mu":
        mu = mu + s0 * g
    elif alt.param == "sigma":
        sigma = s0 * np.exp(g)
    else:
        xi = xi + g
    return mu, sigma, xi


def gev_k_panel(gammas, mu, sigma, xi):
    """k largest per period from GEV_k(mu_t, sigma_t, xi_t). gammas: (T, k) cumulative exponentials.

    Returns (T, k), rows descending. Parameters are scalars or (T,) arrays.
    """
    T = gammas.shape[0]
    mu, sigma, xi = (p[:, None] for p in lk._as_period_params(mu, sigma, xi, T))
    small = np.abs(xi) < 1e-12
    xi_safe = np.where(small, 1.0, xi)
    with np.errstate(over="ignore"):
        x = np.where(small, -np.log(gammas), (gammas ** (-xi_safe) - 1.0) / xi_safe)
    return mu + sigma * x


def return_level_change(alt: Alternative, T, base=(0.0, 1.0, 0.0), years=(10, 100), brownian=None):
    """Change in the m-year return level (GEV_1 quantile 1 - 1/m) over the sample, in units of the
    baseline sigma: last period minus first, or peak minus first for the hump."""
    mu, sigma, xi = parameter_paths(alt, T, base, brownian)
    out = {}
    for m in years:
        q = lk.gev_quantile(1 - 1 / m, mu, sigma, xi)
        q = np.asarray(q, float)
        end = q.max() if alt.shape == "hump" else q[-1]
        out[m] = float((end - q[0]) / base[1])
    return out
