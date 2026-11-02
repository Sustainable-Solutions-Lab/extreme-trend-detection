"""Stage B pseudo-data: daily series from known parent distributions (docs/power_study_plan.md).

Every parent is built from a daily Gaussian series Z with AR(1) correlation phi (stationary,
unit variance), mapped through the parent's quantile function (a Gaussian copula):

    X_d = m_t + s_t * eps(Z_d; tail_t)

where m_t, s_t and the tail parameter vary between years but are constant within a year.

* normal: eps = Z
* gamma:  eps = (G - a)/sqrt(a), G the gamma(a) quantile at Phi(Z); skewed like precipitation
* t:      eps = t_nu quantile at Phi(Z), divided by its sd sqrt(nu/(nu-2)); heavy tail, xi = 1/nu
* tanh:   eps = c tanh(Z/c); normal for Z << c, soft cap at m + s c (c = headroom in units of s)
* beta:   eps = 3 (2 B - 1), B the beta(alpha, beta) quantile at Phi(Z), baseline
          alpha = beta = 4 (unit variance; xi = -0.25, typical of temperature extremes).
          Hard bounds at m -/+ 3 s. The true tail index is
          xi = -1/beta exactly (the density falls like (cap - x)^(beta-1)), whatever alpha is.
          alpha moves the mass within the fixed bounds: the mean is alpha/(alpha + beta), so
          raising alpha pushes the bulk and the maxima toward the upper bound with xi unchanged.

At baseline every parent has eps with mean 0 and variance about 1 (tanh slightly less), so
changes in m are in units of the daily standard deviation for all of them.

For a fixed year the map Z -> X is increasing, so the k largest X are the k largest Z mapped
through it, and declustering picks the same days. The selection is therefore done once on Z per
replicate and shared by all alternatives (common random numbers), and only the k values per year
are transformed. The true distribution of the annual maximum is exact for independent days:
P(max X <= x) = Phi(z(x))^D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm
from scipy.stats import t as t_dist

FAMILIES = ("normal", "gamma", "t", "tanh", "beta")
BASE_TAIL = {"normal": None, "gamma": 2.0, "t": 6.0, "tanh": 3.0, "beta": 4.0}
TAIL_NAME = {"gamma": "shape a", "t": "degrees of freedom", "tanh": "headroom c = L/s",
             "beta": "b (xi = -1/b)"}
KNOBS = ("mean", "width", "tail", "alpha", "capped_warming")
SHAPES = ("linear", "step50", "step80", "exp")
DAYS = 365
BETA_SCALE = np.sqrt(2 * 4.0 + 1)          # = 3; fixed, so the bounds do not move with alpha, beta


def _beta_ab(tail):
    """(alpha, beta) from a beta tail argument: None (baseline), a scalar (alpha = beta), or an
    array whose last axis is (alpha, beta)."""
    if tail is None:
        return 4.0, 4.0
    tail = np.asarray(tail, float)
    if tail.ndim and tail.shape[-1] == 2:
        return tail[..., 0], tail[..., 1]
    return tail, tail


# ---------------------------------------------------------------------------
# Daily Gaussian series and k-largest selection
# ---------------------------------------------------------------------------

def ar1_gaussian(n_rep, n_days, phi, rng):
    """Stationary AR(1) with unit variance, shape (n_rep, n_days)."""
    e = rng.standard_normal((n_rep, n_days))
    if phi == 0:
        return e
    b = np.sqrt(1.0 - phi**2)
    e[:, 0] /= b                          # so that Z_0 ~ N(0, 1)
    return lfilter([b], [1.0, -phi], e, axis=1)


def k_largest_by_year(z, T, k, r=0, days=DAYS):
    """k largest values per year, descending, at least r days apart (greedy). z: (n, T*days).

    Returns (n, T, k).
    """
    n = z.shape[0]
    zy = z.reshape(n * T, days)
    if r == 0:
        part = -np.partition(-zy, k - 1, axis=1)[:, :k]
        out = -np.sort(-part, axis=1)
    else:
        zy = zy.copy()
        out = np.empty((n * T, k))
        rows = np.arange(n * T)
        d = np.arange(days)
        for j in range(k):
            idx = np.argmax(zy, axis=1)
            out[:, j] = zy[rows, idx]
            zy[np.abs(d[None, :] - idx[:, None]) <= r] = -np.inf
    return out.reshape(n, T, k)


# ---------------------------------------------------------------------------
# Parent transforms
# ---------------------------------------------------------------------------

def standard_eps(family, z, tail=None):
    """eps(z; tail) for one family. z: array; tail: scalar or array broadcastable to z (for the
    beta, optionally with a last axis of length 2 holding (alpha, beta))."""
    z = np.asarray(z, float)
    if family == "normal":
        return z
    if family == "tanh":
        return tail * np.tanh(z / tail)
    p_up = norm.sf(z)                      # upper-tail probability, accurate for large z
    if family == "t":
        return t_dist.isf(p_up, tail) / np.sqrt(tail / (tail - 2.0))
    if family == "gamma":
        return (gamma_dist.isf(p_up, tail) - tail) / np.sqrt(tail)
    if family == "beta":
        a, b = _beta_ab(tail)
        return (2.0 * beta_dist.isf(p_up, a, b) - 1.0) * BETA_SCALE
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


def transform(family, z_top, m, s, tail):
    """X = m_t + s_t eps(z; tail_t) for the selected values z_top (..., T, k); parameters (T,)."""
    m, s = np.asarray(m, float)[:, None], np.asarray(s, float)[:, None]
    tail = None if tail is None else np.asarray(tail, float)
    if tail is not None:
        tail = tail[:, None, :] if tail.ndim == 2 else tail[:, None]
    return m + s * standard_eps(family, z_top, tail)


# ---------------------------------------------------------------------------
# Alternatives
# ---------------------------------------------------------------------------

def time_index(T):
    return np.linspace(0.0, 1.0, T) if T > 1 else np.zeros(1)


def shape_values(shape, T):
    """Centred path g(u) with g(1) - g(0) = 1."""
    u = time_index(T)
    if shape == "linear":
        g = u
    elif shape == "step50":
        g = (u >= 0.5).astype(float)
    elif shape == "step80":
        g = (u >= 0.8).astype(float)
    elif shape == "exp":                   # exponential growth: slow, then fast
        g = np.expm1(3.0 * u) / np.expm1(3.0)
    else:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    return g - g.mean()


@dataclass(frozen=True)
class Change:
    """Which knob moves, along which path, by how much.

    Amount units: mean and capped_warming in units of s (the baseline width, 1); width in log s;
    tail and alpha in log of the parameter. For the beta, "tail" changes beta (the upper-tail
    shape, xi = -1/beta) and "alpha" changes alpha (the position of the mass within the fixed
    bounds), each with the other held at its baseline of 4. capped_warming (tanh only): the centre m rises while
    the absolute cap stays fixed.
    """
    knob: str
    shape: str
    amount: float

    @property
    def label(self):
        return f"{self.knob}:{self.shape}:{self.amount:g}"


NULL = Change("mean", "linear", 0.0)


def parameter_paths(family, change: Change, T, tail0=None):
    """(m_t, s_t, tail_t) for a family and a change; tail_t is None for the normal."""
    tail0 = BASE_TAIL[family] if tail0 is None else tail0
    m = np.zeros(T)
    s = np.ones(T)
    tail = None if tail0 is None else np.full(T, float(tail0))
    if family == "beta":
        tail = np.full((T, 2), float(tail0))             # columns: alpha, beta
    if change.amount == 0:
        return m, s, tail
    g = change.amount * shape_values(change.shape, T)
    if change.knob == "mean":
        m = m + g
    elif change.knob == "width":
        s = np.exp(g)
    elif change.knob == "tail":
        if tail is None:
            raise ValueError("the normal has no tail parameter")
        if family == "beta":
            tail[:, 1] = tail0 * np.exp(g)
        else:
            tail = tail0 * np.exp(g)
    elif change.knob == "alpha":
        if family != "beta":
            raise ValueError("alpha applies to the beta parent only")
        tail[:, 0] = tail0 * np.exp(g)
    elif change.knob == "capped_warming":
        if family != "tanh":
            raise ValueError("capped_warming applies to tanh; for the beta use the alpha knob")
        m = m + g
        cap = cap_height(family, tail0)                 # absolute cap, fixed
        headroom = cap - m                              # distance from centre to cap
        if np.any(headroom <= 0):
            raise ValueError("the centre reaches the cap")
        if family == "tanh":                            # cap = m + s c with s = 1: c = headroom
            tail = headroom
        else:                                           # beta: cap = m + s*scale; shrink s, keep b
            s = headroom / BETA_SCALE
    else:
        raise ValueError(f"unknown knob {change.knob!r}")
    return m, s, tail


def cap_height(family, tail0=None):
    """Upper bound of X at baseline (m = 0, s = 1)."""
    tail0 = BASE_TAIL[family] if tail0 is None else tail0
    if family == "tanh":
        return float(tail0)
    if family == "beta":
        return float(BETA_SCALE)
    return np.inf


def default_changes(family):
    out = [NULL]
    # The annual maximum of 365 days sits ~2.8 daily sd above the mean with a spread of ~0.35 sd,
    # so a mean shift of 0.25 daily sd is already ~0.7 GEV sigma.
    amounts = {"mean": (0.1, 0.25, 0.5, 1.0), "width": (0.05, 0.1, 0.2, 0.4),
               "tail": (0.25, 0.5, 1.0, 1.5), "alpha": (0.25, 0.5, 1.0, 1.5),
               "capped_warming": (0.1, 0.25, 0.5, 1.0)}
    if family == "beta":
        # Keep beta >= 2.68 (true xi >= -0.37) in every year. GEV maximum likelihood is regular
        # only for xi > -0.5 (Smith 1985), and a constant GEV fitted across a large change in a
        # bounded variable is pushed well below the true xi (to about -0.9), where the fitted
        # endpoint hugs the sample maximum and the Nyblom information matrix does not exist.
        # With beta0 = 4 the most negative centred path value is -0.5, so log changes <= 0.8;
        # the sizes below stay well inside that (beta >= 3.27).
        amounts["tail"] = (0.05, 0.1, 0.2, 0.4)
        # The beta maxima sit close to the cap with a small spread, so changes of the size used
        # for the unbounded parents already give power ~1; these sizes keep the curves informative.
        amounts["mean"] = (0.05, 0.1, 0.25, 0.5)
        amounts["width"] = (0.025, 0.05, 0.1, 0.2)
        amounts["alpha"] = (0.1, 0.25, 0.5, 1.0)
    knobs = ["mean", "width"]
    if BASE_TAIL[family] is not None:
        knobs.append("tail")
    if family == "beta":
        knobs.append("alpha")
    if family == "tanh":
        knobs.append("capped_warming")
    for knob in knobs:
        for shape in SHAPES:
            out += [Change(knob, shape, a) for a in amounts[knob]]
    return out


# ---------------------------------------------------------------------------
# Truth (independent days)
# ---------------------------------------------------------------------------

def true_return_level(family, m, s, tail, years=100, days=DAYS):
    """Exact m-year return level of the annual maximum per year, for independent days (phi = 0).

    P(max X <= x) = Phi(z)^days, so the level is the map applied to z = Phi^-1((1-1/m)^(1/days)).
    """
    z = norm.isf(-np.expm1(np.log1p(-1.0 / years) / days))
    tail = None if tail is None else np.asarray(tail, float)
    return np.asarray(m, float) + np.asarray(s, float) * standard_eps(family, z, tail)


# ---------------------------------------------------------------------------
# Truth for any phi: quantiles of the annual maximum of the daily Gaussian series
# ---------------------------------------------------------------------------

_ZMAX_CACHE = {}


def zmax_quantile(p, phi, days=DAYS, n_years=500_000, seed=20261101, cache_dir=None):
    """p-quantile of the annual maximum of the stationary AR(1) Gaussian series.

    Exact for phi = 0 (Phi^-1(p^(1/days))). For phi > 0 there is no formula: the annual maxima
    of n_years independent stationary years are simulated once per phi and cached (in memory,
    and in cache_dir as .npy if given). With 500,000 years the Monte Carlo error of the 99%
    quantile is about 0.005 in units of the daily sd, negligible next to estimation error.
    """
    p = np.asarray(p, float)
    if phi == 0:
        return norm.ppf(np.exp(np.log(p) / days))
    key = (float(phi), days, n_years, seed)
    if key not in _ZMAX_CACHE:
        path = None
        if cache_dir is not None:
            path = cache_dir / f"zmax_phi{phi:.2f}_d{days}_n{n_years}_s{seed}.npy"
        if path is not None and path.exists():
            mx = np.load(path)
        else:
            rng = np.random.default_rng(seed)
            mx = np.concatenate([ar1_gaussian(m, days, phi, rng).max(axis=1)
                                 for m in np.diff(np.r_[0:n_years:20_000, n_years])])
            mx.sort()
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, mx)
        _ZMAX_CACHE[key] = mx
    return np.quantile(_ZMAX_CACHE[key], p)


def true_quantile_path(family, m, s, tail, p, phi, **kw):
    """True p-quantile of the annual maximum in every year, (T,), for known parameter paths.

    The map from the Gaussian series to X is increasing within a year, so the annual maximum of
    X is the map applied to the annual maximum of Z, and so are its quantiles.
    """
    z = zmax_quantile(p, phi, **kw)
    tail = None if tail is None else np.asarray(tail, float)
    return np.asarray(m, float) + np.asarray(s, float) * standard_eps(family, z, tail)
