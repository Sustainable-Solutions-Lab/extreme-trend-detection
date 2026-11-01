"""Small-sample critical values for the k-largest tests (paper §2.2.2, §3.1 and Appendix B).

The null distribution of each statistic depends on the GEV parameters only through xi. Rather than
using a sup-bound critical value, the paper scales the statistic by

    adj(xi_hat) = exp(a0 + a1 xi_hat + a2 xi_hat^2)

and rejects when adj * stat > 1. The coefficients are fitted by simulation so that the test has
size <= alpha at every xi on a grid, while being as close to alpha as possible (approximately
unbiased). Tests, numbered as in the authors' code:

    0 "nyblom"  Nyblom L_T for constant parameters    xi grid [-0.5, 1.5], 10 points
    1 "xi"      LR for xi = xi0                        no nuisance parameter: plain quantile
                                                       of LR per xi0, on 18 points in [-0.5, 1.5]
    2 "q90"     LR for q_0.9 = q0                      xi grid [-0.5, 1.5], 10 points
    3 "pareto"  LR for mu = sigma/xi                   xi grid [0.03, 1.5], 10 points
    4 "zipf"    LR for mu = sigma/xi and xi = 1        single DGP (xi = 1)

The procedure follows the authors' Fortran (``Fortran/compute.f90``, ``adjcvmod.f90``):

1. Draw N panels of T periods from GEV_k(0, 1, xi) by the exponential representation of eq. (2),
   using the same exponential draws at every grid point (common random numbers). Under test 3 the
   DGP has mu = sigma/xi; under test 4, xi = 1 and mu = sigma = 1.
2. For each panel compute the statistic and xi_hat. Fits start at the true parameters.
3. Smooth the rejection probabilities with a normal kernel (bandwidth h = 0.3 x the 93rd-97th
   percentile range of the statistic at the middle grid point) and choose (a0, a1, a2) to minimize
   sum_j linex(logit P(accept)_j - logit(1 - alpha)) with linex(x) = exp(-c x) + c x - 1. The
   asymmetry penalizes over-rejection. The constant c is raised in steps 0.3, 3, 12, and a small
   ridge 0.001 (a1^2 + a2^2) is added.
4. Shift a0 so the largest rejection rate on the grid is exactly alpha.
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import norm

from . import likelihood as lk
from .mle import fit_local
from .nyblom import nyblom_test

TESTS = ("nyblom", "xi", "q90", "pareto", "zipf")
TEST_NUMBER = {name: i for i, name in enumerate(TESTS)}

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration"   # committed to git
ADJ_FILE = DATA_DIR / "cv_adjustment.csv"
XI_FILE = DATA_DIR / "cv_xi.csv"
ADJ_FIELDS = ["source", "scheme", "test", "k", "T", "alpha", "a0", "a1", "a2", "N", "seed",
              "grid_min", "grid_max", "n_grid", "max_rejection", "min_rejection", "failures",
              "created"]
XI_FIELDS = ["source", "scheme", "k", "T", "alpha", "xi0", "cv", "N", "seed", "created"]


# ---------------------------------------------------------------------------
# Null grids and DGPs
# ---------------------------------------------------------------------------

def xi_grid(test):
    """The authors' grid of true xi values for size control."""
    if test == "xi":
        g = np.linspace(-0.5, 1.5, 18)
    elif test in ("pareto", "zipf"):
        g = np.linspace(0.03, 1.5, 10)
    else:
        g = np.linspace(-0.5, 1.5, 10)
    if test == "zipf":
        g = np.array([1.0])            # the null fixes xi = 1; every grid point is the same DGP
    return np.where(np.abs(g) < 1e-4, 1e-3, g)


def null_dgp(test, xi):
    """(mu, sigma, xi) of the simulated null DGP and the restriction for the restricted fit."""
    mu, sigma = 0.0, 1.0
    if test == "pareto":
        mu = sigma / xi
        return (mu, sigma, xi), dict(pareto=True)
    if test == "zipf":
        return (1.0, 1.0, 1.0), dict(pareto=True, fix_xi=1.0)
    if test == "xi":
        return (mu, sigma, xi), dict(fix_xi=xi)
    if test == "q90":
        q = float(lk.gev_quantile(0.9, mu, sigma, xi))
        return (mu, sigma, xi), dict(quantile=(q, 0.9))
    return (mu, sigma, xi), None       # nyblom


def draw_gammas(N, T, k, seed):
    """Cumulative sums of standard exponentials, shape (N, T, k): the building block of eq. (2)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_exponential((N, T, k)), axis=2)


def panels_from_gammas(gammas, mu, sigma, xi):
    """GEV_k(mu, sigma, xi) panels (rows descending) from the exponential cumulative sums."""
    if abs(xi) < 1e-12:
        return mu - sigma * np.log(gammas)
    return mu + sigma * (gammas ** (-xi) - 1.0) / xi


# ---------------------------------------------------------------------------
# Simulation of the null distribution
# ---------------------------------------------------------------------------

def statistic(test, X, truth, restriction):
    """(statistic, xi_hat) for one simulated panel, with fits started at the truth."""
    unr = fit_local(X, truth)
    if test == "nyblom":
        if not unr.success:
            return np.nan, np.nan
        return nyblom_test(X, unr, interior_tol=0.1), unr.xi
    res = fit_local(X, truth, **restriction)
    if res.loglik > unr.loglik + 1e-9:            # unrestricted fit got stuck: retry from res
        unr2 = fit_local(X, res.params)
        if unr2.loglik > unr.loglik:
            unr = unr2
    if not (unr.success and res.success):
        return np.nan, np.nan
    return max(unr.loglik - res.loglik, 0.0), unr.xi


def _single_thread():
    """Worker initializer: one BLAS/OpenMP thread per process, so workers don't oversubscribe."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _simulate_chunk(args):
    test, grid, gammas = args
    out = np.full((len(grid), gammas.shape[0], 2), np.nan)
    for j, xi in enumerate(grid):
        truth, restriction = null_dgp(test, xi)
        for i in range(gammas.shape[0]):
            X = panels_from_gammas(gammas[i], *truth)
            out[j, i] = statistic(test, X, truth, restriction)
    return out


@dataclass
class NullSimulation:
    test: str
    k: int
    T: int
    N: int
    seed: int
    grid: np.ndarray
    stats: np.ndarray            # (n_grid, N): statistic
    xi_hat: np.ndarray           # (n_grid, N)
    meta: dict = field(default_factory=dict)

    @property
    def failures(self):
        return int(np.isnan(self.stats).sum())


def simulate_null(test, k, T, N=10_000, seed=1, workers=None, grid=None, chunk=None):
    """Simulate (statistic, xi_hat) under the null at every grid point. Parallel over replications."""
    if test not in TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {TESTS}")
    grid = xi_grid(test) if grid is None else np.asarray(grid, float)
    gammas = draw_gammas(N, T, k, seed)
    workers = workers or max(1, (os.cpu_count() or 3) - 2)    # leave two cores free
    if chunk is None:                                          # ~4 jobs per worker
        chunk = max(1, -(-N // (4 * workers)))
    jobs = [(test, grid, gammas[s:s + chunk]) for s in range(0, N, chunk)]
    if workers == 1:
        parts = [_simulate_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_single_thread) as ex:
            parts = list(ex.map(_simulate_chunk, jobs))
    out = np.concatenate(parts, axis=1)
    return NullSimulation(test, k, T, N, seed, grid, out[..., 0], out[..., 1])


# ---------------------------------------------------------------------------
# Fitting the critical-value function
# ---------------------------------------------------------------------------

def adjustment(a, xi_hat):
    a = np.asarray(a, float)
    return np.exp(a[0] + a[1] * xi_hat + a[2] * xi_hat**2)


def rejection_rates(sim: NullSimulation, a):
    """Share of simulated statistics with adj(xi_hat) * stat > 1, per grid point (NaNs dropped)."""
    adj_stat = adjustment(a, sim.xi_hat) * sim.stats
    ok = ~np.isnan(adj_stat)
    return np.array([np.mean(adj_stat[j, ok[j]] > 1) for j in range(len(sim.grid))])


def fit_adjustment(sim: NullSimulation, alpha=0.05):
    """Coefficients (a0, a1, a2) per Appendix B. Returns (a, rejection rates on the grid)."""
    stats, xi_hat = sim.stats, sim.xi_hat
    ok = ~np.isnan(stats)
    mid = max(len(sim.grid) // 2 - 1, 0)        # authors' "middle" grid point (5th of 10)
    q93, q97 = np.quantile(stats[mid, ok[mid]], [0.93, 0.97])
    h = 0.3 * (q97 - q93)
    target = logit(1 - alpha)
    s = np.where(ok, stats, 0.0)
    x = np.where(ok, xi_hat, 0.0)
    n_ok = ok.sum(axis=1)

    def accept_prob(a):
        p = (norm.cdf((1 - adjustment(a, x) * s) / h) * ok).sum(axis=1) / n_ok
        return np.clip(p, 1e-12, 1 - 1e-12)

    def objective(a, c):
        z = logit(accept_prob(a)) - target
        return np.sum(np.exp(-c * z) + c * z - 1) + 0.001 * (a[1] ** 2 + a[2] ** 2)

    a = np.zeros(3)
    for c in (0.3, 3.0, 12.0):
        a = minimize(objective, a, args=(c,), method="BFGS").x
        a = minimize(objective, a, args=(c,), method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-12, maxiter=5000)).x
    # Exact size on the grid: scale so the largest (1 - alpha) quantile of adj * stat equals 1.
    adj_stat = adjustment(a, xi_hat) * stats
    worst = max(np.quantile(adj_stat[j, ok[j]], 1 - alpha) for j in range(len(sim.grid)))
    a[0] -= np.log(worst)
    return a, rejection_rates(sim, a)


def xi_test_critical_values(sim: NullSimulation, alpha=0.05):
    """Test 1 (xi = xi0): the (1 - alpha) quantile of LR at each xi0 on the grid."""
    return np.array([np.nanquantile(sim.stats[j], 1 - alpha) for j in range(len(sim.grid))])


# ---------------------------------------------------------------------------
# Storage and lookup
# ---------------------------------------------------------------------------

def _read(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _key(r, extra=()):
    return tuple(str(r[c]) for c in ("source", "scheme", "test", "k", "T", "alpha", *extra)
                 if c in r)


def save_adjustment(sim, a, rates, alpha=0.05, source="tvextremes", scheme="klargest", path=ADJ_FILE):
    """Add or replace the coefficient row for (source, scheme, test, k, T, alpha)."""
    row = dict(source=source, scheme=scheme, test=sim.test, k=sim.k, T=sim.T, alpha=alpha,
               a0=f"{a[0]:.10g}", a1=f"{a[1]:.10g}", a2=f"{a[2]:.10g}", N=sim.N, seed=sim.seed,
               grid_min=f"{sim.grid.min():.4g}", grid_max=f"{sim.grid.max():.4g}",
               n_grid=len(sim.grid), max_rejection=f"{rates.max():.5f}",
               min_rejection=f"{rates.min():.5f}", failures=sim.failures,
               created=date.today().isoformat())
    rows = [r for r in _read(path) if _key(r) != _key(row)]
    rows.append(row)
    rows.sort(key=lambda r: (r["source"], r["scheme"], int(r["T"]), int(r["k"] or 0),
                             TEST_NUMBER.get(r["test"], 9)))
    _write(path, ADJ_FIELDS, rows)


def save_xi_cvs(sim, cvs, alpha=0.05, source="tvextremes", scheme="klargest", path=XI_FILE):
    rows = [r for r in _read(path)
            if (r["source"], r["scheme"], r["k"], r["T"], r["alpha"])
            != (source, scheme, str(sim.k), str(sim.T), str(alpha))]
    for xi0, cv in zip(sim.grid, cvs):
        rows.append(dict(source=source, scheme=scheme, k=sim.k, T=sim.T, alpha=alpha,
                         xi0=f"{xi0:.6g}", cv=f"{cv:.6g}", N=sim.N, seed=sim.seed,
                         created=date.today().isoformat()))
    rows.sort(key=lambda r: (r["source"], r["scheme"], int(r["T"]), int(r["k"]), float(r["xi0"])))
    _write(path, XI_FIELDS, rows)


def get_adjustment(test, k, T, alpha=0.05, scheme="klargest", source=None, path=ADJ_FILE):
    """Stored (a0, a1, a2). source=None prefers our own calibration, then the authors' ('mw2024')."""
    rows = [r for r in _read(path)
            if r["test"] == test and r["scheme"] == scheme and r["k"] == str(k)
            and r["T"] == str(T) and float(r["alpha"]) == alpha]
    for src in ([source] if source else ["tvextremes", "mw2024"]):
        for r in rows:
            if r["source"] == src:
                return np.array([float(r["a0"]), float(r["a1"]), float(r["a2"])])
    raise KeyError(f"no stored critical-value coefficients for test={test}, k={k}, T={T}, "
                   f"alpha={alpha}; run scripts/calibrate_cv.py --k {k} --T {T}")


def get_xi_cv(xi0, k, T, alpha=0.05, scheme="klargest", source=None, path=XI_FILE):
    """Critical value of the xi = xi0 LR test, interpolated linearly in xi0 between grid points."""
    rows = [r for r in _read(path) if r["scheme"] == scheme and r["k"] == str(k)
            and r["T"] == str(T) and float(r["alpha"]) == alpha]
    for src in ([source] if source else ["tvextremes", "mw2024"]):
        sel = sorted((float(r["xi0"]), float(r["cv"])) for r in rows if r["source"] == src)
        if sel:
            g, c = np.array(sel).T
            if not g[0] <= xi0 <= g[-1]:
                raise ValueError(f"xi0={xi0} outside calibrated range [{g[0]}, {g[-1]}]")
            return float(np.interp(xi0, g, c))
    raise KeyError(f"no stored xi critical values for k={k}, T={T}, alpha={alpha}")


def adjusted_statistic(stat, xi_hat, test, k, T, alpha=0.05, **kw):
    """adj(xi_hat) * stat; reject at level alpha when > 1."""
    return float(adjustment(get_adjustment(test, k, T, alpha, **kw), xi_hat) * stat)
