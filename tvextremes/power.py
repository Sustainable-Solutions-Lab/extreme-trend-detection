"""Power study, Stage A: exact GEV_k data with known time variation (docs/power_study_plan.md).

For one design (k, T) and one baseline xi0, ``simulate_design`` draws N panels per alternative and
records, per panel:

* nyblom   Nyblom L_T at the constant-parameter MLE
* xi_hat   the constant-parameter xi_hat (needed for the paper's xi_hat-adjusted critical value)
* trend_lr non-stationary GEV linear-location-trend LR (log-likelihood difference)
* mu1_hat  its estimated location change over the sample
* mk_z     Mann-Kendall Z for the per-period maxima

All alternatives reuse the same exponential draws and Brownian paths (common random numbers), so
differences across alternatives and effect sizes are not blurred by independent noise. The null
(amount 0) is simulated once and shared.

Rejection rules are applied afterwards (``rejection_table``), so critical values can be changed
without re-simulating:

* nominal: Nyblom with the paper's xi_hat adjustment (adj(xi_hat) L_T > 1); trend LR with chi^2_1;
  Mann-Kendall with the normal approximation.
* size-adjusted: each statistic against the 95% quantile of its own simulated null at the true xi0
  (oracle critical value), which compares tests fairly when one of them is over-sized.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2, norm

from . import critical_values as cv
from . import pseudo
from .mle import fit_local
from .nyblom import nyblom_test
from .trend_tests import mann_kendall, trend_lr_test

STATS = ("nyblom", "xi_hat", "trend_lr", "mu1_hat", "mk_z")

# Default alternatives. Amount units: mu in baseline sigma, sigma in log sigma, xi in xi.
MU_AMOUNTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SIGMA_AMOUNTS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8)
XI_AMOUNTS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)


def default_alternatives():
    alts = [pseudo.NULL]
    for shape in pseudo.SHAPES:
        alts += [pseudo.Alternative("mu", shape, a) for a in MU_AMOUNTS]
    alts += [pseudo.Alternative("sigma", "linear", a) for a in SIGMA_AMOUNTS]
    alts += [pseudo.Alternative("xi", "linear", a) for a in XI_AMOUNTS]
    return alts


def panel_statistics(X, start):
    """The five statistics for one panel X (T, k). start: (mu, sigma, xi) for the constant fit."""
    out = np.full(len(STATS), np.nan)
    fit = fit_local(X, start)
    # A second start from the data guards against a poor start under strong alternatives.
    alt_start = (float(X[:, 0].mean()), float(max(X[:, 0].std(), 1e-3)), 0.1)
    fit2 = fit_local(X, alt_start)
    if fit2.success and (not fit.success or fit2.loglik > fit.loglik):
        fit = fit2
    if fit.success:
        out[0] = nyblom_test(X, fit, interior_tol=0.1)
        out[1] = fit.xi
        lr, mu1, _ = trend_lr_test(X, fit)
        out[2], out[3] = lr, mu1
    if X.shape[0] > 2:
        out[4] = mann_kendall(X[:, 0])[0]
    return out


def _chunk(args):
    alts, k, T, xi0, gammas, brownian = args
    base = (0.0, 1.0, xi0)
    out = np.full((len(alts), gammas.shape[0], len(STATS)), np.nan)
    for a, alt in enumerate(alts):
        for i in range(gammas.shape[0]):
            mu, sigma, xi = pseudo.parameter_paths(alt, T, base, brownian[i])
            X = pseudo.gev_k_panel(gammas[i], mu, sigma, xi)
            start = (float(mu.mean()), float(sigma.mean()), float(xi.mean()))
            out[a, i] = panel_statistics(X, start)
    return out


@dataclass
class DesignResult:
    k: int
    T: int
    xi0: float
    N: int
    seed: int
    alternatives: list
    stats: np.ndarray                 # (n_alt, N, len(STATS))
    meta: dict = field(default_factory=dict)

    def stat(self, name):
        return self.stats[..., STATS.index(name)]

    def null_index(self):
        return next(i for i, a in enumerate(self.alternatives) if a.amount == 0)


def simulate_design(k, T, xi0, N=2000, seed=1, alternatives=None, workers=None):
    """Simulate every alternative for one design and baseline xi0 (parallel over replications)."""
    alts = list(alternatives or default_alternatives())
    rng = np.random.default_rng(seed)
    gammas = np.cumsum(rng.standard_exponential((N, T, k)), axis=2)
    brownian = pseudo.brownian_paths(N, T, rng)
    workers = workers or max(1, (os.cpu_count() or 3) - 2)
    chunk = max(1, -(-N // (4 * workers)))
    jobs = [(alts, k, T, xi0, gammas[s:s + chunk], brownian[s:s + chunk]) for s in range(0, N, chunk)]
    if workers == 1:
        parts = [_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=cv._single_thread) as ex:
            parts = list(ex.map(_chunk, jobs))
    return DesignResult(k, T, xi0, N, seed, alts, np.concatenate(parts, axis=1))


def save(result: DesignResult, path):
    labels = np.array([a.label for a in result.alternatives])
    np.savez_compressed(path, stats=result.stats, labels=labels, k=result.k, T=result.T,
                        xi0=result.xi0, N=result.N, seed=result.seed, stat_names=np.array(STATS))


def load(path):
    d = np.load(path)
    alts = []
    for lab in d["labels"]:
        p, s, a = str(lab).split(":")
        alts.append(pseudo.Alternative(p, s, float(a)))
    return DesignResult(int(d["k"]), int(d["T"]), float(d["xi0"]), int(d["N"]), int(d["seed"]),
                        alts, d["stats"])


def rejection_table(result: DesignResult, alpha=0.05, nyblom_coef=None):
    """One row per alternative: nominal and size-adjusted rejection rates of each test.

    nyblom_coef: (a0, a1, a2) of the xi_hat adjustment; looked up in data/calibration if None.
    If no calibration exists for (k, T), the nominal Nyblom rate is NaN.
    """
    if nyblom_coef is None:
        try:
            nyblom_coef = cv.get_adjustment("nyblom", result.k, result.T, alpha)
        except KeyError:
            nyblom_coef = None
    L, xh = result.stat("nyblom"), result.stat("xi_hat")
    lr, mk = result.stat("trend_lr"), np.abs(result.stat("mk_z"))
    j0 = result.null_index()
    oracle = {name: np.nanquantile(s[j0], 1 - alpha) for name, s in
              (("nyblom", L), ("trend_lr", lr), ("mk", mk))}
    rows = []
    for j, alt in enumerate(result.alternatives):
        rl = pseudo.return_level_change(alt, result.T, (0.0, 1.0, result.xi0)) \
            if alt.shape != "randomwalk" else {10: np.nan, 100: np.nan}
        def rate(x):
            x = x[~np.isnan(x)]
            return float(x.mean()) if x.size else np.nan
        row = dict(k=result.k, T=result.T, xi0=result.xi0, param=alt.param, shape=alt.shape,
                   amount=alt.amount, d_rl10=rl[10], d_rl100=rl[100], N=result.N,
                   failures=int(np.isnan(L[j]).sum()))
        row["nyblom_nominal"] = (rate(np.where(np.isnan(L[j]), np.nan,
                                               cv.adjustment(nyblom_coef, xh[j]) * L[j] > 1))
                                 if nyblom_coef is not None else np.nan)
        row["trend_lr_nominal"] = rate(np.where(np.isnan(lr[j]), np.nan,
                                                2 * lr[j] > chi2.ppf(1 - alpha, 1)))
        row["mk_nominal"] = rate(np.where(np.isnan(mk[j]), np.nan, mk[j] > norm.ppf(1 - alpha / 2)))
        for name, s in (("nyblom", L[j]), ("trend_lr", lr[j]), ("mk", mk[j])):
            row[f"{name}_sizeadj"] = rate(np.where(np.isnan(s), np.nan, s > oracle[name]))
        rows.append(row)
    return rows
