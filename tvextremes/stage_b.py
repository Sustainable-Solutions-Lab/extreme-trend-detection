"""Power study, Stage B: k largest per year from daily parent-distribution series.

One configuration is (family, k, T, phi, r): parent, number of extremes per year, years, AR(1)
coefficient of the daily series, and declustering gap in days. For each configuration
``simulate_config`` draws the daily Gaussian series once per replicate, selects the k largest
per year (shared by all changes; see ``parents``), and computes the test statistics of
``power.panel_statistics`` for the null (N_null replicates) and for every change (the first
N_alt replicates). Results are summarized by ``summary_rows``.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from . import critical_values as cv
from . import parents as pa
from .power import STATS, panel_statistics

CHUNK = 50          # replicates per job; fixed so results do not depend on the number of workers


def _start(X):
    top = X[:, 0]
    sd = max(float(top.std()), 1e-3)
    sigma = sd * np.sqrt(6) / np.pi
    return float(top.mean() - 0.5772 * sigma), sigma, -0.1


def _job(args):
    family, changes, k, T, phi, r, seed, i0, n, n_alt = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, i0]))
    z = pa.ar1_gaussian(n, T * pa.DAYS, phi, rng)
    ztop = pa.k_largest_by_year(z, T, k, r)                      # (n, T, k)
    del z
    paths = [pa.parameter_paths(family, ch, T) for ch in changes]
    out = np.full((len(changes), n, len(STATS)), np.nan)
    for j, (m, s, tail) in enumerate(paths):
        last = n if changes[j].amount == 0 else min(n, max(0, n_alt - i0))
        if last == 0:
            continue
        X = pa.transform(family, ztop[:last], m, s, tail)        # (last, T, k)
        for i in range(last):
            out[j, i] = panel_statistics(X[i], _start(X[i]))
    return i0, out


@dataclass
class ConfigResult:
    family: str
    k: int
    T: int
    phi: float
    r: int
    seed: int
    changes: list
    stats: np.ndarray            # (n_changes, N_null, len(STATS)); NaN beyond N_alt for changes
    n_alt: int

    def null_index(self):
        return next(i for i, c in enumerate(self.changes) if c.amount == 0)


def simulate_config(family, k, T, phi, r=0, N_null=10_000, N_alt=2_000, seed=1, changes=None,
                    workers=None):
    changes = list(changes or pa.default_changes(family))
    workers = workers or max(1, (os.cpu_count() or 3) - 2)
    jobs = [(family, changes, k, T, phi, r, seed, i0, min(CHUNK, N_null - i0), N_alt)
            for i0 in range(0, N_null, CHUNK)]
    stats = np.full((len(changes), N_null, len(STATS)), np.nan)
    if workers == 1:
        parts = map(_job, jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers, initializer=cv._single_thread)
        parts = ex.map(_job, jobs)
    for i0, out in parts:
        stats[:, i0:i0 + out.shape[1]] = out
    if workers != 1:
        ex.shutdown()
    return ConfigResult(family, k, T, phi, r, seed, changes, stats, N_alt)


def save(res: ConfigResult, path):
    np.savez_compressed(path, stats=res.stats.astype(np.float32),
                        labels=np.array([c.label for c in res.changes]), family=res.family,
                        k=res.k, T=res.T, phi=res.phi, r=res.r, seed=res.seed, n_alt=res.n_alt,
                        stat_names=np.array(STATS))


def load(path):
    d = np.load(path)
    changes = []
    for lab in d["labels"]:
        knob, shape, amount = str(lab).split(":")
        changes.append(pa.Change(knob, shape, float(amount)))
    return ConfigResult(str(d["family"]), int(d["k"]), int(d["T"]), float(d["phi"]), int(d["r"]),
                        int(d["seed"]), changes, d["stats"].astype(float), int(d["n_alt"]))


def summary_rows(res: ConfigResult, alpha=0.05):
    """One row per change: nominal and size-adjusted rejection rates, mean xi_hat, true changes."""
    S = {name: res.stats[..., i] for i, name in enumerate(STATS)}
    try:
        coef = cv.get_adjustment("nyblom", res.k, res.T, alpha)
    except KeyError:
        coef = None
    L, xh, lr, mk = S["nyblom"], S["xi_hat"], S["trend_lr"], np.abs(S["mk_z"])
    j0 = res.null_index()
    oracle = {"nyblom": np.nanquantile(L[j0], 1 - alpha), "trend_lr": np.nanquantile(lr[j0], 1 - alpha),
              "mk": np.nanquantile(mk[j0], 1 - alpha)}

    def rate(x, ok):
        x = x[ok]
        return float(x.mean()) if x.size else np.nan

    rows = []
    for j, ch in enumerate(res.changes):
        ok = ~np.isnan(L[j])
        m, s, tail = pa.parameter_paths(res.family, ch, res.T)
        rl = pa.true_return_level(res.family, m, s, tail, 100)
        rl10 = pa.true_return_level(res.family, m, s, tail, 10)
        row = dict(family=res.family, tail0=pa.BASE_TAIL[res.family], k=res.k, T=res.T,
                   phi=res.phi, r=res.r, knob=ch.knob, shape=ch.shape, amount=ch.amount,
                   N=int(ok.sum()), d_rl10_indep=float(rl10[-1] - rl10[0]), d_rl100_indep=float(rl[-1] - rl[0]),
                   xi_hat_mean=rate(xh[j], ok))
        # replicates that were simulated (Mann-Kendall present) but whose GEV fit failed
        row["failures"] = int(np.isnan(L[j][~np.isnan(S["mk_z"][j])]).sum())
        row["nyblom_nominal"] = (rate(cv.adjustment(coef, xh[j]) * L[j] > 1, ok)
                                 if coef is not None else np.nan)
        row["trend_lr_nominal"] = rate(2 * lr[j] > chi2.ppf(1 - alpha, 1), ok)
        row["mk_nominal"] = rate(mk[j] > norm.ppf(1 - alpha / 2), ok)
        row["nyblom_sizeadj"] = rate(L[j] > oracle["nyblom"], ok)
        row["trend_lr_sizeadj"] = rate(lr[j] > oracle["trend_lr"], ok)
        row["mk_sizeadj"] = rate(mk[j] > oracle["mk"], ok)
        rows.append(row)
    return rows


RL_PROBS = (0.5, 0.9, 0.95, 0.99)


def return_level_rows(res: ConfigResult, probs=RL_PROBS, cache_dir=None):
    """Accuracy of estimated annual-maximum quantiles X_p against the truth implied by the known
    generating function. One row per (change, p).

    X_p is the level that the annual maximum stays below in a fraction p of years. Two estimators:
    'const' (constant-parameter GEV fit: the same X_p every year, so an estimated change of 0)
    and 'trend' (GEV with a linear trend in location: mu_t = mu0 + mu1 (u_t - 1/2)). Each is
    scored in the first year, the last year, and on the change between them: bias (mean error)
    and RMSE.
    """
    from .likelihood import gev_quantile
    S = {name: res.stats[..., i] for i, name in enumerate(STATS)}
    rows = []
    for j, ch in enumerate(res.changes):
        ok = ~np.isnan(S["nyblom"][j]) & ~np.isnan(S["trend_mu0"][j])
        if not ok.any():
            continue
        m, s, tail = pa.parameter_paths(res.family, ch, res.T)
        mu, sg, xi = S["mu_hat"][j][ok], S["sigma_hat"][j][ok], S["xi_hat"][j][ok]
        m0, m1 = S["trend_mu0"][j][ok], S["mu1_hat"][j][ok]
        ts, tx = S["trend_sigma"][j][ok], S["trend_xi"][j][ok]
        for p in probs:
            truth = pa.true_quantile_path(res.family, m, s, tail, p, res.phi, cache_dir=cache_dir)
            t_first, t_last = float(truth[0]), float(truth[-1])
            const = gev_quantile(p, mu, sg, xi)
            est = {"const": (const, const, np.zeros_like(const)),
                   "trend": (gev_quantile(p, m0 - m1 / 2, ts, tx), gev_quantile(p, m0 + m1 / 2, ts, tx),
                             None)}
            est["trend"] = (est["trend"][0], est["trend"][1], est["trend"][1] - est["trend"][0])
            row = dict(family=res.family, k=res.k, T=res.T, phi=res.phi, r=res.r, knob=ch.knob,
                       shape=ch.shape, amount=ch.amount, p=p, N=int(ok.sum()),
                       true_first=t_first, true_last=t_last, true_change=t_last - t_first)
            for name, (e_first, e_last, e_change) in est.items():
                for which, e, t in (("first", e_first, t_first), ("last", e_last, t_last),
                                    ("change", e_change, t_last - t_first)):
                    err = np.asarray(e, float) - t
                    err = err[np.isfinite(err)]
                    row[f"{name}_{which}_bias"] = float(err.mean()) if err.size else np.nan
                    row[f"{name}_{which}_rmse"] = float(np.sqrt(np.mean(err**2))) if err.size else np.nan
            rows.append(row)
    return rows
