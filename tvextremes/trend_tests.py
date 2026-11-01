"""Conventional trend tests for extremes, as baselines for the power study.

* ``fit_gev_trend`` / ``trend_lr_test``: non-stationary GEV with a linear trend in the location,
  mu_t = mu0 + mu1 (u_t - 1/2) with u_t = t/(T-1), and sigma, xi constant (Coles 2001, ch. 6;
  the standard approach in climate science, e.g. extRemes::fevd with location.fun = ~t). Fitted to
  the k largest per period, so it uses the same data as the Nyblom test. The LR statistic is the
  log-likelihood difference (the package convention); the usual chi^2_1 test rejects when
  2 LR > 3.84.
* ``mann_kendall``: the Mann-Kendall rank test for a monotone trend in the per-period maxima.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from . import likelihood as lk
from .mle import XI_MAX, XI_MIN, FitResult, fit_local
from .pseudo import time_index

_BIG = 1e12


@dataclass
class TrendFit:
    mu0: float
    mu1: float          # change in location from the first to the last period
    sigma: float
    xi: float
    loglik: float
    success: bool


def fit_gev_trend(X, start: FitResult | None = None):
    """MLE of the linear-location-trend GEV_k model from X (T, k), rows descending.

    start: a constant-parameter fit to start from (mu1 = 0). Fitted with fit_local if omitted.
    """
    X = lk._klargest_array(X)
    T = X.shape[0]
    c = time_index(T) - 0.5
    if start is None:
        start = fit_local(X, (float(X[:, 0].mean()), float(max(X[:, 0].std(), 1e-3)), 0.1))
    cpoint = X[:, -1]

    def unpack(th):
        xi, logs, mu0, mu1 = th
        return mu0 + mu1 * c, np.exp(logs), xi

    def fun_grad(th):
        if not XI_MIN < th[0] <= XI_MAX:
            return _BIG, np.zeros(4)
        mu, sigma, xi = unpack(th)
        ll = float(np.sum(lk._core_logpdf(X, cpoint, mu, sigma, xi)))
        if not np.isfinite(ll):
            return _BIG, np.zeros(4)
        s = lk._core_score(X, cpoint, mu, sigma, xi)        # (T, 3): xi, sigma, mu
        g = np.array([s[:, 0].sum(), sigma * s[:, 1].sum(), s[:, 2].sum(), (s[:, 2] * c).sum()])
        if not np.all(np.isfinite(g)):
            return _BIG, np.zeros(4)
        return -ll, -g

    th0 = np.array([min(max(start.xi, XI_MIN + 0.01), XI_MAX - 0.01), np.log(start.sigma),
                    start.mu, 0.0])
    if fun_grad(th0)[0] >= _BIG:
        return TrendFit(np.nan, np.nan, np.nan, np.nan, -np.inf, False)
    res = minimize(fun_grad, th0, jac=True, method="L-BFGS-B",
                   bounds=[(XI_MIN, XI_MAX), (None, None), (None, None), (None, None)],
                   options=dict(maxiter=500, ftol=1e-13, gtol=1e-7))
    x, f = res.x, res.fun
    _, g = fun_grad(x)
    if not res.success or f >= _BIG or np.max(np.abs(g) * (1 + np.abs(x))) > 1e-3:
        nm = minimize(lambda th: fun_grad(th)[0], x if f < _BIG else th0, method="Nelder-Mead",
                      options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000, maxfev=40000))
        if nm.fun < f:
            x, f = nm.x, nm.fun
    xi, logs, mu0, mu1 = x
    return TrendFit(float(mu0), float(mu1), float(np.exp(logs)), float(xi), float(-f), bool(f < _BIG))


def trend_lr_test(X, constant: FitResult, alpha=0.05):
    """(LR, mu1_hat, reject) for H0: no linear trend in the location, chi^2_1 critical value."""
    tr = fit_gev_trend(X, start=constant)
    lr = max(tr.loglik - constant.loglik, 0.0) if tr.success else np.nan
    return lr, tr.mu1, bool(2 * lr > chi2.ppf(1 - alpha, 1))


def mann_kendall(x):
    """(Z, two-sided p-value) of the Mann-Kendall test, with continuity correction and no ties."""
    x = np.asarray(x, float)
    n = x.size
    i, j = np.triu_indices(n, k=1)
    s = float(np.sign(x[j] - x[i]).sum())
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if var > 0 else 0.0
    return z, float(2 * norm.sf(abs(z)))
