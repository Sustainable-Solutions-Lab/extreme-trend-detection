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


@dataclass
class CovariateFit:
    beta: np.ndarray    # (3, 2): rows mu, log sigma, xi; columns intercept, slope on c
    c: np.ndarray       # the covariate used, (T,)
    loglik: float
    success: bool

    def params(self, c=None):
        """(mu_t, sigma_t, xi_t) at covariate values c (default: those fitted)."""
        c = self.c if c is None else np.asarray(c, float)
        mu = self.beta[0, 0] + self.beta[0, 1] * c
        sigma = np.exp(self.beta[1, 0] + self.beta[1, 1] * c)
        xi = self.beta[2, 0] + self.beta[2, 1] * c
        return mu, sigma, xi


def fit_gev_covariate(X, c, start: FitResult | None = None, vary=("mu", "sigma", "xi")):
    """MLE of the GEV_k model with mu_t, log sigma_t and xi_t each linear in a covariate c_t.

    With c_t = time this is a linear trend in all three parameters; with c_t an estimate of the
    mean state (e.g. a smoothed local or global mean temperature) the parameters are functions of
    that state. ``vary`` selects which parameters get a slope (the others are constant).
    c is centred and scaled internally for the optimizer; the returned beta is on the scale of c.
    """
    X = lk._klargest_array(X)
    T = X.shape[0]
    c = np.asarray(c, float)
    c0, cs = float(c.mean()), float(c.std()) or 1.0
    cc = (c - c0) / cs
    if start is None:
        start = fit_local(X, (float(X[:, 0].mean()), float(max(X[:, 0].std(), 1e-3)), 0.1))
    cpoint = X[:, -1]
    free = [v in vary for v in ("mu", "sigma", "xi")]

    def unpack(th):
        b = np.array([[th[0], th[3] if free[0] else 0.0],
                      [th[1], th[4] if free[1] else 0.0],
                      [th[2], th[5] if free[2] else 0.0]])
        return b[0, 0] + b[0, 1] * cc, np.exp(b[1, 0] + b[1, 1] * cc), b[2, 0] + b[2, 1] * cc

    def fun_grad(th):
        mu, sigma, xi = unpack(th)
        if np.any(xi <= XI_MIN) or np.any(xi > XI_MAX):
            return _BIG, np.zeros(6)
        ll = float(np.sum(lk._core_logpdf(X, cpoint, mu, sigma, xi)))
        if not np.isfinite(ll):
            return _BIG, np.zeros(6)
        s = lk._core_score(X, cpoint, mu, sigma, xi)        # (T, 3): xi, sigma, mu
        gm, gs, gx = s[:, 2], s[:, 1] * sigma, s[:, 0]
        g = np.array([gm.sum(), gs.sum(), gx.sum(), (gm * cc).sum() * free[0],
                      (gs * cc).sum() * free[1], (gx * cc).sum() * free[2]])
        if not np.all(np.isfinite(g)):
            return _BIG, np.zeros(6)
        return -ll, -g

    th0 = np.array([start.mu, np.log(start.sigma),
                    min(max(start.xi, XI_MIN + 0.01), XI_MAX - 0.01), 0.0, 0.0, 0.0])
    if fun_grad(th0)[0] >= _BIG:
        return CovariateFit(np.full((3, 2), np.nan), c, -np.inf, False)
    res = minimize(fun_grad, th0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=1000, ftol=1e-13, gtol=1e-7))
    x, f = res.x, res.fun
    _, g = fun_grad(x)
    if not res.success or f >= _BIG or np.max(np.abs(g) * (1 + np.abs(x))) > 1e-3:
        nm = minimize(lambda th: fun_grad(th)[0], x if f < _BIG else th0, method="Nelder-Mead",
                      options=dict(xatol=1e-9, fatol=1e-11, maxiter=40000, maxfev=80000))
        if nm.fun < f:
            x, f = nm.x, nm.fun
    # back to the scale of c: a + b (c - c0)/cs = (a - b c0/cs) + (b/cs) c
    b = np.array([[x[0], x[3] * free[0]], [x[1], x[4] * free[1]], [x[2], x[5] * free[2]]])
    beta = np.column_stack([b[:, 0] - b[:, 1] * c0 / cs, b[:, 1] / cs])
    return CovariateFit(beta, c, float(-f), bool(f < _BIG))
