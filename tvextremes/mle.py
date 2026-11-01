"""Maximum likelihood for constant-parameter panels of extremes, with the paper's null restrictions.

Restrictions (paper §2.2.3), selected by keyword:

* ``fix_xi=xi0``            H0: xi = xi0
* ``quantile=(q0, p)``      H0: p-quantile of GEV_1 equals q0 (the paper uses p = 0.9)
* ``pareto=True``           H0: mu = sigma/xi (pure Pareto tail)
* ``pareto=True, fix_xi=1`` H0: mu = sigma/xi and xi = 1 (Zipf's law)

The unrestricted space is Theta_1 = {sigma > 0, xi > -0.99}. Under the Pareto null the paper
restricts xi >= eps = 0.03; in the authors' code that value is the lower end of the xi grid used
for size control, while the restricted fit itself is bounded at xi >= 0.01 (Fortran) or not at all
(MATLAB). We follow the Fortran code, which produced the published critical values.

Two fitting routes:

* ``fit_gev`` / ``fit_exceedance``: robust multi-start Nelder-Mead, for real data.
* ``fit_local``: a single gradient-based (L-BFGS-B) fit from a given start, with Nelder-Mead
  fallback. Much faster; used for Monte Carlo where the true parameters are a good start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import likelihood as lk

XI_MIN = -0.99        # lower bound on xi in Theta_1
XI_MAX = 10.0         # upper bound used by the authors' Fortran optimizer
PARETO_XI_LB = 0.01   # lower bound on xi for the restricted fit under mu = sigma/xi
_START_XI = (-0.2, 0.1, 0.4, 0.8, 1.2)
_BIG = 1e12


@dataclass
class FitResult:
    mu: float
    sigma: float
    xi: float
    loglik: float
    success: bool

    @property
    def params(self):
        return self.mu, self.sigma, self.xi

    def quantile(self, p=0.9):
        return float(lk.gev_quantile(p, self.mu, self.sigma, self.xi))


class _Parametrization:
    """Maps a free vector theta to (mu, sigma, xi) under a given restriction, with its Jacobian."""

    def __init__(self, fix_xi=None, pareto=False, quantile=None):
        if quantile is not None and pareto:
            raise ValueError("quantile and pareto restrictions cannot be combined")
        self.fix_xi, self.pareto, self.quantile = fix_xi, pareto, quantile
        self.xi_lo = PARETO_XI_LB if pareto else XI_MIN
        self.free_xi = fix_xi is None
        self.free_mu = not pareto and quantile is None

    def to_gev(self, theta):
        theta = list(theta)
        xi = theta.pop(0) if self.free_xi else self.fix_xi
        sigma = np.exp(theta.pop(0))
        if self.pareto:
            mu = sigma / xi
        elif self.quantile is not None:
            q0, p = self.quantile
            mu = float(lk.gev_mu_from_quantile(q0, p, sigma, xi))
        else:
            mu = theta.pop(0)
        return mu, sigma, xi

    def from_gev(self, mu, sigma, xi):
        theta = [min(max(xi, self.xi_lo + 0.01), XI_MAX - 0.01)] if self.free_xi else []
        theta.append(np.log(sigma))
        if self.free_mu:
            theta.append(mu)
        return np.array(theta, float)

    def bounds(self):
        b = [(self.xi_lo, XI_MAX)] if self.free_xi else []
        b.append((None, None))
        if self.free_mu:
            b.append((None, None))
        return b

    def feasible(self, xi):
        return self.xi_lo <= xi <= XI_MAX if self.pareto else self.xi_lo < xi <= XI_MAX

    def chain(self, theta, g):
        """Gradient wrt theta, given g = d loglik / d(xi, sigma, mu)."""
        mu, sigma, xi = self.to_gev(theta)
        g_xi, g_sigma, g_mu = g
        if self.pareto:                   # mu = sigma/xi
            d_xi = g_xi - g_mu * sigma / xi**2
            d_logs = sigma * (g_sigma + g_mu / xi)
        elif self.quantile is not None:   # mu = q0 - sigma c(xi)
            q0, p = self.quantile
            y = -np.log(p)
            c = float(lk.gev_quantile(p, 0.0, 1.0, xi))
            if abs(xi) < 1e-6:
                dc = np.log(y) ** 2 / 2
            else:
                dc = (-np.log(y) * y ** (-xi) * xi - (y ** (-xi) - 1)) / xi**2
            d_xi = g_xi - g_mu * sigma * dc
            d_logs = sigma * (g_sigma - g_mu * c)
        else:
            d_xi = g_xi
            d_logs = sigma * g_sigma
        out = [d_xi] if self.free_xi else []
        out.append(d_logs)
        if self.free_mu:
            out.append(g_mu)
        return np.array(out, float)


def _negll_fn(loglik_fn, par):
    def negll(theta):
        mu, sigma, xi = par.to_gev(theta)
        if not par.feasible(xi) or not np.isfinite(mu):
            return _BIG
        ll = loglik_fn(mu, sigma, xi)
        return -ll if np.isfinite(ll) else _BIG
    return negll


def _maximize(loglik_fn, par, starts):
    """Multi-start Nelder-Mead on -loglik, followed by restarts from the best point."""
    negll = _negll_fn(loglik_fn, par)
    best = None
    for mu, sigma, xi in starts:
        # Inflate sigma until the starting point is inside the support.
        for _ in range(30):
            theta0 = par.from_gev(mu, sigma, xi)
            if negll(theta0) < _BIG:
                break
            sigma *= 2.0
        else:
            continue
        res = minimize(negll, theta0, method="Nelder-Mead",
                       options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000, maxfev=40000))
        if best is None or res.fun < best[1]:
            best = (res.x, res.fun)
    if best is None:
        raise RuntimeError("no feasible starting value found")

    # Restart from the best point until no further improvement (guards against NM stalling).
    x, f = best
    for _ in range(5):
        res = minimize(negll, x, method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000, maxfev=40000))
        if res.fun >= f - 1e-10:
            break
        x, f = res.x, res.fun
    mu, sigma, xi = par.to_gev(x)
    return FitResult(float(mu), float(sigma), float(xi), float(-f), bool(f < _BIG))


def _gumbel_start(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    sd = values.std() if values.size > 1 else max(abs(values[0]), 1.0)
    sigma = max(sd, 1e-8) * np.sqrt(6) / np.pi
    return values.mean() - 0.5772 * sigma, sigma


def fit_gev(X, fix_xi=None, pareto=False, quantile=None, starts=None):
    """MLE of (mu, sigma, xi) from the k largest observations per period. X: (T, k), rows descending."""
    X = lk._klargest_array(X)
    par = _Parametrization(fix_xi, pareto, quantile)
    if starts is None:
        mu0, s0 = _gumbel_start(X[:, 0])
        sd_all = X.std() if X.size > 1 else s0
        starts = [(mu0, max(s0, sd_all * 0.5), xi) for xi in _START_XI]
    return _maximize(lambda m, s, x: lk.gev_loglik(X, m, s, x), par, starts)


def fit_exceedance(Y, tau, fix_xi=None, pareto=False, quantile=None, starts=None):
    """MLE of GEV (mu, sigma, xi) from threshold exceedances. Y: (T, k_max) NaN-padded; tau scalar or (T,)."""
    Y, tau = lk._exceedance_arrays(Y, tau)
    par = _Parametrization(fix_xi, pareto, quantile)
    if starts is None:
        counts = (~np.isnan(Y)).sum(axis=1)
        if counts.sum() == 0:
            raise ValueError("no exceedances")
        lam0 = counts.mean()
        tau_bar = tau.mean()
        mean_excess = np.nanmean(Y - tau[:, None])
        starts = []
        for xi in _START_XI:
            psi = mean_excess * (1.0 - xi) if xi < 0.9 else mean_excess * 0.1
            mu, sigma = lk.pgp_to_gev(lam0, psi, xi, tau_bar)
            starts.append((float(mu), float(sigma), xi))
    return _maximize(lambda m, s, x: lk.exceedance_loglik(Y, tau, m, s, x), par, starts)


def fit_local(data, start, tau=None, fix_xi=None, pareto=False, quantile=None):
    """Fast single-start MLE: L-BFGS-B with analytic gradient, Nelder-Mead fallback.

    data: X (k largest) if tau is None, otherwise exceedances Y with threshold tau.
    start: (mu, sigma, xi) starting point, e.g. the true parameters in a simulation.
    """
    if tau is None:
        X = lk._klargest_array(data)
        ll_fn = lambda m, s, x: float(np.sum(lk._core_logpdf(X, X[:, -1], m, s, x)))
        sc_fn = lambda m, s, x: lk._core_score(X, X[:, -1], m, s, x).sum(axis=0)
        const = 0.0
    else:
        Y, tau_v = lk._exceedance_arrays(data, tau)
        const = -float(np.sum(lk.gammaln((~np.isnan(Y)).sum(axis=1) + 1.0)))
        ll_fn = lambda m, s, x: float(np.sum(lk._core_logpdf(Y, tau_v, m, s, x)))
        sc_fn = lambda m, s, x: lk._core_score(Y, tau_v, m, s, x).sum(axis=0)
    par = _Parametrization(fix_xi, pareto, quantile)
    negll = _negll_fn(ll_fn, par)

    def fun_grad(theta):
        f = negll(theta)
        if f >= _BIG:
            return f, np.zeros_like(theta)
        mu, sigma, xi = par.to_gev(theta)
        g = par.chain(theta, sc_fn(mu, sigma, xi))
        if not np.all(np.isfinite(g)):
            return _BIG, np.zeros_like(theta)
        return f, -g

    mu0, s0, xi0 = start
    theta0 = None
    for _ in range(30):                     # inflate sigma until feasible
        cand = par.from_gev(mu0, s0, xi0)
        if negll(cand) < _BIG:
            theta0 = cand
            break
        s0 *= 2.0
    if theta0 is None:
        return FitResult(np.nan, np.nan, np.nan, -np.inf, False)

    res = minimize(fun_grad, theta0, jac=True, method="L-BFGS-B", bounds=par.bounds(),
                   options=dict(maxiter=500, ftol=1e-13, gtol=1e-7))
    x, f = res.x, res.fun
    _, g = fun_grad(x)
    # Fallback when L-BFGS-B stalls (e.g. against the support boundary).
    if not res.success or f >= _BIG or np.max(np.abs(g) * (1 + np.abs(x))) > 1e-3:
        nm = minimize(negll, x if f < _BIG else theta0, method="Nelder-Mead",
                      options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000, maxfev=40000))
        if nm.fun < f:
            x, f = nm.x, nm.fun
    mu, sigma, xi = par.to_gev(x)
    return FitResult(float(mu), float(sigma), float(xi), float(-f + const), bool(f < _BIG))


def lr_statistic(unrestricted: FitResult, restricted: FitResult) -> float:
    """Generalized LR statistic of paper eq. (6): sup ln L over Theta_1 minus sup over Theta_0.

    Note the paper's LR is the log-likelihood *difference* (no factor of 2).
    """
    return unrestricted.loglik - restricted.loglik
