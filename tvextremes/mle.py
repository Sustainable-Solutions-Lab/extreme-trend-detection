"""Maximum likelihood for constant-parameter panels of extremes, with the paper's null restrictions.

Restrictions (paper §2.2.3), selected by keyword:

* ``fix_xi=xi0``            H0: xi = xi0
* ``quantile=(q0, p)``      H0: p-quantile of GEV_1 equals q0 (the paper uses p = 0.9)
* ``pareto=True``           H0: mu = sigma/xi with xi >= eps (pure Pareto tail; eps = 0.03)
* ``pareto=True, fix_xi=1`` H0: mu = sigma/xi and xi = 1 (Zipf's law)

The unrestricted space is Theta_1 = {sigma > 0, xi > -0.99}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import likelihood as lk

XI_MIN = -0.99      # lower bound on xi in Theta_1
PARETO_EPS = 0.03   # xi >= eps under the Pareto null
_START_XI = (-0.2, 0.1, 0.4, 0.8, 1.2)


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
    """Maps a free vector theta to (mu, sigma, xi) under a given restriction."""

    def __init__(self, fix_xi=None, pareto=False, quantile=None):
        if quantile is not None and pareto:
            raise ValueError("quantile and pareto restrictions cannot be combined")
        self.fix_xi, self.pareto, self.quantile = fix_xi, pareto, quantile
        self.xi_lo = PARETO_EPS if pareto else XI_MIN

    def to_gev(self, theta):
        theta = list(theta)
        xi = self.fix_xi if self.fix_xi is not None else theta.pop(0)
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
        theta = [] if self.fix_xi is not None else [max(xi, self.xi_lo + 0.01)]
        theta.append(np.log(sigma))
        if not self.pareto and self.quantile is None:
            theta.append(mu)
        return np.array(theta, float)

    def feasible(self, xi):
        return xi > self.xi_lo if not self.pareto else xi >= self.xi_lo


def _maximize(loglik_fn, par, starts):
    """Multi-start Nelder-Mead on -loglik, followed by restarts from the best point."""

    def negll(theta):
        mu, sigma, xi = par.to_gev(theta)
        if not par.feasible(xi) or not np.isfinite(mu):
            return 1e12
        ll = loglik_fn(mu, sigma, xi)
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for mu, sigma, xi in starts:
        # Inflate sigma until the starting point is inside the support.
        for _ in range(30):
            theta0 = par.from_gev(mu, sigma, xi)
            if negll(theta0) < 1e12:
                break
            sigma *= 2.0
        else:
            continue
        if len(theta0) == 0:            # nothing free (cannot happen with current options)
            res_x, res_f = theta0, negll(theta0)
        else:
            res = minimize(negll, theta0, method="Nelder-Mead",
                           options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000, maxfev=40000))
            res_x, res_f = res.x, res.fun
        if best is None or res_f < best[1]:
            best = (res_x, res_f)
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
    return FitResult(float(mu), float(sigma), float(xi), float(-f), bool(f < 1e12))


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


def lr_statistic(unrestricted: FitResult, restricted: FitResult) -> float:
    """Generalized LR statistic of paper eq. (6): sup ln L over Theta_1 minus sup over Theta_0.

    Note the paper's LR is the log-likelihood *difference* (no factor of 2).
    """
    return unrestricted.loglik - restricted.loglik
