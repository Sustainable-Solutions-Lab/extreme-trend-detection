"""Likelihoods and scores for panels of extremes (Müller & Watson 2024, "Time Varying Extremes").

Two observation schemes, both governed by GEV parameters (mu, sigma, xi):

* **k largest** per period: ``X`` has shape (T, k), each row sorted descending.
  Joint GEV_k density, paper eq. (4):

      ln f = -k ln(sigma) - exp(h(z_k)) + (1 + xi) * sum_j h(z_j)

* **Exceedances** of a threshold ``tau_t``: ``Y`` has shape (T, k_max), padded
  with NaN; the number of non-NaN entries in row t is k_t. Paper eq. (7):

      ln f = -ln(k_t!) - k_t ln(sigma) - exp(h(z_tau)) + (1 + xi) * sum_j h(z_j)

where z = (x - mu) / sigma and h(z) = ln[(1 + xi z)^(-1/xi)] = -log1p(xi z) / xi
(h(z) = -z when xi = 0). Both share the same core, with the "censoring point"
being x_k for the k-largest case and tau_t for exceedances.

Parameters may be scalars or arrays of length T (time-varying), which the
time-varying model needs later. Every log-density returns one value per period;
sum it for the panel log-likelihood. Values outside the support give -inf.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

# Below this |xi| the xi -> 0 (Gumbel) series expansions are used.
_XI_SERIES = 1e-6


# ---------------------------------------------------------------------------
# Core functions h(z) and dh/dxi
# ---------------------------------------------------------------------------

def _h(z, xi):
    """h(z) = -log1p(xi z)/xi, with the xi -> 0 limit -z. NaN where 1 + xi z <= 0."""
    z, xi = np.broadcast_arrays(np.asarray(z, float), np.asarray(xi, float))
    small = np.abs(xi) < _XI_SERIES
    xi_safe = np.where(small, 1.0, xi)
    u = xi * z
    with np.errstate(invalid="ignore", divide="ignore"):
        exact = -np.log1p(np.where(1.0 + u > 0, u, np.nan)) / xi_safe
    series = -z + xi * z**2 / 2 - xi**2 * z**3 / 3
    return np.where(small, series, exact)


def _dh_dxi(z, xi, h):
    """Partial derivative of h(z) with respect to xi, holding z fixed."""
    z, xi = np.broadcast_arrays(np.asarray(z, float), np.asarray(xi, float))
    small = np.abs(xi) < _XI_SERIES
    xi_safe = np.where(small, 1.0, xi)
    with np.errstate(invalid="ignore", divide="ignore"):
        exact = -h / xi_safe - (z / xi_safe) / (1.0 + xi * z)
    series = z**2 / 2 - 2 * xi * z**3 / 3 + 3 * xi**2 * z**4 / 4
    return np.where(small, series, exact)


def h_extended(z, xi):
    """Extended-support h~(z) of paper §3.2.3 (linear extrapolation where 1 + xi z < chi).

    chi = exp(-6|xi|). Defined on all of R; equals h(z) inside the support region
    1 + xi z >= chi. Used for HMC; reweight by f/f~ afterwards.
    """
    z, xi = np.broadcast_arrays(np.asarray(z, float), np.asarray(xi, float))
    chi = np.exp(-6.0 * np.abs(xi))
    small = np.abs(xi) < _XI_SERIES
    xi_safe = np.where(small, 1.0, xi)
    outside = (1.0 + xi * z < chi) & ~small
    lin = -np.log(chi) / xi_safe - (z - (chi - 1.0) / xi_safe) / chi
    return np.where(outside, lin, _h(np.where(outside, 0.0, z), xi))


# ---------------------------------------------------------------------------
# Shared log-density / score core
# ---------------------------------------------------------------------------

def _as_period_params(mu, sigma, xi, T):
    return (np.broadcast_to(np.asarray(mu, float), (T,)),
            np.broadcast_to(np.asarray(sigma, float), (T,)),
            np.broadcast_to(np.asarray(xi, float), (T,)))


def _core_logpdf(obs, cpoint, mu, sigma, xi, extended=False):
    """Per-period log density: -k ln s - exp(h(z_c)) + (1+xi) sum_j h(z_j).

    obs: (T, kmax) with NaN padding; cpoint: (T,) censoring point.
    Does not include the -ln(k!) term.
    """
    T = obs.shape[0]
    mu, sigma, xi = _as_period_params(mu, sigma, xi, T)
    hfun = h_extended if extended else _h
    valid = ~np.isnan(obs)
    k = valid.sum(axis=1)
    zc = (cpoint - mu) / sigma
    z = (np.where(valid, obs, 0.0) - mu[:, None]) / sigma[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        hz = np.where(valid, hfun(z, xi[:, None]), 0.0)
        hc = hfun(zc, xi)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        ll = -k * np.log(sigma) - np.exp(hc) + (1.0 + xi) * np.sum(hz, axis=1)
    bad = (sigma <= 0) | np.isnan(ll)
    return np.where(bad, -np.inf, ll)


def _core_score(obs, cpoint, mu, sigma, xi):
    """Per-period score d ln f / d(xi, sigma, mu), shape (T, 3).

    For the k-largest case x_k appears both as the censoring point and in the sum;
    the two contributions are additive, so the same code serves both schemes.
    """
    T = obs.shape[0]
    mu, sigma, xi = _as_period_params(mu, sigma, xi, T)
    valid = ~np.isnan(obs)
    k = valid.sum(axis=1)
    x = np.where(valid, obs, 0.0)
    xi2, s2, mu2 = xi[:, None], sigma[:, None], mu[:, None]

    # Terms from the sum over observations: (1+xi) sum_j h(z_j)
    z = (x - mu2) / s2
    hz = _h(z, xi2)
    d = 1.0 / (1.0 + xi2 * z)                       # -dh/dz
    dldz = np.where(valid, -(1.0 + xi2) * d, 0.0)  # d/dz_j of (1+xi) h(z_j)
    g_xi = np.where(valid, hz + (1.0 + xi2) * _dh_dxi(z, xi2, hz), 0.0).sum(axis=1)
    g_sigma = (-z * dldz / s2).sum(axis=1)
    g_mu = (-dldz / s2).sum(axis=1)

    # Terms from -exp(h(z_c))
    zc = (cpoint - mu) / sigma
    hc = _h(zc, xi)
    ec = np.exp(hc)
    dldzc = ec / (1.0 + xi * zc)                   # d/dz_c of -exp(h(z_c))
    g_xi = g_xi - ec * _dh_dxi(zc, xi, hc)
    g_sigma = g_sigma - zc * dldzc / sigma - k / sigma
    g_mu = g_mu - dldzc / sigma
    return np.column_stack([g_xi, g_sigma, g_mu])


# ---------------------------------------------------------------------------
# Public API: k largest observations
# ---------------------------------------------------------------------------

def _klargest_array(X):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if np.isnan(X).any():
        raise ValueError("k-largest data must not contain NaN; use exceedance functions for ragged data")
    if np.any(np.diff(X, axis=1) > 0):
        raise ValueError("each row of X must be sorted in descending order")
    return X


def gev_logpdf(X, mu, sigma, xi, extended=False):
    """Per-period log density of the k largest observations, GEV_k(mu, sigma, xi). Eq. (4).

    X: (T, k) array, rows sorted descending (or a length-T vector for k = 1).
    mu, sigma, xi: scalars or length-T arrays.
    extended: use the extended-support likelihood of §3.2.3.
    Returns an array of shape (T,).
    """
    X = _klargest_array(X)
    return _core_logpdf(X, X[:, -1], mu, sigma, xi, extended=extended)


def gev_loglik(X, mu, sigma, xi):
    """Panel log-likelihood sum_t ln f_GEV(X_t | mu, sigma, xi)."""
    with np.errstate(over="ignore", invalid="ignore"):   # far outside the support -> -inf
        return float(np.sum(gev_logpdf(X, mu, sigma, xi)))


def gev_score(X, mu, sigma, xi):
    """Per-period analytic score with respect to (xi, sigma, mu); shape (T, 3)."""
    X = _klargest_array(X)
    return _core_score(X, X[:, -1], mu, sigma, xi)


# ---------------------------------------------------------------------------
# Public API: exceedances of a threshold
# ---------------------------------------------------------------------------

def _exceedance_arrays(Y, tau):
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T = Y.shape[0]
    tau = np.broadcast_to(np.asarray(tau, float), (T,))
    if np.any(Y < tau[:, None]):
        raise ValueError("exceedances must be >= their threshold tau_t")
    return Y, tau


def exceedance_logpdf(Y, tau, mu, sigma, xi, extended=False):
    """Per-period log density of the exceedances of tau_t, in GEV parameterization. Eq. (7).

    Y: (T, k_max) array of exceedances padded with NaN (order within a row is irrelevant).
    tau: scalar or length-T threshold.
    Returns an array of shape (T,), including the -ln(k_t!) term.
    """
    Y, tau = _exceedance_arrays(Y, tau)
    k = (~np.isnan(Y)).sum(axis=1)
    return _core_logpdf(Y, tau, mu, sigma, xi, extended=extended) - gammaln(k + 1.0)


def exceedance_loglik(Y, tau, mu, sigma, xi):
    """Panel log-likelihood for exceedances."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.sum(exceedance_logpdf(Y, tau, mu, sigma, xi)))


def exceedance_score(Y, tau, mu, sigma, xi):
    """Per-period analytic score with respect to (xi, sigma, mu); shape (T, 3)."""
    Y, tau = _exceedance_arrays(Y, tau)
    return _core_score(Y, tau, mu, sigma, xi)


# ---------------------------------------------------------------------------
# Parameter mappings
# ---------------------------------------------------------------------------

def gev_quantile(p, mu, sigma, xi):
    """p-quantile of GEV_1(mu, sigma, xi); e.g. q_0.9 used throughout the paper."""
    y = -np.log(p)
    xi = np.asarray(xi, float)
    small = np.abs(xi) < _XI_SERIES
    xi_safe = np.where(small, 1.0, xi)
    factor = np.where(small, -np.log(y), (y ** (-xi_safe) - 1.0) / xi_safe)
    return mu + sigma * factor


def gev_mu_from_quantile(q, p, sigma, xi):
    """Location mu such that the p-quantile of GEV_1(mu, sigma, xi) equals q."""
    return q - (gev_quantile(p, 0.0, sigma, xi))


def gev_to_pgp(mu, sigma, xi, tau):
    """GEV (mu, sigma, xi) -> Poisson-GP (lambda, psi) at threshold tau. Paper eq. (8).

    lambda = expected number of exceedances of tau; psi = GP scale of the excesses.
    """
    psi = sigma + xi * (tau - mu)
    lam = np.exp(_h((tau - mu) / sigma, xi))
    return lam, psi


def pgp_to_gev(lam, psi, xi, tau):
    """Poisson-GP (lambda, psi) at threshold tau -> GEV (mu, sigma)."""
    lam, psi, xi = (np.asarray(a, float) for a in (lam, psi, xi))
    small = np.abs(xi) < _XI_SERIES
    xi_safe = np.where(small, 1.0, xi)
    sigma = psi * lam ** xi
    mu = np.where(small, tau + psi * np.log(lam), tau + (sigma - psi) / xi_safe)
    return mu, sigma


def gp_to_gev(nu, omega, xi, n):
    """Underlying GP tail (nu, omega, xi) and sample size n -> GEV (mu, sigma). Paper eq. (3)."""
    xi = np.asarray(xi, float)
    small = np.abs(xi) < _XI_SERIES
    xi_safe = np.where(small, 1.0, xi)
    sigma = omega * n ** xi
    mu = np.where(small, nu + omega * np.log(n), nu + omega * (n ** xi_safe - 1.0) / xi_safe)
    return mu, sigma


def tvp_to_gev(xi, alpha, s, m):
    """TVP-GEV parameters (xi, alpha, s, m) -> GEV (mu, sigma). Paper eq. (14).

    sigma = s alpha^xi, mu = m + s (alpha^xi - 1)/xi. alpha is sample size relative to n0.
    """
    return gp_to_gev(m, s, xi, alpha)
