"""Nyblom (1989) test of parameter constancy, paper eq. (12).

    L_T = T^-2 sum_t (sum_{l>=t} S_l)' V^-1 (sum_{l>=t} S_l)

S_t are per-period scores at the MLE, V = -T^-1 sum_t dS_t/dtheta' (average information from the
Hessian, as in the authors' code). Under H0 and T -> infinity, L_T converges to the integral of a
squared Brownian bridge; the paper instead uses small-sample critical values that depend on xi_hat
(see ``critical_values``).
"""

from __future__ import annotations

import numpy as np

from . import likelihood as lk


def nyblom_statistic(scores, info):
    """L_T from a (T, p) matrix of scores and a (p, p) average information matrix."""
    scores = np.asarray(scores, float)
    T = scores.shape[0]
    vinv = np.linalg.pinv(info, hermitian=True)
    partial = np.cumsum(scores[::-1], axis=0)        # sum_{l >= t} S_l
    return float(np.einsum("ti,ij,tj->", partial, vinv, partial) / T**2)


def _score_fn(data, tau):
    if tau is None:
        X = lk._klargest_array(data)
        return lambda m, s, x: lk._core_score(X, X[:, -1], m, s, x)
    Y, tau_v = lk._exceedance_arrays(data, tau)
    return lambda m, s, x: lk._core_score(Y, tau_v, m, s, x)


def information_hessian(data, mu, sigma, xi, tau=None, step=None):
    """Average information -T^-1 sum_t dS_t/dtheta', by central differences of the analytic score.

    Parameter order (xi, sigma, mu), matching the scores.
    """
    score = _score_fn(data, tau)
    theta = np.array([xi, sigma, mu], float)
    eps = np.finfo(float).eps ** (1 / 3) if step is None else step
    H = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = max(abs(theta[i]) * 1e-4, eps) if step is None else step
        sp = score(theta[2] + d[2], theta[1] + d[1], theta[0] + d[0]).mean(axis=0)
        sm = score(theta[2] - d[2], theta[1] - d[1], theta[0] - d[0]).mean(axis=0)
        H[:, i] = (sp - sm) / (2 * d[i])
    info = -0.5 * (H + H.T)
    return info


def nyblom_test(data, fit, tau=None, demean=True, interior_tol=None):
    """Nyblom L_T at a fitted constant-parameter model.

    fit: object with mu, sigma, xi (e.g. ``FitResult``).
    demean: subtract the average score first (the authors' MATLAB code does this; at an exact
        interior MLE it changes nothing).
    interior_tol: if given, return 0 when T * Sbar' V^-1 Sbar exceeds it, i.e. the fit is not an
        interior maximum (the authors' Fortran uses 0.1 in its simulations).
    """
    scores = _score_fn(data, tau)(fit.mu, fit.sigma, fit.xi)
    info = information_hessian(data, fit.mu, fit.sigma, fit.xi, tau=tau)
    if interior_tol is not None:
        sbar = scores.sum(axis=0)
        T = scores.shape[0]
        if sbar @ np.linalg.pinv(info, hermitian=True) @ sbar / T > interior_tol:
            return 0.0
    if demean:
        scores = scores - scores.mean(axis=0)
    return nyblom_statistic(scores, info)
