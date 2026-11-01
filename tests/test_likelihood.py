"""Self-contained checks of the likelihood module (no external data needed)."""

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.optimize import approx_fprime

from tvextremes import likelihood as lk
from tvextremes.mle import fit_gev, fit_exceedance

XIS = [-0.4, -1e-9, 0.0, 0.3, 1.2]


def simulate_klargest(T, k, mu, sigma, xi, rng):
    """Draw T vectors of k largest from GEV_k via the representation in paper eq. (2)."""
    S = np.cumsum(rng.exponential(size=(T, k)), axis=1)
    if xi == 0:
        return mu - sigma * np.log(S)
    return mu + sigma * (S ** (-xi) - 1.0) / xi


def simulate_exceedances(T, tau, mu, sigma, xi, rng):
    """Poisson number of GP excesses above tau (paper eq. 8), NaN-padded."""
    lam, psi = lk.gev_to_pgp(mu, sigma, xi, tau)
    k = rng.poisson(lam, size=T)
    Y = np.full((T, max(k.max(), 1)), np.nan)
    for t in range(T):
        Y[t, :k[t]] = tau + stats.genpareto.rvs(xi, scale=psi, size=k[t], random_state=rng)
    return Y


@pytest.mark.parametrize("xi", XIS)
def test_k1_matches_scipy_genextreme(xi):
    x = np.linspace(-1.5, 4.0, 23)
    mu, sigma = 0.3, 1.7
    ref = stats.genextreme.logpdf(x, -xi, loc=mu, scale=sigma)   # scipy's c = -xi
    got = lk.gev_logpdf(x, mu, sigma, xi)
    ok = np.isfinite(ref)
    np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-7, atol=1e-7)
    assert np.all(np.isneginf(got[~ok]))


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.4])
def test_k2_density_integrates_to_one(xi):
    mu, sigma = 0.0, 1.0
    f = lambda x2, x1: np.exp(lk.gev_logpdf(np.array([[x1, x2]]), mu, sigma, xi)[0])
    lo, hi = -6.0, 60.0
    if xi < 0:
        hi = mu - sigma / xi
    if xi > 0:
        lo = mu - sigma / xi
    total, _ = integrate.dblquad(f, lo, hi, lambda x1: lo, lambda x1: x1, epsabs=1e-9)
    if xi >= 0:  # add back the mass with X1 > hi, P = 1 - exp(-exp(h(hi)))
        total += 1 - np.exp(-np.exp(lk._h((hi - mu) / sigma, xi)))
    assert total == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("xi", XIS)
def test_exceedance_equals_poisson_times_gp(xi):
    rng = np.random.default_rng(1)
    mu, sigma, tau = 0.5, 1.3, 1.0
    Y = simulate_exceedances(40, tau, mu, sigma, xi, rng)
    lam, psi = lk.gev_to_pgp(mu, sigma, xi, tau)
    k = (~np.isnan(Y)).sum(axis=1)
    ref = stats.poisson.logpmf(k, lam) + np.nansum(
        stats.genpareto.logpdf(Y - tau, xi, scale=psi), axis=1)
    np.testing.assert_allclose(lk.exceedance_logpdf(Y, tau, mu, sigma, xi), ref, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("xi", XIS)
def test_gev_score_matches_numerical_gradient(xi):
    rng = np.random.default_rng(2)
    X = simulate_klargest(6, 3, 0.2, 1.1, xi, rng)
    theta = np.array([xi, 1.1, 0.2])
    score = lk.gev_score(X, theta[2], theta[1], theta[0])
    for t in range(X.shape[0]):
        f = lambda th: lk.gev_logpdf(X[t:t + 1], th[2], th[1], th[0])[0]
        num = approx_fprime(theta, f, 1e-7)
        np.testing.assert_allclose(score[t], num, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("xi", XIS)
def test_exceedance_score_matches_numerical_gradient(xi):
    rng = np.random.default_rng(3)
    tau = np.linspace(0.8, 1.2, 8)               # time-varying threshold
    Y = np.full((len(tau), 3), np.nan)
    for t in range(len(tau)):
        row = simulate_exceedances(1, tau[t], 0.4, 1.0, xi, rng)[0, :3]
        Y[t, :row.size] = row
    theta = np.array([xi, 1.0, 0.4])
    score = lk.exceedance_score(Y, tau, theta[2], theta[1], theta[0])
    for t in range(len(tau)):
        f = lambda th: lk.exceedance_logpdf(Y[t:t + 1], tau[t], th[2], th[1], th[0])[0]
        num = approx_fprime(theta, f, 1e-7)
        np.testing.assert_allclose(score[t], num, rtol=1e-4, atol=1e-5)


def test_score_has_mean_zero_at_truth():
    rng = np.random.default_rng(4)
    X = simulate_klargest(200_000, 2, 1.0, 2.0, 0.25, rng)
    s = lk.gev_score(X, 1.0, 2.0, 0.25)
    assert np.all(np.abs(s.mean(axis=0)) < 4 * s.std(axis=0) / np.sqrt(len(s)))


def test_time_varying_parameters_broadcast():
    rng = np.random.default_rng(5)
    X = simulate_klargest(5, 2, 0.0, 1.0, 0.2, rng)
    mu = np.linspace(-0.1, 0.1, 5)
    per = lk.gev_logpdf(X, mu, 1.0, 0.2)
    for t in range(5):
        assert per[t] == pytest.approx(lk.gev_logpdf(X[t:t + 1], mu[t], 1.0, 0.2)[0])


@pytest.mark.parametrize("xi", XIS)
def test_parameter_mappings_round_trip(xi):
    mu, sigma, tau = 0.7, 1.9, 2.0
    lam, psi = lk.gev_to_pgp(mu, sigma, xi, tau)
    mu2, sigma2 = lk.pgp_to_gev(lam, psi, xi, tau)
    assert (mu2, sigma2) == pytest.approx((mu, sigma), rel=1e-6, abs=1e-6)
    q = lk.gev_quantile(0.9, mu, sigma, xi)
    assert lk.gev_mu_from_quantile(q, 0.9, sigma, xi) == pytest.approx(mu, abs=1e-6)
    assert stats.genextreme.cdf(q, -xi, loc=mu, scale=sigma) == pytest.approx(0.9, abs=1e-6)


def test_gp_to_gev_matches_order_statistics_of_gp_sample():
    """Paper eq. (3): the maximum of n GP(nu, omega, xi) draws is ~ GEV(mu, sigma, xi)."""
    rng = np.random.default_rng(6)
    nu, omega, xi, n = 1.0, 2.0, 0.3, 2000
    W = stats.genpareto.rvs(xi, loc=nu - 0 * omega, scale=omega, size=(3000, n), random_state=rng)
    mx = W.max(axis=1)
    mu, sigma = lk.gp_to_gev(nu, omega, xi, n)
    ks = stats.kstest(mx, stats.genextreme(-xi, loc=mu, scale=sigma).cdf)
    assert ks.pvalue > 0.01


def test_tvp_to_gev_is_eq14():
    xi, alpha, s, m = 0.4, 1.7, 2.0, 0.5
    mu, sigma = lk.tvp_to_gev(xi, alpha, s, m)
    assert sigma == pytest.approx(s * alpha ** xi)
    assert mu == pytest.approx(m + s * (alpha ** xi - 1) / xi)


@pytest.mark.parametrize("xi", [-0.3, 0.5, 1.5])
def test_extended_h_agrees_inside_support_and_is_continuous(xi):
    z = np.linspace(-3, 3, 101)
    inside = 1 + xi * z > np.exp(-6 * abs(xi))
    np.testing.assert_allclose(lk.h_extended(z[inside], xi), lk._h(z[inside], xi))
    assert np.all(np.isfinite(lk.h_extended(z, xi)))
    zb = (np.exp(-6 * abs(xi)) - 1) / xi           # boundary point
    eps = 1e-9
    slope = np.exp(6 * abs(xi))                     # |dh/dz| = 1/chi at the boundary
    h_b = -np.log(np.exp(-6 * abs(xi))) / xi        # h(zb) = -ln(chi)/xi
    for side in (-1, 1):
        assert lk.h_extended(zb + side * eps, xi) == pytest.approx(h_b, abs=2 * eps * slope + 1e-9)


@pytest.mark.parametrize("xi", [-0.2, 0.3, 0.9])
def test_mle_recovers_parameters_klargest(xi):
    rng = np.random.default_rng(7)
    X = simulate_klargest(400, 3, 2.0, 0.8, xi, rng)
    fit = fit_gev(X)
    assert fit.xi == pytest.approx(xi, abs=0.1)
    assert fit.sigma == pytest.approx(0.8, rel=0.15)
    assert fit.mu == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("xi", [-0.2, 0.3, 0.9])
def test_mle_recovers_parameters_exceedance(xi):
    rng = np.random.default_rng(8)
    Y = simulate_exceedances(1500, 1.0, 0.5, 1.0, xi, rng)
    fit = fit_exceedance(Y, 1.0)
    assert fit.xi == pytest.approx(xi, abs=0.12)
    assert fit.sigma == pytest.approx(1.0, rel=0.2)
    assert fit.mu == pytest.approx(0.5, abs=0.2)


def test_restricted_fits_satisfy_restrictions():
    rng = np.random.default_rng(9)
    X = simulate_klargest(100, 2, 3.0, 1.0, 0.4, rng)
    unr = fit_gev(X)
    for kw in [dict(fix_xi=0.2), dict(pareto=True), dict(pareto=True, fix_xi=1.0),
               dict(quantile=(5.0, 0.9))]:
        r = fit_gev(X, **kw)
        assert r.loglik <= unr.loglik + 1e-6
        if "fix_xi" in kw:
            assert r.xi == kw["fix_xi"]
        if kw.get("pareto"):
            assert r.mu == pytest.approx(r.sigma / r.xi)
        if "quantile" in kw:
            assert r.quantile(0.9) == pytest.approx(5.0)
