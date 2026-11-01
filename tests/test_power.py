import numpy as np
import pytest
from scipy.stats import genextreme, kstest

from tvextremes import critical_values as cv
from tvextremes import power, pseudo
from tvextremes.mle import fit_local
from tvextremes.trend_tests import fit_gev_trend, mann_kendall, trend_lr_test


@pytest.mark.parametrize("shape", pseudo.SHAPES)
def test_shapes_centred_and_scaled(shape):
    T = 85
    W = pseudo.brownian_paths(1, T, np.random.default_rng(0))[0]
    g = pseudo.shape_values(shape, T, W)
    assert abs(g.mean()) < 1e-12
    if shape in ("linear", "accel", "step50", "step80"):
        assert g[-1] - g[0] == pytest.approx(1.0)
    if shape == "hump":
        assert g.max() - g[0] == pytest.approx(1.0, abs=1e-3)


def test_parameter_paths_units():
    T = 50
    mu, s, xi = pseudo.parameter_paths(pseudo.Alternative("mu", "linear", 2.0), T, (1.0, 0.5, 0.1))
    assert mu[-1] - mu[0] == pytest.approx(1.0)          # 2 sigma with sigma = 0.5
    assert np.all(s == 0.5) and np.all(xi == 0.1)
    mu, s, xi = pseudo.parameter_paths(pseudo.Alternative("sigma", "linear", 0.4), T, (0, 1, 0))
    assert np.log(s[-1] / s[0]) == pytest.approx(0.4)
    mu, s, xi = pseudo.parameter_paths(pseudo.Alternative("xi", "linear", 0.2), T, (0, 1, 0.1))
    assert xi[-1] - xi[0] == pytest.approx(0.2) and xi.mean() == pytest.approx(0.1)


def test_panel_matches_calibration_generator_when_constant():
    g = cv.draw_gammas(1, 20, 4, seed=3)[0]
    for xi in (-0.3, 0.0, 0.4):
        np.testing.assert_allclose(pseudo.gev_k_panel(g, 0.5, 2.0, xi),
                                   cv.panels_from_gammas(g, 0.5, 2.0, xi))


def test_time_varying_panel_has_right_marginals():
    """Probability-integral transform of each period's maximum must be uniform."""
    rng = np.random.default_rng(5)
    T, N = 40, 500
    alt = pseudo.Alternative("xi", "linear", 0.6)
    mu, s, xi = pseudo.parameter_paths(alt, T, (0.0, 1.0, 0.1))
    mu = mu + np.linspace(-1, 1, T)                           # move mu and sigma as well
    s = s * np.exp(np.linspace(-0.3, 0.3, T))
    gam = np.cumsum(rng.standard_exponential((N, T, 3)), axis=2)
    u = []
    for n in range(N):
        X = pseudo.gev_k_panel(gam[n], mu, s, xi)
        assert np.all(np.diff(X, axis=1) <= 0)
        u.append(genextreme.cdf(X[:, 0], -xi, loc=mu, scale=s))   # scipy's c = -xi
    assert kstest(np.concatenate(u), "uniform").pvalue > 0.01


def test_mann_kendall_known_value():
    z, p = mann_kendall(np.arange(10.0))
    assert z == pytest.approx(44 / np.sqrt(125))
    assert mann_kendall(-np.arange(10.0))[0] == pytest.approx(-z)


def test_trend_fit_recovers_trend():
    rng = np.random.default_rng(2)
    T = 400
    alt = pseudo.Alternative("mu", "linear", 2.0)
    mu, s, xi = pseudo.parameter_paths(alt, T, (0.0, 1.0, 0.1))
    X = pseudo.gev_k_panel(np.cumsum(rng.standard_exponential((T, 2)), axis=1), mu, s, xi)
    const = fit_local(X, (0.0, 1.0, 0.1))
    tr = fit_gev_trend(X, const)
    assert tr.mu1 == pytest.approx(2.0, abs=0.3)
    assert tr.xi == pytest.approx(0.1, abs=0.1)
    lr, mu1, reject = trend_lr_test(X, const)
    assert reject and lr > 10


def test_simulate_design_small():
    alts = [pseudo.NULL, pseudo.Alternative("mu", "linear", 3.0),
            pseudo.Alternative("mu", "randomwalk", 3.0)]
    res = power.simulate_design(1, 85, 0.1, N=60, seed=4, alternatives=alts, workers=1)
    assert res.stats.shape == (3, 60, len(power.STATS))
    assert np.isnan(res.stats[..., :4]).sum() == 0
    rows = power.rejection_table(res, nyblom_coef=np.zeros(3))
    assert len(rows) == 3
    assert rows[1]["trend_lr_nominal"] > 0.9 and rows[1]["mk_nominal"] > 0.9
    assert rows[1]["nyblom_sizeadj"] > 0.8
    assert rows[0]["d_rl10"] == 0.0 and rows[1]["d_rl10"] == pytest.approx(3.0)


@pytest.mark.parametrize("alt", [pseudo.NULL, pseudo.Alternative("mu", "linear", 1.5),
                                 pseudo.Alternative("mu", "step80", 1.0),
                                 pseudo.Alternative("sigma", "linear", 0.6),
                                 pseudo.Alternative("xi", "linear", 0.3),
                                 pseudo.Alternative("mu", "randomwalk", 2.0)])
@pytest.mark.parametrize("k", [1, 3])
def test_time_reversal(alt, k):
    """Reversing the order of the periods must leave the constant fit, L_T and the trend LR
    unchanged and flip the sign of the fitted trend and of the Mann-Kendall Z."""
    rng = np.random.default_rng(11)
    T = 85
    W = pseudo.brownian_paths(1, T, rng)[0]
    mu, s, xi = pseudo.parameter_paths(alt, T, (0.0, 1.0, 0.2), W)
    X = pseudo.gev_k_panel(np.cumsum(rng.standard_exponential((T, k)), axis=1), mu, s, xi)
    start = (float(mu.mean()), float(s.mean()), float(xi.mean()))
    fwd = power.panel_statistics(X, start)
    rev = power.panel_statistics(X[::-1], start)
    i = {n: power.STATS.index(n) for n in power.STATS}
    for name in ("nyblom", "xi_hat", "trend_lr"):
        assert rev[i[name]] == pytest.approx(fwd[i[name]], rel=1e-5, abs=1e-7), name
    assert rev[i["mu1_hat"]] == pytest.approx(-fwd[i["mu1_hat"]], rel=1e-4, abs=1e-6)
    assert rev[i["mk_z"]] == pytest.approx(-fwd[i["mk_z"]])


def test_reversed_path_is_reversed_alternative():
    """The generator itself: reversing a linear path equals the path with the opposite amount."""
    T = 30
    for p in pseudo.PARAMS:
        a = pseudo.parameter_paths(pseudo.Alternative(p, "linear", 0.3), T, (0.0, 1.0, 0.1))
        b = pseudo.parameter_paths(pseudo.Alternative(p, "linear", -0.3), T, (0.0, 1.0, 0.1))
        for x, y in zip(a, b):
            np.testing.assert_allclose(x[::-1], y, atol=1e-12)
