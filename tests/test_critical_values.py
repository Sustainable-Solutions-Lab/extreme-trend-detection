"""Tests for the critical-value calibration machinery (fast; full calibrations live in scripts/)."""

import numpy as np
import pytest

from tvextremes import critical_values as cv
from tvextremes.nyblom import nyblom_statistic


def synthetic_sim(N=20000, seed=0):
    """A fake null simulation where the LR scale grows with xi: stat = exp(0.4 xi) * chi2_1 / 2."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-0.5, 1.5, 10)
    stats = np.exp(0.4 * grid)[:, None] * rng.chisquare(1, (10, N)) / 2
    xi_hat = grid[:, None] + 0.1 * rng.standard_normal((10, N))
    return cv.NullSimulation("q90", 1, 50, N, seed, grid, stats, xi_hat)


def test_fit_adjustment_gives_exact_size_on_grid_and_near_unbiased():
    sim = synthetic_sim()
    a, rates = cv.fit_adjustment(sim, alpha=0.05)
    assert rates.max() == pytest.approx(0.05, abs=1.5 / sim.N)
    assert rates.min() > 0.04                   # close to unbiased across xi
    assert a[1] < 0                             # larger xi -> larger LR -> smaller adjustment
    # Fresh draws: the size should stay close to 5%.
    fresh = synthetic_sim(seed=1)
    assert cv.rejection_rates(fresh, a).max() < 0.05 + 3 * np.sqrt(0.05 * 0.95 / fresh.N)


def test_panels_from_gammas_are_descending_gev():
    g = cv.draw_gammas(500, 3, 4, seed=2)
    for xi in (-0.4, 0.0, 0.7):
        X = cv.panels_from_gammas(g, 0.5, 2.0, xi)
        assert np.all(np.diff(X, axis=2) <= 0)
    from scipy import stats
    X = cv.panels_from_gammas(cv.draw_gammas(20000, 1, 1, seed=3), 0.5, 2.0, 0.3)[:, 0, 0]
    assert stats.kstest(X, stats.genextreme(-0.3, loc=0.5, scale=2.0).cdf).pvalue > 0.01


def test_null_dgp_satisfies_restrictions():
    (mu, s, xi), r = cv.null_dgp("pareto", 0.4)
    assert mu == pytest.approx(s / xi) and r == dict(pareto=True)
    (mu, s, xi), r = cv.null_dgp("zipf", 0.4)
    assert (mu, s, xi) == (1.0, 1.0, 1.0)
    (mu, s, xi), r = cv.null_dgp("q90", 0.2)
    assert r["quantile"][0] == pytest.approx(float(cv.lk.gev_quantile(0.9, mu, s, xi)))


@pytest.mark.parametrize("test", ["nyblom", "pareto"])
def test_simulate_null_runs_and_is_reproducible(test):
    a = cv.simulate_null(test, 2, 30, N=12, seed=4, workers=1, grid=[0.2, 0.8])
    b = cv.simulate_null(test, 2, 30, N=12, seed=4, workers=1, grid=[0.2, 0.8])
    assert a.stats.shape == (2, 12) and a.failures == 0
    np.testing.assert_array_equal(a.stats, b.stats)
    assert np.all(a.stats >= 0)


def test_nyblom_statistic_known_value():
    S = np.array([[1.0], [-1.0], [2.0], [-2.0]])
    # partial sums from the end: -2, 0, -1, 0 -> sum of squares 5; / T^2 = 5/16 with V = 1
    assert nyblom_statistic(S, np.eye(1)) == pytest.approx(5 / 16)


def test_stored_authors_coefficients_load():
    a = cv.get_adjustment("pareto", 1, 194, source="mw2024")
    np.testing.assert_allclose(a, [-0.6959448029540886, 0.01958998038518359, -0.02523176527282205])
    assert cv.get_xi_cv(-0.5, 1, 194, source="mw2024") == pytest.approx(2.130883, abs=1e-5)


def test_save_and_lookup_roundtrip(tmp_path):
    sim = synthetic_sim(N=2000)
    a, rates = cv.fit_adjustment(sim)
    path = tmp_path / "adj.csv"
    cv.save_adjustment(sim, a, rates, path=path)
    cv.save_adjustment(sim, a, rates, path=path)              # replacing, not duplicating
    assert len(cv._read(path)) == 1
    np.testing.assert_allclose(cv.get_adjustment("q90", 1, 50, path=path), a, rtol=1e-9)
