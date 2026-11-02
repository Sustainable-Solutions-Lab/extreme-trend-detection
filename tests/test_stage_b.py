import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm
from scipy.stats import t as t_dist

from tvextremes import parents as pa
from tvextremes import stage_b


def test_ar1_stationary():
    z = pa.ar1_gaussian(400, 2000, 0.8, np.random.default_rng(0))
    assert z[:, 0].var() == pytest.approx(1, abs=0.2)
    assert z.var() == pytest.approx(1, abs=0.05)
    lag1 = np.mean(z[:, 1:] * z[:, :-1])
    assert lag1 == pytest.approx(0.8, abs=0.02)


def test_k_largest_plain_and_declustered():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((3, 2 * 50))
    top = pa.k_largest_by_year(z, 2, 4, r=0, days=50)
    for i in range(3):
        for y in range(2):
            np.testing.assert_allclose(top[i, y], np.sort(z[i, y * 50:(y + 1) * 50])[::-1][:4])
    # declustered: brute-force greedy
    top = pa.k_largest_by_year(z, 2, 3, r=2, days=50)
    for i in range(3):
        for y in range(2):
            v = z[i, y * 50:(y + 1) * 50].copy()
            exp = []
            for _ in range(3):
                j = int(np.argmax(v))
                exp.append(v[j])
                v[max(0, j - 2):j + 3] = -np.inf
            np.testing.assert_allclose(top[i, y], exp)


@pytest.mark.parametrize("family", pa.FAMILIES)
def test_marginals_and_monotone(family):
    z = np.linspace(-6, 6, 2001)
    tail = pa.BASE_TAIL[family]
    eps = pa.standard_eps(family, z, tail)
    assert np.all(np.diff(eps) > 0)
    u = norm.cdf(z[::100])
    e = eps[::100]
    if family == "t":
        np.testing.assert_allclose(e * np.sqrt(tail / (tail - 2)), t_dist.ppf(u, tail), rtol=1e-6, atol=1e-9)
    if family == "gamma":
        np.testing.assert_allclose(e * np.sqrt(tail) + tail, gamma_dist.ppf(u, tail), rtol=1e-6, atol=1e-9)
    if family == "beta":
        np.testing.assert_allclose((e / pa.BETA_SCALE + 1) / 2, beta_dist.ppf(u, tail, tail),
                                   rtol=1e-6, atol=1e-12)
        assert eps.max() < pa.cap_height("beta")
    if family != "tanh":                           # unit variance at baseline
        zz = np.random.default_rng(2).standard_normal(400_000)
        x = pa.standard_eps(family, zz, tail)
        assert x.mean() == pytest.approx(0, abs=0.01)
        assert x.var() == pytest.approx(1, abs=0.02)


@pytest.mark.parametrize("family", ["tanh", "beta"])
def test_capped_warming_keeps_cap(family):
    ch = pa.Change("capped_warming", "linear", 1.0)
    m, s, tail = pa.parameter_paths(family, ch, 85)
    cap = m + s * (tail if family == "tanh" else pa.BETA_SCALE)
    np.testing.assert_allclose(cap, pa.cap_height(family))
    assert m[-1] - m[0] == pytest.approx(1.0)


def test_true_return_level_matches_simulation():
    rng = np.random.default_rng(3)
    n_years = 20000
    z = rng.standard_normal((n_years, pa.DAYS)).max(axis=1)
    for family in ("gamma", "tanh"):
        tail = pa.BASE_TAIL[family]
        mx = pa.standard_eps(family, z, tail)
        rl = pa.true_return_level(family, 0.0, 1.0, tail, years=10)
        assert np.mean(mx <= rl) == pytest.approx(0.9, abs=0.007)


def test_beta_tail_index():
    """GEV fit to block maxima of the beta parent should give xi near -1/b."""
    from tvextremes.mle import fit_gev
    rng = np.random.default_rng(4)
    z = rng.standard_normal((3000, pa.DAYS)).max(axis=1)
    mx = pa.standard_eps("beta", z, 3.0)
    assert fit_gev(mx[:, None]).xi == pytest.approx(-1 / 3, abs=0.06)


def test_simulate_config_small():
    changes = [pa.NULL, pa.Change("mean", "linear", 1.0), pa.Change("tail", "step50", 1.0)]
    res = stage_b.simulate_config("t", 3, 40, 0.5, r=2, N_null=100, N_alt=50, seed=5,
                                  changes=changes, workers=1)
    assert res.stats.shape[:2] == (3, 100)
    assert np.isnan(res.stats[0, :, 0]).sum() == 0
    assert np.isnan(res.stats[1, :50, 0]).sum() == 0 and np.isnan(res.stats[1, 50:, 0]).all()
    rows = stage_b.summary_rows(res)
    assert rows[0]["N"] == 100 and rows[1]["N"] == 50
    assert rows[1]["trend_lr_nominal"] > 0.9
    assert rows[1]["d_rl100_indep"] == pytest.approx(1.0)


def test_zmax_quantile_independent_matches_formula_and_simulation():
    q = pa.zmax_quantile([0.5, 0.99], 0.0)
    assert q[0] == pytest.approx(norm.ppf(0.5 ** (1 / 365)))
    # phi > 0 by simulation; persistence lowers the quantiles of the annual maximum slightly
    q5 = pa.zmax_quantile([0.5, 0.99], 0.5, n_years=50_000)
    assert q5[0] < q[0] and abs(q5[1] - q[1]) < 0.1


def test_return_level_rows_null_unbiased_ish():
    res = stage_b.simulate_config("normal", 1, 85, 0.0, N_null=200, N_alt=0, seed=6,
                                  changes=[pa.NULL], workers=1)
    rows = stage_b.return_level_rows(res, probs=(0.9,))
    r = rows[0]
    assert r["true_change"] == 0
    assert abs(r["const_first_bias"]) < 0.05 and r["const_first_rmse"] < 0.2
    assert abs(r["trend_change_bias"]) < 0.1


def test_return_level_rows_trend_tracks_mean_shift():
    ch = pa.Change("mean", "linear", 1.0)
    res = stage_b.simulate_config("normal", 3, 85, 0.0, N_null=100, N_alt=100, seed=7,
                                  changes=[pa.NULL, ch], workers=1)
    rows = [r for r in stage_b.return_level_rows(res, probs=(0.99,)) if r["knob"] == "mean"
            and r["amount"] == 1.0]
    r = rows[0]
    assert r["true_change"] == pytest.approx(1.0)
    assert r["const_change_bias"] == pytest.approx(-1.0)
    assert abs(r["trend_change_bias"]) < 0.15
