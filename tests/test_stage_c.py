import numpy as np
import pytest

cmdstanpy = pytest.importorskip("cmdstanpy")
try:
    cmdstanpy.cmdstan_path()
except ValueError:
    pytest.skip("CmdStan is not installed", allow_module_level=True)

from tvextremes import parents as pa
from tvextremes import stage_c

ORIGINAL = stage_c.ROOT / "external" / "muller-watson-2024" / "files" / "Stan" / "Stan_Code" / "GEV_4RW_Model.stan"


def _panel(T=30, k=2):
    return stage_c._panel("normal", pa.Change("mean", "linear", 0.5), k, T, 0.5, 0, 1, 0)


@pytest.mark.skipif(not ORIGINAL.exists(), reason="authors' replication archive not downloaded")
def test_same_log_density_as_authors_model():
    X = _panel()
    T, k = X.shape
    data = dict(T=T, nobs=k, xi_min=stage_c.XI_MIN, xi_max=stage_c.XI_MAX, sg_xi=stage_c.SG_XI,
                sg_alpha=stage_c.SG_ALPHA, sg_s=stage_c.SG_S, y=X.T.tolist(), np=1, probs=[0.99])
    orig = cmdstanpy.CmdStanModel(stan_file=str(ORIGINAL))
    rng = np.random.default_rng(0)
    for _ in range(3):
        p = dict(trans_xi_level=rng.normal(0, 0.5), ln_s_level=rng.normal(-1, 0.2),
                 m_level=float(X[:, 0].mean()) + rng.normal(0, 0.1),
                 g_xi=0.3, g_alpha=0.2, g_s=0.4, g_m=0.5,
                 **{f"h_{v}": rng.standard_normal(T).tolist() for v in ("xi", "alpha", "s", "m")})
        a = orig.log_prob(p, data=data).iloc[0, 0]
        b = stage_c.model().log_prob(p, data=data).iloc[0, 0]
        assert a == pytest.approx(b, rel=1e-10)


def test_fit_and_score_small():
    X = _panel()
    f = stage_c.fit_tvp(X, chains=2, warmup=200, samples=200, seed=3)
    assert f.q_mean.shape == (len(stage_c.PROBS), X.shape[0])
    assert np.all(f.q_lo <= f.q_median) and np.all(f.q_median <= f.q_hi)
    assert np.all(np.diff(f.q_median, axis=0) > 0)                  # X_0.5 < X_0.9 < X_0.99


def test_true_exceedance_inverts_truth():
    m, s, tail = pa.parameter_paths("t", pa.Change("tail", "linear", 1.0), 40)
    for phi in (0.0, 0.5):
        x = pa.true_quantile_path("t", m, s, tail, 0.99, phi, cache_dir=stage_c.ROOT / "data" / "output" / "stage_b")
        exc = stage_c.true_exceedance("t", x[-1], m[-1], s[-1], np.asarray(tail)[-1], phi,
                                      cache_dir=stage_c.ROOT / "data" / "output" / "stage_b")
        assert exc == pytest.approx(0.01, abs=5e-4)


def test_mle_paths_match_stage_b_endpoints():
    from tvextremes.power import panel_statistics
    from tvextremes.stage_b import _start
    X = _panel(T=40, k=1)
    st = panel_statistics(X, _start(X))
    q = stage_c.mle_quantile_paths(st, 40)
    from tvextremes.likelihood import gev_quantile
    from tvextremes.power import STATS
    S = dict(zip(STATS, st))
    assert q[1, 2, -1] == pytest.approx(gev_quantile(0.99, S["trend_mu0"] + S["mu1_hat"] / 2,
                                                     S["trend_sigma"], S["trend_xi"]))
    assert q[1, 2, 0] == pytest.approx(gev_quantile(0.99, S["trend_mu0"] - S["mu1_hat"] / 2,
                                                    S["trend_sigma"], S["trend_xi"]))
    assert np.all(q[0, 2] == q[0, 2, 0])                               # constant fit is flat


def test_smoother_exact_on_linear_and_reduces_noise():
    t = np.arange(60.0)
    line = 1.5 - 0.02 * t
    assert np.allclose(stage_c.smooth_local_linear(line), line)
    noise = np.random.default_rng(0).standard_normal(60)
    assert stage_c.smooth_local_linear(noise).std() < 0.5 * noise.std()


def test_extra_estimators_by_name():
    ch = pa.Change("mean", "linear", 0.5)
    names = stage_c.EXTRA_NAMES
    _, all_ = stage_c._extra_job(("normal", ch, 1, 40, 0.5, 0, 1, 0, names))
    _, two = stage_c._extra_job(("normal", ch, 1, 40, 0.5, 0, 1, 0, ("cov2", "recent30")))
    assert np.allclose(two[0], all_[names.index("cov2")], equal_nan=True)
    assert np.allclose(two[1], all_[names.index("recent30")], equal_nan=True)
    assert np.all(np.isfinite(all_))
    assert np.ptp(all_[names.index("recent30")], axis=1).max() == 0     # flat path
