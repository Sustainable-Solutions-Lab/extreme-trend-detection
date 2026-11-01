"""Reproduce the constant-parameter MLEs and LR tests of Müller & Watson (2024), Table 1.

Reference numbers come from the authors' MATLAB output files (Matlab/out/*LikelihoodResults.txt)
in the replication archive, doi:10.7910/DVN/RZG8QK, unpacked under external/muller-watson-2024/files.
Skipped when that folder is absent.
"""

from pathlib import Path

import numpy as np
import pytest

from tvextremes.mle import fit_exceedance, fit_gev, lr_statistic

ROOT = Path(__file__).resolve().parents[1] / "external" / "muller-watson-2024" / "files" / "Matlab"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="replication archive not downloaded")
sio = pytest.importorskip("scipy.io")


def _cv_adjust(name, xi_hat):
    a = np.loadtxt(ROOT / "CVS_Files" / name, delimiter=",")
    return np.exp(a[0] + a[1] * xi_hat + a[2] * xi_hat ** 2)


def load(app):
    m = lambda f: sio.loadmat(ROOT / "mat" / f)
    if app in ("Returns_Large", "Returns_Small"):
        return dict(X=m(app + ".mat")["ret"], cv="1_194")
    if app == "City":
        return dict(X=m("CitySize_100.mat")["cs_data"][:, :30], cv="30_4")
    if app == "Firm":
        d = m("FirmSize_top100_byYear.mat")
        yrs = d["calvec"].ravel()
        rows = [np.where(yrs == y)[0][0] for y in (1950, 1973, 1996, 2019)]
        return dict(X=d["emp_largest_normalized"][rows, :30], cv="30_4")
    if app == "Weather":
        d = m("Disasters.mat")
        Y = d["y"].T.astype(float)
        return dict(Y=np.where(Y == -1, np.nan, Y), tau=1.0, cv="tau_522")
    if app == "Weather_norm":
        d = m("Disasters_norm.mat")
        Y = d["y_norm"].T.astype(float)
        return dict(Y=np.where(Y == -1, np.nan, Y), tau=d["tau_norm"].ravel(), cv="tvt_522")
    raise KeyError(app)


# app: (xi, sigma, mu, q90, adjusted LR for mu=sigma/xi, adjusted LR for Zipf)
TABLE1 = {
    "Weather":       (0.8421, 0.8607, 0.4455, 6.22, 1.46, 1.25),
    "Weather_norm":  (0.9971, 0.8688, 0.9048, 8.25, 0.01, 0.02),
    "Returns_Large": (0.0564, 0.5408, 2.4339, 3.73, 2.27, 64.79),
    "Returns_Small": (0.2750, 0.6792, 2.8461, 4.96, 0.08, 56.06),
    "City":          (0.6508, 1.3440, 2.0177, 8.89, 0.18, 1.57),
    "Firm":          (0.4245, 0.4788, 1.0015, 2.81, 0.90, 3.49),
}


def _fit(d, **kw):
    if "Y" in d:
        return fit_exceedance(d["Y"], d["tau"], **kw)
    return fit_gev(d["X"], **kw)


@pytest.mark.parametrize("app", list(TABLE1))
def test_table1(app):
    xi, sigma, mu, q90, lr_pareto, lr_zipf = TABLE1[app]
    d = load(app)
    unr = _fit(d)
    assert unr.xi == pytest.approx(xi, abs=6e-4)
    assert unr.sigma == pytest.approx(sigma, abs=6e-4)
    assert unr.mu == pytest.approx(mu, abs=6e-4)
    assert unr.quantile(0.9) == pytest.approx(q90, abs=0.006)

    adj3 = _cv_adjust(f"cvx_test3_{d['cv']}.csv", unr.xi)
    adj4 = _cv_adjust(f"cvx_test4_{d['cv']}.csv", unr.xi)
    stat3 = adj3 * lr_statistic(unr, _fit(d, pareto=True))
    stat4 = adj4 * lr_statistic(unr, _fit(d, pareto=True, fix_xi=1.0))
    assert stat3 == pytest.approx(lr_pareto, abs=0.006)
    assert stat4 == pytest.approx(lr_zipf, abs=0.006 * max(1, lr_zipf))


# Table 2: xi_hat-adjusted Nyblom statistics (authors' output files, 2 decimals)
TABLE2 = {"Weather": 11.58, "Weather_norm": 4.24, "Returns_Large": 1.63, "Returns_Small": 1.36,
          "City": 1.11, "Firm": 0.73}


@pytest.mark.parametrize("app", list(TABLE2))
def test_table2_nyblom(app):
    from tvextremes.nyblom import nyblom_test
    d = load(app)
    fit = _fit(d)
    data = d["Y"] if "Y" in d else d["X"]
    L = nyblom_test(data, fit, tau=d.get("tau"))
    stat = _cv_adjust(f"cvx_test0_{d['cv']}.csv", fit.xi) * L
    assert stat == pytest.approx(TABLE2[app], abs=0.006)
