"""Stage C: recovery of known paths by the Müller–Watson Bayesian time-varying GEV model.

The model (``stan/gev_4rw.stan``, the authors' GEV_4RW_Model with leaner output) gives each of
xi, alpha, s and m a random walk; its posterior gives paths of mu_t, sigma_t, xi_t and of the
annual-maximum quantiles X_p(t). Data come from Stage B's parent-distribution generator, where the
true X_p(t) is known exactly, so the estimated paths can be scored against the truth.

``fit_tvp`` fits one panel and returns posterior summaries (mean, median and 5/95% quantiles of
X_p(t) for each p). ``simulate_scenario`` draws panels for one (family, change, k, T, phi, r) and
fits each, in parallel processes (one single-threaded Stan run of 4 sequential chains per panel).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import parents as pa

ROOT = Path(__file__).resolve().parents[1]
STAN_FILE = ROOT / "stan" / "gev_4rw.stan"
PROBS = (0.5, 0.9, 0.99)

# The authors' settings (Stan/Returns/4RW_Model.r)
XI_MIN, XI_MAX = -0.9, 2.1
SG_XI, SG_ALPHA, SG_S = 1.0, 0.5, 0.5

# Prior scale sets (sg_xi, sg_alpha, sg_s) for the random walks' innovation scales. "mw" is the
# authors'; "tight" shrinks the walks of the shape and scale parameters by a factor 2-4 (the
# location walk, whose scale is set by the data, is unchanged): a prior-sensitivity check.
PRIORS = {"mw": (SG_XI, SG_ALPHA, SG_S), "tight": (0.25, 0.25, 0.25)}

_MODEL = None


def model():
    """Compiled CmdStan model (compiled once per process; the executable is cached on disk)."""
    global _MODEL
    if _MODEL is None:
        import logging

        import cmdstanpy
        cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
        _MODEL = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
    return _MODEL


def _inits(X, chains):
    top = X[:, 0]
    sd = max(float(top.std()), 1e-3)
    base = dict(trans_xi_level=0.0, ln_s_level=float(np.log(sd)), m_level=float(top.mean()),
                g_xi=0.2, g_alpha=0.2, g_s=0.2, g_m=0.2)
    # small offsets between chains, as in the authors' driver
    offs = [(0, 0, 0), (0.1, 0.1, 0.1), (-0.1, -0.1, -0.1), (0.0, 0.1, 0.1)]
    out = []
    for c in range(chains):
        d = dict(base)
        o = offs[c % len(offs)]
        d["trans_xi_level"] += o[0]
        d["ln_s_level"] += o[1]
        d["m_level"] += o[2] * sd
        out.append(d)
    return out


# Posterior predictive summaries. The predictive p-quantile x*(t) solves
#   mean over draws d of P(max_t > x | theta_d) = 1 - p,
# i.e. the p-quantile of the posterior predictive distribution of the year-t maximum (the
# posterior integrated over, then the quantile taken). At x* the draws' own exceedance
# probabilities average exactly 1 - p; their spread over draws (PRED_EXC_Q) is the model's own
# uncertainty about the risk at that level.
PRED_EXC_Q = (0.05, 0.25, 0.5, 0.75, 0.95)


def gev_survival(x, mu, sigma, xi):
    """P(Y > x) for Y ~ GEV(mu, sigma, xi), elementwise (0 above an upper end point, 1 below a
    lower one)."""
    z = (x - mu) / sigma
    small = np.abs(xi) < 1e-6
    xs = np.where(small, 1.0, xi)
    a = 1.0 + xs * z
    with np.errstate(all="ignore"):
        t = np.where(small, np.exp(-z), np.where(a > 0, np.maximum(a, 1e-300) ** (-1.0 / xs),
                                                 np.where(xs > 0, np.inf, 0.0)))
    return -np.expm1(-t)


def predictive_quantile(mu, sigma, xi, p, lo, hi, iters=50):
    """p-quantile of the predictive distribution: mixture over draws (rows) of GEV(mu, sigma, xi),
    separately for each column (year). lo, hi: brackets per column (e.g. the smallest and largest
    of the draws' own p-quantiles, between which the mixture quantile always lies)."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    target = 1.0 - p
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = gev_survival(mid[None, :], mu, sigma, xi).mean(axis=0) > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def predictive_summaries(mu, sigma, xi, q, probs, chains):
    """From draws mu, sigma, xi (D, T) and the draws' quantiles q (D, n_p, T):
    pred (n_p, T) predictive quantiles; pred_chain (chains, n_p) the final-year predictive
    quantile from each chain's draws alone (Monte Carlo error); pred_exc (len(PRED_EXC_Q), n_p, T)
    quantiles over draws of each draw's exceedance probability of the predictive level."""
    D, T = mu.shape
    n_p = len(probs)
    pred = np.empty((n_p, T))
    pred_exc = np.empty((len(PRED_EXC_Q), n_p, T))
    pred_chain = np.empty((chains, n_p))
    per = D // chains
    for i, p in enumerate(probs):
        pred[i] = predictive_quantile(mu, sigma, xi, p, q[:, i].min(axis=0), q[:, i].max(axis=0))
        S = gev_survival(pred[i][None, :], mu, sigma, xi)
        pred_exc[:, i] = np.quantile(S, PRED_EXC_Q, axis=0)
        for c in range(chains):
            sl = slice(c * per, (c + 1) * per)
            pred_chain[c, i] = predictive_quantile(mu[sl, -1:], sigma[sl, -1:], xi[sl, -1:], p,
                                                   q[sl, i, -1:].min(axis=0),
                                                   q[sl, i, -1:].max(axis=0))[0]
    return pred, pred_chain, pred_exc


@dataclass
class TVPFit:
    probs: tuple
    q_mean: np.ndarray        # (n_p, T) posterior mean of X_p(t)
    q_median: np.ndarray      # (n_p, T)
    q_lo: np.ndarray          # (n_p, T) 5% posterior quantile
    q_hi: np.ndarray          # (n_p, T) 95%
    par_mean: np.ndarray      # (3, T) posterior means of mu, sigma, xi
    divergences: int
    max_rhat: float
    min_ess: float
    seconds: float
    pred: np.ndarray = None        # (n_p, T) posterior predictive quantiles
    pred_chain: np.ndarray = None  # (chains, n_p) final-year predictive quantile, chain by chain
    pred_exc: np.ndarray = None    # (len(PRED_EXC_Q), n_p, T) draws' exceedance at pred
    last_draws: np.ndarray = None  # (D, 3) final-year draws of mu, sigma, xi


def fit_tvp(X, probs=PROBS, chains=4, warmup=1000, samples=1000, seed=1, adapt_delta=0.95,
            prior="mw"):
    """Fit the 4-random-walk GEV model to one panel X (T, k), rows sorted in decreasing order.
    prior: a key of PRIORS (the random walks' prior scales)."""
    import time
    X = np.asarray(X, float)
    T, k = X.shape
    sg_xi, sg_alpha, sg_s = PRIORS[prior]
    data = dict(T=T, nobs=k, xi_min=XI_MIN, xi_max=XI_MAX, sg_xi=sg_xi, sg_alpha=sg_alpha,
                sg_s=sg_s, y=X.T.tolist(), np=len(probs), probs=list(probs))
    tmp = tempfile.mkdtemp(prefix="stage_c_")
    t0 = time.time()
    try:
        fit = model().sample(data=data, chains=chains, parallel_chains=1, iter_warmup=warmup,
                             iter_sampling=samples, seed=seed, inits=_inits(X, chains),
                             adapt_delta=adapt_delta, output_dir=tmp, show_progress=False,
                             show_console=False)
        q = fit.stan_variable("q")                       # (draws, n_p, T)
        draws = [fit.stan_variable(v) for v in ("mu", "sigma", "xi")]    # each (draws, T)
        par = np.stack([d.mean(axis=0) for d in draws])
        div = int(np.sum(fit.method_variables()["divergent__"]))
        summ = fit.summary(percentiles=(50,), sig_figs=6)
        rows = summ.index.str.match(r"^(mu|sigma|xi|q)\[")
        rhat = float(np.nanmax(summ.loc[rows, "R_hat"]))
        ess_col = "ESS_bulk" if "ESS_bulk" in summ.columns else "N_Eff"
        ess = float(np.nanmin(summ.loc[rows, ess_col]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lo, med, hi = np.quantile(q, [0.05, 0.5, 0.95], axis=0)
    pred, pred_chain, pred_exc = predictive_summaries(*draws, q, probs, chains)
    last = np.stack([d[:, -1] for d in draws], axis=1)
    return TVPFit(tuple(probs), q.mean(axis=0), med, lo, hi, par, div, rhat, ess, time.time() - t0,
                  pred, pred_chain, pred_exc, last)


# ---------------------------------------------------------------------------
# Monte Carlo over panels
# ---------------------------------------------------------------------------

def true_exceedance(family, x, m, s, tail, phi, cache_dir=None):
    """P(annual maximum > x) in a year with parent parameters (m, s, tail): the exact
    exceedance probability of an estimated level x (any shape)."""
    from scipy.stats import norm
    zg = np.linspace(-3.0, 9.0, 6001)
    eps = pa.standard_eps(family, zg, tail)
    z = np.interp((np.asarray(x, float) - m) / s, eps, zg, left=-3.0, right=9.0)
    if phi == 0:
        return -np.expm1(pa.DAYS * norm.logcdf(z))
    pa.zmax_quantile(0.5, phi, cache_dir=cache_dir)                 # loads or builds the cache
    mx = pa._ZMAX_CACHE[(float(phi), pa.DAYS, 500_000, 20261101)]
    return 1.0 - np.searchsorted(mx, z) / mx.size


def mle_quantile_paths(stats, T, probs=PROBS):
    """X_p(t) paths (n_est, n_p, T) of the constant, linear-trend and two-segment MLE fits from
    one row of ``power.panel_statistics``."""
    from .likelihood import gev_quantile
    from .power import STATS
    S = dict(zip(STATS, stats))
    from .pseudo import time_index
    u = time_index(T)                                   # as in trend_tests.fit_gev_trend
    h = T // 2
    out = np.full((3, len(probs), T), np.nan)
    for i, p in enumerate(probs):
        out[0, i] = gev_quantile(p, S["mu_hat"], S["sigma_hat"], S["xi_hat"])
        out[1, i] = gev_quantile(p, S["trend_mu0"] + S["mu1_hat"] * (u - 0.5), S["trend_sigma"],
                                 S["trend_xi"])
        out[2, i, :h] = gev_quantile(p, S["h1_mu"], S["h1_sigma"], S["h1_xi"])
        out[2, i, h:] = gev_quantile(p, S["h2_mu"], S["h2_sigma"], S["h2_xi"])
    return out


MLE_NAMES = ("const", "trend", "split")


def _panel(family, change, k, T, phi, r, seed, rep, with_mean=False):
    """Panel X (T, k) of replicate rep; with_mean also returns the annual mean of all days."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
    z = pa.ar1_gaussian(1, T * pa.DAYS, phi, rng)
    zt = pa.k_largest_by_year(z, T, k, r)
    m, s, tail = pa.parameter_paths(family, change, T)
    X = pa.transform(family, zt, m, s, tail)[0]
    if with_mean:
        return X, pa.annual_mean(family, z, T, m, s, tail)[0]
    return X


# Further MLE estimators, computed after the fact by regenerating the panels from their seeds
# (cheap: no Stan). Each fits GEV parameters linear in a covariate c_t (trend_tests.
# fit_gev_covariate):
#   trend3     mu_t, log sigma_t and xi_t linear in time
#   trend2     mu_t and log sigma_t linear in time, constant xi
#   cov3       all three linear in the annual mean of all days of year t (what an analyst can
#              compute; unstable: xi follows the year-to-year noise of the covariate)
#   cov3_true  all three linear in the true mean path m_t (an oracle covariate)
#   cov2       mu_t and log sigma_t linear in a smoothed annual mean (local-linear smoother,
#              Gaussian kernel, SMOOTH_YEARS bandwidth), constant xi: the usual climate practice
#   cov2_true  mu_t and log sigma_t linear in the true m_t, constant xi (oracle)
#   recent30   constant GEV fitted to the last 30 years only (a common practical shortcut;
#              equal to the constant fit when T <= 30); its path is flat
#   aic        when the form of the change is unknown, let the data choose: the candidate with the
#              lowest AIC among AIC_CANDIDATES (constant, location trend, location + width trend,
#              mean-state covariate)
#   aic_avg    the Akaike-weighted average of the candidates' X_p paths
EXTRA_NAMES = ("trend3", "cov3", "cov3_true", "trend2", "recent30", "cov2", "cov2_true", "aic",
               "aic_avg")
AIC_CANDIDATES = ("const", "trend", "trend2", "cov2")
RECENT = 30
SMOOTH_YEARS = 10.0


def smooth_local_linear(y, h=SMOOTH_YEARS):
    """Local-linear kernel smoother (Gaussian kernel, bandwidth h in years) of a yearly series.
    Unlike a centred moving average it has no boundary bias for a linear trend, which matters
    because the last year is the one scored."""
    y = np.asarray(y, float)
    t = np.arange(len(y), dtype=float)
    out = np.empty_like(y)
    for i in range(len(y)):
        w = np.exp(-0.5 * ((t - t[i]) / h) ** 2)
        d = t - t[i]
        s0, s1, s2 = w.sum(), (w * d).sum(), (w * d * d).sum()
        out[i] = ((s2 * w - s1 * w * d) @ y) / (s0 * s2 - s1 * s1)
    return out


def aic_fits(X, amean, start):
    """Fits of the AIC candidates to one record: {name: (mu_t, sigma_t, xi_t, loglik, n_params)}
    (missing where a fit failed)."""
    from .pseudo import time_index
    from .trend_tests import fit_gev_covariate
    T = X.shape[0]
    out = {"const": (np.full(T, start.mu), np.full(T, start.sigma), np.full(T, start.xi),
                     start.loglik, 3)}
    spec = {"trend": (time_index(T), ("mu",), 4), "trend2": (time_index(T), ("mu", "sigma"), 5),
            "cov2": (smooth_local_linear(amean), ("mu", "sigma"), 5)}
    for name, (c, vary, npar) in spec.items():
        c = np.asarray(c, float)
        if np.ptp(c) == 0:
            c = np.zeros(T)
        try:
            f = fit_gev_covariate(X, c, start=start, vary=vary)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if f.success:
            out[name] = (*f.params(), f.loglik, npar)
    return out


def aic_weights(fits):
    """Akaike weights {name: w} of the fitted candidates."""
    aic = {n: 2 * f[4] - 2 * f[3] for n, f in fits.items()}
    best = min(aic.values())
    w = {n: np.exp(-(a - best) / 2) for n, a in aic.items()}
    tot = sum(w.values())
    return {n: v / tot for n, v in w.items()}


def aic_choice(args):
    """Index into AIC_CANDIDATES of the model AIC selects for one record (-1 if none fitted)."""
    from .mle import fit_local
    from .stage_b import _start
    family, change, k, T, phi, r, seed, rep = args
    X, amean = _panel(family, change, k, T, phi, r, seed, rep, with_mean=True)
    with np.errstate(all="ignore"):
        start = fit_local(X, _start(X))
        if not start.success:
            return rep, -1
        w = aic_weights(aic_fits(X, amean, start))
    return rep, AIC_CANDIDATES.index(max(w, key=w.get))


def _extra_job(args):
    from .likelihood import gev_quantile
    from .mle import fit_local
    from .pseudo import time_index
    from .stage_b import _start
    from .trend_tests import fit_gev_covariate
    family, change, k, T, phi, r, seed, rep, names = args
    X, amean = _panel(family, change, k, T, phi, r, seed, rep, with_mean=True)
    m_true = pa.parameter_paths(family, change, T)[0]
    covariate = {"trend3": time_index(T), "trend2": time_index(T), "cov3": amean,
                 "cov3_true": m_true, "cov2": smooth_local_linear(amean), "cov2_true": m_true}
    out = np.full((len(names), len(PROBS), T), np.nan)
    with np.errstate(all="ignore"):
        start = fit_local(X, _start(X))
        if not start.success:
            return rep, out
        if "aic" in names or "aic_avg" in names:
            fits = aic_fits(X, amean, start)
            w = aic_weights(fits)
            qs = {n: np.stack([gev_quantile(p, *fits[n][:3]) for p in PROBS]) for n in fits}
            if "aic" in names:
                out[names.index("aic")] = qs[max(w, key=w.get)]
            if "aic_avg" in names:
                out[names.index("aic_avg")] = sum(w[n] * qs[n] for n in fits)
        for i, name in enumerate(names):
            if name in ("aic", "aic_avg"):
                continue
            if name == "recent30":
                Xr = X[-RECENT:]
                fr = fit_local(Xr, _start(Xr)) if T > RECENT else start
                if fr.success:
                    for ip, p in enumerate(PROBS):
                        out[i, ip] = fr.quantile(p)
                continue
            c = np.asarray(covariate[name], float)
            if np.ptp(c) == 0:                           # no variation: the constant fit
                c = np.zeros(T)
            vary = ("mu", "sigma", "xi") if "3" in name else ("mu", "sigma")    # *3: xi too
            try:
                f = fit_gev_covariate(X, c, start=start, vary=vary)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
            if f.success:
                mu, sg, xi = f.params()
                for ip, p in enumerate(PROBS):
                    out[i, ip] = gev_quantile(p, mu, sg, xi)
    return rep, out


def extra_estimates(res, workers=4, cache_dir=None):
    """X_p paths (N, len(EXTRA_NAMES), n_p, T) of the further MLE estimators for a scenario,
    cached as extra_<scenario>.npz in cache_dir if given. Only estimators missing from the cache
    are computed."""
    N = res.stats.shape[0]
    path, have = None, {}
    if cache_dir is not None:
        ch = res.change
        path = Path(cache_dir) / (f"extra_{res.family}_{ch.knob}_{ch.shape}_{ch.amount:g}_k{res.k}"
                                  f"_T{res.T}_phi{res.phi:.2f}_r{res.r}.npz")
        if path.exists():
            d = np.load(path)
            if d["q"].shape[0] == N:
                have = {str(n): d["q"][:, i] for i, n in enumerate(d["names"])}
    todo = tuple(n for n in EXTRA_NAMES if n not in have)
    if todo:
        jobs = [(res.family, res.change, res.k, res.T, res.phi, res.r, res.seed, i, todo)
                for i in range(N)]
        new = np.full((N, len(todo), len(PROBS), res.T), np.nan)
        if workers == 1:
            parts = map(_extra_job, jobs)
        else:
            from . import critical_values as cv
            ex = ProcessPoolExecutor(max_workers=workers, initializer=cv._single_thread)
            parts = ex.map(_extra_job, jobs, chunksize=4)
        for i, o in parts:
            new[i] = o
        if workers != 1:
            ex.shutdown()
        have.update({n: new[:, j] for j, n in enumerate(todo)})
    q = np.stack([have[n] for n in EXTRA_NAMES], axis=1).astype(float)
    if path is not None and todo:
        np.savez_compressed(path, q=q.astype(np.float32), names=np.array(EXTRA_NAMES))
    return q


def _job(args):
    from .power import panel_statistics
    from .stage_b import _start
    family, change, k, T, phi, r, seed, rep, mcmc = args
    X = _panel(family, change, k, T, phi, r, seed, rep)
    with np.errstate(all="ignore"):
        stats = panel_statistics(X, _start(X))
    chains = mcmc.get("chains", 4)
    pred = dict(pred_q=np.full((len(PROBS), T), np.nan),
                pred_chain=np.full((chains, len(PROBS)), np.nan),
                pred_exc=np.full((len(PRED_EXC_Q), len(PROBS), T), np.nan),
                last_draws=np.full((chains * mcmc.get("samples", 1000), 3), np.nan))
    try:
        f = fit_tvp(X, seed=int(seed % 2**31) + rep, **mcmc)
        tvp = np.stack([f.q_mean, f.q_median, f.q_lo, f.q_hi])
        diag = np.array([f.divergences, f.max_rhat, f.min_ess, f.seconds])
        pred = dict(pred_q=f.pred, pred_chain=f.pred_chain, pred_exc=f.pred_exc,
                    last_draws=f.last_draws)
    except Exception:                                   # a failed Stan run is recorded as NaN
        tvp = np.full((4, len(PROBS), T), np.nan)
        diag = np.full(4, np.nan)
    return rep, stats, mle_quantile_paths(stats, T), tvp, diag, pred


@dataclass
class ScenarioResult:
    family: str
    change: object
    k: int
    T: int
    phi: float
    r: int
    seed: int
    stats: np.ndarray        # (N, len(power.STATS))
    mle_q: np.ndarray        # (N, 3, n_p, T)  const, trend, split
    tvp_q: np.ndarray        # (N, 4, n_p, T)  posterior mean, median, 5%, 95%
    diag: np.ndarray         # (N, 4)  divergences, max R-hat, min bulk ESS, seconds
    prior: str = "mw"        # key of PRIORS used for the TVP fits
    # posterior predictive output (runs from 2026-11-03 on; None in older files)
    pred_q: np.ndarray = None      # (N, n_p, T) predictive quantiles X*_p(t)
    pred_chain: np.ndarray = None  # (N, chains, n_p) final-year predictive quantile per chain
    pred_exc: np.ndarray = None    # (N, len(PRED_EXC_Q), n_p, T) draws' exceedance at X*_p(t)
    last_draws: np.ndarray = None  # (N, D, 3) final-year posterior draws of mu, sigma, xi


def simulate_scenario(family, change, k=1, T=85, phi=0.5, r=0, N=100, seed=1, workers=None,
                      mcmc=None):
    """Fit the TVP model and the MLE estimators to N panels of one scenario (parallel)."""
    model()                                             # compile once before forking
    mcmc = dict(mcmc or {})
    workers = workers or max(1, (os.cpu_count() or 3) - 2)
    jobs = [(family, change, k, T, phi, r, seed, i, mcmc) for i in range(N)]
    from .power import STATS
    stats = np.full((N, len(STATS)), np.nan)
    mle_q = np.full((N, 3, len(PROBS), T), np.nan)
    tvp_q = np.full((N, 4, len(PROBS), T), np.nan)
    diag = np.full((N, 4), np.nan)
    pred = {}
    if workers == 1:
        parts = map(_job, jobs)
    else:
        from . import critical_values as cv
        ex = ProcessPoolExecutor(max_workers=workers, initializer=cv._single_thread)
        parts = ex.map(_job, jobs)
    for i, st, mq, tq, dg, pr in parts:
        stats[i], mle_q[i], tvp_q[i], diag[i] = st, mq, tq, dg
        for key, v in pr.items():
            if key not in pred:
                pred[key] = np.full((N,) + v.shape, np.nan)
            pred[key][i] = v
    if workers != 1:
        ex.shutdown()
    return ScenarioResult(family, change, k, T, phi, r, seed, stats, mle_q, tvp_q, diag,
                          mcmc.get("prior", "mw"), **pred)


def save(res: ScenarioResult, path):
    np.savez_compressed(path, stats=res.stats, mle_q=res.mle_q.astype(np.float32),
                        tvp_q=res.tvp_q.astype(np.float32), diag=res.diag, family=res.family,
                        change=res.change.label, k=res.k, T=res.T, phi=res.phi, r=res.r,
                        seed=res.seed, probs=np.array(PROBS), prior=res.prior,
                        **{key: getattr(res, key).astype(np.float32) for key in PRED_FIELDS
                           if getattr(res, key) is not None})


PRED_FIELDS = ("pred_q", "pred_chain", "pred_exc", "last_draws")


def load(path):
    d = np.load(path)
    knob, shape, amount = str(d["change"]).split(":")
    return ScenarioResult(str(d["family"]), pa.Change(knob, shape, float(amount)), int(d["k"]),
                          int(d["T"]), float(d["phi"]), int(d["r"]), int(d["seed"]), d["stats"],
                          d["mle_q"].astype(float), d["tvp_q"].astype(float), d["diag"],
                          str(d["prior"]) if "prior" in d else "mw",
                          **{key: d[key].astype(float) for key in PRED_FIELDS if key in d})


def score_rows(res: ScenarioResult, cache_dir=None, extra=None):
    """One row per (estimator, p): accuracy of the estimated X_p path against the truth.

    Columns (years: 'last' = final year, 'all' = averaged over all years):
    * exc_med_last, exc_mean_last: median and mean over panels of the true exceedance probability
      of the estimated X_p in the last year (ideal: 1 - p)
    * exc_gt2x_last: share of panels where that probability exceeds 2 (1 - p), i.e. risk
      understated by more than a factor of 2
    * bias_last, rmse_last, rmse_all: error of the level itself (units of the daily parent sd)
    * change_bias, change_rmse: error of the estimated change X_p(last) - X_p(first)
    * cover90_last, cover90_all: coverage of the 90% credible interval (TVP only)

    extra: X_p paths of the further MLE estimators (``extra_estimates``), scored as well.
    """
    m, s, tail = pa.parameter_paths(res.family, res.change, res.T)
    tail_last = None if tail is None else np.asarray(tail)[-1]
    ests = {name: res.mle_q[:, i] for i, name in enumerate(MLE_NAMES)}
    if extra is not None:
        ests.update({name: extra[:, i] for i, name in enumerate(EXTRA_NAMES)})
    ests["tvp_mean"] = res.tvp_q[:, 0]
    ests["tvp_median"] = res.tvp_q[:, 1]
    if res.pred_q is not None:
        ests["tvp_pred"] = res.pred_q
    diag = res.diag
    ok_tvp = np.isfinite(diag[:, 0])
    rows = []
    for ip, p in enumerate(PROBS):
        truth = pa.true_quantile_path(res.family, m, s, tail, p, res.phi, cache_dir=cache_dir)
        for name, q in ests.items():
            q = q[:, ip]                                            # (N, T)
            good = np.all(np.isfinite(q), axis=1)
            q = q[good]
            row = dict(family=res.family, knob=res.change.knob, shape=res.change.shape,
                       amount=res.change.amount, k=res.k, T=res.T, phi=res.phi, r=res.r,
                       prior=res.prior, p=p,
                       estimator=name, N=int(good.sum()),
                       true_first=float(truth[0]), true_last=float(truth[-1]))
            if q.shape[0]:
                exc = true_exceedance(res.family, q[:, -1], m[-1], s[-1], tail_last, res.phi,
                                      cache_dir)
                err = q - truth
                ch_err = (q[:, -1] - q[:, 0]) - (truth[-1] - truth[0])
                row.update(exc_med_last=float(np.median(exc)), exc_mean_last=float(exc.mean()),
                           exc_gt2x_last=float(np.mean(exc > 2 * (1 - p))),
                           bias_last=float(err[:, -1].mean()),
                           rmse_last=float(np.sqrt(np.mean(err[:, -1] ** 2))),
                           rmse_all=float(np.sqrt(np.mean(err ** 2))),
                           change_bias=float(ch_err.mean()),
                           change_rmse=float(np.sqrt(np.mean(ch_err ** 2))))
            if name.startswith("tvp"):
                lo, hi = res.tvp_q[ok_tvp, 2, ip], res.tvp_q[ok_tvp, 3, ip]
                inside = (lo <= truth) & (truth <= hi)
                row.update(cover90_last=float(inside[:, -1].mean()),
                           cover90_all=float(inside.mean()),
                           ci90_width_last=float(np.mean(hi[:, -1] - lo[:, -1])),
                           divergent_share=float(np.mean(diag[ok_tvp, 0] > 0)),
                           max_rhat=float(np.nanmax(diag[ok_tvp, 1])),
                           tvp_failures=int((~ok_tvp).sum()))
            rows.append(row)
    return rows
