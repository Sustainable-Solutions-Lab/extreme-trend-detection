"""Figures, tables and numbers for the note on the posterior predictive 99% level.

The posterior predictive p-quantile integrates the posterior over the GEV parameters first and
takes the quantile of the resulting (mixture) distribution, instead of taking a statistic (median,
mean) of the draws' own quantiles. This script scores it against the known truth in the Stage C
records refitted by ``scripts/stage_c.py --replay`` (data/output/stage_c_pred), and measures where
its uncertainty comes from.

Writes:
* data/output/report/pred_records.pkl: one row per (design, case, estimator, record)
* data/results/predictive_headline.csv: accuracy by design and estimator (committed)
* data/results/predictive_uncertainty.csv: uncertainty measures of the predictive level by design
* report/figures/pred_*.pdf, report/tables/pred_*.tex

Example: python scripts/predictive_note.py --png /tmp/previews
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_figures", ROOT / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

import matplotlib.pyplot as plt                              # noqa: E402  (style set by mf)
from matplotlib.lines import Line2D                          # noqa: E402

from tvextremes import parents as pa                         # noqa: E402
from tvextremes import stage_c                               # noqa: E402
from tvextremes.power import STATS                           # noqa: E402
from tvextremes.pseudo import time_index                     # noqa: E402

run_c = mf.run_c
PRED = ROOT / "data" / "output" / "stage_c_pred"
OUT_C, CACHE, RES, FIG, TAB, WORK = mf.OUT_C, mf.CACHE, mf.RES, mf.FIG, mf.TAB, mf.WORK
P99 = mf.P99
TARGET = 0.01

PRED_COLOR = mf.VIOLET
EST = {"tvp_pred": ("Bayes, predictive", PRED_COLOR),
       "tvp_median": ("Bayes, posterior median", mf.ORANGE),
       "tvp_mean": ("Bayes, posterior mean", "#f3a27f"),
       "trend": ("Location trend (MLE)", mf.BLUE),
       "aic_avg": ("AIC-weighted average (MLE)", mf.RED),
       "const": ("Constant (MLE)", mf.MUTED)}
DESIGNS = mf.DESIGNS
KIND_LABEL = {"none": "No change", "mean": "Mean", "width": "Width", "tail": "Tail",
              "alpha": "Position (beta)", "coupled": "Follows mean state",
              "time": "Time-driven"}


def scenario_list(dkw):
    for family in mf.FAMILIES:
        for kind in ("standard", "coupled", "time"):
            for fam, ch in mf.cases(kind, family):
                yield kind, fam, ch


def pred_path(family, ch, T=85, k=1, phi=0.5, r=0, prior="mw"):
    return PRED / mf.scenario_path(family, ch, T, k, phi, r, prior).name


def records():
    """Per-record table over all designs (cached)."""
    rows = []
    for dname, dkw in DESIGNS.items():
        T = dkw.get("T", 85)
        for kind, family, ch in scenario_list(dkw):
            path = pred_path(family, ch, **dkw)
            if not path.exists():
                continue
            res = stage_c.load(path)
            extra = stage_c.extra_estimates(res, workers=1, cache_dir=OUT_C)
            m, s, tail = pa.parameter_paths(family, ch, T)
            tl = None if tail is None else np.asarray(tail)[-1]
            phi = dkw.get("phi", 0.5)
            truth = pa.true_quantile_path(family, m, s, tail, 0.99, phi, cache_dir=CACHE)[-1]
            exc = lambda x: stage_c.true_exceedance(family, x, m[-1], s[-1], tl, phi, CACHE)   # noqa
            ests = {"const": res.mle_q[:, 0], "trend": res.mle_q[:, 1],
                    "aic_avg": extra[:, stage_c.EXTRA_NAMES.index("aic_avg")],
                    "tvp_median": res.tvp_q[:, 1], "tvp_mean": res.tvp_q[:, 0],
                    "tvp_pred": res.pred_q}
            N = res.stats.shape[0]
            # Bayesian draws in the final year
            D = res.last_draws                               # (N, draws, 3) mu, sigma, xi
            mu, sg, xi = D[..., 0], D[..., 1], D[..., 2]
            xs = res.pred_q[:, P99, -1]
            S_at_pred = stage_c.gev_survival(xs[:, None], mu, sg, xi)     # (N, draws)
            S_at_true = stage_c.gev_survival(truth, mu, sg, xi)
            srt = -np.sort(-S_at_pred, axis=1)
            n5 = max(1, int(round(0.05 * srt.shape[1])))
            top5 = srt[:, :n5].sum(axis=1) / srt.sum(axis=1)
            ess = srt.sum(axis=1) ** 2 / (srt ** 2).sum(axis=1) / srt.shape[1]
            ch_x = res.pred_chain[:, :, P99]
            ch_exc = np.where(np.isfinite(ch_x), exc(np.nan_to_num(ch_x, nan=truth)), np.nan)
            # MLE price of the true 1% level (location-trend and constant fits)
            St = pd.DataFrame(res.stats, columns=STATS)
            u = time_index(T)[-1]
            price_trend = stage_c.gev_survival(truth, St.trend_mu0 + St.mu1_hat * (u - 0.5),
                                               St.trend_sigma, St.trend_xi)
            price_const = stage_c.gev_survival(truth, St.mu_hat, St.sigma_hat, St.xi_hat)
            group = kind if kind != "standard" else ("none" if ch.amount == 0 else ch.knob)
            for name, q in ests.items():
                x = q[:, P99, -1]
                v = np.full(N, np.nan)
                ok = np.isfinite(x)
                v[ok] = exc(x[ok])
                df = pd.DataFrame(dict(design=dname, kind=kind, group=group, family=family,
                                       case=mf.change_label(ch), estimator=name, rep=np.arange(N),
                                       x=x, truth=truth, exc=v, m_last=m[-1], s_last=s[-1],
                                       tail_last=float(tl) if tl is not None and np.size(tl) == 1 else np.nan))
                if name == "tvp_pred":
                    df = df.assign(
                        d05=res.pred_exc[:, 0, P99, -1], d25=res.pred_exc[:, 1, P99, -1],
                        d50=res.pred_exc[:, 2, P99, -1], d75=res.pred_exc[:, 3, P99, -1],
                        d95=res.pred_exc[:, 4, P99, -1], top5=top5, ess=ess,
                        chain_sd=np.nanstd(np.log(np.clip(ch_exc, 1e-6, None)), axis=1, ddof=1),
                        price=S_at_true.mean(axis=1), price_draw_med=np.median(S_at_true, axis=1),
                        price_trend=price_trend, price_const=price_const,
                        xi_mean=xi.mean(axis=1), xi_gt05=(xi > 0.5).mean(axis=1),
                        divergences=res.diag[:, 0])
                rows.append(df)
    rec = pd.concat(rows, ignore_index=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rec.to_pickle(WORK / "pred_records.pkl")
    return rec


def miss(e):
    return (e > 2 * TARGET) | (e < TARGET / 2)


def headline(rec):
    g = rec.dropna(subset=["exc"]).groupby(["design", "kind", "estimator"], sort=False).exc
    out = g.agg(records="size", exc_median="median", exc_mean="mean",
                exc_p10=lambda e: e.quantile(0.1), exc_p90=lambda e: e.quantile(0.9),
                under2x=lambda e: np.mean(e > 2 * TARGET),
                over2x=lambda e: np.mean(e < TARGET / 2)).reset_index()
    out["off2x"] = out.under2x + out.over2x
    out.to_csv(RES / "predictive_headline.csv", index=False, float_format="%.5f")
    return out


def paired(rec, design, kind, a, b):
    """Difference in the missed share a - b over the same records, and its standard error."""
    r = rec[(rec.design == design) & (rec.kind == kind)]
    w = r.pivot_table(index=["family", "case", "rep"], columns="estimator", values="exc")
    w = w[[a, b]].dropna()
    d = miss(w[a]).astype(float) - miss(w[b]).astype(float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def uncertainty(rec):
    p = rec[(rec.estimator == "tvp_pred") & (rec.kind == "standard")].dropna(subset=["exc"])
    rows = []
    for dname, g in p.groupby("design", sort=False):
        le = np.log(np.clip(g.exc, 1e-6, None) / TARGET)
        rsd = (le.quantile(0.75) - le.quantile(0.25)) / 1.349
        mc = g.chain_sd / 2                                  # 4 chains -> all draws
        rows.append(dict(design=dname, records=len(g),
                         d05=g.d05.median(), d25=g.d25.median(), d50=g.d50.median(),
                         d75=g.d75.median(), d95=g.d95.median(),
                         cover90=np.mean((g.exc >= g.d05) & (g.exc <= g.d95)),
                         cover50=np.mean((g.exc >= g.d25) & (g.exc <= g.d75)),
                         top5=g.top5.median(), ess=g.ess.median(),
                         mc_sd=mc.median(), mc_sd90=mc.quantile(0.9), spread_sd=rsd,
                         mc_share=float(np.median(mc) ** 2 / rsd ** 2),
                         price_med=g.price.median(), price_mean=g.price.mean(),
                         price_p10=g.price.quantile(0.1), price_p90=g.price.quantile(0.9),
                         price_trend_med=g.price_trend.median(),
                         price_trend_p10=g.price_trend.quantile(0.1),
                         price_trend_p90=g.price_trend.quantile(0.9),
                         xi_gt05_share=np.mean(g.xi_gt05 > 0.05)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "predictive_uncertainty.csv", index=False, float_format="%.5f")
    return out


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def _save(fig, name, png):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    if png:
        fig.savefig(Path(png) / f"{name}.png", dpi=170)
    plt.close(fig)


def pick_typical(rec, design="85 yr", family="gamma", case="no change"):
    """The record whose predictive level's true exceedance is the median of its case."""
    r = rec[(rec.design == design) & (rec.family == family) & (rec.case == case)
            & (rec.estimator == "tvp_pred")].dropna(subset=["exc"])
    i = (r.exc - r.exc.median()).abs().idxmin()
    return int(r.loc[i, "rep"])


def fig_anatomy(rec, png):
    """One record: the draws' exceedance curves, the predictive (mixture) curve and the truth;
    and the distribution over draws of the exceedance probability at the predictive level."""
    family, ch = "gamma", pa.NULL
    rep = pick_typical(rec, family=family)
    res = stage_c.load(pred_path(family, ch, T=85))
    D = res.last_draws[rep]
    mu, sg, xi = D[:, 0], D[:, 1], D[:, 2]
    m, s, tail = pa.parameter_paths(family, ch, 85)
    tl = None if tail is None else np.asarray(tail)[-1]
    xs = np.linspace(res.tvp_q[rep, 1, 0, -1], 1.7 * res.pred_q[rep, P99, -1], 400)
    S = stage_c.gev_survival(xs[None, :], mu[:, None], sg[:, None], xi[:, None])   # (draws, x)
    true_S = stage_c.true_exceedance(family, xs, m[-1], s[-1], tl, 0.5, CACHE)
    x_pred, x_med = res.pred_q[rep, P99, -1], res.tvp_q[rep, 1, P99, -1]
    x_true = pa.true_quantile_path(family, m, s, tail, 0.99, 0.5, cache_dir=CACHE)[-1]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    rng = np.random.default_rng(1)
    for j in rng.choice(S.shape[0], 120, replace=False):
        a.plot(xs, S[j], color=mf.AXIS, lw=0.5, alpha=0.6, zorder=1)
    a.plot(xs, np.median(S, axis=0), color=mf.ORANGE, lw=1.8, zorder=3)
    a.plot(xs, S.mean(axis=0), color=PRED_COLOR, lw=2.0, zorder=4)
    a.plot(xs, true_S, color=mf.INK, lw=1.8, zorder=5)
    a.axhline(TARGET, color=mf.INK2, lw=0.8, ls=":", zorder=2)
    for x, c in [(x_pred, PRED_COLOR), (x_med, mf.ORANGE), (x_true, mf.INK)]:
        a.plot([x], [TARGET], "o", ms=5, color=c, mec="white", mew=1.0, zorder=6)
    a.set_yscale("log")
    a.set_ylim(1e-4, 0.3)
    a.set_xlim(xs[0], xs[-1])
    a.set_xlabel("Level of the annual maximum (baseline daily sd)")
    a.set_ylabel("Chance per year of exceeding it")
    a.set_title("Exceedance curves in the final year", loc="left")
    a.legend(handles=[Line2D([], [], color=mf.AXIS, lw=0.8, label="Individual posterior draws"),
                      Line2D([], [], color=mf.ORANGE, lw=1.8, label="Median of the draws"),
                      Line2D([], [], color=PRED_COLOR, lw=2, label="Predictive (average of draws)"),
                      Line2D([], [], color=mf.INK, lw=1.8, label="Truth")],
             loc="lower left", fontsize=6.8)

    Sp = stage_c.gev_survival(x_pred, mu, sg, xi)
    lo = 1e-6
    bins = np.logspace(np.log10(lo), np.log10(max(Sp.max(), 0.2)), 40)
    b.hist(np.clip(Sp, lo, None), bins=bins, color=PRED_COLOR, alpha=0.55, edgecolor="white", lw=0.4)
    b.set_xscale("log")
    for v, c, lab in [(Sp.mean(), PRED_COLOR, "average = 1%"),
                      (np.median(Sp), mf.ORANGE, f"median draw = {100 * np.median(Sp):.2f}%"),
                      (float(stage_c.true_exceedance(family, x_pred, m[-1], s[-1], tl, 0.5, CACHE)),
                       mf.INK, None)]:
        b.axvline(max(v, lo), color=c, lw=1.6)
    t_exc = float(stage_c.true_exceedance(family, x_pred, m[-1], s[-1], tl, 0.5, CACHE))
    srt = -np.sort(-Sp)
    top = srt[:len(srt) // 20].sum() / srt.sum()
    b.set_title("Draws' chance of exceeding the predictive level", loc="left")
    b.set_xlabel("Chance per year (log scale)")
    b.set_ylabel("Number of draws")
    b.set_ylim(0, 2.1 * np.histogram(np.clip(Sp, lo, None), bins=bins)[0].max())
    b.annotate("draws whose upper\nend point lies\nbelow the level", xy=(1.3e-6, 0.5 * (Sp <= lo).sum()), xytext=(4e-6, 0.95 * (Sp <= lo).sum()), fontsize=6.3, color=mf.INK2, arrowprops=dict(arrowstyle="-", color=mf.MUTED, lw=0.6))
    b.text(0.03, 0.97, f"average over draws: 1% (by construction)\n"
                       f"median draw: {100 * np.median(Sp):.2f}%\n"
                       f"truth: {100 * t_exc:.2f}%\n"
                       f"top 5% of draws carry {100 * top:.0f}% of the average",
           transform=b.transAxes, va="top", ha="left", fontsize=6.8, color=mf.INK2,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=mf.GRID, lw=0.6), zorder=10)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "pred_anatomy", png)
    return dict(rep=rep, x_pred=x_pred, x_med=x_med, x_true=x_true, med_draw=np.median(Sp),
                true_exc=t_exc, top5=top)


def _rangebars(ax, groups, names, getter, ylabel, hline=TARGET, band=(0.005, 0.02), ylim=None):
    """Percentile bars (10-90 thin, 25-75 thick, median dot, mean diamond) per group and estimator."""
    width = 0.8 / len(names)
    for gi, gname in enumerate(groups):
        for ei, name in enumerate(names):
            v = getter(gname, name)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            x = gi + (ei - (len(names) - 1) / 2) * width
            q10, q25, q50, q75, q90 = np.quantile(np.clip(v, 1e-6, None), [.1, .25, .5, .75, .9])
            c = EST[name][1]
            ax.plot([x, x], [q10, q90], color=c, lw=1.2, solid_capstyle="butt")
            ax.plot([x, x], [q25, q75], color=c, lw=5.0, solid_capstyle="butt", alpha=0.9)
            ax.plot([x], [q50], "o", ms=4.2, color="white", mec=c, mew=1.4, zorder=5)
            ax.plot([x], [max(v.mean(), 1e-6)], "D", ms=3.2, color=mf.INK, zorder=6)
    if band:
        ax.axhspan(*band, color=mf.GRID, alpha=0.6, zorder=0, lw=0)
    ax.axhline(hline, color=mf.INK2, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)


def _legend(ax, names, loc=None):
    h = [Line2D([], [], color=EST[n][1], lw=5, label=EST[n][0]) for n in names]
    h.append(Line2D([], [], marker="D", ls="", color=mf.INK, ms=3.2, label="Average over records"))
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=7, ncol=3,
              borderaxespad=0, handlelength=1.6, columnspacing=1.2)


def fig_exceedance(rec, png):
    names = ["tvp_pred", "tvp_median", "trend", "aic_avg"]
    groups = [g for g in ["30 yr", "45 yr", "85 yr", "85 yr, 3/yr", "85 yr, tight prior"]
              if g in set(rec.design)]
    r = rec[rec.kind == "standard"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    _rangebars(ax, [g.replace(", ", "\n") for g in groups], names,
               lambda g, n: r[(r.design == g.replace("\n", ", ")) & (r.estimator == n)].exc.values,
               "True chance per year of exceeding\nthe reported 1-in-100-year level",
               ylim=(1e-5, 0.2))
    _legend(ax, names)
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["0.001%", "0.01%", "0.1%", "1%", "10%"])
    fig.tight_layout()
    _save(fig, "pred_exceedance", png)


def fig_price(rec, png):
    """The chance each model assigns to exceeding the TRUE 1-in-100-year level: what it would
    charge, as a pure premium rate, for cover whose true risk is 1% a year."""
    groups = [g for g in ["30 yr", "45 yr", "85 yr", "85 yr, 3/yr", "85 yr, tight prior"]
              if g in set(rec.design)]
    r = rec[(rec.kind == "standard") & (rec.estimator == "tvp_pred")]
    cols = {"tvp_pred": "price", "trend": "price_trend", "const": "price_const"}
    names = list(cols)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    _rangebars(ax, [g.replace(", ", "\n") for g in groups], names,
               lambda g, n: r[r.design == g.replace("\n", ", ")][cols[n]].values,
               "Model's chance per year of exceeding\nthe true 1-in-100-year level", ylim=(1e-4, 0.3))
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%"])
    _legend(ax, names)
    fig.tight_layout()
    _save(fig, "pred_price", png)


def fig_xi(rec, png):
    """True exceedance of the predictive level against the posterior mean of the GEV shape xi in
    the final year: records whose posterior leans to heavy tails get the highest levels."""
    r = rec[(rec.kind == "standard") & (rec.estimator == "tvp_pred")
            & rec.design.isin(["30 yr", "45 yr", "85 yr"])].dropna(subset=["exc", "xi_mean"])
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for dname, c in zip(["85 yr", "45 yr", "30 yr"], [mf.RAMP[3], mf.RAMP[2], mf.RAMP[0]]):
        g = r[r.design == dname]
        ax.scatter(g.xi_mean, np.clip(g.exc, 1e-6, None), s=4, color=c, alpha=0.35, lw=0)
        q = pd.qcut(g.xi_mean, 8)
        med = g.groupby(q, observed=True).agg(x=("xi_mean", "median"),
                                              y=("exc", lambda e: np.median(np.clip(e, 1e-6, None))))
        ax.plot(med.x, med.y, color=c, lw=2.0, marker="o", ms=3.5, mec="white", mew=0.8,
                label=f"{dname} (median by octile)")
    ax.axhspan(0.005, 0.02, color=mf.GRID, alpha=0.6, zorder=0, lw=0)
    ax.axhline(TARGET, color=mf.INK2, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.2)
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["≤0.0001%", "0.001%", "0.01%", "0.1%", "1%", "10%"])
    ax.set_xlabel(r"Posterior mean of the GEV shape $\xi$ in the final year")
    ax.set_ylabel("True chance of exceeding\nthe predictive 1-in-100 level")
    ax.set_title("Records whose posterior leans to heavy tails are overstated most", loc="left")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    _save(fig, "pred_xi", png)


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------
def _pct(v, nd=0):
    return f"{100 * v:.{nd}f}"


def _pct_small(v):
    v = 100 * v
    return f"{v:.2f}" if v < 0.1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"


def table_designs(head, rec):
    names = ["tvp_pred", "tvp_median", "trend", "aic_avg"]
    lines = [r"\begin{tabular}{l rrrrr rrr r}", r"\toprule",
             r" & \multicolumn{5}{c}{Bayes, predictive} & \multicolumn{3}{c}{Missed, others} & \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
             r"Design & median & avg. & under & over & \textbf{missed} & \shortstack{post.\\median} & \shortstack{loc.\\trend} & \shortstack{AIC\\avg.} & \shortstack{pred.\ $-$\\median} \\",
             r"\midrule"]
    for d in DESIGNS:
        h = head[(head.design == d) & (head.kind == "standard")].set_index("estimator")
        if "tvp_pred" not in h.index:
            continue
        p = h.loc["tvp_pred"]
        diff, se = paired(rec, d, "standard", "tvp_pred", "tvp_median")
        lines.append(f"{d} & {_pct_small(p.exc_median)}\\% & {_pct_small(p.exc_mean)}\\% & "
                     f"{_pct(p.under2x)} & {_pct(p.over2x)} & \\textbf{{{_pct(p.off2x)}}} & "
                     + " & ".join(_pct(h.loc[n].off2x) for n in names[1:])
                     + f" & ${100 * diff:+.0f}\\pm{100 * se:.0f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "pred_designs.tex").write_text("\n".join(lines) + "\n")


def table_groups(rec, design="85 yr"):
    r = rec[rec.design == design].dropna(subset=["exc"])
    names = ["tvp_pred", "tvp_median", "trend", "aic_avg"]
    order = ["none", "mean", "width", "tail", "alpha", "coupled", "time"]
    lines = [r"\begin{tabular}{l r rrr r rrr}", r"\toprule",
             r" & & \multicolumn{4}{c}{Bayes, predictive} & \multicolumn{3}{c}{Missed, others} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
             r"Kind of change & records & median & under & over & \textbf{missed} & \shortstack{post.\\median} & \shortstack{loc.\\trend} & \shortstack{AIC\\avg.} \\",
             r"\midrule"]
    for g in order:
        gg = r[r.group == g]
        if gg.empty:
            continue
        p = gg[gg.estimator == "tvp_pred"].exc
        cells = [KIND_LABEL[g], str(len(p)), f"{_pct_small(p.median())}\\%",
                 _pct(np.mean(p > 0.02)), _pct(np.mean(p < 0.005)),
                 rf"\textbf{{{_pct(np.mean(miss(p)))}}}"]
        cells += [_pct(np.mean(miss(gg[gg.estimator == n].exc))) for n in names[1:]]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for fam in mf.FAMILIES:
        gg = r[(r.family == fam) & (r.kind == "standard")]
        p = gg[gg.estimator == "tvp_pred"].exc
        cells = [f"{mf.FAMILY[fam]} (standard cases)", str(len(p)),
                 f"{_pct_small(p.median())}\\%", _pct(np.mean(p > 0.02)), _pct(np.mean(p < 0.005)),
                 rf"\textbf{{{_pct(np.mean(miss(p)))}}}"]
        cells += [_pct(np.mean(miss(gg[gg.estimator == n].exc))) for n in names[1:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"pred_groups_{design.replace(' ', '').replace(',', '_').replace('/', '')}.tex").write_text(
        "\n".join(lines) + "\n")


def table_uncertainty(unc):
    lines = [r"\begin{tabular}{l rrr r rr r r}", r"\toprule",
             r" & \multicolumn{3}{c}{\shortstack{Draws' chance of\\exceeding the level}} & & \multicolumn{2}{c}{\shortstack{Truth\\inside}} & & \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
             r"Design & 5\% & median & 95\% & \shortstack{heavy-\\tailed} & 50\% & 90\% & \shortstack{top\\5\%} & \shortstack{Monte\\Carlo} \\",
             r"\midrule"]
    for _, u in unc.iterrows():
        lines.append(f"{u.design} & {_pct_small(u.d05)}\\% & {_pct_small(u.d50)}\\% & "
                     f"{_pct_small(u.d95)}\\% & {_pct(u.xi_gt05_share)}\\% & {_pct(u.cover50)}\\% & "
                     f"{_pct(u.cover90)}\\% & {_pct(u.top5)}\\% & $\\times${np.exp(u.mc_sd):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "pred_uncertainty.tex").write_text("\n".join(lines) + "\n")


def rain_ratios(rec):
    g = rec[(rec.family == "gamma") & (rec.kind == "standard")].dropna(subset=["x"]).copy()
    a = g.tail_last
    g["ratio"] = (a + np.sqrt(a) * (g.x - g.m_last) / g.s_last) / (
        a + np.sqrt(a) * (g.truth - g.m_last) / g.s_last)
    out = g.groupby(["design", "estimator"], sort=False).ratio.describe(
        percentiles=[.1, .25, .5, .75, .9])[["10%", "25%", "50%", "75%", "90%"]]
    out.reset_index().to_csv(RES / "predictive_rain_ratio.csv", index=False, float_format="%.4f")
    return out


def numbers(rec, head, unc, anat):
    """Key numbers quoted in the text, printed for checking."""
    pd.set_option("display.width", 220)
    print(head[head.kind == "standard"].pivot_table(index="design", columns="estimator",
                                                    values="off2x", sort=False).round(3))
    print(unc.round(4).to_string())
    print("anatomy:", anat)
    for d in DESIGNS:
        for kind in ("standard", "coupled"):
            for b in ("tvp_median", "trend", "aic_avg", "tvp_mean"):
                try:
                    diff, se = paired(rec, d, kind, "tvp_pred", b)
                except KeyError:
                    continue
                print(f"{d:20s} {kind:9s} pred - {b:10s}: {100 * diff:+5.1f} ± {100 * se:.1f}")
    # prior sensitivity on the same records
    for T in ("45 yr", "85 yr"):
        a = rec[(rec.design == T) & (rec.estimator == "tvp_pred")].set_index(["family", "case", "rep"])
        b = rec[(rec.design == f"{T}, tight prior") & (rec.estimator == "tvp_pred")].set_index(
            ["family", "case", "rep"])
        j = a[["x", "exc"]].join(b[["x", "exc"]], rsuffix="_tight", how="inner")
        if len(j):
            ratio = np.clip(j.exc_tight, 1e-6, None) / np.clip(j.exc, 1e-6, None)
            print(f"{T}: tight/mw exceedance ratio median {ratio.median():.2f}, "
                  f"10-90% {ratio.quantile(.1):.2f}-{ratio.quantile(.9):.2f}; level change median "
                  f"{(j.x_tight - j.x).median():.3f}")
    # gamma (rainfall-like): levels as a ratio to the true level on the rainfall scale. The
    # gamma parent is X = m + s (G - a)/sqrt(a), G ~ gamma(a), so G = a + sqrt(a) (X - m)/s is
    # rainfall in units of the gamma scale parameter (final-year m, s, a).
    print(rain_ratios(rec).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", default=None)
    ap.add_argument("--cached", action="store_true", help="reuse data/output/report/pred_records.pkl")
    a = ap.parse_args()
    if a.png:
        Path(a.png).mkdir(parents=True, exist_ok=True)
    rec = pd.read_pickle(WORK / "pred_records.pkl") if a.cached else records()
    head = headline(rec)
    unc = uncertainty(rec)
    TAB.mkdir(parents=True, exist_ok=True)
    table_designs(head, rec)
    table_groups(rec, "85 yr")
    table_groups(rec, "30 yr")
    table_uncertainty(unc)
    anat = fig_anatomy(rec, a.png)
    fig_exceedance(rec, a.png)
    fig_price(rec, a.png)
    fig_xi(rec, a.png)
    numbers(rec, head, unc, anat)


if __name__ == "__main__":
    main()
