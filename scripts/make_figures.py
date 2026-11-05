"""Figures and tables for the report.

Reads the committed summary tables in data/results and the raw Stage C outputs in
data/output/stage_c (estimated paths per simulated record), and writes:

* report/figures/*.pdf: the figures
* report/tables/*.tex: LaTeX tables
* data/results/stage_c_headline.csv: return-level accuracy pooled over cases, by design and
  estimator (committed; the per-record values behind it stay in data/output/report)

Example: python scripts/make_figures.py            (add --png DIR to also write PNG previews)
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tvextremes import parents as pa
from tvextremes import stage_c

ROOT = Path(__file__).resolve().parents[1]
OUT_C = ROOT / "data" / "output" / "stage_c"
CACHE = ROOT / "data" / "output" / "stage_b"
RES = ROOT / "data" / "results"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
WORK = ROOT / "data" / "output" / "report"

_spec = importlib.util.spec_from_file_location("stage_c_script", ROOT / "scripts" / "stage_c.py")
run_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_c)

# ---------------------------------------------------------------------------------------------
# Style: one fixed colour per estimator in every figure (validated categorical palette, light
# surface); dashed = a variant of the same estimator (oracle covariate, posterior mean).
# ---------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # ordinal blue (record length)

EST = {   # name: (label, colour, linestyle)
    "const": ("Constant", MUTED, "-"),
    "trend": ("Location trend", BLUE, "-"),
    "tvp_median": ("Bayesian (median)", ORANGE, "-"),
    "trend2": ("Location + width trend", AQUA, "-"),
    "cov2": ("Mean-state covariate", YELLOW, "-"),
    "trend3": ("All-parameter trend", MAGENTA, "-"),
    "recent30": ("Last 30 years", GREEN, "-"),
    "cov3": ("Covariate, all parameters", VIOLET, "-"),
    "split": ("Two halves", MUTED, "--"),
    "aic": ("Data-chosen (AIC)", RED, "-"),
    "aic_avg": ("AIC-weighted average", RED, "--"),
    "cov2_true": ("Covariate (true mean)", YELLOW, "--"),
    "cov3_true": ("Covariate, all par. (true mean)", VIOLET, "--"),
    "tvp_mean": ("Bayesian (mean)", ORANGE, "--"),
}
MAIN = ["const", "trend", "trend2", "cov2", "tvp_median"]
TABLE_EST = ["const", "recent30", "split", "trend", "trend2", "trend3", "cov2", "cov3",
             "aic", "aic_avg", "tvp_median", "tvp_mean", "cov2_true", "cov3_true"]
FAMILY = {"normal": "Normal", "gamma": "Gamma", "t": "Student-t", "beta": "Beta"}
FAMILIES = ["normal", "gamma", "t", "beta"]

# Designs scored in Stage C (all phi = 0.5 unless stated).
DESIGNS = {
    "30 yr": dict(T=30), "45 yr": dict(T=45), "85 yr": dict(T=85),
    "45 yr, 3/yr": dict(T=45, k=3, r=3), "85 yr, 3/yr": dict(T=85, k=3, r=3),
    "85 yr, indep. days": dict(T=85, phi=0.0),
    "45 yr, tight prior": dict(T=45, prior="tight"), "85 yr, tight prior": dict(T=85, prior="tight"),
}
P99 = stage_c.PROBS.index(0.99)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def change_label(ch):
    """Plain description of a change, e.g. 'mean +0.5 sd (linear)'."""
    if ch.amount == 0:
        return "no change"
    a = ch.amount
    what = {"mean": f"mean +{a:g} sd", "width": f"width ×{np.exp(a):.2f}",
            "tail": f"tail: log(shape) +{a:g}", "alpha": f"position: log α +{a:g}",
            "warm_coupled": f"warming +{a:g} sd, width & tail follow",
            "warm_time": f"warming +{a:g} sd, width & tail linear in time",
            "precip_scaling": f"wetter: all quantiles ×{np.exp(a):.2f}",
            "precip_shape": f"wetter ×{np.exp(a):.2f}, heavier tail"}[ch.knob]
    return what + {"linear": " (linear)", "step50": " (step)", "exp": " (accel.)"}[ch.shape]


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def scenario_path(family, ch, T=85, k=1, phi=0.5, r=0, prior="mw"):
    tag = "" if prior == "mw" else f"_prior-{prior}"
    return OUT_C / f"{family}_{ch.knob}_{ch.shape}_{ch.amount:g}_k{k}_T{T}_phi{phi:.2f}_r{r}{tag}.npz"


class Scenario:
    """Estimated X_p paths of every estimator for one scenario, and the truth."""

    def __init__(self, family, ch, T=85, k=1, phi=0.5, r=0, prior="mw"):
        self.family, self.ch, self.T, self.k, self.phi, self.r, self.prior = family, ch, T, k, phi, r, prior
        res = stage_c.load(scenario_path(family, ch, T, k, phi, r, prior))
        extra = stage_c.extra_estimates(res, workers=4, cache_dir=OUT_C)
        self.res = res
        self.est = {n: res.mle_q[:, i] for i, n in enumerate(stage_c.MLE_NAMES)}
        self.est.update({n: extra[:, i] for i, n in enumerate(stage_c.EXTRA_NAMES)})
        self.est["tvp_mean"], self.est["tvp_median"] = res.tvp_q[:, 0], res.tvp_q[:, 1]
        self.tvp_lo, self.tvp_hi = res.tvp_q[:, 2], res.tvp_q[:, 3]
        self.m, self.s, self.tail = pa.parameter_paths(family, ch, T)
        self.truth = np.stack([pa.true_quantile_path(family, self.m, self.s, self.tail, p, phi,
                                                     cache_dir=CACHE) for p in stage_c.PROBS])

    def exceedance(self, name, ip=P99):
        """True exceedance probability of each record's estimated X_p in the last year (NaN where
        the fit failed)."""
        x = self.est[name][:, ip, -1]
        out = np.full(x.shape, np.nan)
        ok = np.isfinite(x)
        tl = None if self.tail is None else np.asarray(self.tail)[-1]
        out[ok] = stage_c.true_exceedance(self.family, x[ok], self.m[-1], self.s[-1], tl,
                                          self.phi, CACHE)
        return out

    def panel(self, rep):
        """The simulated record (annual maxima, first column) of replicate rep."""
        return stage_c._panel(self.family, self.ch, self.k, self.T, self.phi, self.r,
                              self.res.seed, rep)


def cases(kind="standard", family=None):
    fams = [family] if family else FAMILIES
    out = []
    for f in fams:
        if kind == "standard":
            out += [(f, ch) for ch in run_c.scenarios(f)]
        elif kind == "coupled":
            out += [(f, ch) for ch in run_c.coupled_scenarios(f) if ch.amount != 0]
        else:                                                        # "time": time-driven
            out += [(f, ch) for ch in run_c.time_driven_scenarios(f)]
    return out


def records_table():
    """One row per (design, case, estimator, record): last-year exceedance of the estimated
    X_0.99, its level error and the error of the estimated change. Cached in data/output/report."""
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for dname, dkw in DESIGNS.items():
        for kind in ("standard", "coupled", "time"):
            for family, ch in cases(kind):
                if not scenario_path(family, ch, **dkw).exists():
                    continue
                sc = Scenario(family, ch, **dkw)
                tr = sc.truth[P99]
                for name in EST:
                    if dkw.get("prior", "mw") != "mw" and not name.startswith("tvp"):
                        continue                         # MLE fits do not depend on the prior
                    q = sc.est[name][:, P99]
                    exc = sc.exceedance(name)
                    df = pd.DataFrame(dict(
                        design=dname, kind=kind, family=family, knob=ch.knob, shape=ch.shape,
                        amount=ch.amount, case=change_label(ch), estimator=name,
                        rep=np.arange(q.shape[0]), exc=exc, err_last=q[:, -1] - tr[-1],
                        est_change=q[:, -1] - q[:, 0], true_change=tr[-1] - tr[0]))
                    if name.startswith("tvp"):
                        df["covered"] = (sc.tvp_lo[:, P99, -1] <= tr[-1]) & (tr[-1] <= sc.tvp_hi[:, P99, -1])
                        df["ci_width"] = sc.tvp_hi[:, P99, -1] - sc.tvp_lo[:, P99, -1]
                        df["divergences"], df["rhat"] = sc.res.diag[:, 0], sc.res.diag[:, 1]
                    rows.append(df)
    rec = pd.concat(rows, ignore_index=True)
    rec.to_pickle(WORK / "records.pkl")
    return rec


def headline(rec):
    """Pooled accuracy by design, kind and estimator."""
    g = rec.dropna(subset=["exc"]).groupby(["design", "kind", "estimator"], sort=False)
    out = g.agg(cases=("case", "nunique"), records=("exc", "size"),
                exc_median=("exc", "median"),
                under2x=("exc", lambda e: np.mean(e > 0.02)),
                over2x=("exc", lambda e: np.mean(e < 0.005)),
                within2x=("exc", lambda e: np.mean((e >= 0.005) & (e <= 0.02)))).reset_index()
    out["off2x"] = out.under2x + out.over2x
    out.to_csv(RES / "stage_c_headline.csv", index=False, float_format="%.4f")
    return out


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def _legend_handles(names, band=False):
    h = [Line2D([], [], color=INK, lw=2.2, label="Truth")]
    h += [Line2D([], [], color=EST[n][1], ls=EST[n][2], lw=1.5, label=EST[n][0]) for n in names]
    if band:
        h.append(Patch(color=ORANGE, alpha=0.15, lw=0, label="Bayesian 90% interval"))
    return h


def _save(fig, name, png):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    if png:
        fig.savefig(Path(png) / f"{name}.png", dpi=110)
    plt.close(fig)


def representative(sc, names):
    """The record of median difficulty: rank records by the mean over the given estimators of
    |log(exceedance / 1%)| in the last year and take the middle one (not the best or worst)."""
    score = np.nanmean([np.abs(np.log(sc.exceedance(n) / 0.01)) for n in names], axis=0)
    order = np.argsort(np.where(np.isfinite(score), score, np.inf))
    return int(order[np.isfinite(score).sum() // 2])


EXAMPLES = [("normal", pa.Change("mean", "linear", 0.5)),
            ("gamma", pa.Change("width", "linear", 0.2)),
            ("t", pa.Change("tail", "linear", 1.0)),
            ("normal", pa.Change("warm_coupled", "exp", 0.5))]


def fig_examples(png):
    names = ["const", "trend", "trend2", "cov2", "tvp_median"]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
    for ax, (family, ch) in zip(axes.flat, EXAMPLES):
        sc = Scenario(family, ch, T=85)
        rep = representative(sc, names[1:])
        X = sc.panel(rep)
        yr = np.arange(1, sc.T + 1)
        ax.fill_between(yr, sc.tvp_lo[rep, P99], sc.tvp_hi[rep, P99], color=ORANGE, alpha=0.13, lw=0)
        ax.plot(yr, X[:, 0], "o", ms=2.6, color=INK2, mec="white", mew=0.4, zorder=3)
        for n in names:
            ax.plot(yr, sc.est[n][rep, P99], color=EST[n][1], ls=EST[n][2], zorder=4)
        ax.plot(yr, sc.truth[P99], color=INK, lw=2.2, zorder=5)
        ax.set_title(f"{FAMILY[family]}: {change_label(ch)}", loc="left")
        ax.set_xlim(1, sc.T)
    for ax in axes[1]:
        ax.set_xlabel("Year of record")
    for ax in axes[:, 0]:
        ax.set_ylabel("Annual maximum (baseline daily sd)")
    h = _legend_handles(names, band=True)
    h.insert(1, Line2D([], [], marker="o", ls="", ms=3, color=INK2, label="Annual maxima"))
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "fig01_examples", png)


# ---------------------------------------------------------------------------------------------
# Single records (like fig01_examples): annual maxima as points, truth and estimates as lines
# ---------------------------------------------------------------------------------------------
REC_EST = ["trend", "aic_avg", "tvp_median"]
REC_SHORT = {"const": "Const.", "trend": "Loc.", "trend2": "L+W", "cov2": "Cov.",
             "aic_avg": "AIC", "tvp_median": "Bayes"}


def pick_record(sc, names, q=0.5):
    """Record at quantile q of difficulty (mean over names of |log(exceedance / 1%)| in the last
    year): q = 0.5 is the typical record, 0.1 an easy one, 0.9 a hard one."""
    score = np.nanmean([np.abs(np.log(sc.exceedance(n) / 0.01)) for n in names], axis=0)
    ok = np.isfinite(score)
    order = np.argsort(np.where(ok, score, np.inf))
    return int(order[int(round(q * (ok.sum() - 1)))])


def _pct(x):
    return "<0.1%" if x < 0.001 else f"{100 * x:.1f}%"


def fig_records(name, panels, names, png, ncols=3, panel_h=1.9, ylabel_all=False):
    """Grid of single records. panels: (title, family, change, design kwargs, difficulty quantile)."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 0.75 + panel_h * nrows), squeeze=False)
    for ax, (title, family, ch, kw, q) in zip(axes.flat, panels):
        sc = get_scenario(family, ch, **kw)
        if sc is None:
            ax.axis("off")
            continue
        rep = pick_record(sc, names, q)
        yr = np.arange(1, sc.T + 1)
        if "tvp_median" in names:
            ax.fill_between(yr, sc.tvp_lo[rep, P99], sc.tvp_hi[rep, P99], color=ORANGE, alpha=0.13, lw=0)
        ax.plot(yr, sc.panel(rep)[:, 0], "o", ms=2.2, color=INK2, mec="white", mew=0.3, zorder=3)
        for n in names:
            ax.plot(yr, sc.est[n][rep, P99], color=EST[n][1], ls=EST[n][2], lw=1.4, zorder=4)
        ax.plot(yr, sc.truth[P99], color=INK, lw=2.0, zorder=5)
        ax.set_xlim(1, sc.T)
        lines = np.concatenate([sc.panel(rep)[:, 0], sc.truth[P99]] + [sc.est[n][rep, P99] for n in names])
        lo, hi = np.nanmin(lines), np.nanmax(lines)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.15 * (hi - lo))    # Bayes band may run off the top
        ax.set_title(title, loc="left", fontsize=7.5, pad=11)
        ax.text(0, 1.015, "   ".join(f"{REC_SHORT[n]} {_pct(sc.exceedance(n)[rep])}" for n in names),
                transform=ax.transAxes, va="bottom", fontsize=5.9, color=INK2, in_layout=False)
        ax.tick_params(labelsize=6.5)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Year of record", fontsize=7)
    for ax in (axes.flat if ylabel_all else axes[:, 0]):
        ax.set_ylabel("Annual maximum (daily sd)", fontsize=7)
    h = _legend_handles(names, band="tvp_median" in names)
    h.insert(1, Line2D([], [], marker="o", ls="", ms=3, color=INK2, label="Annual maxima"))
    fig.legend(handles=h, loc="lower center", ncol=3 if len(h) > 4 else len(h), bbox_to_anchor=(0.5, 0.0),
               fontsize=7)
    fig.tight_layout(rect=(0, 0.075 if nrows > 1 else 0.14, 1, 1), h_pad=1.6)
    _save(fig, name, png)


RECORD_PARENT = {
    "normal": [pa.Change("mean", "linear", 0.0), pa.Change("mean", "linear", 0.5),
               pa.Change("mean", "step50", 0.5), pa.Change("width", "linear", 0.2),
               pa.Change("warm_coupled", "exp", 0.5), pa.Change("warm_time", "exp", 0.5)],
    "gamma": [pa.Change("mean", "linear", 0.0), pa.Change("mean", "linear", 0.5),
              pa.Change("mean", "step50", 0.5), pa.Change("width", "linear", 0.2),
              pa.Change("tail", "linear", 1.0), pa.Change("precip_scaling", "exp", 0.2)],
    "t": [pa.Change("mean", "linear", 0.0), pa.Change("mean", "linear", 0.5),
          pa.Change("mean", "step50", 0.5), pa.Change("width", "linear", 0.2),
          pa.Change("tail", "linear", 1.0), pa.Change("warm_coupled", "exp", 0.5)],
    "beta": [pa.Change("mean", "linear", 0.0), pa.Change("mean", "linear", 0.25),
             pa.Change("mean", "step50", 0.25), pa.Change("width", "linear", 0.1),
             pa.Change("tail", "linear", 0.2), pa.Change("alpha", "linear", 0.5)],
}


def record_title(ch):
    """Short panel title for the single-record figures."""
    a = ch.amount
    if a == 0:
        return "no change"
    return {"mean": f"mean +{a:g} sd" + (" (step)" if ch.shape == "step50" else " (linear)"),
            "width": f"width ×{np.exp(a):.2f}", "tail": f"tail: log(shape) +{a:g}",
            "alpha": f"position: log α +{a:g}", "warm_coupled": "warming, width follows mean",
            "warm_time": "warming, width follows time", "precip_scaling": f"wetter ×{np.exp(a):.2f}",
            "precip_shape": "wetter, heavier tail"}[ch.knob]


def fig_records_all(png):
    for family, chs in RECORD_PARENT.items():
        panels = [(record_title(ch), family, ch, dict(T=85), 0.5) for ch in chs]
        fig_records(f"fig_records_{family}", panels, REC_EST, png)
    luck = []
    for family, ch in [("normal", pa.Change("mean", "linear", 0.5)),
                       ("normal", pa.Change("warm_coupled", "exp", 0.5))]:
        for q, lab in [(0.1, "an easy record"), (0.5, "a typical record"), (0.9, "a hard record")]:
            luck.append((f"{record_title(ch).split(',')[0].split(' (')[0]}: {lab}", family, ch, dict(T=85), q))
    fig_records("fig_records_luck", luck, REC_EST, png)
    length = []
    for family, ch in [("normal", pa.Change("mean", "linear", 0.5)),
                       ("normal", pa.Change("warm_coupled", "exp", 0.5))]:
        for T in (30, 45, 85):
            length.append((f"{T} years: {record_title(ch).split(',')[0].split(' (')[0]}", family, ch, dict(T=T), 0.5))
    fig_records("fig_records_length", length, REC_EST, png)
    wrong = [(f"{FAMILY[fam]}: {record_title(ch)}", fam, ch, dict(T=85), 0.5) for fam, ch in [
        ("normal", pa.Change("warm_coupled", "exp", 0.5)), ("normal", pa.Change("warm_time", "exp", 0.5)),
        ("normal", pa.Change("width", "linear", 0.2)), ("normal", pa.Change("mean", "step50", 0.5)),
        ("gamma", pa.Change("tail", "linear", 1.0)), ("normal", pa.Change("mean", "linear", 0.0))]]
    fig_records("fig_records_wrong", wrong, ["trend2", "cov2", "aic_avg"], png)


def _spread(ax, sc, name, color):
    q = sc.est[name][:, P99]
    q = q[np.all(np.isfinite(q), axis=1)]
    yr = np.arange(1, sc.T + 1)
    lo, q25, med, q75, hi = np.quantile(q, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.fill_between(yr, lo, hi, color=color, alpha=0.14, lw=0)
    ax.fill_between(yr, q25, q75, color=color, alpha=0.28, lw=0)
    ax.plot(yr, med, color=color)
    ax.plot(yr, sc.truth[P99], color=INK, lw=2.0)
    ax.set_xlim(1, sc.T)


ROW_LABEL = {("normal", "mean"): "Normal: mean +0.5 sd\nX$_{0.99}$",
             ("normal", "warm_coupled"): "Normal: warming,\nwidth & tail follow",
             ("gamma", "precip_scaling"): "Gamma: wetter,\nquantiles ×1.22"}


def fig_spread(png):
    rows = [("normal", pa.Change("mean", "linear", 0.5)),
            ("normal", pa.Change("warm_coupled", "exp", 0.5)),
            ("gamma", pa.Change("precip_scaling", "exp", 0.2))]
    names = ["const", "trend", "trend2", "cov2", "tvp_median"]
    fig, axes = plt.subplots(len(rows), len(names), figsize=(6.5, 5.6), sharex=True, sharey="row")
    for i, (family, ch) in enumerate(rows):
        sc = Scenario(family, ch, T=85)
        for j, n in enumerate(names):
            _spread(axes[i, j], sc, n, EST[n][1])
            if i == 0:
                axes[i, j].set_title(EST[n][0], fontsize=7.5)
        axes[i, 0].set_ylabel(ROW_LABEL[(family, ch.knob)], fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    h = [Line2D([], [], color=INK, lw=2, label="Truth"),
         Line2D([], [], color=INK2, lw=1.5, label="Median of 100 estimates"),
         Patch(color=INK2, alpha=0.35, lw=0, label="25–75%"),
         Patch(color=INK2, alpha=0.15, lw=0, label="5–95%")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, "fig02_spread", png)


_SCEN = {}


def get_scenario(family, ch, **kw):
    """Scenario, loaded once per run (or None if that scenario was not run)."""
    key = (family, ch, tuple(sorted(kw.items())))
    if key not in _SCEN:
        _SCEN[key] = Scenario(family, ch, **kw) if scenario_path(family, ch, **kw).exists() else None
    return _SCEN[key]


def short_label(family, ch):
    """Two-line row label: parent, then the change."""
    a = ch.amount
    what = {"mean": f"mean +{a:g} sd", "width": f"width ×{np.exp(a):.2f}",
            "tail": f"tail: log(shape) +{a:g}", "alpha": f"position: log α +{a:g}",
            "warm_coupled": "warming, width follows mean", "warm_time": "warming, width follows time",
            "precip_scaling": f"wetter ×{np.exp(a):.2f}",
            "precip_shape": "wetter, heavier tail"}[ch.knob] if a else "no change"
    coupled = ch.knob in ("warm_coupled", "warm_time", "precip_scaling", "precip_shape")
    shape = {"linear": " (linear)" if coupled else "", "step50": " (step)",
             "exp": " (accel.)"}[ch.shape] if a else ""
    if ", " in what:                                # long labels: shape after the first part,
        head, rest = what.split(", ", 1)            # the rest on a third line
        return f"{FAMILY[family]}\n{head}{shape},\n{rest}"
    return f"{FAMILY[family]}\n{what}{shape}"


def fig_bands(name, rows, cols, names, png, row_height=1.45, note=None):
    """Small multiples of the spread of the estimated X_0.99 paths over 100 records (median,
    25-75%, 5-95%) against the truth. rows: (family, change); cols: (title, design kwargs);
    names: estimators overlaid in each panel. Rows share a y-axis."""
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6.5, 0.7 + row_height * len(rows)),
                             sharey="row", squeeze=False)
    for i, (family, ch) in enumerate(rows):
        for j, (title, kw) in enumerate(cols):
            ax = axes[i, j]
            sc = get_scenario(family, ch, **kw)
            if sc is None:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center",
                        color=MUTED, fontsize=7)
                continue
            for n in names:
                _spread(ax, sc, n, EST[n][1])
            if i == 0:
                ax.set_title(title, fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("Year")
        axes[i, 0].set_ylabel(short_label(family, ch), fontsize=7.5)
    h = [Line2D([], [], color=INK, lw=2, label="Truth")]
    h += [Line2D([], [], color=EST[n][1], ls=EST[n][2], label=f"{EST[n][0]}: median, 25–75%, 5–95%")
          for n in names]
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.01),
               fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, name, png)


BY_LENGTH = [("30 years", dict(T=30)), ("45 years", dict(T=45)), ("85 years", dict(T=85))]


def fig_record_length(png):
    rows = [("normal", pa.Change("mean", "linear", 0.5)),
            ("normal", pa.Change("warm_coupled", "exp", 0.5))]
    fig_bands("fig03_record_length", rows, BY_LENGTH, ["trend", "tvp_median"], png, row_height=1.6)


def fig_paths_by_parent(png):
    """One figure per parent: its standard changes (rows) by record length (columns)."""
    for fam in FAMILIES:
        rows = [(fam, ch) for ch in run_c.scenarios(fam) if ch.amount != 0]
        fig_bands(f"fig_paths_{fam}", rows, BY_LENGTH, ["trend", "tvp_median"], png)


def fig_paths_covariate(png):
    warm = [(f, ch) for f in ("normal", "t", "beta") for ch in run_c.coupled_scenarios(f) if ch.amount]
    wet = [("gamma", ch) for ch in run_c.coupled_scenarios("gamma") if ch.amount]
    fig_bands("fig_paths_warming", warm, BY_LENGTH, ["trend", "cov2"], png)
    fig_bands("fig_paths_wetter", wet, BY_LENGTH, ["trend", "cov2"], png)


def fig_paths_values_per_year(png):
    rows = [("normal", pa.Change("mean", "linear", 0.5)), ("gamma", pa.Change("width", "linear", 0.2)),
            ("t", pa.Change("tail", "linear", 1.0)), ("beta", pa.Change("alpha", "linear", 0.5))]
    cols = [("45 yr, 1/yr", dict(T=45)), ("45 yr, 3/yr", dict(T=45, k=3, r=3)),
            ("85 yr, 1/yr", dict(T=85)), ("85 yr, 3/yr", dict(T=85, k=3, r=3))]
    fig_bands("fig_paths_values_per_year", rows, cols, ["trend", "tvp_median"], png)


def fig_paths_prior(png):
    rows = [("normal", pa.Change("mean", "linear", 0.5)), ("gamma", pa.Change("mean", "step50", 0.5)),
            ("t", pa.Change("tail", "linear", 1.0)), ("beta", pa.Change("width", "linear", 0.1))]
    cols = [("45 yr, authors' prior", dict(T=45)), ("45 yr, tighter prior", dict(T=45, prior="tight")),
            ("85 yr, authors' prior", dict(T=85)), ("85 yr, tighter prior", dict(T=85, prior="tight"))]
    fig_bands("fig_paths_prior", rows, cols, ["tvp_median"], png)
    warm = [("normal", pa.Change("warm_coupled", "exp", 0.5)), ("t", pa.Change("warm_coupled", "exp", 0.5)),
            ("gamma", pa.Change("precip_scaling", "exp", 0.2)), ("gamma", pa.Change("precip_shape", "exp", 0.2))]
    cols = [("85 yr, authors' prior", dict(T=85)), ("85 yr, tighter prior", dict(T=85, prior="tight"))]
    fig_bands("fig_paths_prior_covariate", warm, cols, ["tvp_median", "cov2"], png)


def fig_power(png):
    d = pd.read_csv(RES / "stage_b_summary.csv")
    d = d[(d.k == 1) & (d.r == 0) & (d.phi == 0.5) & (d["shape"] == "linear")]
    knobs = [("mean", "Mean shift"), ("width", "Width change"), ("tail", "Tail change")]
    tests = [("nyblom_nominal", "Nyblom (paper's test)", "-", "o"),
             ("trend_lr_nominal", "Location-trend LR", "--", "s")]
    fig, axes = plt.subplots(len(knobs), 4, figsize=(6.5, 5.0), sharey=True)
    for i, (knob, ktitle) in enumerate(knobs):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i, j]
            s = d[(d.family == fam) & ((d.knob == knob) | ((d.knob == "mean") & (d.amount == 0)))]
            if knob not in set(d[d.family == fam].knob):
                ax.set_visible(False)
                continue
            for (T, c) in zip([30, 45, 85, 150], RAMP):
                st = s[s["T"] == T].assign(x=lambda f: f.d_rl100_indep.abs()).sort_values("x")
                for col, _, ls, mk in tests:
                    ax.plot(st.x, st[col], ls=ls, marker=mk, ms=2.5, color=c, lw=1.2)
            ax.axhline(0.05, color=MUTED, lw=0.8)
            ax.set_ylim(0, 1.02)
            if i == 0 or (knob == "tail" and fam == "gamma"):
                ax.set_title(FAMILY[fam] if i == 0 else "", fontsize=8)
            if j == 0 or (knob == "tail" and fam == "gamma"):
                ax.set_ylabel(f"{ktitle}\nrejection rate")
                ax.tick_params(labelleft=True)
            if i == len(knobs) - 1:
                ax.set_xlabel("|change in X$_{0.99}$| (daily sd)")
    h = [Line2D([], [], color=c, lw=1.5, label=f"{T} years") for T, c in zip([30, 45, 85, 150], RAMP)]
    h += [Line2D([], [], color=INK2, ls=ls, marker=mk, ms=3, lw=1.2, label=lab) for _, lab, ls, mk in tests]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, "fig04_power", png)


def fig_size(png):
    d = pd.read_csv(RES / "stage_b_summary.csv")
    n = d[d.amount == 0]
    designs = [(1, 0.0, 0, 85), (1, 0.5, 0, 30), (1, 0.5, 0, 45), (1, 0.5, 0, 85), (1, 0.5, 0, 150),
               (1, 0.8, 0, 85), (3, 0.0, 0, 85), (3, 0.5, 0, 30), (3, 0.5, 0, 85), (3, 0.8, 0, 85),
               (10, 0.8, 0, 85), (10, 0.8, 3, 85)]
    tests = [("nyblom_nominal", "Nyblom", BLUE), ("trend_lr_nominal", "Location-trend LR", ORANGE),
             ("split_nominal", "Two-halves", AQUA), ("mk_nominal", "Mann–Kendall", YELLOW)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = []
    for y, (k, phi, r, T) in enumerate(designs):
        s = n[(n.k == k) & (n.phi == phi) & (n.r == r) & (n["T"] == T)]
        labels.append(f"k={k}, φ={phi:g}, T={T}" + (", declustered" if r else ""))
        for t, (col, lab, c) in enumerate(tests):
            yy = y + (t - 1.5) * 0.17
            ax.plot([s[col].min(), s[col].max()], [yy, yy], color=c, lw=1.2)
            ax.plot(s[col].mean(), yy, "o", ms=4, color=c, mec="white", mew=0.6)
    ax.axvline(0.05, color=INK, lw=0.9)
    ax.set_yticks(range(len(designs)), labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([0.04, 0.05, 0.1, 0.2, 0.5], ["4%", "5%", "10%", "20%", "50%"])
    ax.set_xlabel("False-rejection rate with no change (mean and range over the four parents)")
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], color=c, marker="o", ms=4, lw=1.2, label=lab) for _, lab, c in tests]
    ax.set_xlim(0.035, 0.65)
    ax.legend(handles=h, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig05_size", png)


def fig_exceedance(rec, png):
    names = ["const", "recent30", "trend", "trend2", "trend3", "cov2", "aic_avg", "tvp_median"]
    panels = [("30 yr", "standard"), ("45 yr", "standard"), ("85 yr", "standard"),
              ("85 yr, 3/yr", "standard"), ("85 yr", "coupled")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 3.0), sharey=True)
    for ax, (dname, kind) in zip(axes, panels):
        s = rec[(rec.design == dname) & (rec.kind == kind)]
        for y, n in enumerate(names):
            e = s[s.estimator == n].exc.dropna().clip(1e-5, 0.5)
            if not len(e):
                continue
            q05, q25, q50, q75, q95 = np.quantile(e, [0.05, 0.25, 0.5, 0.75, 0.95])
            c = EST[n][1]
            ax.plot([q05, q95], [y, y], color=c, lw=1.0)
            ax.plot([q25, q75], [y, y], color=c, lw=5, solid_capstyle="butt", alpha=0.85)
            ax.plot(q50, y, "o", ms=4.5, color="white", mec=c, mew=1.4, zorder=4)
        ax.axvspan(0.005, 0.02, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.axvline(0.01, color=INK, lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(2e-4, 0.3)
        ax.set_xticks([1e-3, 1e-2, 1e-1], ["0.1%", "1%", "10%"])
        ax.set_title(dname + ("\ncovariate cases" if kind == "coupled" else "\nstandard cases"), fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), [EST[n][0] for n in names])
    axes[0].invert_yaxis()
    fig.supxlabel("True probability that the estimated 99% level is exceeded in the last year "
                  "(ideal 1%; shaded: within a factor 2)", fontsize=7.5, color=INK2)
    fig.tight_layout()
    _save(fig, "fig06_exceedance", png)


def fig_change(rec, png):
    names = ["trend", "trend2", "cov2", "tvp_median"]
    s0 = rec[(rec.design == "85 yr") & rec.estimator.isin(names)]
    ncase = [s0[s0.family == f].case.nunique() for f in FAMILIES]
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8.2), gridspec_kw=dict(height_ratios=ncase))
    for ax, fam in zip(axes, FAMILIES):
        s = s0[s0.family == fam]
        cs = list(dict.fromkeys(s.case))
        for y, case in enumerate(cs):
            sc = s[s.case == case]
            for t, n in enumerate(names):
                e = sc[sc.estimator == n].est_change.dropna()
                yy = y + (t - 1.5) * 0.18
                lo, med, hi = np.quantile(e, [0.1, 0.5, 0.9])
                ax.plot([lo, hi], [yy, yy], color=EST[n][1], lw=1.1)
                ax.plot(med, yy, "o", ms=3.5, color=EST[n][1], mec="white", mew=0.5)
            ax.plot(sc.true_change.iloc[0], y, "|", ms=13, mew=2.2, color=INK, zorder=5)
        ax.set_yticks(range(len(cs)), cs, fontsize=7)
        ax.set_ylim(len(cs) - 0.5, -0.5)
        ax.set_title(FAMILY[fam] + " parent", fontsize=8, loc="left")
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=AXIS, lw=0.8)
    axes[-1].set_xlabel("Change in X$_{0.99}$, first year to last (baseline daily sd), 85-year records")
    h = [Line2D([], [], marker="|", ls="", ms=9, mew=2, color=INK, label="Truth")]
    h += [Line2D([], [], color=EST[n][1], marker="o", ms=3.5, lw=1.1, label=EST[n][0]) for n in names]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=0.8)
    _save(fig, "fig08_change", png)


def fig_bayes(rec, png):
    s = rec[(rec.estimator == "tvp_median") & (rec.kind == "standard")]
    designs = ["30 yr", "45 yr", "85 yr", "45 yr, 3/yr", "85 yr, 3/yr", "85 yr, indep. days",
               "45 yr, tight prior", "85 yr, tight prior"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    for y, dname in enumerate(designs):
        sd = s[s.design == dname]
        if not len(sd):
            continue
        for t, fam in enumerate(FAMILIES):
            sf = sd[sd.family == fam]
            yy = y + (t - 1.5) * 0.17
            c = [BLUE, ORANGE, AQUA, YELLOW][t]
            axes[0].plot(sf.covered.mean(), yy, "o", ms=4, color=c, mec="white", mew=0.5)
            # interval width relative to the spread of the point estimate
            if not len(sf):
                continue
            e = sf.err_last.to_numpy(float)
            spread = np.nanquantile(e, 0.95) - np.nanquantile(e, 0.05)
            axes[1].plot(sf.ci_width.median() / spread, yy, "o", ms=4,
                         color=c, mec="white", mew=0.5)
    axes[0].axvline(0.9, color=INK, lw=0.9)
    axes[0].set_xlim(0.75, 1.0)
    axes[0].set_xlabel("Coverage of the 90% interval for X$_{0.99}$, last year")
    axes[1].axvline(1.0, color=INK, lw=0.9)
    axes[1].set_xlabel("Interval width ÷ 5–95% range of errors")
    axes[0].set_yticks(range(len(designs)), designs)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
    h = [Line2D([], [], color=c, marker="o", ls="", ms=4, label=FAMILY[f])
         for c, f in zip([BLUE, ORANGE, AQUA, YELLOW], FAMILIES)]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.06))
    fig.tight_layout()
    _save(fig, "fig09_bayes_intervals", png)


def fig_heatmap(head, png):
    names = ["const", "recent30", "split", "trend", "trend2", "trend3", "cov2", "cov3", "aic", "aic_avg",
             "tvp_median", "cov2_true"]
    cols = [("30 yr", "standard"), ("45 yr", "standard"), ("85 yr", "standard"), ("85 yr, 3/yr", "standard"),
            ("30 yr", "coupled"), ("45 yr", "coupled"), ("85 yr", "coupled")]
    collab = ["30", "45", "85", "85\n3/yr", "30", "45", "85"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5",
                                                      "#1c5cab", "#0d366b"])
    head = head.assign(off2x=head.under2x + head.over2x)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.0), sharey=True)
    for ax, (metric, title) in zip(axes, [("off2x", "Missed > 2× either way"),
                                         ("under2x", "Understated > 2×\n(true exceedance > 2%)"),
                                         ("over2x", "Overstated > 2×\n(true exceedance < 0.5%)")]):
        M = np.full((len(names), len(cols)), np.nan)
        for j, (dname, kind) in enumerate(cols):
            for i, n in enumerate(names):
                v = head[(head.design == dname) & (head.kind == kind) & (head.estimator == n)][metric]
                if len(v):
                    M[i, j] = v.iloc[0]
        ax.imshow(M, cmap=cmap, vmin=0, vmax=0.9, aspect="auto")
        for i in range(len(names)):
            for j in range(len(cols)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if M[i, j] > 0.45 else INK)
        ax.set_xticks(range(len(cols)), collab, fontsize=6.5)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=7.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axvline(3.5, color="white", lw=3)
        ax.text(1.5, len(names) + 0.9, "standard", ha="center", va="top", fontsize=6.5, color=INK2)
        ax.text(5.0, len(names) + 0.9, "covariate", ha="center", va="top", fontsize=6.5, color=INK2)
    axes[0].set_yticks(range(len(names)), [EST[n][0] for n in names], fontsize=7)
    fig.text(0.5, 0.01, "Share of simulated records, %, pooled over cases. Columns: record length in years; "
             "3/yr: three values per year.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig07_heatmap", png)


def fig_prior_phi(head, png):
    """Appendix: tight vs authors' prior (Bayesian), and independent vs correlated days."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    pairs = [("45 yr", "45 yr, tight prior"), ("85 yr", "85 yr, tight prior")]
    ax = axes[0]
    for y, (a, b) in enumerate(pairs):
        for t, (metric, lab, c) in enumerate([("under2x", "understated > 2×", BLUE),
                                               ("over2x", "overstated > 2×", ORANGE)]):
            va = head[(head.design == a) & (head.kind == "standard") & (head.estimator == "tvp_median")][metric]
            vb = head[(head.design == b) & (head.kind == "standard") & (head.estimator == "tvp_median")][metric]
            if len(va) and len(vb):
                yy = y + (t - 0.5) * 0.25
                ax.annotate("", xy=(vb.iloc[0], yy), xytext=(va.iloc[0], yy),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
                ax.plot(va.iloc[0], yy, "o", ms=4, color=c, mec="white", mew=0.5, label=lab if y == 0 else None)
    ax.set_yticks(range(len(pairs)), [a for a, _ in pairs])
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_xlabel("Share of records (Bayesian median)")
    ax.set_title("Authors' prior (dot) → tighter prior (arrow head)", fontsize=7.5)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    names = ["const", "trend", "trend2", "cov2", "aic_avg", "tvp_median"]
    off = head.under2x + head.over2x
    for y, n in enumerate(names):
        va = off[(head.design == "85 yr") & (head.kind == "standard") & (head.estimator == n)]
        vb = off[(head.design == "85 yr, indep. days") & (head.kind == "standard") & (head.estimator == n)]
        if len(va) and len(vb):
            ax.plot([va.iloc[0], vb.iloc[0]], [y, y], color=EST[n][1], lw=1.2)
            ax.plot(va.iloc[0], y, "o", ms=4.5, color=EST[n][1], mec="white", mew=0.5)
            ax.plot(vb.iloc[0], y, "s", ms=4.5, color="white", mec=EST[n][1], mew=1.3)
    ax.set_yticks(range(len(names)), [EST[n][0] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Share of records missed > 2× either way")
    ax.set_title("85 yr: correlated days φ = 0.5 (●) vs independent (□)", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "figA1_prior_phi", png)


# ---------------------------------------------------------------------------------------------
# Using the wrong model: cases classified by the true process
# ---------------------------------------------------------------------------------------------
PROCESSES = [   # (key, label, models with the right form: time-based, mean-state-based)
    ("none", "No change", "both"),
    ("mean_lin", "Mean shift, linear in time", "both"),
    ("mean_step", "Mean shift, step", "mean state"),
    ("width_const_mean", "Width change, mean constant", "time"),
    ("tail_const_mean", "Tail change, mean constant", "neither"),
    ("state_lin", "Follows mean state, mean linear in time", "both"),
    ("state_acc", "Follows mean state, mean accelerating", "mean state"),
    ("time_acc", "Follows time, mean accelerating", "neither"),
]
PROC_LABEL = {k: lab for k, lab, _ in PROCESSES}
WRONG_EST = ["const", "trend", "trend2", "tvp_median", "cov2", "aic", "aic_avg", "cov2_true"]
# right form: estimator -> set of process keys for which its form contains the truth (approximately)
RIGHT_FORM = {
    "const": {"none"},
    "trend": {"none", "mean_lin"},
    "trend2": {"none", "mean_lin", "width_const_mean", "state_lin"},
    "tvp_median": {"none", "mean_lin", "mean_step", "width_const_mean", "tail_const_mean", "state_lin",
                   "state_acc", "time_acc"},
    "cov2": {"none", "mean_lin", "mean_step", "state_lin", "state_acc"},
    "cov2_true": {"none", "mean_lin", "mean_step", "state_lin", "state_acc"},
}


def process_of(knob, shape, amount):
    if amount == 0:
        return "none"
    if knob == "mean":
        return "mean_lin" if shape == "linear" else "mean_step"
    if knob in ("width", "tail"):
        return f"{knob}_const_mean"
    if knob in ("warm_coupled", "precip_scaling", "precip_shape"):
        return "state_lin" if shape == "linear" else "state_acc"
    if knob == "warm_time":
        return "time_acc"
    return None                                     # beta position change: not classified


def wrong_model_table(rec):
    r = rec.assign(process=[process_of(k, sh, a) for k, sh, a in zip(rec.knob, rec["shape"], rec.amount)])
    r = r[r.process.notna() & r.design.isin(["30 yr", "45 yr", "85 yr"]) & r.estimator.isin(WRONG_EST)]
    g = r.dropna(subset=["exc"]).groupby(["design", "process", "estimator"])
    out = g.agg(cases=("case", "nunique"), under2x=("exc", lambda e: np.mean(e > 0.02)),
                over2x=("exc", lambda e: np.mean(e < 0.005)), exc_median=("exc", "median")).reset_index()
    out["off2x"] = out.under2x + out.over2x
    return out


def aic_choices(rec):
    """Share of records for which AIC picks each candidate, by design and process (cached)."""
    path = WORK / "aic_choice.pkl"
    keys = rec[rec.design.isin(["30 yr", "45 yr", "85 yr"]) & (rec.estimator == "trend")][
        ["design", "family", "knob", "shape", "amount"]].drop_duplicates()
    old = pd.read_pickle(path) if path.exists() else pd.DataFrame()
    rows = [old] if len(old) else []
    from concurrent.futures import ProcessPoolExecutor
    from tvextremes import critical_values as cv
    with ProcessPoolExecutor(max_workers=4, initializer=cv._single_thread) as ex:
        for _, kr in keys.iterrows():
            if len(old) and ((old.design == kr.design) & (old.family == kr.family) & (old.knob == kr.knob)
                             & (old["shape"] == kr["shape"]) & np.isclose(old.amount, kr.amount)).any():
                continue
            T = DESIGNS[kr.design]["T"]
            ch = pa.Change(kr.knob, kr["shape"], float(kr.amount))
            res = stage_c.load(scenario_path(kr.family, ch, T=T))
            jobs = [(kr.family, ch, 1, T, 0.5, 0, res.seed, i) for i in range(res.stats.shape[0])]
            ch_idx = np.array([c for _, c in ex.map(stage_c.aic_choice, jobs, chunksize=5)])
            rows.append(pd.DataFrame([dict(design=kr.design, family=kr.family, knob=kr.knob,
                                           shape=kr["shape"], amount=kr.amount, model=m,
                                           share=float(np.mean(ch_idx == i)))
                                      for i, m in enumerate(stage_c.AIC_CANDIDATES)]))
    out = pd.concat(rows, ignore_index=True)
    out.to_pickle(path)
    out["process"] = [process_of(k, sh, a) for k, sh, a in zip(out.knob, out["shape"], out.amount)]
    return out


def fig_wrong_model(rec, png):
    w = wrong_model_table(rec)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5",
                                                      "#1c5cab", "#0d366b"])
    designs = ["30 yr", "85 yr"]
    short = {"const": "Const.", "trend": "Loc.\ntrend", "trend2": "Loc. + wid.\ntrend",
             "tvp_median": "Bayes\n(median)", "cov2": "Mean-state\ncovariate", "aic": "AIC\nchoice",
             "aic_avg": "AIC\naverage", "cov2_true": "Oracle\ncovariate"}
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.4), sharex=True)
    for ax, d in zip(axes, designs):
        M = np.full((len(PROCESSES), len(WRONG_EST)), np.nan)
        for i, (pk, _, _) in enumerate(PROCESSES):
            for j, n in enumerate(WRONG_EST):
                v = w[(w.design == d) & (w.process == pk) & (w.estimator == n)].off2x
                if len(v):
                    M[i, j] = v.iloc[0]
        ax.imshow(M, cmap=cmap, vmin=0.3, vmax=0.95, aspect="auto")
        for i, (pk, _, _) in enumerate(PROCESSES):
            for j, n in enumerate(WRONG_EST):
                if not np.isfinite(M[i, j]):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=7.5, color=MUTED)
                    continue
                right = pk in RIGHT_FORM.get(n, set())
                ax.text(j, i, f"{100 * M[i, j]:.0f}" + ("" if right or n.startswith("aic") else "×"),
                        ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.7 else INK, fontweight="bold" if right else "normal")
        ax.set_yticks(range(len(PROCESSES)), [lab for _, lab, _ in PROCESSES], fontsize=7.5)
        ax.set_title(d.replace("yr", "years"), fontsize=8.5, loc="left")
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axvline(4.5, color="white", lw=4)
        ax.axvline(6.5, color="white", lw=4)
    axes[-1].set_xticks(range(len(WRONG_EST)), [short[n] for n in WRONG_EST], fontsize=7.5)
    fig.suptitle("Share of records in which the estimated 99% level in the last year misses the true "
                 "risk by more than a factor of 2 (either way), %", fontsize=8.5)
    fig.text(0.5, 0.005, "Bold: the model's form contains the truth (approximately); ×: it does not.\n"
             "AIC: choice among, or average of, the first, second, third and fifth models. "
             "Oracle: covariate model given the true mean.", ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig_wrong_model", png)


def fig_paths_world(png):
    """Same mean path and end states; width follows the mean (top of each pair) or time."""
    rows = [("normal", pa.Change("warm_coupled", "exp", 0.5)), ("normal", pa.Change("warm_time", "exp", 0.5)),
            ("t", pa.Change("warm_coupled", "exp", 0.5)), ("t", pa.Change("warm_time", "exp", 0.5))]
    fig_bands("fig_paths_world", rows, BY_LENGTH, ["trend2", "cov2"], png)
    rows = [("normal", pa.Change("width", "linear", 0.2)), ("gamma", pa.Change("width", "linear", 0.2)),
            ("normal", pa.Change("mean", "step50", 0.5)), ("gamma", pa.Change("tail", "linear", 1.0))]
    fig_bands("fig_paths_wrong", rows, BY_LENGTH, ["trend2", "cov2"], png)


def table_wrong_model(rec):
    """Main table: share missed by > 2x either way (30 and 85 years) and the AIC choices; and a
    detail table with the two directions separately at 85 years."""
    w = wrong_model_table(rec)
    ch = aic_choices(rec)
    est = ["const", "trend", "trend2", "tvp_median", "cov2", "aic", "aic_avg"]
    short = {"const": "Const.", "trend": "Loc.", "trend2": "Loc.+wid.", "tvp_median": "Bayes",
             "cov2": "Cov.", "aic": "AIC", "aic_avg": "AIC avg."}

    def cell(d, pk, n, metric):
        v = w[(w.design == d) & (w.process == pk) & (w.estimator == n)][metric]
        c = f"{100 * v.iloc[0]:.0f}" if len(v) else "--"
        return r"\textbf{" + c + "}" if len(v) and pk in RIGHT_FORM.get(n, set()) else c

    lines = [r"\begin{tabular}{l" + "r" * len(est) + "rrrr}", r"\toprule",
             r" & \multicolumn{%d}{c}{Missed by $>2\times$ either way, \%% of records} & "
             r"\multicolumn{4}{c}{AIC picks, \%% of records} \\" % len(est),
             r"\cmidrule(lr){2-%d} \cmidrule(lr){%d-%d}" % (1 + len(est), 2 + len(est), 5 + len(est)),
             "True process & " + " & ".join(short[n] for n in est)
             + r" & Const. & Loc. & Loc.+wid. & Cov. \\", r"\midrule"]
    for d in ["30 yr", "85 yr"]:
        lines.append(rf"\multicolumn{{{5 + len(est)}}}{{l}}{{\emph{{{d}}}}} \\")
        for pk, lab, _ in PROCESSES:
            cells = [cell(d, pk, n, "off2x") for n in est]
            for m in stage_c.AIC_CANDIDATES:
                v = ch[(ch.design == d) & (ch.process == pk) & (ch.model == m)].share
                cells.append(f"{100 * v.mean():.0f}" if len(v) else "--")
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "wrong_model.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{l" + "r" * len(est) + "}", r"\toprule",
             "True process & " + " & ".join(short[n] for n in est) + r" \\", r"\midrule"]
    for metric, title in [("under2x", "Understated $>2\\times$"), ("over2x", "Overstated $>2\\times$")]:
        lines.append(rf"\multicolumn{{{1 + len(est)}}}{{l}}{{\emph{{{title}, 85 years}}}} \\")
        for pk, lab, _ in PROCESSES:
            lines.append(f"{lab} & " + " & ".join(cell("85 yr", pk, n, metric) for n in est) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "wrong_model_detail.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------
TABLE_LABEL = {"trend2": "Loc. + width trend", "trend3": "All-param. trend", "cov2": "Mean-state cov.",
               "cov3": "Cov., all param.", "cov2_true": "Cov., true mean", "cov3_true": "Cov., all, true mean",
               "aic_avg": "AIC-weighted avg.", "tvp_median": "Bayes (median)", "tvp_mean": "Bayes (mean)"}


def table_headline(head):
    designs = ["30 yr", "45 yr", "85 yr", "85 yr, 3/yr"]
    lines = [r"\begin{tabular}{l" + "rrrr" * len(designs) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{d}}}" for d in designs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(designs))),
             "Estimator & " + " & ".join(["med.", "miss", r"$>$2\%", r"$<$0.5\%"] * len(designs)) + r" \\",
             r"\midrule"]
    for kind, title in [("standard", "Standard cases"), ("coupled", "Covariate-model cases")]:
        lines.append(rf"\multicolumn{{{1 + 4 * len(designs)}}}{{l}}{{\emph{{{title}}}}} \\")
        for n in TABLE_EST:
            cells = []
            for d in designs:
                v = head[(head.design == d) & (head.kind == kind) & (head.estimator == n)]
                if len(v):
                    r = v.iloc[0]
                    cells += [f"{100 * r.exc_median:.2f}", rf"\textbf{{{100 * (r.under2x + r.over2x):.0f}}}",
                              f"{100 * r.under2x:.0f}", f"{100 * r.over2x:.0f}"]
                else:
                    cells += ["", "", "", ""]
            if any(cells):
                lines.append(f"{TABLE_LABEL.get(n, EST[n][0])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "headline.tex").write_text("\n".join(lines) + "\n")


def table_power():
    d = pd.read_csv(RES / "stage_b_summary.csv")
    d = d[(d.k == 1) & (d.r == 0) & (d.phi == 0.5) & (d["shape"] == "linear")]
    rows = [("normal", "mean", 0.5), ("normal", "width", 0.2), ("gamma", "mean", 0.5),
            ("gamma", "width", 0.2), ("gamma", "tail", 1.0), ("t", "mean", 0.5), ("t", "tail", 1.0),
            ("beta", "mean", 0.25), ("beta", "width", 0.1), ("beta", "tail", 0.2), ("beta", "alpha", 0.5)]
    Ts = [30, 45, 85, 150]
    tests = [("nyblom_nominal", "Nyb."), ("trend_lr_nominal", "LR"), ("mk_nominal", "MK"), ("split_nominal", "Hlv.")]
    n = len(tests)
    lines = [r"\begin{tabular}{llr" + "r" * n * len(Ts) + "}", r"\toprule",
             r" & & & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{$T={T}$}}" for T in Ts) + r" \\",
             " ".join(rf"\cmidrule(lr){{{4 + n * i}-{3 + n * (i + 1)}}}" for i in range(len(Ts))),
             r"Parent & Change & $\Delta X_{0.99}$ & " + " & ".join([lab for _, lab in tests] * len(Ts)) + r" \\",
             r"\midrule"]
    for fam, knob, amt in rows:
        s = d[(d.family == fam) & (d.knob == knob) & np.isclose(d.amount, amt)]
        cells = []
        for T in Ts:
            r = s[s["T"] == T].iloc[0]
            vals = [int(np.floor(100 * r[c] + 0.5)) for c, _ in tests]     # round half up, as in the text
            cells += [rf"\textbf{{{v}}}" if v == max(vals) else f"{v}" for v in vals]
        dx = s.d_rl100_indep.iloc[0]
        lines.append(f"{FAMILY[fam]} & {knob} {amt:g} & {dx:+.2f} & " + " & ".join(cells) + r" \\")
    s0 = d[(d.knob == "mean") & (d.amount == 0)]
    cells = []
    for T in Ts:
        st = s0[s0["T"] == T]
        cells += [f"{100 * st[c].mean():.1f}" for c, _ in tests]
    lines += [r"\midrule", r"All (mean) & no change & 0 & " + " & ".join(cells) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "power.tex").write_text("\n".join(lines) + "\n")


def table_cases(rec):
    """Stage C cases with the true change in X_0.99 over an 85-year record."""
    s = rec[(rec.design == "85 yr") & (rec.estimator == "trend")]
    lines = [r"\begin{tabular}{lllr}", r"\toprule",
             r"Set & Parent & Change over the record & $\Delta X_{0.99}$ \\", r"\midrule"]
    for kind, title in [("standard", "Standard"), ("coupled", "Covariate model"), ("time", "Time-driven")]:
        sk = s[s.kind == kind]
        if not len(sk):
            continue
        for fam in FAMILIES:
            for case in dict.fromkeys(sk[sk.family == fam].case):
                tc = sk[(sk.family == fam) & (sk.case == case)].true_change.iloc[0]
                lines.append(f"{title} & {FAMILY[fam]} & {case.replace('&', chr(92) + '&')} & {tc:+.2f} \\\\")
                title = ""
        if kind != "time":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(lines) + "\n"
    txt = txt.replace("×", r"$\times$").replace("α", r"$\alpha$")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "cases.tex").write_text(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", default=None, help="also write PNG previews to this directory")
    ap.add_argument("--reuse", action="store_true", help="reuse the cached per-record table")
    ap.add_argument("--only", nargs="*", default=None, help="figure function names to run")
    args = ap.parse_args()
    if args.png:
        Path(args.png).mkdir(parents=True, exist_ok=True)
    os.nice(10)
    cache = {}

    def rec():
        if "rec" not in cache:
            path = WORK / "records.pkl"
            cache["rec"] = pd.read_pickle(path) if args.reuse and path.exists() else records_table()
        return cache["rec"]

    def head():
        if "head" not in cache:
            cache["head"] = headline(rec())
        return cache["head"]

    jobs = {"examples": lambda: fig_examples(args.png), "records": lambda: fig_records_all(args.png),
            "spread": lambda: fig_spread(args.png),
            "record_length": lambda: fig_record_length(args.png), "power": lambda: fig_power(args.png),
            "paths_parent": lambda: fig_paths_by_parent(args.png),
            "paths_covariate": lambda: fig_paths_covariate(args.png),
            "paths_k": lambda: fig_paths_values_per_year(args.png),
            "paths_prior": lambda: fig_paths_prior(args.png),
            "size": lambda: fig_size(args.png), "exceedance": lambda: fig_exceedance(rec(), args.png),
            "heatmap": lambda: fig_heatmap(head(), args.png), "change": lambda: fig_change(rec(), args.png),
            "bayes": lambda: fig_bayes(rec(), args.png), "prior_phi": lambda: fig_prior_phi(head(), args.png),
            "wrong_model": lambda: (fig_wrong_model(rec(), args.png), table_wrong_model(rec())),
            "paths_world": lambda: fig_paths_world(args.png),
            "tables": lambda: (table_headline(head()), table_power(), table_cases(rec()))}
    for name, f in jobs.items():
        if args.only is None or name in args.only:
            f()
            print("done", name, flush=True)


if __name__ == "__main__":
    main()
