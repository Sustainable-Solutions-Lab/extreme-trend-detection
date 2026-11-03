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
    "split": ("Two halves", RED, "-"),
    "cov2_true": ("Covariate (true mean)", YELLOW, "--"),
    "cov3_true": ("Covariate, all par. (true mean)", VIOLET, "--"),
    "tvp_mean": ("Bayesian (mean)", ORANGE, "--"),
}
MAIN = ["const", "trend", "trend2", "cov2", "tvp_median"]
TABLE_EST = ["const", "recent30", "split", "trend", "trend2", "trend3", "cov2", "cov3",
             "tvp_median", "tvp_mean", "cov2_true", "cov3_true"]
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
        else:
            out += [(f, ch) for ch in run_c.coupled_scenarios(f) if ch.amount != 0]
    return out


def records_table():
    """One row per (design, case, estimator, record): last-year exceedance of the estimated
    X_0.99, its level error and the error of the estimated change. Cached in data/output/report."""
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for dname, dkw in DESIGNS.items():
        for kind in ("standard", "coupled"):
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


def fig_record_length(png):
    rows = [("normal", pa.Change("mean", "linear", 0.5)),
            ("normal", pa.Change("warm_coupled", "exp", 0.5))]
    Ts = [30, 45, 85]
    fig, axes = plt.subplots(len(rows), len(Ts), figsize=(6.5, 3.9), sharey="row")
    for i, (family, ch) in enumerate(rows):
        for j, T in enumerate(Ts):
            sc = Scenario(family, ch, T=T)
            for n in ("trend", "tvp_median"):
                _spread(axes[i, j], sc, n, EST[n][1])
            if i == 0:
                axes[i, j].set_title(f"{T} years", fontsize=8)
        axes[i, 0].set_ylabel(ROW_LABEL[(family, ch.knob)], fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    h = [Line2D([], [], color=INK, lw=2, label="Truth"),
         Line2D([], [], color=BLUE, label="Location trend: median, 25–75%, 5–95%"),
         Line2D([], [], color=ORANGE, label="Bayesian: median, 25–75%, 5–95%")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    _save(fig, "fig03_record_length", png)


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
    names = ["const", "recent30", "trend", "trend2", "trend3", "cov2", "tvp_median"]
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


def fig_heatmap(head, png):
    names = ["const", "recent30", "split", "trend", "trend2", "trend3", "cov2", "cov3",
             "tvp_median", "cov2_true"]
    cols = [("30 yr", "standard"), ("45 yr", "standard"), ("85 yr", "standard"),
            ("45 yr, 3/yr", "standard"), ("85 yr, 3/yr", "standard"), ("85 yr, indep. days", "standard"),
            ("30 yr", "coupled"), ("45 yr", "coupled"), ("85 yr", "coupled")]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5",
                                                      "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), sharey=True)
    for ax, (metric, title) in zip(axes, [("under2x", "Understated > 2× (true exceedance > 2%)"),
                                         ("over2x", "Overstated > 2× (true exceedance < 0.5%)")]):
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
        ax.set_xticks(range(len(cols)), [d + ("*" if k == "coupled" else "") for d, k in cols],
                      rotation=60, ha="right")
        ax.set_title(title, fontsize=7.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axvline(5.5, color="white", lw=3)
    axes[0].set_yticks(range(len(names)), [EST[n][0] for n in names])
    fig.text(0.5, -0.02, "Share of simulated records, %, pooled over cases. * covariate-model cases.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    _save(fig, "fig07_heatmap", png)


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
    names = ["const", "trend", "trend2", "cov2", "tvp_median"]
    for y, n in enumerate(names):
        va = head[(head.design == "85 yr") & (head.kind == "standard") & (head.estimator == n)].under2x
        vb = head[(head.design == "85 yr, indep. days") & (head.kind == "standard") & (head.estimator == n)].under2x
        if len(va) and len(vb):
            ax.plot([va.iloc[0], vb.iloc[0]], [y, y], color=EST[n][1], lw=1.2)
            ax.plot(va.iloc[0], y, "o", ms=4.5, color=EST[n][1], mec="white", mew=0.5)
            ax.plot(vb.iloc[0], y, "s", ms=4.5, color="white", mec=EST[n][1], mew=1.3)
    ax.set_yticks(range(len(names)), [EST[n][0] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Share of records with risk understated > 2×")
    ax.set_title("85 yr: correlated days φ = 0.5 (●) vs independent (□)", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "figA1_prior_phi", png)


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------
def table_headline(head):
    designs = ["30 yr", "45 yr", "85 yr", "85 yr, 3/yr"]
    lines = [r"\begin{tabular}{l" + "rrr" * len(designs) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{d}}}" for d in designs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(designs))),
             "Estimator & " + " & ".join(["med.", r"$>$2\%", r"$<$0.5\%"] * len(designs)) + r" \\",
             r"\midrule"]
    for kind, title in [("standard", "Standard cases"), ("coupled", "Covariate-model cases")]:
        lines.append(rf"\multicolumn{{{1 + 3 * len(designs)}}}{{l}}{{\emph{{{title}}}}} \\")
        for n in TABLE_EST:
            cells = []
            for d in designs:
                v = head[(head.design == d) & (head.kind == kind) & (head.estimator == n)]
                if len(v):
                    r = v.iloc[0]
                    cells += [f"{100 * r.exc_median:.2f}", f"{100 * r.under2x:.0f}", f"{100 * r.over2x:.0f}"]
                else:
                    cells += ["", "", ""]
            if any(cells):
                lines.append(f"{EST[n][0]} & " + " & ".join(cells) + r" \\")
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
    lines = [r"\begin{tabular}{llr" + "rr" * len(Ts) + "}", r"\toprule",
             r" & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$T={T}$}}" for T in Ts) + r" \\",
             " ".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(Ts))),
             r"Parent & Change & $\Delta X_{0.99}$ & " + " & ".join(["Nyb.", "LR"] * len(Ts)) + r" \\",
             r"\midrule"]
    for fam, knob, amt in rows:
        s = d[(d.family == fam) & (d.knob == knob) & np.isclose(d.amount, amt)]
        cells = []
        for T in Ts:
            r = s[s["T"] == T].iloc[0]
            cells += [f"{r.nyblom_nominal:.2f}", f"{r.trend_lr_nominal:.2f}"]
        dx = s.d_rl100_indep.iloc[0]
        lines.append(f"{FAMILY[fam]} & {knob} {amt:g} & {dx:+.2f} & " + " & ".join(cells) + r" \\")
    s0 = d[(d.knob == "mean") & (d.amount == 0)]
    cells = []
    for T in Ts:
        st = s0[s0["T"] == T]
        cells += [f"{st.nyblom_nominal.mean():.3f}", f"{st.trend_lr_nominal.mean():.3f}"]
    lines += [r"\midrule", r"All (mean) & no change & 0 & " + " & ".join(cells) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "power.tex").write_text("\n".join(lines) + "\n")


def table_cases(rec):
    """Stage C cases with the true change in X_0.99 over an 85-year record."""
    s = rec[(rec.design == "85 yr") & (rec.estimator == "trend")]
    lines = [r"\begin{tabular}{lllr}", r"\toprule",
             r"Set & Parent & Change over the record & $\Delta X_{0.99}$ \\", r"\midrule"]
    for kind, title in [("standard", "Standard"), ("coupled", "Covariate model")]:
        sk = s[s.kind == kind]
        for fam in FAMILIES:
            for case in dict.fromkeys(sk[sk.family == fam].case):
                tc = sk[(sk.family == fam) & (sk.case == case)].true_change.iloc[0]
                lines.append(f"{title} & {FAMILY[fam]} & {case.replace('&', chr(92) + '&')} & {tc:+.2f} \\\\")
                title = ""
        if kind == "standard":
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

    jobs = {"examples": lambda: fig_examples(args.png), "spread": lambda: fig_spread(args.png),
            "record_length": lambda: fig_record_length(args.png), "power": lambda: fig_power(args.png),
            "size": lambda: fig_size(args.png), "exceedance": lambda: fig_exceedance(rec(), args.png),
            "heatmap": lambda: fig_heatmap(head(), args.png), "change": lambda: fig_change(rec(), args.png),
            "bayes": lambda: fig_bayes(rec(), args.png), "prior_phi": lambda: fig_prior_phi(head(), args.png),
            "tables": lambda: (table_headline(head()), table_power(), table_cases(rec()))}
    for name, f in jobs.items():
        if args.only is None or name in args.only:
            f()
            print("done", name, flush=True)


if __name__ == "__main__":
    main()
