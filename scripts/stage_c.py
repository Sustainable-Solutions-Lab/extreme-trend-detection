"""Run Stage C: recovery of known return-level paths by the Müller–Watson Bayesian model.

Each scenario (family, change, k, T, phi, r) draws N panels from Stage B's parent-distribution
generator and fits, to the same panels, the authors' four-random-walk GEV model (Stan) and the
three MLE estimators (constant, linear location trend, two-segment). Every estimated X_p(t) path
is scored against the truth implied by the generating function (tvextremes.stage_c.score_rows).

Outputs:
* data/output/stage_c/*.npz: estimated paths per panel (not in git)
* data/results/stage_c_summary.csv: one row per scenario, estimator and p (committed)

Example: python scripts/stage_c.py --N 100 --skip-existing
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import time
from pathlib import Path

from tvextremes import parents as pa
from tvextremes import stage_c

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output" / "stage_c"
CACHE = ROOT / "data" / "output" / "stage_b"           # cached annual-max distributions
SUMMARY = ROOT / "data" / "results" / "stage_c_summary.csv"
KEY = ("family", "knob", "shape", "amount", "k", "T", "phi", "r", "prior")

# One change of each kind per family, sized to be detectable about half to most of the time at
# T = 85, k = 1 (Stage B).
MEAN = {"normal": 0.5, "gamma": 0.5, "t": 0.5, "beta": 0.25}
WIDTH = {"normal": 0.2, "gamma": 0.2, "t": 0.2, "beta": 0.1}
TAIL = {"gamma": 1.0, "t": 1.0, "beta": 0.2}


def scenarios(family):
    out = [pa.NULL, pa.Change("mean", "linear", MEAN[family]),
           pa.Change("mean", "step50", MEAN[family]), pa.Change("width", "linear", WIDTH[family])]
    if family in TAIL:
        out.append(pa.Change("tail", "linear", TAIL[family]))
    if family == "beta":
        out.append(pa.Change("alpha", "linear", 0.5))
    return out


# Covariate-model test cases: parameters move with the mean state, along an accelerating path
# ("exp", slow then fast, like observed warming) and, for comparison, a linear one. Amounts:
# mean rise in daily sd (temperature-like) or log scaling of the gamma (precipitation-like; 0.2 is
# a 22% rise in the mean and every quantile).
COUPLED = {"normal": [("warm_coupled", "exp", 0.5), ("warm_coupled", "linear", 0.5)],
           "t": [("warm_coupled", "exp", 0.5)],
           "beta": [("warm_coupled", "exp", 0.25)],
           "gamma": [("precip_scaling", "exp", 0.2), ("precip_scaling", "linear", 0.2),
                     ("precip_shape", "exp", 0.2)]}


# Time-driven counterparts of the accelerating warm_coupled cases: the same mean path and the same
# first- and last-year distributions, but the width and tail change linearly in time instead of
# following the mean state. Together with COUPLED they test what happens when the change is
# modelled as a function of the wrong variable.
TIME_DRIVEN = {"normal": [("warm_time", "exp", 0.5)], "t": [("warm_time", "exp", 0.5)],
               "beta": [("warm_time", "exp", 0.25)]}


def time_driven_scenarios(family):
    return [pa.Change(*c) for c in TIME_DRIVEN.get(family, [])]


def coupled_scenarios(family):
    return [pa.NULL] + [pa.Change(*c) for c in COUPLED.get(family, [])]


def path_for(family, ch, k, T, phi, r, prior="mw"):
    tag = "" if prior == "mw" else f"_prior-{prior}"
    return OUT / f"{family}_{ch.knob}_{ch.shape}_{ch.amount:g}_k{k}_T{T}_phi{phi:.2f}_r{r}{tag}.npz"


def _key(row):
    row = {"prior": "mw", **{c: v for c, v in row.items() if v != ""}}   # old rows: no prior column
    return tuple(f"{float(row[c]):.4f}" if c in ("amount", "phi") else str(row[c]) for c in KEY)


def write_rows(rows):
    if not rows:
        return
    old = []
    if SUMMARY.exists():
        with open(SUMMARY, newline="") as f:
            old = list(csv.DictReader(f))
    new = {_key(r) for r in rows}
    allrows = [r for r in old if _key(r) not in new]
    allrows += [{c: (f"{v:.4f}" if isinstance(v, float) else v) for c, v in r.items()} for r in rows]
    fields = []
    for r in allrows:
        fields += [c for c in r if c not in fields]
    for r in allrows:
        r.setdefault("prior", "mw")
        if r["prior"] == "":
            r["prior"] = "mw"
    allrows.sort(key=lambda r: (int(r["T"]), int(r["k"]), int(r["r"]), float(r["phi"]), r["prior"],
                                r["family"],
                                r["knob"], r["shape"], float(r["amount"]), float(r["p"]),
                                r["estimator"]))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(allrows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", nargs="+", default=["normal", "gamma", "t", "beta"], choices=pa.FAMILIES)
    p.add_argument("--k", nargs="+", type=int, default=[1])
    p.add_argument("--T", nargs="+", type=int, default=[85])
    p.add_argument("--phi", type=float, default=0.5)
    p.add_argument("--r", type=int, default=0)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--nice", type=int, default=19)
    p.add_argument("--coupled", action="store_true",
                   help="run the covariate-model test cases instead of the standard changes")
    p.add_argument("--time-driven", action="store_true",
                   help="run the time-driven counterparts of the coupled cases (TIME_DRIVEN)")
    p.add_argument("--prior", default="mw", choices=sorted(stage_c.PRIORS),
                   help="prior scales of the random walks (default: the authors')")
    p.add_argument("--rescore", action="store_true",
                   help="only rescore saved scenarios (adds the further MLE estimators)")
    p.add_argument("--replay", action="store_true",
                   help="refit every scenario saved in data/output/stage_c with its own settings "
                        "and seeds, writing to data/output/stage_c_pred with the posterior "
                        "predictive output added (and checking the old summaries reproduce)")
    args = p.parse_args()
    if hasattr(os, "nice") and args.nice > 0:
        os.nice(args.nice)
    if args.replay:
        return replay(args)
    OUT.mkdir(parents=True, exist_ok=True)
    for T in args.T:
        for k in args.k:
            for family in args.families:
                which = (time_driven_scenarios if args.time_driven else
                         coupled_scenarios if args.coupled else scenarios)
                for ch in which(family):
                    path = path_for(family, ch, k, T, args.phi, args.r, args.prior)
                    if args.rescore and not path.exists():
                        continue
                    if (args.skip_existing or args.rescore) and path.exists():
                        res = stage_c.load(path)
                        print(f"{path.name}: reused", flush=True)
                    else:
                        t0 = time.time()
                        res = stage_c.simulate_scenario(family, ch, k=k, T=T, phi=args.phi,
                                                        r=args.r, N=args.N, seed=args.seed,
                                                        workers=args.workers,
                                                        mcmc=dict(prior=args.prior))
                        stage_c.save(res, path)
                        print(f"{family:6s} {ch.label:22s} k={k} T={T}: {args.N} fits in "
                              f"{time.time() - t0:.0f} s", flush=True)
                    extra = stage_c.extra_estimates(res, workers=args.workers or 4, cache_dir=OUT)
                    rows = stage_c.score_rows(res, cache_dir=CACHE, extra=extra)
                    write_rows(rows)
                    for r in rows:
                        if r["p"] == 0.99 and r["estimator"] in ("trend", "trend2", "recent30", "cov3", "tvp_median"):
                            print(f"   {r['estimator']:10s} X99 last year: true exceedance median "
                                  f"{100 * r.get('exc_med_last', float('nan')):.2f}%, >2%: "
                                  f"{100 * r.get('exc_gt2x_last', float('nan')):.0f}% of panels"
                                  + (f", 90% CI coverage {r['cover90_last']:.2f}"
                                     if "cover90_last" in r else ""), flush=True)


PRED_OUT = ROOT / "data" / "output" / "stage_c_pred"


def _replay_order(res):
    """Most-used designs first: k = 1 at 85, 45, 30 years, then 3 values per year, the tight
    prior and phi = 0."""
    design = (res.k, res.prior != "mw", res.phi != 0.5)
    return (design, -res.T)


def replay(args):
    import numpy as np
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    old = []
    for path in sorted(OUT.glob("*.npz")):
        if path.name.startswith("extra_"):
            continue
        old.append((path, stage_c.load(path)))
    old.sort(key=lambda pr: _replay_order(pr[1]))
    print(f"{len(old)} scenarios, {sum(r.stats.shape[0] for _, r in old)} fits", flush=True)
    for path, prev in old:
        out = PRED_OUT / path.name
        if out.exists():
            res = stage_c.load(out)
            print(f"{out.name}: reused", flush=True)
        else:
            t0 = time.time()
            res = stage_c.simulate_scenario(prev.family, prev.change, k=prev.k, T=prev.T,
                                            phi=prev.phi, r=prev.r, N=prev.stats.shape[0],
                                            seed=prev.seed, workers=args.workers,
                                            mcmc=dict(prior=prev.prior))
            stage_c.save(res, out)
            dq = np.nanmax(np.abs(res.tvp_q[:, 1] - prev.tvp_q[:, 1]))
            print(f"{out.name}: {res.stats.shape[0]} fits in {time.time() - t0:.0f} s; "
                  f"max change in posterior median vs the earlier run {dq:.2g}", flush=True)
        extra = stage_c.extra_estimates(res, workers=args.workers or 4, cache_dir=OUT)
        write_rows(stage_c.score_rows(res, cache_dir=CACHE, extra=extra))


if __name__ == "__main__":
    main()
