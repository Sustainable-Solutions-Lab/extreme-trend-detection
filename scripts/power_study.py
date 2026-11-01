"""Run Stage A of the power study (docs/power_study_plan.md) and write a summary table.

Example (annual maxima over 85 years, four baseline tail shapes):

    python scripts/power_study.py --designs 1x85 --xi0 -0.2 0 0.2 0.5 --N 2000

Raw statistics go to data/output/power/<design>_xi<xi0>.npz (not in git). The summary (one row per
design, xi0 and alternative) goes to data/results/power_summary.csv, which is committed; rows for
the same (k, T, xi0) are replaced on re-runs. Use --summarize-only to rebuild the summary from
saved .npz files (for example after a new critical-value calibration).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from tvextremes import power

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output" / "power"
SUMMARY = ROOT / "data" / "results" / "power_summary.csv"   # committed to git


def design_seed(seed, k, T, xi0):
    ss = np.random.SeedSequence([seed, k, T, int(round(xi0 * 1000)) + 10_000])
    return int(ss.generate_state(1)[0])


def npz_path(k, T, xi0):
    return OUT / f"k{k}_T{T}_xi{xi0:+.2f}.npz"


def write_summary(rows):
    old = []
    if SUMMARY.exists():
        with open(SUMMARY, newline="") as f:
            old = list(csv.DictReader(f))
    keys = {(str(r["k"]), str(r["T"]), f"{float(r['xi0']):.4f}") for r in rows}
    keep = [r for r in old if (r["k"], r["T"], f"{float(r['xi0']):.4f}") not in keys]
    allrows = keep + [{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows]
    allrows.sort(key=lambda r: (int(r["k"]), int(r["T"]), float(r["xi0"]), r["param"], r["shape"],
                                float(r["amount"])))
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(allrows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--designs", nargs="+", default=["1x85"], help="k x T, e.g. 1x85 3x85")
    p.add_argument("--xi0", nargs="+", type=float, default=[-0.2, 0.0, 0.2, 0.5])
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse saved .npz files instead of re-simulating them (e.g. after an interruption)")
    p.add_argument("--nice", type=int, default=19)
    args = p.parse_args()
    if hasattr(os, "nice") and args.nice > 0:
        os.nice(args.nice)
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    for design in args.designs:
        k, T = (int(v) for v in design.lower().split("x"))
        for xi0 in args.xi0:
            path = npz_path(k, T, xi0)
            if args.summarize_only or (args.skip_existing and path.exists()):
                if not path.exists():
                    print(f"missing {path.name}, skipped")
                    continue
                res = power.load(path)
            else:
                t0 = time.time()
                res = power.simulate_design(k, T, xi0, N=args.N, seed=design_seed(args.seed, k, T, xi0),
                                            workers=args.workers)
                power.save(res, path)
                print(f"k={k} T={T} xi0={xi0:+.2f}: {len(res.alternatives)} alternatives x N={args.N} "
                      f"in {time.time() - t0:.0f} s", flush=True)
            rows = power.rejection_table(res)
            write_summary(rows)
            null = rows[res.null_index()]
            print(f"  size (nominal): nyblom {null['nyblom_nominal']:.3f}  trend LR "
                  f"{null['trend_lr_nominal']:.3f}  MK {null['mk_nominal']:.3f}; "
                  f"failed fits {null['failures']}", flush=True)


if __name__ == "__main__":
    main()
