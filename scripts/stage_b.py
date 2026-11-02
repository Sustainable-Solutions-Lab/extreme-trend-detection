"""Run Stage B of the power study: daily parent-distribution series (docs/power_study_plan.md).

Example (the main T = 85 grid with annual maxima):

    python scripts/stage_b.py --designs 1:0 --T 85 --phi 0 0.5 0.8

A design is k:r, the number of largest daily values kept per year and the declustering gap in
days. Designs with r > 0 are skipped at phi = 0, where declustering has nothing to remove.

Outputs:
* data/output/stage_b/*.npz: raw statistics per configuration (not in git)
* data/results/stage_b_summary.csv: size and power per configuration and change (committed)
* data/results/stage_b_return_levels.csv: accuracy of estimated X_p (p = 0.5, 0.9, 0.95, 0.99)
  against the truth implied by the generating function (committed)

Rows of the same configuration are replaced on re-runs. --skip-existing reuses saved
configurations, so an interrupted run can be resumed.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from tvextremes import parents as pa
from tvextremes import stage_b

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output" / "stage_b"
RESULTS = ROOT / "data" / "results"
SUMMARY = RESULTS / "stage_b_summary.csv"
RETURN_LEVELS = RESULTS / "stage_b_return_levels.csv"
KEY = ("family", "k", "T", "phi", "r")


def config_seed(seed, family, k, T, phi, r):
    ss = np.random.SeedSequence([seed, pa.FAMILIES.index(family), k, T, int(round(phi * 100)), r])
    return int(ss.generate_state(1)[0])


def npz_path(family, k, T, phi, r):
    return OUT / f"{family}_k{k}_T{T}_phi{phi:.2f}_r{r}.npz"


def _key(row):
    return tuple(f"{float(row[c]):.4f}" if c == "phi" else str(row[c]) for c in KEY)


def write_rows(path, rows, sort_extra=()):
    if not rows:
        return
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    new_keys = {_key(r) for r in rows}
    keep = [r for r in old if _key(r) not in new_keys]
    fmt = [{c: (f"{v:.4f}" if isinstance(v, float) else v) for c, v in r.items()} for r in rows]
    allrows = keep + fmt
    allrows.sort(key=lambda r: (int(r["T"]), int(r["k"]), int(r["r"]), float(r["phi"]), r["family"],
                                r["knob"], r["shape"], float(r["amount"]))
                 + tuple(float(r[c]) for c in sort_extra))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(allrows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--designs", nargs="+", default=["1:0"], help="k:r, e.g. 1:0 3:0 10:3")
    p.add_argument("--T", nargs="+", type=int, default=[85])
    p.add_argument("--phi", nargs="+", type=float, default=[0.0, 0.5, 0.8])
    p.add_argument("--families", nargs="+", default=list(pa.FAMILIES), choices=pa.FAMILIES)
    p.add_argument("--N-null", type=int, default=10_000)
    p.add_argument("--N-alt", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--nice", type=int, default=19)
    args = p.parse_args()
    if hasattr(os, "nice") and args.nice > 0:
        os.nice(args.nice)
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    for T in args.T:
        for design in args.designs:
            k, r = (int(v) for v in design.split(":"))
            for phi in args.phi:
                if r > 0 and phi == 0:
                    continue
                for family in args.families:
                    path = npz_path(family, k, T, phi, r)
                    if args.skip_existing and path.exists():
                        res = stage_b.load(path)
                        print(f"{path.name}: reused", flush=True)
                    else:
                        t0 = time.time()
                        res = stage_b.simulate_config(
                            family, k, T, phi, r, N_null=args.N_null, N_alt=args.N_alt,
                            seed=config_seed(args.seed, family, k, T, phi, r), workers=args.workers)
                        stage_b.save(res, path)
                        print(f"{family:6s} T={T} k={k} r={r} phi={phi:.1f}: {len(res.changes)} "
                              f"changes in {time.time() - t0:.0f} s", flush=True)
                    rows = stage_b.summary_rows(res)
                    write_rows(SUMMARY, rows)
                    write_rows(RETURN_LEVELS, stage_b.return_level_rows(res, cache_dir=OUT),
                               sort_extra=("p",))
                    null = rows[res.null_index()]
                    print(f"   size: nyblom {null['nyblom_nominal']:.3f}  trend LR "
                          f"{null['trend_lr_nominal']:.3f}  MK {null['mk_nominal']:.3f}  "
                          f"mean xi_hat {null['xi_hat_mean']:+.3f}  failed fits {null['failures']}",
                          flush=True)


if __name__ == "__main__":
    main()
