"""Calibrate small-sample critical values for the k-largest tests and store them in data/calibration.

Example (the paper's returns application, k = 1, T = 194):

    python scripts/calibrate_cv.py --k 1 --T 194 --N 10000 --seed 1

Add --validate-seed S to draw an independent set of null panels and report the rejection rates of
the new coefficients (and of the authors' coefficients, where they exist) at the grid points and
at the midpoints between grid points. Raw simulations are saved to data/output/cv_sims/ (not in git).
"""

import os

# One BLAS/OpenMP thread per process: the workers already use the cores, and per-worker thread
# pools oversubscribe the machine. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
from pathlib import Path

import numpy as np

from tvextremes import critical_values as cv

ROOT = Path(__file__).resolve().parents[1]


def midpoints(grid):
    return (grid[:-1] + grid[1:]) / 2 if len(grid) > 1 else grid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--tests", nargs="+", default=list(cv.TESTS), choices=cv.TESTS)
    p.add_argument("--N", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--validate-seed", type=int, default=None)
    p.add_argument("--validate-N", type=int, default=10_000, help="replications for the validation draws")
    p.add_argument("--no-save", action="store_true", help="do not write to data/calibration")
    p.add_argument("--nice", type=int, default=19,
                   help="raise the niceness of this process and its workers (default 19, lowest priority)")
    args = p.parse_args()
    if hasattr(os, "nice") and args.nice > 0:
        os.nice(args.nice)        # inherited by the worker processes

    sim_dir = ROOT / "data" / "output" / "cv_sims"
    sim_dir.mkdir(parents=True, exist_ok=True)
    for test in args.tests:
        t0 = time.time()
        sim = cv.simulate_null(test, args.k, args.T, N=args.N, seed=args.seed, workers=args.workers)
        np.savez_compressed(sim_dir / f"{test}_k{args.k}_T{args.T}_N{args.N}_seed{args.seed}.npz",
                            grid=sim.grid, stats=sim.stats, xi_hat=sim.xi_hat)
        print(f"\n[{test}] k={args.k} T={args.T} N={args.N}: simulated in {time.time() - t0:.0f} s, "
              f"{sim.failures} failed fits")
        if test == "xi":
            cvs = cv.xi_test_critical_values(sim, args.alpha)
            print("  xi0:", np.round(sim.grid, 3))
            print("  cv: ", np.round(cvs, 3))
            if not args.no_save:
                cv.save_xi_cvs(sim, cvs, args.alpha)
            continue
        a, rates = cv.fit_adjustment(sim, args.alpha)
        print(f"  a = ({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}); rejection on grid "
              f"min {rates.min():.4f} max {rates.max():.4f}")
        if not args.no_save:
            cv.save_adjustment(sim, a, rates, args.alpha)

        if args.validate_seed is not None:
            grids = {"grid": sim.grid, "midpoints": midpoints(sim.grid)}
            coefs = {"ours": a}
            try:
                coefs["mw2024"] = cv.get_adjustment(test, args.k, args.T, args.alpha, source="mw2024")
            except KeyError:
                pass
            for gname, g in grids.items():
                if test == "zipf" and gname == "midpoints":
                    continue
                vsim = cv.simulate_null(test, args.k, args.T, N=args.validate_N, seed=args.validate_seed,
                                        workers=args.workers, grid=g)
                for cname, c in coefs.items():
                    r = cv.rejection_rates(vsim, c)
                    print(f"  validation ({gname}, seed {args.validate_seed}) {cname:7s}: "
                          f"max {r.max():.4f} min {r.min():.4f}  " + " ".join(f"{x:.3f}" for x in r))


if __name__ == "__main__":
    main()
