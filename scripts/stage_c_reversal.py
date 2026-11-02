"""Time-reversal check of the Bayesian model: fitting the panel in reverse year order should give
the reversed posterior paths, up to MCMC noise (the prior is symmetric in time)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tvextremes import parents as pa, stage_c

CASES = [("normal", pa.Change("mean", "linear", 0.5)), ("t", pa.Change("tail", "linear", 1.0)),
         ("beta", pa.Change("width", "linear", 0.1)), ("gamma", pa.Change("mean", "step50", 0.5))]


def run(a):
    (family, ch), rev = a
    X = stage_c._panel(family, ch, 1, 85, 0.5, 0, 9, 0)
    f = stage_c.fit_tvp(X[::-1] if rev else X, seed=21, warmup=1000, samples=4000)
    q = f.q_median[:, ::-1] if rev else f.q_median
    return family, ch.label, rev, q, f.q_hi - f.q_lo


if __name__ == "__main__":
    os.nice(19)
    stage_c.model()
    jobs = [(c, rev) for c in CASES for rev in (False, True)]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(run, jobs))
    for i in range(0, len(out), 2):
        fam, lab, _, qf, wf = out[i]
        _, _, _, qr, _ = out[i + 1]
        d = np.abs(qf - qr).max(axis=1)
        print(f"{fam:6s} {lab:20s} max |forward - reversed| of median X_p path, p=0.5/0.9/0.99: "
              + " ".join(f"{v:.3f}" for v in d)
              + "   (90% CI width, last year: " + " ".join(f"{v:.2f}" for v in wf[:, -1]) + ")")
