"""Copy the authors' critical-value coefficients into data/calibration (source = "mw2024").

Reads Matlab/CVS_Files from the replication archive (doi:10.7910/DVN/RZG8QK, CC0 1.0):
  cvx_test{0..4}_{k}_{T}.csv     (a0, a1, a2) for the k-largest tests, level 0.05
  cvx_test{0..4}_{tau|tvt}_522.csv  same for the weather-damage exceedance data, level 0.04
                                    (Bonferroni: 1% for the first-step confidence set)
  cvs_test1_{k}_{T}.tab          xi0 grid and critical values of the xi = xi0 LR test

Usage: python scripts/import_mw2024_cvs.py [path/to/CVS_Files]
"""

import re
import sys
from pathlib import Path

import numpy as np

from tvextremes import critical_values as cv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "external" / "muller-watson-2024" / "files" / "Matlab" / "CVS_Files"
GRID_N = {0: 10, 1: 18, 2: 10, 3: 10, 4: 10}


def main(src: Path):
    adj_rows = [r for r in cv._read(cv.ADJ_FILE) if r["source"] != "mw2024"]
    xi_rows = [r for r in cv._read(cv.XI_FILE) if r["source"] != "mw2024"]
    for f in sorted(src.glob("cvx_test*.csv")):
        m = re.fullmatch(r"cvx_test(\d)_(\w+?)_(\d+)\.csv", f.name)
        test, kk, T = int(m.group(1)), m.group(2), int(m.group(3))
        a = np.loadtxt(f, delimiter=",")
        if kk.isdigit():
            scheme, k, alpha = "klargest", int(kk), 0.05
            g = cv.xi_grid(cv.TESTS[test])
            gmin, gmax, ng = g.min(), g.max(), GRID_N[test]
        else:
            scheme = "exceedance:weather" if kk == "tau" else "exceedance:weather_norm"
            k, alpha, gmin, gmax, ng = "", 0.04, "", "", 50
        adj_rows.append(dict(source="mw2024", scheme=scheme, test=cv.TESTS[test], k=k, T=T,
                             alpha=alpha, a0=f"{a[0]:.10g}", a1=f"{a[1]:.10g}", a2=f"{a[2]:.10g}",
                             N=10000, seed="", grid_min=gmin, grid_max=gmax, n_grid=ng,
                             max_rejection="", min_rejection="", failures="",
                             created="2024-09-03"))
    for f in sorted(src.glob("cvs_test1_*.tab")):
        m = re.fullmatch(r"cvs_test1_(\d+)_(\d+)\.tab", f.name)
        k, T = int(m.group(1)), int(m.group(2))
        grid, cvs = np.loadtxt(f)
        for xi0, c in zip(grid, cvs):
            xi_rows.append(dict(source="mw2024", scheme="klargest", k=k, T=T, alpha=0.05,
                                xi0=f"{xi0:.6g}", cv=f"{c:.6g}", N=10000, seed="",
                                created="2024-09-03"))
    adj_rows.sort(key=lambda r: (r["source"], r["scheme"], int(r["T"]), int(r["k"] or 0),
                                 cv.TEST_NUMBER.get(r["test"], 9)))
    xi_rows.sort(key=lambda r: (r["source"], r["scheme"], int(r["T"]), int(r["k"]), float(r["xi0"])))
    cv._write(cv.ADJ_FILE, cv.ADJ_FIELDS, adj_rows)
    cv._write(cv.XI_FILE, cv.XI_FIELDS, xi_rows)
    print(f"wrote {cv.ADJ_FILE} and {cv.XI_FILE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
