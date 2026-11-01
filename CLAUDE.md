# extreme-trend-detection: notes for Claude

See `README.md` for the project and its status.

## Data layout

- `data/input/`: downloaded or raw input data (e.g. ERA5). Not in git.
- `data/output/`: generated results, including raw Monte Carlo draws (`data/output/cv_sims/`).
  Not in git.
- `data/calibration/`: small, expensive-to-regenerate calibration tables (critical-value
  coefficients). **Committed to git.** Each row records source, N, seed and achieved size.
- `data/results/`: small summary tables of study results (e.g. `power_summary.csv`, one row per
  design and alternative). **Committed to git.** The raw statistics behind them stay in
  `data/output/`.
- `external/`: third-party code and data, e.g. the Müller–Watson replication archive. Not in
  git; the README says how to download it.

## Running heavy jobs

Monte Carlo calibration is CPU-heavy. `scripts/calibrate_cv.py` lowers its own priority
(`--nice 19` by default), and its workers inherit that. Run anything else long with
`nice -n 19`, and leave a core or two free.

## Conventions

- The Python package is `tvextremes`, installed editable in `.venv`. Run tests with
  `.venv/bin/python -m pytest`.
- GEV parameter order in scores and the authors' code is (xi, sigma, mu). Function arguments are
  `(mu, sigma, xi)`.
- The paper's LR statistic is a log-likelihood *difference*, with no factor of 2.
