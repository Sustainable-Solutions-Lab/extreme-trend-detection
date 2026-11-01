# extreme-trend-detection

Methods for detecting and estimating trends in the extremes of a time series of samples. The
starting point is a Python replication of

> Müller, U. K. and M. W. Watson (2024), "Time Varying Extremes," *Review of Economics and
> Statistics*, doi:10.1162/rest_a_01528.

We will then test these methods on pseudo-data with known properties, and apply them to ERA5
climate variables.

## The method in brief

For each period t = 1, …, T we keep either the **k largest observations** or **all exceedances of
a threshold τ_t**. By extreme value theory these follow a GEV distribution with location µ, scale
σ and shape (tail index) ξ, under asymptotics that fix k and T. Neither the number of extremes
nor the number of periods needs to be large.

| Paper section | Method | Status here |
|---|---|---|
| §2.1, eq. (4) | GEV_k likelihood for the k largest per period | done: `likelihood.py` |
| §2.3, eq. (7)–(8) | Exceedance likelihood (Poisson count × GP excesses) | done: `likelihood.py` |
| §2.2.2–2.2.3 | MLE and generalized LR tests: ξ = ξ₀, q₀.₉ = q₀, µ = σ/ξ (Pareto), Zipf | done: `mle.py` (LR statistic) |
| §2.2.2, App. B | ξ̂-dependent critical values `cv(ξ̂) = exp(−a₀−a₁ξ̂−a₂ξ̂²)` | done for k largest: `critical_values.py`, `scripts/calibrate_cv.py` |
| §2.4, App. B.2 | Bonferroni critical values for exceedances | to do |
| §3.1, eq. (12)–(13) | Nyblom test of parameter constancy, and the joint L* test | L_T done (`nyblom.py`, reproduces Table 2); L* to do |
| §3.2, eq. (14)–(15) | Bayesian TVP-GEV: random walks in (ξ, α, s, m), fitted by HMC in Stan | to do (parameter map and extended-support likelihood done) |
| §3.2.4 | Bayes factors via Meng & Wong's simple identity | to do |

In the time-varying model, α is the effective sample size (how many events occur). The parameters
(ξ, s, m) describe the tail of the underlying distribution. The Bayes factors compare these two
explanations: are extremes changing because there are *more events*, or because the distribution
itself has shifted?

## Repository layout

```
tvextremes/
  likelihood.py   per-period log-densities, analytic scores (d/d(ξ, σ, µ)), extended-support h~,
                  parameter maps (GEV ↔ Poisson-GP, GP tail + n → GEV, TVP (ξ, α, s, m) → GEV)
  mle.py          constant-parameter MLE (unrestricted and under each null; robust multi-start
                  and fast gradient-based fits), LR statistic
  nyblom.py       Nyblom L_T stability statistic and Hessian-based information
  critical_values.py  Monte Carlo calibration of the xi_hat-adjusted critical values; lookup
  pseudo.py       pseudo-data: GEV_k panels with known parameter paths (linear, accelerating, steps,
                  hump, random walk in mu, sigma or xi)
  trend_tests.py  baselines: GEV with a linear trend in location (LR test), Mann-Kendall
  power.py        power study: simulate a design, nominal and size-adjusted rejection rates
scripts/
  calibrate_cv.py      run a calibration for given (k, T) and store it in data/calibration
  import_mw2024_cvs.py copy the authors' coefficients into data/calibration
  power_study.py       run the Stage A power study and write data/results/power_summary.csv
tests/
  test_likelihood.py       self-contained checks (scipy cross-checks, normalization, scores, simulation)
  test_critical_values.py  calibration machinery
  test_power.py            pseudo-data generator, trend tests, time-reversal invariance
  test_replication.py      reproduces Tables 1 and 2 of the paper from the authors' data
docs/
  power_study_plan.md      design of the pseudo-data power and robustness study
data/
  calibration/    critical-value tables (committed)
  results/        small summary tables of study results (committed)
  input/, output/ raw inputs and generated results (not in git)
external/         authors' replication archive (not in git; see below)
```

### Data conventions

- **k largest:** `X` has shape (T, k). Each row is sorted in descending order.
- **Exceedances:** `Y` has shape (T, k_max), padded with `NaN`. Row t has k_t ≥ 0 values, each at
  least `tau[t]`. `tau` is a scalar or a length-T vector.
- **Parameters:** these are always (µ, σ, ξ) in GEV form. Any of them may be a length-T array
  (time-varying).
- **Scores:** returned as (T, 3) arrays, with columns ordered (ξ, σ, µ) as in the authors' code.
- **LR statistic:** the paper defines it as a *log-likelihood difference*, with no factor of 2.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

### Authors' replication files

The replication tests need the authors' archive, which is not in git (`external/` is ignored).
Download it from Harvard Dataverse. Watson's Princeton page blocks scripted downloads.

```bash
mkdir -p external/muller-watson-2024 && cd external/muller-watson-2024
curl -L -o replication.zip \
  'https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/RZG8QK'
unzip -q replication.zip -d files
```

The archive contains:

- MATLAB scripts (`Matlab/`, `Matlab_Functions/`)
- the Fortran critical-value code, which needs the Intel compiler and IMSL (`Fortran/`)
- R/cmdstanr and Stan models for every specification, including the exceedance TVP model that the
  paper does not print (`Stan/`)
- raw and processed data (`Data/`, `Matlab/mat/`)
- fitted critical-value coefficients (`Matlab/CVS_Files/`)
- the authors' printed results (`Matlab/out/`)

`test_replication.py` is skipped if the archive is absent.

### Replication status

`test_replication.py` re-fits all six applications in Table 1: weather damages (raw and
normalized), largest and smallest S&P 500 returns, and city and firm sizes with k = 30. It checks:

- ξ̂, σ̂ and µ̂ match the authors' output to about 4 decimals
- q̂₀.₉ matches to their printed 2 decimals
- the ξ̂-adjusted LR statistics for µ = σ/ξ and for Zipf's law match to their printed 2 decimals

The critical-value coefficients for these checks come from the authors' CSV files.

## Critical values

The paper's tests reject when adj(ξ̂)·stat > 1, with adj(ξ̂) = exp(a₀ + a₁ξ̂ + a₂ξ̂²) chosen by
simulation so that the test rejects at most 5% of the time for every true ξ on a grid (Appendix
B). `critical_values.py` follows the authors' Fortran: common random numbers across the ξ grid,
fits started at the truth, a kernel-smoothed linex objective that penalizes over-rejection, and a
final shift of a₀ so that the worst grid point rejects exactly 5%.

**Draws.** The authors used N = 10,000 panels per grid point. We use N = 40,000 for the fit
(seed 2026) and check the result on 10,000 independent draws (seed 7), both at the grid points and
at the midpoints between them. The Monte Carlo standard error of a 5% rejection rate is 0.22
percentage points at N = 10,000 and 0.11 at N = 40,000.

**Results** for the paper's two k-largest designs: worst-case rejection rate on the independent
validation draws, over grid points and midpoints (target 5%).

| Test | k = 1, T = 194: ours | authors' | k = 30, T = 4: ours | authors' |
|---|---|---|---|---|
| Nyblom L_T | 4.8% | 4.6% | 4.7% | 5.3% |
| q₀.₉ = q₀ | 4.9% | 5.0% | 5.0% | 5.1% |
| µ = σ/ξ (Pareto) | 5.2% | 5.0% | 5.1% | 5.4% |
| Zipf | 5.2% | 5.2% | 5.1% | 4.9% |

The ξ = ξ₀ test has no nuisance parameter, so its critical value is simply the 95% quantile of LR
at each ξ₀. Ours are 1.93–2.06 for k = 1, T = 194 and 1.95–2.19 for k = 30, T = 4. They differ
from the authors' by up to 0.07, and the differences have a consistent sign over stretches of the
grid: ours are lower for ξ₀ < 0.5 at k = 1, and higher for ξ₀ > 0.4 at k = 30. The Monte Carlo
standard error of one quantile is about 0.04 at the authors' N = 10,000. Common random numbers
make neighbouring grid points share their errors, so this is probably noise, but it has not been
confirmed. The calibration script does not yet validate the ξ test on independent draws, and
doing so would settle it.

Both sets of coefficients control size to within Monte Carlo error. Most grid points reject less
than 5%, because the three-coefficient curve cannot fit the rejection rate exactly at every ξ and
the worst point is set to 5%. For the Nyblom test at k = 30 the lowest grid point rejects only
3.5%. The remaining error is far below what we expect from departures from the GEV model, which
the pseudo-data study measures.

All rows, with source, N, seed and achieved size, are in `data/calibration/cv_adjustment.csv`
and `cv_xi.csv`. `get_adjustment` prefers our own calibration where one exists and falls back to
the authors' (`source = mw2024`). New designs, such as the 85-year ERA5 record, are calibrated
with `scripts/calibrate_cv.py --k K --T T`.

## Did Müller and Watson validate with pseudo-data?

Only in a narrow sense. Simulation appears in two places.

1. **Calibrating critical values (Appendix B).** For each value on a grid of 10 values of ξ, they
   draw N = 10,000 synthetic panels from the GEV_k model, using the exponential representation of
   eq. (2). They then fit the coefficients (a₀, a₁, a₂) so that the ξ̂-adjusted test rejects at
   most 5% of the time across ξ ∈ [ξ_min, 1.5], and probe ξ values between grid points for
   over-rejection. The exceedance tests (via a 99% confidence set of 50 parameter points) and the
   Nyblom test are calibrated the same way. Size control is therefore verified by simulation, but
   only for data generated *exactly* from the GEV model.
2. **One dependence check (footnote 11).** The same cities stay among the 30 largest for decades,
   so successive samples are dependent. To see how this affects the Nyblom test, they simulated
   Gabaix's (1999) Pareto city-growth process. The ξ̂-adjusted Nyblom test came out somewhat
   *under*-sized. The paper reports this in one sentence and no table. The code is in
   `Fortran/dependence.f90` and `Size_Dependence` in `Fortran/compute.f90`.

They do **not** report:

- **power** of the LR or Nyblom tests against trends of a given size, for given (k, T)
- **recovery of known parameter paths** by the Bayesian TVP-GEV model: no simulation with known
  time-varying (ξ, α, s, m) to check whether posterior paths and credible bands cover the truth
- **whether the Bayes factors pick the true source of change**, e.g. more events (α) versus a
  shifted distribution (ξ, s, m) (Table 3)
- **accuracy of the GEV approximation itself:** all simulations draw directly from the GEV. None
  start from an underlying distribution (normal, Student-t, gamma, …), take the k largest of n
  draws, and measure the resulting error. For that, and for the choice of k, they refer to Müller
  & Wang (2017).
- **robustness to serial dependence** within or across periods, beyond the city-size check

These gaps define the pseudo-data work here, and they are the questions that matter for climate
data:

- Can the tests detect trends of realistic size with typical ERA5 record lengths?
- Do posterior paths recover a known trend?
- Can the method tell "more events" apart from "a shifted distribution"?
- How robust are the results when daily values are autocorrelated (persistent heatwaves, multi-day
  storms) and the parent distribution is not GP-like?

## Roadmap

1. ~~Critical-value calibration in Python (Appendix B)~~ done for k largest; exceedance
   (Bonferroni) calibration still to do.
2. ~~Nyblom stability test~~ done; the joint L* test still to do.
3. Pseudo-data size and power studies (`docs/power_study_plan.md`). Stage A (exact GEV_k data with
   known paths, compared with linear-trend LR and Mann-Kendall) is built and running. Stage B
   (parent distributions such as normal, Student-t, raised cosine and tanh-normal, with serial
   dependence and seasonality) is next.
4. A TVP-GEV/exceedance model in Stan (via cmdstanpy), adapted from the authors' models, with
   importance reweighting and Bayes factors. Check that it recovers known paths.
5. ERA5: build k-largest or exceedance panels per grid cell or region, decluster to handle serial
   dependence, then run the tests and the TVP model.
