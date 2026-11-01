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
scripts/
  calibrate_cv.py      run a calibration for given (k, T) and store it in data/calibration
  import_mw2024_cvs.py copy the authors' coefficients into data/calibration
tests/
  test_likelihood.py       self-contained checks (scipy cross-checks, normalization, scores, simulation)
  test_critical_values.py  calibration machinery
  test_replication.py      reproduces Tables 1 and 2 of the paper from the authors' data
data/
  calibration/    critical-value tables (committed)
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

1. Critical-value calibration in Python (Appendix B), so that we can handle (k, T) combinations
   the authors did not tabulate. The template is `Matlab/Compute_a0a1a2_CVFunction.m`.
2. Nyblom stability test and the joint L* test.
3. A pseudo-data generator with known parameter paths, both directly from GEV_k and from parent
   distributions with optional serial dependence. Use it to run the size and power studies above.
4. A TVP-GEV/exceedance model in Stan (via cmdstanpy), adapted from the authors' models, with
   importance reweighting and Bayes factors. Check that it recovers known paths.
5. ERA5: build k-largest or exceedance panels per grid cell or region, decluster to handle serial
   dependence, then run the tests and the TVP model.
