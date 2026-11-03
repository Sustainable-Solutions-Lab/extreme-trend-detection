# extreme-trend-detection

Methods for detecting and estimating trends in the extremes of a time series of samples. The
starting point is a Python replication of

> Müller, U. K. and M. W. Watson (2024), "Time Varying Extremes," *Review of Economics and
> Statistics*, doi:10.1162/rest_a_01528.

We will then test these methods on pseudo-data with known properties, and apply them to ERA5
climate variables.

**Report:** [`report/report.pdf`](report/report.pdf) (source `report/report.tex`, figures and
tables built by `scripts/make_figures.py`) summarizes the pseudo-data results: how well change is
detected, and how well the current 99% annual-maximum level and its change are estimated.

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
| §3.2, eq. (14)–(15) | Bayesian TVP-GEV: random walks in (ξ, α, s, m), fitted by HMC in Stan | done: `stan/gev_4rw.stan` (authors' model, leaner output), `stage_c.py` |
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
  parents.py      Stage B: daily AR(1) Gaussian series mapped to known parents (normal, gamma,
                  Student-t, beta; tanh-normal available but not run), k largest per year
                  with declustering, exact truth for annual-maximum quantiles
  stage_b.py      Stage B: simulate a configuration, size/power and return-level accuracy
  stage_c.py      Stage C: Bayesian TVP fits (cmdstanpy) and MLE estimators (constant, trends,
                  covariate models) of the X_p paths, scored against the exact truth
stan/
  gev_4rw.stan    the authors' four-random-walk GEV model (same density; leaner output)
scripts/
  calibrate_cv.py      run a calibration for given (k, T) and store it in data/calibration
  import_mw2024_cvs.py copy the authors' coefficients into data/calibration
  power_study.py       run the Stage A power study and write data/results/power_summary.csv
  stage_b.py           run Stage B and write data/results/stage_b_*.csv
  stage_c.py           run Stage C and write data/results/stage_c_summary.csv
  stage_c_reversal.py  time-reversal check of the Bayesian model
  make_figures.py      report figures and tables, data/results/stage_c_headline.csv
report/                report.tex, report.pdf, figures/, tables/ (LaTeX; build with latexmk -pdf)
tests/
  test_likelihood.py       self-contained checks (scipy cross-checks, normalization, scores, simulation)
  test_critical_values.py  calibration machinery
  test_power.py            pseudo-data generator, trend tests, time-reversal invariance
  test_stage_b.py          parent transforms, declustering, exact truth, return-level scoring
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

A copy of the paper with its supplement, if you have one, goes in the same folder as
`external/muller-watson-2024/Muller-Watson_2024_REStat_with_supplement.pdf` (also not in git).

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

## Power study, Stage A: first results

These results come from data drawn exactly from GEV_k, with known changes in µ, σ or ξ. The
setup:

- **Designs:** the 85-year ERA5 record with k = 1, 3 and 10 largest values per year, plus the
  paper's two designs (k = 1, T = 194 and k = 30, T = 4).
- **Baseline tail shapes:** ξ₀ = −0.2, 0, 0.2 and 0.5.
- **Replications:** 2,000 per cell, with common random numbers across alternatives.
- **Tests:** the paper's ξ̂-adjusted Nyblom test, the conventional GEV likelihood-ratio test for a
  linear trend in location (χ²₁ critical value), and Mann–Kendall on the annual maxima.

The full table, with nominal and size-adjusted power and the implied change in the 10- and
100-year return levels, is `data/results/power_summary.csv`. The design is in
`docs/power_study_plan.md`.

**Size with no change** (target 5%, over the four ξ₀):

| Test | Rejection rate |
|---|---|
| Nyblom | 3.9–5.7% |
| Linear-trend LR | 4.9–6.7% |
| Mann–Kendall | 3.9–5.7% |

The linear-trend LR is the most often over-sized, because the χ² approximation is loose at these
sample sizes. Mann–Kendall cannot reject at T = 4.

**Power at ξ₀ = 0** (nominal rejection rate: Nyblom / linear-trend LR / Mann–Kendall). Changes in
µ are in units of σ over the record. Changes in σ are in log points, and changes in ξ are
absolute.

| Change | k=1, T=85 | k=3, T=85 | k=10, T=85 | k=1, T=194 |
|---|---|---|---|---|
| µ linear, 0.5σ | .20/.30/.21 | .46/.64/.22 | .96/.99/.20 | .37/.54/.42 |
| µ linear, 1σ | .59/.75/.63 | .98/1.0/.61 | 1.0/1.0/.60 | .94/.98/.94 |
| µ step at 80% of record, 1σ | .46/.53/.54 | .95/.94/.52 | 1.0/1.0/.51 | .87/.89/.91 |
| µ rise then fall, 1σ | .19/.07/.06 | .56/.06/.05 | 1.0/.05/.06 | .43/.06/.06 |
| µ random walk, 1σ | .47/.51/.45 | .71/.69/.46 | .90/.82/.45 | .65/.65/.60 |
| σ linear, 0.4 | .18/.09/.07 | .71/.61/.06 | 1.0/1.0/.07 | .41/.09/.07 |
| σ linear, 0.8 | .63/.15/.10 | 1.0/.99/.09 | 1.0/1.0/.08 | .95/.26/.12 |
| ξ linear, 0.2 | .09/.07/.06 | .17/.13/.05 | 1.0/1.0/.06 | .13/.07/.06 |
| ξ linear, 0.4 | .17/.08/.07 | .59/.38/.06 | 1.0/1.0/.07 | .39/.10/.08 |

What the table shows:

- **For a linear trend in location**, the linear-trend LR is the most powerful test, as it should
  be. Its edge over Nyblom is modest, and part of it comes from being over-sized.
- **For any other kind of change**, Nyblom is better or much better. A rise-then-fall change and a
  change in scale or tail are almost invisible to both trend tests at k = 1.
- **Annual maxima alone (k = 1) barely detect a change in tail shape.** A rise in ξ of 0.4 raises
  the 100-year return level by about 4.5σ, yet even the best test detects it less than 20% of the
  time.
- **Using more values per year helps enormously in this idealized setting.** With k = 10, almost
  every alternative is detected. Mann–Kendall uses only the annual maxima and does not benefit.
  The extra information comes from the lower order statistics, and in real daily data those are
  the values least likely to follow the GEV_k model (heatwave clusters, seasonality). Stage B
  tests whether the gain survives.

Calibration note for k = 10, T = 85: on two independent validation sets (seeds 7 and 8), the
Nyblom coefficients reject 5.3–5.5% at ξ ≈ −0.3 to −0.1, which is slightly above 5%. A
recalibration with a new seed (2027) gave almost the same coefficients, so the excess, of about
0.3 percentage points, is a small real bias rather than noise in the fit.

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

## Stage C: how well is the 99% level predicted?

Each estimator is fitted to the same simulated records (Stage B generator; 100 records per case)
and scored on the *true* probability that its estimated 99% annual-maximum level for the last
year is exceeded (ideal 1%). An estimate *misses* when that probability is above 2% (risk
understated at least twofold) or below 0.5% (overstated at least twofold); the main score is the
share of records missed either way, reported with its two parts. Pooled results are in
`data/results/stage_c_headline.csv`, per-case results in `data/results/stage_c_summary.csv`, and
the full write-up in the report. Main findings, for 85-year records with one value per year unless
stated:

- **A noise floor dominates.** With no change and the correct constant model, 44% of records miss
  (27% understated, 18% overstated); 68% for 45 years and 71% for 30 years.
- **An AIC-weighted average of simple models is the most accurate estimator** over the standard
  cases: 51% missed (30% understated, 20% overstated), against 56% for a location trend (30% / 26%)
  or a location + width trend (33% / 23%), 65% for a constant fit or the last 30 years, and 70% for
  trends in all three parameters.
- **The authors' Bayesian model** (posterior median) is about as accurate as the simple trends:
  75%, 70% and 59% missed at 30, 45 and 85 years, against 73%, 69% and 56% for the location trend,
  erring toward understatement at 85 years (39% against 30%). Its posterior mean rarely
  understates but overstates in 51% of records. It shrinks estimated changes toward zero. Its 90%
  intervals cover the truth 94–100% of the time, but are 1.3–3 times wider than the actual errors.
  With three values per year, 10–12% of fits do not converge (R-hat > 1.5, mostly the bounded beta
  parent); a tighter prior on the random-walk scales removes this.
- **With 30 or 45 years the method hardly matters**: all the sensible estimators miss 73–76% and
  68–72% of records, close to the no-change floor.
- **When width and tail move with the mean state** (covariate-model cases), the time-trend methods
  miss 69–74% of records, mostly by understating the risk. A GEV with location and log scale
  linear in a smoothed annual mean of all days (`cov2`, constant ξ) misses 62%, as does its oracle
  with the true mean (63%). Letting ξ depend on the raw annual mean (`cov3`) is unstable.
- **Using the wrong model.** No model is right for every physical process, so we classified the
  cases by true process and scored every model on all of them (two-sided: the share of records
  in which the estimate misses the true risk by more than a factor of 2 either way; 44% with no
  change and the correct model). With the wrong driving variable the share of misses rises from 65% to 75%
  when the width follows an accelerating mean (a linear time trend cannot follow it), and from 55%
  to 69% when the width changes at constant mean (a mean-state model cannot see it). When the mean rises at an increasing rate but the width changes linearly in time
  (`warm_time` cases), the mean-state model still does as well as any (64% against 68% for the
  location + width trend), because following the mean path matters more than the driver of the
  width. AIC cannot reliably identify the driver, but an AIC-weighted average of the constant,
  time-trend and mean-state models (`aic_avg`) is within 4 points of the best single model for
  every true process, while each single model is 11–30 points worse for some process. With 30
  years every choice misses 66–87% of records.
- The Bayesian model passes a time-reversal check (forward and reversed posterior paths agree to
  within 1% of the interval width).

## Roadmap

1. ~~Critical-value calibration in Python (Appendix B)~~ done for k largest; exceedance
   (Bonferroni) calibration still to do.
2. ~~Nyblom stability test~~ done; the joint L* test still to do.
3. Pseudo-data size and power studies (`docs/power_study_plan.md`).
   - Stage A (exact GEV_k data with known paths) is done; see the results above.
   - Stage B (daily data from known parents: normal, gamma, Student-t and a bounded beta, with
     serial correlation) is running. It measures size and power when the extremes are only
     approximately GEV, comparing the paper's Nyblom test with the GEV linear-trend test,
     Mann–Kendall, and a simple two-segment test (separate GEV fits to the early and recent
     halves of the record). It also compares estimated return levels X_p (the level that 50%, 90%,
     95% or 99% of annual maxima stay below) with the truth implied by the generating function,
     and tests sensitivity to record length (30, 45, 85 and 150 years).
   - Stage B is done (results in `data/results/stage_b_*.csv`). With one value per year all
     tests hold their 5% size at any persistence; several values per year from a persistent
     daily series inflate false alarms badly unless the extremes are declustered. With one value
     per year the Nyblom test is no more powerful than Mann–Kendall (at least as powerful, within a
     point, in 95% of the 768 simulated changes; average power 47% against 45%); the location-trend
     LR test is stronger for location shifts but weak for tail changes. With three values per year
     the Nyblom test overtakes Mann–Kendall (average 56% against 47%, size-corrected).
4. ~~Stage C~~ done (see above and the report): 30, 45 and 85 years, one or three values per
   year, correlated and independent days, the authors' prior and a tighter one. Importance
   reweighting and Bayes factors are not yet done.
5. Alternative models for the change in return levels (see below): implemented as Stage C
   estimators; an external covariate (global mean temperature) is still to do.
6. ERA5: build k-largest or exceedance panels per grid cell or region, decluster to handle serial
   dependence, then run the tests and the estimators.

### Alternative models for the change

The Bayesian model lets every parameter wander as a random walk, which asks a lot of a few
decades of extremes. Two more constrained models are probably closer to what the data can
support:

1. **Linear trends in all three GEV parameters.** μ_t, log σ_t and ξ_t each linear in time. This
   extends the location-only trend fit, which misses changes in width and tail. Implemented as
   `trend3`, with `trend2` (ξ constant). Neither beats the location-only trend on average: the
   gain where the width or tail really changes is outweighed by the extra variance elsewhere.
2. **Parameters as monotone functions of the mean state (covariate model).** The GEV parameters
   depend on a covariate c_t, e.g. μ_t = a_μ + b_μ c_t, log σ_t = a_σ + b_σ c_t,
   ξ_t = a_ξ + b_ξ c_t, where c_t is a smoothed estimate of the local mean (or a global-mean
   temperature series). This asks how the distribution changes as the location warms (for
   temperature) or as mean precipitation rises (for precipitation), and lets nonlinear change in
   time enter through c_t. It needs test cases where it is the correct model, for example:
   - *Temperature-like:* a warming path that is flat and then accelerates, with the width and
     tail of the daily distribution changing in proportion to the warming.
   - *Precipitation-like:* a gamma parent whose scale is proportional to the mean, so every
     quantile scales with the mean (a Clausius–Clapeyron-type change), possibly with the shape
     changing too.
   The covariate would be estimated from all days of the year, not just the extremes, so it
   carries much more information than the extremes alone. Each case should also be fitted with
   the time-trend and Bayesian models, to see how much is lost when the change is not linear
   in time.

   Implemented: the test cases are the `warm_coupled`, `precip_scaling` and `precip_shape` knobs
   in `parents.py` (`scripts/stage_c.py --coupled`), and the estimators are `cov2` (μ and log σ
   linear in a local-linear smooth, 10-year bandwidth, of the annual mean; ξ constant),
   `cov3` (all three linear in the raw annual mean; unstable) and their oracles with the true mean
   path. `cov2` is the best estimator in these cases (results above).
