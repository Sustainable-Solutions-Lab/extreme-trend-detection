# Plan: pseudo-data power study (draft, 2026-11-01)

Müller & Watson (2024) calibrated the size of their tests by simulation but did not report power,
did not check that the tests behave when the data are only approximately GEV, and did little on
serial dependence. This study does those three things, with designs chosen to match ERA5 use.

## Questions

1. **Power.** How often does each test detect a change in the extremes of a given size, shape and
   sample design (k, T)?
2. **Comparison with what climate scientists use.** The standard tool is a non-stationary GEV with a
   linear trend in the location (LR test), or Mann–Kendall on annual maxima. When does the Nyblom
   test, which assumes no shape for the change, lose to these, and when does it win (steps,
   non-monotone change, change in scale or tail)?
3. **Robustness of size.** Under no change, do the tests keep 5% when the k largest come from a real
   parent distribution (finite block, not exactly GEV) with serial correlation and seasonality, as
   ERA5 daily data do?

## Stage A: exact GEV data (power and comparison)

Draw the k largest of each period from the GEV_k density with period-specific (µ_t, σ_t, ξ_t), using
the same exponential-sum construction as the calibration (`panels_from_gammas`, generalized to
time-varying parameters).

**Designs (k, T):**
- (1, 85): annual maxima, ERA5 1940–2024. The main case.
- (3, 85) and (10, 85): several largest per year. These are only valid if the k largest within a
  year are roughly independent; Stage B tests that.
- (1, 194) and (30, 4): the paper's designs. Calibration for both exists already.

**Alternatives:** paths over t ∈ [0, 1], each scaled by an effect size b.
- Linear trend in µ (warming shifts the distribution).
- Step change in µ at t = 0.5 and t = 0.8.
- Linear trend in σ (more variable extremes).
- Linear trend in ξ (fatter tail).
- Random walk µ_t = µ + b·σ·W(t), with W a standard Brownian motion on [0, 1]: the alternative the
  Nyblom test is designed for, and the one the TVP model assumes. Each replication draws a new path.
- Hump (rise then fall) in µ, where a linear-trend test should fail.

**Effect size:** the total change over the sample in units of σ: b ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2}.
For reporting, translate each into the change in the 10- and 100-year return levels.

**Null background:** ξ ∈ {−0.2, 0, 0.2, 0.5}, which covers temperature (ξ < 0) through precipitation
(ξ > 0), with µ = 0 and σ = 1.

**Tests:**
- Nyblom L_T with the ξ̂-adjusted critical value (the paper's procedure).
- Non-stationary GEV, LR test for a linear trend in µ, χ²₁ critical value. A version with a size-
  corrected critical value is also needed, since the χ² approximation is poor at T = 85.
- Mann–Kendall on the per-period maxima.
- Later: the paper's L* test, once implemented.

**Two kinds of power:**
- Nominal: rejection rate using each test's own critical value. This is what a user gets.
- Size-adjusted: critical value taken from the exact null at the true ξ. This compares the tests
  fairly when one of them over-rejects.

**Replications:** 2,000 per cell. That gives ±1 percentage point on a rate near 50%, which is enough
to draw power curves. Rough cell count: 5 designs × 6 path shapes × 7 effect sizes × 4 ξ ≈ 840
cells, about 1.7 M fits per test. The Nyblom test needs one GEV fit; the trend test needs two.
At the calibration's speed this is a few hours, niced, on 14 cores.

**Calibration needed:** the Nyblom ξ̂-adjustment for (1, 85), (3, 85) and (10, 85). The Nyblom run
for one design took 8 minutes at N = 40,000, so this is cheap.

## Stage B: daily data from known parent distributions

Simulate daily series from a known generating function, keep the k largest values of each year,
and run the same tests and fits as in Stage A. Everything about the generating function is known,
so the truth is exact. The GEV parameters of the extremes are not known exactly: the yearly
maxima are only approximately GEV, and that approximation is what this stage measures.

**Construction.** A daily Gaussian series Z with AR(1) correlation φ (stationary, unit variance)
is mapped through the parent's quantile function:

    X_d = m_t + s_t · ε(Z_d; tail_t)

where m_t (centre), s_t (width) and the tail parameter vary between years but are constant within
a year (`tvextremes/parents.py`).

**Parents.** Two candidates were dropped. The raised cosine went because the beta is the bounded
case with an exact tail index. Tanh-normal (soft cap, c·tanh(Z/c)) went because the beta already
gives a bounded variable whose location, width and tail can all change; its code stays in
`parents.py`.

| Parent | ε | Tail parameter | Notes |
|---|---|---|---|
| Normal | Z | — | light tail; ξ → 0 slowly |
| Gamma | standardized gamma(a) | shape a = 2 | skewed, like precipitation |
| Student-t | standardized t_ν | ν = 6 | heavy tail, ξ = 1/ν |
| Beta | 3·(2B − 1), B ~ beta(α, β) | α = β = 4 | hard bounds at ±3 daily sd; ξ = −1/β exactly (−0.25 at baseline, typical of temperature) |

For a given year the map from Z to X is increasing. So the k largest values of X fall on the same
days as the k largest of Z, and declustering picks the same days. Selection is therefore done
once per replicate on Z and shared by all changes (common random numbers).

**Truth.** For the same reason, the annual maximum of X is that year's map applied to the annual
maximum of Z, and so are its quantiles. The true X_p is the level the annual maximum stays below
in a fraction p of years:
- For independent days it is exact: X_p = map(Φ⁻¹(p^(1/365))).
- For correlated days, the quantiles of the annual maximum of Z come from one simulation of
  500,000 years per φ. The resulting error (about 0.005 daily sd at p = 0.99) is negligible.

**Changes**, one knob at a time, mirroring the three GEV parameters:

| Knob | Moves | Sizes (over the record) |
|---|---|---|
| mean | m_t, all parents | 0.1, 0.25, 0.5, 1 daily sd |
| width | s_t, all parents (beta: cap stretches with it) | 0.05, 0.1, 0.2, 0.4 in log s |
| tail | gamma a, t ν, beta β | 0.25, 0.5, 1, 1.5 in log of the parameter (beta: 0.2–0.8) |
| α (beta only) | position of the mass within the fixed bounds, ξ unchanged | 0.25, 0.5, 1, 1.5 in log α |

- **Central case:** no change.
- **Path shapes:** linear, step at mid-record, step at 80%, and exponential.
- **Direction:** only upward changes are run. Every statistic is invariant under time reversal
  (tested), so a downward change gives the same results as the reversed upward one.
- **Beta: α and β vary separately**, with the bounds fixed. β sets the upper tail, ξ = −1/β,
  whatever α is. α moves the mass within the range: the mean is α/(α + β), so raising α pushes
  the bulk and the maxima toward the fixed upper bound without changing ξ. This is the bounded
  case in which location, width and tail of the extremes all shift as the headroom to a fixed
  limit shrinks. It replaces an earlier "capped warming" that shrank the whole distribution.
- **The beta tail range is limited** to β ≥ 2.68 (true ξ ≥ −0.37), via log-β changes of at most
  0.8 from β = 4. GEV maximum likelihood is regular only for ξ > −0.5 (Smith 1985). Moreover, a
  constant GEV fitted across a large change in a bounded variable is pushed far below the true ξ,
  to about −0.9, where the fitted endpoint hugs the sample maximum and the Nyblom information
  matrix does not exist. With β = 3 at baseline and β down to 2, 60–84% of fits failed in the
  largest step change; with β = 4 and β ≥ 2.68 it was 0–7%. The few failures that remain are
  reported per case. Since the constant fit is the null model of the Nyblom test, this is a real
  limit of the method for strongly changing bounded variables, not only of this simulation.
- **The β knob also changes the width.** The scale factor 3 is fixed at the baseline, so the daily
  variance is 9·Var(2B − 1), which moves with β (and α).

**Two-segment test (added 2026-11-01).** Fit a GEV separately to the first and second halves
of the record and test whether (µ, σ, ξ) differ: LR = ll(first) + ll(second) − ll(both), with 3
restrictions. It asks only "are the two halves different?", the least demanding question about
change, and it gives the most interpretable answer ("X₉₉ in the recent half vs the early
half"). Each half is nearly homogeneous, so it does not hit the irregular fits that break the
constant fit for large bounded changes (beta: 158 of 2,000 Nyblom failures and no split failures
in the largest step). Expected weaknesses: a rise-then-fall change, and changes near the end of
the record. Its nominal critical value is χ²₃ until it is calibrated like the paper's tests
(`calibrate_cv.py --tests split`); summaries switch to the calibrated value automatically.

**Scoring:**
1. **Size and power:** the Nyblom test (ξ̂-adjusted), the GEV linear-trend LR test,
   Mann–Kendall and the two-segment test, at their nominal critical values and size-adjusted.
   Each test is scored on the replicates where its own fit succeeded, with failures counted per
   test. The output is
   `data/results/stage_b_summary.csv`.
2. **Return levels against the truth:** for p = 0.5, 0.9, 0.95 and 0.99, compare estimated X_p
   with the truth implied by the generating function. Scores are bias and RMSE, in the first year,
   in the last year, and for the change between them. The output is
   `data/results/stage_b_return_levels.csv`. Two estimators so far:
   - the constant-parameter GEV fit, which estimates no change;
   - the GEV fit with a linear trend in location;
   - the two-segment fits: the first half scored against the first year, the second half against
     the last year (the question "what is X_p now?").

   The Bayesian time-varying model (Stage C) will be scored the same way.

**Replicates:** 10,000 for the no-change case (±0.22 percentage points on a 5% size) and 2,000 for
each change (±1.1 points on a 50% power).

**Overnight run (2026-11-01), in priority order:**

| Step | What | Estimated time |
|---|---|---|
| 1 | T = 85, k = 1, φ ∈ {0, 0.5, 0.8}, all 4 parents | 1.2 h |
| 2 | T = 85, k = 3, same grid | 1.2 h |
| 3 | Record length: Nyblom calibration, then T ∈ {30, 45, 150}, k = 1, φ = 0.5, all parents | 1.5 h |
| 4 | T = 85, k = 10, φ = 0.8, without declustering and with r = 3 days | 1.6 h |

T = 30 is a climate-normal period, T = 45 the satellite era (1979 onward), T = 85 ERA5, and
T = 150 a long station record. Results are committed after each step.

**Later:**
- Seasonality: a seasonal cycle in mean and variance. It breaks the shared-selection shortcut,
  so it needs the full daily transform and runs on a subset.
- The full record-length grid (all φ and k).
- Declustering gap r = 7 days.

## Stage C: how well are return levels predicted?

The question is not only whether time variation can be detected, but how well each method
predicts a return level such as X_0.99, the level the annual maximum exceeds in 1% of years,
especially in the most recent year.

**Estimators**, all fitted to the same simulated panels:
- `const`: constant-parameter GEV (ignores change).
- `trend`: GEV with a linear trend in location.
- `split`: separate GEV fits to the two halves.
- Added after the first run (`tvextremes.stage_c.EXTRA_NAMES`, MLE, computed by regenerating the
  records from their seeds): `trend2` (µ and log σ linear in time), `trend3` (µ, log σ and ξ linear
  in time), `recent30` (constant fit to the last 30 years), `cov2` (µ and log σ linear in a
  local-linear smooth of the annual mean of all days, 10-year Gaussian bandwidth, ξ constant),
  `cov3` (all three linear in the raw annual mean; unstable), and the oracles `cov2_true` and
  `cov3_true` with the true mean path as covariate.
- `tvp_mean`, `tvp_median`: posterior mean and median of X_p(t) from the authors' Bayesian model
  with four random walks (`stan/gev_4rw.stan`, their `GEV_4RW_Model.stan` with leaner output;
  `tests/test_stage_c.py` checks the log density is identical). Their prior settings: ξ in
  (−0.9, 2.1), random-walk scale factors 1.0 (ξ), 0.5 (α), 0.5 (s), exponential(1) priors on the
  unscaled innovation sds.

**MCMC:** 4 chains of 1,000 warmup and 1,000 sampling iterations, `adapt_delta` 0.95. The authors
use 5,000 + 10,000 for single published fits, which is too slow for a Monte Carlo study. With
0.9 a few percent of transitions diverged; 0.95 cuts that to about 0.5%, and the estimated
X_0.99 did not change (to 0.02) between 0.9, 0.95 and 0.99. Divergences, R-hat and ESS are stored
for every fit.

**Data:** Stage B's parent-distribution generator, so the true X_p(t) is known exactly.

**Metrics** (`tvextremes.stage_c.score_rows`), for p = 0.5, 0.9, 0.99:
- *True exceedance probability* of the estimated level in the last year (ideal 1 − p), its
  median and mean over panels, and the share of panels where it exceeds 2(1 − p) (risk understated
  by more than a factor of 2).
- Bias and RMSE of the level in the last year and over all years (units of the daily parent sd).
- Bias and RMSE of the estimated change, first year to last.
- Bayesian only: coverage and width of the 90% credible interval, last year and all years.

**First run** (`scripts/stage_c.py`): T = 85, k = 1, φ = 0.5, 100 panels per scenario. Per
parent: no change, a linear and a mid-record step change in the mean, a linear change in width,
a linear tail change (gamma, t, beta) and, for the beta, a change in α. Amounts are mid-range
Stage B values (detected roughly half to most of the time).

A time-reversal check belongs here too: fit the model to the data in reverse order and check that
the estimated paths come out reversed. For the constant-parameter fits, Nyblom, trend LR and
Mann–Kendall this holds exactly and is already a unit test (`tests/test_power.py`). For the
Bayesian model it should also hold. The authors' Stan code subtracts each random walk's own
mean, so the level parameters are time averages, not first-period values. Their priors are flat
or symmetric, and the random-walk increments are i.i.d. Gaussian. Together these make the prior
symmetric in time. So reversed data should give reversed posterior paths, up to MCMC noise, and
any systematic asymmetry would point to a bug.

## Code and outputs

- `tvextremes/pseudo.py`: path generators, GEV_k panels with time-varying parameters, and parent-
  distribution simulators with declustered k-largest extraction.
- `tvextremes/trend_tests.py`: non-stationary GEV linear-trend LR test and Mann–Kendall.
- `scripts/power_study.py`: runs a grid of cells in parallel (niced, one BLAS thread per worker) and
  writes one row per cell.
- Raw results go to `data/output/power/`. The summary table of power by cell is small and could be
  committed (e.g. `data/results/power_summary.csv`).
- Unit tests: generated panels have the right distribution (compare with `gev_logpdf`, as for the
  calibration), and at b = 0 Stage A reproduces the calibrated 5% size.
