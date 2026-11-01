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

## Stage B: parent-distribution data (robustness of size, then power)

Simulate daily series, take the k largest per year, and run the same tests.
- Parents:
  - Gaussian: light tail; ξ → 0 slowly.
  - Gamma or Weibull: like precipitation.
  - Student-t with 4–10 degrees of freedom: heavy tail.
  - Raised cosine, density (1 + cos(π(x − µ)/s))/(2s) on [µ − s, µ + s]. Bounded, with a density
    that vanishes quadratically at the endpoint, so ξ = −1/3 exactly. Annual-maximum fits give
    ξ̂ = −0.35, a clean check of the bounded-tail (ξ < 0) case.
  - Tanh-normal (soft-saturating): X = c·tanh(Z/c), Z ~ N(0, 1). It is normal for |Z| ≪ c and
    saturates at ±c (physically, a cap such as a wet-bulb or soil-moisture limit). It is an affine
    transform of the logit-normal, since tanh(u) = 2·logistic(2u) − 1. Its tail near the cap
    decays faster than any power, so it is formally in the Gumbel domain (ξ = 0). In finite
    blocks, though, the fitted ξ̂ is clearly negative and depends on c (annual maxima: −0.44,
    −0.24, −0.17 for c = 1.5, 3, 5). It is the main test of what happens when "the" ξ is not a
    stable constant.
  - Warming for the saturating parents can enter as a shift in the latent mean (Z → Z + δ_t) with
    a fixed cap. Then the extremes approach the cap, their trend flattens, and the effective ξ
    drifts downward. That is a change in shape the paper's model would read as time variation
    in ξ. Also run a version where the cap itself moves.
- Serial correlation: AR(1) with φ ∈ {0, 0.5, 0.8}, the range seen in daily temperature anomalies.
- Seasonality: a seasonal cycle in mean and variance, so the yearly maximum comes mostly from
  one season.
- Block size: 365 days, and the 92 days of a single season.
- Declustering for k > 1: take the k largest values separated by at least r days, with
  r ∈ {0, 3, 7}. This shows whether clusters (one heatwave giving several of the k largest) break
  the size of the k > 1 designs.

First measure size (b = 0). Then measure power for a trend in the parent mean or variance, which is
how warming actually enters, rather than a trend in the GEV parameters directly.

## Stage C (later, separate): recovery of known paths

Check that the Bayesian TVP model recovers the true parameter paths and credible-interval coverage.
This needs the Stan model and is not part of this study.

A time-reversal check belongs here too: fit the model to the data in reverse order and check that
the estimated paths come out reversed. For the constant-parameter fits, Nyblom, trend LR and
Mann–Kendall this holds exactly and is already a unit test (`tests/test_power.py`). For the
Bayesian model it holds only if the prior is symmetric in time. A random-walk prior anchored at
the first period is not, so any asymmetry seen will come either from the prior or from a bug, and
the check tells the two apart.

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
