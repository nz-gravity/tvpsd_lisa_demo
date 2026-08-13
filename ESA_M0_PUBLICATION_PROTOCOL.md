# ESA-orbit M0 publication analysis protocol

The consolidated reader-facing account of the superseded pre-fix analysis is
[`ESA_M0_RESULTS_AND_FINDINGS.md`](ESA_M0_RESULTS_AND_FINDINGS.md). This file
retains the prospective protocol and claim rules for the corrected reruns.

Status: corrected coarse-reference likelihood and WDM-projected references
implemented; continuous and seven-day-gap production anchors pending rerun.
Independent noise realizations and an external LDC dataset remain final
replication exercises.

## Inputs available in a blind analysis

- X/Y/Z or A/E/T TDI series, cadence, timestamps, and gap metadata.
- Orbit/light-travel-time ephemerides and the exact TDI convention.
- Public nominal OMS and test-mass requirement spectra.
- No injected source series, noise realization, component truth, or injected
  noise amplitude.

## Preprocessing and inference contract

- Band: requested `1e-4`--`1e-1 Hz`; report the realized WDM cell centers.
- Numerical scale: all frequency cells in retained training rows only.
- Adaptive likelihood bins: the same inference population, including
  response-null cells; excluded time rows and injected truth cannot enter.
- Inference mask: retained training rows only. The response-null mask never
  removes a cell from the likelihood.
- Evaluation mask: the nominal orbit/TDI response threshold and dilation are
  used for stable relative/log truth accuracy and continuum whitening. Report
  an additional all-cell held-out whitening check including the nulls.
- Estimand: analytic OMS/TM references and injected truth are integrated over
  the compact squared Meyer frequency kernel for each interior WDM cell.
  Point-evaluated response is used only to define diagnostic null corridors.
- Split: disjoint training, validation, and test time blocks. Once method
  tuning is complete on this realization, it is no longer an untouched final
  test; independent data supply the final confirmation.
- Reference residual: `log S = log R + g(f) + h(t,f)`. The interaction has
  zero time mean and its own five-interior-knot time basis, selected on the
  validation cohort before the final five-knot production run.

## Required comparisons

- Reference only `R`.
- Matched stationary residual `R+g`.
- Matched time-varying residual `R+g+h`.
- Free total TV surface as a reference-agnostic robustness fit.

## Primary blind metrics

- Held-out posterior-predictive Whittle log density for TV versus stationary,
  integrated over posterior draws with paired complete-time-block bootstrap
  uncertainty. Plug-in geometric-mean scores remain secondary diagnostics.
- Held-out `mean(z)`, `mean(z^2)`, median `z^2 / median(chi2_1)`, and central
  90% fraction for `z=w/sqrt(S)`.
- Lag-one time and frequency products of held-out whitened coefficients.
- The same diagnostics separated into low (`<=3 mHz`), retained full, and high
  continuum (`>=20 mHz`) bands.

## Simulation-only metrics

- Log-RMSE, log bias, and 90% truth inclusion on validation/test cells.
- These never enter a blind-data fitting decision.

## Sampler acceptance gates

- zero divergences;
- finite maximum R-hat `<=1.05`;
- minimum ESS `>=50`;
- tree-depth saturation `<=0.05`;
- minimum E-BFMI `>=0.3`.

## Required sensitivity checks on the existing realization

- interaction time-knot count;
- interaction-scale prior width;
- nominal OMS/TM amplitude and broad spectral-tilt misspecification;
- response-null threshold and dilation;
- gap duration and WDM-row buffer.

These checks selected five interaction time knots and
a one-pixel gap-edge buffer. Prior/reference perturbations and the one-/thirty-
day gap runs that do not pass production sampler gates are retained as
point-estimate stress tests only. The prior anchors passed all gates, but their
numerical results are superseded by the coarse-reference and WDM-projection
corrections. Replacement runs must pass the same gates before manuscript use.

## Superseded claims to re-evaluate after rerun

- The nested interaction materially improves held-out prediction below 3 mHz,
  gives a small full-band improvement, and gives no meaningful improvement in
  the already-correct high-frequency continuum.
- Observed boundary whitening supports excluding one WDM time pixel on each
  side of a time-domain gap; a second pixel provides no measurable benefit.
- Smooth interpolation remains accurate for the simulated one-, seven-, and
  thirty-day gaps, with modest degradation at thirty days. This is a
  truth-only simulation statement, not a diagnostic available on external
  data.
- Posterior intervals are not yet demonstrated to have calibrated repeated-run
  coverage. The low-band interval contains the injected truth in 70.7% of test
  cells on this realization, rather than the nominal 90%.

## Final replication after method freeze

- multiple independent ESA-orbit noise/Galactic realizations for empirical
  coverage and run-to-run variability;
- one external LDC-style dataset using only released data, orbit files, and
  public nominal noise curves;
- optional A/E/T replication if component separation is promoted beyond M0.
