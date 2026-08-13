# ESA-orbit M0 method-improvement audit

See [`ESA_M0_RESULTS_AND_FINDINGS.md`](ESA_M0_RESULTS_AND_FINDINGS.md) for the
superseded pre-fix scientific results. The coarse reference-offset correction
and rerun plan are recorded in
[`COARSE_REFERENCE_OFFSET_CORRECTION.md`](COARSE_REFERENCE_OFFSET_CORRECTION.md).
This file remains the implementation and method-tuning audit trail.

This note separates changes that can be validated on the existing X2
realization from studies that require newly generated independent data. It is
an analysis record, not manuscript text.

## Implemented on the existing realization

1. **Truth-free adaptive likelihood bins.** Frequency bins are selected from
   robust time-block medians of training-coefficient log power. Response-null
   training cells enter the corrected pilot and likelihood; validation, test,
   and gap-contaminated rows cannot enter.
2. **Matched stationary P-spline.** The stationary Bayesian comparator now uses
   the same frequency knots, adaptive bins, inference population, training rows,
   coefficient units, smoothing-prior family, and sampler settings as the TV
   fit. When a reference PSD is supplied, the comparator uses the same reference
   and estimates only a frequency-dependent residual,
   `log S(t,f) = log S_ref(t,f) + g(f)`. The per-frequency stationary
   training-power mean is retained as a more flexible benchmark.
3. **Prospective split.** Seven-fold time blocks reserve separate validation
   and test folds. Both are excluded from the pilot and likelihood. The 48-knot
   free-surface specification was locked before inspecting the new test fold.
4. **Reference-PSD residual model.** An optional nominal analytic X2
   instrumental PSD supplies a fixed log-PSD offset. Orbit ephemerides, the TDI
   convention, and transfer geometry are treated as known; the nominal OMS and
   test-mass noise spectra and levels are assumptions rather than consequences
   of knowing the response. The P-spline remains a smooth multiplicative
   residual, so the reference amplitude is not fixed. Null corridors remain
   excluded. Controlled OMS/TM scale and spectral-tilt perturbations test
   sensitivity to an imperfect reference.
5. **Gap and notch stress controls.** The runner supports response-threshold and
   dilation sensitivity, adjustable WDM-row buffers, and single contiguous gaps
   with configurable duration and location.
6. **Rejected full-rank nested residual.** The first `g(f)+h(t,f)` prototype
   mean-centred the unrestricted tensor model's existing time basis. It added
   one scale parameter without reducing the interaction rank and was stopped
   after its production run became materially slower. The retained model gives
   `h` an independent coarse time basis. Validation selected five interior
   knots: three underfit the annual low-frequency structure, while five retain
   25% fewer surface parameters than the unrestricted tensor and improve its
   full/high-band error. Thus slow variation and a lower parameter count are
   encoded structurally.

## Completed single-realization checks

- The selected five-knot continuous fit passes every sampler gate: zero
  divergences, maximum R-hat 1.022, minimum ESS 168, no tree-depth saturation,
  and minimum E-BFMI 0.863. Its interaction has 361 parameters versus 480 for
  the unrestricted tensor surface.
- On the untouched test blocks, adding `h(t,f)` to `R+g` improves mean Whittle
  log score by 0.0212 per retained cell below 3 mHz and 0.00076 over the full
  retained band. The change above 20 mHz is -1.2e-5: the flexible interaction
  does not manufacture a meaningful high-frequency improvement where the
  nominal reference is already correct.
- The continuous low-band truth log-RMSE is 0.0459. Full-band log-RMSE is
  0.0137. Truth coverage is below nominal in the low band (0.707), so the
  credible band must not be presented as calibrated empirical coverage from a
  single realization.
- Interaction-scale prior widths 0.25, 0.5, and 1.0 give essentially unchanged
  point estimates. The short 0.25 and 1.0 sensitivity chains fail production
  convergence gates and support robustness of the fitted surface only, not
  posterior-tail claims.
- Deliberately perturbed OMS/TM amplitudes and broad reference tilts are absorbed
  by the residual surface with stable point estimates. Some short perturbation
  fits fail sampler gates, so these are directional misspecification checks,
  not additional production posteriors.
- Rescoring the locked fit over nine response-null masks (ratio thresholds
  0.20--0.50 and dilations of 3--9 frequency bins) leaves the nested full-band
  log-RMSE in 0.01370--0.01378 and the high-continuum log-RMSE in
  0.00923--0.00924. The null-aware accuracy conclusion is therefore insensitive
  to the precise reported notch boundary over this prespecified grid.
- For a seven-day gap, the 300-warmup/300-draw four-chain TV fit passes all
  sampler gates. With no boundary buffer, the blind low-band mean `w^2/S` on
  observed rows adjacent to the gap is 2.69. One- and two-pixel buffers reduce
  it to 1.08 and 1.09, respectively. One pixel is therefore selected without
  spending the extra observed rows required by two pixels.
- With the one-pixel rule fixed, truth-only low-band interpolation log-RMSE is
  0.0417, 0.0401, and 0.0542 for one-, seven-, and thirty-day gaps. The one- and
  thirty-day sensitivity chains are lower-budget and do not pass all production
  sampler gates; use them as interpolation stress checks only.

## Acceptance criteria before manuscript use

- Zero divergences, max R-hat <= 1.05, minimum ESS >= 50, tree-depth
  saturation <= 0.05, and minimum E-BFMI >= 0.3.
- Report validation and final-test scores separately.
- Report matched stationary P-spline and empirical per-frequency stationary
  benchmarks on identical cell cohorts.
- For the response-offset model, report baseline misspecification magnitude and
  never describe it as the same estimand/prior information as the free-surface
  model.
- Compare the time-varying residual only against a stationary residual using
  the identical reference PSD; otherwise the stationarity comparison is
  confounded by different physical information.
- Treat short pilot chains only as geometry or point-estimate diagnostics.
- Continue excluding response-null corridors from relative/log accuracy and
  coverage calculations; report all-cell and notched whitening separately, the
  excluded evaluation fraction, and mask sensitivity. Do not exclude nulls
  from inference or the adaptive-bin pilot.

## Deferred until new independent data are generated

1. Repeated-realization empirical coverage and sampling-variability estimates.
2. Final untouched evaluation after any further method tuning.
3. Channel replication in Y2/Z2 or a physically complete multichannel study.
## WDM projection question: closed for the present grids

`wdm_psd/notes/WDM_PROJECTION_VALIDITY.md` directly measures the installed
transform. Its Meyer frequency kernel is compactly supported within
`|f-f_m| <= 2 df / 3`, with only `3.3e-23` of power outside that support. There
is therefore no long-range leakage path from a response-null cell to a distant
continuum at any dynamic range. End-to-end simulations are unbiased in nulls.

For the present `nt=2048` X2 analysis, only the local curvature approximation
remains: the note scales the measured `nt=32` error to roughly 0.2% at the
coarser grid. This is below the current statistical and model errors and is not
a blocker. Re-open the check only for the note's stated conditions: still
larger `nt`, the lowest few channels, features narrower than `df`, or a changed
Meyer-window parameter `a`.

## Locked model hierarchy for blind use

All models use identical retained cells and supplied orbit/TDI geometry:

1. `R`: nominal reference PSD only.
2. `R+g`: `log S = log R + g(f)`, a stationary spectral correction.
3. `R+g+h`: `log S = log R + g(f) + h(t,f)`, where `h` has zero temporal
   mean, a separate coarse time basis, and a hierarchical global scale.
4. `free-TV`: no physical reference; retained as a robustness analysis, not a
   like-for-like prior-information comparison.

Model comparison on unknown data uses held-out mean Whittle log score and
whitening diagnostics. Truth RMSE and interval inclusion are simulation-only
validation metrics. NUTS samples are not marginal-likelihood evidence; no
Bayes-factor claim is permitted without a separate normalized-prior evidence
calculation.
