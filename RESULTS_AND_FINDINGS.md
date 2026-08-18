# ESA-orbit LISA surface PSD analysis: results and findings

**Analysis status:** superseded pending corrected coarse-reference reruns
**Last updated:** 2026-08-13  
**Scientific scope:** estimation of the total time-varying PSD (H_agn/H_orb), without
separating instrumental noise and the Galactic component

> **Correction notice (2026-08-13):** the response-informed coarse likelihood
> averaged `log(R)` and reused the geometric-mean reference in its quadratic
> term. The correct sufficient statistic is `sum(w^2/R)`, with division by the
> cell-level reference before block summation. The shared library is fixed and
> the continuous and seven-day-gap anchors are being rerun. All numerical surface-study
> results below document the superseded pre-fix analysis and must not be quoted
> as current results. See
> [`COARSE_REFERENCE_OFFSET_CORRECTION.md`](COARSE_REFERENCE_OFFSET_CORRECTION.md).
> The pre-fix run also used the response-null mask in its likelihood. The
> corrected rerun uses all frequency cells in retained training rows and
> reserves the null mask for relative/log accuracy and notched whitening only.

## Technical summary

The response-informed, stationary-plus-interaction model

\[
\log S(t,f)=\log R(t,f)+g(f)+h(t,f)
\]

recovers the annual low-frequency modulation in the simulated X2 data while
remaining effectively inactive in the high-frequency continuum where the
reference is already correct. The interaction `h(t,f)` has its own coarse
five-knot time basis and zero temporal mean. It uses 361 coefficients, compared
with 480 for the unrestricted tensor surface, so slow temporal variation and a
lower parameter count are encoded in the model rather than inferred from a
full-resolution basis.

The continuous four-chain production fit passes every prespecified sampler
gate. On untouched test blocks, adding `h(t,f)` to the matched stationary model
improves mean Whittle log score by 0.0212 per retained cell below 3 mHz and by
0.00076 over the full retained band. Above 20 mHz the change is
`-1.2e-5`, which is scientifically negligible and slightly favours the simpler
stationary correction. This is the desired behavior: flexibility is used where
the data support time variation and does not create a spurious high-frequency
gain.

For gaps, blind whitening of observed cells adjacent to a seven-day gap
supports removing one WDM time pixel on each side. With no buffer, the
low-frequency mean normalized power is 2.69; with one and two pixels it is 1.08
and 1.09. A second pixel therefore removes more data without a measurable
benefit. In the simulation-only gap interior, low-band log-PSD RMSE is 0.0417,
0.0401, and 0.0542 for one-, seven-, and thirty-day gaps, respectively.

These results establish accurate point estimation and useful blind predictive
behavior on this realization. They do **not** yet establish repeated-run
coverage calibration, generalization across independent realizations, or
performance on external LISA Data Challenge data. In particular, the nominal
90% interval contains the injected low-band truth in only 70.7% of retained
test cells on this realization.

## Key findings

### A coarse time interaction captures the low-frequency modulation

The selected five-knot interaction follows the injected annual structure below
3 mHz. A matched stationary residual captures the time-averaged spectrum but
cannot follow this modulation. The middle panel below shows that most of the
stationary model's error is concentrated at low frequencies, while the
right-hand panel shows the incremental predictive value of the interaction.

![Continuous nested-model results](publication_results/esa_x2_continuous_nested.png)

The figure uses three distinct comparisons:

1. low-band modulation relative to the median injected PSD;
2. frequency-resolved log-PSD RMSE on response-retained test cells;
3. held-out mean Whittle log-score gain from adding `h(t,f)` to `R exp(g)`.

The response-null corridors are excluded from relative/log accuracy
calculations because we do not expect to hit moving transfer-function zeros
exactly. In the corrected analysis this is an evaluation choice only: null
cells remain in the likelihood and adaptive-bin pilot, and all-cell held-out
whitening supplies the stringent null-inclusive diagnostic.

### One WDM pixel is sufficient to protect gap boundaries

The gap analysis separates three questions that are otherwise easy to conflate:

- interpolation through the missing interval, assessed using simulation truth;
- whitening immediately outside the missing interval, assessed from observed
  coefficients without truth;
- ordinary held-out prediction away from the gap.

![Gap robustness results](publication_results/esa_x2_gap_robustness.png)

The blind boundary result selects the buffer. The gap-interior truth error is a
stress test only and was not used to choose the buffer. This distinction is
important for later external-data analyses, where the PSD inside a real gap is
unknown.

### Whitening checks are necessary but belong in validation material

For the selected continuous model, held-out normalized coefficient power is
close to its `chi-square(1)` target:

| Band | Mean `z^2` | Median `z^2 / median(chi-square(1))` | Central 90% fraction | Lag-1 time product | Lag-1 frequency product |
|---|---:|---:|---:|---:|---:|
| Below 3 mHz | 1.0121 | 1.0183 | 0.8997 | -0.0091 | 0.0014 |
| Full retained band | 1.0023 | 1.0034 | 0.8997 | 0.0014 | 0.0009 |
| Above 20 mHz | 1.0021 | 1.0037 | 0.8996 | 0.0010 | 0.0019 |

Here `z=w/sqrt(S_hat)`. These diagnostics are useful evidence that the fitted
surface predicts unseen WDM coefficients at the correct scale. They should be
reported compactly in the results and shown graphically only in validation or
appendix material; they should not replace the reader-facing PSD recovery
figures.

## Scope and data provenance

The analysis uses the archived dataset
`lisa_data_generation/combined_esa_xyz.h5`, generated from:

- instrumental noise: `lisa_data_generation/noise2a/tdi.h5`;
- ESA orbit ephemerides: `lisa_data_generation/noise2a/orbits.h5`;
- a calibrated Galactic realization, with stored amplitude scale
  `0.8432771371`.

The present study uses the second-generation Michelson `X2` channel. The time
series spans 364.09 days. The WDM representation has 2,046 time pixels and
3,069 frequency channels. The requested frequency range is
`1e-4`--`1e-1 Hz`; the realized WDM channel centers span
`1.30208e-4`--`1e-1 Hz`, with spacing `3.25521e-5 Hz`. One WDM time pixel spans
approximately 4.27 hours.

The archive contains the injected instrumental, Galactic, and total PSD
surfaces. Those truth arrays are used only for simulation validation. They are
not used to set the numerical PSD scale, likelihood bins, priors, model
selection scores, or gap buffer.

## Model specification

### What the reference PSD contains

`R(t,f)` is a nominal analytic X2 instrumental PSD. It combines:

- orbit/light-travel-time ephemerides;
- the TDI convention and time-dependent transfer geometry;
- assumed nominal optical metrology system (OMS) and test-mass (TM) noise
  spectra and amplitudes.

Treating the orbit-dependent response as known is appropriate when the orbit
solution and TDI construction are supplied with the data. It does **not** imply
that the physical OMS and TM noise levels are known exactly. Those are baseline
assumptions, and `g(f)+h(t,f)` is a free multiplicative correction around them.

For external data, the same analysis therefore needs only the TDI series,
timestamps and gaps, orbit/TDI metadata, and public nominal noise curves. It
does not require injected truth. Reference misspecification is checked by
perturbing the OMS/TM mixture and adding broad spectral tilts.

### Nested residual decomposition

The residual is decomposed into:

- `g(f)`: a stationary frequency P-spline correction;
- `h(t,f)`: a zero-time-mean interaction using a separate coarse time basis;
- `sigma_interaction`: a global hierarchical scale for the interaction.

The zero-time-mean constraint makes `g(f)` the time-averaged residual and stops
the stationary and time-varying components from duplicating the same degree of
freedom. The selected interaction uses five interior time knots. With 36
interior frequency knots, the nested model has 361 residual parameters. The
plain tensor model with eight time knots has 480.

The initial prototype was rejected because it mean-centered the full tensor
time basis and dropped one column, leaving essentially the same time-frequency
rank while adding `g(f)` and a scale parameter. It was slower and did not encode
slow variation. No result from that stopped run is used.

### Model hierarchy

All comparisons use identical retained cells and the same orbit/TDI geometry:

1. `R`: nominal reference only;
2. `R exp(g)`: matched stationary residual;
3. `R exp(g+h)`: matched stationary plus slow time-frequency interaction;
4. free-TV: a reference-agnostic total PSD surface used as a robustness fit.

The primary stationarity comparison is between models 2 and 3. Comparing the
response-informed TV model only to a free or empirical stationary spectrum
would confound time variation with different physical prior information.

## Experimental design and metrics

### Prospective splits

Time pixels are assigned to a seven-fold repeating block split. Fold 5 is used
for validation and fold 6 for final testing. Both are excluded from likelihood
construction and adaptive-bin pilot calculations. After the present method
tuning, this realization is no longer an untouched final test; future final
confirmation must use independently generated data.

### Coarse likelihood evaluation

The superseded continuous analysis evaluated the block-summed likelihood over
512 time bins and 623 adaptive frequency bins rather than every WDM cell
individually. The bins are constructed from retained training coefficients
only. That run incorrectly excluded response-null cells from the bin pilot and
likelihood. The corrected rerun includes them; gaps, validation rows, test rows,
and injected truth still cannot enter.

The numerical PSD scale is the median retained training WDM power divided by
the median of `chi-square(1)`. This replaced an earlier truth-derived scale.

### Metric definitions

- **Mean Whittle log score:** mean held-out log likelihood per retained WDM
  cell. Higher is better. The incremental value of time variation is the score
  of `R exp(g+h)` minus that of `R exp(g)` on the same cells.
- **Log-PSD RMSE:** root mean square of `log(S_hat)-log(S_truth)` on a stated
  truth-available cohort. Lower is better.
- **Truth coverage:** fraction of truth-available cells whose injected PSD lies
  inside the pointwise nominal 90% posterior interval. This is not the same as
  repeated-realization coverage.
- **Normalized power:** `z^2=w^2/S_hat`; under the diagonal WDM likelihood its
  mean should be one and its distribution should follow `chi-square(1)`.
- **Lag-one products:** empirical neighboring products of whitened
  coefficients in time or frequency. Values near zero are expected under the
  diagonal approximation.

## Continuous-data production results

The selected fit used 300 warmup and 300 retained draws in each of four chains.
Its total wall time was 1,185 s, including 854 s in NUTS.

### Sampler diagnostics

| Diagnostic | Acceptance gate | Result | Status |
|---|---:|---:|---|
| Divergences | 0 | 0 | Pass |
| Maximum R-hat | `<=1.05` | 1.0219 | Pass |
| Minimum ESS | `>=50` | 168.4 | Pass |
| Tree-depth saturation | `<=0.05` | 0.000 | Pass |
| Minimum E-BFMI | `>=0.3` | 0.863 | Pass |

The matched stationary comparator also passes: maximum R-hat 1.0476, minimum
ESS 86.4, zero divergences, no tree-depth saturation, and minimum E-BFMI 1.206.

The posterior median interaction scale is 0.04785, with a 5th--95th percentile
interval of 0.04382--0.05251.

### Blind predictive comparison on test cells

| Band | `R` score | `R exp(g)` gain over `R` | `R exp(g+h)` gain over `R` | Increment from `h` |
|---|---:|---:|---:|---:|
| Below 3 mHz | 12.8664 | 2.5549 | 2.5761 | **0.02118** |
| Full retained band | 11.7863 | 0.08131 | 0.08207 | **0.000758** |
| Above 20 mHz | 11.5188 | 0.000002 | -0.000010 | **-0.000012** |

The low-band improvement is both scientifically coherent and visible in the
recovered modulation. The full-band average is smaller because most retained
cells lie at frequencies where the reference or stationary correction is
already adequate. The high-frequency result is a useful negative control.

### Simulation-truth accuracy on test cells

| Band | Cells | Log-PSD RMSE | Log bias | Nominal 90% truth inclusion |
|---|---:|---:|---:|---:|
| Below 3 mHz | 25,988 | 0.04591 | -0.00217 | 0.7073 |
| Full retained band | 862,760 | 0.01374 | 0.00023 | 0.8862 |
| Above 20 mHz | 684,348 | 0.00923 | 0.00042 | 0.8989 |

The point estimate is nearly unbiased. The low-band interval is too narrow or
misspecified relative to the local truth variation on this realization. This
must be stated whenever the credible interval is plotted.

## Why five interaction time knots were selected

The three- and five-knot models both pass production sampler gates, but three
knots underfit the annual low-frequency shape.

| Interaction time knots | Residual parameters | Validation low-band RMSE | Test low-band RMSE | Full-band RMSE | High-band RMSE |
|---:|---:|---:|---:|---:|---:|
| 3 | 281 | 0.06392 | 0.06395 | 0.01588 | 0.00871 |
| 5 | 361 | **0.04593** | **0.04591** | **0.01374** | 0.00923 |

Five knots were selected using the validation cohort. The slightly lower
high-frequency RMSE from three knots was not used to override the clear
low-frequency underfit, particularly because the high-frequency reference is
already exact in this simulation and neither model gains predictive score
there.

## Gap analysis

### Seven-day production anchor

The seven-day, one-pixel-buffer fit used 300 warmup and 300 retained draws in
each of four chains. It passes all TV sampler gates: zero divergences, maximum
R-hat 1.0195, minimum ESS 226.4, no tree-depth saturation, and minimum E-BFMI
0.732. The matched stationary comparator has maximum R-hat 1.077 and therefore
does not support stationary posterior-interval claims in this run.

The gap removes 2.05% of WDM time rows after boundary buffering. Inside the
truth-available missing interval, low-band log-PSD RMSE is 0.04015 and nominal
90% truth inclusion is 0.784. On ordinary held-out low-band cells, the TV model
improves mean log score over the stationary residual by 0.02145 per cell.

### Buffer sensitivity at fixed seven-day duration

| Buffer pixels per edge | Excluded row fraction | Adjacent observed mean `z^2` | Adjacent central 90% fraction | Gap-interior RMSE | Sampler gates |
|---:|---:|---:|---:|---:|---|
| 0 | 0.0196 | **2.6905** | 0.8624 | 0.03940 | Pass |
| 1 | 0.0205 | **1.0796** | 0.8848 | 0.04015 | Pass, production |
| 2 | 0.0215 | **1.0866** | 0.8904 | 0.04001 | Fail: R-hat 1.056 |

The no-buffer result exposes contaminated boundary coefficients even though its
truth-only interpolation RMSE is marginally smaller. This is why the buffer is
selected using observed boundary whitening rather than gap-interior truth.

### Gap-duration sensitivity with one-pixel buffer

| Gap duration | Excluded row fraction | Adjacent observed mean `z^2` | Gap-interior log-PSD RMSE | Gap truth inclusion | Sampler gates |
|---:|---:|---:|---:|---:|---|
| 1 day | 0.0039 | 1.1144 | 0.04174 | 0.7416 | Fail: R-hat 1.062 |
| 7 days | 0.0205 | 1.0796 | 0.04015 | 0.7841 | Pass, production |
| 30 days | 0.0836 | 1.1200 | 0.05424 | 0.7779 | Fail: R-hat 1.052 and ESS 39.9 |

The thirty-day gap produces modest degradation rather than catastrophic
failure. The one- and thirty-day results are point-estimate stress checks only
because their short chains miss the production sampler gates.

## Robustness checks

### Interaction-scale prior

Changing the HalfNormal prior width from 0.25 to 1.0 leaves the fitted point
surface essentially unchanged:

| Prior width | Median interaction scale | Low-band RMSE | Full-band RMSE | High-band RMSE | Maximum R-hat |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.04777 | 0.04589 | 0.01377 | 0.00929 | 1.061 |
| 0.50, selected | 0.04785 | 0.04591 | 0.01374 | 0.00923 | 1.022 |
| 1.00 | 0.04792 | 0.04595 | 0.01377 | 0.00926 | 1.089 |

The 0.25 and 1.0 runs were deliberately shorter and do not support posterior
tail comparisons. They support only the conclusion that the point estimate is
insensitive to this prior range.

### Reference misspecification

Two deliberately perturbed references were fitted:

| Reference perturbation | Median reference log error | Low-band RMSE | Full-band RMSE | High-band RMSE | Maximum R-hat |
|---|---:|---:|---:|---:|---:|
| OMS `1.5`, TM `0.7`, tilt `+0.15` | 0.6495 | 0.04590 | 0.01373 | 0.00923 | 1.050 |
| OMS `0.7`, TM `1.5`, tilt `-0.15` | 0.6033 | 0.04570 | 0.01379 | 0.00924 | 1.055 |

The residual surface absorbs these broad baseline errors without materially
changing the recovered total PSD. The second run narrowly misses the R-hat
gate, and the stationary comparators in these short fits are less stable.
Therefore this is evidence of directional point-estimate robustness, not proof
that arbitrary reference errors are harmless.

### Response-null mask sensitivity

The locked fit was rescored under nine masks formed from response-ratio
thresholds 0.20, 0.35, and 0.50 and frequency dilations of 3, 6, and 9 bins.
Across this grid:

- full-band nested log-RMSE remains 0.01370--0.01378;
- high-continuum nested log-RMSE remains 0.00923--0.00924;
- the excluded full-band cell fraction ranges from 2.62% to 4.72%.

The reported accuracy is therefore insensitive to the precise notch boundary
over this grid. The mask is a diagnostic boundary, not a tuned result selector.

### WDM projection validity

The installed Meyer frequency kernel is compactly supported within
`|f-f_m| <= 2 df/3`. The measured power outside this support is `3.3e-23`, so
there is no long-range frequency leakage path from a moving response null into
distant continuum cells, regardless of dynamic range.

For the present `nt=2048` configuration, the remaining local curvature error is
estimated at approximately 0.2%, below the present statistical and model
errors. The projection check should be reopened only for substantially larger
`nt`, the lowest few frequency channels, features narrower than `df`, or a
different Meyer-window parameter.

## What the present study supports

The present realization supports the following claims:

1. A response-informed P-spline residual can recover the total time-varying X2
   PSD over the retained `1.3e-4`--`1e-1 Hz` WDM band.
2. A separately coarse interaction basis materially improves prediction below
   3 mHz relative to a matched stationary residual.
3. The interaction gives no meaningful gain in the high-frequency continuum,
   providing a useful negative control against gratuitous flexibility.
4. One excluded WDM pixel on each side of a time-domain gap is supported by
   blind whitening immediately outside the gap.
5. Smooth interpolation remains accurate in the tested simulated gaps, with
   modest degradation for a thirty-day gap.
6. The fitted total PSD is robust at the point-estimate level to the tested
   interaction priors, broad OMS/TM reference perturbations, and response-null
   mask definitions.

## What the present study does not support

The following claims would overstate the evidence:

- calibrated nominal 90% coverage across repeated realizations;
- a Bayes factor for stationarity versus time variation;
- separate recovery of instrumental noise and the Galactic PSD (H_para);
- performance across arbitrary LISA channels or TDI combinations;
- generalization to external LDC data;
- robustness to narrow or structurally incorrect reference features outside
  the tested perturbation family;
- direct goodness-of-fit validation inside a real data gap, where the truth is
  unobserved.

NUTS posterior draws do not provide a marginal likelihood. A Bayes-factor claim
would require a separate evidence calculation with normalized priors and
identical likelihood/mask contracts.

## Recommended manuscript use

For LISA Part A, the primary result should compare `R`, `R exp(g)`, and
`R exp(g+h)` on continuous data. The continuous figure above can carry the main
scientific argument: recovered modulation, frequency-resolved accuracy, and
incremental held-out predictive score. Report the compact whitening numbers in
the text and move distributional whitening plots to an appendix.

For LISA Part B, use the seven-day, one-pixel-buffer fit as the production
anchor. The gap figure should explain interpolation, the blind buffer decision,
and the duration stress test. Clearly label gap-interior RMSE as simulation
truth only.

Do not promote the short prior/reference or one-/thirty-day sensitivity chains
to posterior results. They belong in a compact robustness subsection or
appendix table. Do not describe truth inclusion on this one realization as
empirical coverage.

## Required next steps

1. Freeze the current split, five-knot interaction, one-pixel buffer, likelihood
   binning, and response-mask rule.
2. Generate multiple independent ESA-orbit noise/Galactic realizations and
   measure run-to-run RMSE, bias, and empirical coverage.
3. Apply the frozen method to one external LDC-style dataset using only released
   TDI data, timestamps/gaps, orbit information, and public nominal noise
   curves.
4. Consider A/E/T replication only if the manuscript expands beyond the X2 surface
   demonstration.
5. Treat H_para component separation as a separate identifiable model, not as two
   unrestricted positive P-spline surfaces.

## Further questions

- Does the low-band interval undercoverage persist across independent
  realizations, and is it caused by the diagonal likelihood, spline bias, or
  posterior interval construction?
- How does performance change for several gaps with realistic duration and
  cadence distributions rather than one centered gap?
- Can a normalized-prior evidence calculation reliably quantify support for
  `R exp(g+h)` over `R exp(g)` without prohibitive cost?
- How stable is the interaction basis choice across channels and independently
  generated Galactic realizations?

## Reproducibility and artifacts

The main implementation and record are:

- `wdm_psd/tv_pspline_psd/model.py`: nested residual model;
- `wdm_psd/tv_pspline_psd/inference.py`: independent interaction basis and
  posterior reconstruction;
- `lisa_data_generation/esa_m0_study.py`: continuous/gap runner, blind scores,
  truth metrics, and summary-only sensitivity mode;
- `lisa_data_generation/plot_esa_method_improvements.py`: publication figures
  and consolidated summary;
- `lisa_data_generation/publication_results/esa_x2_publication_summary.json`:
  machine-readable selected results;
- `lisa_data_generation/PUBLICATION_PROTOCOL.md`: frozen analysis
  protocol and claim boundary;
- `lisa_data_generation/METHOD_IMPROVEMENTS.md`: implementation and
  method-tuning audit;
- `wdm_psd/notes/WDM_PROJECTION_VALIDITY.md`: compact-support and curvature
  validation.

Complete posterior surfaces are retained for the selected continuous fit and
the seven-day gap anchor. Lower-budget sensitivity runs retain their JSON audit
metrics without duplicating large posterior-surface archives.
