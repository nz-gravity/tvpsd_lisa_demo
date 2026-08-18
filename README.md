# LISA ESA-orbit PSD analyses

Analysis code for the ESA-orbit LISA study. The estimator itself -- including
the AET rotation algebra (`tv_pspline_psd.lisa_aet`) and the multichannel
physical-component model (`tv_pspline_psd.multichannel`) -- lives in the
sibling package `../wdm_psd/tv_pspline_psd`. This directory holds only what is
specific to this archive and this study: data generation, the hypothesis
runners, OzSTAR job scripts, protocol documents, and diagnostics.

## The three-rung comparison

The manuscript compares three models on the **same data, grid, cells and
splits**, in order of how much physics they assume:

| rung | hypothesis | model | time dependence of the noise | assumes |
|---|---|---|---|---|
| 1 | **H_agn** (agnostic) | free TV surface | free P-spline | nothing |
| 2 | **H_orb** (orbit/transfer aware) | `log S = log R(t,f) + g(f) + h(t,f)` | **known**, from `L(t)` via TDI transfer | orbits, TDI convention, transfer geometry |
| 3 | **H_para** (parametric components) | `S = T_TM S_TM + T_OMS S_OMS + A_gal T_gal` | same | + OMS/TM spectral forms, Galactic template |

H_agn and H_orb are both `run_surface_study.py`, selected with
`--hypothesis {agn,orb}`; the resolved label is recorded in each run's metadata
so the plotting and diagnostic scripts read it rather than inferring it from
which flags were passed. H_para is `run_component_study.py --component-noise`.

The point of the ladder is not that more physics gives smaller error bars —
it will — but to show **where the narrowing stops being calibrated**. Report
interval width, coverage, and bias at every rung.

## Entry points

```bash
# H_agn (rung 1), X2 channel
python run_surface_study.py --hypothesis agn --mode continuous

# H_orb (rung 2), X2 channel, continuous and gapped
python run_surface_study.py --hypothesis orb --mode continuous
python run_surface_study.py --hypothesis orb --mode gapped --single-gap-days 7 --gap-buffer-pixels 1

# H_para (rung 3), A/E/T jointly
python run_component_study.py --component-noise --warmup 500 --samples 500
```

H_para runs on the surface study's grid and protocol: `nt=2048`, ~512 time bins, its
`training_data_pilot_log_psd` for truth-free adaptive frequency bins, and the
same seven-fold split (validation fold 5, test fold 6) excluded from both the
likelihood and the bin pilot.

## Protocol and results

- `PUBLICATION_PROTOCOL.md` — analysis protocol and claim boundary
- `RESULTS_AND_FINDINGS.md` — superseded results, pending the post-CSD-fix rerun
- `COARSE_REFERENCE_OFFSET_CORRECTION.md` — likelihood correction and rerun plan
- `METHOD_IMPROVEMENTS.md` — method-tuning audit
- `diagnostics/README.md` — every contested design choice, with the number that settled it

## Data (not tracked)

Inputs and outputs total ~12 GB and are excluded by `.gitignore`:

- `combined_esa_xyz.h5` (1.1 GB) — the archived X2/Y2/Z2 realization
- `noise2a/` — instrumental noise `tdi.h5` and ESA `orbits.h5`
- `gb1/` — galactic binary inputs
- `esa_m0_*/`, `*.npz` — fit outputs and posterior surfaces

`publication_results/esa_x2_publication_summary.json` **is** tracked: it
is the machine-readable record of the selected results. Those numbers are
**superseded** — they predate the XYZ cross-spectrum fix in the archive
(see below) and are kept only as a record of what the earlier run produced.

## Known issues

- ~~**Held-out folds are excluded but not yet scored.**~~ Resolved:
  `heldout_binned_diagnostics` scores the validation and test cohorts on the
  fit's own bins, in the surface study's three bands, with the gain taken against the analytic
  OMS+TM reference. `mean_z` and the lag-one products are not reported, since
  time pooling discards coefficient signs; `central_90_fraction` uses
  `chi^2_nu` quantiles rather than a fixed normal cut.
- **Coverage.** Surface-study low-band truth inclusion is 0.707 (0.886 full band); H_para is
  ~0.65-0.92 depending on channel. Neither is repeated-realization coverage.
  Whether this is the diagonal likelihood, spline bias, or interval
  construction is open, and it is the same question for both models.
- **Archive rebuilt (XYZ cross spectra).** The Galactic foreground was
  previously drawn independently per XYZ channel from auto-PSDs alone, which
  destroyed the TDI cross spectra: corr(X,Y) was -0.05 instead of -0.56, and
  the Galaxy sat in the A/E/T `T` channel at 0.86 of `A` instead of ~1e-4 of
  it. The synthesis now draws through the Cholesky factor of the full response
  covariance and the archive stores `truth/*_csd`. **All fit results predating
  this rebuild are invalid and must be rerun.** Note that `T` is only an
  *approximate* null: the ESA orbits have unequal, breathing arms, so both the
  Galaxy and the instrument noise leak into `T` at the ~1e-4 (power) level. The
  Galaxy still supplies ~40% of `T`-channel power (up to 87%) over 0.3-10 mHz,
  so it must **not** be approximated as zero there.
- **T channel.** The simulated T noise exceeds the analytic model by up to 20x
  in the counts-weighted mean below 0.3 mHz (A and E agree to <3%). Cause not
  established; equal-arm geometry, residual laser noise, WDM projection and
  interpolation artifacts are all ruled out. Measured at `nt=32`; may be much
  smaller at `nt=2048`.
