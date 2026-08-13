# LISA ESA-orbit PSD analyses

Analysis code for the ESA-orbit LISA study. The estimator itself -- including
the AET rotation algebra (`tv_pspline_psd.lisa_aet`) and the multichannel
physical-component model (`tv_pspline_psd.multichannel`) -- lives in the
sibling package `../wdm_psd/tv_pspline_psd`. This directory holds only what is
specific to this archive and this study: data generation, the M0/M1 study
runners, OzSTAR job scripts, protocol documents, and diagnostics.

## The three-rung comparison

The manuscript compares three models on the **same data, grid, cells and
splits**, in order of how much physics they assume:

| rung | model | time dependence of the noise | assumes |
|---|---|---|---|
| 1. agnostic | free TV surface | free P-spline | nothing |
| 2. transfer-function aware | `log S = log R(t,f) + g(f) + h(t,f)` | **known**, from `L(t)` via TDI transfer | orbits, TDI convention, transfer geometry |
| 3. component separation | `S = T_TM S_TM + T_OMS S_OMS + A_gal T_gal` | same | + OMS/TM spectral forms, Galactic template |

Rungs 1 and 2 are M0 (`esa_m0_study.py`, models 4 and 3 of its hierarchy).
Rung 3 is M1 (`run_aet_diagonal_pilot.py --component-noise`).

The point of the ladder is not that more physics gives smaller error bars —
it will — but to show **where the narrowing stops being calibrated**. Report
interval width, coverage, and bias at every rung.

## Entry points

```bash
# M0 (rungs 1-2), X2 channel
python esa_m0_study.py --mode continuous
python esa_m0_study.py --mode gapped --single-gap-days 7 --gap-buffer-pixels 1

# M1 (rung 3), A/E/T jointly
python run_aet_diagonal_pilot.py --warmup 500 --samples 500
```

M1 runs on M0's grid and protocol: `nt=2048`, ~512 time bins, M0's
`training_data_pilot_log_psd` for truth-free adaptive frequency bins, and the
same seven-fold split (validation fold 5, test fold 6) excluded from both the
likelihood and the bin pilot.

## Protocol and results

- `ESA_M0_PUBLICATION_PROTOCOL.md` — analysis protocol and claim boundary
- `ESA_M0_RESULTS_AND_FINDINGS.md` — superseded pre-fix M0 results pending rerun
- `COARSE_REFERENCE_OFFSET_CORRECTION.md` — likelihood correction and rerun plan
- `ESA_M0_METHOD_IMPROVEMENTS.md` — method-tuning audit
- `diagnostics/README.md` — every contested design choice, with the number that settled it

## Data (not tracked)

Inputs and outputs total ~12 GB and are excluded by `.gitignore`:

- `combined_esa_xyz.h5` (1.1 GB) — the archived X2/Y2/Z2 realization
- `noise2a/` — instrumental noise `tdi.h5` and ESA `orbits.h5`
- `gb1/` — galactic binary inputs
- `esa_m0_*/`, `*.npz` — fit outputs and posterior surfaces

`esa_m0_publication_results/esa_x2_publication_summary.json` **is** tracked: it
is the machine-readable record of the selected results.

## Known issues

- **Held-out folds are excluded but not yet scored.** The split removes
  validation/test rows from the M1 likelihood, but no held-out Whittle score is
  computed, so M1's reported numbers are still in-sample. M0 does score its
  test cells.
- **Coverage.** M0 low-band truth inclusion is 0.707 (0.886 full band); M1 is
  ~0.65-0.92 depending on channel. Neither is repeated-realization coverage.
  Whether this is the diagonal likelihood, spline bias, or interval
  construction is open, and it is the same question for both models.
- **T channel.** The simulated T noise exceeds the analytic model by up to 20x
  in the counts-weighted mean below 0.3 mHz (A and E agree to <3%). Cause not
  established; equal-arm geometry, residual laser noise, WDM projection and
  interpolation artifacts are all ruled out. Measured at `nt=32`; may be much
  smaller at `nt=2048`.
