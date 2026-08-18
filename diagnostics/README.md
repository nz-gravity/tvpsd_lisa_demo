# Diagnostics

One-off measurement scripts. Each exists because a design choice was contested
and needed a number rather than an argument. They are records of *what was
established*, not part of any fit.

**They read fitted `.npz` files by absolute path under `/tmp`.** Those paths are
stale. To re-run one, repoint it at a current output. The findings below are
what they produced when run, and are the evidence behind
`METHOD_IMPROVEMENTS.md`/`RESULTS_AND_FINDINGS.md`/`aet_component_pspline_nuts.py` docstrings and the project notes.

## WDM representation

| script | established |
|---|---|
| `reference_null_smearing.py` | Interpolating the analytic reference from the archive's 256-point frequency grid loses **7.5 nats** of TDI-null depth (16.6 → 9.0). Fixed by evaluating transfer functions directly on the analysis grid. |
| `adaptive_bin_count.py` | An unsmoothed data-driven bin pilot never merges: 39,125 bins from 39,125 channels, a 32x larger inference problem (one fit ran 13 h instead of ~20 min). Frequency smoothing restores ~1,250 bins. |

## Sampler geometry

| script | established |
|---|---|
| `para_surface_conditioning.py` | The tensor-spline model whitens the *prior*, but with millions of counts the *likelihood* dominates: condition 5.0e3 → 1.9e1 when whitened correctly. Whittle Fisher is `counts/2` per cell and parameter-independent, so the ideal preconditioner is a known constant matrix. |
| `para_component_conditioning.py` | Same disease in the physical-component model, with a twist: TM is prior-pinned (likelihood/prior = 1.7e-4) while OMS is data-dominated (62x). Prior-only whitening is right for one block and wrong for the other → condition 1.06e4, ~103 leapfrog steps. Fixed: 746 s → 203 s, ESS 485 → 945. |
| `agn_vs_para_geometry.py` | H_agn vs H_para on identical data and machinery, isolating what costs leapfrog steps. |
| `para_noise_parameterisations.py` | Compared noise parameterisations (tensor spline / stationary+drift / offset). Recorded because two of the three were *worse* — the intuitive fixes failed. |

## Identifiability (the core H_para difficulty)

| script | established |
|---|---|
| `para_amplitude_redundancy.py` | Free OMS/TM amplitudes are reproduced by the frequency spline to **R² = 1.00000000** (1 part in 1e11) — ridges ~1e5x longer than wide, *between* parameter blocks, invisible to any per-block preconditioner. Hence they are not sampled. |
| `galaxy_identifiability.py` | A free noise spline absorbs **99.8%** of the Galactic amplitude direction at 7 time basis functions; static-in-time leaves 15% → **67x** more identifying information. Time rank is the lever. |
| `identifiability_map.py` | Per channel and band: component fractions and annual modulation depth. Galaxy modulates 1.47 nats vs instrument noise 0.065 → **23x contrast**, the identifier a stationary analysis lacks. Also shows T is ~100% Galaxy below 3 mHz *in the analytic model*, which is the trap that biased `A_gal`. |
| `phi_prior_width_dex.py` | `phi` is not scale-free: `phi=1e8` permits a 0.145 dex (±18%) 90% prior width at 1 mHz; `phi=1e4` permits 14.5 dex (unconstrained). Explains why varying `phi_TM` over four decades did nothing. Convert to dex before quoting. |

## Noise model vs simulation

| script | established |
|---|---|
| `transfer_function_factorisation.py` | `S_noise,c(t,f) = T_TM,c S_TM(f) + T_OMS,c S_OMS(f)` holds to **4.4e-16** with *stationary* component spectra — all time dependence, including the null drifting 59.83 → 60.94 mHz, lives in the orbit-derived transfer functions. This is why the noise needs no time-varying free parameters. |
| `simulated_vs_analytic_noise.py` | Simulated vs analytic noise per band. A and E agree to <3%; T disagrees by up to 20x in the counts-weighted mean below 0.3 mHz. Measured against the *directly evaluated* reference, so not an interpolation artifact. |
| `t_channel_residual_shape.py` | The T excess is ~23%/14%/1.5% in the **median** across the low bands — the 20x mean is dominated by a minority of deep-null cells. Both statistics matter: the median says the model is broadly fine, the mean says the likelihood is dominated by pathological cells. |
| `para_scalar_level_pull.py` | A single scalar noise level is underdetermined by ~2 orders of magnitude against the true dynamic range, forcing the spline >20 prior sigma from its mode at any usable roughness penalty. |

## Calibration

| script | established |
|---|---|
| `para_bias_vs_width_split.py` | Separates *bias* from *interval width* in the coverage deficit. Found 64% of the miss was systematic — which is why Safe-Bayes tempering was **rejected**: it would have widened intervals around a wrong centre. |

## Superseded

`../wdm_psd/notes/WDM_PROJECTION_VALIDITY.md` settled that the WDM projection
`E[w_nm^2] = S(f_m)` is **exact** on these grids (compact kernel support,
out-of-support power 3.3e-23; predicted bias 6e-5 against a required 1900%).
An earlier hypothesis blaming the T-channel excess on projection leakage is
therefore wrong and should not be reopened.
