"""Cheap offline diagnostic (no MCMC): how much dynamic range does the free
spline have to carry, at fixed phi=100, when the noise reference is (a) one
scalar per channel vs (b) a 2-parameter OMS/TM amplitude fit?

If the required eigen-coefficient magnitude is many multiples of the prior
scale implied by phi=100, the likelihood must drag the posterior far from the
prior mode through the exp() nonlinearity before it can even start exploring
locally -- exactly the kind of large-gradient regime that forces long NUTS
trajectories, independent of any frequency-translating feature.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from backgrounds import noise as background_noise, tdi
from scipy.interpolate import RegularGridInterpolator

HERE = Path("/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "wdm_psd"))

from run_aet_diagonal_pilot import (
    SECONDS_PER_YEAR,
    CARRIER_FREQUENCY_HZ,
    load_esa_orbits,
    interpolate_surface,
    bin_channels,
    wdm_valid_length,
    armlength_ratio,
)
from tv_pspline_psd.lisa_aet import AET_CHANNELS, xyz_covariance_to_aet_diagonal
from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
from tv_pspline_psd.inference import adaptive_frequency_bin_starts
from tv_pspline_psd.model import whiten_penalty_pair, eigen_prior_scale
from tv_pspline_psd.splines import create_bspline_basis, create_difference_penalty_matrix

ARCHIVE = HERE / "combined_esa_xyz.h5"
NT = 32
FMAX_HZ = 0.075
N_TIME_KNOTS, N_FREQUENCY_KNOTS = 3, 40
PHI = 100.0

with h5py.File(ARCHIVE, "r") as hdf:
    dt = float(hdf.attrs["dt_seconds"])
    t0_tcb = float(hdf.attrs["t0_tcb"])
    n_archive = int(hdf.attrs["n_samples"])
    truth_time = hdf["truth/time_tcb"][:]
    truth_frequency = hdf["truth/frequency_hz"][:]

n_total = wdm_valid_length(n_archive, NT)
nf = n_total // NT
df = 1.0 / (2.0 * nf * dt)
config = PSplineConfig(
    n_interior_knots_time=N_TIME_KNOTS,
    n_interior_knots_freq=N_FREQUENCY_KNOTS,
    trim_time_bins=1,
    trim_low_freq_channels=max(1, int(np.ceil(1.0e-4 / df))),
    trim_high_freq_channels=max(0, nf - int(np.floor(FMAX_HZ / df))),
    freq_knot_strategy="log",
    centered=True,
)
# Grid only -- cheap, no coefficient computation needed for this diagnostic.
dummy = np.zeros(n_total)
_, fit_time_grid, fit_frequency = wdm_analysis_coefficients(dummy, dt, NT, config)
band = fit_frequency <= FMAX_HZ
fit_frequency = fit_frequency[band]
absolute_time = t0_tcb + np.asarray(fit_time_grid) * n_total * dt

orbits = load_esa_orbits(HERE / "noise2a" / "orbits.h5")
oms = background_noise.AnalyticOMSNoiseModel(
    truth_frequency, truth_time, orbits, tdi_tf_func=tdi.compute_tdi_tf,
    gen="2.0", oms_isi_carrier_asds=7.9e-12, fs=0.5, duration=SECONDS_PER_YEAR,
)
test_mass = background_noise.AnalyticTMNoiseModel(
    truth_frequency, truth_time, orbits, tdi_tf_func=tdi.compute_tdi_tf_tm,
    gen="2.0", tm_isi_carrier_asds=2.4e-15, fs=0.5, duration=SECONDS_PER_YEAR,
)
oms_cov = oms.compute_covariances(0.0)
tm_cov = test_mass.compute_covariances(0.0)


def to_aet_time_freq(covariance):
    diagonal = xyz_covariance_to_aet_diagonal(covariance)
    return np.moveaxis(diagonal, (0, 1, 2), (2, 1, 0)) * CARRIER_FREQUENCY_HZ**2


oms_aet = to_aet_time_freq(oms_cov)
tm_aet = to_aet_time_freq(tm_cov)
total_aet = oms_aet + tm_aet


def interp(source):
    return interpolate_surface(
        source, truth_time, truth_frequency, absolute_time, fit_frequency
    )


oms_unbinned = interp(oms_aet)
tm_unbinned = interp(tm_aet)
total_unbinned = interp(total_aet)

pilot_log_psd = np.concatenate(
    [np.log(total_unbinned[c]) for c in range(len(AET_CHANNELS))], axis=0
)
bin_starts = adaptive_frequency_bin_starts(pilot_log_psd, max_log_range=0.15, max_bin=32)
oms_binned, _, grouped_frequency = bin_channels(oms_unbinned, fit_frequency, bin_starts)
tm_binned, _, _ = bin_channels(tm_unbinned, fit_frequency, bin_starts)
total_binned, counts, _ = bin_channels(total_unbinned, fit_frequency, bin_starts)
fit_valid = counts > 0.0

print(f"grid: {absolute_time.size} time x {grouped_frequency.size} freq bins "
      f"(from {fit_frequency.size} WDM channels, adaptive)\n")

# --- Build the exact whitened tensor basis the model uses ---
time_unit = (absolute_time - absolute_time[0]) / (absolute_time[-1] - absolute_time[0])
log_frequency = np.log(grouped_frequency)
frequency_unit = (log_frequency - log_frequency[0]) / (log_frequency[-1] - log_frequency[0])
basis_time, _ = create_bspline_basis(time_unit, N_TIME_KNOTS, degree=3)
basis_frequency, _ = create_bspline_basis(frequency_unit, N_FREQUENCY_KNOTS, degree=3)
penalty_time = create_difference_penalty_matrix(basis_time.shape[1], diff_order=2)
penalty_frequency = create_difference_penalty_matrix(basis_frequency.shape[1], diff_order=2)
whitened = whiten_penalty_pair(penalty_time, penalty_frequency)
basis_eig_time = basis_time @ whitened["U_time"]
basis_eig_frequency = basis_frequency @ whitened["U_freq"]
prior_scale = np.asarray(
    eigen_prior_scale(PHI, PHI, whitened["lam_time"], whitened["lam_freq"], whitened["joint_null"], config)
)
design = np.einsum("ti,fj->tfij", basis_eig_time, basis_eig_frequency, optimize=True).reshape(
    absolute_time.size * grouped_frequency.size, -1
)


def penalized_fit(log_target, weight):
    """Ridge-penalized least squares with the model's actual per-coefficient
    prior precision -- i.e. exactly the MAP the spline prior pulls toward."""
    selected = weight.ravel() > 0
    y = log_target.ravel()[selected]
    X = design[selected]
    precision = 1.0 / prior_scale.ravel() ** 2
    coefficients = np.linalg.solve(
        X.T @ X + np.diag(precision), X.T @ y
    )
    residual = y - X @ coefficients
    coeff_in_prior_sigmas = np.abs(coefficients) / prior_scale.ravel()
    return coefficients, residual, coeff_in_prior_sigmas


print("=== A: single scalar reference per channel (current M1) ===")
for c, name in enumerate(AET_CHANNELS):
    level = float(np.exp(np.median(np.log(total_binned[c][fit_valid[c]]))))
    log_target = np.log(total_binned[c] / level)
    coeffs, residual, sigmas = penalized_fit(log_target, fit_valid[c])
    print(
        f"  {name}: log-target range [{log_target[fit_valid[c]].min():.2f}, "
        f"{log_target[fit_valid[c]].max():.2f}]  "
        f"post-fit residual RMS {np.sqrt(np.mean(residual**2)):.3f}  "
        f"max |coeff|/prior_sigma {sigmas.max():.1f}x  "
        f"(>3x means the prior is fighting the data at phi={PHI:.0f})"
    )

print("\n=== B: 2-parameter OMS/TM amplitude reference per channel ===")
for c, name in enumerate(AET_CHANNELS):
    valid = fit_valid[c]
    y = total_binned[c][valid]
    A = np.stack([oms_binned[c][valid], tm_binned[c][valid]], axis=1)
    amplitudes, *_ = np.linalg.lstsq(A, y, rcond=None)
    amplitudes = np.maximum(amplitudes, 1.0e-3 * amplitudes.max())
    reference = amplitudes[0] * oms_binned[c] + amplitudes[1] * tm_binned[c]
    log_target = np.log(total_binned[c] / reference)
    coeffs, residual, sigmas = penalized_fit(log_target, valid)
    print(
        f"  {name}: fitted (a_oms,a_tm)=({amplitudes[0]:.3f},{amplitudes[1]:.3f})  "
        f"log-target range [{log_target[valid].min():.2f}, {log_target[valid].max():.2f}]  "
        f"post-fit residual RMS {np.sqrt(np.mean(residual**2)):.3f}  "
        f"max |coeff|/prior_sigma {sigmas.max():.1f}x"
    )
