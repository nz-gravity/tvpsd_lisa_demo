"""Does the archive's SIMULATED noise match the ANALYTIC noise model?

Everything in this study validates against analytic OMS/TM, and the H_para prior
is centred on it. But the archive's noise came from a simulator. If the two
differ, the fit follows the data (correctly) and every residual against the
analytic 'truth' is offset -- with the mismatch pushed into whichever free
component can absorb it.

Runs the noise-only series through the identical WDM pipeline as the pilot,
using the now-exact analytic calibration (C_m = N), and compares to the
analytic model evaluated directly on the same grid.
"""

import sys

import h5py
import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from run_component_study import (
    aet_noise_transfer_functions,
    bin_channels,
    oms_theory_psd,
    tm_theory_psd,
    wdm_valid_length,
)
from tv_pspline_psd.lisa_aet import AET_CHANNELS, xyz_to_aet_series
from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
from tv_pspline_psd.inference import adaptive_frequency_bin_starts

HERE = "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation"
ARCHIVE = f"{HERE}/combined_esa_xyz.h5"
NT, FMAX = 32, 0.02

with h5py.File(ARCHIVE, "r") as hdf:
    dt = float(hdf.attrs["dt_seconds"])
    t0 = float(hdf.attrs["t0_tcb"])
    n_archive = int(hdf.attrs["n_samples"])
    n_total = wdm_valid_length(n_archive, NT)
    print("archive tdi datasets:", list(hdf["tdi"].keys()))
    xyz_noise = hdf["tdi/noise"][:, :n_total]

aet_noise = xyz_to_aet_series(xyz_noise)
del xyz_noise

nf = n_total // NT
df = 1.0 / (2.0 * nf * dt)
config = PSplineConfig(
    n_interior_knots_time=3,
    n_interior_knots_freq=40,
    trim_time_bins=1,
    trim_low_freq_channels=max(1, int(np.ceil(1.0e-4 / df))),
    trim_high_freq_channels=max(0, nf - int(np.floor(FMAX / df))),
    freq_knot_strategy="log",
    centered=True,
)

coefficients, fit_time_grid, fit_frequency = [], None, None
for series in aet_noise:
    c, t, f = wdm_analysis_coefficients(series, dt, NT, config)
    coefficients.append(c)
    fit_time_grid, fit_frequency = t, f
del aet_noise
coefficients = np.stack(coefficients)
band = fit_frequency <= FMAX
coefficients = coefficients[:, :, band]
fit_frequency = fit_frequency[band]
absolute_time = t0 + np.asarray(fit_time_grid) * n_total * dt

# Exact calibration: E[w^2] = N S / (2 dt).
observed = coefficients**2 * (2.0 * dt / float(n_total))
del coefficients

transfer_tm, transfer_oms = aet_noise_transfer_functions(
    f"{HERE}/noise2a/orbits.h5", absolute_time, fit_frequency
)
analytic = (
    transfer_tm * tm_theory_psd(fit_frequency)[None, None, :]
    + transfer_oms * oms_theory_psd(fit_frequency)[None, None, :]
)

pilot = np.concatenate([np.log(analytic[c]) for c in range(3)], axis=0)
bin_starts = adaptive_frequency_bin_starts(pilot, max_log_range=0.15, max_bin=32)
observed_binned, counts, grouped = bin_channels(observed, fit_frequency, bin_starts)
analytic_binned, _, _ = bin_channels(analytic, fit_frequency, bin_starts)

print()
print("SIMULATED noise / ANALYTIC noise model  (1.000 = they agree)")
print("counts-weighted; Whittle-unbiased, so departures are model error\n")
print(f"{'band':>18} {'A':>9} {'E':>9} {'T':>9}")
edges = [(1e-4, 3e-4), (3e-4, 5e-4), (5e-4, 1e-3), (1e-3, 2e-3),
         (2e-3, 3e-3), (3e-3, 8e-3), (8e-3, 2e-2)]
for a, b in edges:
    s = (grouped >= a) & (grouped < b)
    if not s.any():
        continue
    row = []
    for c in range(3):
        o, m, w = observed_binned[c][:, s], analytic_binned[c][:, s], counts[c][:, s]
        row.append(np.sum(w * o / m) / np.sum(w))
    print(f"{a*1e3:7.2f}-{b*1e3:6.2f} mHz {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f}")

print()
overall = [
    np.sum(counts[c] * observed_binned[c] / analytic_binned[c]) / np.sum(counts[c])
    for c in range(3)
]
print(f"  overall: A {overall[0]:.4f}  E {overall[1]:.4f}  T {overall[2]:.4f}")
print(f"  total effective counts per channel: {counts[0].sum():.3g} "
      f"-> statistical error on these ratios ~{1/np.sqrt(counts[0].sum()):.5f}")
