"""What shape is the T-channel excess? Determines the right model for it.

T is a near-null: it sits ~7000x below A, so a sub-percent error in the
cancelling combination becomes an order-of-magnitude error in T. Whatever is
left over has to be modelled, not discarded, if T is to anchor SGWB detection.

Tests whether  T_sim - T_analytic  is better described as
  (a) a fixed fraction of the A-channel noise   -> imperfect common-mode nulling
  (b) OMS-shaped                                -> OMS leakage
  (c) TM-shaped                                 -> TM leakage
  (d) a flat floor
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
from tv_pspline_psd.lisa_aet import xyz_to_aet_series
from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients

HERE = "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation"
NT, FMAX = 32, 0.02

with h5py.File(f"{HERE}/combined_esa_xyz.h5", "r") as hdf:
    dt = float(hdf.attrs["dt_seconds"])
    t0 = float(hdf.attrs["t0_tcb"])
    n_archive = int(hdf.attrs["n_samples"])
    n_total = wdm_valid_length(n_archive, NT)
    xyz = hdf["tdi/noise"][:, :n_total]

aet = xyz_to_aet_series(xyz)
del xyz
nf = n_total // NT
df = 1.0 / (2.0 * nf * dt)
config = PSplineConfig(
    n_interior_knots_time=3, n_interior_knots_freq=40, trim_time_bins=1,
    trim_low_freq_channels=max(1, int(np.ceil(1.0e-4 / df))),
    trim_high_freq_channels=max(0, nf - int(np.floor(FMAX / df))),
    freq_knot_strategy="log", centered=True,
)
coefficients, grid_t, grid_f = [], None, None
for series in aet:
    c, t, f = wdm_analysis_coefficients(series, dt, NT, config)
    coefficients.append(c); grid_t, grid_f = t, f
del aet
coefficients = np.stack(coefficients)
band = grid_f <= FMAX
observed = coefficients[:, :, band] ** 2 * (2.0 * dt / float(n_total))
del coefficients
frequency = grid_f[band]
time = t0 + np.asarray(grid_t) * n_total * dt

transfer_tm, transfer_oms = aet_noise_transfer_functions(
    f"{HERE}/noise2a/orbits.h5", time, frequency
)
tm_part = transfer_tm * tm_theory_psd(frequency)[None, None, :]
oms_part = transfer_oms * oms_theory_psd(frequency)[None, None, :]
analytic = tm_part + oms_part

# Pool in frequency so the chi-square scatter does not swamp the comparison.
starts = np.arange(0, frequency.size, 64, dtype=int)
obs_b, counts, freq_b = bin_channels(observed, frequency, starts)
ana_b, _, _ = bin_channels(analytic, frequency, starts)
tm_b, _, _ = bin_channels(tm_part, frequency, starts)
oms_b, _, _ = bin_channels(oms_part, frequency, starts)

T = 2
excess = obs_b[T] - ana_b[T]          # what T has that the model lacks
a_noise = ana_b[0]                    # A-channel analytic noise

print("T-channel excess: simulated minus analytic\n")
print(f"{'band':>16} {'excess/T_ana':>13} {'excess/A_ana':>13} {'excess/OMS_T':>13} {'excess/TM_T':>12}")
for lo, hi in [(1e-4,3e-4),(3e-4,1e-3),(1e-3,3e-3),(3e-3,8e-3),(8e-3,2e-2)]:
    s = (freq_b >= lo) & (freq_b < hi)
    if not s.any():
        continue
    e = np.median(excess[:, s])
    print(f"{lo*1e3:6.2f}-{hi*1e3:6.2f} mHz "
          f"{e/np.median(ana_b[T][:, s]):13.4f} "
          f"{e/np.median(a_noise[:, s]):13.3e} "
          f"{e/np.median(oms_b[T][:, s]):13.4f} "
          f"{e/np.median(tm_b[T][:, s]):12.4f}")

print()
print("If 'excess/A_ana' is roughly CONSTANT, the excess is a fixed fraction of")
print("the A-channel noise, i.e. imperfect common-mode nulling -- and the right")
print("model is S_T += epsilon * S_A,noise with epsilon small and slowly varying.")
