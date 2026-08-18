import sys, time
import numpy as np, h5py

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")
from run_component_study import (
    wdm_valid_length, time_block_log_pilot, analytic_aet_noise_components_psd,
    interpolate_surface, aet_noise_transfer_functions, oms_theory_psd, tm_theory_psd,
)
from tv_pspline_psd.lisa_aet import xyz_to_aet_series, AET_CHANNELS
from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
from tv_pspline_psd.inference import adaptive_frequency_bin_starts

HERE = "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation"
NT, FMAX = 32, 0.02
with h5py.File(f"{HERE}/combined_esa_xyz.h5", "r") as hdf:
    dt = float(hdf.attrs["dt_seconds"]); t0 = float(hdf.attrs["t0_tcb"])
    n_archive = int(hdf.attrs["n_samples"])
    n_total = wdm_valid_length(n_archive, NT)
    xyz = hdf["tdi/total"][:, :n_total]
    injected_amplitude = float(hdf.attrs["galactic_amplitude_scale"])

aet = xyz_to_aet_series(xyz); del xyz
nf = n_total // NT
df = 1.0 / (2.0 * nf * dt)
config = PSplineConfig(n_interior_knots_time=3, n_interior_knots_freq=40, trim_time_bins=1,
    trim_low_freq_channels=max(1, int(np.ceil(1e-4/df))),
    trim_high_freq_channels=max(0, nf - int(np.floor(FMAX/df))),
    freq_knot_strategy="log", centered=True)

coefficients, grid_t, grid_f = [], None, None
for series in aet:
    c, t, f = wdm_analysis_coefficients(series, dt, NT, config)
    coefficients.append(c); grid_t, grid_f = t, f
del aet
coefficients = np.stack(coefficients)
band = grid_f <= FMAX
observed = coefficients[:, :, band]**2 * (2.0*dt/float(n_total))
del coefficients
frequency = grid_f[band]
wdm_time = t0 + np.asarray(grid_t)*n_total*dt

transfer_tm, transfer_oms = aet_noise_transfer_functions(f"{HERE}/noise2a/orbits.h5", wdm_time, frequency)
analytic = transfer_tm*tm_theory_psd(frequency)[None,None,:] + transfer_oms*oms_theory_psd(frequency)[None,None,:]

print("observed shape:", observed.shape)

t0_ = time.time()
pilot_analytic = np.concatenate([np.log(analytic[c]) for c in range(3)], axis=0)
starts_a = adaptive_frequency_bin_starts(pilot_analytic, max_log_range=0.15, max_bin=32)
print(f"analytic-seeded bins: {starts_a.size}  ({time.time()-t0_:.1f}s)")

t0_ = time.time()
pilot_tf = time_block_log_pilot(observed, 6)
print("truth-free pilot shape:", pilot_tf.shape, "  time:", time.time()-t0_)
t0_ = time.time()
starts_tf = adaptive_frequency_bin_starts(pilot_tf, max_log_range=0.15, max_bin=32)
print(f"truth-free bins: {starts_tf.size}  ({time.time()-t0_:.1f}s)")
