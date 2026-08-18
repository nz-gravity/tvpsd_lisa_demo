"""Is the null in the OMS/TM reference real, or smeared by interpolation?

The reference is evaluated on the archive's 256-point truth grid (spacing
~1.6 mHz near 60 mHz) and then log-interpolated onto the ~6000-point fit
grid. The TDI null is ~1 mHz wide, so the reference cannot resolve it.
Compare against evaluating the same analytic model DIRECTLY on the fit grid.
"""

import sys

import h5py
import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from run_component_study import (
    analytic_aet_noise_components_psd,
    interpolate_surface,
)

HERE = "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation"
d = np.load("/tmp/aet_shared_J_full.npz", allow_pickle=True)
grouped = d["frequency_hz"]
time_days = d["time_days"]

with h5py.File(f"{HERE}/combined_esa_xyz.h5", "r") as hdf:
    truth_time = hdf["truth/time_tcb"][:]
    truth_frequency = hdf["truth/frequency_hz"][:]
    t0 = float(hdf.attrs["t0_tcb"])
absolute_time = t0 + time_days * 86400.0

band = (grouped >= 0.050) & (grouped <= 0.070)
fine = grouped[band]
orbits = f"{HERE}/noise2a/orbits.h5"

print(f"evaluating analytic OMS/TM on {fine.size} fit-grid points in 50-70 mHz ...")
oms_direct, tm_direct = analytic_aet_noise_components_psd(orbits, absolute_time, fine)
direct = (oms_direct + tm_direct)[0]

print(f"evaluating on the {truth_frequency.size}-point truth grid, then interpolating ...")
oms_coarse, tm_coarse = analytic_aet_noise_components_psd(
    orbits, truth_time, truth_frequency
)
coarse_full = oms_coarse + tm_coarse
interpolated = interpolate_surface(
    coarse_full, truth_time, truth_frequency, absolute_time, fine
)[0]

row = 0
depth_direct = np.log(direct[row].max() / direct[row].min())
depth_interp = np.log(interpolated[row].max() / interpolated[row].min())
print()
print(f"  null depth, evaluated directly on fit grid : {depth_direct:7.2f} nats")
print(f"  null depth, interpolated from 256-pt grid  : {depth_interp:7.2f} nats")
print(f"  depth LOST to interpolation                : {depth_direct - depth_interp:7.2f} nats")
print()
ratio = np.log(interpolated / direct)
print(f"  median |log(interpolated / direct)| over 50-70 mHz: {np.nanmedian(np.abs(ratio)):.4f}")
print(f"  max    |log(interpolated / direct)|              : {np.nanmax(np.abs(ratio)):.4f}")
print()
print("  compare: run J's total |log err| in this band was 0.1279,")
print("           and its claimed posterior width was 0.0040")
