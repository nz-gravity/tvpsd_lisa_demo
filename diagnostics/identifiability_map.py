"""Time-varying version of Nazeela et al. Appendix A.

Their stationary version asks: which channel constrains which noise component?
In the time-varying case there is a second, orthogonal question: which
component is *modulated*, since modulation is what separates the Galaxy from
the instrument once the noise's own time dependence is locked to L(t).

Per channel and frequency band, reports
  - fractional contributions p_TM, p_OMS, p_gal
  - annual modulation depth of each (peak-to-peak in log)
so you can read off where component separation has leverage.
"""

import sys

import h5py
import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from run_aet_diagonal_pilot import (
    aet_noise_transfer_functions,
    interpolate_surface,
    oms_theory_psd,
    tm_theory_psd,
)
from tv_pspline_psd.lisa_aet import diagonal_xyz_psd_to_aet

HERE = "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation"
with h5py.File(f"{HERE}/combined_esa_xyz.h5", "r") as hdf:
    t0 = float(hdf.attrs["t0_tcb"])
    truth_time = hdf["truth/time_tcb"][:]
    truth_frequency = hdf["truth/frequency_hz"][:]
    galactic_xyz = hdf["truth/galactic_psd"][:]

frequency = np.geomspace(1e-4, 2e-2, 400)
time = np.linspace(truth_time[0], truth_time[-1], 30)

transfer_tm, transfer_oms = aet_noise_transfer_functions(
    f"{HERE}/noise2a/orbits.h5", time, frequency
)
tm_part = transfer_tm * tm_theory_psd(frequency)[None, None, :]
oms_part = transfer_oms * oms_theory_psd(frequency)[None, None, :]
galactic = interpolate_surface(
    diagonal_xyz_psd_to_aet(galactic_xyz), truth_time, truth_frequency, time, frequency
)
total = tm_part + oms_part + galactic


def depth(surface):
    """Annual modulation depth: peak-to-peak of log over time, median over f."""
    return float(np.nanmedian(np.ptp(np.log(surface), axis=0)))


bands = [(1e-4, 3e-4), (3e-4, 1e-3), (1e-3, 3e-3), (3e-3, 8e-3), (8e-3, 2e-2)]
print("Fractional contribution / annual modulation depth in nats\n")
for c, name in enumerate("AET"):
    print(f"  channel {name}")
    print(f"    {'band':>16} {'p_TM':>7} {'p_OMS':>7} {'p_gal':>7} | "
          f"{'mod TM':>7} {'mod OMS':>8} {'mod gal':>8}")
    for a, b in bands:
        s = (frequency >= a) & (frequency < b)
        if not s.any():
            continue
        tot = total[c][:, s]
        p_tm = np.nanmean(tm_part[c][:, s] / tot)
        p_oms = np.nanmean(oms_part[c][:, s] / tot)
        p_gal = np.nanmean(galactic[c][:, s] / tot)
        print(
            f"    {a*1e3:6.2f}-{b*1e3:6.2f} mHz {p_tm:7.3f} {p_oms:7.3f} {p_gal:7.3f} | "
            f"{depth(tm_part[c][:, s]):7.3f} {depth(oms_part[c][:, s]):8.3f} "
            f"{depth(galactic[c][:, s]):8.3f}"
        )
    print()

print("Reading: separation needs a component that is BOTH non-negligible (p)")
print("AND distinguishable. The Galaxy's modulation depth vs the instrument's")
print("is the discriminant a stationary analysis does not have.")
