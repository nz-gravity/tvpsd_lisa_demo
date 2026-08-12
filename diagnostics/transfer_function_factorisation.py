"""Can we split the analytic AET noise into (known transfer function) x (stationary spectrum)?

    S_noise,c(t,f) = T_TM,c(t,f) * S_TM(f) + T_OMS,c(t,f) * S_OMS(f)

If so, all the time-variation lives in T (computable from orbits at full
resolution, so the drifting null is exact) and the free splines are functions
of frequency only. Checks:
  1. T carries the null and is genuinely time-varying.
  2. T_TM != T_OMS, so the two components are distinguishable per channel.
  3. T for the T (null) channel differs from A/E -- the channel structure.
"""

import sys

import h5py
import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from run_aet_diagonal_pilot import analytic_aet_noise_components_psd

HERE = "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation"
C_LIGHT = 299792458.0
A_TM, F1, F2 = 2.4e-15, 4.0e-4, 8.0e-3
A_OMS, F3 = 7.9e-12, 2.0e-3


def s_tm_theory(f):
    return A_TM**2 * (1 + (F1 / f) ** 2) * (1 + (f / F2) ** 4) * (1.0 / (2 * np.pi * f * C_LIGHT)) ** 2


def s_oms_theory(f):
    return A_OMS**2 * (1 + (F3 / f) ** 4) * (2 * np.pi * f / C_LIGHT) ** 2


d = np.load("/tmp/aet_shared_J_full.npz", allow_pickle=True)
time_days = d["time_days"]
with h5py.File(f"{HERE}/combined_esa_xyz.h5", "r") as hdf:
    t0 = float(hdf.attrs["t0_tcb"])
absolute_time = t0 + time_days * 86400.0

# Evaluate DIRECTLY on a null-resolving grid -- no 256-point interpolation.
frequency = np.geomspace(1e-4, 0.1, 3000)
print(f"evaluating analytic OMS/TM on {frequency.size} log-spaced points, "
      f"{absolute_time.size} times (directly, no interpolation) ...")
oms_aet, tm_aet = analytic_aet_noise_components_psd(
    f"{HERE}/noise2a/orbits.h5", absolute_time, frequency
)

transfer_oms = oms_aet / s_oms_theory(frequency)[None, None, :]
transfer_tm = tm_aet / s_tm_theory(frequency)[None, None, :]

print()
print("1) Does the transfer function carry the drifting null?")
band = (frequency > 0.055) & (frequency < 0.068)
for name, transfer in (("OMS", transfer_oms), ("TM", transfer_tm)):
    t_a = transfer[0]
    null_f = frequency[band][np.argmin(t_a[:, band], axis=1)]
    depth = np.log(t_a[:, band].max() / t_a[:, band].min())
    print(f"   A-channel {name}: null moves {1e3*null_f.min():.3f} -> {1e3*null_f.max():.3f} mHz "
          f"(drift {1e6*(null_f.max()-null_f.min()):.0f} uHz), depth {depth:.1f} nats")

print()
print("2) Are TM and OMS transfer functions distinguishable? (ratio must vary with f)")
for c, name in enumerate("AET"):
    ratio = np.log(transfer_tm[c, 0] / transfer_oms[c, 0])
    print(f"   {name}: log(T_TM/T_OMS) spans {ratio.max()-ratio.min():.2f} nats over the band")

print()
print("3) Channel structure -- T should suppress TM relative to A (null channel)")
mid = np.argmin(np.abs(frequency - 2e-3))
for c, name in enumerate("AET"):
    print(f"   {name} @2mHz: T_TM={transfer_tm[c,0,mid]:.4e}  T_OMS={transfer_oms[c,0,mid]:.4e}  "
          f"TM/OMS={transfer_tm[c,0,mid]/transfer_oms[c,0,mid]:.4f}")

print()
print("4) Is the *stationary* assumption on S_TM/S_OMS consistent, i.e. does")
print("   T alone explain the time variation of the full analytic surface?")
for c, name in enumerate("AET"):
    total = oms_aet[c] + tm_aet[c]
    # Reconstruct using time-varying T but a single (t-independent) spectrum.
    reconstructed = (
        transfer_oms[c] * s_oms_theory(frequency)[None, :]
        + transfer_tm[c] * s_tm_theory(frequency)[None, :]
    )
    err = np.max(np.abs(np.log(reconstructed / total)))
    print(f"   {name}: max |log(reconstructed/analytic)| = {err:.3e}")
