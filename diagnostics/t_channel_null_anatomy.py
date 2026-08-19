"""Why the T channel is not signal-free on breathing arms.

Compares the T/A power ratio for the instrumental noise and for the Galactic
foreground, under a rigid equal-arm constellation and under the ESA trailing
orbits actually used in the study. Produces the numbers quoted in the
T-channel appendix.

With equal arms the foreground is nulled ~2000x harder than the noise, so T is
effectively signal-free. Unequal, breathing arms leak a common fraction of the
A-channel content into T; that leak carries A's composition, so the two floors
collide and the foreground is no longer negligible in T.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import warnings

import h5py
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

import lisaorbits  # noqa: E402
from backgrounds import StochasticBackgroundResponse, noise as bn, signal, tdi  # noqa: E402

from tv_pspline_psd.lisa_aet import xyz_covariance_to_aet_diagonal as to_aet  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FREQUENCY_HZ = np.geomspace(1.0e-4, 5.0e-2, 60)


def component_surfaces(orbits, time_tcb):
    """Return time-averaged (noise, galactic) A/E/T auto-PSDs on FREQUENCY_HZ."""
    oms = bn.AnalyticOMSNoiseModel(
        FREQUENCY_HZ, time_tcb, orbits, tdi_tf_func=tdi.compute_tdi_tf, gen="2.0",
        oms_isi_carrier_asds=7.9e-12, fs=0.5, duration=365.25 * 86400,
    )
    test_mass = bn.AnalyticTMNoiseModel(
        FREQUENCY_HZ, time_tcb, orbits, tdi_tf_func=tdi.compute_tdi_tf_tm, gen="2.0",
        tm_isi_carrier_asds=2.4e-15, fs=0.5, duration=365.25 * 86400,
    )
    oms_aet = to_aet(oms.compute_covariances(0.0))
    tm_aet = to_aet(test_mass.compute_covariances(0.0))

    galaxy = signal.Galaxy(nside=8, tobs_yrs=1.0)
    power = galaxy.compute_map(n_points=400, lmin=1.0e-3, lmax=20.0, xsun=-8.1, coord="C")
    kernel = StochasticBackgroundResponse(
        np.sqrt(power / power.sum()), orbits=orbits
    ).compute_tdi_kernel(FREQUENCY_HZ, time_tcb, tdi_var="xyz", gen="2.0")
    galactic = to_aet(kernel) * galaxy.psd(FREQUENCY_HZ)[:, None, None]
    return oms_aet.mean(axis=1), tm_aet.mean(axis=1), galactic.mean(axis=1)


def esa_orbits():
    with h5py.File(f"{HERE}/../noise2a/orbits.h5", "r") as hdf:
        t0 = float(hdf.attrs["t0"])
        dt = float(hdf.attrs["dt"])
        size = int(hdf.attrs["size"])
        return lisaorbits.InterpolatedOrbits(
            t0 + dt * np.arange(size), hdf["tcb/x"][:], t_init=t0,
            interp_order=5, extrapolate=False,
        )


def equal_arm_orbits():
    base = lisaorbits.EqualArmlengthOrbits()
    grid = np.arange(0.0, 632 * 1.0e5, 1.0e5)
    return lisaorbits.InterpolatedOrbits(
        grid, base.compute_position(grid), t_init=0.0, interp_order=5, extrapolate=False
    )


def main() -> None:
    oms_e, tm_e, gal_e = component_surfaces(
        equal_arm_orbits(), np.linspace(100 * 86400, 400 * 86400, 4)
    )
    oms_s, tm_s, gal_s = component_surfaces(
        esa_orbits(), 2073211230.8175 + np.linspace(0, 300 * 86400, 6)
    )
    noise_e, noise_s = oms_e + tm_e, oms_s + tm_s

    print("T/A power ratio, noise and Galactic foreground")
    print(f"{'f [mHz]':>9} | {'equal noise':>11} {'equal gal':>11} "
          f"| {'ESA noise':>11} {'ESA gal':>11}")
    for i in range(0, FREQUENCY_HZ.size, 6):
        print(f"{FREQUENCY_HZ[i]*1e3:9.3f} | {noise_e[i,2]/noise_e[i,0]:11.2e} "
              f"{gal_e[i,2]/gal_e[i,0]:11.2e} | {noise_s[i,2]/noise_s[i,0]:11.2e} "
              f"{gal_s[i,2]/gal_s[i,0]:11.2e}")

    print("\nTest-mass fraction of the channel NOISE, p^TM_C")
    print(f"{'band [mHz]':>12} | {'ESA A':>8} {'ESA T':>8} | {'equal A':>8} {'equal T':>8}")
    for lo, hi in ((0.1, 1.0), (0.1, 3.0), (3.0, 20.0)):
        b = (FREQUENCY_HZ >= lo * 1e-3) & (FREQUENCY_HZ < hi * 1e-3)
        print(f"{lo:5.1f}-{hi:5.1f} | "
              f"{(tm_s[b,0]/noise_s[b,0]).mean():8.4f} {(tm_s[b,2]/noise_s[b,2]).mean():8.4f} | "
              f"{(tm_e[b,0]/noise_e[b,0]).mean():8.4f} {(tm_e[b,2]/noise_e[b,2]).mean():8.4f}")


if __name__ == "__main__":
    main()
