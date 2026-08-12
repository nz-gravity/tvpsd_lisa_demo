"""Is A_gal identified by the DATA, or only by the roughness prior?

Perturbing log A_gal moves log S_total along  g = A_gal*T_gal / S_total.
Perturbing the noise spline moves it along  (1-g) * b_t(t) * b_f(f).
If g lies in the span of the spline directions, the noise surface can absorb
the Galaxy entirely and the component split is prior-driven, not likelihood-
driven.

Swept over the number of TIME basis functions, because time is the physical
discriminant: the Galaxy is annually modulated (~1.4 nats) while the
instrument noise is nearly stationary (~0.05 nats) once the known armlength
drift is accounted for.
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from tv_pspline_psd.splines import create_bspline_basis

d = np.load("/tmp/aet_precond_G.npz", allow_pickle=True)
counts = d["counts"]
frequency = d["frequency_hz"]
time_days = d["time_days"]
truth_noise = d["truth_noise"]
truth_galactic = d["truth_galactic"]
N_FREQUENCY_KNOTS = 40

time_unit = (time_days - time_days[0]) / (time_days[-1] - time_days[0])
log_frequency = np.log(frequency)
frequency_unit = (log_frequency - log_frequency[0]) / (log_frequency[-1] - log_frequency[0])
basis_frequency, _ = create_bspline_basis(frequency_unit, N_FREQUENCY_KNOTS, degree=3)


def weighted_r_squared(basis_time, weight, galactic_fraction):
    """R^2 of projecting the Galactic direction onto the noise-spline span."""
    noise_fraction = 1.0 - galactic_fraction
    n_t, n_f = basis_time.shape[1], basis_frequency.shape[1]
    normal = np.zeros((n_t, n_f, n_t, n_f))
    rhs = np.zeros((n_t, n_f))
    for t in range(basis_time.shape[0]):
        v = weight[t] * noise_fraction[t] ** 2
        u = weight[t] * noise_fraction[t] * galactic_fraction[t]
        gram_f = basis_frequency.T @ (v[:, None] * basis_frequency)
        normal += (
            basis_time[t][:, None, None, None]
            * basis_time[t][None, None, :, None]
            * gram_f[None, :, None, :]
        )
        rhs += basis_time[t][:, None] * (basis_frequency.T @ u)[None, :]
    normal = normal.reshape(n_t * n_f, n_t * n_f)
    rhs = rhs.ravel()
    jacobi = 1.0 / np.sqrt(np.maximum(np.diag(normal), 1e-300))
    solution, *_ = np.linalg.lstsq(
        jacobi[:, None] * normal * jacobi[None, :], jacobi * rhs, rcond=1e-12
    )
    coefficients = jacobi * solution
    total = float(np.sum(weight * galactic_fraction**2))
    explained = float(coefficients @ rhs)
    return 1.0 - (total - explained) / total


print("Can the free noise spline imitate the Galactic amplitude direction?")
print("(R^2 -> 1 means A_gal is identified only by the roughness prior)\n")
print(f"{'channel':>8} {'time basis':>11} {'R^2':>14} {'unexplained':>13}")
for channel_index, channel in enumerate("AET"):
    noise = truth_noise[channel_index]
    total_psd = noise + truth_galactic[channel_index]
    galactic_fraction = truth_galactic[channel_index] / total_psd
    weight = counts[channel_index]
    valid = np.isfinite(galactic_fraction) & np.isfinite(weight)
    weight = np.where(valid, weight, 0.0)
    galactic_fraction = np.where(valid, galactic_fraction, 0.0)
    for n_time_knots in (3, 1, 0, None):
        if n_time_knots is None:
            basis_time = np.ones((time_days.size, 1))
            label = "1 (static)"
        else:
            basis_time, _ = create_bspline_basis(time_unit, n_time_knots, degree=3)
            label = str(basis_time.shape[1])
        r2 = weighted_r_squared(basis_time, weight, galactic_fraction)
        print(f"{channel:>8} {label:>11} {r2:>14.8f} {1.0 - r2:>13.3e}")
    print()
