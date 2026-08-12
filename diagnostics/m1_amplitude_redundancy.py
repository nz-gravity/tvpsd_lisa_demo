"""Are the free OMS/TM amplitudes redundant with the noise spline?

Both act on log S. The amplitude directions are
    d log S_noise / d log a_m = a_m * M_m / (a_oms*OMS + a_tm*TM)
If those surfaces lie (nearly) in the span of the tensor spline basis, the
joint posterior has a near-flat ridge no per-block preconditioner can fix,
because the degeneracy is *between* blocks.

Reported as the weighted R^2 of projecting each amplitude direction onto the
spline column space, using the Whittle Fisher weights counts * (S_noise/S_tot)^2.
R^2 -> 1 means the spline can imitate that amplitude exactly.
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from tv_pspline_psd.splines import create_bspline_basis, create_difference_penalty_matrix

d = np.load("/tmp/aet_precond_G.npz", allow_pickle=True)
counts = d["counts"]
frequency = d["frequency_hz"]
time_days = d["time_days"]
oms = d["oms_reference_psd"]
tm = d["tm_reference_psd"]
truth_noise = d["truth_noise"]
truth_galactic = d["truth_galactic"]
N_TIME_KNOTS, N_FREQUENCY_KNOTS = 3, 40

time_unit = (time_days - time_days[0]) / (time_days[-1] - time_days[0])
log_frequency = np.log(frequency)
frequency_unit = (log_frequency - log_frequency[0]) / (log_frequency[-1] - log_frequency[0])
basis_time, _ = create_bspline_basis(time_unit, N_TIME_KNOTS, degree=3)
basis_frequency, _ = create_bspline_basis(frequency_unit, N_FREQUENCY_KNOTS, degree=3)

print(f"spline basis: {basis_time.shape[1]} time x {basis_frequency.shape[1]} freq")
print(f"grid: {time_days.size} x {frequency.size}\n")

for channel_index, channel in enumerate("AET"):
    noise = truth_noise[channel_index]
    total = noise + truth_galactic[channel_index]
    # Whittle Fisher weight on log S_noise, including galactic dilution.
    weight = counts[channel_index] * (noise / total) ** 2
    reference = oms[channel_index] + tm[channel_index]
    directions = {
        "log_a_oms": oms[channel_index] / reference,
        "log_a_tm": tm[channel_index] / reference,
    }
    # interpolate_surface fills out-of-band cells with NaN; drop them.
    valid = np.isfinite(weight) & np.isfinite(reference) & (reference > 0.0)
    for target in directions.values():
        valid &= np.isfinite(target)
    weight = np.where(valid, weight, 0.0)
    directions = {k: np.where(valid, v, 0.0) for k, v in directions.items()}
    print(f"  {channel}: {valid.sum()}/{valid.size} cells finite")

    # Weighted normal equations for the tensor spline, without forming the
    # full (cells x coefficients) design.
    # Rows are (time_basis i, freq_basis j), columns (k, l) -- the index order
    # must match vec(W) with W of shape (n_time_basis, n_freq_basis).
    gram = np.einsum(
        "tf,ti,fj,tk,fl->ijkl",
        weight, basis_time, basis_frequency, basis_time, basis_frequency,
        optimize=True,
    ).reshape(basis_time.shape[1] * basis_frequency.shape[1], -1)

    # Jacobi-scale before solving: the raw Gram spans a huge dynamic range and
    # a direct solve on the normal equations loses all precision.
    jacobi = 1.0 / np.sqrt(np.diag(gram))
    scaled_gram = jacobi[:, None] * gram * jacobi[None, :]
    print(f"  {channel}: weighted spline Gram condition number "
          f"{np.linalg.cond(scaled_gram):.3e} (after Jacobi scaling)")

    for name, target in directions.items():
        rhs = np.einsum(
            "tf,ti,fj->ij", weight * target, basis_time, basis_frequency, optimize=True
        ).ravel()
        solution, *_ = np.linalg.lstsq(scaled_gram, jacobi * rhs, rcond=1e-12)
        coefficients = jacobi * solution
        fitted = np.einsum(
            "ti,ij,fj->tf",
            basis_time,
            coefficients.reshape(basis_time.shape[1], basis_frequency.shape[1]),
            basis_frequency,
            optimize=True,
        )
        residual = float(np.sum(weight * (target - fitted) ** 2))
        total_ss = float(np.sum(weight * target**2))
        r_squared = 1.0 - residual / total_ss
        print(
            f"  {channel} {name:10s} R^2 = {r_squared:.8f}   "
            f"unexplained {1.0 - r_squared:.3e}   "
            f"-> effective ridge length ~{1.0/np.sqrt(max(1.0-r_squared,1e-16)):.0f}x"
        )
    print()
