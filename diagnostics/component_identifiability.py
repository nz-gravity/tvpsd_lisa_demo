"""Can the data separate S_TM from S_OMS, and does trying cost anything?

Computes the Whittle Fisher information for the component spectra at the
injected truth, per frequency, summed over all three A/E/T channels and all
fitted times. Two models are compared:

  two spectra   S_noise,c = T_TM,c S_TM(f) + T_OMS,c S_OMS(f)
  recalibration S_noise,c = a(f) R_c(t,f)

For a Whittle cell with nu coefficients, the information for log S is nu/2 and
d log S_total / d log S_i is the fractional contribution p_i, so
F_ij = sum_{c,t} (nu/2) p_i p_j.

Result on the archived continuous fit: the two-spectrum split is resolved
(sigma < 0.1 nats) over only 3.2-5.0 mHz, 28 of 877 bins, with the two spectra
at correlation -0.99 to -1.00 elsewhere. The recalibration a(f) is resolved
over 856 of 877 bins, and the Galactic amplitude is measured roughly twice as
well because it no longer has to be marginalised against a degenerate ridge.
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent / "results" / "h_para_aet_continuous_15647844.npz"


def channel_fractions(archive, channel, k):
    """Fitted-cell (p_TM, p_OMS, p_gal) and counts at frequency index ``k``."""
    selected = archive["fit_mask"][channel][:, k]
    if not selected.any():
        return None
    tm = archive["tm_reference_psd"][channel][selected, k]
    oms = archive["oms_reference_psd"][channel][selected, k]
    galactic = archive["truth_galactic"][channel][selected, k]
    total = tm + oms + galactic
    counts = archive["counts"]
    nu = counts[selected, k] if counts.ndim == 2 else counts[channel][selected, k]
    return np.stack([tm / total, oms / total, galactic / total]), nu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="sigma(log S) in nats defining 'resolved'")
    args = parser.parse_args()

    archive = np.load(args.archive, allow_pickle=True)
    frequency = archive["frequency_hz"]
    n = frequency.size

    sigma_tm = np.full(n, np.inf)
    sigma_oms = np.full(n, np.inf)
    sigma_a = np.full(n, np.inf)
    correlation = np.full(n, np.nan)
    amplitude_two = 0.0
    fisher_aa = np.zeros(n)
    fisher_aA = np.zeros(n)
    fisher_AA = 0.0

    for k in range(n):
        pair = np.zeros((3, 3))
        single = np.zeros(3)
        for channel in range(3):
            got = channel_fractions(archive, channel, k)
            if got is None:
                continue
            p, nu = got
            weight = nu / 2.0
            pair += np.einsum("it,jt,t->ij", p, p, weight)
            noise_fraction = p[0] + p[1]
            single += np.array([
                np.sum(weight * noise_fraction**2),
                np.sum(weight * noise_fraction * p[2]),
                np.sum(weight * p[2] ** 2),
            ])
        if np.linalg.cond(pair[:2, :2]) < 1e12:
            covariance = np.linalg.inv(pair[:2, :2])
            sigma_tm[k], sigma_oms[k] = np.sqrt(np.diag(covariance))
            correlation[k] = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
            amplitude_two += pair[2, 2] - pair[2, :2] @ np.linalg.solve(pair[:2, :2], pair[:2, 2])
        fisher_aa[k], fisher_aA[k] = single[0], single[1]
        fisher_AA += single[2]
        if single[0] > 0.0:
            sigma_a[k] = 1.0 / np.sqrt(single[0])

    good = fisher_aa > 0.0
    amplitude_one = fisher_AA - np.sum(fisher_aA[good] ** 2 / fisher_aa[good])

    def band(sigma):
        resolved = frequency[sigma < args.threshold] * 1e3
        if resolved.size == 0:
            return "never resolved"
        return f"{resolved.min():7.3f} - {resolved.max():7.3f} mHz ({resolved.size:3d}/{n} bins)"

    print(f"archive: {args.archive.name}")
    print(f"resolved means sigma(log S) < {args.threshold} nats\n")
    print("two-spectrum model")
    print(f"  S_TM            {band(sigma_tm)}")
    print(f"  S_OMS           {band(sigma_oms)}")
    both = (sigma_tm < args.threshold) & (sigma_oms < args.threshold)
    print(f"  both together   {band(np.where(both, 0.0, np.inf))}")
    finite = np.isfinite(correlation)
    print(f"  corr(S_TM,S_OMS) median {np.median(correlation[finite]):+.3f}")
    print(f"  sigma(log A_gal) marginalised {1/np.sqrt(amplitude_two):.5f} nats")
    print("\nrecalibration model")
    print(f"  a(f)            {band(sigma_a)}")
    print(f"  sigma(log A_gal) marginalised {1/np.sqrt(amplitude_one):.5f} nats")
    print(f"  amplitude information retained {amplitude_one/fisher_AA:.4f}")
    print(f"\nGalactic amplitude is measured {np.sqrt(amplitude_one/amplitude_two):.2f}x "
          f"better under the recalibration model.")


if __name__ == "__main__":
    main()
