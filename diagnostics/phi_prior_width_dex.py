"""How much freedom does each phi actually give? (Nazeela Appendix B, in dex)

lam ~ N(lam_loc, (phi P)^-1) is not scale-free, so the numbers 1e4 / 1e8 are
uninterpretable on their own. Convert to the quantity that matters: the 90%
prior width of log10 S(f) at representative frequencies.

Also reports how far the FITTED noise moved from the prior centre, to check
whether the data informed the splines or the answer is just the initialization.
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from aet_component_pspline_nuts import penalized_spline_prior_centre
from run_aet_diagonal_pilot import oms_theory_psd, tm_theory_psd
from tv_pspline_psd.splines import (
    create_bspline_basis,
    create_difference_penalty_matrix,
)

frequency = np.geomspace(1e-4, 2e-2, 400)
n_knots = 12
log_f = np.log(frequency)
unit = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
basis, _ = create_bspline_basis(unit, n_knots, degree=3)
penalty = create_difference_penalty_matrix(basis.shape[1], diff_order=2)
penalty = penalty + 1e-6 * np.eye(basis.shape[1]) * np.trace(penalty) / basis.shape[1]

probe = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
rng = np.random.default_rng(0)
cov = np.linalg.inv(penalty)
chol = np.linalg.cholesky(cov)

print("90% prior width of log10 S(f), in dex\n")
print(f"{'phi':>8} " + " ".join(f"{p*1e3:>8.2f}mHz" for p in probe))
for phi in (1e2, 1e4, 1e6, 1e8):
    draws = (chol @ rng.normal(size=(basis.shape[1], 4000))) / np.sqrt(phi)
    log_s = (basis @ draws) / np.log(10.0)  # deviation from centre, in dex
    widths = []
    for p in probe:
        j = int(np.argmin(np.abs(frequency - p)))
        lo, hi = np.quantile(log_s[j], [0.05, 0.95])
        widths.append(hi - lo)
    print(f"{phi:8.0e} " + " ".join(f"{w:11.4f}" for w in widths))

print()
print("Interpretation: phi=1e8 permits ~0.001 dex (0.2%) departures from theory.")
print("The T-channel TM excess measured today is 10-23%, i.e. 0.04-0.09 dex --")
print("about 50-100x wider than that prior allows.")
print()

# How far did the fit actually move?
d = np.load("/tmp/aet_final.npz", allow_pickle=True)
fitted, analytic = d["posterior_noise"], d["truth_noise"]
freq_b, mask = d["frequency_hz"], d["fit_mask"]
print("Fitted noise vs analytic centre (did the data move the splines?)")
print(f"{'band':>16} {'A dex':>9} {'E dex':>9} {'T dex':>9}")
for lo, hi in [(1e-4, 1e-3), (1e-3, 3e-3), (3e-3, 8e-3), (8e-3, 2e-2)]:
    s = (freq_b >= lo) & (freq_b < hi)
    if not s.any():
        continue
    row = []
    for c in range(3):
        v = mask[c][:, s] & np.isfinite(analytic[c][:, s])
        if not v.any():
            row.append(np.nan); continue
        row.append(np.median(np.log10(fitted[c][:, s][v] / analytic[c][:, s][v])))
    print(f"{lo*1e3:6.2f}-{hi*1e3:6.2f} mHz " + " ".join(f"{x:9.4f}" for x in row))
