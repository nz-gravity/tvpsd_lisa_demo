"""Is the under-coverage a WIDTH problem or a BIAS problem?

This decides whether tempering (Safe Bayes) is the right tool.

  * If standardized residuals z = log(median/truth) / sigma_post have mean ~0
    and std ~k>1, the posterior is simply too narrow: the likelihood
    over-counts independent information (correlated WDM coefficients).
    Tempering with eta = 1/k^2 is then the CORRECT effective-DOF correction,
    and eta is measurable rather than tuned.

  * If z has a systematic nonzero mean, the centre is wrong. Tempering would
    widen the intervals until they cover a biased answer -- masking the
    problem rather than fixing it.
"""

import numpy as np

d = np.load("/tmp/aet_v3_component.npz", allow_pickle=True)
total = d["posterior_total"]
lower = d["posterior_total_lower"]
upper = d["posterior_total_upper"]
truth = d["truth_noise"] + d["truth_galactic"]
frequency = d["frequency_hz"]
mask = d["fit_mask"]

# 90% interval -> sigma via the Gaussian factor on the log scale.
Z90 = 1.6448536269514722

print("standardized residuals  z = log(median/truth) / sigma_post")
print("(mean ~0 and std k>1 => too narrow; mean != 0 => biased centre)\n")
print(f"{'chan':>5} {'mean z':>9} {'std z':>8} {'implied k':>10} {'eta=1/k^2':>10} {'cov':>7}")
for c, name in enumerate("AET"):
    valid = mask[c] & np.isfinite(truth[c]) & (truth[c] > 0)
    resid = np.log(total[c][valid] / truth[c][valid])
    sigma = np.log(upper[c][valid] / lower[c][valid]) / (2 * Z90)
    z = resid / sigma
    coverage = np.mean(
        (truth[c][valid] >= lower[c][valid]) & (truth[c][valid] <= upper[c][valid])
    )
    k = float(np.std(z))
    print(
        f"{name:>5} {np.mean(z):9.3f} {k:8.2f} {k:10.2f} {1.0/max(k,1e-9)**2:10.4f} {coverage:7.3f}"
    )

print()
print("Decomposition of the total squared residual:")
for c, name in enumerate("AET"):
    valid = mask[c] & np.isfinite(truth[c]) & (truth[c] > 0)
    resid = np.log(total[c][valid] / truth[c][valid])
    sigma = np.log(upper[c][valid] / lower[c][valid]) / (2 * Z90)
    z = resid / sigma
    systematic = np.mean(z) ** 2
    scatter = np.var(z)
    print(
        f"  {name}: mean(z)^2 = {systematic:8.3f}   var(z) = {scatter:8.3f}   "
        f"-> {100*systematic/(systematic+scatter):5.1f}% of the miss is systematic"
    )

print()
print("Where does the residual sit in frequency?")
edges = [(1e-4, 5e-4), (5e-4, 1e-3), (1e-3, 3e-3), (3e-3, 8e-3), (8e-3, 2e-2)]
print(f"{'band':>16} {'mean z (A)':>11} {'std z (A)':>10} {'gal frac':>9}")
gal_frac = d["truth_galactic"] / truth
for a, b in edges:
    s = (frequency >= a) & (frequency < b)
    if not s.any():
        continue
    c = 0
    valid = mask[c][:, s] & np.isfinite(truth[c][:, s])
    resid = np.log(total[c][:, s][valid] / truth[c][:, s][valid])
    sigma = np.log(upper[c][:, s][valid] / lower[c][:, s][valid]) / (2 * Z90)
    z = resid / sigma
    print(
        f"{a*1e3:6.2f}-{b*1e3:5.1f} mHz {np.mean(z):11.2f} {np.std(z):10.2f} "
        f"{np.nanmean(gal_frac[c][:, s]):9.3f}"
    )
