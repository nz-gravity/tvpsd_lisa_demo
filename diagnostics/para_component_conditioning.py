"""Is the component model's prior-only whitening actually costing anything?

fit_aet_component_noise_nuts samples z with lam = lam_loc + L^-T z / sqrt(phi),
where P = L L^T is the PRIOR penalty. That makes the PRIOR isotropic in z. But
the Whittle Fisher is ~counts/2 per cell, and with phi_oms = 1e4 (deliberately
loose) the OMS direction should be data-dominated -- so the posterior in z may
be badly conditioned even though the prior is perfect.

Reports the condition number after ideal diagonal (mass-matrix) preconditioning,
which is what NUTS actually sees. Leapfrog steps scale like its square root.
"""

import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from run_component_study import oms_theory_psd, tm_theory_psd
from tv_pspline_psd.splines import create_bspline_basis, create_difference_penalty_matrix

d = np.load("/tmp/aet_s12.npz", allow_pickle=True)
frequency = d["frequency_hz"]
counts = d["counts"]
truth_noise = d["truth_noise"]
truth_galactic = d["truth_galactic"]
mask = d["fit_mask"]
oms_ref = d["oms_reference_psd"]
tm_ref = d["tm_reference_psd"]

N_KNOTS = 12
PHI_TM, PHI_OMS = 1.0e8, 1.0e4

log_f = np.log(frequency)
unit = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
basis, _ = create_bspline_basis(unit, N_KNOTS, degree=3)
penalty = create_difference_penalty_matrix(basis.shape[1], diff_order=2)
penalty = penalty + 1e-6 * np.eye(basis.shape[1]) * np.trace(penalty) / basis.shape[1]
k = basis.shape[1]

# Transfer functions implied by the saved references and the theory spectra.
transfer_tm = tm_ref / tm_theory_psd(frequency)[None, None, :]
transfer_oms = oms_ref / oms_theory_psd(frequency)[None, None, :]
spectrum_tm = tm_theory_psd(frequency)
spectrum_oms = oms_theory_psd(frequency)

# Fisher w.r.t. (lam_tm, lam_oms): d logS_total / d lam_i_j = frac_i * b_j.
fisher = np.zeros((2 * k, 2 * k))
for c in range(3):
    valid = mask[c] & np.isfinite(truth_noise[c]) & (truth_noise[c] > 0)
    tm_part = transfer_tm[c] * spectrum_tm[None, :]
    oms_part = transfer_oms[c] * spectrum_oms[None, :]
    total = tm_part + oms_part + truth_galactic[c]
    w = np.where(valid, counts[c], 0.0)
    frac_tm = np.where(valid, tm_part / total, 0.0)
    frac_oms = np.where(valid, oms_part / total, 0.0)
    # sum over time of w * frac_i * frac_j gives a per-frequency weight
    for (fi, fj, r0, c0) in (
        (frac_tm, frac_tm, 0, 0),
        (frac_tm, frac_oms, 0, k),
        (frac_oms, frac_oms, k, k),
    ):
        weight = np.sum(w * fi * fj, axis=0)
        block = basis.T @ (weight[:, None] * basis)
        fisher[r0:r0 + k, c0:c0 + k] += 0.5 * block
fisher[k:, :k] = fisher[:k, k:].T

prior_precision = np.zeros((2 * k, 2 * k))
prior_precision[:k, :k] = PHI_TM * penalty
prior_precision[k:, k:] = PHI_OMS * penalty


def preconditioned_condition(hessian):
    scale = 1.0 / np.sqrt(np.diag(hessian))
    return np.linalg.cond(scale[:, None] * hessian * scale[None, :])


posterior = fisher + prior_precision

print("Fisher vs prior magnitude (trace ratio), per block:")
print(f"  TM  : likelihood/prior = {np.trace(fisher[:k,:k])/np.trace(PHI_TM*penalty):.3e}")
print(f"  OMS : likelihood/prior = {np.trace(fisher[k:,k:])/np.trace(PHI_OMS*penalty):.3e}")
print()

# (A) current coordinates: z, with lam = lam_loc + L^-T z / sqrt(phi)
chol_inv = sla.solve_triangular(np.linalg.cholesky(penalty).T, np.eye(k), lower=False)
jac = np.zeros((2 * k, 2 * k))
jac[:k, :k] = chol_inv / np.sqrt(PHI_TM)
jac[k:, k:] = chol_inv / np.sqrt(PHI_OMS)
posterior_z = jac.T @ posterior @ jac
cond_current = preconditioned_condition(posterior_z)

# (B) likelihood-whitened: Cholesky of the full posterior precision
chol_post = np.linalg.cholesky(posterior)
amat = sla.solve_triangular(chol_post.T, np.eye(2 * k), lower=False)
posterior_w = amat.T @ posterior @ amat
cond_whitened = preconditioned_condition(posterior_w)

print(f"  A) current (prior-whitened)      cond {cond_current:12.3e}  ~steps {np.sqrt(cond_current):8.1f}")
print(f"  B) likelihood-whitened           cond {cond_whitened:12.3e}  ~steps {np.sqrt(cond_whitened):8.1f}")
print()
print(f"  improvement: {cond_current/cond_whitened:.3e}x in condition number")
print(f"  tree depth needed: current >= {np.ceil(np.log2(max(np.sqrt(cond_current),1))):.0f}, "
      f"whitened >= {np.ceil(np.log2(max(np.sqrt(cond_whitened),1))):.0f}")
