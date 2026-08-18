"""Posterior conditioning of the H_para spline block, exactly, without MCMC.

For the Whittle likelihood with a log link, the Fisher information w.r.t.
Lambda = log S is counts/2 per cell. With Lambda = B_t W B_f^T and counts
separable (here counts vary only with frequency), the information w.r.t.
vec(W) is an exact Kronecker product:

    H_lik = 0.5 * (B_t^T B_t) kron (B_f^T diag(c_f) B_f)

so the whole 308-per-channel posterior precision is available from two small
matrices. Number of NUTS leapfrog steps scales like sqrt(condition number)
AFTER ideal diagonal mass adaptation, so that is what we report.

Compares:
  A) current coordinates -- eigenbasis of the PENALTY alone (whiten_penalty_pair)
  B) Demmler-Reinsch coordinates -- generalized eigenbasis of (penalty, Fisher),
     which diagonalizes the likelihood exactly instead of the prior.
"""

import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/wdm_psd")

from tv_pspline_psd.splines import create_bspline_basis, create_difference_penalty_matrix

d = np.load("/tmp/aet_omstm_D.npz", allow_pickle=True)
counts = d["counts"][0]
frequency = d["frequency_hz"]
time_days = d["time_days"]
PHI = 100.0
NULL_SD = 5.0
N_TIME_KNOTS, N_FREQUENCY_KNOTS = 3, 40

c_f = counts[0]
assert np.allclose(counts, c_f[None, :]), "counts must be separable for the exact Kronecker form"

time_unit = (time_days - time_days[0]) / (time_days[-1] - time_days[0])
log_frequency = np.log(frequency)
frequency_unit = (log_frequency - log_frequency[0]) / (log_frequency[-1] - log_frequency[0])
basis_time, _ = create_bspline_basis(time_unit, N_TIME_KNOTS, degree=3)
basis_frequency, _ = create_bspline_basis(frequency_unit, N_FREQUENCY_KNOTS, degree=3)
penalty_time = create_difference_penalty_matrix(basis_time.shape[1], diff_order=2)
penalty_frequency = create_difference_penalty_matrix(basis_frequency.shape[1], diff_order=2)

# Likelihood Fisher factors (separable counts).
gram_time = basis_time.T @ basis_time
gram_frequency = basis_frequency.T @ (c_f[:, None] * basis_frequency)


def preconditioned_condition_number(hessian):
    """Condition number after ideal diagonal (mass-matrix) preconditioning."""
    scale = 1.0 / np.sqrt(np.diag(hessian))
    return np.linalg.cond(scale[:, None] * hessian * scale[None, :])


def report(label, transform_time, transform_frequency):
    bt = basis_time @ transform_time
    bf = basis_frequency @ transform_frequency
    lik = 0.5 * np.kron(bt.T @ bt, bf.T @ (c_f[:, None] * bf))
    lam_t = transform_time.T @ penalty_time @ transform_time
    lam_f = transform_frequency.T @ penalty_frequency @ transform_frequency
    prior = PHI * np.kron(lam_t, transform_frequency.T @ transform_frequency) + PHI * np.kron(
        transform_time.T @ transform_time, lam_f
    )
    # Weak proper prior on the joint null space, as the model does.
    null = np.diag(prior) < 1e-8 * max(np.diag(prior).max(), 1.0)
    prior[np.diag_indices_from(prior)] += np.where(null, 1.0 / NULL_SD**2, 0.0)
    hessian = lik + prior
    condition = preconditioned_condition_number(hessian)
    print(
        f"  {label:42s} cond {condition:12.3e}   ~leapfrog steps {np.sqrt(condition):9.0f}"
    )
    return condition


print(f"spline block: {basis_time.shape[1]} time x {basis_frequency.shape[1]} freq "
      f"= {basis_time.shape[1]*basis_frequency.shape[1]} coefficients per channel\n")

# A) Current: eigenbasis of the penalty alone.
_, u_time = np.linalg.eigh(penalty_time)
_, u_frequency = np.linalg.eigh(penalty_frequency)
current = report("A) current (penalty eigenbasis)", u_time, u_frequency)

# B) Demmler-Reinsch: generalized eigenbasis of (penalty, Fisher).
ridge_t = 1e-10 * np.trace(gram_time) / gram_time.shape[0]
ridge_f = 1e-10 * np.trace(gram_frequency) / gram_frequency.shape[0]
_, v_time = sla.eigh(penalty_time, gram_time + ridge_t * np.eye(gram_time.shape[0]))
_, v_frequency = sla.eigh(
    penalty_frequency, gram_frequency + ridge_f * np.eye(gram_frequency.shape[0])
)
generalized = report("B) Demmler-Reinsch (likelihood-whitened)", v_time, v_frequency)

print(f"\n  improvement: {current/generalized:.3e}x in condition number, "
      f"{np.sqrt(current/generalized):.0f}x fewer leapfrog steps")
print(f"  current needs tree depth >= {np.ceil(np.log2(np.sqrt(current))):.0f} "
      f"(cap is 10); Demmler-Reinsch needs >= {np.ceil(np.log2(np.sqrt(generalized))):.0f}")
