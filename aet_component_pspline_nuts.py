"""Joint diagonal-AET NUTS model for Galactic/noise PSD separation.

Each A/E/T channel has an independent univariate Whittle likelihood.  Galactic
amplitude and knee frequency are shared. The noise term is the analytic
OMS+TM surface as a fixed multiplicative offset, times a free log-P-spline
residual, so the *shape* of the noise -- including its drifting TDI null --
enters the likelihood while the spline retains full freedom to depart from it.

Two measurements drove this design, both in the scratchpad diagnostics:

* A single scalar noise level is underdetermined by ~2 orders of magnitude
  against the true surface's dynamic range, forcing the spline into a
  >20-prior-sigma pull at any usable roughness penalty. Supplying the shape
  as an offset removed that and improved recovery ~9x.
* Free OMS/TM *amplitudes* are reproduced by the frequency spline to one part
  in 1e11 (R^2 = 1.00000000), i.e. almost perfectly redundant. Sampling them
  adds ridges ~1e5 times longer than wide, between parameter blocks, which no
  per-block preconditioner can remove. They are therefore not sampled.

Coefficients are sampled in likelihood-whitened coordinates
(``coefficient_preconditioner``) rather than the penalty eigenbasis, because
with millions of pooled counts the likelihood dominates the prior.

Cross-spectra are intentionally ignored.  Consequently this is a diagonal
likelihood approximation, not a full AET covariance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import time as walltime

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value

import scipy.linalg as sla

from tv_pspline_psd.config import PSplineConfig
from tv_pspline_psd.model import (
    eigen_prior_scale,
    whiten_penalty_pair,
)
from tv_pspline_psd.splines import (
    create_bspline_basis,
    create_difference_penalty_matrix,
    evaluate_bspline_basis,
)

from aet_diagonal import AET_CHANNELS
from component_pspline_nuts import _stable_log_knee_ratio


@dataclass(frozen=True)
class AETDiagonalPosterior:
    """Posterior summaries with channel order A, E, T."""

    noise_median: np.ndarray
    noise_lower: np.ndarray
    noise_upper: np.ndarray
    galactic_median: np.ndarray
    galactic_lower: np.ndarray
    galactic_upper: np.ndarray
    total_median: np.ndarray
    total_lower: np.ndarray
    total_upper: np.ndarray
    amplitude_draws: np.ndarray
    f_knee_draws_hz: np.ndarray
    diagnostics: dict[str, float | int]
    samples: dict[str, np.ndarray]
    mcmc: MCMC
    runtime_seconds: float
    phi_time: float
    phi_frequency: float
    noise_level_log_sd: float


def coefficient_preconditioner(
    basis_eig_time: np.ndarray,
    basis_eig_frequency: np.ndarray,
    counts_per_frequency: np.ndarray,
    coefficient_scale: np.ndarray,
) -> np.ndarray:
    """Return ``A`` mapping isotropic ``z`` to spline coefficients ``s = A z``.

    For a Whittle likelihood with a log link the Fisher information w.r.t.
    ``log S`` is ``counts / 2`` per cell and does **not** depend on the
    parameter values, so the Gaussian approximation of the posterior precision

        H = 0.5 * (B_t^T B_t) kron (B_f^T diag(counts) B_f) + diag(1 / scale^2)

    is an exact constant, not a mode-dependent linearization. Taking
    ``H = L L^T`` and ``A = L^-T`` gives ``cov(s) = A A^T = H^-1``, i.e. the
    posterior is isotropic in ``z`` up to the (mild) non-Gaussianity of the
    Whittle likelihood.

    The existing penalty eigenbasis whitens the *prior*, which is the wrong
    target here: with millions of pooled counts the likelihood dominates the
    prior by orders of magnitude, leaving a condition number of order 1e3 and
    forcing NUTS to tree depth 7 even with an ideal mass matrix. Whitening
    against the likelihood instead brings it to order 10.

    ``A`` is only a change of coordinates -- the sampled posterior is
    unchanged -- so approximating a non-separable ``counts`` by its
    frequency profile costs accuracy in the preconditioner, never in the
    result.
    """
    if basis_eig_frequency.ndim == 3:
        basis_eig_frequency = basis_eig_frequency.mean(axis=0)
    gram_time = basis_eig_time.T @ basis_eig_time
    gram_frequency = basis_eig_frequency.T @ (
        counts_per_frequency[:, None] * basis_eig_frequency
    )
    hessian = 0.5 * np.kron(gram_time, gram_frequency)
    hessian[np.diag_indices_from(hessian)] += (
        1.0 / np.asarray(coefficient_scale, dtype=float).ravel() ** 2
    )
    cholesky = np.linalg.cholesky(hessian)
    return sla.solve_triangular(
        cholesky.T, np.eye(cholesky.shape[0]), lower=False
    )


def _sample_preconditioned_coefficients(name, preconditioner, coefficient_scale, shape):
    """Sample spline coefficients in isotropic coordinates, exact prior kept.

    ``z`` carries a unit-normal reference measure; the factor replaces it with
    the model's actual ``Normal(0, coefficient_scale)`` prior on the
    coefficients. The Jacobian of the fixed linear map is constant and so
    drops out of the posterior.
    """
    n_weights = int(np.prod(shape))
    z = numpyro.sample(name, dist.Normal(0.0, 1.0).expand([n_weights]).to_event(1))
    coefficients = preconditioner @ z
    scale_flat = jnp.reshape(coefficient_scale, (-1,))
    numpyro.factor(
        f"{name}_prior",
        -0.5 * jnp.sum((coefficients / scale_flat) ** 2) + 0.5 * jnp.sum(z**2),
    )
    return coefficients.reshape(shape)


def _log_spline_surface(basis_eig_time, basis_eig_frequency, coefficients):
    """Evaluate ``B_t W B_f^T``, allowing a per-time-row frequency basis.

    A 2-D ``basis_eig_frequency`` is the usual separable tensor product. A 3-D
    ``(n_time, n_frequency, n_basis)`` basis comes from evaluating the same
    frequency spline at a time-dependent warped frequency, which lets one set
    of coefficients describe a spectral feature that translates in frequency.
    """
    if basis_eig_frequency.ndim == 2:
        return basis_eig_time @ coefficients @ basis_eig_frequency.T
    return jnp.einsum(
        "ta,tfb,ab->tf", basis_eig_time, basis_eig_frequency, coefficients
    )


def diagonal_aet_model(
    summed_power,
    counts,
    basis_eig_time,
    basis_eig_frequency,
    coefficient_scale,
    preconditioner,
    log_noise_reference,
    galactic_template,
    frequency_hz,
    reference_f_knee_hz,
    config,
    share_noise_residual=False,
    delta_log_sd=0.3,
    gamma=1680.0,
):
    """Three scalar Whittle likelihoods with shared Galactic parameters.

    ``share_noise_residual`` switches the noise term from a free spline
    residual *per channel* to ONE spline residual shared across A/E/T plus a
    per-channel scalar offset ``delta_c``. The per-channel version measurably
    absorbs the Galactic amplitude direction into its own flexibility (R^2 up
    to 0.998 against the Galactic component, see the module docstring); the
    shared form has far less freedom to do that, and lets the T (null)
    channel constrain the shared instrument shape for A and E.
    """
    log_amplitude = numpyro.sample("log_amplitude", dist.Normal(0.0, 0.5))
    log_f_knee = numpyro.sample(
        "log_f_knee", dist.Normal(jnp.log(reference_f_knee_hz), 0.25)
    )
    f_knee = jnp.exp(log_f_knee)
    log_knee_ratio = _stable_log_knee_ratio(
        frequency_hz[None, :], f_knee, reference_f_knee_hz, gamma
    )

    if share_noise_residual:
        shared_eig_coefficients = _sample_preconditioned_coefficients(
            "z_shared",
            preconditioner,
            coefficient_scale,
            (basis_eig_time.shape[1], basis_eig_frequency.shape[-1]),
        )
        shared_log_spline_residual = _log_spline_surface(
            basis_eig_time, basis_eig_frequency, shared_eig_coefficients
        )

    log_likelihood = 0.0
    for channel_index, channel_name in enumerate(AET_CHANNELS):
        if share_noise_residual:
            delta = numpyro.sample(
                f"delta_{channel_name}", dist.Normal(0.0, delta_log_sd)
            )
            log_noise = (
                log_noise_reference[channel_index]
                + delta
                + shared_log_spline_residual
            )
        else:
            eig_coefficients = _sample_preconditioned_coefficients(
                f"z_{channel_name}",
                preconditioner,
                coefficient_scale,
                (basis_eig_time.shape[1], basis_eig_frequency.shape[-1]),
            )
            log_spline_residual = _log_spline_surface(
                basis_eig_time, basis_eig_frequency, eig_coefficients
            )
            log_noise = log_noise_reference[channel_index] + log_spline_residual
        log_galactic = (
            jnp.log(galactic_template[channel_index])
            + log_amplitude
            + log_knee_ratio
        )
        log_total = jnp.logaddexp(log_noise, log_galactic)
        log_likelihood = log_likelihood - 0.5 * jnp.sum(
            counts[channel_index] * log_total
            + summed_power[channel_index] * jnp.exp(-log_total)
        )

    numpyro.deterministic("log_likelihood", log_likelihood)
    numpyro.factor("diagonal_aet_whittle", log_likelihood)


def component_noise_preconditioner(
    basis: np.ndarray,
    transfer_tm: np.ndarray,
    transfer_oms: np.ndarray,
    spectrum_tm: np.ndarray,
    spectrum_oms: np.ndarray,
    galactic_template: np.ndarray,
    counts: np.ndarray,
    penalty: np.ndarray,
    phi_tm: float,
    phi_oms: float,
) -> np.ndarray:
    """Return ``A`` mapping isotropic ``z`` to ``[lam_tm, lam_oms]`` offsets.

    Whitening against the prior ``phi P`` alone is the wrong target whenever
    the likelihood dominates it, and here the two blocks sit on OPPOSITE sides
    of that line: with the shipped defaults the Whittle Fisher is ~1.7e-4 of
    the prior for TM (``phi_tm = 1e8`` pins it) but ~62x the prior for OMS
    (``phi_oms = 1e4`` is deliberately loose). Prior-only whitening is
    therefore right for one block and wrong for the other, leaving a
    preconditioned condition number ~1.1e4 -- about 100 leapfrog steps per
    iteration, tree depth 7, where 1 would do.

    Unlike ``coefficient_preconditioner``, the Fisher here is NOT parameter
    independent: ``d log S_total / d lam_i = (S_i / S_total) b(f)`` depends on
    the component fractions. It is evaluated once at the prior centre (the
    analytic spectra), which the fit only departs from by a few percent. That
    makes this an approximate metric rather than an exact one -- but it is
    still an exact change of coordinates, so the posterior is unchanged
    regardless of how good the approximation is.
    """
    n_basis = basis.shape[1]
    fisher = np.zeros((2 * n_basis, 2 * n_basis))
    for channel in range(transfer_tm.shape[0]):
        tm_part = transfer_tm[channel] * spectrum_tm[None, :]
        oms_part = transfer_oms[channel] * spectrum_oms[None, :]
        total = tm_part + oms_part + galactic_template[channel]
        weight = counts[channel]
        fraction_tm = tm_part / total
        fraction_oms = oms_part / total
        for first, second, row, column in (
            (fraction_tm, fraction_tm, 0, 0),
            (fraction_tm, fraction_oms, 0, n_basis),
            (fraction_oms, fraction_oms, n_basis, n_basis),
        ):
            pooled = np.sum(weight * first * second, axis=0)
            fisher[row:row + n_basis, column:column + n_basis] += 0.5 * (
                basis.T @ (pooled[:, None] * basis)
            )
    fisher[n_basis:, :n_basis] = fisher[:n_basis, n_basis:].T

    hessian = fisher.copy()
    hessian[:n_basis, :n_basis] += phi_tm * penalty
    hessian[n_basis:, n_basis:] += phi_oms * penalty
    cholesky = np.linalg.cholesky(hessian)
    return sla.solve_triangular(
        cholesky.T, np.eye(2 * n_basis), lower=False
    )


def penalized_spline_prior_centre(
    basis: np.ndarray, log_target: np.ndarray, penalty: np.ndarray, ridge: float = 1e-6
) -> np.ndarray:
    """Penalized least-squares fit of ``basis @ lam`` to ``log_target``.

    Gives the prior mean ``lam_loc`` that centres the spline on the analytic
    instrument spectrum, following Nazeela et al.
    """
    normal = basis.T @ basis + ridge * penalty * float(np.trace(basis.T @ basis))
    return np.linalg.solve(normal, basis.T @ log_target)


def component_noise_model(
    summed_power,
    counts,
    basis,
    transfer_tm,
    transfer_oms,
    lam_loc_tm,
    lam_loc_oms,
    prior_cholesky_inverse,
    penalty,
    phi_tm,
    phi_oms,
    galactic_template,
    frequency_hz,
    reference_f_knee_hz,
    t_leakage_log_centre=None,
    t_leakage_log_sd=2.0,
    gamma=1680.0,
):
    r"""Physical-component AET noise model with a parametric Galaxy.

    The instrument noise is built from the two *physical* noise sources,

        S_noise,c(t,f) = T_TM,c(t,f) S_TM(f) + T_OMS,c(t,f) S_OMS(f),

    where the transfer functions ``T`` are computed from the orbits and carry
    ALL of the time dependence -- including the TDI nulls at ``n/(2L(t))``,
    which drift as the armlength drifts. The free parameters are two
    log-P-splines in frequency only, ``log S_i(f) = B lam_i``, with
    ``lam_i ~ N(lam_loc_i, (phi_i P_i)^-1)`` centred on the analytic
    instrument spectra.

    This is exact, not an approximation: the analytic AET surface factorizes
    into (time-varying transfer) x (stationary spectrum) to machine precision.
    Consequently the noise cannot manufacture arbitrary time structure, so the
    Galaxy's annual modulation is the only free time-varying component and the
    noise/Galaxy degeneracy is broken by construction rather than by a
    roughness prior.

    ``t_leakage_log_centre`` (if not None) adds one free parameter for
    imperfect test-mass cancellation in the null channel::

        S_noise,T = (T_TM,T + kappa T_TM,A) S_TM + T_OMS,T S_OMS

    T is a near-null, sitting ~7000x below A, so its predicted level is a
    small residual of large cancelling terms and a sub-percent modelling
    error becomes an order-unity error there. Measured against the archive's
    simulated noise, T carries ~10-23% more power than the analytic model
    below 1 mHz, and the excess tracks T's own test-mass contribution
    (excess/TM_T = 0.23, 0.19, 0.11 across the low bands, while the OMS- and
    A-referenced ratios vary by orders of magnitude). Expressed as a fraction
    of the A-channel test-mass response that fails to cancel, that is
    kappa ~ 2e-5 to 5e-5. Fitting kappa keeps those cells in the likelihood
    with their uncertainty propagated, rather than discarding them.
    """
    log_amplitude = numpyro.sample("log_amplitude", dist.Normal(0.0, 0.5))
    log_f_knee = numpyro.sample(
        "log_f_knee", dist.Normal(jnp.log(reference_f_knee_hz), 0.25)
    )
    log_knee_ratio = _stable_log_knee_ratio(
        frequency_hz[None, :], jnp.exp(log_f_knee), reference_f_knee_hz, gamma
    )

    n_basis = basis.shape[1]
    # One joint z for both blocks: the TM/OMS cross term in the Fisher is not
    # negligible, so whitening them separately would leave that correlation.
    z = numpyro.sample(
        "z_noise", dist.Normal(0.0, 1.0).expand([2 * n_basis]).to_event(1)
    )
    offset = prior_cholesky_inverse @ z
    lam_tm = lam_loc_tm + offset[:n_basis]
    lam_oms = lam_loc_oms + offset[n_basis:]
    numpyro.deterministic("lam_tm", lam_tm)
    numpyro.deterministic("lam_oms", lam_oms)
    # z carries a unit-normal reference measure; this factor replaces it with
    # the model's actual N(lam_loc, (phi P)^-1) prior on each block. The
    # Jacobian of the fixed linear map is constant and drops out.
    numpyro.factor(
        "noise_spline_prior",
        -0.5
        * (
            phi_tm * (offset[:n_basis] @ (penalty @ offset[:n_basis]))
            + phi_oms * (offset[n_basis:] @ (penalty @ offset[n_basis:]))
        )
        + 0.5 * jnp.sum(z**2),
    )

    spectrum_tm = jnp.exp(basis @ lam_tm)
    spectrum_oms = jnp.exp(basis @ lam_oms)

    t_index = AET_CHANNELS.index("T")
    a_index = AET_CHANNELS.index("A")
    if t_leakage_log_centre is not None:
        # kappa(f) = kappa0 (f / 1 mHz)^slope. The measured residual drifts
        # (kappa ~ 2e-5, 3.1e-5, 5.4e-5 at 0.2, 0.65, 2 mHz), so a single
        # scalar corrects the mean offset but misfits the shape.
        log_kappa0 = numpyro.sample(
            "log_t_leakage", dist.Normal(t_leakage_log_centre, t_leakage_log_sd)
        )
        leakage_slope = numpyro.sample("t_leakage_slope", dist.Normal(0.43, 0.5))
        kappa = jnp.exp(log_kappa0) * (frequency_hz / 1.0e-3) ** leakage_slope

    log_likelihood = 0.0
    for channel_index in range(len(AET_CHANNELS)):
        tm_transfer = transfer_tm[channel_index]
        if t_leakage_log_centre is not None and channel_index == t_index:
            tm_transfer = tm_transfer + kappa[None, :] * transfer_tm[a_index]
        noise = (
            tm_transfer * spectrum_tm[None, :]
            + transfer_oms[channel_index] * spectrum_oms[None, :]
        )
        log_galactic = (
            jnp.log(galactic_template[channel_index]) + log_amplitude + log_knee_ratio
        )
        log_total = jnp.logaddexp(jnp.log(noise), log_galactic)
        log_likelihood = log_likelihood - 0.5 * jnp.sum(
            counts[channel_index] * log_total
            + summed_power[channel_index] * jnp.exp(-log_total)
        )

    numpyro.deterministic("log_likelihood", log_likelihood)
    numpyro.factor("component_aet_whittle", log_likelihood)


def _diagnostics(mcmc: MCMC, max_tree_depth: int) -> dict[str, float | int]:
    grouped = mcmc.get_samples(group_by_chain=True)
    diagnostics = summary(grouped, group_by_chain=True)
    rhat_values: list[float] = []
    ess_values: list[float] = []
    tracked_sites = ["log_amplitude", "log_f_knee"]
    if "z_noise" in diagnostics:  # physical-component model
        tracked_sites.append("z_noise")
        if "log_t_leakage" in diagnostics:
            tracked_sites += ["log_t_leakage", "t_leakage_slope"]
    elif "z_shared" in diagnostics:
        tracked_sites.append("z_shared")
        tracked_sites += [f"delta_{channel_name}" for channel_name in AET_CHANNELS]
    else:
        tracked_sites += [f"z_{channel_name}" for channel_name in AET_CHANNELS]
    for site in tracked_sites:
        values = diagnostics[site]
        rhat_values.extend(np.ravel(np.asarray(values["r_hat"], dtype=float)))
        ess_values.extend(np.ravel(np.asarray(values["n_eff"], dtype=float)))
    extra = mcmc.get_extra_fields(group_by_chain=True)
    steps = np.asarray(extra["num_steps"])
    energy = np.asarray(extra["potential_energy"])
    bfmi = []
    for chain_energy in energy:
        variance = np.var(chain_energy)
        bfmi.append(
            np.nan
            if variance == 0.0
            else float(np.mean(np.diff(chain_energy) ** 2) / variance)
        )
    return {
        "num_chains": int(grouped["log_amplitude"].shape[0]),
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "max_r_hat": float(np.nanmax(rhat_values)),
        "min_effective_sample_size": float(np.nanmin(ess_values)),
        "mean_accept_probability": float(np.mean(np.asarray(extra["accept_prob"]))),
        "max_num_steps": int(np.max(steps)),
        "tree_depth_saturation_fraction": float(
            np.mean(steps >= 2**max_tree_depth - 1)
        ),
        "min_ebfmi": float(np.nanmin(bfmi)),
    }


def _intervals(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.median(draws, axis=0),
        np.quantile(draws, 0.05, axis=0),
        np.quantile(draws, 0.95, axis=0),
    )


def fit_aet_diagonal_nuts(
    observed_psd: np.ndarray,
    counts: np.ndarray,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    oms_reference_psd: np.ndarray,
    tm_reference_psd: np.ndarray,
    galactic_template_psd: np.ndarray,
    reference_f_knee_hz: float = 2.15e-3,
    mask: np.ndarray | None = None,
    n_time_knots: int = 2,
    n_frequency_knots: int = 8,
    warped_frequency_hz: np.ndarray | None = None,
    share_noise_residual: bool = False,
    delta_log_sd: float = 0.3,
    phi_time: float = 100.0,
    phi_frequency: float = 100.0,
    noise_level_log_sd: float = 5.0,
    n_warmup: int = 300,
    n_samples: int = 300,
    num_chains: int = 2,
    random_seed: int = 20260811,
    target_accept_probability: float = 0.95,
    max_tree_depth: int = 10,
    progress_bar: bool = True,
) -> AETDiagonalPosterior:
    """Fit the diagonal-AET free-noise-spline component model.

    PSD surfaces have shape ``(3, n_time, n_frequency)`` in A/E/T order.
    ``oms_reference_psd`` and ``tm_reference_psd`` each have that shape: the
    analytic OMS and test-mass noise components (same units as
    ``observed_psd``). Their **sum** is a fixed multiplicative offset, and the
    free log-P-spline models the residual on top of it. This puts the OMS/TM
    time-frequency *shape* into the likelihood -- more than a single weak
    scalar level -- but the spline retains full freedom to depart from it,
    including rescaling it, so nothing about the noise is actually fixed.

    The amplitudes are deliberately **not** sampled. Their direction
    ``d log S / d log a = M / (OMS + TM)`` is reproduced by the frequency
    spline to one part in 1e11 (R^2 = 1.00000000), so sampling them adds
    ridges ~1e5 times longer than wide that no per-block preconditioner can
    remove. Centering the spline's prior on the correct shape is what buys
    the accuracy; free amplitudes only buy a degeneracy.

    ``share_noise_residual`` replaces the per-channel free spline with ONE
    spline shared across A/E/T (static in time except through the frequency
    warp) plus a per-channel scalar offset ``delta_c`` (prior SD
    ``delta_log_sd``). The default per-channel spline is measured to absorb
    up to 99.8% of the Galactic amplitude direction (R^2 = 0.998, unexplained
    variance 0.2%), meaning A_gal is then constrained mostly by the roughness
    prior, not the data -- and it costs 308 parameters per channel to do it.
    Sharing removes almost all of that freedom, cuts the parameter count
    roughly 20x, and lets the T (Galaxy-null) channel directly inform the
    instrument shape for A and E.

    ``warped_frequency_hz`` optionally supplies a ``(n_time, n_frequency)``
    time-dependent frequency coordinate at which the frequency spline is
    evaluated, in place of the fixed grid.  The intended choice is
    ``frequency_hz * L(t) / mean(L)`` from the orbit file: the TDI transfer
    minima sit at ``1 / (2 L(t))`` and therefore drift in frequency over the
    year, which a separable ``B_t W B_f^T`` surface can only follow at very
    high rank in both directions.  Warping by the known armlength makes those
    features stationary in the warped coordinate, so the time direction only
    has to describe what is genuinely slowly varying.  This supplies orbit
    geometry, not a noise PSD; no OMS/TM surface enters the likelihood.

    ``noise_level_log_sd`` is the standard deviation of the
    unpenalized log-spline null space, so a value of one permits order-unity
    multiplicative departures from those scalar levels. ``phi_time`` and
    ``phi_frequency`` are fixed roughness precisions; there are no sampled
    spline-scale hyperparameters.
    """
    observed = np.asarray(observed_psd, dtype=float)
    cell_counts = np.asarray(counts, dtype=float)
    oms_reference = np.asarray(oms_reference_psd, dtype=float)
    tm_reference = np.asarray(tm_reference_psd, dtype=float)
    galactic_template = np.asarray(galactic_template_psd, dtype=float)
    time = np.asarray(time_tcb, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    if observed.ndim != 3 or observed.shape[0] != len(AET_CHANNELS):
        raise ValueError("observed_psd must have shape (3, time, frequency)")
    if any(
        array.shape != observed.shape
        for array in (cell_counts, galactic_template, oms_reference, tm_reference)
    ):
        raise ValueError("all AET surfaces must share shape")
    if time.shape != (observed.shape[1],) or frequency.shape != (observed.shape[2],):
        raise ValueError("time/frequency coordinates do not match the AET surfaces")
    if np.any(np.diff(time) <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("time and frequency coordinates must be strictly increasing")
    retained = cell_counts > 0.0
    if mask is not None:
        supplied_mask = np.asarray(mask, dtype=bool)
        if supplied_mask.shape != observed.shape:
            raise ValueError("mask must have the same shape as observed_psd")
        retained &= supplied_mask
    if not np.all(np.any(retained, axis=(1, 2))):
        raise ValueError("each AET channel must retain at least one cell")
    for name, array, positive in (
        ("observed_psd", observed, True),
        ("counts", cell_counts, True),
        ("galactic_template_psd", galactic_template, False),
        ("oms_reference_psd", oms_reference, True),
        ("tm_reference_psd", tm_reference, True),
    ):
        selected = array[retained]
        invalid = np.any(~np.isfinite(selected))
        invalid |= np.any(selected <= 0.0) if positive else np.any(selected < 0.0)
        if invalid:
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"retained {name} values must be finite and {qualifier}")
    if num_chains < 2:
        raise ValueError("AET posterior inference requires at least two chains")
    if min(phi_time, phi_frequency, noise_level_log_sd) <= 0.0:
        raise ValueError("fixed precisions and noise_level_log_sd must be positive")

    # Channel-specific numerical scales do not encode physical information.
    # They cancel when the component draws are restored to physical units.
    data_scale = np.asarray(
        [np.median(observed[c][retained[c]]) for c in range(len(AET_CHANNELS))]
    )
    scaled_observed = observed / data_scale[:, None, None]
    scaled_noise_reference = (oms_reference + tm_reference) / data_scale[:, None, None]
    scaled_template = galactic_template / data_scale[:, None, None]
    counts_fit = np.where(retained, cell_counts, 0.0)
    summed_power = counts_fit * np.where(retained, scaled_observed, 1.0)
    positive_template = scaled_template[retained & (scaled_template > 0.0)]
    template_floor = (
        float(np.min(positive_template)) * 1.0e-6
        if positive_template.size
        else np.finfo(float).tiny
    )
    template_fit = np.where(
        retained, np.maximum(scaled_template, template_floor), template_floor
    )

    time_unit = (time - time[0]) / (time[-1] - time[0])
    log_frequency = np.log(frequency)
    frequency_unit = (log_frequency - log_frequency[0]) / (
        log_frequency[-1] - log_frequency[0]
    )
    if share_noise_residual:
        # Genuinely static except through the frequency warp: one column, no
        # B-spline time basis at all, so it has no way to track anything
        # channel-specific or time-varying beyond what the warp supplies.
        basis_time = np.ones((time.size, 1))
    else:
        basis_time, _ = create_bspline_basis(time_unit, n_time_knots, degree=3)
    basis_frequency, frequency_knots = create_bspline_basis(
        frequency_unit, n_frequency_knots, degree=3
    )
    if warped_frequency_hz is not None:
        warped = np.asarray(warped_frequency_hz, dtype=float)
        if warped.shape != observed.shape[1:]:
            raise ValueError("warped_frequency_hz must have shape (time, frequency)")
        if np.any(~np.isfinite(warped)) or np.any(warped <= 0.0):
            raise ValueError("warped_frequency_hz values must be finite and positive")
        # Same normalization as the grid, so the knots are shared; the warp
        # only moves evaluation points, and is clipped at the band edges.
        warped_unit = (np.log(warped) - log_frequency[0]) / (
            log_frequency[-1] - log_frequency[0]
        )
        basis_frequency = evaluate_bspline_basis(
            np.clip(warped_unit, 0.0, 1.0).ravel(), frequency_knots, degree=3
        ).reshape(*warped.shape, basis_frequency.shape[1])
    penalty_time = (
        np.zeros((1, 1))
        if share_noise_residual
        else create_difference_penalty_matrix(basis_time.shape[1], diff_order=2)
    )
    penalty_frequency = create_difference_penalty_matrix(
        basis_frequency.shape[-1], diff_order=2
    )
    whitened = whiten_penalty_pair(penalty_time, penalty_frequency)
    basis_eig_time = basis_time @ whitened["U_time"]
    basis_eig_frequency = basis_frequency @ whitened["U_freq"]
    config = PSplineConfig(
        n_interior_knots_time=0 if share_noise_residual else n_time_knots,
        n_interior_knots_freq=n_frequency_knots,
        freq_knot_strategy="log",
        centered=True,
        trim_time_bins=0,
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        null_precision=1.0 / noise_level_log_sd**2,
        ridge_eps=1.0e-4,
    )
    coefficient_scale = eigen_prior_scale(
        jnp.asarray(phi_time),
        jnp.asarray(phi_frequency),
        jnp.asarray(whitened["lam_time"]),
        jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]),
        config,
    )

    # Preconditioner: whiten against the likelihood, not the penalty. counts
    # vary only with frequency here, so its frequency profile is exact. A
    # shared residual is driven by all three channels' likelihoods at once,
    # so its Fisher information sums their counts rather than averaging.
    counts_per_frequency = (
        counts_fit.sum(axis=0).mean(axis=0)
        if share_noise_residual
        else counts_fit.mean(axis=(0, 1))
    )
    preconditioner = coefficient_preconditioner(
        basis_eig_time,
        basis_eig_frequency,
        counts_per_frequency,
        np.asarray(coefficient_scale),
    )
    inverse_preconditioner = np.linalg.inv(preconditioner)

    coefficient_count = basis_time.shape[1] * basis_frequency.shape[-1]
    # Start the free spline near a data-informed decomposition instead of at a
    # flat scalar level. This changes only NUTS initialization, not the model:
    # the target subtracts the unit-amplitude Galactic template and floors the
    # remainder at 5% of the observed total before a smooth least-squares
    # projection onto the spline basis.
    design_subscripts = (
        "ti,fj->tfij" if basis_eig_frequency.ndim == 2 else "ti,tfj->tfij"
    )
    spline_design = np.einsum(
        design_subscripts, basis_eig_time, basis_eig_frequency, optimize=True
    ).reshape(-1, coefficient_count)
    initial_noise_scaled = np.maximum(
        scaled_observed - scaled_template, 0.05 * scaled_observed
    )
    init_values = {
        "log_amplitude": np.asarray(0.0),
        "log_f_knee": np.asarray(np.log(reference_f_knee_hz)),
    }
    channel_targets = []
    for channel_index, channel in enumerate(AET_CHANNELS):
        selected = retained[channel_index].ravel()
        reference_at_unit_amplitude = (
            scaled_noise_reference[channel_index]
        ).ravel()[selected]
        target = np.log(
            initial_noise_scaled[channel_index].ravel()[selected]
            / reference_at_unit_amplitude
        )
        channel_targets.append((selected, np.clip(target, -20.0, 20.0)))
        if not share_noise_residual:
            spline_coefficients, *_ = np.linalg.lstsq(
                spline_design[selected], channel_targets[-1][1], rcond=1.0e-8
            )
            init_values[f"z_{channel}"] = inverse_preconditioner @ spline_coefficients
    if share_noise_residual:
        # Shared spline starts from the channel-averaged residual target; the
        # per-channel offsets that would explain any remaining channel-level
        # difference start at zero and let the sampler find them.
        pooled_target = np.concatenate([target for _, target in channel_targets])
        pooled_design = np.concatenate(
            [spline_design[selected] for selected, _ in channel_targets]
        )
        spline_coefficients, *_ = np.linalg.lstsq(
            pooled_design, pooled_target, rcond=1.0e-8
        )
        init_values["z_shared"] = inverse_preconditioner @ spline_coefficients
        for channel in AET_CHANNELS:
            init_values[f"delta_{channel}"] = np.asarray(0.0)
    model_args = (
        jnp.asarray(summed_power),
        jnp.asarray(counts_fit),
        jnp.asarray(basis_eig_time),
        jnp.asarray(basis_eig_frequency),
        jnp.asarray(coefficient_scale),
        jnp.asarray(preconditioner),
        jnp.asarray(np.log(scaled_noise_reference)),
        jnp.asarray(template_fit),
        jnp.asarray(frequency),
        reference_f_knee_hz,
        config,
        share_noise_residual,
        delta_log_sd,
    )
    kernel = NUTS(
        diagonal_aet_model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=target_accept_probability,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
    )
    start = walltime.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(random_seed),
        *model_args,
        extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy"),
    )
    runtime_seconds = walltime.perf_counter() - start
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}

    amplitude_draws = np.exp(samples["log_amplitude"])
    f_knee_draws = np.exp(samples["log_f_knee"])
    log_knee_ratio = np.asarray(
        _stable_log_knee_ratio(
            jnp.asarray(frequency)[None, :],
            jnp.asarray(f_knee_draws)[:, None],
            reference_f_knee_hz,
            1680.0,
        )
    )
    def reconstruct_log_spline(coefficient_draws: np.ndarray) -> np.ndarray:
        eigen_draws = (coefficient_draws @ preconditioner.T).reshape(
            -1, basis_time.shape[1], basis_frequency.shape[-1]
        )
        return np.einsum(
            "ti,sij,fj->stf" if basis_eig_frequency.ndim == 2 else "ti,sij,tfj->stf",
            basis_eig_time,
            eigen_draws,
            basis_eig_frequency,
            optimize=True,
        )

    if share_noise_residual:
        shared_log_spline = reconstruct_log_spline(samples["z_shared"])

    noise_draws_by_channel = []
    galactic_draws_by_channel = []
    total_draws_by_channel = []
    for channel_index, channel_name in enumerate(AET_CHANNELS):
        if share_noise_residual:
            delta_draws = samples[f"delta_{channel_name}"]
            log_spline = delta_draws[:, None, None] + shared_log_spline
        else:
            log_spline = reconstruct_log_spline(samples[f"z_{channel_name}"])
        noise_draws = (
            data_scale[channel_index]
            * scaled_noise_reference[channel_index][None, :, :]
            * np.exp(log_spline)
        )
        galactic_draws = (
            data_scale[channel_index]
            * template_fit[channel_index][None, :, :]
            * amplitude_draws[:, None, None]
            * np.exp(log_knee_ratio[:, None, :])
        )
        noise_draws_by_channel.append(_intervals(noise_draws))
        galactic_draws_by_channel.append(_intervals(galactic_draws))
        total_draws_by_channel.append(_intervals(noise_draws + galactic_draws))

    def stack_intervals(channel_intervals, interval_index):
        return np.stack([item[interval_index] for item in channel_intervals])

    return AETDiagonalPosterior(
        noise_median=stack_intervals(noise_draws_by_channel, 0),
        noise_lower=stack_intervals(noise_draws_by_channel, 1),
        noise_upper=stack_intervals(noise_draws_by_channel, 2),
        galactic_median=stack_intervals(galactic_draws_by_channel, 0),
        galactic_lower=stack_intervals(galactic_draws_by_channel, 1),
        galactic_upper=stack_intervals(galactic_draws_by_channel, 2),
        total_median=stack_intervals(total_draws_by_channel, 0),
        total_lower=stack_intervals(total_draws_by_channel, 1),
        total_upper=stack_intervals(total_draws_by_channel, 2),
        amplitude_draws=amplitude_draws,
        f_knee_draws_hz=f_knee_draws,
        diagnostics=_diagnostics(mcmc, max_tree_depth),
        samples=samples,
        mcmc=mcmc,
        runtime_seconds=runtime_seconds,
        phi_time=float(phi_time),
        phi_frequency=float(phi_frequency),
        noise_level_log_sd=float(noise_level_log_sd),
    )


def fit_aet_component_noise_nuts(
    observed_psd: np.ndarray,
    counts: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    transfer_tm: np.ndarray,
    transfer_oms: np.ndarray,
    tm_theory_psd: np.ndarray,
    oms_theory_psd: np.ndarray,
    galactic_template_psd: np.ndarray,
    reference_f_knee_hz: float = 2.15e-3,
    mask: np.ndarray | None = None,
    n_frequency_knots: int = 12,
    phi_tm: float = 1.0e8,
    phi_oms: float = 1.0e4,
    fit_t_null_leakage: bool = True,
    t_leakage_centre: float = 3.0e-5,
    t_leakage_log_sd: float = 2.0,
    n_warmup: int = 500,
    n_samples: int = 500,
    num_chains: int = 2,
    random_seed: int = 20260811,
    target_accept_probability: float = 0.9,
    max_tree_depth: int = 10,
    progress_bar: bool = True,
) -> AETDiagonalPosterior:
    """Fit the physical-component AET noise model (M1 v3).

    Instead of a free log-P-spline *surface* per channel, the noise is built
    from the two physical sources through the known TDI transfer functions::

        S_noise,c(t,f) = transfer_tm[c](t,f) * S_TM(f)
                       + transfer_oms[c](t,f) * S_OMS(f)

    ``transfer_tm`` / ``transfer_oms`` have shape ``(3, n_time, n_frequency)``
    and must be computed from the orbits on the analysis grid, so they resolve
    the drifting TDI nulls exactly; they carry all of the time dependence.
    ``tm_theory_psd`` / ``oms_theory_psd`` are ``(n_frequency,)`` analytic
    instrument spectra that centre each spline's prior (Bayle & Hartwig
    instrument specifications), following Nazeela et al.

    ``phi_tm`` defaults far tighter than ``phi_oms`` on purpose: the T channel
    is ~27x less sensitive to test-mass than to readout noise, so S_TM is the
    component that a low-frequency stochastic signal can imitate. Loosening
    ``phi_tm`` makes the model more agnostic at the cost of that degeneracy.
    """
    observed = np.asarray(observed_psd, dtype=float)
    cell_counts = np.asarray(counts, dtype=float)
    transfer_tm = np.asarray(transfer_tm, dtype=float)
    transfer_oms = np.asarray(transfer_oms, dtype=float)
    galactic_template = np.asarray(galactic_template_psd, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    tm_theory = np.asarray(tm_theory_psd, dtype=float)
    oms_theory = np.asarray(oms_theory_psd, dtype=float)
    if observed.ndim != 3 or observed.shape[0] != len(AET_CHANNELS):
        raise ValueError("observed_psd must have shape (3, time, frequency)")
    if any(
        array.shape != observed.shape
        for array in (cell_counts, galactic_template, transfer_tm, transfer_oms)
    ):
        raise ValueError("all AET surfaces must share shape")
    if tm_theory.shape != (observed.shape[2],) or oms_theory.shape != (observed.shape[2],):
        raise ValueError("theory spectra must have shape (n_frequency,)")
    if frequency.shape != (observed.shape[2],):
        raise ValueError("frequency does not match the AET surfaces")
    if np.any(np.diff(frequency) <= 0.0):
        raise ValueError("frequency must be strictly increasing")
    if num_chains < 2:
        raise ValueError("AET posterior inference requires at least two chains")
    if min(phi_tm, phi_oms) <= 0.0:
        raise ValueError("phi_tm and phi_oms must be positive")
    retained = cell_counts > 0.0
    if mask is not None:
        retained &= np.asarray(mask, dtype=bool)
    if not np.all(np.any(retained, axis=(1, 2))):
        raise ValueError("each AET channel must retain at least one cell")
    for name, array in (
        ("observed_psd", observed),
        ("transfer_tm", transfer_tm),
        ("transfer_oms", transfer_oms),
    ):
        selected = array[retained]
        if np.any(~np.isfinite(selected)) or np.any(selected <= 0.0):
            raise ValueError(f"retained {name} values must be finite and positive")
    if np.any(~np.isfinite(tm_theory)) or np.any(tm_theory <= 0.0):
        raise ValueError("tm_theory_psd must be finite and positive")
    if np.any(~np.isfinite(oms_theory)) or np.any(oms_theory <= 0.0):
        raise ValueError("oms_theory_psd must be finite and positive")

    counts_fit = np.where(retained, cell_counts, 0.0)
    summed_power = counts_fit * np.where(retained, observed, 1.0)
    positive_template = galactic_template[retained & (galactic_template > 0.0)]
    template_floor = (
        float(np.min(positive_template)) * 1.0e-6
        if positive_template.size
        else np.finfo(float).tiny
    )
    template_fit = np.where(
        retained, np.maximum(galactic_template, template_floor), template_floor
    )

    # Log-spaced knots, as in Nazeela et al.
    log_frequency = np.log(frequency)
    frequency_unit = (log_frequency - log_frequency[0]) / (
        log_frequency[-1] - log_frequency[0]
    )
    basis, _ = create_bspline_basis(frequency_unit, n_frequency_knots, degree=3)
    penalty = create_difference_penalty_matrix(basis.shape[1], diff_order=2)
    # Full-rank penalty: the difference penalty alone leaves the low-order
    # polynomial null space unconstrained, so the prior would be improper.
    penalty = penalty + 1.0e-6 * np.eye(basis.shape[1]) * float(np.trace(penalty)) / basis.shape[1]
    lam_loc_tm = penalized_spline_prior_centre(basis, np.log(tm_theory), penalty)
    lam_loc_oms = penalized_spline_prior_centre(basis, np.log(oms_theory), penalty)
    # Whiten against prior AND likelihood: the two blocks sit on opposite
    # sides of the prior/likelihood balance (TM prior-pinned, OMS data-
    # dominated ~60x), so prior-only whitening leaves condition ~1e4.
    prior_cholesky_inverse = component_noise_preconditioner(
        basis,
        transfer_tm,
        transfer_oms,
        tm_theory,
        oms_theory,
        template_fit,
        counts_fit,
        penalty,
        phi_tm,
        phi_oms,
    )

    init_values = {
        "log_amplitude": np.asarray(0.0),
        "log_f_knee": np.asarray(np.log(reference_f_knee_hz)),
        "z_noise": np.zeros(2 * basis.shape[1]),
    }
    if fit_t_null_leakage:
        init_values["log_t_leakage"] = np.asarray(np.log(t_leakage_centre))
        init_values["t_leakage_slope"] = np.asarray(0.43)
    model_args = (
        jnp.asarray(summed_power),
        jnp.asarray(counts_fit),
        jnp.asarray(basis),
        jnp.asarray(transfer_tm),
        jnp.asarray(transfer_oms),
        jnp.asarray(lam_loc_tm),
        jnp.asarray(lam_loc_oms),
        jnp.asarray(prior_cholesky_inverse),
        jnp.asarray(penalty),
        phi_tm,
        phi_oms,
        jnp.asarray(template_fit),
        jnp.asarray(frequency),
        reference_f_knee_hz,
        float(np.log(t_leakage_centre)) if fit_t_null_leakage else None,
        t_leakage_log_sd,
    )
    kernel = NUTS(
        component_noise_model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=target_accept_probability,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
    )
    start = walltime.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(random_seed),
        *model_args,
        extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy"),
    )
    runtime_seconds = walltime.perf_counter() - start
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}

    amplitude_draws = np.exp(samples["log_amplitude"])
    f_knee_draws = np.exp(samples["log_f_knee"])
    log_knee_ratio = np.asarray(
        _stable_log_knee_ratio(
            jnp.asarray(frequency)[None, :],
            jnp.asarray(f_knee_draws)[:, None],
            reference_f_knee_hz,
            1680.0,
        )
    )
    spectrum_tm = np.exp(samples["lam_tm"] @ basis.T)
    spectrum_oms = np.exp(samples["lam_oms"] @ basis.T)

    t_index = AET_CHANNELS.index("T")
    a_index = AET_CHANNELS.index("A")
    kappa_draws = (
        np.exp(samples["log_t_leakage"])[:, None]
        * (frequency[None, :] / 1.0e-3) ** samples["t_leakage_slope"][:, None]
        if fit_t_null_leakage
        else None
    )
    noise_by_channel, galactic_by_channel, total_by_channel = [], [], []
    for channel_index in range(len(AET_CHANNELS)):
        tm_transfer = transfer_tm[channel_index][None, :, :]
        if fit_t_null_leakage and channel_index == t_index:
            tm_transfer = tm_transfer + (
                kappa_draws[:, None, :] * transfer_tm[a_index][None, :, :]
            )
        noise_draws = (
            tm_transfer * spectrum_tm[:, None, :]
            + transfer_oms[channel_index][None, :, :] * spectrum_oms[:, None, :]
        )
        galactic_draws = (
            template_fit[channel_index][None, :, :]
            * amplitude_draws[:, None, None]
            * np.exp(log_knee_ratio[:, None, :])
        )
        noise_by_channel.append(_intervals(noise_draws))
        galactic_by_channel.append(_intervals(galactic_draws))
        total_by_channel.append(_intervals(noise_draws + galactic_draws))

    def stack(channel_intervals, index):
        return np.stack([item[index] for item in channel_intervals])

    return AETDiagonalPosterior(
        noise_median=stack(noise_by_channel, 0),
        noise_lower=stack(noise_by_channel, 1),
        noise_upper=stack(noise_by_channel, 2),
        galactic_median=stack(galactic_by_channel, 0),
        galactic_lower=stack(galactic_by_channel, 1),
        galactic_upper=stack(galactic_by_channel, 2),
        total_median=stack(total_by_channel, 0),
        total_lower=stack(total_by_channel, 1),
        total_upper=stack(total_by_channel, 2),
        amplitude_draws=amplitude_draws,
        f_knee_draws_hz=f_knee_draws,
        diagnostics=_diagnostics(mcmc, max_tree_depth),
        samples=samples,
        mcmc=mcmc,
        runtime_seconds=runtime_seconds,
        phi_time=float(phi_tm),
        phi_frequency=float(phi_oms),
        noise_level_log_sd=float("nan"),
    )


__all__ = [
    "AETDiagonalPosterior",
    "component_noise_model",
    "diagonal_aet_model",
    "fit_aet_component_noise_nuts",
    "fit_aet_diagonal_nuts",
    "penalized_spline_prior_centre",
]
