"""Likelihood-aware preconditioner for the M1 physical-component noise model.

`component_noise_model` (aet_component_pspline_nuts.py) samples the two
frequency splines (S_TM, S_OMS) in coordinates that whiten the PRIOR
precision only (``prior_cholesky_inverse``). The Whittle likelihood's
curvature (``~counts/2`` per cell) is not accounted for -- exactly the
pathology already diagnosed and fixed for the tensor-spline model via
``coefficient_preconditioner`` (see its docstring in aet_component_pspline_nuts.py).

This module applies the same fix to the physical-component model: build each
spline's preconditioner from prior-plus-likelihood Fisher information
(evaluated at the analytic init point, since it is only a change of sampling
coordinates -- an approximate weight costs sampler efficiency, never
correctness), Cholesky it, and sample in those isotropic coordinates.

Everything else (data prep, posterior reconstruction, diagnostics) is reused
unmodified from aet_component_pspline_nuts.py.
"""

from __future__ import annotations

import time as walltime

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import scipy.linalg as sla

from aet_component_pspline_nuts import (
    AETDiagonalPosterior,
    _diagnostics,
    _intervals,
    penalized_spline_prior_centre,
)
from aet_diagonal import AET_CHANNELS
from component_pspline_nuts import _stable_log_knee_ratio
from tv_pspline_psd.splines import create_bspline_basis, create_difference_penalty_matrix


def _fisher_preconditioner(
    basis: np.ndarray, fisher_per_frequency: np.ndarray, phi: float, penalty: np.ndarray
) -> np.ndarray:
    """Cholesky preconditioner for ``lam`` under prior-plus-likelihood curvature.

    ``H = basis^T diag(fisher) basis + phi * penalty`` approximates the
    posterior precision of the log-spline coefficients; ``A = L^-T`` (``H = LL^T``)
    maps isotropic ``z`` to ``lam``-space with roughly unit posterior variance.
    """
    gram = basis.T @ (fisher_per_frequency[:, None] * basis)
    hessian = gram + phi * penalty
    cholesky = np.linalg.cholesky(hessian)
    return sla.solve_triangular(cholesky.T, np.eye(cholesky.shape[0]), lower=False)


def _sample_preconditioned_lam(name, preconditioner, lam_loc, phi, penalty):
    """Sample ``lam`` in preconditioned coordinates, exact prior kept.

    ``z`` carries an isotropic reference measure from ``numpyro.sample``; the
    factor removes that automatic ``N(0, 1)`` density and replaces it with the
    model's actual ``N(lam_loc, (phi * penalty)^-1)`` prior on ``lam``. The
    preconditioner only changes sampling geometry, never the target density.
    """
    n = lam_loc.size
    z = numpyro.sample(name, dist.Normal(0.0, 1.0).expand([n]).to_event(1))
    lam = lam_loc + preconditioner @ z
    delta = lam - lam_loc
    numpyro.factor(
        f"{name}_prior",
        -0.5 * phi * jnp.dot(delta, penalty @ delta) + 0.5 * jnp.sum(z**2),
    )
    return lam


def component_noise_model_preconditioned(
    summed_power,
    counts,
    basis,
    transfer_tm,
    transfer_oms,
    lam_loc_tm,
    lam_loc_oms,
    preconditioner_tm,
    preconditioner_oms,
    penalty,
    phi_tm,
    phi_oms,
    galactic_template,
    frequency_hz,
    reference_f_knee_hz,
    gamma=1680.0,
):
    """``component_noise_model`` with a likelihood-aware spline preconditioner.

    Same physical model, ``S_noise,c = T_tm,c S_TM + T_oms,c S_OMS``; only the
    sampling coordinates for the two frequency splines change. T null-leakage
    is dropped -- out of scope for this preconditioner sanity check.
    """
    log_amplitude = numpyro.sample("log_amplitude", dist.Normal(0.0, 0.5))
    log_f_knee = numpyro.sample(
        "log_f_knee", dist.Normal(jnp.log(reference_f_knee_hz), 0.25)
    )
    log_knee_ratio = _stable_log_knee_ratio(
        frequency_hz[None, :], jnp.exp(log_f_knee), reference_f_knee_hz, gamma
    )

    lam_tm = _sample_preconditioned_lam(
        "z_tm", preconditioner_tm, lam_loc_tm, phi_tm, penalty
    )
    lam_oms = _sample_preconditioned_lam(
        "z_oms", preconditioner_oms, lam_loc_oms, phi_oms, penalty
    )
    numpyro.deterministic("lam_tm", lam_tm)
    numpyro.deterministic("lam_oms", lam_oms)
    spectrum_tm = jnp.exp(basis @ lam_tm)
    spectrum_oms = jnp.exp(basis @ lam_oms)

    log_likelihood = 0.0
    for channel_index in range(len(AET_CHANNELS)):
        noise = (
            transfer_tm[channel_index] * spectrum_tm[None, :]
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


def fit_aet_component_noise_nuts_preconditioned(
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
    n_warmup: int = 500,
    n_samples: int = 500,
    num_chains: int = 2,
    random_seed: int = 20260811,
    target_accept_probability: float = 0.9,
    max_tree_depth: int = 10,
    progress_bar: bool = True,
) -> AETDiagonalPosterior:
    """``fit_aet_component_noise_nuts`` with the likelihood-aware preconditioner.

    Data handling mirrors the original function exactly; only the spline
    sampling coordinates differ. No T null-leakage term.
    """
    observed = np.asarray(observed_psd, dtype=float)
    cell_counts = np.asarray(counts, dtype=float)
    transfer_tm = np.asarray(transfer_tm, dtype=float)
    transfer_oms = np.asarray(transfer_oms, dtype=float)
    galactic_template = np.asarray(galactic_template_psd, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    tm_theory = np.asarray(tm_theory_psd, dtype=float)
    oms_theory = np.asarray(oms_theory_psd, dtype=float)
    if num_chains < 2:
        raise ValueError("AET posterior inference requires at least two chains")

    retained = cell_counts > 0.0
    if mask is not None:
        retained &= np.asarray(mask, dtype=bool)
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

    log_frequency = np.log(frequency)
    frequency_unit = (log_frequency - log_frequency[0]) / (
        log_frequency[-1] - log_frequency[0]
    )
    basis, _ = create_bspline_basis(frequency_unit, n_frequency_knots, degree=3)
    penalty = create_difference_penalty_matrix(basis.shape[1], diff_order=2)
    penalty = (
        penalty
        + 1.0e-6 * np.eye(basis.shape[1]) * float(np.trace(penalty)) / basis.shape[1]
    )
    lam_loc_tm = penalized_spline_prior_centre(basis, np.log(tm_theory), penalty)
    lam_loc_oms = penalized_spline_prior_centre(basis, np.log(oms_theory), penalty)

    # Likelihood Fisher info per frequency channel at the analytic init point
    # (noise-dominated approximation, ignoring the Galaxy's small curvature
    # contribution -- a preconditioner only needs to be approximately right).
    noise0 = (
        transfer_tm * tm_theory[None, None, :] + transfer_oms * oms_theory[None, None, :]
    )
    weight_tm = transfer_tm * tm_theory[None, None, :] / noise0
    weight_oms = transfer_oms * oms_theory[None, None, :] / noise0
    fisher_tm = 0.5 * np.sum(counts_fit * weight_tm**2, axis=(0, 1))
    fisher_oms = 0.5 * np.sum(counts_fit * weight_oms**2, axis=(0, 1))
    preconditioner_tm = _fisher_preconditioner(basis, fisher_tm, phi_tm, penalty)
    preconditioner_oms = _fisher_preconditioner(basis, fisher_oms, phi_oms, penalty)

    init_values = {
        "log_amplitude": np.asarray(0.0),
        "log_f_knee": np.asarray(np.log(reference_f_knee_hz)),
        "z_tm": np.zeros(basis.shape[1]),
        "z_oms": np.zeros(basis.shape[1]),
    }
    model_args = (
        jnp.asarray(summed_power),
        jnp.asarray(counts_fit),
        jnp.asarray(basis),
        jnp.asarray(transfer_tm),
        jnp.asarray(transfer_oms),
        jnp.asarray(lam_loc_tm),
        jnp.asarray(lam_loc_oms),
        jnp.asarray(preconditioner_tm),
        jnp.asarray(preconditioner_oms),
        jnp.asarray(penalty),
        phi_tm,
        phi_oms,
        jnp.asarray(template_fit),
        jnp.asarray(frequency),
        reference_f_knee_hz,
    )
    kernel = NUTS(
        component_noise_model_preconditioned,
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

    noise_by_channel, galactic_by_channel, total_by_channel = [], [], []
    for channel_index in range(len(AET_CHANNELS)):
        noise_draws = (
            transfer_tm[channel_index][None, :, :] * spectrum_tm[:, None, :]
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
    "component_noise_model_preconditioned",
    "fit_aet_component_noise_nuts_preconditioned",
]
