"""NumPyro NUTS posterior for free-spline-noise H_para PSD decomposition."""

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

from tv_pspline_psd.config import PSplineConfig
from tv_pspline_psd.inference import reconstruct_eig_coeff_samples
from tv_pspline_psd.model import sample_tensor_eigen_coefficients, whiten_penalty_pair
from tv_pspline_psd.splines import create_bspline_basis, create_difference_penalty_matrix

from component_pspline_fit import ComponentPSplineFit


@dataclass(frozen=True)
class ComponentPSplinePosterior:
    """H_para posterior summaries on the fitted grouped time-frequency grid."""

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
    noise_level_log_sd: float


def _stable_log_knee_ratio(frequency, f_knee, reference_f_knee, gamma):
    """Log of the RCL tanh-knee ratio using a stable logistic identity."""
    return (
        jax.nn.log_sigmoid(2.0 * gamma * (f_knee - frequency))
        - jax.nn.log_sigmoid(2.0 * gamma * (reference_f_knee - frequency))
    )


def h_para_component_model(
    summed_power,
    counts,
    basis_eig_time,
    basis_eig_frequency,
    lam_time,
    lam_frequency,
    joint_null,
    noise_prior_level,
    galactic_template,
    frequency_hz,
    reference_f_knee_hz,
    config,
    gamma=1680.0,
):
    """H_para Whittle model with a proper tensor-spline prior and Galactic priors."""
    eig_coefficients = sample_tensor_eigen_coefficients(
        basis_eig_time,
        basis_eig_frequency,
        lam_time,
        lam_frequency,
        joint_null,
        config,
    )
    log_noise_residual = basis_eig_time @ eig_coefficients @ basis_eig_frequency.T
    log_amplitude = numpyro.sample("log_amplitude", dist.Normal(0.0, 0.5))
    log_f_knee = numpyro.sample(
        "log_f_knee", dist.Normal(jnp.log(reference_f_knee_hz), 0.25)
    )
    f_knee = jnp.exp(log_f_knee)

    log_noise = jnp.log(noise_prior_level) + log_noise_residual
    log_galactic = (
        jnp.log(galactic_template)
        + log_amplitude
        + _stable_log_knee_ratio(
            frequency_hz[None, :], f_knee, reference_f_knee_hz, gamma
        )
    )
    log_total = jnp.logaddexp(log_noise, log_galactic)
    log_likelihood = -0.5 * jnp.sum(
        counts * log_total + summed_power * jnp.exp(-log_total)
    )
    numpyro.deterministic("log_likelihood", log_likelihood)
    numpyro.factor("whittle", log_likelihood)


def _mcmc_diagnostics(mcmc: MCMC, max_tree_depth: int) -> dict[str, float | int]:
    grouped = mcmc.get_samples(group_by_chain=True)
    diagnostics = summary(grouped, group_by_chain=True)
    rhat_values = []
    ess_values = []
    for site in ("phi_time", "phi_freq", "log_amplitude", "log_f_knee", "s"):
        if site not in diagnostics:
            continue
        rhat_values.extend(np.ravel(np.asarray(diagnostics[site]["r_hat"], dtype=float)))
        ess_values.extend(np.ravel(np.asarray(diagnostics[site]["n_eff"], dtype=float)))
    extra = mcmc.get_extra_fields(group_by_chain=True)
    divergences = int(np.sum(np.asarray(extra["diverging"])))
    steps = np.asarray(extra["num_steps"])
    energy = np.asarray(extra["potential_energy"])
    bfmi_by_chain = []
    for chain_energy in energy:
        variance = np.var(chain_energy)
        bfmi_by_chain.append(
            np.nan if variance == 0.0 else float(np.mean(np.diff(chain_energy) ** 2) / variance)
        )
    return {
        "num_chains": int(grouped["log_amplitude"].shape[0]),
        "divergences": divergences,
        "max_r_hat": float(np.nanmax(rhat_values)),
        "min_effective_sample_size": float(np.nanmin(ess_values)),
        "mean_accept_probability": float(np.mean(np.asarray(extra["accept_prob"]))),
        "max_num_steps": int(np.max(steps)),
        "tree_depth_saturation_fraction": float(np.mean(steps >= 2**max_tree_depth - 1)),
        "min_ebfmi": float(np.nanmin(bfmi_by_chain)),
    }


def fit_component_pspline_nuts(
    observed_psd: np.ndarray,
    counts: np.ndarray,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    noise_prior_level_psd: float,
    galactic_template_psd: np.ndarray,
    reference_f_knee_hz: float = 2.15e-3,
    mask: np.ndarray | None = None,
    n_time_knots: int = 5,
    n_frequency_knots: int = 12,
    noise_level_log_sd: float = 5.0,
    n_warmup: int = 500,
    n_samples: int = 500,
    num_chains: int = 2,
    random_seed: int = 20260810,
    target_accept_probability: float = 0.95,
    max_tree_depth: int = 12,
    progress_bar: bool = True,
    map_fit: ComponentPSplineFit | None = None,
) -> ComponentPSplinePosterior:
    """Sample the H_para posterior on a pre-grouped WDM Whittle grid.

    ``observed_psd`` is the mean squared WDM power in each group and ``counts``
    is the number of retained real WDM coefficients in that group.  Both
    components and the observations must share physical PSD units.
    ``noise_prior_level_psd`` is one positive scalar: it centres the proper
    prior for the free log-P-spline noise surface but supplies no time- or
    frequency-dependent noise template. A fixed scaling is removed internally
    to protect float64 gradient conditioning.
    """
    observed = np.asarray(observed_psd, dtype=float)
    counts = np.asarray(counts, dtype=float)
    noise_prior_level = np.asarray(noise_prior_level_psd, dtype=float)
    galactic_template = np.asarray(galactic_template_psd, dtype=float)
    time = np.asarray(time_tcb, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    if observed.ndim != 2:
        raise ValueError("observed_psd must have shape (time, frequency)")
    if any(array.shape != observed.shape for array in (counts, galactic_template)):
        raise ValueError("observed, counts, and template surfaces must share shape")
    if noise_prior_level.shape != ():
        raise ValueError("noise_prior_level_psd must be a positive scalar")
    if time.shape != (observed.shape[0],) or frequency.shape != (observed.shape[1],):
        raise ValueError("time/frequency coordinates do not match the PSD surface")
    if np.any(np.diff(time) <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("time and frequency coordinates must be strictly increasing")
    if mask is None:
        retained = counts > 0.0
    else:
        retained = np.asarray(mask, dtype=bool)
        if retained.shape != observed.shape:
            raise ValueError("mask must have the same shape as observed_psd")
        retained &= counts > 0.0
    if not np.any(retained):
        raise ValueError("no retained cells")
    for name, array, strictly_positive in (
        ("observed_psd", observed, True),
        ("counts", counts, True),
        ("galactic_template_psd", galactic_template, False),
    ):
        selected = array[retained]
        if np.any(~np.isfinite(selected)) or (
            np.any(selected <= 0.0) if strictly_positive else np.any(selected < 0.0)
        ):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"retained {name} values must be finite and {qualifier}")
    if not np.isfinite(noise_prior_level) or noise_prior_level <= 0.0:
        raise ValueError("noise_prior_level_psd must be a finite positive scalar")
    if num_chains < 2:
        raise ValueError("H_para posterior inference requires at least two chains")
    if noise_level_log_sd <= 0.0:
        raise ValueError("noise_level_log_sd must be positive")

    # Replace masked placeholders before JAX sees them; zero counts remove both
    # their power and log-variance contributions from the likelihood.
    data_scale = float(np.median(observed[retained]))
    observed_scaled = np.where(retained, observed / data_scale, 1.0)
    counts_fit = np.where(retained, counts, 0.0)
    summed_power = counts_fit * observed_scaled
    noise_level_scaled = float(noise_prior_level / data_scale)
    positive_template = galactic_template[retained & (galactic_template > 0.0)]
    template_floor = (
        float(np.min(positive_template)) * 1.0e-6 / data_scale
        if positive_template.size else np.finfo(float).tiny
    )
    galactic_scaled = np.where(
        retained, np.maximum(galactic_template / data_scale, template_floor), template_floor
    )

    time_unit = (time - time[0]) / (time[-1] - time[0])
    log_frequency = np.log(frequency)
    frequency_unit = (log_frequency - log_frequency[0]) / (
        log_frequency[-1] - log_frequency[0]
    )
    basis_time, _ = create_bspline_basis(time_unit, n_time_knots, degree=3)
    basis_frequency, _ = create_bspline_basis(frequency_unit, n_frequency_knots, degree=3)
    penalty_time = create_difference_penalty_matrix(basis_time.shape[1], diff_order=2)
    penalty_frequency = create_difference_penalty_matrix(basis_frequency.shape[1], diff_order=2)
    whitened = whiten_penalty_pair(penalty_time, penalty_frequency)
    basis_eig_time = basis_time @ whitened["U_time"]
    basis_eig_frequency = basis_frequency @ whitened["U_freq"]
    config = PSplineConfig(
        n_interior_knots_time=n_time_knots,
        n_interior_knots_freq=n_frequency_knots,
        freq_knot_strategy="log",
        centered=True,
        trim_time_bins=0,
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        null_precision=1.0 / noise_level_log_sd**2,
        ridge_eps=1.0e-4,
    )

    init_values = {
        "phi_time": np.asarray(np.log(30.0)),
        "phi_freq": np.asarray(np.log(30.0)),
        "log_amplitude": np.asarray(0.0),
        "log_f_knee": np.asarray(np.log(reference_f_knee_hz)),
        "s": np.zeros(basis_time.shape[1] * basis_frequency.shape[1]),
    }
    if map_fit is not None:
        if map_fit.spline_coefficients.shape != (basis_time.shape[1], basis_frequency.shape[1]):
            raise ValueError("map_fit spline basis does not match the requested NUTS basis")
        map_eigen = (
            whitened["U_time"].T
            @ map_fit.spline_coefficients
            @ whitened["U_freq"]
        )
        init_values.update({
            "s": map_eigen.ravel(),
            "log_amplitude": np.asarray(map_fit.log_galactic_amplitude),
            "log_f_knee": np.asarray(np.log(map_fit.f_knee_hz)),
        })

    model_args = (
        jnp.asarray(summed_power),
        jnp.asarray(counts_fit),
        jnp.asarray(basis_eig_time),
        jnp.asarray(basis_eig_frequency),
        jnp.asarray(whitened["lam_time"]),
        jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]),
        jnp.asarray(noise_level_scaled),
        jnp.asarray(galactic_scaled),
        jnp.asarray(frequency),
        reference_f_knee_hz,
        config,
    )
    kernel = NUTS(
        h_para_component_model,
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
    eigen_draws = reconstruct_eig_coeff_samples(samples, whitened, config)
    log_noise_residual = np.einsum(
        "ti,sij,fj->stf", basis_eig_time, eigen_draws, basis_eig_frequency,
        optimize=True,
    )
    noise_draws = data_scale * noise_level_scaled * np.exp(log_noise_residual)
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
    galactic_draws = (
        data_scale
        * galactic_scaled[None, :, :]
        * amplitude_draws[:, None, None]
        * np.exp(log_knee_ratio[:, None, :])
    )
    total_draws = noise_draws + galactic_draws

    def intervals(draws):
        return (
            np.median(draws, axis=0),
            np.quantile(draws, 0.05, axis=0),
            np.quantile(draws, 0.95, axis=0),
        )

    noise_median, noise_lower, noise_upper = intervals(noise_draws)
    galactic_median, galactic_lower, galactic_upper = intervals(galactic_draws)
    total_median, total_lower, total_upper = intervals(total_draws)
    return ComponentPSplinePosterior(
        noise_median=noise_median,
        noise_lower=noise_lower,
        noise_upper=noise_upper,
        galactic_median=galactic_median,
        galactic_lower=galactic_lower,
        galactic_upper=galactic_upper,
        total_median=total_median,
        total_lower=total_lower,
        total_upper=total_upper,
        amplitude_draws=amplitude_draws,
        f_knee_draws_hz=f_knee_draws,
        diagnostics=_mcmc_diagnostics(mcmc, max_tree_depth),
        samples=samples,
        mcmc=mcmc,
        runtime_seconds=runtime_seconds,
        noise_level_log_sd=float(noise_level_log_sd),
    )


__all__ = [
    "ComponentPSplinePosterior",
    "fit_component_pspline_nuts",
    "h_para_component_model",
]
