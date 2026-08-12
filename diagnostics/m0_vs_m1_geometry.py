"""M0 vs M1 on identical data and identical spline machinery, one channel.

Isolates what actually costs leapfrog steps: the component split, the spline
parameterization, or the mass matrix.
"""

import sys
import time as walltime

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
from tv_pspline_psd.config import PSplineConfig
from tv_pspline_psd.model import (
    eigen_prior_scale,
    sample_eigen_coefficients,
    whiten_penalty_pair,
)
from tv_pspline_psd.splines import (
    create_bspline_basis,
    create_difference_penalty_matrix,
)
from component_pspline_nuts import _stable_log_knee_ratio

CHANNEL = 0
d = np.load(
    "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation/"
    "component_models_aet_diagonal_weakcal_nuts.npz",
    allow_pickle=True,
)
truth_noise = d["truth_noise"][CHANNEL]
truth_galactic = d["truth_galactic"][CHANNEL]
amp_true = float(d["injected_amplitude"])
counts = np.maximum(np.round(d["counts"][CHANNEL]), 1.0)
freq = d["frequency_hz"]
time = d["time_days"] * 86400.0

rng = np.random.default_rng(0)
total = truth_noise + truth_galactic
observed = total * rng.chisquare(counts) / counts

scale = float(np.median(observed))
observed_s = observed / scale
summed_power = counts * observed_s
template_s = truth_galactic / amp_true / scale
noise_level = float(np.exp(np.median(np.log(truth_noise)))) / scale
total_level = float(np.exp(np.median(np.log(total)))) / scale
F_KNEE = 2.15e-3


def bases(n_time_knots, n_freq_knots, time_poly_degree):
    tu = (time - time[0]) / (time[-1] - time[0])
    lf = np.log(freq)
    fu = (lf - lf[0]) / (lf[-1] - lf[0])
    if time_poly_degree is None:
        Bt, _ = create_bspline_basis(tu, n_time_knots, degree=3)
        Pt = create_difference_penalty_matrix(Bt.shape[1], diff_order=2)
    else:
        Bt = np.vander(tu - 0.5, time_poly_degree + 1, increasing=True)
        Pt = np.zeros((Bt.shape[1],) * 2)
    Bf, _ = create_bspline_basis(fu, n_freq_knots, degree=3)
    Pf = create_difference_penalty_matrix(Bf.shape[1], diff_order=2)
    w = whiten_penalty_pair(Pt, Pf)
    return Bt @ w["U_time"], Bf @ w["U_freq"], w


def model(Bt, Bf, cscale, level, template, with_galaxy, config):
    s = sample_eigen_coefficients(
        "s", cscale, (Bt.shape[1], Bf.shape[1]), config
    )
    log_component = jnp.log(level) + Bt @ s @ Bf.T
    if with_galaxy:
        log_a = numpyro.sample("log_amplitude", dist.Normal(0.0, 0.5))
        log_fk = numpyro.sample("log_f_knee", dist.Normal(jnp.log(F_KNEE), 0.25))
        log_gal = (
            jnp.log(template)
            + log_a
            + _stable_log_knee_ratio(freq[None, :], jnp.exp(log_fk), F_KNEE, 1680.0)
        )
        log_total = jnp.logaddexp(log_component, log_gal)
    else:
        log_total = log_component
    numpyro.factor(
        "whittle",
        -0.5 * jnp.sum(counts * log_total + summed_power * jnp.exp(-log_total)),
    )


def run(name, *, with_galaxy, n_time_knots=2, n_freq_knots=8, time_poly_degree=None,
        centered=True, dense_mass=False, phi=100.0, warmup=300, samples=300,
        max_tree_depth=10):
    Bt, Bf, w = bases(n_time_knots, n_freq_knots, time_poly_degree)
    config = PSplineConfig(
        n_interior_knots_time=max(n_time_knots, 1),
        n_interior_knots_freq=n_freq_knots,
        freq_knot_strategy="log",
        centered=centered,
        trim_time_bins=0,
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        null_precision=1.0 / 5.0**2,
        ridge_eps=1.0e-4,
    )
    cscale = eigen_prior_scale(
        jnp.asarray(phi), jnp.asarray(phi),
        jnp.asarray(w["lam_time"]), jnp.asarray(w["lam_freq"]),
        jnp.asarray(w["joint_null"]), config,
    )
    init = {"s": np.zeros(Bt.shape[1] * Bf.shape[1])}
    if with_galaxy:
        init |= {"log_amplitude": np.asarray(0.0),
                 "log_f_knee": np.asarray(np.log(F_KNEE))}
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init),
        target_accept_prob=0.95,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
    )
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=2,
                chain_method="sequential", progress_bar=False)
    t0 = walltime.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(1),
        jnp.asarray(Bt), jnp.asarray(Bf), cscale,
        noise_level if with_galaxy else total_level,
        jnp.asarray(template_s), with_galaxy, config,
        extra_fields=("num_steps", "diverging"),
    )
    dt = walltime.perf_counter() - t0
    steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
    sam = mcmc.get_samples(group_by_chain=True)
    ess = float(np.min(numpyro.diagnostics.effective_sample_size(np.asarray(sam["s"]))))
    npar = Bt.shape[1] * Bf.shape[1] + (2 if with_galaxy else 0)
    print(
        f"{name:38s} npar {npar:4d} | {dt:6.1f}s | steps mean {steps.mean():7.1f}"
        f" max {steps.max():5d} | div {int(np.sum(np.asarray(mcmc.get_extra_fields()['diverging'])))}"
        f" | ESS(s) {ess:6.1f} | ESS/s {ess/dt:5.2f}",
        flush=True,
    )


run("M0 total spline", with_galaxy=False)
run("M1 noise spline + galaxy", with_galaxy=True)
run("M1 dense_mass", with_galaxy=True, dense_mass=True)
run("M1 non-centered", with_galaxy=True, centered=False)
run("M1 stationary time (poly deg 1)", with_galaxy=True, time_poly_degree=1)
run("M0 stationary time (poly deg 1)", with_galaxy=False, time_poly_degree=1)
