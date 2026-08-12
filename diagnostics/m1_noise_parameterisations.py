"""Why is M1 slow? Compare noise-model parameterizations on synthetic Whittle
data drawn from the pilot's own truth surfaces."""

import sys
import numpy as np

sys.path.insert(0, "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
from aet_component_pspline_nuts import fit_aet_diagonal_nuts

d = np.load(
    "/Users/avi/Documents/projects/wdm_psd/lisa_data_generation/"
    "component_models_aet_diagonal_weakcal_nuts.npz",
    allow_pickle=True,
)
truth_noise = d["truth_noise"]
truth_galactic = d["truth_galactic"]
amp_true = float(d["injected_amplitude"])
template = truth_galactic / amp_true
counts = np.maximum(np.round(d["counts"]), 1.0)
freq = d["frequency_hz"]
time = d["time_days"] * 86400.0

rng = np.random.default_rng(0)
total = truth_noise + truth_galactic
observed = total * rng.chisquare(counts) / counts

level_scalar = np.exp(np.array([np.median(np.log(truth_noise[c])) for c in range(3)]))

common = dict(
    counts=counts,
    time_tcb=time,
    frequency_hz=freq,
    galactic_template_psd=template,
    n_frequency_knots=8,
    n_warmup=150,
    n_samples=150,
    num_chains=2,
    progress_bar=False,
)

cases = {
    "A current: tensor t-f spline, scalar level": dict(
        noise_prior_level_psd=level_scalar, n_time_knots=2
    ),
    "B stationary+drift in t, scalar level": dict(
        noise_prior_level_psd=level_scalar, time_poly_degree=1
    ),
    "C stationary+drift in t, noise surface offset": dict(
        noise_prior_level_psd=truth_noise, time_poly_degree=1
    ),
}

for name, kwargs in cases.items():
    post = fit_aet_diagonal_nuts(observed, **common, **kwargs)
    steps = np.asarray(post.mcmc.get_extra_fields()["num_steps"])
    amp = post.amplitude_draws
    print(
        f"{name}\n"
        f"   runtime {post.runtime_seconds:6.1f}s | mean steps {steps.mean():7.1f}"
        f" | max {steps.max():5d} | div {post.diagnostics['divergences']}"
        f" | maxRhat {post.diagnostics['max_r_hat']:.3f}"
        f" | minESS {post.diagnostics['min_effective_sample_size']:6.1f}\n"
        f"   A_gal {np.median(amp):.3f} (true {amp_true:.3f})"
        f"  90% [{np.quantile(amp, .05):.3f}, {np.quantile(amp, .95):.3f}]"
        f" | f_knee {1e3*np.median(post.f_knee_draws_hz):.3f} mHz",
        flush=True,
    )
