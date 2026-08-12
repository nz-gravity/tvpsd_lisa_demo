import numpy as np
import pytest

from component_pspline_nuts import fit_component_pspline_nuts


def _small_problem():
    time = np.linspace(0.0, 365.25 * 86400.0, 6)
    frequency = np.geomspace(1.0e-4, 8.0e-3, 10)
    phase = 2.0 * np.pi * time / time[-1]
    noise = (1.0 + 0.05 * np.sin(phase))[:, None] * np.ones((1, frequency.size))
    galactic = (1.0 + 0.4 * np.cos(phase))[:, None] * 4.0 * (
        frequency / frequency[0]
    )[None, :] ** -1.0
    counts = np.full(noise.shape, 50.0)
    observed = noise + galactic
    noise_prior_level = float(np.exp(np.median(np.log(noise))))
    return observed, counts, time, frequency, noise_prior_level, galactic


def test_m1_nuts_requires_two_chains():
    observed, counts, time, frequency, noise_prior_level, galactic = _small_problem()
    with pytest.raises(ValueError, match="at least two chains"):
        fit_component_pspline_nuts(
            observed, counts, time, frequency,
            noise_prior_level_psd=noise_prior_level, galactic_template_psd=galactic,
            num_chains=1, n_warmup=2, n_samples=2, progress_bar=False,
        )


def test_m1_nuts_smoke_returns_component_intervals():
    observed, counts, time, frequency, noise_prior_level, galactic = _small_problem()
    posterior = fit_component_pspline_nuts(
        observed, counts, time, frequency,
        noise_prior_level_psd=noise_prior_level, galactic_template_psd=galactic,
        n_time_knots=0, n_frequency_knots=1,
        noise_level_log_sd=1.0,
        num_chains=2, n_warmup=20, n_samples=20,
        max_tree_depth=7, progress_bar=False,
    )
    assert posterior.diagnostics["num_chains"] == 2
    assert posterior.diagnostics["divergences"] == 0
    for surface in (
        posterior.noise_median, posterior.noise_lower, posterior.noise_upper,
        posterior.galactic_median, posterior.galactic_lower, posterior.galactic_upper,
        posterior.total_median, posterior.total_lower, posterior.total_upper,
    ):
        assert surface.shape == observed.shape
        assert np.all(np.isfinite(surface)) and np.all(surface > 0.0)
    assert np.all(posterior.total_lower <= posterior.total_median)
    assert np.all(posterior.total_median <= posterior.total_upper)


def test_m1_nuts_rejects_a_time_frequency_noise_template():
    observed, counts, time, frequency, _, galactic = _small_problem()
    with pytest.raises(ValueError, match="positive scalar"):
        fit_component_pspline_nuts(
            observed,
            counts,
            time,
            frequency,
            noise_prior_level_psd=np.ones_like(observed),
            galactic_template_psd=galactic,
            num_chains=2,
            n_warmup=2,
            n_samples=2,
            progress_bar=False,
        )
