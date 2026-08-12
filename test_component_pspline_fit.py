"""Recovery tests for the parametric-Galaxy/free-noise-spline fit."""

import numpy as np

from component_pspline_fit import fit_component_pspline, rcl_knee_ratio


def test_parametric_galactic_component_and_free_noise_spline_recover():
    n_time, n_frequency = 12, 31
    time = np.linspace(0.0, 365.25 * 86400.0, n_time)
    frequency = np.geomspace(1.0e-4, 1.5e-2, n_frequency)
    phase = 2.0 * np.pi * time / time[-1]
    log_frequency = np.log(frequency / frequency[0]) / np.log(frequency[-1] / frequency[0])
    noise_shape = (1.0 + 0.2 * np.sin(phase))[:, None] * (
        1.0 + 0.7 * (frequency / frequency[-1]) ** 0.4
    )[None, :]
    galactic_template = (1.0 + 0.5 * np.cos(phase))[:, None] * 12.0 * (
        frequency / frequency[0]
    )[None, :] ** -1.1
    injected_noise = noise_shape * np.exp(
        0.05 * np.sin(phase)[:, None] * log_frequency[None, :]
    )
    injected_galactic = np.exp(0.3) * galactic_template * rcl_knee_ratio(
        frequency, 2.8e-3, 2.15e-3
    )[None, :]
    observed = injected_noise + injected_galactic

    fit = fit_component_pspline(
        observed, time, frequency,
        noise_prior_level_psd=float(np.exp(np.median(np.log(injected_noise)))),
        galactic_template_psd=galactic_template,
        n_time_knots=4, n_frequency_knots=5,
        smoothing_time=10.0, smoothing_frequency=10.0,
        effective_dof=1000.0,
    )

    assert fit.optimizer.success
    np.testing.assert_allclose(fit.log_galactic_amplitude, 0.3, atol=1.0e-2)
    np.testing.assert_allclose(fit.f_knee_hz, 2.8e-3, rtol=3.0e-3)
    assert np.max(np.abs(fit.total_psd / observed - 1.0)) < 0.02
    assert np.sqrt(np.mean(np.log(fit.noise_psd / injected_noise) ** 2)) < 0.05
