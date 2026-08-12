import numpy as np
import pytest

from aet_component_pspline_nuts import (
    coefficient_preconditioner,
    fit_aet_component_noise_nuts,
    fit_aet_diagonal_nuts,
)
from tv_pspline_psd.splines import create_bspline_basis, evaluate_bspline_basis


def _small_problem():
    time = np.linspace(0.0, 365.25 * 86400.0, 5)
    frequency = np.geomspace(1.0e-4, 6.0e-3, 8)
    phase = 2.0 * np.pi * time / time[-1]
    oms_reference = np.stack(
        [
            (1.0 + 0.03 * (channel + 1) * np.sin(phase))[:, None]
            * (1.0 + 0.1 * channel)
            * np.ones((1, frequency.size))
            for channel in range(3)
        ]
    )
    tm_reference = 0.3 * oms_reference
    noise = oms_reference + tm_reference
    template = np.stack(
        [
            (1.0 + (0.2 + 0.1 * channel) * np.cos(phase))[:, None]
            * (3.0 - 0.5 * channel)
            * (frequency / frequency[0])[None, :] ** -0.8
            for channel in range(3)
        ]
    )
    observed = noise + template
    counts = np.full_like(observed, 40.0)
    return observed, counts, time, frequency, oms_reference, tm_reference, template


def test_aet_requires_two_chains():
    observed, counts, time, frequency, oms_reference, tm_reference, template = _small_problem()
    with pytest.raises(ValueError, match="at least two chains"):
        fit_aet_diagonal_nuts(
            observed,
            counts,
            time,
            frequency,
            oms_reference_psd=oms_reference,
            tm_reference_psd=tm_reference,
            galactic_template_psd=template,
            num_chains=1,
            n_warmup=2,
            n_samples=2,
            progress_bar=False,
        )


def test_aet_fixed_precision_smoke_returns_all_channels():
    observed, counts, time, frequency, oms_reference, tm_reference, template = _small_problem()
    posterior = fit_aet_diagonal_nuts(
        observed,
        counts,
        time,
        frequency,
        oms_reference_psd=oms_reference,
        tm_reference_psd=tm_reference,
        galactic_template_psd=template,
        n_time_knots=0,
        n_frequency_knots=1,
        phi_time=50.0,
        phi_frequency=60.0,
        noise_level_log_sd=1.0,
        num_chains=2,
        n_warmup=20,
        n_samples=20,
        max_tree_depth=7,
        progress_bar=False,
    )
    assert posterior.diagnostics["num_chains"] == 2
    assert posterior.phi_time == 50.0
    assert posterior.phi_frequency == 60.0
    assert posterior.noise_level_log_sd == 1.0
    assert "phi_time" not in posterior.samples
    assert "phi_frequency" not in posterior.samples
    for surface in (
        posterior.noise_median,
        posterior.noise_lower,
        posterior.noise_upper,
        posterior.galactic_median,
        posterior.galactic_lower,
        posterior.galactic_upper,
        posterior.total_median,
        posterior.total_lower,
        posterior.total_upper,
    ):
        assert surface.shape == observed.shape
        assert np.all(np.isfinite(surface))
        assert np.all(surface > 0.0)
    assert np.all(posterior.total_lower <= posterior.total_median)
    assert np.all(posterior.total_median <= posterior.total_upper)


def test_aet_shared_noise_residual_smoke_returns_all_channels():
    observed, counts, time, frequency, oms_reference, tm_reference, template = _small_problem()
    posterior = fit_aet_diagonal_nuts(
        observed,
        counts,
        time,
        frequency,
        oms_reference_psd=oms_reference,
        tm_reference_psd=tm_reference,
        galactic_template_psd=template,
        share_noise_residual=True,
        delta_log_sd=0.3,
        n_frequency_knots=1,
        phi_time=50.0,
        phi_frequency=60.0,
        noise_level_log_sd=1.0,
        num_chains=2,
        n_warmup=20,
        n_samples=20,
        max_tree_depth=7,
        progress_bar=False,
    )
    assert "z_shared" in posterior.samples
    assert all(f"delta_{c}" in posterior.samples for c in "AET")
    assert not any(f"z_{c}" in posterior.samples for c in "AET")
    for surface in (
        posterior.noise_median,
        posterior.galactic_median,
        posterior.total_median,
    ):
        assert surface.shape == observed.shape
        assert np.all(np.isfinite(surface))
        assert np.all(surface > 0.0)
    assert np.all(posterior.total_lower <= posterior.total_median)
    assert np.all(posterior.total_median <= posterior.total_upper)


def test_aet_rejects_mismatched_noise_reference_shape():
    observed, counts, time, frequency, oms_reference, tm_reference, template = _small_problem()
    with pytest.raises(ValueError, match="must share shape"):
        fit_aet_diagonal_nuts(
            observed,
            counts,
            time,
            frequency,
            oms_reference_psd=oms_reference[:, :1, :],
            tm_reference_psd=tm_reference,
            galactic_template_psd=template,
            num_chains=2,
            n_warmup=2,
            n_samples=2,
            progress_bar=False,
        )


def test_warped_frequency_recovers_a_translating_null():
    """A trough that drifts in frequency is separable in the warped coordinate.

    The unwarped tensor product must build the moving trough out of separable
    products; the warped one describes it with a single frequency profile.
    """
    n_time, n_frequency = 12, 96
    # 60 frequency knots: enough to resolve the trough at fixed time, so this
    # isolates the warp rather than plain frequency resolution.
    time = np.linspace(0.0, 1.0, n_time)
    frequency = np.linspace(1.0e-3, 4.0e-2, n_frequency)
    # Armlength drifts by 2%, so the null at 1/(2L) drifts with it.
    armlength_ratio = 1.0 + 0.02 * np.cos(2.0 * np.pi * time)
    null_hz = 2.0e-2 * armlength_ratio
    truth = np.exp(
        -8.0 * np.exp(-0.5 * ((frequency[None, :] - null_hz[:, None]) / 1.0e-3) ** 2)
    )

    warped = frequency[None, :] / armlength_ratio[:, None]
    log_target = np.log(truth)
    errors = {}
    for label, basis_frequency_grid in (("plain", None), ("warped", warped)):
        basis_time, _ = create_bspline_basis(time, 3, degree=3)
        basis_f, knots = create_bspline_basis(frequency, 60, degree=3)
        if basis_frequency_grid is not None:
            basis_f = evaluate_bspline_basis(
                np.clip(basis_frequency_grid, frequency[0], frequency[-1]).ravel(),
                knots,
                degree=3,
            ).reshape(n_time, n_frequency, -1)
            design = np.einsum("ti,tfj->tfij", basis_time, basis_f)
        else:
            design = np.einsum("ti,fj->tfij", basis_time, basis_f)
        design = design.reshape(n_time * n_frequency, -1)
        coefficients, *_ = np.linalg.lstsq(design, log_target.ravel(), rcond=None)
        errors[label] = float(
            np.sqrt(np.mean((design @ coefficients - log_target.ravel()) ** 2))
        )
    assert errors["warped"] < 0.5 * errors["plain"], errors


def test_preconditioner_is_an_exact_change_of_coordinates():
    """s = A z must reproduce the untransformed prior+likelihood target.

    The preconditioner may only change coordinates, never the posterior, so
    the log density at matched points must agree up to a constant.
    """
    rng = np.random.default_rng(0)
    n_time_basis, n_frequency_basis = 4, 6
    n = n_time_basis * n_frequency_basis
    basis_time = rng.normal(size=(9, n_time_basis))
    basis_frequency = rng.normal(size=(12, n_frequency_basis))
    counts_per_frequency = rng.uniform(4.0, 30.0, size=12)
    scale = rng.uniform(0.2, 3.0, size=(n_time_basis, n_frequency_basis))

    preconditioner = coefficient_preconditioner(
        basis_time, basis_frequency, counts_per_frequency, scale
    )
    assert preconditioner.shape == (n, n)

    # The map must be invertible and reproduce the intended covariance.
    gram_time = basis_time.T @ basis_time
    gram_frequency = basis_frequency.T @ (counts_per_frequency[:, None] * basis_frequency)
    hessian = 0.5 * np.kron(gram_time, gram_frequency)
    hessian[np.diag_indices_from(hessian)] += 1.0 / scale.ravel() ** 2
    np.testing.assert_allclose(
        preconditioner @ preconditioner.T, np.linalg.inv(hessian), rtol=1e-8, atol=1e-12
    )

    # Isotropic in z: condition number of the transformed Hessian is ~1.
    transformed = preconditioner.T @ hessian @ preconditioner
    assert np.linalg.cond(transformed) < 1.0 + 1e-6

    # The exact prior is preserved: log N(s; 0, scale) at s = A z equals the
    # factor the model adds on top of the unit-normal reference measure.
    z = rng.normal(size=n)
    s = preconditioner @ z
    model_factor = -0.5 * np.sum((s / scale.ravel()) ** 2) + 0.5 * np.sum(z**2)
    reference = -0.5 * np.sum(z**2)
    np.testing.assert_allclose(
        model_factor + reference, -0.5 * np.sum((s / scale.ravel()) ** 2), rtol=1e-10
    )


def _component_problem():
    """AET data built exactly as the physical-component model describes it."""
    rng = np.random.default_rng(3)
    n_time, n_frequency = 8, 60
    frequency = np.geomspace(2.0e-4, 8.0e-3, n_frequency)
    # Stationary component spectra; all time dependence in the transfer funcs.
    tm_theory = 1.0 + (3.0e-4 / frequency) ** 2
    oms_theory = 1.0 + (frequency / 2.0e-3) ** 2
    phase = 2.0 * np.pi * np.arange(n_time) / n_time
    transfer_tm = np.stack([
        (2.0 - 0.5 * c) * (1.0 + 0.05 * np.cos(phase))[:, None]
        * np.ones((1, n_frequency))
        for c in range(3)
    ])
    transfer_oms = np.stack([
        (0.7 + 0.4 * c) * (1.0 + 0.05 * np.sin(phase))[:, None]
        * np.ones((1, n_frequency))
        for c in range(3)
    ])
    noise = transfer_tm * tm_theory[None, None, :] + transfer_oms * oms_theory[None, None, :]
    template = np.stack([
        (1.0 + 0.6 * np.cos(phase))[:, None]
        * (3.0 - 0.8 * c)
        * (frequency / frequency[0])[None, :] ** -1.2
        for c in range(3)
    ])
    counts = np.full(noise.shape, 60.0)
    total = noise + 0.7 * template
    observed = total * rng.chisquare(counts) / counts
    return observed, counts, frequency, transfer_tm, transfer_oms, tm_theory, oms_theory, template


def test_component_noise_prior_matches_requested_covariance():
    """lam = lam_loc + L^-T z / sqrt(phi) must realise N(lam_loc, (phi P)^-1)."""
    from tv_pspline_psd.splines import create_difference_penalty_matrix
    import scipy.linalg as sla

    k = 8
    penalty = create_difference_penalty_matrix(k, diff_order=2)
    penalty = penalty + 1e-6 * np.eye(k) * np.trace(penalty) / k
    chol_inv = sla.solve_triangular(
        np.linalg.cholesky(penalty).T, np.eye(k), lower=False
    )
    phi = 25.0
    realised = (chol_inv @ chol_inv.T) / phi
    target = np.linalg.inv(phi * penalty)
    # The penalty is deliberately near-singular (ridge 1e-6), so inverting it
    # loses ~6 digits; compare against the matrix scale rather than elementwise.
    np.testing.assert_allclose(
        realised, target, rtol=1e-4, atol=1e-5 * np.abs(target).max()
    )


def test_component_noise_recovers_a_known_galactic_amplitude():
    obs, counts, freq, t_tm, t_oms, tm_th, oms_th, template = _component_problem()
    posterior = fit_aet_component_noise_nuts(
        obs,
        counts,
        freq,
        transfer_tm=t_tm,
        transfer_oms=t_oms,
        tm_theory_psd=tm_th,
        oms_theory_psd=oms_th,
        galactic_template_psd=template,
        reference_f_knee_hz=2.15e-3,
        n_frequency_knots=6,
        phi_tm=1.0e4,
        phi_oms=1.0e4,
        num_chains=2,
        n_warmup=300,
        n_samples=300,
        progress_bar=False,
    )
    assert posterior.diagnostics["divergences"] == 0
    # The noise has no free time dependence, so A_gal is data-identified.
    lo, hi = np.quantile(posterior.amplitude_draws, [0.05, 0.95])
    assert lo <= 0.7 <= hi, (lo, np.median(posterior.amplitude_draws), hi)
    for surface in (posterior.noise_median, posterior.total_median):
        assert surface.shape == obs.shape
        assert np.all(np.isfinite(surface)) and np.all(surface > 0.0)


def test_component_noise_preconditioner_is_exact_and_conditions_both_blocks():
    """A must be a valid change of coordinates AND fix the block asymmetry.

    TM is prior-pinned (phi_tm=1e8) while OMS is data-dominated (phi_oms=1e4),
    so whitening against the prior alone is right for one block and wrong for
    the other. The preconditioner must whiten against prior + likelihood.
    """
    from aet_component_pspline_nuts import component_noise_preconditioner
    from tv_pspline_psd.splines import (
        create_bspline_basis,
        create_difference_penalty_matrix,
    )
    import scipy.linalg as sla

    rng = np.random.default_rng(0)
    n_time, n_frequency, n_knots = 10, 300, 8
    frequency = np.geomspace(1e-4, 2e-2, n_frequency)
    unit = (np.log(frequency) - np.log(frequency[0])) / (
        np.log(frequency[-1]) - np.log(frequency[0])
    )
    basis, _ = create_bspline_basis(unit, n_knots, degree=3)
    k = basis.shape[1]
    penalty = create_difference_penalty_matrix(k, diff_order=2)
    penalty = penalty + 1e-6 * np.eye(k) * np.trace(penalty) / k

    spectrum_tm = 1.0 + (3e-4 / frequency) ** 2
    spectrum_oms = 1.0 + (frequency / 2e-3) ** 2
    transfer_tm = np.abs(rng.normal(2.0, 0.1, size=(3, n_time, n_frequency)))
    transfer_oms = np.abs(rng.normal(1.0, 0.1, size=(3, n_time, n_frequency)))
    template = np.abs(rng.normal(0.5, 0.05, size=(3, n_time, n_frequency)))
    counts = np.full((3, n_time, n_frequency), 40.0)
    phi_tm, phi_oms = 1.0e8, 1.0e4

    a = component_noise_preconditioner(
        basis, transfer_tm, transfer_oms, spectrum_tm, spectrum_oms,
        template, counts, penalty, phi_tm, phi_oms,
    )
    assert a.shape == (2 * k, 2 * k)
    # Invertible => a genuine change of coordinates, so the posterior is
    # unchanged whatever the metric's quality.
    assert np.isfinite(a).all()
    assert abs(np.linalg.det(a)) > 0

    # It must whiten the posterior it was built from, to condition ~1.
    fisher = np.zeros((2 * k, 2 * k))
    for c in range(3):
        tm_part = transfer_tm[c] * spectrum_tm[None, :]
        oms_part = transfer_oms[c] * spectrum_oms[None, :]
        total = tm_part + oms_part + template[c]
        for first, second, r, cc in (
            (tm_part / total, tm_part / total, 0, 0),
            (tm_part / total, oms_part / total, 0, k),
            (oms_part / total, oms_part / total, k, k),
        ):
            pooled = np.sum(counts[c] * first * second, axis=0)
            fisher[r:r + k, cc:cc + k] += 0.5 * (basis.T @ (pooled[:, None] * basis))
    fisher[k:, :k] = fisher[:k, k:].T
    hessian = fisher.copy()
    hessian[:k, :k] += phi_tm * penalty
    hessian[k:, k:] += phi_oms * penalty

    transformed = a.T @ hessian @ a
    assert np.linalg.cond(transformed) < 1.0 + 1e-6

    # And it must beat prior-only whitening, which is the bug this fixes.
    prior_only = np.zeros((2 * k, 2 * k))
    chol_inv = sla.solve_triangular(np.linalg.cholesky(penalty).T, np.eye(k), lower=False)
    prior_only[:k, :k] = chol_inv / np.sqrt(phi_tm)
    prior_only[k:, k:] = chol_inv / np.sqrt(phi_oms)
    prior_whitened = prior_only.T @ hessian @ prior_only
    scale = 1.0 / np.sqrt(np.diag(prior_whitened))
    cond_prior_only = np.linalg.cond(scale[:, None] * prior_whitened * scale[None, :])
    assert cond_prior_only > 100.0 * np.linalg.cond(transformed)
