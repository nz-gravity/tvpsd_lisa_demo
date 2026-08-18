from types import SimpleNamespace

import numpy as np

from run_component_study import (
    AET_CHANNELS,
    heldout_binned_diagnostics,
    recovery_metrics,
)


def test_aet_whitening_mask_is_diagnostic_only():
    shape = (len(AET_CHANNELS), 1, 3)
    noise = np.ones(shape)
    galactic = np.ones(shape)
    truth_total = noise + galactic
    observed = truth_total.copy()
    observed[:, :, 1] = 200.0
    counts = np.ones(shape)
    whitening_mask = np.ones(shape, dtype=bool)
    whitening_mask[:, :, 1] = False
    posterior = SimpleNamespace(
        noise_median=noise,
        galactic_median=galactic,
        total_median=truth_total,
        total_lower=0.9 * truth_total,
        total_upper=1.1 * truth_total,
    )
    metrics = recovery_metrics(
        observed,
        counts,
        noise,
        galactic,
        posterior,
        whitening_mask=whitening_mask,
    )
    for channel in ("a", "e", "t"):
        assert metrics[f"{channel}_fit_effective_cells"] == 3.0
        assert metrics[f"{channel}_whitening_effective_cells"] == 2.0
        assert metrics[f"{channel}_continuum_mean_z2"] == 1.0


def test_heldout_binned_diagnostics_is_calibrated_and_proper():
    """A correct model must score unit power, nominal coverage, and best."""
    rng = np.random.default_rng(0)
    shape = (400, 300)
    surface = np.exp(rng.normal(0.0, 2.0, shape))
    # Counts spanning the range the production pooling actually produces.
    counts = rng.integers(2, 97, shape).astype(float)
    observed = rng.chisquare(counts) * surface / counts
    mask = np.ones(shape, dtype=bool)

    scores = heldout_binned_diagnostics(observed, counts, surface, mask)
    assert abs(scores["mean_z2"] - 1.0) < 0.01
    assert abs(scores["central_90_fraction"] - 0.90) < 0.01
    assert abs(scores["median_ratio_vs_chi2_nu_median"] - 1.0) < 0.02

    # Properness: the true surface must beat any rescaling of it.
    for factor in (0.7, 0.85, 1.15, 1.4):
        rescaled = heldout_binned_diagnostics(
            observed, counts, surface * factor, mask
        )
        assert (
            rescaled["mean_whittle_log_score"]
            < scores["mean_whittle_log_score"]
        )

    # At one coefficient per cell this must reduce to the surface study's coefficient form,
    # which is what makes the two ladders' scores comparable.
    coefficients = rng.normal(0.0, 1.0, shape) * np.sqrt(surface)
    single = heldout_binned_diagnostics(
        coefficients**2, np.ones(shape), surface, mask
    )
    surface_form = float(
        np.mean(-0.5 * (np.log(surface) + coefficients**2 / surface))
    )
    assert abs(single["mean_whittle_log_score"] - surface_form) < 1.0e-12

    # Zero-count cells are excluded rather than divided by.
    zeroed = counts.copy()
    zeroed[:10] = 0.0
    excluded = heldout_binned_diagnostics(observed, zeroed, surface, mask)
    assert excluded["n_cells"] == mask.sum() - 10 * shape[1]
