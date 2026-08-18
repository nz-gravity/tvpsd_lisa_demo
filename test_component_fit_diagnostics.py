import numpy as np

from component_fit_diagnostics import component_recovery_metrics, masked_frequency_bin_mean


def test_masked_frequency_bin_mean_tracks_counts_and_empty_bins():
    values = np.arange(12.0).reshape(2, 6)
    retained = np.array([
        [True, False, True, True, False, False],
        [False, False, True, False, True, True],
    ])
    means, counts, frequency = masked_frequency_bin_mean(
        values, retained, np.arange(1.0, 7.0), np.array([0, 2, 4])
    )
    np.testing.assert_allclose(counts, [[1, 2, 0], [0, 1, 2]])
    np.testing.assert_allclose(means[0, :2], [0.0, 2.5])
    assert np.isnan(means[0, 2]) and np.isnan(means[1, 0])
    np.testing.assert_allclose(frequency, [1.5, 3.5, 5.5])


def test_recovery_metrics_separate_total_and_component_accuracy():
    shape = (2, 3)
    noise = np.full(shape, 2.0)
    galactic = np.full(shape, 3.0)
    truth_total = noise + galactic
    observed = truth_total.copy()
    counts = np.ones(shape)
    metrics = component_recovery_metrics(
        observed, counts, noise, galactic,
        truth_total, 0.9 * truth_total, 1.1 * truth_total,
        noise, galactic,
    )
    assert metrics["surface_total_median_abs_log_error"] == 0.0
    assert metrics["h_para_total_median_abs_log_error"] == 0.0
    assert metrics["h_para_noise_median_abs_log_error_visible"] == 0.0
    assert metrics["h_para_galactic_median_abs_log_error_visible"] == 0.0
    assert metrics["surface_continuum_mean_z2"] == 1.0
    assert metrics["h_para_continuum_mean_z2"] == 1.0
    assert metrics["surface_pointwise_90_coverage"] == 1.0


def test_whitening_mask_does_not_remove_cells_from_fit_metrics():
    shape = (1, 3)
    noise = np.ones(shape)
    galactic = np.ones(shape)
    truth_total = noise + galactic
    observed = truth_total.copy()
    observed[0, 1] = 200.0
    counts = np.ones(shape)
    whitening_mask = np.array([[True, False, True]])
    metrics = component_recovery_metrics(
        observed,
        counts,
        noise,
        galactic,
        truth_total,
        0.9 * truth_total,
        1.1 * truth_total,
        noise,
        galactic,
        whitening_mask=whitening_mask,
    )
    assert metrics["fit_effective_cells"] == 3.0
    assert metrics["whitening_effective_cells"] == 2.0
    assert metrics["surface_continuum_mean_z2"] == 1.0


def test_time_block_log_pilot_is_truth_free_and_robust():
    """Pilot must come from the data and be steadier than a single time row."""
    import sys
    sys.path.insert(0, ".")
    from run_component_study import time_block_log_pilot

    rng = np.random.default_rng(0)
    # Scaled to the same ratios as the real AET pilot grid (nt=32, ~39000
    # channels, null ~2000 channels wide, default smoothing window 201): a
    # notch much wider than the smoothing window, on a grid much larger than
    # the window, so the window can suppress channel noise without smearing
    # the notch it is meant to preserve.
    n_channels, n_time, n_frequency = 3, 24, 4000
    smoothing_bins = 21
    frequency = np.linspace(1.0, 2.0, n_frequency)
    truth = 1.0 / frequency**2
    truth = truth * (1.0 - 0.99 * np.exp(-0.5 * ((frequency - 1.5) / 0.01) ** 2))
    observed = truth[None, None, :] * rng.chisquare(1, size=(n_channels, n_time, n_frequency))

    pilot = time_block_log_pilot(observed, 6, frequency_smoothing_bins=smoothing_bins)
    assert pilot.shape == (n_channels * 6, n_frequency)
    assert np.all(np.isfinite(pilot))

    # The smoothed block medians must track the notch far better than a
    # single time row, i.e. the pilot suppresses per-channel sampling noise.
    single_row = np.log(np.maximum(observed[0, 0], 1e-300))
    block_error = np.median(np.abs(pilot[0] - np.log(truth)))
    row_error = np.median(np.abs(single_row - np.log(truth)))
    assert block_error < row_error

    # And the notch itself must survive: its depth in the pilot should be
    # close to the true depth, not smoothed away by the frequency window.
    true_depth = np.log(truth.max() / truth.min())
    pilot_depth = pilot[0].max() - pilot[0].min()
    assert pilot_depth > 0.8 * true_depth


def test_truth_free_pilot_produces_a_coarse_bin_layout():
    """Regression guard: the greedy binner must actually merge, not degenerate
    to one bin per channel. This is a real bug that happened once: an
    unsmoothed truth-free pilot left every channel's noise above the merge
    threshold, producing 39125 bins instead of ~1200 and a 32x larger
    inference problem (a fit that normally finishes in ~20 min ran for 13
    hours before being killed).
    """
    import sys
    sys.path.insert(0, ".")
    from run_component_study import time_block_log_pilot
    from tv_pspline_psd.inference import adaptive_frequency_bin_starts

    rng = np.random.default_rng(1)
    n_channels, n_time, n_frequency = 3, 30, 4000
    frequency = np.linspace(1.0e-4, 2.0e-2, n_frequency)
    truth = (1.0 + (2.0e-3 / frequency) ** 2) * (1.0 + (frequency / 6.0e-2) ** 4)
    observed = truth[None, None, :] * rng.chisquare(1, size=(n_channels, n_time, n_frequency))

    pilot = time_block_log_pilot(observed, 6)
    bin_starts = adaptive_frequency_bin_starts(pilot, max_log_range=0.15, max_bin=32)
    # A degenerate layout is one bin per channel; a working one merges the
    # smooth continuum into wide bins. Require at least a 3x reduction.
    assert bin_starts.size < n_frequency / 3
