import numpy as np
import pytest
from esa_m0_study import (
    analysis_masks,
    analysis_row_split,
    blind_whitening_diagnostics,
    block_holdout,
    convergence_gate_status,
    gate_gaps,
    good_time_bins,
    hybrid_frequency_knots,
    low_band_modulation_posterior,
    partition_starts,
    posterior_predictive_score_comparison,
    projected_interpolated_positive_surface,
    response_null_mask,
    robust_training_psd_scale,
    training_data_pilot_log_psd,
    wdm_valid_length,
)


def test_wdm_valid_length_is_even_geometry():
    n = wdm_valid_length(15_728_640, 2048)
    assert n <= 15_728_640
    assert n % 2048 == 0
    assert (n // 2048) % 2 == 0


def test_gap_gate_and_good_rows_are_conservative():
    dt = 1.0
    data = np.ones(100)
    gaps = [(40.0, 50.0)]
    gated = gate_gaps(data, dt, gaps, taper_s=5.0)
    assert np.all(gated[40:51] == 0.0)
    assert gated[39] < 1.0
    assert gated[51] < 1.0
    grid = (np.arange(10) + 0.5) / 10
    keep = good_time_bins(grid, 100.0, gaps, nt=10, taper_s=5.0)
    assert np.any(~keep)
    assert keep[0] and keep[-1]


def test_holdout_and_partition_preserve_transitions():
    heldout = block_holdout(23, block=4, every=5)
    assert np.array_equal(np.flatnonzero(heldout), np.array([16, 17, 18, 19]))
    good = np.ones(23, dtype=bool)
    good[9:12] = False
    starts = partition_starts(23, 4, heldout, good)
    assert {0, 4, 8, 9, 12, 16, 20}.issubset(set(starts))


def test_analysis_split_is_disjoint_and_complete():
    training, validation, test = analysis_row_split(
        71, block=3, cycle=7, validation_fold=5, test_fold=6
    )
    assert np.all(training | validation | test)
    assert not np.any(training & validation)
    assert not np.any(training & test)
    assert not np.any(validation & test)
    assert np.any(validation) and np.any(test)


def test_inference_and_evaluation_masks_are_separate():
    training = np.array([True, False, True])
    response_keep = np.array(
        [
            [True, False, True, False],
            [True, True, False, False],
            [False, True, True, False],
        ]
    )

    inference, evaluation = analysis_masks(training, response_keep)

    assert np.all(inference[training])
    assert not np.any(inference[~training])
    assert np.array_equal(evaluation, response_keep)
    assert np.any(inference & ~evaluation)


def test_adaptive_pilot_includes_response_null_training_cells():
    coefficients = np.ones((8, 9))
    training = np.ones(8, dtype=bool)
    response_keep = np.ones_like(coefficients, dtype=bool)
    response_keep[:, 4] = False
    inference, evaluation = analysis_masks(training, response_keep)

    baseline = training_data_pilot_log_psd(
        coefficients,
        inference,
        n_time_profiles=2,
        frequency_width=1,
    )
    changed = coefficients.copy()
    changed[:, 4] = 1.0e3
    changed_pilot = training_data_pilot_log_psd(
        changed,
        inference,
        n_time_profiles=2,
        frequency_width=1,
    )

    assert not np.allclose(baseline[:, 4], changed_pilot[:, 4])
    assert not np.any(evaluation[:, 4])


def test_training_pilot_does_not_use_excluded_values():
    rng = np.random.default_rng(9)
    coefficients = rng.normal(size=(20, 30))
    retained = np.ones_like(coefficients, dtype=bool)
    retained[5:8] = False
    retained[:, 12:15] = False
    pilot = training_data_pilot_log_psd(
        coefficients, retained, n_time_profiles=5, frequency_width=5
    )
    changed = coefficients.copy()
    changed[~retained] = 1.0e100
    pilot_changed = training_data_pilot_log_psd(
        changed, retained, n_time_profiles=5, frequency_width=5
    )
    assert np.all(np.isfinite(pilot))
    assert np.allclose(pilot, pilot_changed)


def test_training_scale_is_blind_to_excluded_cells_and_scale_equivariant():
    rng = np.random.default_rng(31)
    coefficients = rng.normal(size=(40, 50)) * 3.0e-12
    retained = np.ones_like(coefficients, dtype=bool)
    retained[::5] = False
    retained[:, 17:21] = False
    baseline = robust_training_psd_scale(coefficients, retained, to_psd=0.25)

    changed = coefficients.copy()
    changed[~retained] = 1.0e100
    assert robust_training_psd_scale(changed, retained, to_psd=0.25) == baseline

    amplitude_scale = 7.5
    scaled = robust_training_psd_scale(
        amplitude_scale * coefficients, retained, to_psd=0.25
    )
    assert scaled == pytest.approx(amplitude_scale**2 * baseline)


def test_blind_whitening_diagnostics_detect_correct_scale():
    rng = np.random.default_rng(81)
    truth = np.linspace(0.5, 2.0, 80)[None, :]
    coefficients = rng.normal(size=(400, 80)) * np.sqrt(truth)
    surface = np.broadcast_to(truth, coefficients.shape)
    mask = np.ones_like(coefficients, dtype=bool)
    mask[::7] = False
    diagnostics = blind_whitening_diagnostics(
        coefficients, surface, mask, to_psd=1.0
    )
    assert abs(diagnostics["mean_z"]) < 0.02
    assert diagnostics["mean_z2"] == pytest.approx(1.0, abs=0.03)
    assert diagnostics["central_90_fraction"] == pytest.approx(0.9, abs=0.02)
    assert abs(diagnostics["lag1_time_product"]) < 0.03
    assert abs(diagnostics["lag1_frequency_product"]) < 0.03


def test_convergence_gate_reports_each_publication_threshold():
    accepted = {
        "divergences": 0,
        "max_rhat": 1.01,
        "min_ess": 100.0,
        "tree_depth_saturation": 0.01,
        "min_ebfmi": 0.5,
    }
    assert convergence_gate_status(accepted)["passed"]
    rejected = accepted | {"max_rhat": 1.06}
    status = convergence_gate_status(rejected)
    assert not status["passed"]
    assert not status["checks"]["max_rhat_le_1.05"]


def test_hybrid_knots_are_strictly_interior_and_ordered():
    frequency = np.linspace(1e-4, 1e-1, 1000)
    knots = hybrid_frequency_knots(frequency, 60)
    assert knots.size == 60
    assert np.all(np.diff(knots) > 0)
    assert knots[0] > frequency[0]
    assert knots[-1] < frequency[-1]


def test_response_mask_only_removes_high_frequency_depressions():
    frequency = np.linspace(1e-4, 1e-1, 401)
    surface = np.ones((3, frequency.size))
    surface[:, 250] = 1e-4
    keep, ratio, continuum = response_null_mask(
        surface,
        frequency,
        high_frequency_hz=0.02,
        ratio_threshold=0.35,
        continuum_width=21,
        dilation_bins=2,
    )
    assert np.all(keep[:, frequency < 0.02])
    assert np.all(~keep[:, 248:253])
    assert np.all(ratio[:, 250] < 0.35)
    assert np.all(continuum > 0)


def test_projected_surface_honors_zero_outside_generated_band():
    source_time = np.array([0.0, 1.0])
    source_frequency = np.array([0.01, 0.02])
    source = np.ones((2, 2))
    target_time = np.array([0.25, 0.75])
    target_frequency = np.array([0.01, 0.02])

    extrapolated = projected_interpolated_positive_surface(
        source,
        source_time,
        source_frequency,
        target_time,
        target_frequency,
        1.0e-3,
        projection_nodes=16,
    )
    bounded = projected_interpolated_positive_surface(
        source,
        source_time,
        source_frequency,
        target_time,
        target_frequency,
        1.0e-3,
        projection_nodes=16,
        zero_outside_frequency=True,
    )

    assert np.allclose(extrapolated, 1.0)
    assert np.all(bounded < extrapolated)
    assert np.all(bounded > 0.0)


def _zero_tensor_fit(n_draws: int = 4) -> dict:
    class Config:
        centered = True

    return {
        "residual_structure": "tensor",
        "config": Config(),
        "B_time": np.eye(2),
        "B_freq": np.eye(3),
        "whitened": {"U_time": np.eye(2), "U_freq": np.eye(3)},
        "samples": {"s": np.zeros((n_draws, 2, 3))},
    }


def test_low_band_modulation_uses_posterior_draws():
    fit = _zero_tensor_fit()
    offset = np.log(np.array([[2.0, 2.0, 9.0], [3.0, 3.0, 9.0]]))
    lower, median, upper = low_band_modulation_posterior(
        fit, offset, np.array([True, True, False])
    )
    assert np.allclose(lower, [2.0, 3.0])
    assert np.allclose(median, [2.0, 3.0])
    assert np.allclose(upper, [2.0, 3.0])


def test_posterior_predictive_comparison_is_paired_by_block():
    fit = _zero_tensor_fit()
    stationary = {"residual_log_psd_samples": np.zeros((4, 3))}
    coefficients = np.ones((2, 3))
    result = posterior_predictive_score_comparison(
        coefficients,
        fit,
        stationary,
        np.zeros_like(coefficients),
        np.array([True, True]),
        {"all": np.ones_like(coefficients, dtype=bool)},
        row_block=1,
        frequency_chunk=2,
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )
    summary = result["cohorts"]["all"]
    assert summary["n_blocks"] == 2
    assert summary["n_cells"] == 6
    assert summary["tv_minus_stationary_mean"] == pytest.approx(0.0)
    assert summary["tv_minus_stationary_block_bootstrap_p05"] == pytest.approx(0.0)
    assert summary["tv_minus_stationary_block_bootstrap_p95"] == pytest.approx(0.0)
