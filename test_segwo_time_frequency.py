"""Tests for the public-SEGWO local TDI estimator."""

import numpy as np

from segwo_time_frequency import add_independent_covariances, estimate_local_tdi_covariance


def test_local_covariance_has_expected_shape_and_is_hermitian():
    rng = np.random.default_rng(17)
    data = rng.normal(size=(3, 4096))
    estimate = estimate_local_tdi_covariance(data, 16.0, segment_samples=1024)

    assert estimate.covariance_hz2_per_hz.shape == (4, 513, 3, 3)
    np.testing.assert_allclose(
        estimate.covariance_hz2_per_hz,
        np.conj(estimate.covariance_hz2_per_hz.swapaxes(-1, -2)),
    )
    assert np.all(estimate.psd_hz2_per_hz >= 0.0)


def test_independent_covariances_add_on_a_shared_grid():
    rng = np.random.default_rng(23)
    first = estimate_local_tdi_covariance(rng.normal(size=(3, 2048)), 16.0, segment_samples=1024)
    second = estimate_local_tdi_covariance(rng.normal(size=(3, 2048)), 16.0, segment_samples=1024)
    combined = add_independent_covariances(first, second)
    np.testing.assert_allclose(
        combined.covariance_hz2_per_hz,
        first.covariance_hz2_per_hz + second.covariance_hz2_per_hz,
    )


def test_frequency_selection_and_binning_reduce_the_grid():
    rng = np.random.default_rng(29)
    estimate = estimate_local_tdi_covariance(
        rng.normal(size=(3, 4096)),
        2.0,
        segment_samples=1024,
        frequency_range_hz=(0.1, 0.2),
        frequency_bin=4,
    )
    assert np.all((estimate.frequency_hz >= 0.1) & (estimate.frequency_hz <= 0.2))
    assert estimate.covariance_hz2_per_hz.shape[1] == estimate.frequency_hz.size
