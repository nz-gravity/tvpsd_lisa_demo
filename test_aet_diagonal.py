import numpy as np
from types import SimpleNamespace

from aet_diagonal import (
    XYZ_TO_AET,
    diagonal_xyz_psd_to_aet,
    xyz_covariance_to_aet_diagonal,
    xyz_to_aet_series,
)
from run_aet_diagonal_pilot import AET_CHANNELS, recovery_metrics


def test_xyz_to_aet_is_orthonormal_and_preserves_sample_power():
    np.testing.assert_allclose(XYZ_TO_AET @ XYZ_TO_AET.T, np.eye(3), atol=1e-15)
    xyz = np.arange(24, dtype=float).reshape(3, 8)
    aet = xyz_to_aet_series(xyz)
    np.testing.assert_allclose(np.sum(aet**2, axis=0), np.sum(xyz**2, axis=0))


def test_diagonal_psd_rotation_matches_zero_csd_covariance():
    xyz_psd = np.asarray([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]])
    aet_psd = diagonal_xyz_psd_to_aet(xyz_psd)
    expected = np.empty_like(aet_psd)
    for index in range(xyz_psd.shape[1]):
        covariance = np.diag(xyz_psd[:, index])
        expected[:, index] = np.diag(
            XYZ_TO_AET @ covariance @ XYZ_TO_AET.T
        )
    np.testing.assert_allclose(aet_psd, expected)


def test_full_covariance_rotation_returns_only_correct_aet_diagonal():
    covariance = np.asarray(
        [
            [2.0, 0.4 + 0.1j, -0.2j],
            [0.4 - 0.1j, 3.0, 0.3],
            [0.2j, 0.3, 5.0],
        ]
    )
    expected = np.diag(XYZ_TO_AET @ covariance @ XYZ_TO_AET.T).real
    np.testing.assert_allclose(
        xyz_covariance_to_aet_diagonal(covariance), expected
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
