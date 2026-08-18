"""Checks on the correlated Galactic synthesis in ``sgwb_data_generation.ipynb``.

The archive previously drew X, Y and Z independently from their auto-PSDs,
which destroyed the TDI cross spectra and left the Galactic foreground in the
A/E/T ``T`` channel at full strength.  These tests pin the convention that
replaced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebuild_archive import NOTEBOOK_PATH, load_generation_namespace

pytest.importorskip("backgrounds")


@pytest.fixture(scope="module")
def synthesis():
    namespace = load_generation_namespace()
    if not namespace["NOISE_TDI_PATH"].exists():
        pytest.skip("noise2a inputs are not present")
    return namespace


def _aet_rotation() -> np.ndarray:
    return np.asarray(
        [
            [-1.0, 0.0, 1.0],
            [1.0, -2.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    ) / np.asarray([[np.sqrt(2.0)], [np.sqrt(6.0)], [np.sqrt(3.0)]])


def test_draw_reproduces_the_full_cross_spectral_matrix(synthesis):
    """The drawn series must match S_ab, not just the diagonal S_aa."""
    draw = synthesis["draw_correlated_series"]
    dt_seconds, n = 2.0, 4096
    frequency = np.fft.rfftfreq(n, dt_seconds)
    band = (frequency >= synthesis["FREQ_MIN_HZ"]) & (frequency <= synthesis["FREQ_MAX_HZ"])

    # A LISA-like fully correlated response: equal auto-PSDs, corr = -1/2.
    correlation = np.full((3, 3), -0.5) + 1.5 * np.eye(3)
    level = 3.0e-40
    covariance = np.zeros((frequency.size, 3, 3), dtype=complex)
    covariance[band] = level * correlation

    rng = np.random.default_rng(20260818)
    estimate = np.zeros((frequency.size, 3, 3), dtype=complex)
    replicates = 3000
    for _ in range(replicates):
        coefficients = np.fft.rfft(draw(covariance, dt_seconds, rng), axis=-1)
        estimate += np.einsum("af,bf->fab", coefficients, coefficients.conj())
    estimate *= 2.0 * dt_seconds / (n * replicates)

    interior = band.copy()
    interior[[0, -1]] = False  # DC and Nyquist have half the degrees of freedom.
    recovered = np.real(estimate[interior]).mean(axis=0) / level
    assert np.abs(recovered - correlation).max() < 0.02

    # The point of retaining the cross spectra: T is null for a correlated
    # response.  With independent draws this ratio would be of order one.
    rotation = _aet_rotation()
    aet = np.einsum("ai,fij,bj->fab", rotation, estimate[interior], rotation)
    power = np.real(np.diagonal(aet, axis1=-2, axis2=-1)).mean(axis=0)
    assert power[2] / power[0] < 1.0e-3


def test_interpolated_covariance_stays_positive_semidefinite(synthesis):
    """Interpolating a correlation matrix between nodes must stay factorizable."""
    build = synthesis["covariance_interpolators"]
    evaluate = synthesis["evaluate_covariance"]
    time_tcb = np.linspace(0.0, 1.0e6, 5)
    frequency_hz = np.geomspace(synthesis["FREQ_MIN_HZ"], synthesis["FREQ_MAX_HZ"], 32)

    rng = np.random.default_rng(7)
    root = rng.standard_normal((time_tcb.size, frequency_hz.size, 3, 3))
    surface = np.einsum("tfac,tfbc->tfab", root, root) * 1.0e-40

    fft_frequency_hz = np.fft.rfftfreq(2048, 2.0)
    covariance = evaluate(build(time_tcb, frequency_hz, surface), 3.7e5, fft_frequency_hz)
    in_band = np.real(np.einsum("fcc->f", covariance)) > 0.0
    assert in_band.any()
    assert np.linalg.eigvalsh(covariance[in_band]).min() >= -1.0e-60
    assert np.allclose(covariance, np.conj(np.swapaxes(covariance, -1, -2)))


def test_notebook_no_longer_advertises_independent_xyz_channels():
    """Guard the archive contract the fit code reads off the documentation."""
    text = NOTEBOOK_PATH.read_text()
    assert "zero-XYZ-cross-spectrum" not in text
    assert "truth/*_csd" in text or "_csd" in text
