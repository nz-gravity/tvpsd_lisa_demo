"""SEGWO-based local frequency-domain covariance estimates for generated TDI.

This module uses only the public :mod:`segwo` covariance API.  In particular,
it contains no code or imports from ``lisa-UK/lisa_noise``.

The estimator produces local, one-sided XYZ covariance matrices from a TDI
time series.  The diagonal is the channel PSD ``S_X, S_Y, S_Z(t, f)`` and the
off-diagonal entries retain the complex cross-spectral densities.  It is an
*estimate* from a finite realization, not the analytic generation truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import get_window
from segwo.cov import construct_covariance_from_measurements


@dataclass(frozen=True)
class LocalTDICovariance:
    """One-sided local XYZ covariance on a rectangular time-frequency grid."""

    time_tcb: np.ndarray
    frequency_hz: np.ndarray
    covariance_hz2_per_hz: np.ndarray

    @property
    def psd_hz2_per_hz(self) -> np.ndarray:
        """Return real auto-PSDs with shape ``(time, frequency, XYZ)``."""
        return np.real(np.diagonal(self.covariance_hz2_per_hz, axis1=-2, axis2=-1))


def _one_sided_scaling(n_samples: int, dt: float, window: np.ndarray) -> np.ndarray:
    """One-sided periodogram scale for ``numpy.fft.rfft`` coefficients."""
    scale = np.full(n_samples // 2 + 1, dt / np.sum(window**2), dtype=float)
    if scale.size > 2:
        scale[1:-1] *= 2.0
    elif scale.size == 2:  # DC and Nyquist only.
        pass
    return scale


def estimate_local_tdi_covariance(
    tdi_xyz: np.ndarray,
    dt: float,
    *,
    t0_tcb: float = 0.0,
    segment_samples: int = 65_536,
    hop_samples: int | None = None,
    window: str = "hann",
    frequency_range_hz: tuple[float, float] | None = None,
    frequency_bin: int = 1,
) -> LocalTDICovariance:
    """Estimate a locally stationary, one-sided XYZ covariance surface.

    Parameters
    ----------
    tdi_xyz
        Real TDI array with shape ``(3, n_samples)`` in the order X, Y, Z and
        units Hz.
    dt
        Sample cadence in seconds.
    t0_tcb
        TCB epoch of the first sample, in seconds.
    segment_samples, hop_samples
        FFT length and step.  The default hop equals one non-overlapping
        segment.  A smaller hop creates overlapping local estimates; it does
        not by itself make each estimate a Welch average.
    window
        Window accepted by :func:`scipy.signal.get_window`.
    frequency_range_hz
        Optional inclusive frequency range retained in the result.  FFTs are
        still formed at their native resolution before this selection.
    frequency_bin
        Average this many adjacent retained frequency bins.  This reduces the
        stored covariance grid while preserving its one-sided PSD convention.

    Returns
    -------
    LocalTDICovariance
        ``covariance_hz2_per_hz`` has shape ``(time, frequency, 3, 3)``.  It
        is formed using SEGWO's public
        :func:`segwo.cov.construct_covariance_from_measurements`.
    """
    tdi_xyz = np.asarray(tdi_xyz, dtype=float)
    if tdi_xyz.ndim != 2 or tdi_xyz.shape[0] != 3:
        raise ValueError("tdi_xyz must have shape (3, n_samples) in X/Y/Z order")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if segment_samples < 2 or segment_samples > tdi_xyz.shape[1]:
        raise ValueError("segment_samples must be in [2, n_samples]")
    if hop_samples is None:
        hop_samples = segment_samples
    if hop_samples < 1:
        raise ValueError("hop_samples must be positive")
    if frequency_bin < 1:
        raise ValueError("frequency_bin must be positive")

    starts = np.arange(0, tdi_xyz.shape[1] - segment_samples + 1, hop_samples)
    if starts.size == 0:
        raise ValueError("no complete local FFT segments")
    taper = get_window(window, segment_samples, fftbins=True)
    full_frequency_hz = np.fft.rfftfreq(segment_samples, dt)
    frequency_mask = np.ones(full_frequency_hz.size, dtype=bool)
    if frequency_range_hz is not None:
        low, high = map(float, frequency_range_hz)
        if not (0.0 <= low < high):
            raise ValueError("frequency_range_hz must satisfy 0 <= low < high")
        frequency_mask &= (full_frequency_hz >= low) & (full_frequency_hz <= high)
    selected_frequency_hz = full_frequency_hz[frequency_mask]
    if selected_frequency_hz.size < frequency_bin:
        raise ValueError("frequency selection has fewer bins than frequency_bin")
    n_binned_frequency = selected_frequency_hz.size // frequency_bin
    retained_bins = n_binned_frequency * frequency_bin
    frequency_mask_indices = np.flatnonzero(frequency_mask)[:retained_bins]
    frequency_hz = full_frequency_hz[frequency_mask_indices].reshape(-1, frequency_bin).mean(axis=1)
    scale = _one_sided_scaling(segment_samples, dt, taper)

    covariance_by_time = []
    for start in starts:
        segment = tdi_xyz[:, start:start + segment_samples]
        coefficients = np.fft.rfft(segment * taper[None, :], axis=-1).T
        covariance = construct_covariance_from_measurements(coefficients)
        covariance *= scale[:, None, None]
        covariance = covariance[frequency_mask_indices]
        covariance_by_time.append(covariance.reshape(n_binned_frequency, frequency_bin, 3, 3).mean(axis=1))

    time_tcb = t0_tcb + (starts + 0.5 * segment_samples) * dt
    return LocalTDICovariance(time_tcb, frequency_hz, np.stack(covariance_by_time))


def estimate_hdf5_tdi_component(
    archive_path: str | Path,
    component: str,
    **estimator_kwargs: object,
) -> LocalTDICovariance:
    """Estimate a component stored as ``tdi/<component>`` in the archive.

    For ``combined_esa_xyz.h5`` the valid components are ``noise``,
    ``galactic``, and ``total``.  Estimating the first two separately is the
    appropriate finite-realization diagnostic for the additive model; the
    single-periodogram estimate of ``total`` also includes random cross terms.
    """
    archive_path = Path(archive_path)
    with h5py.File(archive_path, "r") as hdf:
        dataset = f"tdi/{component}"
        if dataset not in hdf:
            raise KeyError(f"{dataset!r} was not found in {archive_path}")
        tdi_xyz = hdf[dataset][:]
        # The generated archive uses the explicit ``dt_seconds`` spelling;
        # source LDC-style files commonly use ``dt``.
        if "dt_seconds" in hdf.attrs:
            dt = float(hdf.attrs["dt_seconds"])
        elif "dt" in hdf.attrs:
            dt = float(hdf.attrs["dt"])
        else:
            raise KeyError("archive has neither a 'dt_seconds' nor a 'dt' attribute")
        t0_tcb = float(hdf.attrs["t0_tcb"])
    return estimate_local_tdi_covariance(tdi_xyz, dt, t0_tcb=t0_tcb, **estimator_kwargs)


def add_independent_covariances(*estimates: LocalTDICovariance) -> LocalTDICovariance:
    """Add covariance estimates on an identical grid for independent processes."""
    if not estimates:
        raise ValueError("at least one covariance estimate is required")
    first = estimates[0]
    for estimate in estimates[1:]:
        if not (
            np.array_equal(estimate.time_tcb, first.time_tcb)
            and np.array_equal(estimate.frequency_hz, first.frequency_hz)
        ):
            raise ValueError("all covariance estimates must use the same time-frequency grid")
    return LocalTDICovariance(
        first.time_tcb,
        first.frequency_hz,
        np.sum([estimate.covariance_hz2_per_hz for estimate in estimates], axis=0),
    )
