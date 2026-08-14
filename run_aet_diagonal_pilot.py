"""Run the diagonal-AET free-noise-spline component-separation study.

This is intentionally a product of three univariate Whittle likelihoods.  It
does not estimate or invert an AET cross-spectral matrix.  Because the source
archive generated independent XYZ Galactic channels, its diagonal PSD truth is
rotated under the same zero-XYZ-CSD contract; results are a controlled toy AET
study rather than a physical multichannel stochastic-response validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
from backgrounds import noise as background_noise, tdi
from lisaorbits import InterpolatedOrbits
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, median_filter, uniform_filter1d
from scipy.stats import chi2


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent / "wdm_psd"
for path in (HERE, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tv_pspline_psd.multichannel import (
    fit_aet_component_noise_nuts,
    fit_aet_diagonal_nuts,
)
from tv_pspline_psd.lisa_aet import (
    AET_CHANNELS,
    diagonal_xyz_psd_to_aet,
    xyz_covariance_to_aet_diagonal,
    xyz_to_aet_series,
)
from component_fit_diagnostics import masked_frequency_bin_mean
from esa_m0_study import (
    analysis_row_split,
    partition_starts,
    projected_analytic_channel_noise_components_psd,
    projected_interpolated_positive_surface,
    training_data_pilot_log_psd,
)
from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
from tv_pspline_psd.datasets import wdm_white_noise_calibration
from tv_pspline_psd.inference import adaptive_frequency_bin_starts


SECONDS_PER_YEAR = 365.25 * 24 * 3600
CARRIER_FREQUENCY_HZ = 281_600_000_000_000.0


def wdm_valid_length(n_requested: int, nt: int) -> int:
    nf = n_requested // nt
    nf -= nf % 2
    if nt % 2 or nf < 2:
        raise ValueError("WDM requires even nt and an even N / nt >= 2")
    return nt * nf


def load_esa_orbits(path: Path) -> InterpolatedOrbits:
    """Build the orbit interpolator used by the archived noise simulation."""
    with h5py.File(path, "r") as hdf:
        t0 = float(hdf.attrs["t0"])
        dt = float(hdf.attrs["dt"])
        size = int(hdf.attrs["size"])
        time_tcb = t0 + dt * np.arange(size)
        positions = hdf["tcb/x"][:]
    return InterpolatedOrbits(
        time_tcb,
        positions,
        t_init=t0,
        interp_order=5,
        extrapolate=False,
    )


def armlength_ratio(orbit_path: Path, time_tcb: np.ndarray) -> np.ndarray:
    """Mean light-travel-time ratio ``L(t) / mean(L)`` across the six links.

    The first TDI null sits at ``1/(2 L(t))`` and drifts with the armlength
    over the year (see the archived null-tracking validation against SEGWO).
    Warping frequency by this ratio makes that drift stationary in the warped
    coordinate; using the six-link mean rather than a per-channel armlength
    is an approximation shared by all A/E/T channels, adequate for a
    near-equilateral constellation.
    """
    orbits = load_esa_orbits(orbit_path)
    links = np.array([12, 23, 31, 13, 32, 21])
    ltt = np.mean(orbits.compute_ltt(time_tcb, links), axis=1)
    return ltt / np.mean(ltt)


def analytic_aet_noise_components_psd(
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    frequency_chunk: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the analytic AET (OMS, test-mass) auto-PSD components.

    Each is ``(3, n_time, n_frequency)``, same order and units as the
    combined surface. Kept separate so callers can inspect them individually;
    the M1 likelihood uses their sum as a fixed offset.

    ``frequency_chunk`` evaluates the analytic model in frequency slices
    instead of materialising the full ``(n_frequency, n_time, 3, 3)`` complex
    covariance tensor at once, matching ``esa_m0_study.py``'s equivalent.
    Required on the fine (unbinned, tens-of-thousands-of-channels) WDM grid --
    without it a full-band call allocates tens of GB. Only the binned
    ``grouped_frequency`` call sites are small enough to omit it.
    """
    orbits = load_esa_orbits(orbit_path)

    def to_aet(xyz_covariance: np.ndarray) -> np.ndarray:
        # Input order is (frequency, time, XYZ, XYZ); return (AET, time, frequency).
        diagonal = xyz_covariance_to_aet_diagonal(xyz_covariance)
        return np.moveaxis(diagonal, (0, 1, 2), (2, 1, 0)) * CARRIER_FREQUENCY_HZ**2

    def build(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        oms = background_noise.AnalyticOMSNoiseModel(
            frequencies,
            time_tcb,
            orbits,
            tdi_tf_func=tdi.compute_tdi_tf,
            gen="2.0",
            oms_isi_carrier_asds=7.9e-12,
            fs=0.5,
            duration=SECONDS_PER_YEAR,
        )
        test_mass = background_noise.AnalyticTMNoiseModel(
            frequencies,
            time_tcb,
            orbits,
            tdi_tf_func=tdi.compute_tdi_tf_tm,
            gen="2.0",
            tm_isi_carrier_asds=2.4e-15,
            fs=0.5,
            duration=SECONDS_PER_YEAR,
        )
        return (
            to_aet(oms.compute_covariances(0.0)),
            to_aet(test_mass.compute_covariances(0.0)),
        )

    if frequency_chunk is None:
        return build(frequency_hz)

    n_time = np.asarray(time_tcb).size
    oms_result = np.empty((3, n_time, frequency_hz.size))
    tm_result = np.empty((3, n_time, frequency_hz.size))
    for start in range(0, frequency_hz.size, frequency_chunk):
        stop = min(start + frequency_chunk, frequency_hz.size)
        oms_chunk, tm_chunk = build(frequency_hz[start:stop])
        oms_result[:, :, start:stop] = oms_chunk
        tm_result[:, :, start:stop] = tm_chunk
    return oms_result, tm_result


def projected_aet_noise_components_psd(
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    delta_f_hz: float,
    *,
    projection_nodes: int = 16,
    frequency_chunk: int = 384,
) -> tuple[np.ndarray, np.ndarray]:
    """AET stack of M0's WDM-kernel-projected analytic OMS/TM components.

    Each returned array is ``(3, n_time, n_frequency)`` in ``AET_CHANNELS``
    order, matching ``analytic_aet_noise_components_psd``'s layout so the two
    are drop-in interchangeable. The per-channel projection itself is
    ``esa_m0_study``'s, so M0 and M1 share one estimand by construction rather
    than by two implementations agreeing.
    """
    oms_result, tm_result = [], []
    for channel in AET_CHANNELS:
        oms, tm = projected_analytic_channel_noise_components_psd(
            channel,
            orbit_path,
            time_tcb,
            frequency_hz,
            delta_f_hz,
            projection_nodes=projection_nodes,
            frequency_chunk=frequency_chunk,
        )
        oms_result.append(oms)
        tm_result.append(tm)
    return np.stack(oms_result), np.stack(tm_result)


def projected_aet_interpolated_surface(
    source: np.ndarray,
    source_time: np.ndarray,
    source_frequency: np.ndarray,
    target_time: np.ndarray,
    target_frequency: np.ndarray,
    delta_f_hz: float,
    *,
    projection_nodes: int = 16,
    frequency_chunk: int = 384,
    zero_outside_frequency: bool = False,
) -> np.ndarray:
    """Per-channel wrapper for M0's single-channel projected interpolator.

    ``interpolate_surface``'s AET signature with the projected estimand:
    ``source`` is ``(3, n_source_time, n_source_frequency)`` and the result is
    ``(3, n_target_time, n_target_frequency)``.
    """
    return np.stack(
        [
            projected_interpolated_positive_surface(
                channel_surface,
                source_time,
                source_frequency,
                target_time,
                target_frequency,
                delta_f_hz,
                projection_nodes=projection_nodes,
                frequency_chunk=frequency_chunk,
                zero_outside_frequency=zero_outside_frequency,
            )
            for channel_surface in source
        ]
    )


def analytic_aet_noise_psd(
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Compute only the three analytic AET auto-PSDs from OMS/TM covariance."""
    oms_aet, tm_aet = analytic_aet_noise_components_psd(orbit_path, time_tcb, frequency_hz)
    return oms_aet + tm_aet


C_LIGHT_M_S = 299792458.0
TM_ACCELERATION_ASD = 2.4e-15
TM_F1_HZ, TM_F2_HZ = 4.0e-4, 8.0e-3
OMS_DISPLACEMENT_ASD = 7.9e-12
OMS_F3_HZ = 2.0e-3


def tm_theory_psd(frequency_hz: np.ndarray) -> np.ndarray:
    """Analytic test-mass acceleration noise PSD (Bayle & Hartwig eq. B3)."""
    f = np.asarray(frequency_hz, dtype=float)
    return (
        TM_ACCELERATION_ASD**2
        * (1.0 + (TM_F1_HZ / f) ** 2)
        * (1.0 + (f / TM_F2_HZ) ** 4)
        * (1.0 / (2.0 * np.pi * f * C_LIGHT_M_S)) ** 2
    )


def oms_theory_psd(frequency_hz: np.ndarray) -> np.ndarray:
    """Analytic OMS readout noise PSD (Bayle & Hartwig eq. B5)."""
    f = np.asarray(frequency_hz, dtype=float)
    return (
        OMS_DISPLACEMENT_ASD**2
        * (1.0 + (OMS_F3_HZ / f) ** 4)
        * (2.0 * np.pi * f / C_LIGHT_M_S) ** 2
    )


def aet_noise_transfer_functions(
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(T_tm, T_oms)``, each ``(3, n_time, n_frequency)``.

    Defined by ``T_i,c(t,f) = analytic_i,c(t,f) / S_i^theory(f)``, so that
    ``S_noise,c = T_tm,c S_TM + T_oms,c S_OMS`` reproduces the analytic
    surface exactly when the spectra equal their theoretical forms. All of the
    time dependence -- including the TDI nulls at ``n/(2L(t))``, which drift
    with the armlength -- lives in these transfer functions.

    Evaluated directly on the supplied grid, never interpolated from a coarse
    one: interpolating the analytic surface from the archive's 256-point
    frequency grid smears ~7.5 nats out of the null, which the free spline
    then cannot recover.
    """
    oms_aet, tm_aet = analytic_aet_noise_components_psd(
        orbit_path, time_tcb, frequency_hz
    )
    return (
        tm_aet / tm_theory_psd(frequency_hz)[None, None, :],
        oms_aet / oms_theory_psd(frequency_hz)[None, None, :],
    )


def interpolate_surface(
    source: np.ndarray,
    source_time: np.ndarray,
    source_frequency: np.ndarray,
    target_time: np.ndarray,
    target_frequency: np.ndarray,
) -> np.ndarray:
    """Interpolate positive channel surfaces in log PSD and log frequency."""
    output = []
    time_mesh, frequency_mesh = np.meshgrid(
        target_time, np.log(target_frequency), indexing="ij"
    )
    points = np.column_stack((time_mesh.ravel(), frequency_mesh.ravel()))
    for channel_surface in source:
        positive = channel_surface[channel_surface > 0.0]
        floor = float(np.min(positive)) * 1.0e-6
        interpolator = RegularGridInterpolator(
            (source_time, np.log(source_frequency)),
            np.log(np.maximum(channel_surface, floor)),
            bounds_error=False,
            fill_value=np.nan,
        )
        output.append(np.exp(interpolator(points)).reshape(time_mesh.shape))
    return np.stack(output)


def bin_channels(
    surfaces: np.ndarray,
    frequency_hz: np.ndarray,
    bin_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency-pool all channels without excluding response-null bins."""
    full_mask = np.ones(surfaces.shape[1:], dtype=bool)
    binned = []
    counts = []
    grouped_frequency = None
    for surface in surfaces:
        mean, channel_counts, grouped_frequency = masked_frequency_bin_mean(
            surface, full_mask, frequency_hz, bin_starts
        )
        binned.append(mean)
        counts.append(channel_counts)
    assert grouped_frequency is not None
    return np.stack(binned), np.stack(counts), grouped_frequency


def heldout_binned_diagnostics(
    observed_psd: np.ndarray,
    counts: np.ndarray,
    surface: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Truth-free held-out checks, the binned analogue of M0's.

    ``esa_m0_study.blind_whitening_diagnostics`` works on single WDM
    coefficients, where ``z = w / sqrt(S)`` is standard normal. M1's
    likelihood is already pooled, so a cell carries ``nu`` coefficients'
    summed power rather than one signed coefficient. The comparable
    quantities are therefore built from the normalized mean power
    ``u = (P / nu) / S``, which has unit expectation exactly as ``z^2``
    does, and the per-coefficient Whittle log score

        -0.5 * (log S + u)

    which reduces to M0's expression at ``nu = 1``. Both are directly
    comparable to M0's ``mean_z2`` and ``mean_whittle_log_score``.

    Two of M0's entries are deliberately absent rather than approximated:
    ``mean_z`` and the lag-one products need signed coefficients, which
    pooling discards. ``central_90_fraction`` is computed against the
    ``chi^2_nu`` quantiles appropriate to each cell's own count instead of
    the fixed 1.645 normal cut, so it remains a nominal-90% check.
    """
    values = np.asarray(observed_psd, dtype=float)
    weights = np.asarray(counts, dtype=float)
    model = np.asarray(surface, dtype=float)
    retained = np.asarray(mask, dtype=bool)
    if not (values.shape == weights.shape == model.shape == retained.shape):
        raise ValueError("observed, counts, surface, and mask must share shape")
    selected = (
        retained
        & np.isfinite(values)
        & np.isfinite(model)
        & (model > 0.0)
        & (weights > 0.0)
    )
    if not np.any(selected):
        raise ValueError("held-out diagnostics retain no valid cells")
    nu = weights[selected]
    u = values[selected] / model[selected]
    # P / S ~ chi^2_nu, so the nominal central 90% band is count-dependent.
    lower = chi2.ppf(0.05, nu)
    upper = chi2.ppf(0.95, nu)
    inside = (nu * u >= lower) & (nu * u <= upper)
    return {
        "n_cells": int(selected.sum()),
        "n_coefficients": float(nu.sum()),
        "mean_z2": float(np.mean(u)),
        "median_ratio_vs_chi2_nu_median": float(
            np.median(nu * u / chi2.ppf(0.5, nu))
        ),
        "central_90_fraction": float(np.mean(inside)),
        "mean_whittle_log_score": float(
            np.mean(-0.5 * (np.log(model[selected]) + u))
        ),
    }


def bin_time_rows(
    surfaces: np.ndarray,
    counts: np.ndarray,
    time_tcb: np.ndarray,
    time_bin_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count-weighted pooling of already frequency-binned surfaces over time.

    M0 evaluates its likelihood on ~512 time bins rather than every WDM row;
    M1 previously kept every row, which is only tolerable because it ran at
    ``nt=32`` (30 rows). At M0's ``nt=2048`` the rows must be pooled the same
    way for the two analyses to share cells.
    """
    values = np.asarray(surfaces, dtype=float)
    weights = np.asarray(counts, dtype=float)
    starts = np.asarray(time_bin_starts, dtype=int)
    stops = np.r_[starts[1:], values.shape[1]]
    pooled = np.empty((values.shape[0], starts.size, values.shape[2]))
    pooled_counts = np.empty_like(pooled)
    pooled_time = np.empty(starts.size)
    for index, (start, stop) in enumerate(zip(starts, stops)):
        block_weight = weights[:, start:stop, :]
        total = np.sum(block_weight, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled[:, index, :] = np.where(
                total > 0.0,
                np.sum(np.nan_to_num(values[:, start:stop, :]) * block_weight, axis=1)
                / np.maximum(total, 1.0),
                np.nan,
            )
        pooled_counts[:, index, :] = total
        pooled_time[index] = float(np.mean(time_tcb[start:stop]))
    return pooled, pooled_counts, pooled_time


def time_block_log_pilot(
    observed_psd: np.ndarray, n_blocks: int, *, frequency_smoothing_bins: int = 201
) -> np.ndarray:
    """Robust log-power pilot from the data alone, for adaptive bin selection.

    Splits the time rows into ``n_blocks`` contiguous blocks and takes the
    median log power within each, per channel and frequency: a block median is
    far steadier than a single row across TIME. But with only a handful of WDM
    time rows (here 30), each frequency channel's block median is still one
    noisy chi-square draw, uncorrelated with its neighbours -- median std stays
    ~0.4-1.0 nat even pooling all rows into one block, an order of magnitude
    above a useful merge threshold. Real spectral curvature, by contrast,
    varies slowly across many channels. A running median FILTER ACROSS
    FREQUENCY (matching ``coarse_null_mask``'s pattern) suppresses the
    channel-to-channel noise the greedy binner would otherwise chase, while
    leaving genuine narrow features (nulls) intact only if
    ``frequency_smoothing_bins`` is well under the feature width in channels.
    A mean filter, not a median filter, does the suppressing: the per-channel
    noise is close to i.i.d., so a moving average cuts its variance by a clean
    ``1 / frequency_smoothing_bins``, whereas a median filter (robust to
    outliers by design) barely dents it -- measured to leave 10x too many
    bins at ``n_blocks=6`` (5 time rows/block) even at this window.

    Returns a ``(n_blocks * n_channels, n_frequency)`` pilot, stacked so a bin
    must satisfy the log-range test in every block of every channel.
    """
    values = np.asarray(observed_psd, dtype=float)
    positive = values[values > 0.0]
    floor = float(np.min(positive)) * 1.0e-6 if positive.size else np.finfo(float).tiny
    log_power = np.log(np.maximum(values, floor))
    blocks = np.array_split(np.arange(values.shape[1]), max(1, n_blocks))
    window = max(1, frequency_smoothing_bins | 1)  # odd, so the filter is centred
    return np.concatenate(
        [
            uniform_filter1d(
                np.median(log_power[channel][rows], axis=0), size=window, mode="nearest"
            )[None, :]
            for channel in range(values.shape[0])
            for rows in blocks
            if rows.size
        ],
        axis=0,
    )


def coarse_null_mask(
    calibration_psd: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    continuum_window_hz: float = 5.0e-3,
    relative_threshold: float = 0.5,
    dilation_bins: int = 1,
) -> np.ndarray:
    """Flag response-null neighborhoods for diagnostics, not for fitting."""
    df = float(np.median(np.diff(frequency_hz)))
    window = max(5, int(round(continuum_window_hz / df)))
    window += 1 - window % 2
    excluded = np.zeros_like(calibration_psd, dtype=bool)
    structure = np.ones(2 * dilation_bins + 1, dtype=bool)
    for channel in range(calibration_psd.shape[0]):
        for row in range(calibration_psd.shape[1]):
            log_psd = np.log(calibration_psd[channel, row])
            continuum = median_filter(log_psd, size=window, mode="nearest")
            core = np.exp(log_psd - continuum) < relative_threshold
            excluded[channel, row] = binary_dilation(core, structure=structure)
    return excluded


def recovery_metrics(
    observed: np.ndarray,
    counts: np.ndarray,
    truth_noise: np.ndarray,
    truth_galactic: np.ndarray,
    posterior,
    *,
    whitening_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize fit-domain recovery and continuum-only accuracy/coverage.

    The likelihood counts remain unchanged: every valid bin is still fitted.
    ``whitening_mask`` selects the continuum cohort -- fitted cells minus the
    response-null corridors -- and per ESA_M0_METHOD_IMPROVEMENTS that cohort
    now carries the headline accuracy and coverage as well as the mean-z2
    scale check. Null corridors are where a point-evaluated reference PSD is
    least meaningful, so scoring interval calibration there conflates model
    error with the estimand being ill-defined. The all-cell values are
    reported alongside, suffixed ``_all_cells``, together with the excluded
    fraction, so the sensitivity to the mask stays visible.
    """
    output: dict[str, float] = {}
    truth_total = truth_noise + truth_galactic
    for channel_index, channel_name in enumerate(AET_CHANNELS):
        fit_valid = (
            (counts[channel_index] > 0.0)
            & np.isfinite(truth_total[channel_index])
        )
        if whitening_mask is None:
            whitening_valid = fit_valid
        else:
            diagnostic = np.asarray(whitening_mask, dtype=bool)
            if diagnostic.shape != observed.shape:
                raise ValueError("whitening_mask must match observed")
            whitening_valid = fit_valid & diagnostic[channel_index]
        if not np.any(whitening_valid):
            raise ValueError(f"no whitening cells for channel {channel_name}")
        # Accuracy and coverage are scored on the continuum cohort.
        noise_visible = whitening_valid & (
            truth_noise[channel_index] / truth_total[channel_index] >= 0.03
        )
        galactic_visible = whitening_valid & (
            truth_galactic[channel_index] / truth_total[channel_index] >= 0.03
        )

        def median_log_error(estimate, truth, selection):
            return float(
                np.median(np.abs(np.log(estimate[selection] / truth[selection])))
            )

        prefix = channel_name.lower()
        output[f"{prefix}_total_median_abs_log_error"] = median_log_error(
            posterior.total_median[channel_index],
            truth_total[channel_index],
            whitening_valid,
        )
        output[f"{prefix}_total_median_abs_log_error_all_cells"] = median_log_error(
            posterior.total_median[channel_index], truth_total[channel_index], fit_valid
        )
        output[f"{prefix}_noise_median_abs_log_error_visible"] = median_log_error(
            posterior.noise_median[channel_index],
            truth_noise[channel_index],
            noise_visible,
        )
        output[f"{prefix}_galactic_median_abs_log_error_visible"] = median_log_error(
            posterior.galactic_median[channel_index],
            truth_galactic[channel_index],
            galactic_visible,
        )
        output[f"{prefix}_continuum_mean_z2"] = float(
            np.sum(
                counts[channel_index][whitening_valid]
                * observed[channel_index][whitening_valid]
                / posterior.total_median[channel_index][whitening_valid]
            )
            / np.sum(counts[channel_index][whitening_valid])
        )
        def coverage(selection):
            return float(
                np.mean(
                    (truth_total[channel_index][selection] >= posterior.total_lower[channel_index][selection])
                    & (truth_total[channel_index][selection] <= posterior.total_upper[channel_index][selection])
                )
            )

        output[f"{prefix}_pointwise_90_coverage"] = coverage(whitening_valid)
        output[f"{prefix}_pointwise_90_coverage_all_cells"] = coverage(fit_valid)
        output[f"{prefix}_null_excluded_cell_fraction"] = float(
            1.0 - np.sum(whitening_valid) / max(np.sum(fit_valid), 1)
        )
        output[f"{prefix}_fit_effective_cells"] = float(
            np.sum(counts[channel_index][fit_valid])
        )
        output[f"{prefix}_whitening_effective_cells"] = float(
            np.sum(counts[channel_index][whitening_valid])
        )
    return output


def plot_surfaces(
    output_path: Path,
    frequency_hz: np.ndarray,
    counts: np.ndarray,
    truth_noise: np.ndarray,
    truth_galactic: np.ndarray,
    posterior,
    *,
    whitening_mask: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    for channel_index, channel_name in enumerate(AET_CHANNELS):
        valid_frequency = np.sum(counts[channel_index], axis=0) > 0.0
        frequency = frequency_hz[valid_frequency]
        truth_total = truth_noise[channel_index] + truth_galactic[channel_index]
        noise_time_median = np.nanmedian(truth_noise[channel_index], axis=0)
        galactic_time_median = np.nanmedian(truth_galactic[channel_index], axis=0)
        log_component_ratio = np.log(noise_time_median / galactic_time_median)
        sign_changes = np.flatnonzero(
            np.signbit(log_component_ratio[:-1])
            != np.signbit(log_component_ratio[1:])
        )
        crossover_index = int(
            sign_changes[-1] + 1
            if sign_changes.size
            else np.nanargmin(np.abs(log_component_ratio))
        )
        crossover_frequency = frequency_hz[crossover_index]
        axis = axes[channel_index, 0]
        axis.loglog(
            frequency,
            np.nanmedian(truth_total[:, valid_frequency], axis=0),
            "k--",
            label="injection total",
        )
        axis.loglog(
            frequency,
            np.nanmedian(posterior.total_median[channel_index][:, valid_frequency], axis=0),
            color="C3",
            label="posterior median",
        )
        axis.fill_between(
            frequency,
            np.nanmedian(posterior.total_lower[channel_index][:, valid_frequency], axis=0),
            np.nanmedian(posterior.total_upper[channel_index][:, valid_frequency], axis=0),
            color="C3",
            alpha=0.2,
            label="pointwise 90% band",
        )
        axis.axvline(
            crossover_frequency,
            color="0.25",
            ls=":",
            lw=1.2,
            label="noise = Galaxy",
        )
        if whitening_mask is not None:
            omitted_frequency = np.any(
                (counts[channel_index] > 0.0)
                & ~np.asarray(whitening_mask[channel_index], dtype=bool),
                axis=0,
            )[valid_frequency]
            for position in np.flatnonzero(omitted_frequency):
                left = frequency[position - 1] if position else frequency[position]
                right = (
                    frequency[position + 1]
                    if position + 1 < frequency.size
                    else frequency[position]
                )
                axis.axvspan(
                    np.sqrt(left * frequency[position]),
                    np.sqrt(frequency[position] * right),
                    color="0.5",
                    alpha=0.08,
                    lw=0,
                )
        axis.set(
            title=f"{channel_name}: total PSD",
            xlabel="frequency [Hz]",
            ylabel="PSD [Hz$^2$/Hz]",
        )
        axis.legend(fontsize=8)

        axis = axes[channel_index, 1]
        axis.loglog(
            frequency,
            np.nanmedian(truth_noise[channel_index][:, valid_frequency], axis=0),
            "--",
            color="C0",
            label="noise truth",
        )
        axis.loglog(
            frequency,
            np.nanmedian(posterior.noise_median[channel_index][:, valid_frequency], axis=0),
            color="C0",
            label="noise spline",
        )
        axis.loglog(
            frequency,
            np.nanmedian(truth_galactic[channel_index][:, valid_frequency], axis=0),
            "--",
            color="C1",
            label="Galactic truth",
        )
        axis.loglog(
            frequency,
            np.nanmedian(posterior.galactic_median[channel_index][:, valid_frequency], axis=0),
            color="C1",
            label="parametric Galactic",
        )
        axis.axvline(
            crossover_frequency,
            color="0.25",
            ls=":",
            lw=1.2,
            label="noise = Galaxy",
        )
        if whitening_mask is not None:
            omitted_frequency = np.any(
                (counts[channel_index] > 0.0)
                & ~np.asarray(whitening_mask[channel_index], dtype=bool),
                axis=0,
            )[valid_frequency]
            for position in np.flatnonzero(omitted_frequency):
                left = frequency[position - 1] if position else frequency[position]
                right = (
                    frequency[position + 1]
                    if position + 1 < frequency.size
                    else frequency[position]
                )
                axis.axvspan(
                    np.sqrt(left * frequency[position]),
                    np.sqrt(frequency[position] * right),
                    color="0.5",
                    alpha=0.08,
                    lw=0,
                )
        axis.set(
            title=f"{channel_name}: component split",
            xlabel="frequency [Hz]",
            ylabel="PSD [Hz$^2$/Hz]",
        )
        axis.legend(fontsize=8)
    fig.suptitle(
        "Diagonal A/E/T: all bins fitted; dotted line marks the noise--Galaxy crossover",
        y=1.01,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parameters(output_path: Path, posterior, injected_amplitude: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].scatter(
        posterior.amplitude_draws,
        1.0e3 * posterior.f_knee_draws_hz,
        s=8,
        alpha=0.25,
    )
    axes[0].axvline(injected_amplitude, color="k", ls="--")
    axes[0].axhline(2.15, color="k", ls="--")
    axes[0].set(xlabel="$A_{\\rm gal}$", ylabel="$f_{\\rm knee}$ [mHz]", title="joint posterior")
    axes[1].hist(posterior.amplitude_draws, bins=30, density=True, alpha=0.8)
    axes[1].axvline(injected_amplitude, color="k", ls="--", label="injected")
    axes[1].set(xlabel="$A_{\\rm gal}$", ylabel="density", title="shared amplitude")
    axes[1].legend()
    axes[2].hist(1.0e3 * posterior.f_knee_draws_hz, bins=30, density=True, alpha=0.8)
    axes[2].axvline(2.15, color="k", ls="--", label="injected")
    axes[2].set(xlabel="$f_{\\rm knee}$ [mHz]", ylabel="density", title="shared knee")
    axes[2].legend()
    fig.suptitle("Diagonal A/E/T shared Galactic parameters", y=1.02)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    surface_plot = Path(args.surface_plot).resolve()
    parameter_plot = Path(args.parameter_plot).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    surface_plot.parent.mkdir(parents=True, exist_ok=True)
    parameter_plot.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(archive, "r") as hdf:
        dt = float(hdf.attrs["dt_seconds"])
        t0_tcb = float(hdf.attrs["t0_tcb"])
        n_archive = int(hdf.attrs["n_samples"])
        n_total = wdm_valid_length(n_archive, args.nt)
        xyz_total = hdf["tdi/total"][:, :n_total]
        truth_time = hdf["truth/time_tcb"][:]
        truth_frequency = hdf["truth/frequency_hz"][:]
        truth_galactic_xyz = hdf["truth/galactic_psd"][:]
        injected_amplitude = float(hdf.attrs["galactic_amplitude_scale"])

    aet_total = xyz_to_aet_series(xyz_total)
    del xyz_total
    nf = n_total // args.nt
    df = 1.0 / (2.0 * nf * dt)
    trim_low = max(1, int(np.ceil(args.fmin_hz / df)))
    trim_high = max(0, nf - int(np.floor(args.fmax_hz / df)))
    config = PSplineConfig(
        n_interior_knots_time=args.time_knots,
        n_interior_knots_freq=args.frequency_knots,
        trim_time_bins=1,
        trim_low_freq_channels=trim_low,
        trim_high_freq_channels=trim_high,
        freq_knot_strategy="log",
        centered=True,
    )

    coefficients = []
    fit_time_grid = None
    fit_frequency = None
    for channel_series in aet_total:
        channel_coefficients, channel_time, channel_frequency = wdm_analysis_coefficients(
            channel_series, dt, args.nt, config
        )
        coefficients.append(channel_coefficients)
        if fit_time_grid is None:
            fit_time_grid = channel_time
            fit_frequency = channel_frequency
        else:
            np.testing.assert_array_equal(channel_time, fit_time_grid)
            np.testing.assert_array_equal(channel_frequency, fit_frequency)
    del aet_total
    coefficients = np.stack(coefficients)
    assert fit_time_grid is not None and fit_frequency is not None
    band = fit_frequency <= args.component_fmax_hz
    coefficients = coefficients[:, :, band]
    fit_frequency = fit_frequency[band]
    absolute_time = t0_tcb + np.asarray(fit_time_grid) * n_total * dt
    time_days = (absolute_time - t0_tcb) / 86400.0

    if args.analytic_calibration:
        # E[w_nm^2] = N * S / (2 dt), so the WDM-to-PSD constant is exactly the
        # sample count N, identical in every frequency channel. Verified against
        # the Monte Carlo estimator: its mean converges to N (1.000123, 1.000017,
        # 0.999966 at 16/64/256 draws) and its per-channel spread is pure sampling
        # noise -- std(log) 0.0646/0.0324/0.0162 versus the predicted
        # sqrt(2/(n_draws*n_time)) 0.0645/0.0323/0.0161, i.e. ratio 1.00 in every
        # case, with no residual structure. Estimating this constant by Monte
        # Carlo therefore only injects a ~6.5% per-channel multiplicative error
        # (at the pilot's 16 draws) into every observed PSD value.
        wdm_variance_calibration = np.full(int(np.sum(band)), float(n_total))
    else:
        wdm_variance_calibration = wdm_white_noise_calibration(
            n_total, dt, args.nt, config,
            n_draws=args.calibration_draws, seed=args.calibration_seed,
        )[band]
    conversion = 2.0 * dt / wdm_variance_calibration[None, None, :]
    observed_psd = coefficients**2 * conversion
    del coefficients

    # The analytic OMS/TM components enter the M1 likelihood directly, each
    # with one free per-channel amplitude (see tv_pspline_psd.multichannel);
    # their sum is also retained for simulation-truth recovery and the
    # response-null diagnostic.
    #
    # PROJECTED onto the WDM frequency kernel, matching esa_m0_study.py's
    # estimand: a WDM coefficient measures the squared-Meyer-window average of
    # S across its atom's compact support, not S at the cell centre. Where the
    # spectrum is smooth the two agree, but inside the drifting TDI null comb
    # they differ by orders of magnitude, and comparing a fit to the wrong one
    # is what made the X2 M0 anchor read log-RMSE 0.81 / coverage 0.19 before
    # the projection landed (0.05 / 0.78 after).
    #
    # Evaluated DIRECTLY on the fine analysis grid (absolute_time,
    # fit_frequency), never interpolated from the archive's coarse 256-point
    # truth_frequency grid: that interpolation smears ~7.5 nats out of the
    # TDI null (see the null-smearing diagnostic), which then propagates into
    # both the truth-comparison arrays AND, since bin_channels frequency-
    # averages this same array, into the fit's own transfer functions.
    # frequency_chunk keeps this from allocating the full fine-grid complex
    # covariance tensor at once.
    #
    # The transfer functions built from these (T_i = projected_i / S_i^theory)
    # keep the exact factorization at S_i = S_i^theory, so the component model
    # still reproduces the projected analytic surface identically. This relies
    # on S_TM/S_OMS being smooth within one atom -- they carry no null
    # structure, which all lives in T -- so proj[T S] = proj[T] S(f_m) holds.
    oms_reference_unbinned, tm_reference_unbinned = (
        projected_aet_noise_components_psd(
            archive.parent / "noise2a" / "orbits.h5",
            absolute_time,
            fit_frequency,
            df,
            projection_nodes=args.wdm_projection_nodes,
            frequency_chunk=args.reference_frequency_chunk,
        )
    )
    noise_reference_unbinned = oms_reference_unbinned + tm_reference_unbinned
    # zero_outside_frequency: the Galactic component was generated only over
    # the archive band, so quadrature nodes beyond it carry no power.
    galactic_truth_unbinned = projected_aet_interpolated_surface(
        diagonal_xyz_psd_to_aet(truth_galactic_xyz),
        truth_time,
        truth_frequency,
        absolute_time,
        fit_frequency,
        df,
        projection_nodes=args.wdm_projection_nodes,
        frequency_chunk=args.reference_frequency_chunk,
        zero_outside_frequency=True,
    )
    galactic_template_unbinned = galactic_truth_unbinned / injected_amplitude

    # Shared adaptive bins: fine where the spectrum varies sharply (the
    # drifting TDI null), coarse on the smooth continuum. This replaces
    # fixed-size pooling, which smears the null wider than its physical width
    # and stiffens the likelihood against the spline.
    # Prospective split, matching M0: seven-fold repeating blocks with the
    # validation and test folds excluded from the likelihood AND the bin pilot,
    # so no reported score is in-sample (ESA_M0_METHOD_IMPROVEMENTS item 3).
    training_rows, validation_rows, test_rows = analysis_row_split(
        absolute_time.size,
        block=args.split_block,
        cycle=args.split_cycle,
        validation_fold=args.validation_fold,
        test_fold=args.test_fold,
    )
    print(
        f"row split: {training_rows.sum()} training / {validation_rows.sum()} "
        f"validation / {test_rows.sum()} test of {absolute_time.size} WDM rows"
    )

    if args.truth_free_bins:
        # M0's own pilot, reused verbatim so both analyses build bins the same
        # way: robust time-profile medians of retained TRAINING coefficients,
        # then a frequency median filter. Never sees truth, validation or test.
        pilot_log_psd = np.concatenate(
            [
                training_data_pilot_log_psd(
                    np.sqrt(np.maximum(observed_psd[c], 0.0)),
                    np.broadcast_to(
                        training_rows[:, None], observed_psd[c].shape
                    ),
                    n_time_profiles=args.pilot_time_profiles,
                    frequency_width=args.pilot_frequency_width,
                )
                for c in range(len(AET_CHANNELS))
            ],
            axis=0,
        )
    else:
        pilot_log_psd = np.concatenate(
            [np.log(noise_reference_unbinned[c]) for c in range(len(AET_CHANNELS))],
            axis=0,
        )
    bin_starts = adaptive_frequency_bin_starts(
        pilot_log_psd,
        max_log_range=args.adaptive_max_log_range,
        max_bin=args.adaptive_max_bin,
    )
    print(
        f"adaptive bins: {bin_starts.size} from {fit_frequency.size} WDM channels "
        f"({fit_frequency.size / max(bin_starts.size, 1):.0f}x merge)"
    )
    observed_binned, counts, grouped_frequency = bin_channels(
        observed_psd, fit_frequency, bin_starts
    )
    noise_reference_binned, _, _ = bin_channels(
        noise_reference_unbinned, fit_frequency, bin_starts
    )
    oms_reference_binned, _, _ = bin_channels(
        oms_reference_unbinned, fit_frequency, bin_starts
    )
    tm_reference_binned, _, _ = bin_channels(
        tm_reference_unbinned, fit_frequency, bin_starts
    )
    noise_truth_binned = noise_reference_binned.copy()
    galactic_truth_binned, _, _ = bin_channels(
        galactic_truth_unbinned, fit_frequency, bin_starts
    )
    galactic_template_binned, _, _ = bin_channels(
        galactic_template_unbinned, fit_frequency, bin_starts
    )
    del observed_psd, noise_reference_unbinned, galactic_truth_unbinned
    del galactic_template_unbinned, oms_reference_unbinned, tm_reference_unbinned

    # Pool time rows the way M0 does (~512 bins). Held-out rows get zero weight
    # so they never enter the likelihood, and every surface is pooled with the
    # SAME weights so a bin means the same cells in the data and the truth.
    if args.time_bin > 1:
        time_bin_starts = partition_starts(
            absolute_time.size, args.time_bin, training_rows
        )
        likelihood_counts = counts * training_rows[None, :, None]
        pooled = {}
        for name, surface in (
            ("observed", observed_binned),
            ("noise_reference", noise_reference_binned),
            ("oms_reference", oms_reference_binned),
            ("tm_reference", tm_reference_binned),
            ("galactic_truth", galactic_truth_binned),
            ("galactic_template", galactic_template_binned),
        ):
            pooled[name], pooled_counts, pooled_time = bin_time_rows(
                surface, likelihood_counts, absolute_time, time_bin_starts
            )
        # Pool the excluded rows separately, on the SAME bin boundaries and
        # with the same count weighting, so each held-out cohort lands in the
        # bins the fit already predicts. Without this the held-out rows are not
        # merely down-weighted, they are absent from every output array: the
        # likelihood pooling gives them zero weight.
        # The analytic reference is pooled per cohort as well: partition_starts
        # splits at holdout transitions, so a held-out bin has zero training
        # weight and the training-pooled reference is NaN exactly there.
        heldout_pools = {}
        for cohort_name, cohort_rows in (
            ("validation", validation_rows),
            ("test", test_rows),
        ):
            cohort_weights = counts * cohort_rows[None, :, None]
            cohort_observed, cohort_counts, _ = bin_time_rows(
                observed_binned, cohort_weights, absolute_time, time_bin_starts
            )
            cohort_reference, _, _ = bin_time_rows(
                noise_reference_binned,
                cohort_weights,
                absolute_time,
                time_bin_starts,
            )
            heldout_pools[cohort_name] = (
                cohort_observed,
                cohort_counts,
                cohort_reference,
            )

        observed_binned = pooled["observed"]
        noise_reference_binned = pooled["noise_reference"]
        oms_reference_binned = pooled["oms_reference"]
        tm_reference_binned = pooled["tm_reference"]
        galactic_truth_binned = pooled["galactic_truth"]
        galactic_template_binned = pooled["galactic_template"]
        noise_truth_binned = noise_reference_binned.copy()
        counts = pooled_counts
        absolute_time = pooled_time
        time_days = (absolute_time - t0_tcb) / 86400.0
        print(
            f"time bins: {absolute_time.size} from {training_rows.size} WDM rows "
            f"({training_rows.sum()} training)"
        )
    else:
        # No time pooling: rows are their own bins, so a cohort is selected by
        # zeroing the counts of every row outside it.
        heldout_pools = {
            cohort_name: (
                observed_binned,
                counts * cohort_rows[None, :, None],
                noise_reference_binned,
            )
            for cohort_name, cohort_rows in (
                ("validation", validation_rows),
                ("test", test_rows),
            )
        }

    diagnostic_null_mask = coarse_null_mask(
        noise_reference_binned, grouped_frequency
    )
    fit_valid = counts > 0.0
    if args.t_channel_fmin_hz > 0.0:
        # The analytic T-channel noise assumes equal arms and perfect laser
        # noise cancellation, so its low-frequency null (8 sin^4(w/2) S_TM ~ f^4)
        # is far deeper than a real drifting-arm constellation achieves. The
        # archive's simulated T exceeds the analytic model by up to ~20x below
        # 1 mHz while A and E agree to <3%. Since S_TM/S_OMS are shared across
        # channels, the fit cannot repair T without corrupting A/E, so the
        # mismatch is pushed into the shared Galactic amplitude.
        t_index = AET_CHANNELS.index("T")
        fit_valid[t_index] &= grouped_frequency[None, :] >= args.t_channel_fmin_hz
    whitening_mask = fit_valid & ~diagnostic_null_mask
    # Held-out cells must NOT inherit fit_valid: partition_starts splits bins
    # at holdout transitions, so a held-out bin carries zero training counts
    # and fit_valid is false there by construction. Reusing it would silently
    # score an empty cohort. Apply only the cuts that are statements about the
    # model's validity -- the response-null corridors and the T-channel
    # low-frequency exclusion -- and let each cohort's own counts decide which
    # cells it actually populates.
    heldout_eligible = ~diagnostic_null_mask
    if args.t_channel_fmin_hz > 0.0:
        heldout_eligible[AET_CHANNELS.index("T")] &= (
            grouped_frequency[None, :] >= args.t_channel_fmin_hz
        )
    warped_frequency_hz = None
    if args.frequency_warp:
        arm_ratio = armlength_ratio(
            archive.parent / "noise2a" / "orbits.h5", absolute_time
        )
        warped_frequency_hz = grouped_frequency[None, :] * arm_ratio[:, None]

    if args.component_noise:
        # Transfer functions from the SAME bin-averaged reference used for
        # truth comparison (oms/tm_reference_binned), not a fresh point
        # evaluation at the bin's nominal frequency. Those differ badly near
        # a null: the null core is ~2 mHz wide and highly nonlinear, while a
        # bin can be sub-mHz to ~1 mHz wide, so bin-averaging the fine-grid
        # reference in linear power (matching how the observed data itself
        # was pooled) gives a materially different, and correct, value from
        # sampling the analytic model once at the bin center. Using the
        # latter for the fit while comparing against the former for "truth"
        # silently miscalibrated the likelihood across the whole
        # null-adjacent band -- not just the narrow core a response-null mask
        # is built to catch -- because the *data* term was bin-averaged
        # correctly but the *model* term was not.
        theory_tm = tm_theory_psd(grouped_frequency)[None, None, :]
        theory_oms = oms_theory_psd(grouped_frequency)[None, None, :]
        transfer_tm = tm_reference_binned / theory_tm
        transfer_oms = oms_reference_binned / theory_oms
        posterior = fit_aet_component_noise_nuts(
            np.where(fit_valid, observed_binned, 1.0),
            np.where(fit_valid, counts, 1.0),
            grouped_frequency,
            transfer_tm=np.where(fit_valid, transfer_tm, 1.0),
            transfer_oms=np.where(fit_valid, transfer_oms, 1.0),
            tm_theory_psd=tm_theory_psd(grouped_frequency),
            oms_theory_psd=oms_theory_psd(grouped_frequency),
            galactic_template_psd=np.where(fit_valid, galactic_template_binned, 0.0),
            mask=fit_valid,
            n_frequency_knots=args.component_frequency_knots,
            phi_tm=args.phi_tm,
            phi_oms=args.phi_oms,
            fit_t_null_leakage=args.fit_t_null_leakage,
            t_leakage_centre=args.t_leakage_centre,
            n_warmup=args.warmup,
            n_samples=args.samples,
            num_chains=2,
            target_accept_probability=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            progress_bar=not args.no_progress,
        )
    else:
        posterior = fit_aet_diagonal_nuts(
            np.where(fit_valid, observed_binned, 1.0),
            np.where(fit_valid, counts, 1.0),
            absolute_time,
            grouped_frequency,
            oms_reference_psd=np.where(fit_valid, oms_reference_binned, 1.0),
            tm_reference_psd=np.where(fit_valid, tm_reference_binned, 1.0),
            galactic_template_psd=np.where(fit_valid, galactic_template_binned, 0.0),
            mask=fit_valid,
            n_time_knots=args.time_knots,
            n_frequency_knots=args.frequency_knots,
            warped_frequency_hz=warped_frequency_hz,
            share_noise_residual=args.share_noise_residual,
            delta_log_sd=args.delta_log_sd,
            phi_time=args.phi_time,
            phi_frequency=args.phi_frequency,
            noise_level_log_sd=args.noise_level_log_sd,
            n_warmup=args.warmup,
            n_samples=args.samples,
            num_chains=2,
            target_accept_probability=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            progress_bar=not args.no_progress,
        )
    diagnostics = posterior.diagnostics
    posterior_usable = bool(
        diagnostics["divergences"] == 0
        and np.isfinite(diagnostics["max_r_hat"])
        and diagnostics["max_r_hat"] <= 1.05
        and diagnostics["min_effective_sample_size"] >= 50.0
        and diagnostics["tree_depth_saturation_fraction"] <= 0.05
        and diagnostics["min_ebfmi"] >= 0.3
    )
    if min(args.warmup, args.samples) < 100:
        posterior_status = "smoke_only"
    elif posterior_usable:
        posterior_status = "converged_candidate"
    else:
        posterior_status = "failed_diagnostics"
    metrics = recovery_metrics(
        observed_binned,
        # Zero the counts of masked cells so the metrics score only what the
        # likelihood actually saw; otherwise dropped cells are graded against a
        # floored template and produce meaningless |log error| values.
        np.where(fit_valid, counts, 0.0),
        noise_truth_binned,
        galactic_truth_binned,
        posterior,
        whitening_mask=whitening_mask,
    )

    # Held-out scores on the rows the likelihood never saw. Until this existed
    # every M1 number was in-sample while M0's were not, so the two models'
    # rows in a ladder table were not measuring the same thing. Bands match
    # esa_m0_study.py exactly so the tables can sit side by side.
    bands = {
        "low": grouped_frequency <= 3.0e-3,
        "retained_full": np.ones(grouped_frequency.size, dtype=bool),
        "high_continuum": grouped_frequency >= args.null_min_frequency,
    }
    heldout_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for cohort_name, (
        cohort_observed,
        cohort_counts,
        cohort_reference,
    ) in heldout_pools.items():
        heldout_scores[cohort_name] = {}
        for channel_index, channel_name in enumerate(AET_CHANNELS):
            channel_scores: dict[str, dict[str, float]] = {}
            for band_name, band_select in bands.items():
                cohort_mask = (
                    heldout_eligible[channel_index]
                    & (cohort_counts[channel_index] > 0.0)
                    & np.isfinite(cohort_observed[channel_index])
                    & band_select[None, :]
                )
                if not np.any(cohort_mask):
                    continue
                # The analytic OMS+TM surface is M1's zero-parameter physics
                # model, the same role M0's reference plays, so the gain is
                # comparable between the two ladders.
                reference = heldout_binned_diagnostics(
                    cohort_observed[channel_index],
                    cohort_counts[channel_index],
                    cohort_reference[channel_index],
                    cohort_mask,
                )
                fitted = heldout_binned_diagnostics(
                    cohort_observed[channel_index],
                    cohort_counts[channel_index],
                    posterior.total_median[channel_index],
                    cohort_mask,
                )
                fitted["mean_log_score_gain_vs_reference"] = float(
                    fitted["mean_whittle_log_score"]
                    - reference["mean_whittle_log_score"]
                )
                channel_scores[band_name] = {
                    "reference_only": reference,
                    "component_model": fitted,
                }
            heldout_scores[cohort_name][channel_name] = channel_scores
    for channel_name, channel_scores in heldout_scores.get("test", {}).items():
        full = channel_scores.get("retained_full")
        if full is None:
            continue
        fitted = full["component_model"]
        print(
            f"held-out test {channel_name}: mean_z2={fitted['mean_z2']:.4f} "
            f"central90={fitted['central_90_fraction']:.4f} "
            f"score_gain={fitted['mean_log_score_gain_vs_reference']:+.4f}"
        )
    plot_surfaces(
        surface_plot,
        grouped_frequency,
        counts,
        noise_truth_binned,
        galactic_truth_binned,
        posterior,
        whitening_mask=whitening_mask,
    )
    plot_parameters(parameter_plot, posterior, injected_amplitude)

    np.savez_compressed(
        output,
        channels=np.asarray(AET_CHANNELS),
        time_days=time_days,
        frequency_hz=grouped_frequency,
        requested_fmin_hz=args.fmin_hz,
        requested_fmax_hz=args.fmax_hz,
        component_fmax_hz=args.component_fmax_hz,
        fitted_frequency_min_hz=float(grouped_frequency.min()),
        fitted_frequency_max_hz=float(grouped_frequency.max()),
        counts=counts,
        fit_mask=fit_valid,
        whitening_mask=whitening_mask,
        diagnostic_null_mask=diagnostic_null_mask,
        observed_total=observed_binned,
        diagnostic_noise_reference=noise_reference_binned,
        oms_reference_psd=oms_reference_binned,
        tm_reference_psd=tm_reference_binned,
        truth_noise=noise_truth_binned,
        truth_galactic=galactic_truth_binned,
        posterior_noise=posterior.noise_median,
        posterior_noise_lower=posterior.noise_lower,
        posterior_noise_upper=posterior.noise_upper,
        posterior_galactic=posterior.galactic_median,
        posterior_galactic_lower=posterior.galactic_lower,
        posterior_galactic_upper=posterior.galactic_upper,
        posterior_total=posterior.total_median,
        posterior_total_lower=posterior.total_lower,
        posterior_total_upper=posterior.total_upper,
        amplitude_draws=posterior.amplitude_draws,
        f_knee_hz_draws=posterior.f_knee_draws_hz,
        t_leakage_draws=posterior.samples.get(
            "log_t_leakage", np.zeros(0)
        ),
        t_leakage_slope_draws=posterior.samples.get(
            "t_leakage_slope", np.zeros(0)
        ),
        injected_amplitude=injected_amplitude,
        injected_f_knee_hz=2.15e-3,
        phi_time=args.phi_time,
        phi_frequency=args.phi_frequency,
        noise_level_log_sd=args.noise_level_log_sd,
        diagnostic_names=np.asarray(list(diagnostics)),
        diagnostic_values=np.asarray(list(diagnostics.values()), dtype=float),
        metric_names=np.asarray(list(metrics)),
        metric_values=np.asarray(list(metrics.values())),
        approximation=np.asarray(
            "product of diagonal AET Whittle likelihoods; AET CSD ignored; "
            "diagonal PSDs rotated under zero-XYZ-CSD archive contract; "
            + (
                "noise is built from the physical TM/OMS frequency spectra "
                "through the orbit-derived TDI transfer functions "
                "(S_noise,c = T_TM,c S_TM + T_OMS,c S_OMS), so all time "
                "dependence including the drifting null comes from the orbits "
                "and no free time-dependent noise parameters are sampled; "
                "component amplitudes are not sampled, being redundant with "
                "the frequency splines"
                if args.component_noise
                else "noise is the analytic OMS+TM surface as a fixed offset "
                "times a free time-frequency P-spline residual"
            )
            + "; estimand is the WDM-frequency-kernel projected marginal power "
            f"({args.wdm_projection_nodes} quadrature nodes per atom), "
            "matching esa_m0_study.py; "
            "all valid bins fitted; response-null neighborhoods omitted only "
            "from continuum-whitening summaries"
        ),
        heldout_scores_json=np.asarray(json.dumps(heldout_scores)),
        wdm_projection_nodes=np.asarray(args.wdm_projection_nodes),
        wdm_projection_delta_f_hz=np.asarray(df),
        wdm_projection_kernel=np.asarray(
            "squared compact-support Meyer frequency window"
        ),
        posterior_usable=posterior_usable,
        posterior_status=np.asarray(posterior_status),
        convergence_gate=np.asarray(
            "divergences=0; finite max R-hat<=1.05; min ESS>=50; "
            "tree-depth saturation<=0.05; min E-BFMI>=0.3"
        ),
    )
    result = {
        "output": str(output),
        "surface_plot": str(surface_plot),
        "parameter_plot": str(parameter_plot),
        "runtime_seconds": posterior.runtime_seconds,
        "posterior_status": posterior_status,
        "posterior_usable": posterior_usable,
        "amplitude_median": float(np.median(posterior.amplitude_draws)),
        "amplitude_90": np.quantile(posterior.amplitude_draws, [0.05, 0.95]).tolist(),
        "f_knee_mhz_median": float(1.0e3 * np.median(posterior.f_knee_draws_hz)),
        "f_knee_mhz_90": (
            1.0e3 * np.quantile(posterior.f_knee_draws_hz, [0.05, 0.95])
        ).tolist(),
        "whitening_omitted_fraction_by_channel": {
            channel: float(np.mean(diagnostic_null_mask[index]))
            for index, channel in enumerate(AET_CHANNELS)
        },
        "heldout_scores": heldout_scores,
        "heldout_score_contract": (
            "cohort rows excluded from the likelihood, pooled on the fit's own "
            "time bins; u = (P/nu)/S with unit expectation, per-coefficient "
            "Whittle log score -0.5(log S + u), central 90% against chi^2_nu; "
            "gain is versus the analytic OMS+TM reference. Comparable to "
            "esa_m0_study.py's blind_diagnostics; mean_z and lag-one products "
            "are omitted because pooling discards coefficient signs"
        ),
        **diagnostics,
        **metrics,
    }
    print(json.dumps(result, indent=2))
    if args.require_convergence and not posterior_usable:
        raise RuntimeError(
            f"posterior failed the required convergence gate: {diagnostics}"
        )
    return result


def parser() -> argparse.ArgumentParser:
    output_dir = HERE / "pspline_aet_plots"
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--archive", default=HERE / "combined_esa_xyz.h5")
    argument_parser.add_argument(
        "--output",
        default=HERE / "component_models_aet_diagonal_fullband_fitall_weakcal_nuts.npz",
    )
    argument_parser.add_argument(
        "--surface-plot",
        default=output_dir / "aet_diagonal_fullband_fitall_component_recovery.png",
    )
    argument_parser.add_argument(
        "--parameter-plot",
        default=output_dir / "aet_diagonal_fullband_fitall_parameter_posterior.png",
    )
    argument_parser.add_argument("--nt", type=int, default=2048,
        help="WDM time pixels; 2048 matches the M0 study so the two share cells")
    argument_parser.add_argument("--fmin-hz", type=float, default=1.0e-4)
    argument_parser.add_argument("--fmax-hz", type=float, default=1.0e-1)
    argument_parser.add_argument("--component-fmax-hz", type=float, default=1.0e-1)
    argument_parser.add_argument(
        "--adaptive-max-log-range",
        type=float,
        default=0.25,
        help="greedy frequency-bin merge threshold on the noise-reference log-PSD range",
    )
    argument_parser.add_argument(
        "--adaptive-max-bin", type=int, default=24, help="widest allowed adaptive frequency bin"
    )
    argument_parser.add_argument(
        "--truth-free-bins",
        dest="truth_free_bins",
        action="store_true",
        default=True,
        help=(
            "choose adaptive frequency bins from robust time-block medians of the "
            "observed log power instead of the analytic reference, so the bin "
            "layout carries no truth information"
        ),
    )
    argument_parser.add_argument(
        "--no-truth-free-bins", dest="truth_free_bins", action="store_false"
    )
    argument_parser.add_argument(
        "--pilot-time-profiles", type=int, default=32,
        help="time profiles for M0's truth-free bin-selection pilot",
    )
    argument_parser.add_argument(
        "--pilot-frequency-width", type=int, default=31,
        help="frequency median-filter width for M0's bin-selection pilot",
    )
    argument_parser.add_argument(
        "--time-bin", type=int, default=4,
        help="WDM rows pooled per likelihood time bin (M0 uses 4)",
    )
    argument_parser.add_argument("--split-block", type=int, default=4)
    argument_parser.add_argument("--split-cycle", type=int, default=7)
    argument_parser.add_argument("--validation-fold", type=int, default=5)
    argument_parser.add_argument("--test-fold", type=int, default=6)
    argument_parser.add_argument("--time-knots", type=int, default=3)
    argument_parser.add_argument("--frequency-knots", type=int, default=40)
    argument_parser.add_argument(
        "--frequency-warp",
        dest="frequency_warp",
        action="store_true",
        default=True,
        help="warp frequency by the orbit armlength ratio so the drifting TDI null is stationary",
    )
    argument_parser.add_argument(
        "--no-frequency-warp", dest="frequency_warp", action="store_false"
    )
    argument_parser.add_argument(
        "--share-noise-residual",
        dest="share_noise_residual",
        action="store_true",
        default=True,
        help=(
            "one noise spline shared across A/E/T plus a per-channel scalar "
            "offset, instead of a free spline per channel (the per-channel "
            "spline is measured to absorb up to 99.8%% of the Galactic "
            "amplitude direction)"
        ),
    )
    argument_parser.add_argument(
        "--no-share-noise-residual", dest="share_noise_residual", action="store_false"
    )
    argument_parser.add_argument(
        "--delta-log-sd",
        type=float,
        default=0.3,
        help="prior SD for the per-channel log noise offset when --share-noise-residual",
    )
    argument_parser.add_argument(
        "--component-noise",
        dest="component_noise",
        action="store_true",
        default=True,
        help=(
            "build the noise from the physical TM/OMS spectra through the known "
            "TDI transfer functions (all time dependence, including the drifting "
            "null, comes from the orbits) instead of a free time-frequency spline"
        ),
    )
    argument_parser.add_argument(
        "--no-component-noise", dest="component_noise", action="store_false"
    )
    argument_parser.add_argument(
        "--component-frequency-knots",
        type=int,
        default=12,
        help="interior knots for each of the TM and OMS frequency splines",
    )
    argument_parser.add_argument(
        "--phi-tm",
        type=float,
        default=1.0e8,
        help=(
            "penalty precision for the test-mass spline; tight by default because "
            "T is ~27x less sensitive to TM than OMS, so TM is what a low-frequency "
            "signal can imitate"
        ),
    )
    argument_parser.add_argument("--phi-oms", type=float, default=1.0e4)
    argument_parser.add_argument(
        "--fit-t-null-leakage",
        dest="fit_t_null_leakage",
        action="store_true",
        default=True,
        help=(
            "fit one free parameter for imperfect test-mass cancellation in T "
            "(S_T gains kappa * T_TM,A * S_TM); T is a near-null so a sub-percent "
            "modelling error there becomes order-unity"
        ),
    )
    argument_parser.add_argument(
        "--no-fit-t-null-leakage", dest="fit_t_null_leakage", action="store_false"
    )
    argument_parser.add_argument(
        "--t-leakage-centre", type=float, default=3.0e-5,
        help="prior centre for the T null-leakage fraction kappa",
    )
    argument_parser.add_argument(
        "--reference-frequency-chunk", type=int, default=96,
        help=(
            "quadrature frequencies evaluated per chunk when projecting the "
            "analytic reference; must be at least --wdm-projection-nodes. Left "
            "at 96 (not M0's 384) because this job requests less memory: it "
            "only costs loop iterations"
        ),
    )
    argument_parser.add_argument(
        "--null-min-frequency", type=float, default=0.02,
        help=(
            "lower edge of the high-continuum band for held-out scores; must "
            "match esa_m0_study.py's value for the two ladders to share bands"
        ),
    )
    argument_parser.add_argument(
        "--wdm-projection-nodes", type=int, default=16,
        help=(
            "quadrature nodes per WDM frequency atom for the projected "
            "estimand. Must match the M0 runs this fit is tabulated beside"
        ),
    )
    argument_parser.add_argument("--phi-time", type=float, default=100.0)
    argument_parser.add_argument("--phi-frequency", type=float, default=100.0)
    argument_parser.add_argument(
        "--noise-level-log-sd",
        "--calibration-log-sd",
        dest="noise_level_log_sd",
        type=float,
        default=5.0,
        help=(
            "prior SD for the unpenalized log-spline modes; the legacy option "
            "name is retained as an alias"
        ),
    )
    argument_parser.add_argument("--warmup", type=int, default=300)
    argument_parser.add_argument("--samples", type=int, default=300)
    argument_parser.add_argument("--target-accept", type=float, default=0.95)
    argument_parser.add_argument("--max-tree-depth", type=int, default=10)
    argument_parser.add_argument(
        "--wdm-calibration-draws",
        "--calibration-draws",
        dest="calibration_draws",
        type=int,
        default=16,
        help="white-noise Monte Carlo draws used only for WDM-to-PSD conversion",
    )
    argument_parser.add_argument(
        "--t-channel-fmin-hz",
        type=float,
        default=0.0,
        help=(
            "drop T-channel cells below this frequency; the equal-arm analytic "
            "T noise model is invalid there (simulated/analytic ~20x at 0.1 mHz)"
        ),
    )
    argument_parser.add_argument(
        "--analytic-calibration",
        dest="analytic_calibration",
        action="store_true",
        default=True,
        help=(
            "use the exact WDM-to-PSD constant N instead of estimating it by "
            "Monte Carlo (the MC estimator converges to N and its per-channel "
            "spread is pure sampling noise)"
        ),
    )
    argument_parser.add_argument(
        "--no-analytic-calibration", dest="analytic_calibration", action="store_false"
    )
    argument_parser.add_argument(
        "--calibration-seed",
        type=int,
        default=20260806,
        help="seed for the Monte Carlo WDM->PSD calibration",
    )
    argument_parser.add_argument("--no-progress", action="store_true")
    argument_parser.add_argument(
        "--require-convergence",
        action="store_true",
        help="raise after saving outputs when the posterior diagnostics fail",
    )
    return argument_parser


if __name__ == "__main__":
    run(parser().parse_args())
