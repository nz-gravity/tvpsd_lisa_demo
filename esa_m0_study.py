"""Full-band M0 WDM/P-spline study for the archived ESA-orbit LISA data.

The study keeps the requested 1e-4--1e-1 Hz domain visible. The analytic
response and simulation truth are projected through the compact
WDM frequency kernel before they enter inference or validation. Response-null
cells remain in the likelihood; a separately recorded null mask is used only
for stable relative-error and continuum-whitening summaries.

Part A fits the uninterrupted X2 realization.  Part B applies LISA-like gaps
and cosine tapers in the time domain before the WDM transform.  Both parts use
the same underlying ESA-orbit realization; different Part-B seeds are gap
schedules, not independent noise realizations.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np
from backgrounds import noise as background_noise
from backgrounds import tdi
from lisaorbits import InterpolatedOrbits
from numpyro.diagnostics import summary as numpyro_summary
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, median_filter
from scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent / "wdm_psd"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tv_pspline_psd import (  # noqa: E402
    PSplineConfig,
    adaptive_frequency_bin_starts,
    collapse_wdm_frequency_projection,
    fit_log_pspline_surface,
    run_stationary_psd_mcmc,
    wdm_analysis_coefficients,
    wdm_frequency_projection_grid,
)
from tv_pspline_psd.lisa_aet import (  # noqa: E402
    diagonal_xyz_psd_to_aet,
    xyz_covariance_to_aet_diagonal,
    xyz_to_aet_series,
)

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
CARRIER_FREQUENCY_HZ = 281_600_000_000_000.0
DEFAULT_ARCHIVE = HERE / "combined_esa_xyz.h5"
DEFAULT_ORBITS = HERE / "noise2a" / "orbits.h5"
DEFAULT_RESULTS = HERE / "esa_m0_results"
CHI_SQUARE_ONE_MEDIAN = 0.4549364231195727
WDM_PROJECTION_CACHE_VERSION = 1

# The archive's "tdi/total", "truth/galactic_psd" and "truth/noise_psd"
# datasets are stored as (X2, Y2, Z2) on their leading axis. A/E/T are not
# stored -- they are the orthogonal rotation of X/Y/Z (see
# tv_pspline_psd.lisa_aet), applied to the time series. Analytic A/E/T
# instrumental references are obtained by rotating the full XYZ covariance;
# only the archived Galactic point surface uses the archive's explicit
# zero-XYZ-cross-spectrum generation contract.
XYZ_CHANNELS = ("X2", "Y2", "Z2")
AET_CHANNELS = ("A", "E", "T")
ALL_CHANNELS = XYZ_CHANNELS + AET_CHANNELS


def load_archive_channel_series(
    hdf: h5py.File,
    dataset: str,
    channel: str,
    n_samples: int,
) -> np.ndarray:
    """Read one XYZ channel or rotate the archived XYZ triplet to A/E/T."""
    if channel in XYZ_CHANNELS:
        return hdf[dataset][XYZ_CHANNELS.index(channel), :n_samples]
    xyz = hdf[dataset][:, :n_samples]
    return xyz_to_aet_series(xyz)[AET_CHANNELS.index(channel)]


def wdm_valid_length(n_requested: int, nt: int) -> int:
    """Largest valid WDM length no larger than ``n_requested``."""
    nf = n_requested // nt
    nf -= nf % 2
    if nt % 2 or nf < 2:
        raise ValueError("WDM requires even nt and even N / nt >= 2")
    return nt * nf


def robust_training_psd_scale(
    coefficients: np.ndarray,
    retained: np.ndarray,
    to_psd: float,
) -> float:
    """Return a truth-free numerical PSD scale from retained training cells.

    For one real Gaussian WDM coefficient, ``w**2 / S`` follows ``chi2_1``.
    Dividing the retained median power by the ``chi2_1`` median therefore gives
    a robust order-of-magnitude PSD scale. This value is only a numerical
    reparameterisation; excluded, validation, test, and gap cells cannot enter.
    """
    values = np.asarray(coefficients, dtype=float)
    mask = np.asarray(retained, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("retained mask must match coefficients")
    if not np.isfinite(to_psd) or to_psd <= 0.0:
        raise ValueError("to_psd must be finite and positive")
    selected = values[mask]
    finite = np.isfinite(selected)
    if not np.any(finite):
        raise ValueError("retained training coefficients contain no finite values")
    power_psd = selected[finite] ** 2 * to_psd
    positive = power_psd[power_psd > 0.0]
    if positive.size == 0:
        raise ValueError("retained training coefficients have no positive power")
    scale = float(np.median(positive) / CHI_SQUARE_ONE_MEDIAN)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("could not construct a positive training-data scale")
    return scale


def load_esa_orbits(path: Path) -> InterpolatedOrbits:
    """Load the same interpolated ESA orbit model used for the archive."""
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


def _channel_diagonal(covariance: np.ndarray, channel: str) -> np.ndarray:
    """Extract one channel's real auto-PSD from a ``(freq, time, 3, 3)`` XYZ
    covariance. XYZ channels read the matrix diagonal directly (frozen X2
    path: ``channel_index=0`` is byte-for-byte the prior hardcoded ``[...,0,0]``
    index); AET channels rotate the full XYZ covariance before taking the
    diagonal, preserving the analytic cross spectra.
    """
    if channel in XYZ_CHANNELS:
        index = XYZ_CHANNELS.index(channel)
        return np.asarray(covariance[..., index, index].real)
    index = AET_CHANNELS.index(channel)
    return xyz_covariance_to_aet_diagonal(covariance)[..., index]


def _analytic_noise_components_for_frequencies(
    channel: str,
    orbits: InterpolatedOrbits,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one bounded frequency block with a preloaded orbit model."""
    oms = background_noise.AnalyticOMSNoiseModel(
        frequency_hz,
        time_tcb,
        orbits,
        tdi_tf_func=tdi.compute_tdi_tf,
        gen="2.0",
        oms_isi_carrier_asds=7.9e-12,
        fs=0.5,
        duration=SECONDS_PER_YEAR,
    )
    test_mass = background_noise.AnalyticTMNoiseModel(
        frequency_hz,
        time_tcb,
        orbits,
        tdi_tf_func=tdi.compute_tdi_tf_tm,
        gen="2.0",
        tm_isi_carrier_asds=2.4e-15,
        fs=0.5,
        duration=SECONDS_PER_YEAR,
    )
    oms_psd = (
        _channel_diagonal(oms.compute_covariances(0.0), channel).T
        * CARRIER_FREQUENCY_HZ**2
    )
    tm_psd = (
        _channel_diagonal(test_mass.compute_covariances(0.0), channel).T
        * CARRIER_FREQUENCY_HZ**2
    )
    return oms_psd, tm_psd


def analytic_channel_noise_components_psd(
    channel: str,
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    frequency_chunk: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct analytic ESA-orbit OMS and TM PSDs for one channel.

    The returned array has shape ``(time, frequency)`` and is in the same
    fractional-frequency-squared per Hz units as the archive truth.  Frequency
    chunking avoids materialising the full XYZ covariance tensor at once.
    """
    if channel not in ALL_CHANNELS:
        raise ValueError(f"channel must be one of {ALL_CHANNELS}, got {channel!r}")
    time_tcb = np.asarray(time_tcb, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    orbits = load_esa_orbits(orbit_path)
    oms_result = np.empty((time_tcb.size, frequency_hz.size), dtype=float)
    tm_result = np.empty((time_tcb.size, frequency_hz.size), dtype=float)
    for start in range(0, frequency_hz.size, frequency_chunk):
        stop = min(start + frequency_chunk, frequency_hz.size)
        oms_block, tm_block = _analytic_noise_components_for_frequencies(
            channel,
            orbits,
            time_tcb,
            frequency_hz[start:stop],
        )
        oms_result[:, start:stop] = oms_block
        tm_result[:, start:stop] = tm_block
    return oms_result, tm_result


def projected_analytic_channel_noise_components_psd(
    channel: str,
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    delta_f_hz: float,
    *,
    projection_nodes: int = 16,
    frequency_chunk: int = 384,
    spectral_tilt: float = 0.0,
    pivot_hz: float = 1.0e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project analytic OMS and TM spectra onto interior WDM cells.

    The response is evaluated inside the exact compact support of each WDM
    frequency atom and collapsed with the squared Meyer-window weights. Work
    is chunked by the total number of quadrature frequencies, bounding the
    temporary covariance tensors on year-long grids.
    """
    if channel not in ALL_CHANNELS:
        raise ValueError(f"channel must be one of {ALL_CHANNELS}, got {channel!r}")
    if frequency_chunk < projection_nodes:
        raise ValueError("frequency_chunk must be at least projection_nodes")
    if pivot_hz <= 0.0:
        raise ValueError("pivot_hz must be positive")
    time_tcb = np.asarray(time_tcb, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    projection_grid, weights = wdm_frequency_projection_grid(
        frequency_hz,
        delta_f_hz,
        n_nodes=projection_nodes,
    )
    orbits = load_esa_orbits(orbit_path)
    oms_result = np.empty((time_tcb.size, frequency_hz.size), dtype=float)
    tm_result = np.empty_like(oms_result)
    centers_per_chunk = max(1, int(frequency_chunk) // int(projection_nodes))
    for start in range(0, frequency_hz.size, centers_per_chunk):
        stop = min(start + centers_per_chunk, frequency_hz.size)
        sample_frequency = projection_grid[start:stop].reshape(-1)
        oms_sample, tm_sample = _analytic_noise_components_for_frequencies(
            channel,
            orbits,
            time_tcb,
            sample_frequency,
        )
        shape = (time_tcb.size, stop - start, projection_nodes)
        if spectral_tilt != 0.0:
            multiplier = (sample_frequency / pivot_hz) ** spectral_tilt
            oms_sample *= multiplier[None, :]
            tm_sample *= multiplier[None, :]
        oms_result[:, start:stop] = collapse_wdm_frequency_projection(
            oms_sample.reshape(shape), weights
        )
        tm_result[:, start:stop] = collapse_wdm_frequency_projection(
            tm_sample.reshape(shape), weights
        )
    return oms_result, tm_result


def analytic_channel_noise_psd(
    channel: str,
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    frequency_chunk: int = 96,
) -> np.ndarray:
    """Direct analytic ESA-orbit OMS+TM PSD for one channel."""
    oms, test_mass = analytic_channel_noise_components_psd(
        channel,
        orbit_path,
        time_tcb,
        frequency_hz,
        frequency_chunk=frequency_chunk,
    )
    return oms + test_mass


def interpolate_positive_surface(
    source: np.ndarray,
    source_time: np.ndarray,
    source_frequency: np.ndarray,
    target_time: np.ndarray,
    target_frequency: np.ndarray,
) -> np.ndarray:
    """Log-interpolate one positive surface in time and log frequency."""
    source = np.asarray(source, dtype=float)
    positive = source[source > 0]
    if positive.size == 0:
        raise ValueError("source surface has no positive entries")
    floor = float(np.min(positive)) * 1.0e-6
    interpolator = RegularGridInterpolator(
        (np.asarray(source_time), np.log(np.asarray(source_frequency))),
        np.log(np.maximum(source, floor)),
        bounds_error=False,
        fill_value=None,
    )
    tm, fm = np.meshgrid(target_time, np.log(target_frequency), indexing="ij")
    points = np.column_stack((tm.ravel(), fm.ravel()))
    return np.exp(interpolator(points).reshape(tm.shape))


def projected_interpolated_positive_surface(
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
    """Log-interpolate a model inside each WDM atom and project its power.

    Set ``zero_outside_frequency`` when the source-generation contract is zero
    beyond its tabulated band. This is used for the Galactic component, which
    was generated only over 1e-4--1e-1 Hz; extrapolating its final positive
    grid value into quadrature nodes beyond 0.1 Hz would add absent power.
    """
    if frequency_chunk < projection_nodes:
        raise ValueError("frequency_chunk must be at least projection_nodes")
    projection_grid, weights = wdm_frequency_projection_grid(
        target_frequency,
        delta_f_hz,
        n_nodes=projection_nodes,
    )
    output = np.empty((len(target_time), len(target_frequency)), dtype=float)
    centers_per_chunk = max(1, int(frequency_chunk) // int(projection_nodes))
    for start in range(0, len(target_frequency), centers_per_chunk):
        stop = min(start + centers_per_chunk, len(target_frequency))
        sampled_frequency = projection_grid[start:stop].reshape(-1)
        if zero_outside_frequency:
            sampled = np.zeros((len(target_time), sampled_frequency.size), dtype=float)
            inside = (
                (sampled_frequency >= float(np.min(source_frequency)))
                & (sampled_frequency <= float(np.max(source_frequency)))
            )
            if np.any(inside):
                sampled[:, inside] = interpolate_positive_surface(
                    source,
                    source_time,
                    source_frequency,
                    target_time,
                    sampled_frequency[inside],
                )
        else:
            sampled = interpolate_positive_surface(
                source,
                source_time,
                source_frequency,
                target_time,
                sampled_frequency,
            )
        output[:, start:stop] = collapse_wdm_frequency_projection(
            sampled.reshape(len(target_time), stop - start, projection_nodes),
            weights,
        )
    return output


def response_null_mask(
    noise_psd: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    high_frequency_hz: float = 0.02,
    ratio_threshold: float = 0.35,
    continuum_width: int = 81,
    dilation_bins: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return response-retained cells and the local response ratio.

    A running median in log PSD estimates the smooth continuum separately at
    each time.  Deep high-frequency depressions, dilated along frequency to
    cover their shoulders, are the excluded null corridors.
    """
    noise_psd = np.asarray(noise_psd, dtype=float)
    width = max(3, int(continuum_width) | 1)
    log_noise = np.log(np.maximum(noise_psd, np.finfo(float).tiny))
    log_continuum = median_filter(log_noise, size=(1, width), mode="nearest")
    ratio = np.exp(log_noise - log_continuum)
    core = (frequency_hz[None, :] >= high_frequency_hz) & (ratio < ratio_threshold)
    if dilation_bins:
        structure = np.ones((1, 2 * int(dilation_bins) + 1), dtype=bool)
        excluded = binary_dilation(core, structure=structure)
    else:
        excluded = core
    return ~excluded, ratio, np.exp(log_continuum)


def lisa_like_gaps(t_obs_s: float, seed: int) -> list[tuple[float, float]]:
    """Scheduled 3.5 h/14 d gaps plus weekly-rate unscheduled outages."""
    gaps = [
        (float(t0), float(min(t0 + 3.5 * 3600.0, t_obs_s)))
        for t0 in np.arange(14 * SECONDS_PER_DAY, t_obs_s, 14 * SECONDS_PER_DAY)
    ]
    rng = np.random.default_rng(seed)
    n_unscheduled = int(rng.poisson(t_obs_s / (7 * SECONDS_PER_DAY)))
    starts = rng.uniform(0.0, t_obs_s, n_unscheduled)
    durations = np.exp(
        rng.uniform(np.log(0.5 * 3600.0), np.log(24.0 * 3600.0), n_unscheduled)
    )
    gaps.extend(
        (float(start), float(min(start + duration, t_obs_s)))
        for start, duration in zip(starts, durations, strict=True)
    )
    return sorted(gaps)


def gate_gaps(
    data: np.ndarray,
    dt: float,
    gaps: list[tuple[float, float]],
    *,
    taper_s: float = 3600.0,
) -> np.ndarray:
    """Zero gaps with one-hour cosine tapers, without a full time array."""
    output = np.asarray(data, dtype=float).copy()
    window = np.ones(output.size, dtype=float)
    for start_s, stop_s in gaps:
        start = max(0, int(np.floor(start_s / dt)))
        stop = min(output.size, int(np.ceil(stop_s / dt)) + 1)
        window[start:stop] = 0.0
        n_taper = max(1, int(np.ceil(taper_s / dt)))
        left = max(0, start - n_taper)
        if start > left:
            u = np.arange(start - left, dtype=float) / max(start - left, 1)
            window[left:start] = np.minimum(window[left:start], 0.5 - 0.5 * np.cos(np.pi * u))
        right = min(output.size, stop + n_taper)
        if right > stop:
            u = np.arange(1, right - stop + 1, dtype=float) / max(right - stop, 1)
            window[stop:right] = np.minimum(window[stop:right], 0.5 - 0.5 * np.cos(np.pi * u))
    return output * window


def good_time_bins(
    time_grid: np.ndarray,
    t_obs_s: float,
    gaps: list[tuple[float, float]],
    nt: int,
    *,
    taper_s: float = 3600.0,
    buffer_pixels: float = 1.0,
) -> np.ndarray:
    """WDM rows unaffected by a tapered gap plus one WDM-pixel buffer."""
    centers = np.asarray(time_grid) * t_obs_s
    pixel = buffer_pixels * t_obs_s / nt
    keep = np.ones(centers.size, dtype=bool)
    for start, stop in gaps:
        keep &= (centers < start - taper_s - pixel) | (centers > stop + taper_s + pixel)
    return keep


def block_holdout(n_time: int, block: int = 4, every: int = 5) -> np.ndarray:
    """Hold out every fifth four-row block for time-predictive scoring."""
    return (np.arange(n_time) // block) % every == every - 1


def analysis_row_split(
    n_time: int,
    *,
    block: int = 4,
    cycle: int = 7,
    validation_fold: int = 5,
    test_fold: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic training/validation/test blocks for prospective scoring.

    The final test fold is excluded from every fit and pilot.  The validation
    fold is available for future tuning but is also excluded from the locked
    production fit, preventing the earlier single holdout cohort from serving
    simultaneously as model-selection and final-test data.
    """
    if cycle < 3 or not (0 <= validation_fold < cycle) or not (0 <= test_fold < cycle):
        raise ValueError("validation/test folds must be distinct members of cycle >= 3")
    if validation_fold == test_fold:
        raise ValueError("validation and test folds must differ")
    fold = (np.arange(n_time) // block) % cycle
    validation = fold == validation_fold
    test = fold == test_fold
    training = ~(validation | test)
    return training, validation, test


def analysis_masks(
    training_rows: np.ndarray,
    response_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distinct inference and response-notched evaluation masks.

    Every frequency cell in a retained training row contributes to inference,
    including cells near response nulls. ``response_keep`` is used only to
    define cohorts where relative/log accuracy and continuum whitening are
    well-conditioned; it must not alter the likelihood population or adaptive
    bin pilot.
    """
    rows = np.asarray(training_rows, dtype=bool)
    evaluation = np.asarray(response_keep, dtype=bool)
    if rows.ndim != 1 or evaluation.ndim != 2 or evaluation.shape[0] != rows.size:
        raise ValueError(
            "training_rows must match the time dimension of response_keep"
        )
    inference = np.broadcast_to(rows[:, None], evaluation.shape).copy()
    return inference, evaluation.copy()


def training_data_pilot_log_psd(
    coefficients: np.ndarray,
    retained_training_mask: np.ndarray,
    *,
    n_time_profiles: int = 32,
    frequency_width: int = 31,
) -> np.ndarray:
    """Build a truth-free smooth pilot from retained training coefficients.

    Training rows are reduced to robust time-block medians before frequency
    smoothing. Missing response cells are interpolated only for pilot
    construction; they remain absent from the likelihood. The pilot selects
    likelihood bins but is never used as an inferential observation. Returning
    a small ``(profile, frequency)`` array also avoids a full-grid median-filter
    temporary on year-long data.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    retained = np.asarray(retained_training_mask, dtype=bool)
    if coefficients.shape != retained.shape or coefficients.ndim != 2:
        raise ValueError("coefficients and retained_training_mask must share a 2-D shape")
    positive = coefficients[retained] ** 2
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("pilot mask retains no positive coefficient power")
    floor = float(np.min(positive)) * 1.0e-6
    raw = np.log(np.maximum(coefficients**2, floor))
    n_profiles = max(1, min(int(n_time_profiles), coefficients.shape[0]))
    edges = np.linspace(0, coefficients.shape[0], n_profiles + 1, dtype=int)
    filled = np.full((n_profiles, coefficients.shape[1]), np.nan)
    index_frequency = np.arange(coefficients.shape[1], dtype=float)
    for profile, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        block = np.where(retained[start:stop], raw[start:stop], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row = np.nanmedian(block, axis=0)
        valid = np.isfinite(row)
        if np.any(valid):
            filled[profile] = np.interp(index_frequency, index_frequency[valid], row[valid])
    valid_profiles = np.flatnonzero(np.isfinite(filled[:, 0]))
    if valid_profiles.size == 0:
        raise ValueError("pilot has no retained training profiles")
    for profile in range(n_profiles):
        if not np.isfinite(filled[profile, 0]):
            nearest = valid_profiles[np.argmin(np.abs(valid_profiles - profile))]
            filled[profile] = filled[nearest]
    return median_filter(
        filled,
        size=(1, max(1, int(frequency_width)) | 1),
        mode="nearest",
    )


def partition_starts(n_time: int, block: int, *states: np.ndarray) -> np.ndarray:
    """Block starts augmented at every supplied state transition."""
    starts = set(range(0, n_time, block))
    for state in states:
        state = np.asarray(state, dtype=bool)
        starts.update((np.flatnonzero(state[1:] != state[:-1]) + 1).tolist())
    return np.asarray(sorted(starts), dtype=int)


def hybrid_frequency_knots(
    frequency_hz: np.ndarray,
    n_knots: int,
    *,
    break_hz: float = 0.01,
    low_fraction: float = 0.5,
) -> np.ndarray:
    """Interior knots log-spaced below 0.01 Hz and linear above."""
    f0, f1 = map(float, (frequency_hz[0], frequency_hz[-1]))
    n_low = max(1, min(n_knots - 1, int(round(n_knots * low_fraction))))
    n_high = n_knots - n_low
    low = np.geomspace(f0, break_hz, n_low + 2)[1:-1]
    high = np.linspace(break_hz, f1, n_high + 2)[1:-1]
    knots = np.unique(np.concatenate((low, high)))
    if knots.size != n_knots:
        raise RuntimeError("hybrid knot construction produced the wrong count")
    return knots


def sampler_diagnostics(result: dict[str, Any], max_tree_depth: int) -> dict[str, float]:
    """Reviewer-facing convergence gates across all sampled sites."""
    mcmc = result["mcmc"]
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = numpyro_summary(grouped, group_by_chain=True)
    rhats: list[np.ndarray] = []
    esses: list[np.ndarray] = []
    for site in diag.values():
        rhats.append(np.asarray(site["r_hat"], dtype=float).ravel())
        esses.append(np.asarray(site["n_eff"], dtype=float).ravel())
    extra = mcmc.get_extra_fields(group_by_chain=True)
    steps = np.asarray(extra["num_steps"], dtype=float)
    energy = np.asarray(extra["potential_energy"], dtype=float)
    variances = np.var(energy, axis=1, ddof=1)
    ebfmi = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / variances
    return {
        "divergences": int(np.asarray(extra["diverging"]).sum()),
        "max_rhat": float(np.nanmax(np.concatenate(rhats))),
        "min_ess": float(np.nanmin(np.concatenate(esses))),
        "tree_depth_saturation": float(np.mean(steps >= (2**max_tree_depth - 1))),
        "min_ebfmi": float(np.nanmin(ebfmi)),
        "mean_accept_prob": float(np.mean(np.asarray(extra["accept_prob"]))),
    }


def convergence_gate_status(diagnostics: dict[str, float]) -> dict[str, Any]:
    """Apply the predeclared publication convergence thresholds."""
    checks = {
        "zero_divergences": diagnostics["divergences"] == 0,
        "max_rhat_le_1.05": diagnostics["max_rhat"] <= 1.05,
        "min_ess_ge_50": diagnostics["min_ess"] >= 50.0,
        "tree_depth_saturation_le_0.05": diagnostics["tree_depth_saturation"] <= 0.05,
        "min_ebfmi_ge_0.3": diagnostics["min_ebfmi"] >= 0.3,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def masked_mean(values: np.ndarray, mask: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Mean with a boolean cell mask and NaN for empty reductions."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    numerator = np.sum(np.where(mask, values, 0.0), axis=axis)
    denominator = np.sum(mask, axis=axis)
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=float),
        where=denominator > 0,
    )


def score_surface(
    estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Truth-recovery scores on an explicitly supplied cell cohort."""
    selected = np.asarray(mask, dtype=bool)
    delta = np.log(np.maximum(estimate[selected], np.finfo(float).tiny)) - np.log(truth[selected])
    return {
        "n_cells": int(selected.sum()),
        "log_rmse": float(np.sqrt(np.mean(delta**2))),
        "log_bias": float(np.mean(delta)),
        "coverage_90": float(np.mean((truth[selected] >= lower[selected]) & (truth[selected] <= upper[selected]))),
    }


def score_stationary(
    estimate_frequency: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Log-error scores for a time-invariant per-frequency comparator."""
    estimate = np.broadcast_to(np.asarray(estimate_frequency)[None, :], truth.shape)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(estimate)
    delta = np.log(estimate[selected]) - np.log(truth[selected])
    return {
        "n_cells": int(selected.sum()),
        "log_rmse": float(np.sqrt(np.mean(delta**2))),
        "log_bias": float(np.mean(delta)),
    }


def score_point_surface(
    estimate: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Truth-based point-estimate score for a complete PSD surface."""
    selected = np.asarray(mask, dtype=bool)
    delta = np.log(np.asarray(estimate)[selected]) - np.log(truth[selected])
    return {
        "n_cells": int(selected.sum()),
        "log_rmse": float(np.sqrt(np.mean(delta**2))),
        "log_bias": float(np.mean(delta)),
    }


def blind_whitening_diagnostics(
    coefficients: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
    to_psd: float,
) -> dict[str, float]:
    """Held-out coefficient checks that require no injected PSD truth."""
    values = np.asarray(coefficients, dtype=float)
    surface = np.asarray(estimate, dtype=float)
    retained = np.asarray(mask, dtype=bool)
    if values.shape != surface.shape or values.shape != retained.shape:
        raise ValueError("coefficients, estimate, and mask must share shape")
    selected = retained & np.isfinite(surface) & (surface > 0.0)
    if not np.any(selected):
        raise ValueError("blind diagnostics retain no finite positive cells")
    whitened = values * np.sqrt(to_psd / surface)
    z = whitened[selected]
    z2 = z**2

    time_pairs = selected[1:] & selected[:-1]
    freq_pairs = selected[:, 1:] & selected[:, :-1]
    time_product = whitened[1:][time_pairs] * whitened[:-1][time_pairs]
    freq_product = whitened[:, 1:][freq_pairs] * whitened[:, :-1][freq_pairs]
    whittle = -0.5 * (np.log(surface[selected]) + z2)
    return {
        "n_cells": int(selected.sum()),
        "mean_z": float(np.mean(z)),
        "mean_z2": float(np.mean(z2)),
        "median_z2_over_chi2_1_median": float(
            np.median(z2) / CHI_SQUARE_ONE_MEDIAN
        ),
        "central_90_fraction": float(np.mean(np.abs(z) <= 1.6448536269514722)),
        "lag1_time_product": (
            float(np.mean(time_product)) if time_product.size else float("nan")
        ),
        "lag1_frequency_product": (
            float(np.mean(freq_product)) if freq_product.size else float("nan")
        ),
        "mean_whittle_log_score": float(np.mean(whittle)),
    }


def validate_archived_components_in_wdm(
    archive_path: Path,
    channel: str,
    n_total: int,
    dt: float,
    nt: int,
    config: PSplineConfig,
    time_grid: np.ndarray,
    frequency_hz: np.ndarray,
    clean_total_coefficients: np.ndarray,
    noise_truth: np.ndarray,
    galactic_truth: np.ndarray,
    point_noise_reference: np.ndarray,
    response_keep: np.ndarray,
    to_psd: float,
) -> dict[str, Any]:
    """Validate the generation components against their WDM-projected truth.

    This uses the archived noise and Galactic time series, not the total data
    used for inference. It is a pre-inference plumbing check and does not tune
    the fit. The total/noise/Galactic WDM linearity residual catches channel
    rotation, trimming, or archive-selection mistakes.
    """
    component_coefficients: dict[str, np.ndarray] = {}
    with h5py.File(archive_path, "r") as hdf:
        for name, dataset in (("noise", "tdi/noise"), ("galactic", "tdi/galactic")):
            series = load_archive_channel_series(hdf, dataset, channel, n_total)
            coefficients, component_time, component_frequency = wdm_analysis_coefficients(
                series, dt, nt, config
            )
            if not (
                np.array_equal(component_time, time_grid)
                and np.array_equal(component_frequency, frequency_hz)
            ):
                raise RuntimeError(f"{name} component WDM grid differs from total grid")
            component_coefficients[name] = coefficients

    cohorts = {
        "low": response_keep & (frequency_hz[None, :] <= 0.003),
        "full_response_notched": response_keep,
        "high_response_notched": response_keep
        & (frequency_hz[None, :] >= 0.02),
        "response_null_cells": ~response_keep,
        "full_all_cells": np.ones_like(response_keep, dtype=bool),
    }
    truth_by_component = {"noise": noise_truth, "galactic": galactic_truth}
    result: dict[str, Any] = {
        "contract": (
            "archived component time series transformed independently; compared "
            "to WDM-frequency-kernel-projected component expectations"
        ),
        "uses_total_inference_data_for_tuning": False,
        "components": {},
    }
    for name, coefficients in component_coefficients.items():
        truth = np.asarray(truth_by_component[name], dtype=float)
        power = coefficients**2 * to_psd
        summaries: dict[str, Any] = {}
        for cohort_name, cohort in cohorts.items():
            selected = cohort & np.isfinite(truth) & (truth > 0.0)
            # The Galactic component is generated only on its source band and
            # can be vanishingly small at the upper boundary. Do not present a
            # null-core Galactic ratio where numerical roundoff dominates.
            if name == "galactic" and cohort_name == "response_null_cells":
                continue
            if not np.any(selected):
                summaries[cohort_name] = {
                    "n_cells": 0,
                    "skipped": "cohort contains no finite positive truth cells",
                }
                continue
            checks = blind_whitening_diagnostics(
                coefficients,
                truth,
                selected,
                to_psd,
            )
            checks["sum_power_over_sum_expectation"] = float(
                np.sum(power[selected]) / np.sum(truth[selected])
            )
            summaries[cohort_name] = checks
        result["components"][name] = summaries

    null_cells = (~response_keep) & (point_noise_reference > 0.0)
    noise_power = component_coefficients["noise"] ** 2 * to_psd
    if np.any(null_cells):
        result["point_vs_projected_null_check"] = {
            "n_cells": int(null_cells.sum()),
            "mean_z2_projected_reference": float(
                np.mean(noise_power[null_cells] / noise_truth[null_cells])
            ),
            "mean_z2_point_reference": float(
                np.mean(noise_power[null_cells] / point_noise_reference[null_cells])
            ),
            "interpretation": (
                "projected should be near one; point evaluation is expected to "
                "over-whiten at deep transfer-function zeros"
            ),
        }
    reconstructed = component_coefficients["noise"] + component_coefficients["galactic"]
    residual = clean_total_coefficients - reconstructed
    scale = max(float(np.max(np.abs(clean_total_coefficients))), np.finfo(float).tiny)
    result["wdm_linearity"] = {
        "max_absolute_residual": float(np.max(np.abs(residual))),
        "max_residual_over_max_total": float(np.max(np.abs(residual)) / scale),
    }
    return result


def _residual_log_psd_draws(
    fit: dict[str, Any],
    time_indices: np.ndarray,
    frequency_indices: np.ndarray,
) -> np.ndarray:
    """Reconstruct selected residual log-PSD draws without a full draw cube."""
    rows = np.asarray(time_indices, dtype=int)
    columns = np.asarray(frequency_indices, dtype=int)
    samples = fit["samples"]
    if fit["residual_structure"] == "stationary_plus_interaction":
        basis_time = np.asarray(fit["basis_interaction_time"])[rows]
        basis_frequency = np.asarray(fit["basis_nested_freq"])[columns]
        n_time_eig = basis_time.shape[1]
        n_frequency_eig = basis_frequency.shape[1]
        g_samples = np.asarray(samples["g"]).reshape(-1, n_frequency_eig)
        h_samples = np.asarray(samples["h"]).reshape(
            -1, n_time_eig, n_frequency_eig
        )
        stationary = g_samples @ basis_frequency.T
        interaction = np.einsum(
            "ta,nab,fb->ntf",
            basis_time,
            h_samples,
            basis_frequency,
            optimize=True,
        )
        return stationary[:, None, :] + interaction

    if not bool(fit["config"].centered):
        raise ValueError("selected-draw reconstruction currently requires centered=True")
    basis_time = (
        np.asarray(fit["B_time"])[rows]
        @ np.asarray(fit["whitened"]["U_time"])
    )
    basis_frequency = (
        np.asarray(fit["B_freq"])[columns]
        @ np.asarray(fit["whitened"]["U_freq"])
    )
    eig_samples = np.asarray(samples["s"]).reshape(
        -1, basis_time.shape[1], basis_frequency.shape[1]
    )
    return np.einsum(
        "ta,nab,fb->ntf",
        basis_time,
        eig_samples,
        basis_frequency,
        optimize=True,
    )


def low_band_modulation_posterior(
    fit: dict[str, Any],
    log_psd_offset: np.ndarray,
    frequency_select: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior interval for the geometric mean PSD in one frequency band."""
    selected = np.flatnonzero(np.asarray(frequency_select, dtype=bool))
    if selected.size == 0:
        raise ValueError("frequency_select retains no channels")
    offset = np.asarray(log_psd_offset, dtype=float)[:, selected].mean(axis=1)
    if fit["residual_structure"] == "stationary_plus_interaction":
        basis_time = np.asarray(fit["basis_interaction_time"])
        basis_frequency_mean = np.asarray(fit["basis_nested_freq"])[selected].mean(axis=0)
        n_time_eig = basis_time.shape[1]
        n_frequency_eig = basis_frequency_mean.size
        samples = fit["samples"]
        g_samples = np.asarray(samples["g"]).reshape(-1, n_frequency_eig)
        h_samples = np.asarray(samples["h"]).reshape(
            -1, n_time_eig, n_frequency_eig
        )
        log_draws = (
            offset[None, :]
            + (g_samples @ basis_frequency_mean)[:, None]
            + np.einsum(
                "ta,nab,b->nt",
                basis_time,
                h_samples,
                basis_frequency_mean,
                optimize=True,
            )
        )
    else:
        if not bool(fit["config"].centered):
            raise ValueError("modulation reconstruction currently requires centered=True")
        basis_time = np.asarray(fit["B_time"]) @ np.asarray(
            fit["whitened"]["U_time"]
        )
        basis_frequency_mean = (
            np.asarray(fit["B_freq"])[selected]
            @ np.asarray(fit["whitened"]["U_freq"])
        ).mean(axis=0)
        eig_samples = np.asarray(fit["samples"]["s"]).reshape(
            -1, basis_time.shape[1], basis_frequency_mean.size
        )
        log_draws = offset[None, :] + np.einsum(
            "ta,nab,b->nt",
            basis_time,
            eig_samples,
            basis_frequency_mean,
            optimize=True,
        )
    lower, median, upper = np.percentile(np.exp(log_draws), [5.0, 50.0, 95.0], axis=0)
    return lower, median, upper


def posterior_predictive_score_comparison(
    normalized_coefficients: np.ndarray,
    tv_fit: dict[str, Any],
    stationary_fit: dict[str, Any],
    log_psd_offset: np.ndarray,
    cohort_rows: np.ndarray,
    cohorts: dict[str, np.ndarray],
    *,
    row_block: int = 4,
    frequency_chunk: int = 256,
    bootstrap_replicates: int = 4000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Compare posterior predictive Whittle scores on held-out row blocks.

    Each model's pointwise score integrates over its training-conditioned
    posterior draws with log-mean-exp. Uncertainty resamples complete held-out
    time blocks, preserving the cohort's intended clustering unit.
    """
    coefficients = np.asarray(normalized_coefficients, dtype=float)
    rows = np.asarray(cohort_rows, dtype=bool)
    offset = np.asarray(log_psd_offset, dtype=float)
    if coefficients.shape != offset.shape or rows.shape != (coefficients.shape[0],):
        raise ValueError("cohort rows and log offset must match coefficients")
    if row_block < 1 or frequency_chunk < 1 or bootstrap_replicates < 1:
        raise ValueError("block, chunk, and bootstrap counts must be positive")
    cohort_masks = {name: np.asarray(mask, dtype=bool) for name, mask in cohorts.items()}
    if any(mask.shape != coefficients.shape for mask in cohort_masks.values()):
        raise ValueError("every cohort mask must match coefficients")

    selected_rows = np.flatnonzero(rows)
    if selected_rows.size == 0:
        raise ValueError("cohort_rows retains no rows")
    run_boundaries = np.flatnonzero(np.diff(selected_rows) > 1) + 1
    runs = np.split(selected_rows, run_boundaries)
    blocks = [
        run[start : start + row_block]
        for run in runs
        for start in range(0, run.size, row_block)
        if run[start : start + row_block].size
    ]
    block_values = {
        name: {"tv": [], "stationary": [], "count": []}
        for name in cohort_masks
    }
    stationary_residual = np.asarray(stationary_fit["residual_log_psd_samples"])
    n_tv_draws = int(np.asarray(next(iter(tv_fit["samples"].values()))).shape[0])
    n_stationary_draws = int(stationary_residual.shape[0])

    for block_rows in blocks:
        totals = {
            name: {"tv": 0.0, "stationary": 0.0, "count": 0}
            for name in cohort_masks
        }
        for start in range(0, coefficients.shape[1], frequency_chunk):
            stop = min(start + frequency_chunk, coefficients.shape[1])
            columns = np.arange(start, stop)
            tv_log_psd = (
                offset[np.ix_(block_rows, columns)][None, :, :]
                + _residual_log_psd_draws(tv_fit, block_rows, columns)
            )
            stationary_log_psd = (
                offset[np.ix_(block_rows, columns)][None, :, :]
                + stationary_residual[:, None, columns]
            )
            power = coefficients[np.ix_(block_rows, columns)] ** 2
            tv_log_likelihood = -0.5 * (
                tv_log_psd + power[None, :, :] * np.exp(np.clip(-tv_log_psd, -745.0, 700.0))
            )
            stationary_log_likelihood = -0.5 * (
                stationary_log_psd
                + power[None, :, :] * np.exp(np.clip(-stationary_log_psd, -745.0, 700.0))
            )
            tv_lpd = logsumexp(tv_log_likelihood, axis=0) - np.log(n_tv_draws)
            stationary_lpd = (
                logsumexp(stationary_log_likelihood, axis=0)
                - np.log(n_stationary_draws)
            )
            for name, mask in cohort_masks.items():
                selected = mask[np.ix_(block_rows, columns)]
                totals[name]["tv"] += float(np.sum(tv_lpd[selected]))
                totals[name]["stationary"] += float(np.sum(stationary_lpd[selected]))
                totals[name]["count"] += int(selected.sum())
        for name in cohort_masks:
            if totals[name]["count"]:
                for key in ("tv", "stationary", "count"):
                    block_values[name][key].append(totals[name][key])

    rng = np.random.default_rng(bootstrap_seed)
    result: dict[str, Any] = {}
    for name, values in block_values.items():
        tv = np.asarray(values["tv"], dtype=float)
        stationary = np.asarray(values["stationary"], dtype=float)
        count = np.asarray(values["count"], dtype=float)
        if count.size == 0:
            raise ValueError(f"cohort {name!r} retains no held-out cells")
        difference = tv - stationary
        draws = np.empty(bootstrap_replicates, dtype=float)
        for index in range(bootstrap_replicates):
            sampled = rng.integers(0, count.size, size=count.size)
            draws[index] = difference[sampled].sum() / count[sampled].sum()
        result[name] = {
            "n_blocks": int(count.size),
            "n_cells": int(count.sum()),
            "tv_mean_log_predictive_density": float(tv.sum() / count.sum()),
            "stationary_mean_log_predictive_density": float(
                stationary.sum() / count.sum()
            ),
            "tv_minus_stationary_mean": float(difference.sum() / count.sum()),
            "tv_minus_stationary_block_bootstrap_p05": float(np.percentile(draws, 5.0)),
            "tv_minus_stationary_block_bootstrap_median": float(np.percentile(draws, 50.0)),
            "tv_minus_stationary_block_bootstrap_p95": float(np.percentile(draws, 95.0)),
        }
    return {
        "estimator": "held-out posterior predictive Whittle log density",
        "posterior_integration": "pointwise log-mean-exp over posterior draws",
        "uncertainty": "paired nonparametric bootstrap of complete held-out time blocks",
        "row_block": int(row_block),
        "bootstrap_replicates": int(bootstrap_replicates),
        "cohorts": result,
    }


def _truth_cache_matches(
    cache: dict[str, np.ndarray],
    time_tcb: np.ndarray,
    frequency: np.ndarray,
    delta_f_hz: float,
    projection_nodes: int,
) -> bool:
    return (
        "projection_cache_version" in cache
        and np.array_equal(cache["time_tcb"], time_tcb)
        and np.array_equal(cache["frequency_hz"], frequency)
        and float(cache["delta_f_hz"]) == float(delta_f_hz)
        and int(cache["projection_nodes"]) == int(projection_nodes)
        and int(cache["projection_cache_version"])
        == WDM_PROJECTION_CACHE_VERSION
    )


def _channel_truth(dataset: np.ndarray, channel: str) -> np.ndarray:
    """Select one channel's PSD from an archived ``(X2, Y2, Z2)`` truth array.

    AET channels rotate under the same zero-XYZ-cross-spectrum contract as
    ``_channel_diagonal``: appropriate for this controlled archive, not a
    physical multichannel response (see ``tv_pspline_psd.lisa_aet.diagonal_xyz_psd_to_aet``).
    """
    if channel in XYZ_CHANNELS:
        return dataset[XYZ_CHANNELS.index(channel)]
    return diagonal_xyz_psd_to_aet(dataset)[AET_CHANNELS.index(channel)]


def load_or_build_truth(
    channel: str,
    cache_path: Path,
    archive_path: Path,
    orbit_path: Path,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    delta_f_hz: float,
    *,
    frequency_chunk: int,
    projection_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return WDM-projected OMS, TM, and Galactic truth for one channel."""
    if cache_path.exists():
        with np.load(cache_path) as cache:
            arrays = {name: cache[name] for name in cache.files}
        if _truth_cache_matches(
            arrays,
            time_tcb,
            frequency_hz,
            delta_f_hz,
            projection_nodes,
        ):
            return (
                arrays["noise_oms_psd"],
                arrays["noise_tm_psd"],
                arrays["galactic_psd"],
                json.loads(str(arrays["validation_json"])),
            )

    started = time.perf_counter()
    noise_oms, noise_tm = projected_analytic_channel_noise_components_psd(
        channel,
        orbit_path,
        time_tcb,
        frequency_hz,
        delta_f_hz,
        projection_nodes=projection_nodes,
        frequency_chunk=frequency_chunk,
    )
    with h5py.File(archive_path, "r") as hdf:
        source_t = hdf["truth/time_tcb"][:]
        source_f = hdf["truth/frequency_hz"][:]
        source_galactic = _channel_truth(hdf["truth/galactic_psd"][:], channel)
        source_noise = _channel_truth(hdf["truth/noise_psd"][:], channel)
    galactic = projected_interpolated_positive_surface(
        source_galactic,
        source_t,
        source_f,
        time_tcb,
        frequency_hz,
        delta_f_hz,
        projection_nodes=projection_nodes,
        frequency_chunk=frequency_chunk,
        zero_outside_frequency=True,
    )
    direct_on_archive = analytic_channel_noise_psd(
        channel,
        orbit_path,
        source_t,
        source_f,
        frequency_chunk=frequency_chunk,
    )
    validation = {
        "direct_truth_runtime_s": float(time.perf_counter() - started),
        "archive_noise_median_abs_log_ratio": float(
            np.median(np.abs(np.log(direct_on_archive / source_noise)))
        ),
        "archive_noise_max_abs_log_ratio": float(
            np.max(np.abs(np.log(direct_on_archive / source_noise)))
        ),
        "estimand": "WDM frequency-kernel projected marginal power",
        "projection_nodes": int(projection_nodes),
        "delta_f_hz": float(delta_f_hz),
        "projection_cache_version": WDM_PROJECTION_CACHE_VERSION,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        time_tcb=time_tcb,
        frequency_hz=frequency_hz,
        delta_f_hz=np.asarray(delta_f_hz),
        projection_nodes=np.asarray(projection_nodes),
        projection_cache_version=np.asarray(WDM_PROJECTION_CACHE_VERSION),
        noise_oms_psd=noise_oms,
        noise_tm_psd=noise_tm,
        galactic_psd=galactic,
        validation_json=np.asarray(json.dumps(validation)),
    )
    return noise_oms, noise_tm, galactic, validation


def _jsonify(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value)}")


def _git_receipt(path: Path) -> dict[str, Any]:
    """Return a compact, non-mutating revision receipt for one checkout."""
    result: dict[str, Any] = {"path": str(path.resolve())}
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result.update({"commit": head, "dirty": bool(dirty.strip())})
    except (OSError, subprocess.CalledProcessError) as error:
        result["unavailable"] = str(error)
    return result


def _file_receipt(path: Path) -> dict[str, Any]:
    """Record stable path/size/mtime identity without hashing multi-GB inputs."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def runtime_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Software, checkout, command, and input receipt stored with every run."""
    versions: dict[str, str] = {}
    for distribution in (
        "numpy",
        "scipy",
        "h5py",
        "jax",
        "jaxlib",
        "numpyro",
        "wdm-transform",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed-as-distribution"
    return {
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": versions,
        "study_checkout": _git_receipt(HERE),
        "package_checkout": _git_receipt(PACKAGE_ROOT),
        "inputs": {
            "archive": _file_receipt(args.archive),
            "orbits": _file_receipt(args.orbits),
        },
    }


def save_chain_archive(
    path: Path,
    fit: dict[str, Any],
    stationary_fit: dict[str, Any],
) -> None:
    """Save chain-preserving parameters and reconstruction bases."""
    payload: dict[str, np.ndarray] = {}
    for prefix, result in (("tv", fit), ("stationary", stationary_fit)):
        grouped_samples = result["mcmc"].get_samples(group_by_chain=True)
        grouped_extra = result["mcmc"].get_extra_fields(group_by_chain=True)
        for name, values in grouped_samples.items():
            payload[f"{prefix}_sample_{name}"] = np.asarray(values)
        for name, values in grouped_extra.items():
            payload[f"{prefix}_extra_{name}"] = np.asarray(values)
    payload["frequency_hz"] = np.asarray(fit["freq_grid"])
    payload["time_grid"] = np.asarray(fit["time_grid"])
    if fit["residual_structure"] == "stationary_plus_interaction":
        payload["tv_basis_time"] = np.asarray(fit["basis_interaction_time"])
        payload["tv_basis_frequency"] = np.asarray(fit["basis_nested_freq"])
    else:
        payload["tv_basis_time"] = np.asarray(fit["B_time"]) @ np.asarray(
            fit["whitened"]["U_time"]
        )
        payload["tv_basis_frequency"] = np.asarray(fit["B_freq"]) @ np.asarray(
            fit["whitened"]["U_freq"]
        )
    payload["stationary_basis_frequency"] = np.asarray(
        stationary_fit["basis_eig_freq"]
    )
    np.savez_compressed(path, **payload)


def run(args: argparse.Namespace) -> Path:
    """Execute one continuous or one gapped ESA-orbit M0 fit for ``args.channel``."""
    if min(args.offset_oms_scale, args.offset_tm_scale, args.offset_pivot_hz) <= 0.0:
        raise ValueError("response-offset scales and pivot frequency must be positive")
    if args.single_gap_days <= 0.0 or args.gap_buffer_pixels < 0.0:
        raise ValueError("single-gap duration must be positive and buffer non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.archive, "r") as hdf:
        dt = float(hdf.attrs["dt_seconds"])
        t0_tcb = float(hdf.attrs["t0_tcb"])
        requested = int(hdf.attrs["n_samples"])
        n_total = wdm_valid_length(requested, args.nt)
        if args.channel in XYZ_CHANNELS:
            clean_data = hdf["tdi/total"][XYZ_CHANNELS.index(args.channel), :n_total]
        else:
            xyz_total = hdf["tdi/total"][:, :n_total]
            clean_data = xyz_to_aet_series(xyz_total)[AET_CHANNELS.index(args.channel)]
        archive_attrs = {
            "noise_source": str(hdf.attrs["noise_source"]),
            "orbit_source": str(hdf.attrs["orbit_source"]),
            "galactic_amplitude_scale": float(hdf.attrs["galactic_amplitude_scale"]),
        }

    nf = n_total // args.nt
    # The real WDM transform has ``nf + 1`` frequency channels over the
    # one-sided band with spacing 1 / (2 * nf * dt).
    df = 1.0 / (2.0 * nf * dt)
    trim_low = max(1, int(np.ceil(args.fmin / df)))
    trim_high = max(1, int(nf - np.floor(args.fmax / df)))
    config = PSplineConfig(
        n_interior_knots_time=args.time_knots,
        n_interior_knots_freq=args.frequency_knots,
        trim_time_bins=1,
        trim_low_freq_channels=trim_low,
        trim_high_freq_channels=trim_high,
        freq_knot_strategy="linear",
        centered=True,
    )

    t_obs_s = n_total * dt
    gaps: list[tuple[float, float]] = []
    analysis_data = clean_data
    if args.mode == "gapped":
        if args.gap_scenario == "lisa_like":
            gaps = lisa_like_gaps(t_obs_s, args.gap_seed)
        else:
            duration = args.single_gap_days * SECONDS_PER_DAY
            center = args.single_gap_center_year * SECONDS_PER_YEAR
            gaps = [
                (
                    max(0.0, center - 0.5 * duration),
                    min(t_obs_s, center + 0.5 * duration),
                )
            ]
        analysis_data = gate_gaps(clean_data, dt, gaps, taper_s=args.taper_hours * 3600.0)

    coefficients, time_grid, frequency_hz = wdm_analysis_coefficients(
        analysis_data,
        dt,
        args.nt,
        config,
    )
    clean_coefficients = coefficients
    if args.mode == "gapped":
        clean_coefficients, clean_time, clean_frequency = wdm_analysis_coefficients(
            clean_data,
            dt,
            args.nt,
            config,
        )
        if not (np.array_equal(time_grid, clean_time) and np.array_equal(frequency_hz, clean_frequency)):
            raise RuntimeError("clean and gapped WDM grids differ")

    time_tcb = t0_tcb + time_grid * t_obs_s
    cache_name = (
        f"esa_{args.channel.lower()}_truth_nt{args.nt}_{args.fmin:g}_{args.fmax:g}"
        f"_wprojv{WDM_PROJECTION_CACHE_VERSION}_n{args.wdm_projection_nodes}.npz"
    )
    noise_oms_truth, noise_tm_truth, galactic_truth, truth_validation = load_or_build_truth(
        args.channel,
        args.output_dir / cache_name,
        args.archive,
        args.orbits,
        time_tcb,
        frequency_hz,
        df,
        frequency_chunk=args.truth_frequency_chunk,
        projection_nodes=args.wdm_projection_nodes,
    )
    noise_truth = noise_oms_truth + noise_tm_truth
    total_truth = noise_truth + galactic_truth
    # Construct the mask and optional offset from a separately evaluated
    # nominal response reference, never from the injected total surface or
    # coefficient realization. In this controlled simulation the nominal
    # parameters match the injection unless explicit misspecification factors
    # are supplied below.
    response_oms_point, response_tm_point = analytic_channel_noise_components_psd(
        args.channel,
        args.orbits,
        time_tcb,
        frequency_hz,
        frequency_chunk=args.truth_frequency_chunk,
    )
    nominal_response_point = response_oms_point + response_tm_point
    response_keep, response_ratio, response_continuum = response_null_mask(
        nominal_response_point,
        frequency_hz,
        high_frequency_hz=args.null_min_frequency,
        ratio_threshold=args.null_ratio_threshold,
        continuum_width=args.null_continuum_width,
        dilation_bins=args.null_dilation_bins,
    )

    good_rows = np.ones(time_grid.size, dtype=bool)
    if gaps:
        good_rows = good_time_bins(
            time_grid,
            t_obs_s,
            gaps,
            args.nt,
            taper_s=args.taper_hours * 3600.0,
            buffer_pixels=args.gap_buffer_pixels,
        )
    split_training, validation_rows, test_rows = analysis_row_split(
        time_grid.size,
        block=args.time_bin,
        cycle=args.split_cycle,
        validation_fold=args.validation_fold,
        test_fold=args.test_fold,
    )
    training_rows = good_rows & split_training
    inference_mask, evaluation_mask = analysis_masks(training_rows, response_keep)
    if not np.any(inference_mask):
        raise RuntimeError("inference mask retained no cells")
    if not np.all(inference_mask[training_rows]):
        raise RuntimeError("inference unexpectedly excludes training frequency cells")

    to_psd = 2.0 * dt / n_total
    component_validation = None
    if not args.skip_component_validation:
        component_validation = validate_archived_components_in_wdm(
            args.archive,
            args.channel,
            n_total,
            dt,
            args.nt,
            config,
            time_grid,
            frequency_hz,
            clean_coefficients,
            noise_truth,
            galactic_truth,
            nominal_response_point,
            response_keep,
            to_psd,
        )
    data_scale = robust_training_psd_scale(coefficients, inference_mask, to_psd)
    normalized_coefficients = coefficients * np.sqrt(to_psd / data_scale)

    pilot_log = training_data_pilot_log_psd(
        normalized_coefficients,
        inference_mask,
        n_time_profiles=args.pilot_time_profiles,
        frequency_width=args.pilot_frequency_width,
    )
    freq_starts = adaptive_frequency_bin_starts(
        pilot_log,
        max_log_range=args.max_log_range,
        max_bin=args.max_frequency_bin,
    )
    time_starts = partition_starts(
        time_grid.size,
        args.time_bin,
        good_rows,
        validation_rows,
        test_rows,
    )
    interior_frequency_knots = hybrid_frequency_knots(
        frequency_hz,
        args.frequency_knots,
        break_hz=args.frequency_knot_break,
    )

    started = time.perf_counter()
    if args.offset_spectral_tilt == 0.0:
        response_oms_projected = noise_oms_truth
        response_tm_projected = noise_tm_truth
    else:
        response_oms_projected, response_tm_projected = (
            projected_analytic_channel_noise_components_psd(
                args.channel,
                args.orbits,
                time_tcb,
                frequency_hz,
                df,
                projection_nodes=args.wdm_projection_nodes,
                frequency_chunk=args.truth_frequency_chunk,
                spectral_tilt=args.offset_spectral_tilt,
                pivot_hz=args.offset_pivot_hz,
            )
        )
    offset_reference = (
        args.offset_oms_scale * response_oms_projected
        + args.offset_tm_scale * response_tm_projected
    )
    response_log_offset = np.log(offset_reference / data_scale) if args.response_offset else None
    fit = fit_log_pspline_surface(
        normalized_coefficients[None, :, :],
        time_grid,
        frequency_hz,
        config=config,
        interior_knots_freq=interior_frequency_knots,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        num_chains=args.chains,
        random_seed=args.random_seed + (args.gap_seed if gaps else 0),
        max_tree_depth=args.max_tree_depth,
        target_accept_prob=args.target_accept,
        progress_bar=not args.no_progress,
        time_bin_starts=time_starts,
        freq_bin_starts=freq_starts,
        likelihood_mask=inference_mask,
        log_psd_offset=response_log_offset,
        residual_structure=args.residual_structure,
        interaction_scale_prior=args.interaction_scale_prior,
        interaction_time_knots=args.interaction_time_knots,
        binning_metadata={
            "time": "fixed blocks split at gap and holdout transitions",
            "frequency": "direct ESA response pilot",
            "max_log_range": args.max_log_range,
            "max_frequency_bin": args.max_frequency_bin,
        },
    )
    wall_s = time.perf_counter() - started
    estimate = np.asarray(fit["psd_geometric_mean"]) * data_scale
    lower = np.asarray(fit["psd_lower"]) * data_scale
    upper = np.asarray(fit["psd_upper"]) * data_scale

    stationary_fit = run_stationary_psd_mcmc(
        normalized_coefficients,
        frequency_hz,
        config=config,
        interior_knots_freq=interior_frequency_knots,
        likelihood_mask=inference_mask,
        freq_bin_starts=freq_starts,
        log_psd_offset=response_log_offset,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        num_chains=args.chains,
        random_seed=args.random_seed + 10_000 + (args.gap_seed if gaps else 0),
        max_tree_depth=args.max_tree_depth,
        target_accept_prob=args.target_accept,
        progress_bar=not args.no_progress,
    )
    stationary_surface = (
        np.asarray(stationary_fit["psd_geometric_mean_surface"]) * data_scale
    )
    stationary_lower_surface = (
        np.asarray(stationary_fit["psd_lower_surface"]) * data_scale
    )
    stationary_upper_surface = (
        np.asarray(stationary_fit["psd_upper_surface"]) * data_scale
    )

    power_psd = coefficients**2 * to_psd
    clean_power_psd = clean_coefficients**2 * to_psd
    stationary_empirical_frequency = masked_mean(
        power_psd,
        inference_mask,
        axis=0,
    )
    ordinary_test = evaluation_mask & test_rows[:, None] & good_rows[:, None]
    ordinary_validation = evaluation_mask & validation_rows[:, None] & good_rows[:, None]
    gap_test = evaluation_mask & (~good_rows)[:, None]
    bands = {
        "low": frequency_hz <= 0.003,
        "retained_full": np.ones(frequency_hz.size, dtype=bool),
        "high_continuum": frequency_hz >= args.null_min_frequency,
    }
    scores: dict[str, Any] = {}
    for label, frequency_select in bands.items():
        ordinary_mask = ordinary_test & frequency_select[None, :]
        scores[f"ordinary_{label}_tv"] = score_surface(
            estimate, lower, upper, total_truth, ordinary_mask
        )
        scores[f"ordinary_{label}_reference"] = score_point_surface(
            offset_reference, total_truth, ordinary_mask
        )
        scores[f"ordinary_{label}_stationary"] = score_surface(
            stationary_surface,
            stationary_lower_surface,
            stationary_upper_surface,
            total_truth,
            ordinary_mask,
        )
        scores[f"ordinary_{label}_stationary_empirical"] = score_stationary(
            stationary_empirical_frequency, total_truth, ordinary_mask
        )
        validation_mask = ordinary_validation & frequency_select[None, :]
        scores[f"validation_{label}_tv"] = score_surface(
            estimate, lower, upper, total_truth, validation_mask
        )
        scores[f"validation_{label}_reference"] = score_point_surface(
            offset_reference, total_truth, validation_mask
        )
        scores[f"validation_{label}_stationary"] = score_surface(
            stationary_surface,
            stationary_lower_surface,
            stationary_upper_surface,
            total_truth,
            validation_mask,
        )
        scores[f"validation_{label}_stationary_empirical"] = score_stationary(
            stationary_empirical_frequency, total_truth, validation_mask
        )
        if gaps:
            gap_mask = gap_test & frequency_select[None, :]
            scores[f"gap_{label}_tv"] = score_surface(
                estimate, lower, upper, total_truth, gap_mask
            )
            scores[f"gap_{label}_reference"] = score_point_surface(
                offset_reference, total_truth, gap_mask
            )
            scores[f"gap_{label}_stationary"] = score_surface(
                stationary_surface,
                stationary_lower_surface,
                stationary_upper_surface,
                total_truth,
                gap_mask,
            )
            scores[f"gap_{label}_stationary_empirical"] = score_stationary(
                stationary_empirical_frequency, total_truth, gap_mask
            )

    blind_diagnostics: dict[str, Any] = {}
    for cohort_name, cohort_rows in (
        ("validation", validation_rows & good_rows),
        ("test", test_rows & good_rows),
    ):
        blind_diagnostics[cohort_name] = {}
        for label, frequency_select in bands.items():
            cohort_mask = (
                response_keep
                & cohort_rows[:, None]
                & frequency_select[None, :]
            )
            model_checks = {
                "reference_only": blind_whitening_diagnostics(
                    coefficients, offset_reference, cohort_mask, to_psd
                ),
                "stationary_residual": blind_whitening_diagnostics(
                    coefficients, stationary_surface, cohort_mask, to_psd
                ),
                "tv_residual": blind_whitening_diagnostics(
                    coefficients, estimate, cohort_mask, to_psd
                ),
            }
            reference_score = model_checks["reference_only"][
                "mean_whittle_log_score"
            ]
            for check in model_checks.values():
                check["mean_log_score_gain_vs_reference"] = float(
                    check["mean_whittle_log_score"] - reference_score
                )
            blind_diagnostics[cohort_name][label] = model_checks
        # The notched cohorts above support stable continuum summaries. This
        # additional truth-free check includes response-null cells and is the
        # stringent diagnostic of the actual inference population.
        all_cell_mask = cohort_rows[:, None] & np.ones_like(evaluation_mask)
        all_cell_checks = {
            "reference_only": blind_whitening_diagnostics(
                coefficients, offset_reference, all_cell_mask, to_psd
            ),
            "stationary_residual": blind_whitening_diagnostics(
                coefficients, stationary_surface, all_cell_mask, to_psd
            ),
            "tv_residual": blind_whitening_diagnostics(
                coefficients, estimate, all_cell_mask, to_psd
            ),
        }
        all_cell_reference_score = all_cell_checks["reference_only"][
            "mean_whittle_log_score"
        ]
        for check in all_cell_checks.values():
            check["mean_log_score_gain_vs_reference"] = float(
                check["mean_whittle_log_score"] - all_cell_reference_score
            )
        blind_diagnostics[f"{cohort_name}_all_cells"] = all_cell_checks
    if gaps:
        # Missing rows have no coefficients to whiten. Assess the first two
        # retained WDM rows on either side of every excluded/tapered region.
        adjacent_rows = good_rows & binary_dilation(
            ~good_rows, structure=np.ones(5, dtype=bool)
        )
        blind_diagnostics["adjacent_gap"] = {}
        for label, frequency_select in bands.items():
            cohort_mask = (
                response_keep
                & adjacent_rows[:, None]
                & frequency_select[None, :]
            )
            blind_diagnostics["adjacent_gap"][label] = {
                "reference_only": blind_whitening_diagnostics(
                    coefficients, offset_reference, cohort_mask, to_psd
                ),
                "stationary_residual": blind_whitening_diagnostics(
                    coefficients, stationary_surface, cohort_mask, to_psd
                ),
                "tv_residual": blind_whitening_diagnostics(
                    coefficients, estimate, cohort_mask, to_psd
                ),
            }
        blind_diagnostics["adjacent_gap_row_count"] = int(adjacent_rows.sum())
        adjacent_all_mask = adjacent_rows[:, None] & np.ones_like(evaluation_mask)
        blind_diagnostics["adjacent_gap_all_cells"] = {
            "reference_only": blind_whitening_diagnostics(
                coefficients, offset_reference, adjacent_all_mask, to_psd
            ),
            "stationary_residual": blind_whitening_diagnostics(
                coefficients, stationary_surface, adjacent_all_mask, to_psd
            ),
            "tv_residual": blind_whitening_diagnostics(
                coefficients, estimate, adjacent_all_mask, to_psd
            ),
        }

    predictive_cohorts = {
        "low_response_notched": response_keep
        & (frequency_hz[None, :] <= 0.003),
        "full_response_notched": response_keep.copy(),
        "high_response_notched": response_keep
        & (frequency_hz[None, :] >= args.null_min_frequency),
        "full_all_cells": np.ones_like(response_keep, dtype=bool),
    }
    posterior_predictive_comparison = posterior_predictive_score_comparison(
        normalized_coefficients,
        fit,
        stationary_fit,
        np.asarray(fit["log_psd_offset"]),
        test_rows & good_rows,
        predictive_cohorts,
        row_block=args.time_bin,
        frequency_chunk=args.predictive_frequency_chunk,
        bootstrap_replicates=args.score_bootstrap_replicates,
        bootstrap_seed=args.random_seed + 20_000 + (args.gap_seed if gaps else 0),
    )

    low_cell = response_keep & (frequency_hz[None, :] <= 0.003)
    truth_modulation = np.exp(masked_mean(np.log(total_truth), low_cell, axis=1))
    estimate_modulation = np.exp(masked_mean(np.log(estimate), low_cell, axis=1))
    stationary_modulation = np.exp(
        masked_mean(np.log(stationary_surface), low_cell, axis=1)
    )
    (
        estimate_modulation_lower,
        estimate_modulation_median,
        estimate_modulation_upper,
    ) = low_band_modulation_posterior(
        fit,
        np.asarray(fit["log_psd_offset"]),
        frequency_hz <= 0.003,
    )
    estimate_modulation_lower *= data_scale
    estimate_modulation_median *= data_scale
    estimate_modulation_upper *= data_scale
    diagnostics = sampler_diagnostics(fit, args.max_tree_depth)
    stationary_diagnostics = sampler_diagnostics(stationary_fit, args.max_tree_depth)
    metrics = {
        "mode": args.mode,
        "gap_seed": args.gap_seed if gaps else None,
        "archive": str(args.archive.resolve()),
        "orbits": str(args.orbits.resolve()),
        "archive_attrs": archive_attrs,
        "runtime_provenance": runtime_provenance(args),
        "channel": args.channel,
        "dt_s": dt,
        "n_samples": n_total,
        "duration_days": t_obs_s / SECONDS_PER_DAY,
        "nt": args.nt,
        "wdm_shape": list(coefficients.shape),
        "requested_frequency_hz": [args.fmin, args.fmax],
        "realized_frequency_hz": [float(frequency_hz[0]), float(frequency_hz[-1])],
        "frequency_spacing_hz": df,
        "wdm_reference_projection": {
            "applied": True,
            "kernel": "squared compact-support Meyer frequency window",
            "quadrature_nodes": args.wdm_projection_nodes,
            "cache_version": WDM_PROJECTION_CACHE_VERSION,
            "delta_f_hz": df,
            "galactic_zero_outside_archive_frequency_band": True,
        },
        "time_pixel_hours": t_obs_s / args.nt / 3600.0,
        "response_mask_fraction_full": float(1.0 - response_keep.mean()),
        "response_mask_fraction_high": float(1.0 - response_keep[:, frequency_hz >= args.null_min_frequency].mean()),
        "training_cell_fraction": float(inference_mask.mean()),
        "inference_cell_fraction": float(inference_mask.mean()),
        "inference_includes_response_null_cells": True,
        "evaluation_response_notched_fraction": float(1.0 - evaluation_mask.mean()),
        "mask_contract": {
            "inference": "all frequency cells in retained training rows",
            "adaptive_bin_pilot": "same inference mask; includes response-null cells",
            "truth_accuracy": "response-notched validation/test/gap cohorts",
            "blind_whitening_primary": "response-notched cohorts",
            "blind_whitening_stress": "all held-out cells including response nulls",
        },
        "validation_row_fraction": float(validation_rows.mean()),
        "test_row_fraction": float(test_rows.mean()),
        "split_contract": {
            "cycle": args.split_cycle,
            "validation_fold": args.validation_fold,
            "test_fold": args.test_fold,
            "status": "prospectively locked after the original five-fold study",
        },
        "gap_row_fraction": float((~good_rows).mean()),
        "n_gaps": len(gaps),
        "gap_scenario": args.gap_scenario if gaps else None,
        "single_gap_days": args.single_gap_days if args.gap_scenario == "single" and gaps else None,
        "gap_buffer_pixels": args.gap_buffer_pixels,
        "gated_hours": float(sum(stop - start for start, stop in gaps) / 3600.0),
        "frequency_likelihood_bins": int(freq_starts.size),
        "time_likelihood_bins": int(time_starts.size),
        "data_scale": data_scale,
        "data_scale_source": (
            "median retained training WDM power divided by median chi-square_1"
        ),
        "reference_psd_offset": {
            "applied": bool(args.response_offset),
            "reference": (
                f"nominal analytic ESA-orbit {args.channel} instrumental reference PSD"
                if args.response_offset
                else None
            ),
            "interpretation": (
                "free log-multiplicative P-spline residual around a reference "
                "that combines known orbit-dependent TDI transfer functions "
                "with assumed nominal OMS and test-mass noise spectra"
                if args.response_offset
                else "free total log-PSD surface"
            ),
            "known_inputs": (
                "ESA orbit ephemerides, TDI convention, and transfer geometry"
                if args.response_offset
                else None
            ),
            "assumed_inputs": (
                "nominal optical-metrology and test-mass noise spectral shapes and levels"
                if args.response_offset
                else None
            ),
            "oms_scale": args.offset_oms_scale,
            "tm_scale": args.offset_tm_scale,
            "spectral_tilt": args.offset_spectral_tilt,
            "pivot_hz": args.offset_pivot_hz,
            "reference_median_abs_log_error_vs_instrument_truth": float(
                np.median(np.abs(np.log(offset_reference / noise_truth)))
            ),
        },
        "truth_validation": truth_validation,
        "component_wdm_validation": component_validation,
        "sampler": diagnostics,
        "sampler_convergence_gate": convergence_gate_status(diagnostics),
        "stationary_sampler": stationary_diagnostics,
        "stationary_sampler_convergence_gate": convergence_gate_status(
            stationary_diagnostics
        ),
        "stationary_comparator": {
            "form": (
                "log S(t,f) = log S_reference(t,f) + g(f)"
                if args.response_offset
                else "log S(t,f) = g(f)"
            ),
            "same_reference_as_tv": bool(args.response_offset),
            "same_mask_bins_knots_and_sampler_settings": True,
        },
        "tv_residual_structure": {
            # The reference term appears only when one was actually applied:
            # without --reference-psd-offset this is the free total surface
            # (ladder rung 1), and naming a reference it never saw would
            # misreport the very comparison the two rungs exist to make.
            "form": (
                (
                    "log S = log S_reference + g(f) + h(t,f)"
                    if args.residual_structure == "stationary_plus_interaction"
                    else "log S = log S_reference + h(t,f)"
                )
                if args.response_offset
                else "log S = h(t,f)  [free total surface, no reference]"
            ),
            "name": args.residual_structure,
            "interaction_zero_time_mean": bool(
                args.residual_structure == "stationary_plus_interaction"
            ),
            "interaction_scale_prior": (
                args.interaction_scale_prior
                if args.residual_structure == "stationary_plus_interaction"
                else None
            ),
            "interaction_time_knots": (
                args.interaction_time_knots
                if args.residual_structure == "stationary_plus_interaction"
                else None
            ),
            "interaction_parameter_count": (
                (
                    args.interaction_time_knots
                    + config.degree_time
                )
                * (args.frequency_knots + config.degree_freq + 1)
                + (args.frequency_knots + config.degree_freq + 1)
                + 1
                if args.residual_structure == "stationary_plus_interaction"
                else None
            ),
            "plain_tensor_parameter_count": (
                (args.time_knots + config.degree_time + 1)
                * (args.frequency_knots + config.degree_freq + 1)
            ),
            "interaction_scale_posterior": (
                {
                    "median": float(np.median(fit["interaction_scale_samples"])),
                    "p05": float(np.percentile(fit["interaction_scale_samples"], 5.0)),
                    "p95": float(np.percentile(fit["interaction_scale_samples"], 95.0)),
                }
                if fit["interaction_scale_samples"] is not None
                else None
            ),
        },
        "fit_wall_s": wall_s,
        "nuts_runtime_s": float(fit["nuts_runtime_s"]),
        "stationary_nuts_runtime_s": float(stationary_fit["nuts_runtime_s"]),
        "scores": scores,
        "score_contract": {
            "scores": "truth-based geometric-posterior-mean surface metrics",
            "blind_diagnostics": "plug-in whitening and Whittle scores at posterior geometric means",
            "posterior_predictive_comparison": (
                "training-conditioned posterior predictive log density with paired "
                "held-out-block uncertainty"
            ),
        },
        "blind_diagnostics": blind_diagnostics,
        "posterior_predictive_comparison": posterior_predictive_comparison,
    }

    tag = "continuous" if not gaps else f"gapped_seed{args.gap_seed}"
    if gaps and args.gap_scenario == "single":
        tag = (
            f"gapped_single_{args.single_gap_days:g}d"
            f"_buffer{args.gap_buffer_pixels:g}"
        )
    if args.response_offset:
        tag += "_reference_offset"
        if (
            args.offset_oms_scale != 1.0
            or args.offset_tm_scale != 1.0
            or args.offset_spectral_tilt != 0.0
        ):
            tag += (
                f"_oms{args.offset_oms_scale:g}_tm{args.offset_tm_scale:g}"
                f"_tilt{args.offset_spectral_tilt:g}"
            )
    if args.residual_structure == "stationary_plus_interaction":
        tag += "_nested"
    tag += f"_wprojv{WDM_PROJECTION_CACHE_VERSION}_n{args.wdm_projection_nodes}"
    output_path = args.output_dir / f"esa_{args.channel.lower()}_m0_{tag}.npz"
    chain_path = args.output_dir / f"esa_{args.channel.lower()}_m0_{tag}_chains.npz"
    if not args.summary_only:
        metrics["artifacts"] = {
            "surface_archive": str(output_path.resolve()),
            "chain_archive": str(chain_path.resolve()),
            "chain_archive_contract": (
                "chain-preserving sampled sites, NUTS diagnostics, and spline "
                "reconstruction bases"
            ),
        }
        np.savez_compressed(
            output_path,
            time_year=time_grid * t_obs_s / SECONDS_PER_YEAR,
            time_tcb=time_tcb,
            frequency_hz=frequency_hz,
            estimate_psd=estimate,
            lower_psd=lower,
            upper_psd=upper,
            truth_psd=total_truth,
            noise_truth_psd=noise_truth,
            galactic_truth_psd=galactic_truth,
            reference_psd=offset_reference,
            point_reference_psd=nominal_response_point,
            response_keep=response_keep,
            response_ratio=response_ratio,
            response_continuum_psd=response_continuum,
            good_rows=good_rows,
            heldout_rows=test_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            likelihood_mask=inference_mask,
            inference_mask=inference_mask,
            evaluation_mask=evaluation_mask,
            stationary_psd=stationary_surface,
            stationary_empirical_psd=stationary_empirical_frequency,
            stationary_lower_psd=stationary_lower_surface,
            stationary_upper_psd=stationary_upper_surface,
            truth_modulation=truth_modulation,
            estimate_modulation=estimate_modulation,
            estimate_modulation_lower=estimate_modulation_lower,
            estimate_modulation_median=estimate_modulation_median,
            estimate_modulation_upper=estimate_modulation_upper,
            stationary_modulation=stationary_modulation,
            time_bin_starts=time_starts,
            frequency_bin_starts=freq_starts,
            clean_power_psd=clean_power_psd if gaps else np.empty((0, 0)),
            gap_schedule_s=np.asarray(gaps, dtype=float).reshape(-1, 2),
            metrics_json=np.asarray(json.dumps(metrics, default=_jsonify, indent=2)),
        )
        save_chain_archive(chain_path, fit, stationary_fit)
    else:
        metrics["archive"] = None
        metrics["archive_policy"] = "summary_only"
    json_path = args.output_dir / f"esa_{args.channel.lower()}_m0_{tag}.json"
    json_path.write_text(
        json.dumps(metrics, default=_jsonify, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, default=_jsonify, indent=2))
    print(f"[saved] {json_path if args.summary_only else output_path}")
    return json_path if args.summary_only else output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("continuous", "gapped"), default="continuous")
    parser.add_argument("--gap-seed", type=int, default=1)
    parser.add_argument("--gap-scenario", choices=("lisa_like", "single"), default="lisa_like")
    parser.add_argument("--single-gap-days", type=float, default=7.0)
    parser.add_argument("--single-gap-center-year", type=float, default=0.5)
    parser.add_argument("--gap-buffer-pixels", type=float, default=1.0)
    parser.add_argument(
        "--channel",
        choices=ALL_CHANNELS,
        default="X2",
        help=(
            "X2/Y2/Z2 read the archive directly; A/E/T are the orthogonal "
            "rotation of X/Y/Z, applied to the time series before the WDM "
            "transform, and (for truth/reference comparison only) to the "
            "per-channel PSD under the zero-XYZ-cross-spectrum contract"
        ),
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--orbits", type=Path, default=DEFAULT_ORBITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fmin", type=float, default=1.0e-4)
    parser.add_argument("--fmax", type=float, default=1.0e-1)
    parser.add_argument("--nt", type=int, default=2048)
    parser.add_argument("--time-knots", type=int, default=8)
    parser.add_argument("--frequency-knots", type=int, default=60)
    parser.add_argument("--frequency-knot-break", type=float, default=1.0e-2)
    parser.add_argument("--time-bin", type=int, default=4)
    parser.add_argument("--split-cycle", type=int, default=7)
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=6)
    parser.add_argument("--pilot-time-profiles", type=int, default=32)
    parser.add_argument("--pilot-frequency-width", type=int, default=31)
    parser.add_argument("--max-log-range", type=float, default=0.25)
    parser.add_argument("--max-frequency-bin", type=int, default=24)
    parser.add_argument("--predictive-frequency-chunk", type=int, default=256)
    parser.add_argument("--score-bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--null-min-frequency", type=float, default=0.02)
    parser.add_argument("--null-ratio-threshold", type=float, default=0.35)
    parser.add_argument("--null-continuum-width", type=int, default=81)
    parser.add_argument("--null-dilation-bins", type=int, default=6)
    parser.add_argument("--taper-hours", type=float, default=1.0)
    parser.add_argument(
        "--truth-frequency-chunk",
        type=int,
        default=384,
        help=(
            "Maximum total quadrature frequencies in one analytic-response "
            "evaluation block."
        ),
    )
    parser.add_argument(
        "--wdm-projection-nodes",
        type=int,
        default=16,
        help="Gauss-Legendre nodes used to project each interior WDM channel.",
    )
    parser.add_argument(
        "--skip-component-validation",
        action="store_true",
        help=(
            "Skip the independent archived noise/Galactic WDM projection check. "
            "Publication runs should leave this enabled."
        ),
    )
    parser.add_argument("--n-warmup", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--max-tree-depth", type=int, default=12)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument(
        "--reference-psd-offset",
        "--response-offset",
        dest="response_offset",
        action="store_true",
        help=(
            "Fit a multiplicative P-spline residual around the nominal analytic "
            "instrumental reference PSD. The --response-offset spelling is retained "
            "as a backwards-compatible alias."
        ),
    )
    parser.add_argument("--offset-oms-scale", type=float, default=1.0)
    parser.add_argument("--offset-tm-scale", type=float, default=1.0)
    parser.add_argument("--offset-spectral-tilt", type=float, default=0.0)
    parser.add_argument("--offset-pivot-hz", type=float, default=1.0e-2)
    parser.add_argument(
        "--residual-structure",
        choices=("tensor", "stationary_plus_interaction"),
        default="tensor",
    )
    parser.add_argument("--interaction-scale-prior", type=float, default=0.5)
    parser.add_argument("--interaction-time-knots", type=int, default=5)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write metrics JSON without the large posterior-surface NPZ archive.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
