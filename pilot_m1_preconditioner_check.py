"""Sanity check: does a likelihood-aware preconditioner fix M1's slow NUTS?

Loads the AET data once (same steps as run_aet_diagonal_pilot.py --component-noise
--no-fit-t-null-leakage), then fits it with BOTH the original
`fit_aet_component_noise_nuts` and the new
`fit_aet_component_noise_nuts_preconditioned` at a small warmup/samples count,
and prints tree-depth-saturation, divergences, and wall time side by side.
Small counts are enough to see whether the preconditioner changes step-size
adaptation and tree depth; it is not meant to produce a publishable posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent / "wdm_psd"
for path in (HERE, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aet_component_pspline_nuts import fit_aet_component_noise_nuts
from aet_component_pspline_nuts_preconditioned import (
    fit_aet_component_noise_nuts_preconditioned,
)
from aet_diagonal import AET_CHANNELS, diagonal_xyz_psd_to_aet, xyz_to_aet_series
from run_aet_diagonal_pilot import (
    aet_noise_transfer_functions,
    analytic_aet_noise_components_psd,
    bin_channels,
    interpolate_surface,
    oms_theory_psd,
    time_block_log_pilot,
    tm_theory_psd,
    wdm_valid_length,
)
from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
from tv_pspline_psd.inference import adaptive_frequency_bin_starts


def prepare_data(args: argparse.Namespace) -> dict:
    archive = Path(args.archive).resolve()
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
        n_interior_knots_time=3,
        n_interior_knots_freq=40,
        trim_time_bins=1,
        trim_low_freq_channels=trim_low,
        trim_high_freq_channels=trim_high,
        freq_knot_strategy="log",
        centered=True,
    )

    coefficients = []
    fit_time_grid = fit_frequency = None
    for channel_series in aet_total:
        channel_coefficients, channel_time, channel_frequency = wdm_analysis_coefficients(
            channel_series, dt, args.nt, config
        )
        coefficients.append(channel_coefficients)
        if fit_time_grid is None:
            fit_time_grid, fit_frequency = channel_time, channel_frequency
    del aet_total
    coefficients = np.stack(coefficients)
    band = fit_frequency <= args.component_fmax_hz
    coefficients = coefficients[:, :, band]
    fit_frequency = fit_frequency[band]
    absolute_time = t0_tcb + np.asarray(fit_time_grid) * n_total * dt

    # Analytic WDM->PSD constant (see run_aet_diagonal_pilot.py:--analytic-calibration).
    conversion = 2.0 * dt / float(n_total)
    observed_psd = coefficients**2 * conversion
    del coefficients

    orbit_path = archive.parent / "noise2a" / "orbits.h5"
    oms_truth_aet, tm_truth_aet = analytic_aet_noise_components_psd(
        orbit_path, truth_time, truth_frequency
    )
    oms_reference_unbinned = interpolate_surface(
        oms_truth_aet, truth_time, truth_frequency, absolute_time, fit_frequency
    )
    tm_reference_unbinned = interpolate_surface(
        tm_truth_aet, truth_time, truth_frequency, absolute_time, fit_frequency
    )
    noise_reference_unbinned = oms_reference_unbinned + tm_reference_unbinned
    galactic_truth_unbinned = interpolate_surface(
        diagonal_xyz_psd_to_aet(truth_galactic_xyz),
        truth_time,
        truth_frequency,
        absolute_time,
        fit_frequency,
    )
    galactic_template_unbinned = galactic_truth_unbinned / injected_amplitude
    del oms_reference_unbinned, tm_reference_unbinned

    pilot_log_psd = time_block_log_pilot(observed_psd, args.pilot_time_blocks)
    bin_starts = adaptive_frequency_bin_starts(
        pilot_log_psd, max_log_range=args.adaptive_max_log_range, max_bin=args.adaptive_max_bin
    )
    observed_binned, counts, grouped_frequency = bin_channels(observed_psd, fit_frequency, bin_starts)
    galactic_template_binned, _, _ = bin_channels(galactic_template_unbinned, fit_frequency, bin_starts)
    del observed_psd, galactic_template_unbinned, noise_reference_unbinned, galactic_truth_unbinned

    fit_valid = counts > 0.0
    if args.t_channel_fmin_hz > 0.0:
        t_index = AET_CHANNELS.index("T")
        fit_valid[t_index] &= grouped_frequency[None, :] >= args.t_channel_fmin_hz

    transfer_tm, transfer_oms = aet_noise_transfer_functions(
        orbit_path, absolute_time, grouped_frequency
    )
    return {
        "observed_binned": np.where(fit_valid, observed_binned, 1.0),
        "counts": np.where(fit_valid, counts, 1.0),
        "grouped_frequency": grouped_frequency,
        "transfer_tm": np.where(fit_valid, transfer_tm, 1.0),
        "transfer_oms": np.where(fit_valid, transfer_oms, 1.0),
        "galactic_template_binned": np.where(fit_valid, galactic_template_binned, 0.0),
        "fit_valid": fit_valid,
        "n_freq_bins": int(grouped_frequency.size),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default=HERE / "combined_esa_xyz.h5")
    p.add_argument("--nt", type=int, default=32)
    p.add_argument("--fmin-hz", type=float, default=1.0e-4)
    p.add_argument("--fmax-hz", type=float, default=0.02)
    p.add_argument("--component-fmax-hz", type=float, default=0.02)
    p.add_argument("--adaptive-max-log-range", type=float, default=0.15)
    p.add_argument("--adaptive-max-bin", type=int, default=32)
    p.add_argument("--pilot-time-blocks", type=int, default=6)
    p.add_argument("--t-channel-fmin-hz", type=float, default=3.0e-3)
    p.add_argument("--component-frequency-knots", type=int, default=12)
    p.add_argument("--phi-tm", type=float, default=1.0e8)
    p.add_argument("--phi-oms", type=float, default=1.0e4)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--samples", type=int, default=150)
    p.add_argument("--target-accept", type=float, default=0.95)
    p.add_argument("--max-tree-depth", type=int, default=10)
    p.add_argument("--no-progress", action="store_true", default=True)
    return p


def run(args: argparse.Namespace) -> None:
    data = prepare_data(args)
    print(f"[data] {data['n_freq_bins']} binned frequency channels, nt={args.nt}")

    common = dict(
        observed_psd=data["observed_binned"],
        counts=data["counts"],
        frequency_hz=data["grouped_frequency"],
        transfer_tm=data["transfer_tm"],
        transfer_oms=data["transfer_oms"],
        tm_theory_psd=tm_theory_psd(data["grouped_frequency"]),
        oms_theory_psd=oms_theory_psd(data["grouped_frequency"]),
        galactic_template_psd=data["galactic_template_binned"],
        mask=data["fit_valid"],
        n_frequency_knots=args.component_frequency_knots,
        phi_tm=args.phi_tm,
        phi_oms=args.phi_oms,
        n_warmup=args.warmup,
        n_samples=args.samples,
        num_chains=2,
        target_accept_probability=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        progress_bar=not args.no_progress,
    )

    print("\n=== ORIGINAL (prior-only whitening) ===")
    original = fit_aet_component_noise_nuts(
        fit_t_null_leakage=False, t_leakage_centre=3.0e-5, **common
    )
    print(json.dumps(original.diagnostics, indent=2))
    print(f"runtime_seconds: {original.runtime_seconds:.1f}")
    print(f"amplitude_median: {float(np.median(original.amplitude_draws)):.4f}")

    print("\n=== PRECONDITIONED (prior + likelihood Fisher whitening) ===")
    preconditioned = fit_aet_component_noise_nuts_preconditioned(**common)
    print(json.dumps(preconditioned.diagnostics, indent=2))
    print(f"runtime_seconds: {preconditioned.runtime_seconds:.1f}")
    print(f"amplitude_median: {float(np.median(preconditioned.amplitude_draws)):.4f}")

    print("\n=== SUMMARY ===")
    speedup = original.runtime_seconds / max(preconditioned.runtime_seconds, 1e-9)
    print(f"speedup: {speedup:.2f}x")
    print(
        "max_num_steps: original="
        f"{original.diagnostics['max_num_steps']} preconditioned="
        f"{preconditioned.diagnostics['max_num_steps']}"
    )
    print(
        "tree_depth_saturation_fraction: original="
        f"{original.diagnostics['tree_depth_saturation_fraction']:.3f} preconditioned="
        f"{preconditioned.diagnostics['tree_depth_saturation_fraction']:.3f}"
    )
    print(
        "amplitude agreement (should be close): original="
        f"{float(np.median(original.amplitude_draws)):.4f} preconditioned="
        f"{float(np.median(preconditioned.amplitude_draws)):.4f}"
    )


if __name__ == "__main__":
    run(parser().parse_args())
