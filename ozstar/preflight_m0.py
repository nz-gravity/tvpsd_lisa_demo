"""Fast OzSTAR preflight for projected-reference M0 publication runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

# Python puts this script's own directory (ozstar/) on sys.path, not the CWD,
# so the study modules in the parent directory are not importable by default
# even when sbatch runs `cd "$LISA_DIR"; python ozstar/preflight_m0.py`.
STUDY_ROOT = Path(__file__).resolve().parent.parent
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

import tv_pspline_psd  # noqa: E402
from esa_m0_study import (  # noqa: E402
    analysis_masks,
    lisa_like_gaps,
    analytic_channel_noise_psd,
    projected_analytic_channel_noise_components_psd,
    wdm_valid_length,
)
from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface  # noqa: E402
from tv_pspline_psd.inference import _reference_scaled_power  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/fred/oz200/avajpeyi/projects/WDM_PSD"),
    )
    parser.add_argument("--lisa-dir", type=Path)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    lisa_dir = (
        args.lisa_dir.resolve()
        if args.lisa_dir is not None
        else base / "tvpsd_lisa_demo"
    )
    package_root = (
        args.package_root.resolve()
        if args.package_root is not None
        else (base / "TVPsplinePSD").resolve()
    )

    imported = Path(tv_pspline_psd.__file__).resolve()
    if package_root not in imported.parents:
        raise RuntimeError(
            f"imported tv_pspline_psd from {imported}, expected checkout {package_root}"
        )

    archive = lisa_dir / "combined_esa_xyz.h5"
    orbits = lisa_dir / "noise2a" / "orbits.h5"
    for path in (archive, orbits):
        if not path.is_file():
            raise FileNotFoundError(path)
    with h5py.File(archive, "r") as hdf:
        for key in ("tdi/total", "truth/noise_psd", "truth/galactic_psd"):
            if key not in hdf:
                raise RuntimeError(f"archive is missing {key}")
        dt = float(hdf.attrs["dt_seconds"])
        t0_tcb = float(hdf.attrs["t0_tcb"])
        requested_samples = int(hdf.attrs["n_samples"])

    power = np.array([[2.0, 8.0, 18.0]])
    reference = np.array([[1.0e-8, 4.0, 9.0]])
    scaled = _reference_scaled_power(power, np.log(reference))
    if not np.allclose(scaled, power / reference):
        raise RuntimeError("cellwise reference scaling failed")
    geometric_pool = power.sum() / np.exp(np.mean(np.log(reference)))
    if np.isclose(scaled.sum(), geometric_pool):
        raise RuntimeError("preflight did not distinguish cellwise and pooled scaling")

    training_rows = np.array([True, False, True])
    response_keep = np.array(
        [[True, False, True], [True, True, False], [False, True, True]]
    )
    inference_mask, evaluation_mask = analysis_masks(training_rows, response_keep)
    if not np.all(inference_mask[training_rows]):
        raise RuntimeError("inference mask excludes response-null training cells")
    if not np.any(inference_mask & ~evaluation_mask):
        raise RuntimeError("inference/evaluation masks were not separated")

    # Exercise the actual coarse fitter and verify that its provenance records
    # the corrected branch. This is deliberately tiny; it validates wiring, not
    # posterior convergence.
    rng = np.random.default_rng(42)
    n_time, n_frequency = 8, 9
    time = np.linspace(0.0, 1.0, n_time)
    frequency = np.linspace(0.01, 0.1, n_frequency)
    reference_frequency = np.resize(np.array([1.0e-4, 1.0, 25.0]), n_frequency)
    reference_surface = np.exp(0.2 * time[:, None]) * reference_frequency[None, :]
    residual = rng.normal(size=(1, n_time, n_frequency))
    physical = residual * np.sqrt(reference_surface)[None, :, :]
    config = PSplineConfig(
        n_interior_knots_time=1,
        n_interior_knots_freq=2,
        centered=True,
        trim_time_bins=0,
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        freq_knot_strategy="linear",
    )
    fit = fit_log_pspline_surface(
        physical,
        time,
        frequency,
        config=config,
        time_bin=2,
        freq_bin=3,
        log_psd_offset=np.log(reference_surface),
        n_warmup=1,
        n_samples=1,
        num_chains=1,
        random_seed=1,
        progress_bar=False,
    )
    handling = fit["provenance"]["log_psd_offset"][
        "coarse_likelihood_handling"
    ]
    expected = "cellwise_power_divided_by_reference_before_block_sum"
    if handling != expected:
        raise RuntimeError(f"incorrect coarse-reference handling: {handling!r}")

    # Check the actual production-grid response, not a synthetic polynomial.
    # Sixteen quadrature nodes must agree with a doubled rule at the deepest
    # sampled X2 nulls, while both must differ materially from point evaluation.
    nt = 2048
    n_total = wdm_valid_length(requested_samples, nt)
    nf = n_total // nt
    delta_f = 1.0 / (2.0 * nf * dt)
    duration = n_total * dt
    check_times = t0_tcb + duration * np.array([0.15, 0.50, 0.85])
    check_frequency = (
        np.arange(np.ceil(0.02 / delta_f), np.floor(0.1 / delta_f) + 1)
        * delta_f
    )
    point_grid = analytic_channel_noise_psd(
        "X2", orbits, check_times, check_frequency, frequency_chunk=384
    )
    deepest_indices = np.argmin(point_grid, axis=1)
    deepest_frequency = check_frequency[deepest_indices]
    point_at_null = point_grid[np.arange(check_times.size), deepest_indices]
    projected: dict[int, np.ndarray] = {}
    for nodes in (16, 32):
        values = []
        for time_tcb, frequency_hz in zip(
            check_times, deepest_frequency, strict=True
        ):
            oms, tm = projected_analytic_channel_noise_components_psd(
                "X2",
                orbits,
                np.array([time_tcb]),
                np.array([frequency_hz]),
                delta_f,
                projection_nodes=nodes,
                frequency_chunk=384,
            )
            values.append(float(oms[0, 0] + tm[0, 0]))
        projected[nodes] = np.asarray(values)
    projection_relative_change = np.abs(projected[16] / projected[32] - 1.0)
    if np.max(projection_relative_change) > 5.0e-4:
        raise RuntimeError(
            "16-node WDM projection failed convergence check: "
            f"max relative change={np.max(projection_relative_change):.3g}"
        )
    point_ratio = projected[16] / point_at_null
    if np.max(point_ratio) < 10.0:
        raise RuntimeError(
            "production-grid preflight did not resolve a point-evaluation null"
        )

    # M1 reaches the same projected estimand through two thin AET wrappers
    # around the functions checked above. Verify them here: the fit takes about
    # an hour, and a shape or normalization error would otherwise surface only
    # in the final metrics.
    from run_aet_diagonal_pilot import (  # noqa: E402
        projected_aet_interpolated_surface,
    )

    flat_time = np.linspace(0.0, 1.0, 5)
    flat_frequency = np.geomspace(1.0e-4, 1.0e-1, 32)
    flat_source = np.full((3, flat_time.size, flat_frequency.size), 7.0)
    flat_target = np.geomspace(2.0e-4, 5.0e-2, 6)
    flat_projected = projected_aet_interpolated_surface(
        flat_source,
        flat_time,
        flat_frequency,
        flat_time,
        flat_target,
        1.0e-6,
        projection_nodes=16,
        frequency_chunk=96,
    )
    expected_shape = (3, flat_time.size, flat_target.size)
    if flat_projected.shape != expected_shape:
        raise RuntimeError(
            f"projected AET surface has shape {flat_projected.shape}, "
            f"expected {expected_shape}"
        )
    # Quadrature weights include the squared Meyer window and sum to one, so a
    # constant spectrum must survive projection unchanged.
    if not np.allclose(flat_projected, 7.0, rtol=1.0e-10):
        raise RuntimeError(
            "AET projection wrapper did not preserve a constant surface: max "
            f"deviation {np.max(np.abs(flat_projected - 7.0)):.3g}"
        )

    # The three-model gap comparison is only meaningful if all three jobs see
    # the SAME outages. The schedule is lisa_like_gaps(t_obs_s, seed), and
    # t_obs_s is derived independently in esa_m0_study and
    # run_aet_diagonal_pilot by two separate copies of wdm_valid_length. They
    # agree today; nothing enforces it. Check here, because a divergence would
    # silently turn "same data, different models" into three different
    # datasets, with no error raised anywhere.
    import run_aet_diagonal_pilot as m1  # noqa: E402

    if m1.wdm_valid_length(requested_samples, nt) != n_total:
        raise RuntimeError(
            f"record length differs between the runners at nt={nt}: "
            f"M0 {n_total} vs M1 {m1.wdm_valid_length(requested_samples, nt)}; "
            "the lisa_like gap schedules would not match"
        )
    for seed in (1, 2):
        if lisa_like_gaps(duration, seed) != m1.lisa_like_gaps(duration, seed):
            raise RuntimeError(
                f"lisa_like gap schedules differ between runners at seed={seed}"
            )
    shared_schedule = lisa_like_gaps(duration, 1)

    print(f"[preflight passed] package={imported}")
    print(f"[preflight passed] archive={archive}")
    print(f"[preflight passed] reference_handling={handling}")
    print("[preflight passed] inference includes response-null training cells")
    print("[preflight passed] AET projection wrappers preserve shape and constants")
    print(
        f"[preflight passed] shared gap schedule: t_obs={t_obs_s:.0f}s, "
        f"{len(shared_schedule)} lisa_like gaps identical across both runners"
    )
    print(
        "[preflight passed] 16-vs-32 node projection max relative change="
        f"{np.max(projection_relative_change):.3g}"
    )
    print(
        "[preflight passed] projected/point response at deepest checked null="
        f"{np.max(point_ratio):.3g}"
    )


if __name__ == "__main__":
    main()
