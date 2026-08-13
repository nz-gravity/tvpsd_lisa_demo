"""Fast OzSTAR preflight for the corrected coarse-reference M0 likelihood."""

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
from esa_m0_study import analysis_masks  # noqa: E402
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

    print(f"[preflight passed] package={imported}")
    print(f"[preflight passed] archive={archive}")
    print(f"[preflight passed] reference_handling={handling}")
    print("[preflight passed] inference includes response-null training cells")


if __name__ == "__main__":
    main()
