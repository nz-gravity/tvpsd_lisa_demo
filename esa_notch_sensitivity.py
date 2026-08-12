"""Score locked ESA M0 fits over a grid of response-notch definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_dilation


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent / "wdm_psd"
for path in (HERE, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def load(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in names}


def log_rmse(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    delta = np.log(estimate[mask]) - np.log(truth[mask])
    return float(np.sqrt(np.mean(delta**2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=HERE / "esa_m0_method_results",
    )
    args = parser.parse_args()
    free = load(
        args.results / "esa_x2_m0_continuous.npz",
        (
            "frequency_hz",
            "truth_psd",
            "response_ratio",
            "test_rows",
            "estimate_psd",
            "stationary_empirical_psd",
        ),
    )
    offset = load(
        args.results
        / "time_knots5"
        / "esa_x2_m0_continuous_reference_offset_nested.npz",
        ("estimate_psd", "stationary_psd"),
    )
    frequency = free["frequency_hz"]
    truth = free["truth_psd"]
    response_ratio = free["response_ratio"]
    test_rows = free["test_rows"].astype(bool)
    bands = {
        "retained_full": np.ones(frequency.size, dtype=bool),
        "high_continuum": frequency >= 0.02,
    }
    rows = []
    for threshold in (0.20, 0.35, 0.50):
        for dilation in (3, 6, 9):
            core = (frequency[None, :] >= 0.02) & (response_ratio < threshold)
            keep = ~binary_dilation(
                core,
                structure=np.ones((1, 2 * dilation + 1), dtype=bool),
            )
            record = {
                "ratio_threshold": threshold,
                "dilation_bins": dilation,
                "excluded_fraction_full": float(1.0 - keep.mean()),
                "excluded_fraction_high": float(1.0 - keep[:, frequency >= 0.02].mean()),
                "scores": {},
            }
            for label, select in bands.items():
                mask = keep & test_rows[:, None] & select[None, :]
                record["scores"][label] = {
                    "free_tv_log_rmse": log_rmse(free["estimate_psd"], truth, mask),
                    "response_aware_tv_log_rmse": log_rmse(offset["estimate_psd"], truth, mask),
                    "response_aware_stationary_log_rmse": log_rmse(
                        offset["stationary_psd"], truth, mask
                    ),
                    "empirical_stationary_log_rmse": log_rmse(
                        np.broadcast_to(free["stationary_empirical_psd"][None, :], truth.shape),
                        truth,
                        mask,
                    ),
                }
            rows.append(record)
    output = {
        "scope": "rescoring locked fits; no mask-specific refitting",
        "rows": rows,
    }
    path = args.results / "time_knots5" / "esa_x2_notch_scoring_sensitivity.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
