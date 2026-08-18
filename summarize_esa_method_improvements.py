"""Summarize the locked free and response-aware ESA surface fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BANDS = ("low", "retained_full", "high_continuum")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "esa_m0_method_results",
    )
    args = parser.parse_args()
    free = load(args.results / "esa_x2_m0_continuous.json")
    offset = load(
        args.results
        / "time_knots5"
        / "esa_x2_m0_continuous_reference_offset_nested.json"
    )

    output = {
        "scope": "single ESA-orbit X2 realization; prospective block test split",
        "deferred": [
            "independent-realization coverage",
            "WDM-window-projected truth",
            "Y2/Z2 replication",
        ],
        "models": {
            "free_total_surface": {
                "frequency_knots": 48,
                "frequency_bins": free["frequency_likelihood_bins"],
                "sampler": free["sampler"],
            },
            "response_aware_stationary_plus_interaction": {
                "frequency_knots": 36,
                "frequency_bins": offset["frequency_likelihood_bins"],
                "sampler": offset["sampler"],
                "reference": offset["reference_psd_offset"],
                "structure": offset["tv_residual_structure"],
            },
            "response_aware_stationary_residual": {
                "sampler": offset["stationary_sampler"],
                "model": offset.get("stationary_comparator"),
            },
            "matched_stationary_pspline": {
                "sampler": free["stationary_sampler"],
            },
        },
        "prospective_test": {},
    }
    for band in BANDS:
        output["prospective_test"][band] = {
            "free_tv": free["scores"][f"ordinary_{band}_tv"],
            "response_aware_tv": offset["scores"][f"ordinary_{band}_tv"],
            "response_aware_stationary": offset["scores"][f"ordinary_{band}_stationary"],
            "matched_stationary_pspline": free["scores"][f"ordinary_{band}_stationary"],
            "empirical_stationary_mean": free["scores"][f"ordinary_{band}_stationary_empirical"],
            "blind": offset["blind_diagnostics"]["test"][band],
        }
    path = args.results / "time_knots5" / "esa_x2_method_comparison.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
