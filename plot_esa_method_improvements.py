"""Publication figures for the selected ESA-orbit X2 nested M0 analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CONTINUOUS = (
    HERE
    / "esa_m0_method_results"
    / "time_knots5"
    / "esa_x2_m0_continuous_reference_offset_nested.npz"
)
GAP_ANCHOR = (
    HERE
    / "esa_m0_gap_results"
    / "duration7_buffer1"
    / "esa_x2_m0_gapped_single_7d_buffer1_reference_offset_nested.npz"
)
OUTPUT = HERE / "esa_m0_publication_results"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def archive_metrics(data: dict[str, np.ndarray]) -> dict:
    return json.loads(str(data["metrics_json"]))


def stationary_surface(data: dict[str, np.ndarray]) -> np.ndarray:
    surface = data["stationary_psd"]
    if surface.ndim == 1:
        surface = np.broadcast_to(surface[None, :], data["truth_psd"].shape)
    return surface


def geometric_low(data: dict[str, np.ndarray], surface: np.ndarray) -> np.ndarray:
    select = data["frequency_hz"] <= 0.003
    keep = data["response_keep"][:, select].astype(bool)
    values = np.log(surface[:, select])
    return np.exp(
        np.divide(
            np.sum(np.where(keep, values, 0.0), axis=1),
            keep.sum(axis=1),
        )
    )


def frequency_rmse(
    estimate: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    smooth: int = 31,
) -> np.ndarray:
    delta2 = (np.log(estimate) - np.log(truth)) ** 2
    count = mask.sum(axis=0)
    raw = np.sqrt(
        np.divide(
            np.sum(np.where(mask, delta2, 0.0), axis=0),
            count,
            out=np.full(count.shape, np.nan, dtype=float),
            where=count > 0,
        )
    )
    kernel = np.ones(smooth)
    valid = np.isfinite(raw)
    denominator = np.convolve(valid.astype(float), kernel, mode="same")
    return np.divide(
        np.convolve(np.where(valid, raw, 0.0), kernel, mode="same"),
        denominator,
        out=np.full_like(raw, np.nan),
        where=denominator > 0,
    )


def save(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def continuous_figure(data: dict[str, np.ndarray], output: Path) -> None:
    metrics = archive_metrics(data)
    time = data["time_year"]
    truth_low = geometric_low(data, data["truth_psd"])
    estimate_low = geometric_low(data, data["estimate_psd"])
    lower_low = geometric_low(data, data["lower_psd"])
    upper_low = geometric_low(data, data["upper_psd"])
    stationary_low = geometric_low(data, stationary_surface(data))
    normalization = np.median(truth_low)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(
        time,
        lower_low / normalization,
        upper_low / normalization,
        color=BLUE,
        alpha=0.18,
        linewidth=0,
        label="90% credible interval",
    )
    ax.plot(time, truth_low / normalization, color="black", linestyle="--", label="Injection")
    ax.plot(time, estimate_low / normalization, color=BLUE, label=r"$R\,e^{g+h}$")
    ax.plot(time, stationary_low / normalization, color=ORANGE, linestyle=":", label=r"$R\,e^g$")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Low-band PSD / median truth")
    ax.set_title("(a) Modulation below 3 mHz")
    ax.legend(frameon=False, ncol=2, loc="upper center")

    ax = axes[1]
    frequency = data["frequency_hz"]
    mask = data["response_keep"].astype(bool) & data["test_rows"].astype(bool)[:, None]
    for estimate, color, line, label in (
        (data["estimate_psd"], BLUE, "-", r"$R\,e^{g+h}$"),
        (stationary_surface(data), ORANGE, "--", r"$R\,e^g$"),
    ):
        ax.plot(
            frequency,
            frequency_rmse(estimate, data["truth_psd"], mask),
            color=color,
            linestyle=line,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlim(frequency[0], frequency[-1])
    ax.set_ylim(0, 0.36)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Held-out log-PSD RMSE")
    ax.set_title("(b) Frequency-resolved accuracy")
    ax.legend(frameon=False)

    ax = axes[2]
    bands = ("low", "retained_full", "high_continuum")
    labels = ("<3 mHz", "Full retained", ">20 mHz")
    interaction_gain = []
    for band in bands:
        values = metrics["blind_diagnostics"]["test"][band]
        interaction_gain.append(
            values["tv_residual"]["mean_whittle_log_score"]
            - values["stationary_residual"]["mean_whittle_log_score"]
        )
    x = np.arange(len(bands))
    ax.plot(x, interaction_gain, color=BLUE, marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=2.0e-5, linscale=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(r"Held-out mean log-score gain of $h$")
    ax.set_title(r"(c) Incremental value of $h(t,f)$")
    save(fig, output)


def gap_paths() -> list[Path]:
    return [
        HERE / "esa_m0_gap_sensitivity/duration1_buffer1/esa_x2_m0_gapped_single_1d_buffer1_reference_offset_nested.json",
        HERE / "esa_m0_gap_results/duration7_buffer1/esa_x2_m0_gapped_single_7d_buffer1_reference_offset_nested.json",
        HERE / "esa_m0_gap_sensitivity/duration30_buffer1/esa_x2_m0_gapped_single_30d_buffer1_reference_offset_nested.json",
        HERE / "esa_m0_gap_sensitivity/duration7_buffer0/esa_x2_m0_gapped_single_7d_buffer0_reference_offset_nested.json",
        HERE / "esa_m0_gap_sensitivity/duration7_buffer2/esa_x2_m0_gapped_single_7d_buffer2_reference_offset_nested.json",
    ]


def gap_figure(data: dict[str, np.ndarray], rows: list[dict], output: Path) -> None:
    time = data["time_year"]
    truth_low = geometric_low(data, data["truth_psd"])
    estimate_low = geometric_low(data, data["estimate_psd"])
    lower_low = geometric_low(data, data["lower_psd"])
    upper_low = geometric_low(data, data["upper_psd"])
    normalization = np.median(truth_low)
    schedule = data["gap_schedule_s"] / (365.25 * 86400.0)
    center = float(np.mean(schedule[0]))
    window = np.abs(time - center) <= 45 / 365.25

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(
        time[window],
        lower_low[window] / normalization,
        upper_low[window] / normalization,
        color=BLUE,
        alpha=0.18,
        linewidth=0,
        label="90% credible interval",
    )
    ax.plot(time[window], truth_low[window] / normalization, color="black", linestyle="--", label="Injection")
    ax.plot(time[window], estimate_low[window] / normalization, color=BLUE, label=r"$R\,e^{g+h}$")
    ax.axvspan(schedule[0, 0], schedule[0, 1], color=GRAY, alpha=0.16, linewidth=0, label="7-day gap")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Low-band PSD / median truth")
    ax.set_title("(a) Interpolation across a gap")
    ax.legend(frameon=False, ncol=2, loc="upper center")

    ax = axes[1]
    buffer_rows = sorted(
        (row for row in rows if row["single_gap_days"] == 7),
        key=lambda row: row["gap_buffer_pixels"],
    )
    buffers = [row["gap_buffer_pixels"] for row in buffer_rows]
    adjacent_z2 = [
        row["blind_diagnostics"]["adjacent_gap"]["low"]["tv_residual"]["mean_z2"]
        for row in buffer_rows
    ]
    ax.plot(buffers, adjacent_z2, color=BLUE, marker="o")
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label=r"$\chi_1^2$ expectation")
    ax.set_xticks(buffers)
    ax.set_xlabel("Excluded WDM pixels per gap edge")
    ax.set_ylabel(r"Adjacent observed mean $w^2/\hat S$")
    ax.set_title("(b) Blind boundary check")
    ax.legend(frameon=False)

    ax = axes[2]
    duration_rows = sorted(
        (row for row in rows if row["gap_buffer_pixels"] == 1),
        key=lambda row: row["single_gap_days"],
    )
    days = [row["single_gap_days"] for row in duration_rows]
    rmse = [row["scores"]["gap_low_tv"]["log_rmse"] for row in duration_rows]
    ax.plot(days, rmse, color=GREEN, marker="o")
    ax.set_xticks(days)
    ax.set_xlabel("Gap duration [days]")
    ax.set_ylabel("Gap-interior log-PSD RMSE")
    ax.set_title("(c) Truth-only interpolation stress test")
    ax.text(
        0.04,
        0.96,
        "Simulation truth only",
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
        color=GRAY,
    )
    save(fig, output)


def sampler_passes(values: dict) -> bool:
    return bool(
        values["divergences"] == 0
        and values["max_rhat"] <= 1.05
        and values["min_ess"] >= 50
        and values["tree_depth_saturation"] <= 0.05
        and values["min_ebfmi"] >= 0.3
    )


def write_summary(continuous: dict[str, np.ndarray], rows: list[dict], path: Path) -> None:
    continuous_metrics = archive_metrics(continuous)
    gap_rows = []
    for row in rows:
        adjacent = row["blind_diagnostics"]["adjacent_gap"]["low"]["tv_residual"]
        gap_rows.append(
            {
                "duration_days": row["single_gap_days"],
                "buffer_pixels": row["gap_buffer_pixels"],
                "gap_row_fraction": row["gap_row_fraction"],
                "sampler": row["sampler"],
                "passes_sampler_gates": sampler_passes(row["sampler"]),
                "adjacent_low_mean_z2": adjacent["mean_z2"],
                "adjacent_low_central_90_fraction": adjacent["central_90_fraction"],
                "gap_low_log_rmse_truth_only": row["scores"]["gap_low_tv"]["log_rmse"],
                "gap_low_coverage_90_truth_only": row["scores"]["gap_low_tv"]["coverage_90"],
                "test_low_tv_minus_stationary_log_score": (
                    row["blind_diagnostics"]["test"]["low"]["tv_residual"]["mean_whittle_log_score"]
                    - row["blind_diagnostics"]["test"]["low"]["stationary_residual"]["mean_whittle_log_score"]
                ),
            }
        )
    summary = {
        "scope": "single archived ESA-orbit X2 realization; M0 total PSD",
        "data": {
            "archive": continuous_metrics["archive"],
            "orbits": continuous_metrics["orbits"],
            "duration_days": continuous_metrics["duration_days"],
            "frequency_hz": continuous_metrics["realized_frequency_hz"],
            "wdm_shape": continuous_metrics["wdm_shape"],
        },
        "selected_model": continuous_metrics["tv_residual_structure"],
        "continuous_sampler": continuous_metrics["sampler"],
        "continuous_passes_sampler_gates": sampler_passes(continuous_metrics["sampler"]),
        "continuous_scores": continuous_metrics["scores"],
        "continuous_blind_diagnostics": continuous_metrics["blind_diagnostics"],
        "gap_runs": gap_rows,
        "interpretation": {
            "selected_gap_buffer_pixels": 1,
            "selection_basis": "blind low-band whitening on observed rows adjacent to the gap",
            "gap_interior_metrics": "truth-only simulation stress tests, not externally observable diagnostics",
            "short_sensitivity_runs": "point-estimate robustness only when sampler gates fail",
        },
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    continuous = load_npz(CONTINUOUS)
    gap_anchor = load_npz(GAP_ANCHOR)
    gap_rows = [load_json(path) for path in gap_paths()]
    continuous_figure(continuous, OUTPUT / "esa_x2_continuous_nested")
    gap_figure(gap_anchor, gap_rows, OUTPUT / "esa_x2_gap_robustness")
    write_summary(continuous, gap_rows, OUTPUT / "esa_x2_publication_summary.json")
    print(OUTPUT)


if __name__ == "__main__":
    main()
