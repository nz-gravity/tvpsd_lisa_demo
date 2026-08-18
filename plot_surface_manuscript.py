"""Publication figures and compact summary for the ESA-orbit X2 surface study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import chi2


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
LIGHT_GRAY = "#E6E6E6"


def paper_style() -> None:
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
            "lines.linewidth": 1.4,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def metrics(data: dict[str, np.ndarray]) -> dict:
    return json.loads(str(data["metrics_json"]))


def masked_frequency_rmse(
    estimate: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    smooth: int = 31,
) -> np.ndarray:
    delta2 = (np.log(estimate) - np.log(truth)) ** 2
    numerator = np.sum(np.where(mask, delta2, 0.0), axis=0)
    denominator = np.sum(mask, axis=0)
    rmse = np.sqrt(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
    if smooth > 1:
        kernel = np.ones(smooth)
        valid = np.isfinite(rmse)
        rmse = np.divide(
            np.convolve(np.where(valid, rmse, 0.0), kernel, mode="same"),
            np.convolve(valid.astype(float), kernel, mode="same"),
            out=np.full_like(rmse, np.nan),
            where=np.convolve(valid.astype(float), kernel, mode="same") > 0,
        )
    return rmse


def low_band_summary(data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    frequency = data["frequency_hz"]
    select = frequency <= 0.003
    keep = data["response_keep"][:, select]

    def geometric(surface: np.ndarray) -> np.ndarray:
        values = np.log(surface[:, select])
        count = keep.sum(axis=1)
        return np.exp(np.sum(np.where(keep, values, 0.0), axis=1) / count)

    return (
        geometric(data["truth_psd"]),
        geometric(data["estimate_psd"]),
        geometric(data["lower_psd"]),
        geometric(data["upper_psd"]),
    )


def figure_continuous(data: dict[str, np.ndarray], output: Path) -> None:
    time_year = data["time_year"]
    frequency = data["frequency_hz"]
    truth = data["truth_psd"]
    estimate = data["estimate_psd"]
    keep = data["response_keep"]
    heldout = data["heldout_rows"]
    stationary = data["stationary_psd"]
    stationary_surface = (
        np.broadcast_to(stationary[None, :], truth.shape)
        if stationary.ndim == 1
        else stationary
    )
    low_select = frequency <= 0.003
    low_keep = keep[:, low_select]
    stationary_low = np.exp(
        np.sum(
            np.where(low_keep, np.log(stationary_surface[:, low_select]), 0.0),
            axis=1,
        )
        / low_keep.sum(axis=1)
    )
    truth_low, estimate_low, lower_low, upper_low = low_band_summary(data)

    fig = plt.figure(figsize=(7.15, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=(1.35, 0.85, 0.9))
    ax0 = fig.add_subplot(gs[0])
    vmax = np.nanpercentile(estimate[keep], 99.7)
    vmin = np.nanpercentile(estimate[keep], 0.3)
    surface = np.ma.array(estimate, mask=~keep)
    image = ax0.pcolormesh(
        time_year,
        frequency,
        surface.T,
        shading="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        rasterized=True,
    )
    ax0.set_yscale("log")
    ax0.set_ylim(frequency[0], frequency[-1])
    ax0.set_ylabel("Frequency [Hz]")
    ax0.set_xlabel("Time [yr]")
    ax0.set_title("(a) Time-varying X2 PSD; response-null corridors are notched")
    colorbar = fig.colorbar(image, ax=ax0, pad=0.015, aspect=28)
    colorbar.set_label(r"PSD [fractional frequency$^2$/Hz]")

    ax1 = fig.add_subplot(gs[1])
    ax1.fill_between(time_year, lower_low, upper_low, color=BLUE, alpha=0.18, linewidth=0, label="90% credible interval")
    ax1.plot(time_year, truth_low, color="black", linestyle="--", label="Injection truth")
    ax1.plot(time_year, estimate_low, color=BLUE, label="TV P-spline")
    ax1.plot(
        time_year,
        stationary_low,
        color=ORANGE,
        linestyle=":",
        label="Stationary residual model",
    )
    ax1.set_yscale("log")
    ax1.set_ylim(top=max(np.nanmax(upper_low), np.nanmax(stationary_low)) * 1.18)
    ax1.set_xlabel("Time [yr]")
    ax1.set_ylabel(r"Geometric mean PSD$^{\dagger}$")
    ax1.set_title("(b) Annual modulation below 3 mHz")
    ax1.legend(ncol=4, frameon=False, loc="upper center")

    ax2 = fig.add_subplot(gs[2])
    test = keep & heldout[:, None]
    tv_rmse = masked_frequency_rmse(estimate, truth, test)
    stationary_rmse = masked_frequency_rmse(stationary_surface, truth, test)
    ax2.plot(frequency, tv_rmse, color=BLUE, label="TV P-spline")
    ax2.plot(frequency, stationary_rmse, color=ORANGE, linestyle="--", label="Stationary estimate")
    ax2.axvline(0.02, color=GRAY, linestyle=":", linewidth=1.0)
    ax2.text(0.0207, 0.95, "moving-null region", transform=ax2.get_xaxis_transform(), color=GRAY, va="top")
    ax2.set_xscale("log")
    ax2.set_xlim(frequency[0], frequency[-1])
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Held-out log-RMSE")
    ax2.set_title("(c) Predictive accuracy on retained cells (31-channel running mean)")
    ax2.legend(frameon=False)
    ax2.text(
        0.0,
        -0.34,
        r"$^{\dagger}$Geometric mean over retained cells from $1.3\times10^{-4}$ to $3\times10^{-3}$ Hz.",
        transform=ax2.transAxes,
        fontsize=7.2,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def figure_gaps(
    representative: dict[str, np.ndarray],
    schedule_data: list[dict[str, np.ndarray]],
    output: Path,
) -> None:
    time_year = representative["time_year"]
    truth_low, estimate_low, lower_low, upper_low = low_band_summary(representative)
    good = representative["good_rows"].astype(bool)
    gaps = representative["gap_schedule_s"] / (365.25 * 86400.0)

    fig = plt.figure(figsize=(7.15, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=(0.9, 0.9, 1.05))
    ax0 = fig.add_subplot(gs[0])
    ax0.fill_between(time_year, lower_low, upper_low, color=BLUE, alpha=0.18, linewidth=0)
    ax0.plot(time_year, truth_low, color="black", linestyle="--", label="Injection truth")
    ax0.plot(time_year, estimate_low, color=BLUE, label="Gapped TV P-spline")
    for left, right in gaps:
        ax0.axvspan(left, right, color=GRAY, alpha=0.12, linewidth=0)
    ax0.set_yscale("log")
    ax0.set_xlabel("Time [yr]")
    ax0.set_ylabel(r"Geometric mean PSD$^{\dagger}$")
    ax0.set_title("(a) Representative schedule (seed 2); shaded intervals are time-domain gaps")
    ax0.legend(frameon=False, ncol=2)

    ax1 = fig.add_subplot(gs[1])
    bad_indices = np.flatnonzero(~good)
    if bad_indices.size:
        runs = np.split(bad_indices, np.where(np.diff(bad_indices) > 1)[0] + 1)
        run = max(runs, key=len)
        center = int(np.mean(run))
    else:
        center = time_year.size // 2
    half_window = max(18, int(round(7.0 / (365.25 / time_year.size))))
    sl = slice(max(0, center - half_window), min(time_year.size, center + half_window + 1))
    ax1.fill_between(time_year[sl], lower_low[sl], upper_low[sl], color=BLUE, alpha=0.18, linewidth=0)
    ax1.plot(time_year[sl], truth_low[sl], color="black", linestyle="--", label="Injection truth")
    ax1.plot(time_year[sl], estimate_low[sl], color=BLUE, label="Gapped TV P-spline")
    bad = ~good[sl]
    ax1.fill_between(time_year[sl], 0, 1, where=bad, transform=ax1.get_xaxis_transform(), color=GRAY, alpha=0.18, step="mid")
    ax1.set_yscale("log")
    ax1.set_xlabel("Time [yr]")
    ax1.set_ylabel(r"Geometric mean PSD$^{\dagger}$")
    ax1.set_title("(b) Fourteen-day view around the longest contaminated WDM interval")

    ax2 = fig.add_subplot(gs[2])
    categories = ["Below 3 mHz", "Retained full band", "High-frequency\ncontinuum"]
    keys = ["low", "retained_full", "high_continuum"]
    x = np.arange(3, dtype=float)
    jitter = np.linspace(-0.12, 0.12, len(schedule_data))
    for index, data in enumerate(schedule_data):
        score = metrics(data)["scores"]
        tv = np.array([score[f"gap_{key}_tv"]["log_rmse"] for key in keys])
        stationary = np.array([score[f"gap_{key}_stationary"]["log_rmse"] for key in keys])
        x_schedule = x + jitter[index]
        for j in range(3):
            ax2.plot([x_schedule[j] - 0.018, x_schedule[j] + 0.018], [tv[j], stationary[j]], color="#BBBBBB", linewidth=0.8, zorder=1)
        ax2.scatter(x_schedule - 0.018, tv, color=BLUE, marker="o", s=24, zorder=3, label="TV P-spline" if index == 0 else None)
        ax2.scatter(x_schedule + 0.018, stationary, facecolor="white", edgecolor=ORANGE, marker="s", s=24, zorder=3, label="Stationary estimate" if index == 0 else None)
    ax2.set_xticks(x, categories)
    ax2.set_xlim(-0.45, 2.45)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Gap-row log-RMSE")
    ax2.set_title("(c) Five gap schedules applied to the same ESA-orbit realization")
    ax2.legend(frameon=False, ncol=2)
    ax2.text(
        0.0,
        -0.34,
        r"$^{\dagger}$Response-null corridors are excluded from all summaries and accuracy scores.",
        transform=ax2.transAxes,
        fontsize=7.2,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def whitening_scores(
    continuous: dict[str, np.ndarray],
    clean_power_psd: np.ndarray,
) -> dict[str, dict[str, float]]:
    frequency = continuous["frequency_hz"]
    base = continuous["response_keep"].astype(bool) & continuous["heldout_rows"].astype(bool)[:, None]
    ratio = clean_power_psd / continuous["estimate_psd"]
    bands = {
        "low": frequency <= 0.003,
        "retained_full": np.ones(frequency.size, dtype=bool),
        "high_continuum": frequency >= 0.02,
    }
    output = {}
    for label, select in bands.items():
        values = ratio[base & select[None, :]]
        output[label] = {
            "n_cells": int(values.size),
            "mean_z2": float(np.mean(values)),
            "median_z2": float(np.median(values)),
            "q90_z2": float(np.quantile(values, 0.90)),
        }
    return output


def figure_whitening(
    continuous: dict[str, np.ndarray],
    clean_power_psd: np.ndarray,
    output: Path,
) -> None:
    """Appendix check of held-out coefficient power divided by fitted PSD."""
    frequency = continuous["frequency_hz"]
    keep = continuous["response_keep"].astype(bool)
    heldout = continuous["heldout_rows"].astype(bool)
    mask = keep & heldout[:, None]
    z2 = clean_power_psd / continuous["estimate_psd"]

    n_blocks = 70
    edges = np.geomspace(frequency[0], frequency[-1], n_blocks + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    means = np.full(n_blocks, np.nan)
    medians = np.full(n_blocks, np.nan)
    for index in range(n_blocks):
        select = (frequency >= edges[index]) & (frequency < edges[index + 1])
        values = z2[mask & select[None, :]]
        if values.size:
            means[index] = np.mean(values)
            medians[index] = np.median(values)

    values = z2[mask]
    # A fixed deterministic thinning keeps the ECDF vector-sized without
    # selecting favourable cells.
    stride = max(1, values.size // 100_000)
    values = np.sort(values[::stride])
    empirical = np.arange(1, values.size + 1) / values.size
    x_reference = np.geomspace(2e-3, 20.0, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), constrained_layout=True)
    axes[0].plot(centers, means, color=BLUE, label=r"Mean $z^2$")
    axes[0].plot(centers, medians, color=GREEN, linestyle="--", label=r"Median $z^2$")
    axes[0].axhline(1.0, color="black", linestyle=":", label=r"$\chi^2_1$ mean")
    axes[0].axhline(chi2.ppf(0.5, 1), color=GRAY, linestyle="-.", label=r"$\chi^2_1$ median")
    axes[0].axvline(0.02, color=LIGHT_GRAY, linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel(r"Held-out normalized power $z^2$")
    axes[0].set_title("(a) Frequency-binned moments")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(values, empirical, color=BLUE, label="Held-out coefficients")
    axes[1].plot(x_reference, chi2.cdf(x_reference, 1), color="black", linestyle="--", label=r"$\chi^2_1$ reference")
    axes[1].set_xscale("log")
    axes[1].set_xlim(x_reference[0], x_reference[-1])
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r"Normalized power $z^2=w^2/S$")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("(b) Empirical distribution")
    axes[1].legend(frameon=False)
    fig.suptitle("Held-out whitening check on response-retained WDM cells", fontsize=9.5)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def write_summary(continuous: dict[str, np.ndarray], schedules: list[dict[str, np.ndarray]], path: Path) -> None:
    continuous_metrics = metrics(continuous)
    rows = []
    for data in schedules:
        item = metrics(data)
        rows.append(
            {
                "gap_seed": item["gap_seed"],
                "n_gaps": item["n_gaps"],
                "gated_hours": item["gated_hours"],
                "gap_row_fraction": item["gap_row_fraction"],
                "max_rhat": item["sampler"]["max_rhat"],
                "min_ess": item["sampler"]["min_ess"],
                "passes_all_sampler_gates": bool(
                    item["sampler"]["divergences"] == 0
                    and item["sampler"]["max_rhat"] <= 1.05
                    and item["sampler"]["min_ess"] >= 50
                    and item["sampler"]["tree_depth_saturation"] <= 0.05
                    and item["sampler"]["min_ebfmi"] >= 0.3
                ),
                "scores": item["scores"],
            }
        )
    summary = {
        "analysis": "ESA-orbit X2 surface total PSD",
        "frequency_contract": continuous_metrics["requested_frequency_hz"],
        "realized_frequency_hz": continuous_metrics["realized_frequency_hz"],
        "null_notch_fraction_full": continuous_metrics["response_mask_fraction_full"],
        "null_notch_fraction_high": continuous_metrics["response_mask_fraction_high"],
        "continuous": continuous_metrics,
        "heldout_whitening": whitening_scores(continuous, schedules[1]["clean_power_psd"]),
        "gap_schedules": rows,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "esa_m0_results_f48")
    parser.add_argument("--representative-seed", type=int, default=2)
    args = parser.parse_args()
    paper_style()
    continuous = load(args.results / "esa_x2_m0_continuous.npz")
    schedules = [load(args.results / f"esa_x2_m0_gapped_seed{seed}.npz") for seed in range(1, 6)]
    representative = schedules[args.representative_seed - 1]
    figure_continuous(continuous, args.results / "esa_x2_part_a_continuous")
    figure_gaps(representative, schedules, args.results / "esa_x2_part_b_gaps")
    figure_whitening(continuous, representative["clean_power_psd"], args.results / "esa_x2_whitening_check")
    write_summary(continuous, schedules, args.results / "esa_x2_m0_summary.json")
    print(f"[saved] figures and summary in {args.results}")


if __name__ == "__main__":
    main()
