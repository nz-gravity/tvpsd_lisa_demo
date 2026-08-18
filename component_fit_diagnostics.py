"""Matched diagnostics for surface-total-spline and H_para component fits."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def masked_frequency_bin_mean(
    values: np.ndarray,
    retained_mask: np.ndarray,
    frequency_hz: np.ndarray,
    bin_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a surface over the retained cells in contiguous frequency bins.

    Returns ``(mean, counts, mean_frequency)``. Empty time/bin combinations are
    represented by NaN with count zero, so they can be excluded consistently
    from both the Whittle power and log-variance terms.
    """
    values = np.asarray(values, dtype=float)
    retained = np.asarray(retained_mask, dtype=bool)
    frequency = np.asarray(frequency_hz, dtype=float)
    starts = np.asarray(bin_starts, dtype=int)
    if values.ndim != 2 or retained.shape != values.shape:
        raise ValueError("values and retained_mask must share shape (time, frequency)")
    if frequency.shape != (values.shape[1],):
        raise ValueError("frequency_hz does not match the surface frequency axis")
    if starts.ndim != 1 or starts.size == 0 or starts[0] != 0:
        raise ValueError("bin_starts must be one-dimensional and begin at zero")
    if np.any(np.diff(starts) <= 0) or starts[-1] >= frequency.size:
        raise ValueError("bin_starts must be strictly increasing valid indices")

    stops = np.r_[starts[1:], frequency.size]
    means = np.full((values.shape[0], starts.size), np.nan)
    counts = np.zeros_like(means)
    mean_frequency = np.empty(starts.size)
    for index, (start, stop) in enumerate(zip(starts, stops)):
        selected = retained[:, start:stop]
        counts[:, index] = np.sum(selected, axis=1)
        numerator = np.sum(np.where(selected, values[:, start:stop], 0.0), axis=1)
        valid = counts[:, index] > 0.0
        means[valid, index] = numerator[valid] / counts[valid, index]
        mean_frequency[index] = np.mean(frequency[start:stop])
    return means, counts, mean_frequency


def component_recovery_metrics(
    observed_total: np.ndarray,
    counts: np.ndarray,
    truth_noise: np.ndarray,
    truth_galactic: np.ndarray,
    surface_total: np.ndarray,
    surface_lower: np.ndarray,
    surface_upper: np.ndarray,
    h_para_noise: np.ndarray,
    h_para_galactic: np.ndarray,
    h_para_total: np.ndarray | None = None,
    whitening_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate fit-domain recovery and continuum-whitening metrics.

    ``counts`` defines the cells used by the likelihood. ``whitening_mask`` is
    diagnostic-only: it can omit narrow response-null neighborhoods without
    removing those cells from the fit or from surface-recovery summaries.
    """
    arrays = (
        observed_total, counts, truth_noise, truth_galactic, surface_total,
        surface_lower, surface_upper, h_para_noise, h_para_galactic,
    )
    if any(np.asarray(array).shape != np.asarray(observed_total).shape for array in arrays):
        raise ValueError("all metric surfaces must have the same shape")
    observed = np.asarray(observed_total, dtype=float)
    weight = np.asarray(counts, dtype=float)
    truth_noise = np.asarray(truth_noise, dtype=float)
    truth_galactic = np.asarray(truth_galactic, dtype=float)
    truth_total = truth_noise + truth_galactic
    h_para_total = (
        np.asarray(h_para_noise) + np.asarray(h_para_galactic)
        if h_para_total is None else np.asarray(h_para_total, dtype=float)
    )
    if h_para_total.shape != observed.shape:
        raise ValueError("h_para_total must have the same shape as observed_total")
    fit_valid = (
        (weight > 0.0) & np.isfinite(observed) & (observed > 0.0)
        & np.isfinite(truth_total) & (truth_total > 0.0)
    )
    if not np.any(fit_valid):
        raise ValueError("no valid fitted cells for component diagnostics")
    if whitening_mask is None:
        whitening_valid = fit_valid
    else:
        whitening = np.asarray(whitening_mask, dtype=bool)
        if whitening.shape != observed.shape:
            raise ValueError("whitening_mask must match observed_total")
        whitening_valid = fit_valid & whitening
    if not np.any(whitening_valid):
        raise ValueError("no valid cells for continuum-whitening diagnostics")

    def median_absolute_log_ratio(estimate: np.ndarray, truth: np.ndarray, selection: np.ndarray) -> float:
        return float(np.median(np.abs(np.log(estimate[selection] / truth[selection]))))

    noise_visible = fit_valid & (truth_noise / truth_total >= 0.03)
    galactic_visible = fit_valid & (truth_galactic / truth_total >= 0.03)
    whitening_weight = np.sum(weight[whitening_valid])
    return {
        "surface_total_median_abs_log_error": median_absolute_log_ratio(surface_total, truth_total, fit_valid),
        "h_para_total_median_abs_log_error": median_absolute_log_ratio(h_para_total, truth_total, fit_valid),
        "h_para_noise_median_abs_log_error_visible": median_absolute_log_ratio(h_para_noise, truth_noise, noise_visible),
        "h_para_galactic_median_abs_log_error_visible": median_absolute_log_ratio(
            h_para_galactic, truth_galactic, galactic_visible
        ),
        "surface_continuum_mean_z2": float(
            np.sum(weight[whitening_valid] * observed[whitening_valid] / surface_total[whitening_valid])
            / whitening_weight
        ),
        "h_para_continuum_mean_z2": float(
            np.sum(weight[whitening_valid] * observed[whitening_valid] / h_para_total[whitening_valid])
            / whitening_weight
        ),
        "surface_pointwise_90_coverage": float(np.mean(
            (truth_total[fit_valid] >= surface_lower[fit_valid])
            & (truth_total[fit_valid] <= surface_upper[fit_valid])
        )),
        "fit_effective_cells": float(np.sum(weight[fit_valid])),
        "whitening_effective_cells": float(whitening_weight),
    }


def plot_component_model_comparison(
    output_path: str | Path,
    *,
    time_days: np.ndarray,
    frequency_hz: np.ndarray,
    observed_total: np.ndarray,
    counts: np.ndarray,
    truth_noise: np.ndarray,
    truth_galactic: np.ndarray,
    surface_total: np.ndarray,
    surface_lower: np.ndarray,
    surface_upper: np.ndarray,
    h_para_noise: np.ndarray,
    h_para_galactic: np.ndarray,
    galactic_amplitude: float,
    f_knee_hz: float,
    h_para_noise_lower: np.ndarray | None = None,
    h_para_noise_upper: np.ndarray | None = None,
    h_para_galactic_lower: np.ndarray | None = None,
    h_para_galactic_upper: np.ndarray | None = None,
    h_para_total_lower: np.ndarray | None = None,
    h_para_total_upper: np.ndarray | None = None,
    h_para_diagnostics: dict[str, float | int] | None = None,
    h_para_total_estimate: np.ndarray | None = None,
    whitening_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Save a six-panel comparison with claims matched to available uncertainty."""
    output_path = Path(output_path)
    truth_total = truth_noise + truth_galactic
    h_para_total = (
        h_para_noise + h_para_galactic
        if h_para_total_estimate is None else np.asarray(h_para_total_estimate, dtype=float)
    )
    h_para_interval_arrays = (
        h_para_noise_lower, h_para_noise_upper, h_para_galactic_lower,
        h_para_galactic_upper, h_para_total_lower, h_para_total_upper,
    )
    has_m1_posterior = all(array is not None for array in h_para_interval_arrays)
    valid = counts > 0.0
    metrics = component_recovery_metrics(
        observed_total, counts, truth_noise, truth_galactic,
        surface_total, surface_lower, surface_upper, h_para_noise, h_para_galactic, h_para_total,
        whitening_mask,
    )
    if has_m1_posterior:
        metrics["h_para_pointwise_90_coverage"] = float(np.mean(
            (truth_total[valid] >= h_para_total_lower[valid])
            & (truth_total[valid] <= h_para_total_upper[valid])
        ))

    def time_median(values: np.ndarray) -> np.ndarray:
        return np.nanmedian(np.where(valid, values, np.nan), axis=0)

    def shade_whitening_omissions(axis: plt.Axes) -> None:
        if whitening_mask is None:
            return
        omitted = np.any(valid & ~np.asarray(whitening_mask, dtype=bool), axis=0)
        for position in np.flatnonzero(omitted):
            left = frequency_hz[position - 1] if position else frequency_hz[position]
            right = (
                frequency_hz[position + 1]
                if position + 1 < frequency_hz.size
                else frequency_hz[position]
            )
            axis.axvspan(
                np.sqrt(left * frequency_hz[position]),
                np.sqrt(frequency_hz[position] * right),
                color="0.5",
                alpha=0.08,
                lw=0,
            )

    fig = plt.figure(figsize=(14, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    total_axis = fig.add_subplot(grid[0, 0])
    component_axis = fig.add_subplot(grid[0, 1])
    text_axis = fig.add_subplot(grid[0, 2])
    surface_axis = fig.add_subplot(grid[1, 0])
    h_para_axis = fig.add_subplot(grid[1, 1], sharex=surface_axis, sharey=surface_axis)
    residual_axis = fig.add_subplot(grid[1, 2])

    total_axis.loglog(frequency_hz, time_median(truth_total), color="k", label="truth total")
    total_axis.loglog(frequency_hz, time_median(surface_total), color="C0", label="surface posterior mean")
    total_axis.fill_between(
        frequency_hz, time_median(surface_lower), time_median(surface_upper),
        color="C0", alpha=0.2, label="surface pointwise 90% band",
    )
    h_para_total_label = "H_para posterior median" if has_m1_posterior else "H_para MAP total"
    total_axis.loglog(frequency_hz, time_median(h_para_total), color="C3", label=h_para_total_label)
    if has_m1_posterior:
        total_axis.fill_between(
            frequency_hz, time_median(h_para_total_lower), time_median(h_para_total_upper),
            color="C3", alpha=0.18, label="H_para pointwise 90% band",
        )
    total_axis.set(xlabel="frequency [Hz]", ylabel=r"PSD [Hz$^2$/Hz]", title="Total-spectrum recovery")
    shade_whitening_omissions(total_axis)
    total_axis.legend(fontsize=7)

    component_axis.loglog(frequency_hz, time_median(truth_noise), "--", color="C0", label="noise truth")
    component_axis.loglog(
        frequency_hz, time_median(h_para_noise), color="C0",
        label="H_para noise posterior median" if has_m1_posterior else "H_para noise MAP",
    )
    component_axis.loglog(frequency_hz, time_median(truth_galactic), "--", color="C1", label="Galactic truth")
    component_axis.loglog(
        frequency_hz, time_median(h_para_galactic), color="C1",
        label="H_para Galactic posterior median" if has_m1_posterior else "H_para Galactic MAP",
    )
    if has_m1_posterior:
        component_axis.fill_between(
            frequency_hz, time_median(h_para_noise_lower), time_median(h_para_noise_upper),
            color="C0", alpha=0.16,
        )
        component_axis.fill_between(
            frequency_hz, time_median(h_para_galactic_lower), time_median(h_para_galactic_upper),
            color="C1", alpha=0.16,
        )
    component_axis.set(xlabel="frequency [Hz]", ylabel=r"PSD [Hz$^2$/Hz]", title="H_para component recovery")
    shade_whitening_omissions(component_axis)
    component_axis.legend(fontsize=7)

    text_axis.axis("off")
    lines = [
        "Matched fit-domain diagnostics",
        f"A_gal = {galactic_amplitude:.4f}",
        f"f_knee = {1e3 * f_knee_hz:.4f} mHz",
        f"surface total median |log error| = {metrics['surface_total_median_abs_log_error']:.3f}",
        f"H_para total median |log error| = {metrics['h_para_total_median_abs_log_error']:.3f}",
        f"H_para noise error (>=3% share) = {metrics['h_para_noise_median_abs_log_error_visible']:.3f}",
        f"H_para Galactic error (>=3% share) = {metrics['h_para_galactic_median_abs_log_error_visible']:.3f}",
        (
            "continuum mean z^2: "
            f"surface={metrics['surface_continuum_mean_z2']:.3f}, "
            f"H_para={metrics['h_para_continuum_mean_z2']:.3f}"
        ),
        f"surface pointwise 90% coverage = {metrics['surface_pointwise_90_coverage']:.1%}",
        *(
            [f"H_para pointwise 90% coverage = {metrics['h_para_pointwise_90_coverage']:.1%}"]
            if has_m1_posterior else []
        ),
        (
            f"H_para NUTS: div={int(h_para_diagnostics['divergences'])}, "
            f"max R-hat={h_para_diagnostics['max_r_hat']:.3f}"
            if has_m1_posterior and h_para_diagnostics is not None
            else "H_para is a conditional MAP fit; no interval is shown."
        ),
    ]
    text_axis.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)

    errors = (
        np.where(valid, np.log10(surface_total / truth_total), np.nan),
        np.where(valid, np.log10(h_para_total / truth_total), np.nan),
    )
    finite = np.concatenate([error[np.isfinite(error)] for error in errors])
    limit = max(0.05, float(np.quantile(np.abs(finite), 0.99)))
    for axis, error, title in (
        (surface_axis, errors[0], "surface log10(total / truth)"),
        (h_para_axis, errors[1], "H_para log10(total / truth)"),
    ):
        image = axis.pcolormesh(
            time_days, frequency_hz, error.T, shading="auto", cmap="coolwarm",
            vmin=-limit, vmax=limit,
        )
        axis.set(xlabel="time [days]", title=title, yscale="log")
        fig.colorbar(image, ax=axis, pad=0.01, label="dex")
    surface_axis.set_ylabel("frequency [Hz]")

    diagnostic_valid = valid if whitening_mask is None else valid & np.asarray(whitening_mask, dtype=bool)
    diagnostic_counts = np.where(diagnostic_valid, counts, 0.0)
    diagnostic_denominator = np.sum(diagnostic_counts, axis=1)
    weighted_m0_z2 = np.nansum(
        diagnostic_counts * observed_total / surface_total, axis=1
    ) / diagnostic_denominator
    weighted_m1_z2 = np.nansum(
        diagnostic_counts * observed_total / h_para_total, axis=1
    ) / diagnostic_denominator
    residual_axis.plot(time_days, weighted_m0_z2, marker="o", ms=3, label="surface")
    residual_axis.plot(time_days, weighted_m1_z2, marker="o", ms=3, label="H_para")
    residual_axis.axhline(1.0, color="k", ls="--", lw=1)
    residual_axis.set(
        xlabel="time [days]", ylabel=r"continuum mean $z^2$",
        title="Whitening by time (dip neighborhoods omitted)",
    )
    residual_axis.legend(fontsize=8)

    comparison_name = "H_para response-informed component posterior" if has_m1_posterior else "H_para response-informed component MAP"
    fig.suptitle(f"X2: surface total P-spline posterior versus {comparison_name}", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_m1_parameter_posterior(
    output_path: str | Path,
    *,
    amplitude_draws: np.ndarray,
    f_knee_draws_hz: np.ndarray,
    phi_time_draws: np.ndarray,
    phi_frequency_draws: np.ndarray,
    injected_amplitude: float | None = None,
    injected_f_knee_hz: float | None = None,
) -> None:
    """Save compact H_para marginal, joint, and smoothing-precision diagnostics."""
    output_path = Path(output_path)
    amplitude = np.asarray(amplitude_draws, dtype=float).ravel()
    knee_mhz = 1.0e3 * np.asarray(f_knee_draws_hz, dtype=float).ravel()
    phi_time = np.asarray(phi_time_draws, dtype=float).ravel()
    phi_frequency = np.asarray(phi_frequency_draws, dtype=float).ravel()
    if not (
        amplitude.size == knee_mhz.size == phi_time.size == phi_frequency.size
        and amplitude.size > 1
    ):
        raise ValueError("all H_para parameter-draw arrays must have the same non-trivial size")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    axes[0, 0].scatter(amplitude, knee_mhz, s=7, alpha=0.22, edgecolors="none")
    if injected_amplitude is not None:
        axes[0, 0].axvline(injected_amplitude, color="k", ls="--", lw=1)
    if injected_f_knee_hz is not None:
        axes[0, 0].axhline(1.0e3 * injected_f_knee_hz, color="k", ls="--", lw=1)
    axes[0, 0].set(
        xlabel=r"$A_{\rm gal}$", ylabel=r"$f_{\rm knee}$ [mHz]",
        title="Joint Galactic posterior",
    )

    axes[0, 1].hist(amplitude, bins=35, density=True, color="C1", alpha=0.75)
    if injected_amplitude is not None:
        axes[0, 1].axvline(injected_amplitude, color="k", ls="--", label="injected")
    amplitude_interval = np.quantile(amplitude, (0.05, 0.5, 0.95))
    axes[0, 1].set(
        xlabel=r"$A_{\rm gal}$", ylabel="posterior density",
        title=f"Amplitude: {amplitude_interval[1]:.4f} [{amplitude_interval[0]:.4f}, {amplitude_interval[2]:.4f}]",
    )
    if injected_amplitude is not None:
        axes[0, 1].legend(fontsize=8)

    axes[1, 0].hist(knee_mhz, bins=35, density=True, color="C2", alpha=0.75)
    if injected_f_knee_hz is not None:
        axes[1, 0].axvline(1.0e3 * injected_f_knee_hz, color="k", ls="--", label="injected")
    knee_interval = np.quantile(knee_mhz, (0.05, 0.5, 0.95))
    axes[1, 0].set(
        xlabel=r"$f_{\rm knee}$ [mHz]", ylabel="posterior density",
        title=f"Knee: {knee_interval[1]:.4f} [{knee_interval[0]:.4f}, {knee_interval[2]:.4f}] mHz",
    )
    if injected_f_knee_hz is not None:
        axes[1, 0].legend(fontsize=8)

    axes[1, 1].hist(np.log10(phi_time), bins=35, density=True, alpha=0.65, label=r"$\phi_t$")
    axes[1, 1].hist(
        np.log10(phi_frequency), bins=35, density=True, alpha=0.65, label=r"$\phi_f$"
    )
    axes[1, 1].set(
        xlabel=r"$\log_{10}$ smoothing precision", ylabel="posterior density",
        title="Noise-residual spline smoothness",
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("X2 H_para NUTS parameter posterior", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "component_recovery_metrics",
    "masked_frequency_bin_mean",
    "plot_component_model_comparison",
    "plot_m1_parameter_posterior",
]
