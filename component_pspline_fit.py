"""Parametric-Galactic / free-P-spline-noise PSD decomposition.

This is intentionally a *constrained* component model.  Given only a local
PSD estimate, two arbitrary positive time--frequency surfaces cannot be
separated: the likelihood observes their sum.  The Galactic component is
therefore a known orbit-and-sky response template, with only an amplitude and
an RCL-knee correction free.  The instrumental component is a regularised
tensor-product log-P-spline. A single positive reference level may weakly
centre its overall scale, but no analytic time-frequency noise surface is used
by the fit.

The implementation uses the gamma/Whittle likelihood for locally averaged
periodograms.  ``effective_dof`` should account for frequency averaging only;
overlapping time segments remain correlated, so this is a MAP estimator and
not yet a calibrated posterior interval calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import OptimizeResult, minimize
from scipy.special import log_expit


@dataclass(frozen=True)
class ComponentPSplineFit:
    """MAP decomposition and the quantities required for diagnostics."""

    noise_psd: np.ndarray
    galactic_psd: np.ndarray
    total_psd: np.ndarray
    log_galactic_amplitude: float
    f_knee_hz: float
    spline_coefficients: np.ndarray
    time_basis: np.ndarray
    frequency_basis: np.ndarray
    objective: float
    optimizer: OptimizeResult


def _unit_interval(values: np.ndarray, *, log: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("coordinate must be a finite one-dimensional array with at least two entries")
    coordinate = np.log(values) if log else values
    span = coordinate[-1] - coordinate[0]
    if not span > 0.0 or np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    return (coordinate - coordinate[0]) / span


def _bspline_basis(coordinate: np.ndarray, n_interior_knots: int, degree: int = 3) -> np.ndarray:
    if n_interior_knots < 0:
        raise ValueError("n_interior_knots must be non-negative")
    if coordinate.size < degree + 1:
        raise ValueError("coordinate grid is too short for the requested spline degree")
    interior = np.linspace(0.0, 1.0, n_interior_knots + 2)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    return BSpline.design_matrix(coordinate, knots, degree, extrapolate=False).toarray()


def rcl_knee_ratio(
    frequency_hz: np.ndarray,
    f_knee_hz: float,
    reference_f_knee_hz: float,
    *,
    gamma: float = 1680.0,
) -> np.ndarray:
    """RCL knee correction relative to the supplied Galactic template.

    The map/orbit response and all non-knee spectral factors cancel.  This
    makes the supplied template the reference model rather than treating an
    archive's injected Galactic surface as data to be fitted.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(frequency <= 0.0) or f_knee_hz <= 0.0 or reference_f_knee_hz <= 0.0:
        raise ValueError("frequency and knee frequencies must be positive")
    log_ratio = log_expit(2.0 * gamma * (f_knee_hz - frequency)) - log_expit(
        2.0 * gamma * (reference_f_knee_hz - frequency)
    )
    return np.exp(log_ratio)


def _second_difference_penalty(coefficients: np.ndarray, axis: int) -> float:
    if coefficients.shape[axis] < 3:
        return 0.0
    return float(np.sum(np.diff(coefficients, n=2, axis=axis) ** 2))


def fit_component_pspline(
    observed_psd: np.ndarray,
    time_tcb: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    noise_prior_level_psd: float,
    galactic_template_psd: np.ndarray,
    reference_f_knee_hz: float = 2.15e-3,
    effective_dof: float | np.ndarray = 16.0,
    mask: np.ndarray | None = None,
    n_time_knots: int = 5,
    n_frequency_knots: int = 12,
    smoothing_time: float = 30.0,
    smoothing_frequency: float = 30.0,
    noise_level_log_sd: float = 5.0,
    initial_log_galactic_amplitude: float = 0.0,
    initial_f_knee_hz: float | None = None,
    knee_bounds_hz: tuple[float, float] = (5.0e-4, 8.0e-3),
    maxiter: int = 1000,
) -> ComponentPSplineFit:
    """Fit a response-informed Galactic component and a noise P-spline.

    All PSD arrays have shape ``(n_time, n_frequency)`` and common physical
    units.  ``galactic_template_psd`` must be computed from the known sky map
    and LISA response at ``reference_f_knee_hz``; it is not estimated from the
    observed total PSD. ``noise_prior_level_psd`` is a single scalar that
    weakly centres the geometric level of the otherwise free noise spline; it
    does not supply a time- or frequency-dependent noise shape. Consequently
    narrow transfer minima must be learned by the spline from the data.
    """
    observed = np.asarray(observed_psd, dtype=float)
    noise_prior_level = np.asarray(noise_prior_level_psd, dtype=float)
    galactic_template = np.asarray(galactic_template_psd, dtype=float)
    if observed.ndim != 2:
        raise ValueError("observed_psd must have shape (time, frequency)")
    if galactic_template.shape != observed.shape:
        raise ValueError("observed and Galactic PSD surfaces must share shape")
    if noise_prior_level.shape != ():
        raise ValueError("noise_prior_level_psd must be a positive scalar")
    if np.any(~np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValueError("observed_psd must be finite and strictly positive")
    if not np.isfinite(noise_prior_level) or noise_prior_level <= 0.0:
        raise ValueError("noise_prior_level_psd must be a finite positive scalar")
    if np.any(~np.isfinite(galactic_template)) or np.any(galactic_template < 0.0):
        raise ValueError("galactic_template_psd must be finite and non-negative")
    if smoothing_time < 0.0 or smoothing_frequency < 0.0:
        raise ValueError("smoothing penalties must be non-negative")
    if noise_level_log_sd <= 0.0:
        raise ValueError("noise_level_log_sd must be positive")
    if not (0.0 < knee_bounds_hz[0] < knee_bounds_hz[1]):
        raise ValueError("knee_bounds_hz must be positive and ordered")

    time_coordinate = _unit_interval(time_tcb)
    frequency_coordinate = _unit_interval(frequency_hz, log=True)
    time_basis = _bspline_basis(time_coordinate, n_time_knots)
    frequency_basis = _bspline_basis(frequency_coordinate, n_frequency_knots)
    n_time_coeff, n_frequency_coeff = time_basis.shape[1], frequency_basis.shape[1]
    if mask is None:
        retained = np.ones_like(observed, dtype=bool)
    else:
        retained = np.asarray(mask, dtype=bool)
        if retained.shape != observed.shape:
            raise ValueError("mask must have the same shape as observed_psd")
    if retained.sum() <= n_time_coeff * n_frequency_coeff:
        raise ValueError("too few retained cells for the requested spline basis")
    dof = np.broadcast_to(np.asarray(effective_dof, dtype=float), observed.shape)
    if np.any(~np.isfinite(dof)) or np.any(dof <= 0.0):
        raise ValueError("effective_dof must be finite and positive")
    if initial_f_knee_hz is None:
        initial_f_knee_hz = reference_f_knee_hz
    if not knee_bounds_hz[0] <= initial_f_knee_hz <= knee_bounds_hz[1]:
        raise ValueError("initial_f_knee_hz lies outside knee_bounds_hz")

    # Physical TDI PSDs are around 1e-16 in this archive.  Work in relative
    # units so L-BFGS sees well-scaled gradients; the dimensionless Whittle
    # ratios are unchanged and the returned component PSDs are restored below.
    data_scale = float(np.median(observed[retained]))
    observed = observed / data_scale
    noise_prior_level = float(noise_prior_level / data_scale)
    galactic_template = galactic_template / data_scale

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float, float]:
        coefficient_count = n_time_coeff * n_frequency_coeff
        theta = parameters[:coefficient_count].reshape(n_time_coeff, n_frequency_coeff)
        return theta, float(parameters[coefficient_count]), float(np.exp(parameters[coefficient_count + 1]))

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        theta, log_amplitude, f_knee = unpack(parameters)
        log_noise_correction = time_basis @ theta @ frequency_basis.T
        noise = noise_prior_level * np.exp(log_noise_correction)
        ratio = rcl_knee_ratio(frequency_hz, f_knee, reference_f_knee_hz)
        galactic = np.exp(log_amplitude) * galactic_template * ratio[None, :]
        total = noise + galactic
        # This algebraic form avoids squaring an exploratory large PSD during
        # optimisation, while retaining the exact Whittle derivative.
        derivative = np.where(retained, 0.5 * dof * (1.0 - observed / total) / total, 0.0)
        likelihood = 0.5 * np.sum(dof[retained] * (np.log(total[retained]) + observed[retained] / total[retained]))
        penalty_time = smoothing_time * _second_difference_penalty(theta, axis=0)
        penalty_frequency = smoothing_frequency * _second_difference_penalty(theta, axis=1)
        mean_log_correction = float(np.mean(log_noise_correction[retained]))
        penalty_level = 0.5 * (mean_log_correction / noise_level_log_sd) ** 2
        gradient_log_noise = derivative * noise
        gradient_theta = time_basis.T @ gradient_log_noise @ frequency_basis
        mean_surface_weight = retained.astype(float) / np.sum(retained)
        gradient_theta += (
            mean_log_correction
            / noise_level_log_sd**2
            * (time_basis.T @ mean_surface_weight @ frequency_basis)
        )
        # Apply the D2^T D2 roughness gradients with natural finite-difference
        # boundaries.  The dimensions here are small compared with the PSD
        # grid, while writing the adjoint explicitly keeps the optimizer fast.
        for axis, smoothing in ((0, smoothing_time), (1, smoothing_frequency)):
            if smoothing == 0.0:
                continue
            size = theta.shape[axis]
            if size < 3:
                continue
            d2 = np.diff(theta, n=2, axis=axis)
            adjoint = np.zeros_like(theta)
            if axis == 0:
                adjoint[:-2] += d2
                adjoint[1:-1] -= 2.0 * d2
                adjoint[2:] += d2
            else:
                adjoint[:, :-2] += d2
                adjoint[:, 1:-1] -= 2.0 * d2
                adjoint[:, 2:] += d2
            gradient_theta += 2.0 * smoothing * adjoint
        gradient_amplitude = float(np.sum((derivative * galactic)[retained]))
        knee_factor = f_knee * 1680.0 * (
            1.0 - np.tanh(1680.0 * (f_knee - frequency_hz))
        )
        gradient_log_knee = float(np.sum((derivative * galactic * knee_factor[None, :])[retained]))
        return likelihood + penalty_time + penalty_frequency + penalty_level, np.r_[
            gradient_theta.ravel(), gradient_amplitude, gradient_log_knee
        ]

    initial = np.zeros(n_time_coeff * n_frequency_coeff + 2)
    initial[-2:] = (initial_log_galactic_amplitude, np.log(initial_f_knee_hz))
    # Bounding individual spline coefficients prevents an exploratory L-BFGS
    # step from overflowing
    # ``exp(B_t theta B_f^T)`` without constraining any plausible PSD fit.
    bounds = [(-20.0, 20.0)] * (initial.size - 2) + [
        (-8.0, 8.0),
        (np.log(knee_bounds_hz[0]), np.log(knee_bounds_hz[1])),
    ]
    optimizer = minimize(
        objective_and_gradient, initial, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1.0e-11, "gtol": 1.0e-7},
    )
    if not optimizer.success:
        raise RuntimeError(f"component fit did not converge: {optimizer.message}")
    theta, log_amplitude, f_knee = unpack(optimizer.x)
    noise = data_scale * noise_prior_level * np.exp(time_basis @ theta @ frequency_basis.T)
    galactic = data_scale * np.exp(log_amplitude) * galactic_template * rcl_knee_ratio(
        frequency_hz, f_knee, reference_f_knee_hz
    )[None, :]
    return ComponentPSplineFit(
        noise_psd=noise,
        galactic_psd=galactic,
        total_psd=noise + galactic,
        log_galactic_amplitude=log_amplitude,
        f_knee_hz=f_knee,
        spline_coefficients=theta,
        time_basis=time_basis,
        frequency_basis=frequency_basis,
        objective=float(optimizer.fun),
        optimizer=optimizer,
    )


__all__ = ["ComponentPSplineFit", "fit_component_pspline", "rcl_knee_ratio"]
