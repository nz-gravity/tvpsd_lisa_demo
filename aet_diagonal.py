"""Diagonal A/E/T helpers for the controlled independent-XYZ archive.

The archive currently stores XYZ time series and diagonal XYZ PSD surfaces.
The orthonormal rotation below is applied to the samples.  A diagonal PSD is
rotated under the archive's explicit zero-XYZ-cross-spectrum simulation
contract, i.e. ``diag(M diag(S_xyz) M.T)``.  This is appropriate for the toy
archive but is not a replacement for a physical full-covariance response.
"""

from __future__ import annotations

import numpy as np


AET_CHANNELS = ("A", "E", "T")
XYZ_TO_AET = np.asarray(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ],
    dtype=float,
)


def xyz_to_aet_series(xyz: np.ndarray) -> np.ndarray:
    """Rotate an array with leading XYZ channel axis into orthonormal A/E/T."""
    values = np.asarray(xyz, dtype=float)
    if values.ndim < 2 or values.shape[0] != 3:
        raise ValueError("xyz must have shape (3, ...)")
    return np.einsum("cx,x...->c...", XYZ_TO_AET, values, optimize=True)


def diagonal_xyz_psd_to_aet(xyz_psd: np.ndarray) -> np.ndarray:
    """Rotate diagonal XYZ PSDs to diagonal A/E/T PSDs.

    This calculation assumes zero XYZ cross spectra.  It is exactly the
    covariance contract used by the current independently drawn XYZ Galactic
    realization, but it is only an approximation for a physical LISA
    stochastic response or correlated instrumental noise.
    """
    values = np.asarray(xyz_psd, dtype=float)
    if values.ndim < 2 or values.shape[0] != 3:
        raise ValueError("xyz_psd must have shape (3, ...)")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("xyz_psd must be finite and non-negative")
    return np.einsum("cx,x...->c...", XYZ_TO_AET**2, values, optimize=True)


def xyz_covariance_to_aet_diagonal(xyz_covariance: np.ndarray) -> np.ndarray:
    """Return diagonal A/E/T PSDs from a full trailing-axis XYZ covariance.

    ``xyz_covariance`` has shape ``(..., 3, 3)`` and may be complex Hermitian.
    Only ``diag(M C M.T)`` is returned; callers need not estimate or retain AET
    cross spectra to use the correct analytic AET auto-PSDs.
    """
    covariance = np.asarray(xyz_covariance)
    if covariance.ndim < 2 or covariance.shape[-2:] != (3, 3):
        raise ValueError("xyz_covariance must have shape (..., 3, 3)")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("xyz_covariance must be finite")
    aet_covariance = np.einsum(
        "ai,...ij,bj->...ab",
        XYZ_TO_AET,
        covariance,
        XYZ_TO_AET,
        optimize=True,
    )
    diagonal = np.real(np.diagonal(aet_covariance, axis1=-2, axis2=-1))
    if np.any(diagonal < 0.0):
        raise ValueError("rotated AET covariance has a negative diagonal")
    return diagonal


__all__ = [
    "AET_CHANNELS",
    "XYZ_TO_AET",
    "diagonal_xyz_psd_to_aet",
    "xyz_covariance_to_aet_diagonal",
    "xyz_to_aet_series",
]
