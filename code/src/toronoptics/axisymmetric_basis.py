"""Covariant Jones basis and transformations between matrices and coefficients."""

from __future__ import annotations

import numpy as np


def axisymmetric_basis(phi: np.ndarray) -> np.ndarray:
    """Return I, K, Sigma_1, Sigma_2 for every polar angle."""

    angle = np.asarray(phi, dtype=np.float64)
    c2 = np.cos(2.0 * angle)
    s2 = np.sin(2.0 * angle)
    result = np.empty(angle.shape + (4, 2, 2), dtype=np.float64)
    result[..., 0, :, :] = ((1.0, 0.0), (0.0, 1.0))
    result[..., 1, :, :] = ((0.0, -1.0), (1.0, 0.0))
    result[..., 2, 0, 0] = c2
    result[..., 2, 0, 1] = s2
    result[..., 2, 1, 0] = s2
    result[..., 2, 1, 1] = -c2
    result[..., 3, 0, 0] = -s2
    result[..., 3, 0, 1] = c2
    result[..., 3, 1, 0] = c2
    result[..., 3, 1, 1] = s2
    return result


def jones_from_coefficients(
    coefficients: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.complex128)
    basis = axisymmetric_basis(phi)
    if values.shape != basis.shape[:-2]:
        raise ValueError(
            "coefficients must have shape phi.shape + (4,), got "
            f"{values.shape!r} for phi shape {np.asarray(phi).shape!r}"
        )
    return np.einsum("...k,...kij->...ij", values, basis, optimize=True)


def coefficients_from_jones(
    jones: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Invert the axisymmetric Jones decomposition at known polar angles."""

    matrix = np.asarray(jones, dtype=np.complex128)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("jones must end in a 2 by 2 matrix")
    angle = np.broadcast_to(np.asarray(phi, dtype=np.float64), matrix.shape[:-2])
    c2 = np.cos(2.0 * angle)
    s2 = np.sin(2.0 * angle)
    a = 0.5 * (matrix[..., 0, 0] + matrix[..., 1, 1])
    b = 0.5 * (matrix[..., 1, 0] - matrix[..., 0, 1])
    radial = 0.5 * (matrix[..., 0, 0] - matrix[..., 1, 1])
    symmetric = 0.5 * (matrix[..., 0, 1] + matrix[..., 1, 0])
    c = c2 * radial + s2 * symmetric
    d = -s2 * radial + c2 * symmetric
    return np.stack((a, b, c, d), axis=-1)
