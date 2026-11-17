"""Polarimetric calibration helpers for the direct HDR experiment.

The detector-channel response and the polarization-dependent response of the
recombining beamsplitter are different transformations.  The former multiplies
the four recorded analyzer intensities, whereas the latter acts coherently on
the two-component field before the analyzer.  This module keeps the two steps
separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STOKES_CHANNELS = ("I0", "I90", "I45", "I135")


@dataclass(frozen=True)
class StokesClosureFit:
    """Relative inverse gains inferred from linear-polarizer closure."""

    inverse_gains: np.ndarray
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class BeamsplitterRatioFit:
    """Independent estimates of the relative output-field coefficient."""

    complex_ratio_from_cross_term: complex
    amplitude_from_orthogonal_intensity: float
    cross_relative_rms: float
    orthogonal_relative_rms: float


def stokes_covariance(intensities: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return the covariance of ``I0 + I90 - I45 - I135`` samples.

    ``intensities`` must end in the channel order given by
    :data:`STOKES_CHANNELS`.  A null eigenvector of the returned matrix gives
    relative inverse detector gains without imposing a polarization state on
    the measured field.
    """

    values = np.asarray(intensities, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 4:
        raise ValueError(f"Expected intensities[...,4], got {values.shape}")
    samples = values.reshape(-1, 4)
    signs = np.asarray((1.0, 1.0, -1.0, -1.0))
    design = samples * signs
    valid = np.all(np.isfinite(design), axis=1)
    if weights is None:
        sample_weights = np.ones(samples.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if sample_weights.size != samples.shape[0]:
            raise ValueError("weights do not match intensities")
        valid &= np.isfinite(sample_weights) & (sample_weights > 0.0)
    if np.count_nonzero(valid) < 4:
        raise ValueError("Too few valid samples for Stokes-closure calibration")
    selected = design[valid]
    selected_weights = sample_weights[valid]
    return (selected.T * selected_weights) @ selected / float(np.sum(selected_weights))


def fit_stokes_closure(covariance: np.ndarray, *, reference_index: int = 0) -> StokesClosureFit:
    """Fit positive inverse gains from a Stokes-closure covariance matrix."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 covariance matrix, got {matrix.shape}")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    gains = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    if gains[int(reference_index)] < 0.0:
        gains = -gains
    if np.any(gains <= 0.0):
        raise ValueError("Stokes-closure null vector is not positive")
    gains /= gains[int(reference_index)]
    return StokesClosureFit(inverse_gains=gains, eigenvalues=eigenvalues)


def closure_relative_rms(intensities: np.ndarray, inverse_gains: np.ndarray) -> float:
    """Return RMS Stokes-closure error relative to the mean total intensity."""

    values = np.asarray(intensities, dtype=np.float64)
    gains = np.asarray(inverse_gains, dtype=np.float64)
    if values.shape[-1] != 4 or gains.shape != (4,):
        raise ValueError("Expected intensities[...,4] and four inverse gains")
    corrected = values * gains
    residual = corrected[..., 0] + corrected[..., 1] - corrected[..., 2] - corrected[..., 3]
    total = 0.5 * np.sum(corrected, axis=-1)
    valid = np.isfinite(residual) & np.isfinite(total)
    return float(
        np.sqrt(np.mean(residual[valid] ** 2))
        / max(np.sqrt(np.mean(total[valid] ** 2)), np.finfo(float).eps)
    )


def radial_output_coherency(gram: np.ndarray, input_angle_deg: float) -> np.ndarray:
    """Contract a vectorized-Jones Gram matrix with one linear input state."""

    values = np.asarray(gram)
    if values.shape[-2:] != (4, 4):
        raise ValueError(f"Expected gram[...,4,4], got {values.shape}")
    angle = np.radians(float(input_angle_deg))
    incident = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.complex128)
    rows = np.asarray(
        (
            (incident[0], incident[1], 0.0, 0.0),
            (0.0, 0.0, incident[0], incident[1]),
        ),
        dtype=np.complex128,
    )
    return np.einsum(
        "ai,...ij,bj->...ab",
        rows,
        values,
        np.conjugate(rows),
        optimize=True,
    )


def post_beamsplitter_intensities(
    coherency: np.ndarray,
    output_ratio: complex,
    *,
    normalization: float = 1.0,
) -> np.ndarray:
    """Evaluate four linear analyzers after ``diag(1, output_ratio)``."""

    matrix = np.asarray(coherency)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected coherency[...,2,2], got {matrix.shape}")
    denominator = float(normalization)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("normalization must be finite and positive")
    ratio = complex(output_ratio)
    xx = np.real(matrix[..., 0, 0]) / denominator
    yy = abs(ratio) ** 2 * np.real(matrix[..., 1, 1]) / denominator
    cross = np.real(np.conjugate(ratio) * matrix[..., 0, 1]) / denominator
    return np.stack(
        (
            xx,
            yy,
            0.5 * (xx + yy) + cross,
            0.5 * (xx + yy) - cross,
        ),
        axis=-1,
    )


def fit_beamsplitter_ratio(
    coherency: np.ndarray,
    intensities: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    normalization: float = 1.0,
) -> BeamsplitterRatioFit:
    """Estimate the relative field response from orthogonal and diagonal data.

    The orthogonal channel determines ``abs(q)`` through
    ``I90 = abs(q)**2 Wyy``.  The diagonal difference determines real and
    imaginary parts independently through
    ``(I45-I135)/2 = Re(conj(q) Wxy)``.
    """

    matrix = np.asarray(coherency)
    observed = np.asarray(intensities, dtype=np.float64)
    if matrix.shape[-2:] != (2, 2) or observed.shape != matrix.shape[:-2] + (4,):
        raise ValueError("coherency and intensity shapes do not match")
    denominator = float(normalization)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("normalization must be finite and positive")
    if weights is None:
        sample_weights = np.ones(matrix.shape[:-2], dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != matrix.shape[:-2]:
            raise ValueError("weights do not match coherency samples")

    yy_model = np.real(matrix[..., 1, 1]) / denominator
    yy_observed = observed[..., 1]
    cross_model = matrix[..., 0, 1] / denominator
    cross_observed = 0.5 * (observed[..., 2] - observed[..., 3])
    valid = (
        np.isfinite(yy_model)
        & np.isfinite(yy_observed)
        & np.isfinite(cross_model)
        & np.isfinite(cross_observed)
        & np.isfinite(sample_weights)
        & (sample_weights > 0.0)
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError("Too few valid samples for beamsplitter calibration")
    weight = sample_weights[valid]
    yy_x = yy_model[valid]
    yy_y = yy_observed[valid]
    magnitude_squared = float(
        np.sum(weight * yy_x * yy_y)
        / max(np.sum(weight * yy_x * yy_x), np.finfo(float).eps)
    )
    amplitude = float(np.sqrt(max(magnitude_squared, 0.0)))

    cross = cross_model[valid]
    design = np.column_stack((np.real(cross), np.imag(cross)))
    root_weight = np.sqrt(weight)
    parameters, *_unused = np.linalg.lstsq(
        design * root_weight[:, None],
        cross_observed[valid] * root_weight,
        rcond=None,
    )
    ratio = complex(float(parameters[0]), float(parameters[1]))
    cross_prediction = design @ parameters
    yy_prediction = magnitude_squared * yy_x
    cross_rms = float(
        np.sqrt(np.sum(weight * (cross_observed[valid] - cross_prediction) ** 2))
        / max(np.sqrt(np.sum(weight * cross_observed[valid] ** 2)), np.finfo(float).eps)
    )
    yy_rms = float(
        np.sqrt(np.sum(weight * (yy_y - yy_prediction) ** 2))
        / max(np.sqrt(np.sum(weight * yy_y**2)), np.finfo(float).eps)
    )
    return BeamsplitterRatioFit(
        complex_ratio_from_cross_term=ratio,
        amplitude_from_orthogonal_intensity=amplitude,
        cross_relative_rms=cross_rms,
        orthogonal_relative_rms=yy_rms,
    )
