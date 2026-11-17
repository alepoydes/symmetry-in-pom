"""Axisymmetric director profiles used by optical control calculations."""

from __future__ import annotations

import numpy as np


def layer_helicity(field: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """Return the nematic in-plane helicity of every director layer."""

    director = np.asarray(field, dtype=np.float64)
    if director.ndim != 4 or director.shape[-1] != 3:
        raise ValueError("field must have shape (x, y, z, 3)")
    x, y = np.indices(director.shape[:2], dtype=np.float64)
    phi = np.arctan2(y - float(center[1]), x - float(center[0]))
    transverse = director[..., 0] + 1j * director[..., 1]
    moment = np.sum(transverse**2 * np.exp(-2j * phi)[..., None], axis=(0, 1))
    return 0.5 * np.unwrap(np.angle(moment))


def radial_polar_profile(
    field: np.ndarray,
    center: tuple[float, float],
    *,
    bin_width_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radial nematic polar angle and coherence for every layer."""

    director = np.asarray(field, dtype=np.float64)
    if director.ndim != 4 or director.shape[-1] != 3:
        raise ValueError("field must have shape (x, y, z, 3)")
    if not np.isfinite(float(bin_width_px)) or float(bin_width_px) <= 0.0:
        raise ValueError("bin_width_px must be positive and finite")

    x, y = np.indices(director.shape[:2], dtype=np.float64)
    radius = np.hypot(x - float(center[0]), y - float(center[1]))
    phi = np.arctan2(y - float(center[1]), x - float(center[0]))
    tau = layer_helicity(director, center)
    transverse = director[..., 0] + 1j * director[..., 1]
    local_axis = np.exp(1j * (phi[..., None] + tau[None, None, :]))
    n_parallel = np.real(transverse * np.conj(local_axis))
    polar_nematic = (director[..., 2] + 1j * n_parallel) ** 2

    bin_index = np.floor(radius / float(bin_width_px)).astype(np.int32)
    n_bins = int(bin_index.max()) + 1
    flat_bin = bin_index.ravel()
    counts = np.bincount(flat_bin, minlength=n_bins).astype(np.float64)
    theta = np.empty((n_bins, director.shape[2]), dtype=np.float64)
    coherence = np.empty_like(theta)
    for z_index in range(director.shape[2]):
        values = polar_nematic[..., z_index].ravel()
        real = np.bincount(flat_bin, weights=values.real, minlength=n_bins)
        imag = np.bincount(flat_bin, weights=values.imag, minlength=n_bins)
        mean = (real + 1j * imag) / np.maximum(counts, 1.0)
        doubled_angle = np.unwrap(np.angle(mean)[::-1])[::-1]
        theta[:, z_index] = 0.5 * doubled_angle
        coherence[:, z_index] = np.abs(mean)

    radius_px = (np.arange(n_bins, dtype=np.float64) + 0.5) * float(bin_width_px)
    populated = counts > 0.0
    return radius_px[populated], theta[populated], coherence[populated]


def scaled_helicity_profile(measured_tau: np.ndarray, scale: float) -> np.ndarray:
    """Scale longitudinal helicity about its midplane value."""

    measured = np.asarray(measured_tau, dtype=np.float64)
    if measured.ndim != 1 or measured.size < 1:
        raise ValueError("measured_tau must be a nonempty one-dimensional array")
    if not np.isfinite(float(scale)) or not 0.0 <= float(scale) <= 1.0:
        raise ValueError(f"helicity scale must lie in [0, 1], got {scale!r}")
    midplane = float(measured[measured.size // 2])
    return midplane + float(scale) * (measured - midplane)
