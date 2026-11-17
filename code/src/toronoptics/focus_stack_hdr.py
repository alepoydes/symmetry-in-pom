"""Radiometric calibration and HDR assembly for focus-stack polarimetry.

The functions in this module operate on linear camera intensities.  Dark-frame
correction, exposure merging, and the measured reference field remain separate
operations: a coherent optical background is not treated as an additive noise
image or divided out implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)


FOCUS_HDR_SCHEMA_VERSION = "toronoptics.focus-hdr.v2"
FOCUS_HDR_CHANNELS = ("I0", "I135", "I45", "I90")
RAW_TO_PROJECT_SOURCE_CHANNEL = {
    "I0": "I0",
    "I135": "I45",
    "I45": "I135",
    "I90": "I90",
}


def fill_small_display_holes(
    intensity_yx: np.ndarray,
    confidence_yx: np.ndarray,
    *,
    maximum_distance_px: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Continue narrow invalid gaps in a display copy of an HDR image.

    The input intensity and confidence arrays are never modified.  Invalid
    regions farther than ``maximum_distance_px`` from measured data remain
    invalid, so this helper cannot conceal a large missing region.
    """

    values = np.asarray(intensity_yx, dtype=np.float64)
    confidence = np.asarray(confidence_yx, dtype=np.float64)
    if values.shape != confidence.shape or values.ndim != 2:
        raise ValueError(
            f"Intensity and confidence must be matching 2-D arrays, got "
            f"{values.shape} and {confidence.shape}"
        )
    valid = np.isfinite(values) & np.isfinite(confidence) & (confidence > 0.0)
    invalid_count = int(np.count_nonzero(~valid))
    if invalid_count == 0:
        return values.copy(), {
            "invalid_pixels": 0,
            "display_filled_pixels": 0,
            "maximum_filled_distance_px": 0.0,
        }
    if not np.any(valid):
        raise ValueError("The displayed HDR crop contains no valid pixels")

    distance, nearest = distance_transform_edt(
        ~valid,
        return_distances=True,
        return_indices=True,
    )
    fillable = (~valid) & (distance <= float(maximum_distance_px))
    nearest_filled = values.copy()
    nearest_filled[~valid] = values[
        tuple(index[~valid] for index in nearest)
    ]
    smoothed = gaussian_filter(nearest_filled, sigma=1.0, mode="nearest")
    displayed = values.copy()
    displayed[fillable] = smoothed[fillable]
    filled_count = int(np.count_nonzero(fillable))
    return displayed, {
        "invalid_pixels": invalid_count,
        "display_filled_pixels": filled_count,
        "maximum_filled_distance_px": (
            float(np.max(distance[fillable])) if filled_count else 0.0
        ),
    }


def raw_camera_to_project(array_yxc: np.ndarray) -> np.ndarray:
    """Reflect a registered raw-camera array and relabel analyzer channels.

    Reflection changes a linear-analyzer angle from ``alpha`` to ``pi-alpha``;
    the diagonal channels must therefore be exchanged together with the image
    reflection.  The input and output channel order is :data:`FOCUS_HDR_CHANNELS`.
    """

    values = np.asarray(array_yxc)
    if values.ndim != 3 or values.shape[-1] != len(FOCUS_HDR_CHANNELS):
        raise ValueError(
            f"Expected array[y,x,{len(FOCUS_HDR_CHANNELS)}], got {values.shape}"
        )
    source_indices = np.asarray(
        [
            FOCUS_HDR_CHANNELS.index(RAW_TO_PROJECT_SOURCE_CHANNEL[name])
            for name in FOCUS_HDR_CHANNELS
        ],
        dtype=np.int64,
    )
    return np.flip(values[..., source_indices], axis=1)


def raw_camera_center_to_project(
    center_xy: tuple[float, float] | np.ndarray,
    *,
    width: int,
) -> np.ndarray:
    """Transform a raw-camera center to the reflected project coordinates."""

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if width < 1:
        raise ValueError("width must be positive")
    return np.asarray((width - 1.0 - center[0], center[1]), dtype=np.float64)


@dataclass(frozen=True)
class DarkFrameCalibration:
    """Denoised dark response and pixels rejected as impulsive defects."""

    dark_yx: np.ndarray
    hot_mask_yx: np.ndarray
    residual_sigma: float


@dataclass(frozen=True)
class RankOneReference:
    """Robust common spatial mode of several positive reference images."""

    mode_yx: np.ndarray
    amplitudes: np.ndarray
    relative_residuals: np.ndarray
    singular_energy_fraction: np.ndarray
    iterations: int


@dataclass(frozen=True)
class LinearScaleFit:
    """Robust through-origin scale between two registered exposures."""

    scale: float
    relative_residual: float
    correlation: float
    used_pixels: int


@dataclass(frozen=True)
class HDRMerge:
    """Merged intensity and per-pixel provenance of the two exposures."""

    intensity_yx: np.ndarray
    confidence_yx: np.ndarray
    low_exposure_fraction_yx: np.ndarray
    valid_yx: np.ndarray


def robust_sigma(values: np.ndarray) -> float:
    """Return the Gaussian-equivalent median absolute deviation."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - center)))


def calibrate_dark_frame(
    image_yx: np.ndarray,
    *,
    full_scale: float = 1.0,
    smooth_sigma_px: float = 8.0,
    hot_threshold_sigma: float = 8.0,
) -> DarkFrameCalibration:
    """Denoise one dark frame without imprinting its shot noise on the data."""

    image = np.asarray(image_yx, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a two-dimensional dark frame, got {image.shape}")
    if full_scale <= 0.0:
        raise ValueError("full_scale must be positive")
    if smooth_sigma_px <= 0.0:
        raise ValueError("smooth_sigma_px must be positive")

    local_median = median_filter(image, size=3, mode="reflect")
    residual = image - local_median
    sigma = robust_sigma(residual)
    quantization_floor = 1.5 / float(full_scale)
    threshold = max(float(hot_threshold_sigma) * sigma, quantization_floor)
    hot = np.abs(residual) > threshold
    repaired = np.where(hot, local_median, image)
    smooth = gaussian_filter(repaired, sigma=float(smooth_sigma_px), mode="reflect")
    return DarkFrameCalibration(
        dark_yx=np.asarray(smooth, dtype=np.float32),
        hot_mask_yx=np.asarray(hot, dtype=bool),
        residual_sigma=float(sigma),
    )


def robust_image_level(image_yx: np.ndarray) -> float:
    """Estimate a positive image level while rejecting sparse structures."""

    values = np.asarray(image_yx, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate a level from an empty image")
    lower, upper = np.quantile(values, (0.05, 0.95))
    central = values[(values >= lower) & (values <= upper)]
    level = float(np.mean(central))
    if not np.isfinite(level) or level <= 0.0:
        raise ValueError(f"Reference image has a non-positive robust level: {level}")
    return level


def robust_rank_one_reference(
    images_yx: Sequence[np.ndarray] | np.ndarray,
    *,
    maximum_iterations: int = 8,
    huber_sigma: float = 3.5,
    singular_downsample: int = 4,
) -> RankOneReference:
    """Fit ``image_i = amplitude_i * mode`` with pixelwise Huber weights.

    The singular spectrum is calculated without centering: centering would
    remove the common illumination mode that this fit is intended to recover.
    Full-resolution robust alternating least squares then suppresses structures
    present only in individual reference frames.
    """

    stack = np.asarray(images_yx, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"Expected at least two 2D images, got {stack.shape}")
    if not np.all(np.isfinite(stack)):
        raise ValueError("Reference images must be finite")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    initial_levels = np.asarray(
        [robust_image_level(image) for image in stack],
        dtype=np.float64,
    )
    normalized = stack / initial_levels[:, None, None].astype(np.float32)

    step = max(int(singular_downsample), 1)
    sampled = normalized[:, ::step, ::step].reshape(stack.shape[0], -1).astype(np.float64)
    _u, singular_values, _vt = np.linalg.svd(sampled, full_matrices=False)
    singular_energy = singular_values * singular_values
    singular_energy /= max(float(np.sum(singular_energy)), np.finfo(float).tiny)

    mode = np.mean(normalized, axis=0, dtype=np.float64)
    mode /= robust_image_level(mode)
    coefficients = np.ones(stack.shape[0], dtype=np.float64)
    relative_residuals = np.empty(stack.shape[0], dtype=np.float64)

    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        previous = mode.copy()

        for index, image in enumerate(normalized):
            residual = image - coefficients[index] * mode
            sigma = max(robust_sigma(residual[::2, ::2]), 1.0e-7)
            limit = float(huber_sigma) * sigma
            weight = np.minimum(1.0, limit / np.maximum(np.abs(residual), limit))
            numerator = np.sum(weight * mode * image, dtype=np.float64)
            denominator = np.sum(weight * mode * mode, dtype=np.float64)
            coefficients[index] = numerator / max(denominator, np.finfo(float).tiny)

        numerator = np.zeros(mode.shape, dtype=np.float64)
        denominator = np.zeros(mode.shape, dtype=np.float64)
        for coefficient, image in zip(coefficients, normalized, strict=True):
            residual = image - coefficient * mode
            sigma = max(robust_sigma(residual[::2, ::2]), 1.0e-7)
            limit = float(huber_sigma) * sigma
            weight = np.minimum(1.0, limit / np.maximum(np.abs(residual), limit))
            numerator += weight * coefficient * image
            denominator += weight * coefficient * coefficient
        mode = numerator / np.maximum(denominator, np.finfo(float).tiny)
        mode_level = robust_image_level(mode)
        mode /= mode_level
        coefficients *= mode_level

        relative_change = np.sqrt(np.mean((mode - previous) ** 2))
        if relative_change < 2.0e-6:
            break

    amplitudes = initial_levels * coefficients
    mode_norm = np.sqrt(np.mean(mode * mode))
    for index, (amplitude, image) in enumerate(zip(amplitudes, stack, strict=True)):
        residual = image - amplitude * mode
        relative_residuals[index] = np.sqrt(np.mean(residual * residual)) / max(
            abs(float(amplitude)) * mode_norm,
            np.finfo(float).tiny,
        )

    return RankOneReference(
        mode_yx=np.asarray(mode, dtype=np.float32),
        amplitudes=np.asarray(amplitudes, dtype=np.float64),
        relative_residuals=relative_residuals,
        singular_energy_fraction=np.asarray(singular_energy, dtype=np.float64),
        iterations=iterations,
    )


def fit_linear_scale(
    low_yx: np.ndarray,
    high_yx: np.ndarray,
    *,
    valid_yx: np.ndarray | None = None,
    minimum_low: float = 0.0,
    maximum_high: float = np.inf,
    sample_step: int = 2,
    maximum_iterations: int = 8,
) -> LinearScaleFit:
    """Robustly fit ``high = scale * low`` on registered linear images."""

    low = np.asarray(low_yx, dtype=np.float64)
    high = np.asarray(high_yx, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 2:
        raise ValueError(f"Exposure shapes differ: {low.shape} and {high.shape}")
    mask = np.isfinite(low) & np.isfinite(high)
    if valid_yx is not None:
        mask &= np.asarray(valid_yx, dtype=bool)
    mask &= low > float(minimum_low)
    mask &= high < float(maximum_high)
    step = max(int(sample_step), 1)
    low_values = low[::step, ::step][mask[::step, ::step]]
    high_values = high[::step, ::step][mask[::step, ::step]]
    if low_values.size < 128:
        raise ValueError(f"Too few pixels for exposure-scale fit: {low_values.size}")

    scale = float(np.median(high_values / low_values))
    for _iteration in range(maximum_iterations):
        residual = high_values - scale * low_values
        sigma = max(robust_sigma(residual), np.finfo(float).eps)
        limit = 3.5 * sigma
        weight = np.minimum(1.0, limit / np.maximum(np.abs(residual), limit))
        updated = float(
            np.sum(weight * low_values * high_values)
            / max(np.sum(weight * low_values * low_values), np.finfo(float).tiny)
        )
        if abs(updated - scale) <= 1.0e-8 * max(abs(scale), 1.0):
            scale = updated
            break
        scale = updated

    residual = high_values - scale * low_values
    denominator = np.sqrt(np.mean(high_values * high_values))
    relative_residual = float(np.sqrt(np.mean(residual * residual)) / max(denominator, 1.0e-15))
    correlation = float(np.corrcoef(low_values, high_values)[0, 1])
    return LinearScaleFit(
        scale=scale,
        relative_residual=relative_residual,
        correlation=correlation,
        used_pixels=int(low_values.size),
    )


def saturation_weight(
    raw_yx: np.ndarray,
    *,
    soft_threshold: float,
    hard_threshold: float,
) -> np.ndarray:
    """Return a smooth reliability taper approaching detector saturation."""

    if not 0.0 <= soft_threshold < hard_threshold <= 1.0:
        raise ValueError("Require 0 <= soft_threshold < hard_threshold <= 1")
    raw = np.asarray(raw_yx, dtype=np.float32)
    coordinate = np.clip(
        (float(hard_threshold) - raw) / (float(hard_threshold) - float(soft_threshold)),
        0.0,
        1.0,
    )
    return coordinate * coordinate * (3.0 - 2.0 * coordinate)


def merge_hdr_exposures(
    low_corrected_yx: np.ndarray,
    high_corrected_yx: np.ndarray,
    *,
    low_raw_yx: np.ndarray,
    high_raw_yx: np.ndarray,
    low_to_high_scale: float,
    high_soft_threshold: float,
    high_hard_threshold: float = 0.995,
    low_soft_threshold: float = 0.94,
    low_hard_threshold: float = 0.995,
) -> HDRMerge:
    """Merge registered exposures in high-exposure detector units."""

    low = np.asarray(low_corrected_yx, dtype=np.float32)
    high = np.asarray(high_corrected_yx, dtype=np.float32)
    low_raw = np.asarray(low_raw_yx, dtype=np.float32)
    high_raw = np.asarray(high_raw_yx, dtype=np.float32)
    if low.shape != high.shape or low.shape != low_raw.shape or low.shape != high_raw.shape:
        raise ValueError("All HDR input arrays must have the same shape")
    if low.ndim != 2:
        raise ValueError(f"Expected two-dimensional HDR inputs, got {low.shape}")
    if low_to_high_scale <= 0.0:
        raise ValueError("low_to_high_scale must be positive")

    finite_low = np.isfinite(low) & np.isfinite(low_raw)
    finite_high = np.isfinite(high) & np.isfinite(high_raw)
    low_reliability = saturation_weight(
        np.nan_to_num(low_raw, nan=1.0),
        soft_threshold=low_soft_threshold,
        hard_threshold=low_hard_threshold,
    )
    high_reliability = saturation_weight(
        np.nan_to_num(high_raw, nan=1.0),
        soft_threshold=high_soft_threshold,
        hard_threshold=high_hard_threshold,
    )
    low_reliability *= finite_low
    high_reliability *= finite_high

    # The two acquisitions are registered but not simultaneous.  Preserve the
    # higher-SNR frame exactly throughout its linear range and introduce the
    # short exposure only through the measured saturation taper.  This avoids
    # blurring coherent rings by continuously averaging slightly different
    # realizations of the scene.
    high_weight = high_reliability
    low_weight = (1.0 - high_reliability) * low_reliability
    low_weight = np.where(finite_high, low_weight, low_reliability)
    total_weight = low_weight + high_weight
    valid = total_weight > 0.0
    low_scaled = float(low_to_high_scale) * low
    low_value = np.where(finite_low, low_scaled, 0.0)
    high_value = np.where(finite_high, high, 0.0)
    intensity = np.full(low.shape, np.nan, dtype=np.float32)
    intensity[valid] = (
        low_weight[valid] * low_value[valid]
        + high_weight[valid] * high_value[valid]
    ) / total_weight[valid]

    low_fraction = np.zeros(low.shape, dtype=np.float32)
    low_fraction[valid] = low_weight[valid] / total_weight[valid]
    confidence = np.maximum(
        high_reliability,
        low_reliability / max(float(low_to_high_scale), 1.0),
    )
    return HDRMerge(
        intensity_yx=intensity,
        confidence_yx=np.asarray(confidence, dtype=np.float32),
        low_exposure_fraction_yx=low_fraction,
        valid_yx=np.asarray(valid, dtype=bool),
    )
