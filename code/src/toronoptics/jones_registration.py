from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy.signal.windows import tukey


@dataclass(frozen=True)
class JonesRegistrationFeature:
    spectrum: np.ndarray
    origin_px: tuple[int, int]
    rough_center_px: tuple[float, float]


@dataclass(frozen=True)
class JonesRegistrationResult:
    centers_px: np.ndarray
    corrections_px: np.ndarray
    pairwise_center_difference_px: np.ndarray
    pairwise_shift_px: np.ndarray
    pairwise_peak: np.ndarray
    pairwise_peak_ratio: np.ndarray
    pairwise_residual_px: np.ndarray
    triangle_closure_px: np.ndarray


def make_registration_feature(
    jones: np.ndarray,
    *,
    rough_center_px: tuple[float, float],
    crop_size_px: int,
    border_fraction: float = 0.06,
) -> JonesRegistrationFeature:
    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected Jones field with shape (x,y,2,2), got {matrix.shape!r}")
    size = int(crop_size_px)
    if size < 16:
        raise ValueError("crop_size_px must be at least 16")
    if not 0.0 < float(border_fraction) < 0.5:
        raise ValueError("border_fraction must lie between 0 and 0.5")

    center_index = 0.5 * (size - 1)
    origin = np.rint(np.asarray(rough_center_px) - center_index).astype(np.int64)
    end = origin + size
    if np.any(origin < 0) or end[0] > matrix.shape[0] or end[1] > matrix.shape[1]:
        raise ValueError("Registration crop extends outside the Jones field")
    crop = np.asarray(
        matrix[origin[0] : end[0], origin[1] : end[1]], dtype=np.complex64
    ).copy()
    border = max(2, int(round(size * float(border_fraction))))
    background_mask = np.zeros((size, size), dtype=bool)
    background_mask[:border] = True
    background_mask[-border:] = True
    background_mask[:, :border] = True
    background_mask[:, -border:] = True
    for row in range(2):
        for column in range(2):
            values = crop[..., row, column][background_mask]
            background = np.median(values.real) + 1j * np.median(values.imag)
            crop[..., row, column] -= background

    window_1d = tukey(size, alpha=0.2, sym=True).astype(np.float32, copy=False)
    crop *= (window_1d[:, None] * window_1d[None, :])[..., None, None]
    channels = np.moveaxis(crop.reshape(size, size, 4), -1, 0)
    spectrum = fft.fftn(channels, axes=(-2, -1), workers=-1)
    return JonesRegistrationFeature(
        spectrum=np.asarray(spectrum, dtype=np.complex64),
        origin_px=(int(origin[0]), int(origin[1])),
        rough_center_px=(float(rough_center_px[0]), float(rough_center_px[1])),
    )


def register_jones_features(
    features: tuple[JonesRegistrationFeature, ...],
    *,
    anchor_centers_px: np.ndarray,
    upsample_factor: int = 20,
) -> JonesRegistrationResult:
    if len(features) < 2:
        raise ValueError("At least two registration features are required")
    anchors = np.asarray(anchor_centers_px, dtype=np.float64)
    if anchors.shape != (len(features), 2):
        raise ValueError(
            f"anchor_centers_px must have shape {(len(features), 2)}, got {anchors.shape}"
        )
    spectral_shape = features[0].spectrum.shape
    if any(item.spectrum.shape != spectral_shape for item in features):
        raise ValueError("All registration spectra must have the same shape")
    if int(upsample_factor) < 1:
        raise ValueError("upsample_factor must be positive")

    count = len(features)
    pair_shift = np.full((count, count, 2), np.nan, dtype=np.float64)
    pair_difference = np.full_like(pair_shift, np.nan)
    pair_peak = np.full((count, count), np.nan, dtype=np.float64)
    pair_ratio = np.full((count, count), np.nan, dtype=np.float64)
    equations: list[tuple[int, int, np.ndarray]] = []
    for reference in range(count):
        pair_shift[reference, reference] = 0.0
        pair_difference[reference, reference] = 0.0
        pair_peak[reference, reference] = 1.0
        pair_ratio[reference, reference] = np.inf
        for moving in range(reference + 1, count):
            shift, peak, peak_ratio = phase_cross_correlation_multichannel(
                features[reference].spectrum,
                features[moving].spectrum,
                upsample_factor=int(upsample_factor),
            )
            origin_reference = np.asarray(features[reference].origin_px, dtype=np.float64)
            origin_moving = np.asarray(features[moving].origin_px, dtype=np.float64)
            difference = origin_moving - origin_reference - shift
            pair_shift[reference, moving] = shift
            pair_shift[moving, reference] = -shift
            pair_difference[reference, moving] = difference
            pair_difference[moving, reference] = -difference
            pair_peak[reference, moving] = pair_peak[moving, reference] = peak
            pair_ratio[reference, moving] = pair_ratio[moving, reference] = peak_ratio
            equations.append((reference, moving, difference))

    corrections = _solve_registration_graph(equations, anchors)
    centers = anchors + corrections
    residual = np.full((count, count), np.nan, dtype=np.float64)
    np.fill_diagonal(residual, 0.0)
    for reference, moving, difference in equations:
        value = np.linalg.norm((centers[moving] - centers[reference]) - difference)
        residual[reference, moving] = residual[moving, reference] = value
    closure = np.asarray(
        [
            np.linalg.norm(
                pair_difference[i, j]
                + pair_difference[j, k]
                - pair_difference[i, k]
            )
            for i in range(count)
            for j in range(i + 1, count)
            for k in range(j + 1, count)
        ],
        dtype=np.float64,
    )
    return JonesRegistrationResult(
        centers_px=centers,
        corrections_px=corrections,
        pairwise_center_difference_px=pair_difference,
        pairwise_shift_px=pair_shift,
        pairwise_peak=pair_peak,
        pairwise_peak_ratio=pair_ratio,
        pairwise_residual_px=residual,
        triangle_closure_px=closure,
    )


def phase_cross_correlation_multichannel(
    reference_spectrum: np.ndarray,
    moving_spectrum: np.ndarray,
    *,
    upsample_factor: int = 20,
) -> tuple[np.ndarray, float, float]:
    reference = np.asarray(reference_spectrum)
    moving = np.asarray(moving_spectrum)
    if reference.shape != moving.shape or reference.ndim != 3:
        raise ValueError("Spectra must have matching shape (channel,x,y)")
    cross = np.sum(reference * np.conj(moving), axis=0, dtype=np.complex128)
    scale = np.sqrt(
        float(np.sum(np.abs(reference) ** 2, dtype=np.float64))
        * float(np.sum(np.abs(moving) ** 2, dtype=np.float64))
    )
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
        raise ValueError("Registration feature has zero or non-finite spectral power")
    normalized = cross / scale
    normalized[(0,) * normalized.ndim] = 0.0

    correlation = fft.ifftn(normalized, workers=-1)
    magnitude = np.abs(correlation)
    maximum = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    shift = np.asarray(maximum, dtype=np.float64)
    midpoint = np.asarray([np.trunc(axis_size / 2) for axis_size in magnitude.shape])
    shape = np.asarray(magnitude.shape)
    shift[shift > midpoint] -= shape[shift > midpoint]
    peak = float(magnitude[maximum])
    peak_ratio = peak / max(_second_peak(magnitude, maximum), np.finfo(float).tiny)

    factor = int(upsample_factor)
    if factor > 1:
        shift = np.round(shift * factor) / factor
        region_size = int(np.ceil(1.5 * factor))
        dft_shift = np.trunc(region_size / 2.0)
        sample_offset = dft_shift - shift * factor
        refined = _upsampled_dft(
            np.conj(normalized),
            upsampled_region_size=(region_size, region_size),
            upsample_factor=factor,
            axis_offsets=sample_offset,
        ).conj()
        refined_magnitude = np.abs(refined)
        refined_maximum = np.unravel_index(
            int(np.argmax(refined_magnitude)), refined_magnitude.shape
        )
        shift += (np.asarray(refined_maximum, dtype=np.float64) - dft_shift) / factor
        peak = float(refined_magnitude[refined_maximum] / normalized.size)
    return shift, peak, float(peak_ratio)


def _upsampled_dft(
    data: np.ndarray,
    *,
    upsampled_region_size: tuple[int, int],
    upsample_factor: int,
    axis_offsets: np.ndarray,
) -> np.ndarray:
    result = np.asarray(data)
    offsets = np.asarray(axis_offsets, dtype=np.float64)
    for axis_size, region_size, offset in zip(
        result.shape[::-1], upsampled_region_size[::-1], offsets[::-1], strict=True
    ):
        kernel = (
            np.arange(region_size, dtype=np.float64)[:, None] - float(offset)
        ) * fft.fftfreq(axis_size, d=float(upsample_factor))[None, :]
        result = np.tensordot(np.exp(-2j * np.pi * kernel), result, axes=(1, -1))
    return result


def _second_peak(magnitude: np.ndarray, maximum: tuple[int, ...]) -> float:
    masked = np.asarray(magnitude).copy()
    radius = 2
    coordinates = np.ogrid[tuple(slice(0, size) for size in masked.shape)]
    distance = np.zeros(masked.shape, dtype=np.float64)
    for coordinate, center, size in zip(coordinates, maximum, masked.shape, strict=True):
        delta = np.minimum(np.abs(coordinate - center), size - np.abs(coordinate - center))
        distance += delta**2
    masked[distance <= radius**2] = 0.0
    return float(np.max(masked))


def _solve_registration_graph(
    equations: list[tuple[int, int, np.ndarray]],
    anchors: np.ndarray,
) -> np.ndarray:
    count = anchors.shape[0]
    design = np.zeros((len(equations) + 1, count), dtype=np.float64)
    target = np.zeros((len(equations) + 1, 2), dtype=np.float64)
    for row, (reference, moving, difference) in enumerate(equations):
        design[row, reference] = -1.0
        design[row, moving] = 1.0
        target[row] = difference - (anchors[moving] - anchors[reference])
    design[-1] = 1.0 / np.sqrt(count)
    weights = np.ones(len(equations), dtype=np.float64)
    solution = np.zeros_like(anchors)
    for _iteration in range(4):
        weighted_design = design.copy()
        weighted_target = target.copy()
        root_weight = np.sqrt(weights)
        weighted_design[:-1] *= root_weight[:, None]
        weighted_target[:-1] *= root_weight[:, None]
        solution = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
        vector_residual = design[:-1] @ solution - target[:-1]
        norm = np.linalg.norm(vector_residual, axis=1)
        scale = 1.4826 * np.median(np.abs(norm - np.median(norm)))
        threshold = max(0.05, 2.5 * scale)
        weights = np.minimum(1.0, threshold / np.maximum(norm, np.finfo(float).tiny))
    return solution
