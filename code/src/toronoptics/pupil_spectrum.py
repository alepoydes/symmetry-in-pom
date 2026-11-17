from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter, windows


@dataclass(frozen=True)
class JonesPupilSpectrum:
    frequency_um_inv: np.ndarray
    power: np.ndarray
    log_power: np.ndarray
    log_slope: np.ndarray
    wavelength_um: float

    @property
    def numerical_aperture(self) -> np.ndarray:
        return self.frequency_um_inv * float(self.wavelength_um)


@dataclass(frozen=True)
class PupilCutoffEstimate:
    frequency_um_inv: float
    effective_na: float
    slope_decades_per_um_inv: float
    half_width_na: float


def radial_jones_pupil_spectrum(
    jones: np.ndarray,
    *,
    pixel_size_um: float,
    wavelength_um: float,
    radial_bin_um_inv: float = 0.005,
    tukey_fraction: float = 0.2,
    background_radius_fraction: float = 0.44,
    smooth_width_um_inv: float = 0.15,
) -> JonesPupilSpectrum:
    """Return the radially averaged spectrum of the background-subtracted Jones field."""
    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected Jones shape (x, y, 2, 2), got {matrix.shape!r}")
    for name, value in (
        ("pixel_size_um", pixel_size_um),
        ("wavelength_um", wavelength_um),
        ("radial_bin_um_inv", radial_bin_um_inv),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    nx, ny = matrix.shape[:2]
    x, y = np.indices((nx, ny), dtype=np.float64)
    center = (0.5 * (nx - 1), 0.5 * (ny - 1))
    radius = np.hypot(x - center[0], y - center[1])
    background_mask = radius >= float(background_radius_fraction) * min(nx, ny)
    if not np.any(background_mask):
        raise ValueError("Background annulus contains no pixels")
    background = np.mean(matrix[background_mask], axis=0)
    perturbation = matrix - background

    window = windows.tukey(nx, alpha=float(tukey_fraction))[:, None]
    window = window * windows.tukey(ny, alpha=float(tukey_fraction))[None, :]
    fx = np.fft.fftfreq(nx, d=float(pixel_size_um))
    fy = np.fft.fftfreq(ny, d=float(pixel_size_um))
    frequency_map = np.hypot(fx[:, None], fy[None, :])
    radial_index = np.floor(frequency_map / float(radial_bin_um_inv)).astype(np.int32)
    bin_count = int(np.max(radial_index)) + 1
    flat_index = radial_index.ravel()
    count = np.bincount(flat_index, minlength=bin_count)
    power = np.zeros(bin_count, dtype=np.float64)
    for output_index, input_index in ((0, 0), (0, 1), (1, 0), (1, 1)):
        spectrum = np.fft.fft2(perturbation[..., output_index, input_index] * window)
        power += np.bincount(
            flat_index,
            weights=np.abs(spectrum).ravel() ** 2,
            minlength=bin_count,
        )
    power /= np.maximum(count, 1)
    frequency = (np.arange(bin_count, dtype=np.float64) + 0.5) * float(
        radial_bin_um_inv
    )
    floor = max(float(np.max(power)) * 1.0e-16, np.finfo(float).tiny)
    raw_log_power = np.log10(np.maximum(power, floor))
    smooth_bins = _odd_window_length(smooth_width_um_inv, radial_bin_um_inv, bin_count)
    log_power = savgol_filter(raw_log_power, smooth_bins, 3)
    log_slope = savgol_filter(
        raw_log_power,
        smooth_bins,
        3,
        deriv=1,
        delta=float(radial_bin_um_inv),
    )
    return JonesPupilSpectrum(
        frequency_um_inv=frequency,
        power=power,
        log_power=log_power,
        log_slope=log_slope,
        wavelength_um=float(wavelength_um),
    )


def estimate_pupil_cutoff(
    profile: JonesPupilSpectrum,
    *,
    na_interval: tuple[float, float] = (0.15, 0.34),
    frequency_interval_um_inv: tuple[float, float] | None = None,
) -> PupilCutoffEstimate:
    """Locate the sharpest downward spectral edge in a physical NA interval."""
    na = profile.numerical_aperture
    if frequency_interval_um_inv is None:
        selected = (na >= float(na_interval[0])) & (na <= float(na_interval[1]))
    else:
        lo, hi = map(float, frequency_interval_um_inv)
        selected = (profile.frequency_um_inv >= lo) & (profile.frequency_um_inv <= hi)
    candidates = np.flatnonzero(selected & np.isfinite(profile.log_slope))
    if candidates.size < 3:
        raise ValueError("NA search interval contains too few frequency bins")
    index = int(candidates[np.argmin(profile.log_slope[candidates])])
    frequency = _quadratic_minimum(
        profile.frequency_um_inv,
        profile.log_slope,
        index,
    )
    slope = float(np.interp(frequency, profile.frequency_um_inv, profile.log_slope))
    half_level = 0.5 * slope
    active = profile.log_slope <= half_level
    left = index
    right = index
    while left > 0 and active[left - 1]:
        left -= 1
    while right + 1 < active.size and active[right + 1]:
        right += 1
    half_width_na = 0.5 * float(profile.wavelength_um) * (
        profile.frequency_um_inv[right] - profile.frequency_um_inv[left]
    )
    return PupilCutoffEstimate(
        frequency_um_inv=float(frequency),
        effective_na=float(frequency * profile.wavelength_um),
        slope_decades_per_um_inv=slope,
        half_width_na=max(half_width_na, 0.0),
    )


def _odd_window_length(width: float, step: float, size: int) -> int:
    value = max(5, int(round(float(width) / float(step))))
    if value % 2 == 0:
        value += 1
    maximum = size if size % 2 == 1 else size - 1
    return min(value, maximum)


def _quadratic_minimum(x: np.ndarray, y: np.ndarray, index: int) -> float:
    if index <= 0 or index >= x.size - 1:
        return float(x[index])
    coefficients = np.polyfit(x[index - 1 : index + 2], y[index - 1 : index + 2], 2)
    if coefficients[0] <= 0.0:
        return float(x[index])
    candidate = -coefficients[1] / (2.0 * coefficients[0])
    return float(np.clip(candidate, x[index - 1], x[index + 1]))
