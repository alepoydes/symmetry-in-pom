"""Through-focus polarimetry anchored by a measured Jones field.

The module keeps the optical forward model separate from experiment-specific
file naming.  Spatial Jones arrays follow the project convention
``J[x, y, output, input]``; camera images use the usual ``image[y, x]`` order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import fft
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter
from scipy.optimize import least_squares
from scipy.signal import windows


CHANNEL_NAMES = ("I0", "I135", "I45", "I90")
ANALYZER_ANGLES_DEG: dict[str, float] = {
    "I0": 0.0,
    "I135": 135.0,
    "I45": 45.0,
    "I90": 90.0,
}
MACRO_QUADRANT_SLICES = {
    "I0": (slice(0, 1024), slice(0, 1224)),
    "I135": (slice(0, 1024), slice(1224, 2448)),
    "I45": (slice(1024, 2048), slice(0, 1224)),
    "I90": (slice(1024, 2048), slice(1224, 2448)),
}
NUMBERED_FRAME_RE = re.compile(r"^(?P<number>\d+)$")


@dataclass(frozen=True)
class MacroPolarimetryFrame:
    """One four-channel camera frame before spatial registration."""

    path: Path
    frame_number: int
    exposure_ms: float
    channels_yx: dict[str, np.ndarray]
    saturation_weights_yx: dict[str, np.ndarray]
    encoded_full_scale: float


@dataclass(frozen=True)
class RadialCenterEstimate:
    """Subpixel radial-symmetry center in image ``(x, y)`` coordinates."""

    center_xy: tuple[float, float]
    coarse_center_xy: tuple[float, float]
    rms_distance_px: float
    used_pixels: int


@dataclass(frozen=True)
class AffineProfileFit:
    """Optimal affine intensity calibration for one radial profile."""

    scale: float
    offset: float
    relative_residual: float
    correlation: float


@dataclass(frozen=True)
class FocusTrajectoryFit:
    """Smooth monotone trajectory fitted to a frame-by-focus cost matrix."""

    focus_um: np.ndarray
    data_cost: np.ndarray
    objective: float
    nominal_step_um: float
    direction: int


def read_macro_polarimetry_frame(
    path: Path,
    *,
    exposure_ms: float,
    soft_saturation_start: float = 0.94,
    saturation_threshold: float = 0.995,
) -> MacroPolarimetryFrame:
    """Read one 2-by-2 macro-quadrant polarimetric PNG.

    The newer 50x acquisition stores four complete 1224-by-1024 analyzer
    images in a single 2448-by-2048 file.  This is deliberately distinct from
    the pixel-interleaved polarization-camera mosaic handled elsewhere.
    """

    source = Path(path).expanduser().resolve()
    with Image.open(source) as image:
        encoded = np.asarray(image)
    if encoded.shape != (2048, 2448):
        raise ValueError(
            f"Expected a 2048x2448 macro-quadrant image, got {encoded.shape}: {source}"
        )
    if not np.issubdtype(encoded.dtype, np.integer):
        raise TypeError(f"Expected an integer camera image, got {encoded.dtype}: {source}")
    if not 0.0 <= soft_saturation_start < saturation_threshold <= 1.0:
        raise ValueError("Require 0 <= soft_saturation_start < saturation_threshold <= 1")

    full_scale = float(np.iinfo(encoded.dtype).max)
    normalized = encoded.astype(np.float32) / full_scale
    channels: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    width = float(saturation_threshold - soft_saturation_start)
    for name, slices in MACRO_QUADRANT_SLICES.items():
        channel = np.ascontiguousarray(normalized[slices])
        coordinate = np.clip(
            (float(saturation_threshold) - channel) / width,
            0.0,
            1.0,
        )
        weight = coordinate * coordinate * (3.0 - 2.0 * coordinate)
        channels[name] = channel
        weights[name] = weight.astype(np.float32, copy=False)

    match = NUMBERED_FRAME_RE.fullmatch(source.stem)
    if match is None:
        raise ValueError(f"Expected a numeric frame filename, got {source.name!r}")
    return MacroPolarimetryFrame(
        path=source,
        frame_number=int(match.group("number")),
        exposure_ms=float(exposure_ms),
        channels_yx=channels,
        saturation_weights_yx=weights,
        encoded_full_scale=full_scale,
    )


def list_macro_polarimetry_frames(
    directory: Path,
    *,
    exposure_ms: float,
) -> tuple[MacroPolarimetryFrame, ...]:
    """Load all numerically named PNG frames in acquisition order."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = []
    for path in root.glob("*.png"):
        match = NUMBERED_FRAME_RE.fullmatch(path.stem)
        if match is not None:
            paths.append((int(match.group("number")), path))
    if not paths:
        raise FileNotFoundError(f"No numerically named PNG frames found in {root}")
    paths.sort(key=lambda item: item[0])
    numbers = [number for number, _path in paths]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate frame numbers in {root}")
    return tuple(
        read_macro_polarimetry_frame(path, exposure_ms=exposure_ms)
        for _number, path in paths
    )


def estimate_radial_center(
    image_yx: np.ndarray,
    *,
    initial_center_xy: tuple[float, float] | None = None,
    search_half_width_px: float = 390.0,
    gradient_sigma_px: float = 1.5,
    highpass_sigma_px: float = 45.0,
    minimum_radius_px: float = 18.0,
    maximum_radius_px: float = 330.0,
    gradient_percentile: float = 72.0,
    irls_iterations: int = 5,
) -> RadialCenterEstimate:
    """Estimate the common center of ring-like image features.

    The gradient of an ideal radial image is parallel to the displacement from
    its center.  That condition is linear in the two center coordinates.  A
    short iteratively reweighted least-squares solve suppresses non-radial
    defects and avoids an expensive two-dimensional center scan.
    """

    image = np.asarray(image_yx, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 64:
        raise ValueError(f"Expected a sufficiently large two-dimensional image, got {image.shape}")
    height, width = image.shape
    if initial_center_xy is None:
        initial = np.asarray((0.5 * (width - 1), 0.5 * (height - 1)), dtype=np.float64)
    else:
        initial = np.asarray(initial_center_xy, dtype=np.float64)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"Invalid initial center: {initial_center_xy!r}")

    finite = np.isfinite(image)
    fill = float(np.nanmedian(image))
    prepared = np.where(finite, image, fill)
    prepared = gaussian_filter(prepared, gradient_sigma_px)
    prepared -= gaussian_filter(prepared, highpass_sigma_px)

    # A local high-frequency-energy maximum provides a robust coarse correction
    # when the toron is not exactly at the image center.
    energy = uniform_filter(prepared * prepared, size=151, mode="nearest")
    yy, xx = np.indices(image.shape, dtype=np.float64)
    allowed = (
        (np.abs(xx - initial[0]) <= float(search_half_width_px))
        & (np.abs(yy - initial[1]) <= float(search_half_width_px))
    )
    energy = np.where(allowed, energy, -np.inf)
    coarse_index = int(np.argmax(energy))
    coarse_y, coarse_x = np.unravel_index(coarse_index, image.shape)
    center = np.asarray((float(coarse_x), float(coarse_y)), dtype=np.float64)

    gy, gx = np.gradient(prepared)
    gradient2 = gx * gx + gy * gy
    radius = np.hypot(xx - center[0], yy - center[1])
    annulus = (
        finite
        & (radius >= float(minimum_radius_px))
        & (radius <= float(maximum_radius_px))
    )
    if np.count_nonzero(annulus) < 100:
        raise ValueError("Center-estimation annulus contains too few finite pixels")
    threshold = float(np.percentile(gradient2[annulus], gradient_percentile))
    selected = annulus & (gradient2 >= threshold)
    x = xx[selected]
    y = yy[selected]
    gx_selected = gx[selected]
    gy_selected = gy[selected]
    magnitude = np.sqrt(gradient2[selected])
    design = np.column_stack((gy_selected, -gx_selected))
    target = gy_selected * x - gx_selected * y
    base_weight = magnitude / max(float(np.median(magnitude)), np.finfo(float).eps)
    robust_weight = np.ones_like(base_weight)

    for _iteration in range(int(irls_iterations)):
        weight = np.sqrt(np.clip(base_weight * robust_weight, 1.0e-6, 1.0e6))
        solution, *_unused = np.linalg.lstsq(
            design * weight[:, None],
            target * weight,
            rcond=None,
        )
        center = np.asarray(solution, dtype=np.float64)
        residual = design @ center - target
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        scale = max(scale, np.finfo(float).eps)
        coordinate = np.abs(residual) / (2.5 * scale)
        robust_weight = np.where(coordinate < 1.0, (1.0 - coordinate * coordinate) ** 2, 0.0)

    residual = design @ center - target
    normalized_distance = residual / np.maximum(magnitude, np.finfo(float).eps)
    return RadialCenterEstimate(
        center_xy=(float(center[0]), float(center[1])),
        coarse_center_xy=(float(coarse_x), float(coarse_y)),
        rms_distance_px=float(np.sqrt(np.mean(normalized_distance * normalized_distance))),
        used_pixels=int(np.count_nonzero(robust_weight > 0.0)),
    )


def sample_radial_profile(
    image_yx: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radii_px: np.ndarray,
    weights_yx: np.ndarray | None = None,
    angle_count: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a soft-weighted angular mean and its effective angular coverage."""

    image = np.asarray(image_yx, dtype=np.float64)
    radii = np.asarray(radii_px, dtype=np.float64)
    if image.ndim != 2 or radii.ndim != 1:
        raise ValueError("image_yx must be 2D and radii_px must be 1D")
    angles = np.linspace(0.0, 2.0 * np.pi, int(angle_count), endpoint=False)
    x = float(center_xy[0]) + radii[:, None] * np.cos(angles)[None, :]
    y = float(center_xy[1]) + radii[:, None] * np.sin(angles)[None, :]
    samples = map_coordinates(
        image,
        (y, x),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    if weights_yx is None:
        weights = np.isfinite(samples).astype(np.float64)
    else:
        weights = map_coordinates(
            np.asarray(weights_yx, dtype=np.float64),
            (y, x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        weights = np.where(np.isfinite(samples), np.clip(weights, 0.0, 1.0), 0.0)
    numerator = np.nansum(np.where(np.isfinite(samples), samples, 0.0) * weights, axis=1)
    denominator = np.sum(weights, axis=1)
    profile = numerator / np.maximum(denominator, np.finfo(float).eps)
    coverage = denominator / float(angle_count)
    return profile, coverage


def fit_affine_profile(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    require_positive_scale: bool = True,
) -> AffineProfileFit:
    """Fit ``observed = scale * predicted + offset`` and report shape error."""

    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    x = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if y.shape != x.shape:
        raise ValueError(f"Profile shapes differ: {y.shape} versus {x.shape}")
    if weights is None:
        weight = np.ones_like(y)
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight.shape != y.shape:
            raise ValueError("weights do not match profile shape")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0.0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("Too few valid radial samples for affine profile fit")
    x = x[valid]
    y = y[valid]
    weight = weight[valid]
    design = np.column_stack((x, np.ones_like(x)))
    root_weight = np.sqrt(weight)
    parameters, *_unused = np.linalg.lstsq(
        design * root_weight[:, None],
        y * root_weight,
        rcond=None,
    )
    scale, offset = map(float, parameters)
    if require_positive_scale and scale < 0.0:
        scale = 0.0
        offset = float(np.sum(weight * y) / np.sum(weight))
    fitted = scale * x + offset
    residual_energy = float(np.sum(weight * (y - fitted) ** 2))
    mean = float(np.sum(weight * y) / np.sum(weight))
    signal_energy = float(np.sum(weight * (y - mean) ** 2))
    x_mean = float(np.sum(weight * x) / np.sum(weight))
    covariance = float(np.sum(weight * (x - x_mean) * (y - mean)))
    x_energy = float(np.sum(weight * (x - x_mean) ** 2))
    correlation = covariance / max(np.sqrt(x_energy * signal_energy), np.finfo(float).eps)
    return AffineProfileFit(
        scale=scale,
        offset=offset,
        relative_residual=residual_energy / max(signal_energy, np.finfo(float).eps),
        correlation=float(correlation),
    )


def affine_profile_costs(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    require_positive_scale: bool = True,
) -> np.ndarray:
    """Vectorized affine-calibrated residual for a bank of predictions.

    ``observed`` has shape ``(radius,)`` and ``predicted`` may have any leading
    dimensions followed by the same radial dimension.  The returned array has
    the leading shape of ``predicted``.
    """

    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    x = np.asarray(predicted, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != y.size:
        raise ValueError(
            f"predicted must end in the observed profile size {y.size}, got {x.shape}"
        )
    if weights is None:
        weight = np.ones_like(y)
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight.shape != y.shape:
            raise ValueError("weights do not match profile shape")
    valid = np.isfinite(y) & np.isfinite(weight) & (weight > 0.0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("Too few valid radial samples for affine profile costs")
    y = y[valid]
    weight = weight[valid]
    x = x[..., valid]
    finite_prediction = np.all(np.isfinite(x), axis=-1)
    total_weight = float(np.sum(weight))
    y_mean = float(np.sum(weight * y) / total_weight)
    y_centered = y - y_mean
    y_energy = float(np.sum(weight * y_centered * y_centered))
    x_mean = np.sum(x * weight, axis=-1) / total_weight
    x_centered = x - x_mean[..., None]
    covariance = np.sum(x_centered * y_centered * weight, axis=-1)
    x_energy = np.sum(x_centered * x_centered * weight, axis=-1)
    scale = covariance / np.maximum(x_energy, np.finfo(float).eps)
    if require_positive_scale:
        scale = np.maximum(scale, 0.0)
    residual = y_centered - scale[..., None] * x_centered
    cost = np.sum(weight * residual * residual, axis=-1) / max(
        y_energy, np.finfo(float).eps
    )
    return np.where(finite_prediction, cost, np.inf)


class AngularSpectrumJonesPropagator:
    """Band-limited free-space propagation of a measured Jones field.

    A constant border Jones response is propagated analytically as a plane
    wave.  Only the localized perturbation is tapered, padded, transformed,
    and propagated, which avoids both periodic copies and an artificial hard
    edge in the incident field.
    """

    def __init__(
        self,
        jones: np.ndarray,
        *,
        pixel_size_um: float,
        wavelength_um: float,
        numerical_aperture: float | None = None,
        padding_factor: float = 1.5,
        taper_fraction: float = 0.15,
        workers: int = -1,
    ) -> None:
        matrix = np.asarray(jones, dtype=np.complex64)
        if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
            raise ValueError(f"Expected J[x,y,2,2], got {matrix.shape}")
        if not np.isfinite(pixel_size_um) or float(pixel_size_um) <= 0.0:
            raise ValueError("pixel_size_um must be positive and finite")
        if not np.isfinite(wavelength_um) or float(wavelength_um) <= 0.0:
            raise ValueError("wavelength_um must be positive and finite")
        if numerical_aperture is not None and not 0.0 < float(numerical_aperture) <= 1.0:
            raise ValueError("numerical_aperture must lie in (0, 1]")
        if float(padding_factor) < 1.0:
            raise ValueError("padding_factor must be at least one")

        self.jones_shape = matrix.shape[:2]
        self.pixel_size_um = float(pixel_size_um)
        self.wavelength_um = float(wavelength_um)
        self.numerical_aperture = None if numerical_aperture is None else float(numerical_aperture)
        self.workers = int(workers)
        self.background = robust_border_jones(matrix)
        perturbation = matrix - self.background
        taper = windows.tukey(matrix.shape[0], alpha=float(taper_fraction))[:, None]
        taper = taper * windows.tukey(matrix.shape[1], alpha=float(taper_fraction))[None, :]
        perturbation = perturbation * taper[..., None, None]

        padded_shape = tuple(
            fft.next_fast_len(int(np.ceil(size * float(padding_factor))))
            for size in matrix.shape[:2]
        )
        self.padded_shape = padded_shape
        self._crop = _centered_slices(padded_shape, matrix.shape[:2])
        padded = np.zeros((*padded_shape, 2, 2), dtype=np.complex64)
        padded[self._crop] = perturbation
        self._spectrum = fft.fft2(padded, axes=(0, 1), workers=self.workers)

        fx = fft.fftfreq(padded_shape[0], d=self.pixel_size_um)
        fy = fft.fftfreq(padded_shape[1], d=self.pixel_size_um)
        kx = 2.0 * np.pi * fx[:, None]
        ky = 2.0 * np.pi * fy[None, :]
        k = 2.0 * np.pi / self.wavelength_um
        transverse2 = kx * kx + ky * ky
        propagating = transverse2 <= k * k
        if self.numerical_aperture is not None:
            propagating &= transverse2 <= (k * self.numerical_aperture) ** 2
        kz = np.sqrt(np.maximum(k * k - transverse2, 0.0))
        self._delta_kz = (kz - k).astype(np.float32)
        self._pupil = propagating.astype(np.float32)

    def propagate_jones(self, distance_um: float) -> np.ndarray:
        """Return ``J[x,y,output,input]`` after free-space propagation."""

        distance = float(distance_um)
        if not np.isfinite(distance):
            raise ValueError("distance_um must be finite")
        phase = self._pupil * np.exp(1j * self._delta_kz * distance)
        propagated = fft.ifft2(
            self._spectrum * phase[..., None, None],
            axes=(0, 1),
            workers=self.workers,
        )[self._crop]
        propagated += self.background
        return np.asarray(propagated, dtype=np.complex64)

    def propagate_field(self, distance_um: float, input_state: np.ndarray) -> np.ndarray:
        """Return the two-component field for one normalized input state."""

        state = np.asarray(input_state, dtype=np.complex64).reshape(-1)
        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError("input_state must contain two finite complex amplitudes")
        norm = float(np.linalg.norm(state))
        if norm <= np.finfo(float).eps:
            raise ValueError("input_state must be nonzero")
        return self.propagate_jones(distance_um) @ (state / norm)

    def transverse_momentum(self, input_state: np.ndarray) -> tuple[float, float]:
        """Return paraxial centroid slopes ``(dx/dz, dy/dz)`` of the field.

        The constant plane-wave background is excluded so that the finite
        localized toron response, rather than an arbitrarily large crop of the
        illumination, defines the moment.
        """

        state = np.asarray(input_state, dtype=np.complex64).reshape(2)
        norm = float(np.linalg.norm(state))
        if norm <= np.finfo(float).eps:
            raise ValueError("input_state must be nonzero")
        state = state / norm
        field_spectrum = self._spectrum @ state
        power = np.sum(np.abs(field_spectrum) ** 2, axis=-1) * self._pupil
        denominator = float(np.sum(power))
        if denominator <= np.finfo(float).eps:
            return 0.0, 0.0
        fx = fft.fftfreq(self.padded_shape[0], d=self.pixel_size_um)
        fy = fft.fftfreq(self.padded_shape[1], d=self.pixel_size_um)
        k = 2.0 * np.pi / self.wavelength_um
        kx = 2.0 * np.pi * fx[:, None]
        ky = 2.0 * np.pi * fy[None, :]
        kz = k + self._delta_kz
        safe_kz = np.maximum(kz, 0.05 * k)
        slope_x = float(np.sum(power * (kx / safe_kz)) / denominator)
        slope_y = float(np.sum(power * (ky / safe_kz)) / denominator)
        return slope_x, slope_y


class InputColumnWavelengthJonesPropagator:
    """Propagate the two Cartesian input columns at specified wavelengths.

    This is a sensitivity-control operator, not a monochromatic Jones
    operator.  It is applied before any change of input-polarization basis,
    because such a basis change would mix the two source wavelengths.
    """

    def __init__(
        self,
        jones: np.ndarray,
        *,
        pixel_size_um: float,
        wavelength_um_by_input: tuple[float, float],
        numerical_aperture: float | None = None,
        padding_factor: float = 1.5,
        taper_fraction: float = 0.15,
        workers: int = -1,
    ) -> None:
        matrix = np.asarray(jones, dtype=np.complex64)
        if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
            raise ValueError(f"Expected J[x,y,2,2], got {matrix.shape}")
        wavelengths = np.asarray(
            wavelength_um_by_input, dtype=np.float64
        ).reshape(-1)
        if (
            wavelengths.shape != (2,)
            or not np.all(np.isfinite(wavelengths))
            or np.any(wavelengths <= 0.0)
        ):
            raise ValueError(
                "wavelength_um_by_input must contain two positive finite values"
            )
        self.wavelength_um_by_input = (
            float(wavelengths[0]),
            float(wavelengths[1]),
        )
        self._common: AngularSpectrumJonesPropagator | None = None
        self._columns: tuple[
            AngularSpectrumJonesPropagator,
            AngularSpectrumJonesPropagator,
        ] | None = None
        if wavelengths[0] == wavelengths[1]:
            self._common = AngularSpectrumJonesPropagator(
                matrix,
                pixel_size_um=pixel_size_um,
                wavelength_um=float(wavelengths[0]),
                numerical_aperture=numerical_aperture,
                padding_factor=padding_factor,
                taper_fraction=taper_fraction,
                workers=workers,
            )
            return

        propagators = []
        for input_index, wavelength_um in enumerate(wavelengths):
            isolated = np.zeros_like(matrix)
            isolated[..., :, input_index] = matrix[..., :, input_index]
            propagators.append(
                AngularSpectrumJonesPropagator(
                    isolated,
                    pixel_size_um=pixel_size_um,
                    wavelength_um=float(wavelength_um),
                    numerical_aperture=numerical_aperture,
                    padding_factor=padding_factor,
                    taper_fraction=taper_fraction,
                    workers=workers,
                )
            )
        self._columns = (propagators[0], propagators[1])

    def propagate_jones(self, distance_um: float) -> np.ndarray:
        """Return the two independently propagated input columns."""

        if self._common is not None:
            return self._common.propagate_jones(distance_um)
        if self._columns is None:
            raise RuntimeError("column propagators were not initialized")
        output = np.empty(
            (*self._columns[0].jones_shape, 2, 2), dtype=np.complex64
        )
        for input_index, propagator in enumerate(self._columns):
            propagated = propagator.propagate_jones(distance_um)
            output[..., :, input_index] = propagated[..., :, input_index]
        return output


def robust_border_jones(jones: np.ndarray) -> np.ndarray:
    """Return the robust constant Jones response at an array border."""

    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected J[x,y,2,2], got {matrix.shape!r}")
    return _robust_border_jones(matrix)


def radial_jones_gram(
    jones: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radii_px: np.ndarray,
    angle_count: int = 360,
) -> np.ndarray:
    """Return ring-averaged Gram matrices of vectorized Jones coefficients."""

    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected J[x,y,2,2], got {matrix.shape}")
    radii = np.asarray(radii_px, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, int(angle_count), endpoint=False)
    x = float(center_xy[0]) + radii[:, None] * np.cos(angles)[None, :]
    y = float(center_xy[1]) + radii[:, None] * np.sin(angles)[None, :]
    vectors = []
    for output_index in range(2):
        for input_index in range(2):
            component_xy = matrix[..., output_index, input_index]
            real = map_coordinates(
                np.real(component_xy),
                (x, y),
                order=1,
                mode="nearest",
                prefilter=False,
            )
            imag = map_coordinates(
                np.imag(component_xy),
                (x, y),
                order=1,
                mode="nearest",
                prefilter=False,
            )
            vectors.append(real + 1j * imag)
    vector = np.stack(vectors, axis=-1)
    return np.mean(vector[..., :, None] * np.conjugate(vector[..., None, :]), axis=1)


def intensity_profile_from_gram(
    gram: np.ndarray,
    *,
    input_angle_deg: float,
    analyzer_angle_deg: float,
) -> np.ndarray:
    """Evaluate one linear-polarizer profile from ring-averaged Jones data."""

    values = np.asarray(gram)
    if values.ndim < 3 or values.shape[-2:] != (4, 4):
        raise ValueError(f"Expected gram[...,r,4,4], got {values.shape}")
    input_angle = np.radians(float(input_angle_deg))
    analyzer_angle = np.radians(float(analyzer_angle_deg))
    incident = np.asarray((np.cos(input_angle), np.sin(input_angle)))
    analyzer = np.asarray((np.cos(analyzer_angle), np.sin(analyzer_angle)))
    coefficient = np.kron(analyzer, incident)
    profile = np.einsum("i,...ij,j->...", coefficient, values, coefficient, optimize=True)
    return np.real(profile)


def fit_smooth_focus_trajectory(
    cost: np.ndarray,
    focus_grid_um: np.ndarray,
    *,
    nominal_step_um: float,
    step_tolerance_um: float = 2.5,
    curvature_tolerance_um: float = 0.75,
) -> FocusTrajectoryFit:
    """Fit a monotone focus sequence to a precomputed cost surface.

    A coarse affine search establishes the propagation direction and avoids
    local ring aliases.  Continuous per-frame corrections are then optimized
    against linearly interpolated profile costs with step and curvature priors.
    """

    surface = np.asarray(cost, dtype=np.float64)
    grid = np.asarray(focus_grid_um, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1] != grid.size:
        raise ValueError("cost must have shape (frame, focus_grid.size)")
    if grid.ndim != 1 or grid.size < 5 or not np.all(np.diff(grid) > 0.0):
        raise ValueError("focus_grid_um must be a strictly increasing 1D grid")
    frame_count = surface.shape[0]
    indices = np.arange(frame_count, dtype=np.float64)
    candidate_steps = np.linspace(
        max(0.25, abs(float(nominal_step_um)) - 3.0 * float(step_tolerance_um)),
        abs(float(nominal_step_um)) + 3.0 * float(step_tolerance_um),
        81,
    )
    offsets = np.linspace(grid[0], grid[-1], min(501, max(101, grid.size)))
    best: tuple[float, float, int] | None = None
    for direction in (-1, 1):
        for step in candidate_steps:
            trajectories = offsets[:, None] + float(direction) * step * indices[None, :]
            valid = np.all(
                (trajectories >= grid[0]) & (trajectories <= grid[-1]),
                axis=1,
            )
            if not np.any(valid):
                continue
            trajectories = trajectories[valid]
            valid_offsets = offsets[valid]
            values = np.zeros(trajectories.shape[0], dtype=np.float64)
            for frame_index in range(frame_count):
                values += np.interp(
                    trajectories[:, frame_index],
                    grid,
                    surface[frame_index],
                )
            position = int(np.argmin(values))
            candidate = (float(values[position]), float(valid_offsets[position]), int(direction))
            if best is None or candidate[0] < best[0]:
                best = candidate
                best_step = float(step)
    if best is None:
        raise ValueError("Focus grid is too narrow for the requested trajectory")
    _coarse_cost, offset, direction = best
    initial = offset + float(direction) * best_step * indices

    row_minimum = np.nanmin(surface, axis=1)
    row_scale = np.empty(frame_count, dtype=np.float64)
    probe_distance = max(float(step_tolerance_um), float(np.median(np.diff(grid))))
    for frame_index in range(frame_count):
        minimum_index = int(np.nanargmin(surface[frame_index]))
        minimum_focus = float(grid[minimum_index])
        left = float(
            np.interp(
                max(grid[0], minimum_focus - probe_distance),
                grid,
                surface[frame_index],
            )
        )
        right = float(
            np.interp(
                min(grid[-1], minimum_focus + probe_distance),
                grid,
                surface[frame_index],
            )
        )
        row_scale[frame_index] = max(
            0.5 * (left + right) - row_minimum[frame_index],
            np.finfo(float).eps,
        )

    def residual(focus: np.ndarray) -> np.ndarray:
        data = np.asarray(
            [
                np.sqrt(
                    max(
                        float(np.interp(value, grid, surface[index]))
                        - row_minimum[index],
                        0.0,
                    )
                    / row_scale[index]
                )
                for index, value in enumerate(focus)
            ]
        )
        steps = np.diff(focus)
        target = float(direction) * abs(float(nominal_step_um))
        step_residual = (steps - target) / max(float(step_tolerance_um), np.finfo(float).eps)
        curvature = np.diff(focus, n=2) / max(
            float(curvature_tolerance_um), np.finfo(float).eps
        )
        return np.concatenate((data, step_residual, curvature))

    optimized = least_squares(
        residual,
        initial,
        bounds=(np.full(frame_count, grid[0]), np.full(frame_count, grid[-1])),
        max_nfev=300,
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    focus = np.asarray(optimized.x)
    data_cost = np.asarray(
        [np.interp(value, grid, surface[index]) for index, value in enumerate(focus)]
    )
    return FocusTrajectoryFit(
        focus_um=focus,
        data_cost=data_cost,
        objective=float(np.sum(residual(focus) ** 2)),
        nominal_step_um=float(np.median(np.diff(focus))),
        direction=int(direction),
    )


def _robust_border_jones(matrix: np.ndarray, border_fraction: float = 0.08) -> np.ndarray:
    nx, ny = matrix.shape[:2]
    width_x = max(2, int(round(border_fraction * nx)))
    width_y = max(2, int(round(border_fraction * ny)))
    mask = np.zeros((nx, ny), dtype=bool)
    mask[:width_x] = True
    mask[-width_x:] = True
    mask[:, :width_y] = True
    mask[:, -width_y:] = True
    values = matrix[mask]
    background = np.empty((2, 2), dtype=np.complex64)
    for output_index in range(2):
        for input_index in range(2):
            component = values[:, output_index, input_index]
            background[output_index, input_index] = np.median(np.real(component)) + 1j * np.median(
                np.imag(component)
            )
    return background


def _centered_slices(
    outer_shape: Sequence[int], inner_shape: Sequence[int]
) -> tuple[slice, slice]:
    slices = []
    for outer, inner in zip(outer_shape, inner_shape, strict=True):
        if inner > outer:
            raise ValueError(f"Inner shape {tuple(inner_shape)} exceeds {tuple(outer_shape)}")
        start = (int(outer) - int(inner)) // 2
        slices.append(slice(start, start + int(inner)))
    return tuple(slices)  # type: ignore[return-value]
