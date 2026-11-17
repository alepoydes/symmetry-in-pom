"""Axisymmetric reductions of spatially resolved Jones fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toronoptics.analysis_specs import BackgroundSpec
from toronoptics.laguerre_gaussian import soft_annular_angular_profiles
from toronoptics.polar_fourier import (
    PolarSamplingGrid,
    angular_fourier,
    cartesian_coordinate_grids,
    sample_polar,
)


@dataclass(frozen=True)
class SubpixelCenterResult:
    center_px: tuple[float, float]
    objective: float
    initial_center_px: tuple[float, float]
    broad_x_px: np.ndarray
    broad_y_px: np.ndarray
    broad_objective: np.ndarray
    broad_all_objective: np.ndarray
    coarse_x_px: np.ndarray
    coarse_y_px: np.ndarray
    coarse_objective: np.ndarray
    coarse_all_objective: np.ndarray
    coarse_record_objective: np.ndarray
    fine_x_px: np.ndarray
    fine_y_px: np.ndarray
    fine_objective: np.ndarray
    fine_all_objective: np.ndarray
    broad_converged: bool
    coarse_converged: bool
    record_centers_px: np.ndarray
    record_objective: np.ndarray


@dataclass(frozen=True)
class IntensityHarmonicProfiles:
    radius_um: np.ndarray
    modes: np.ndarray
    coefficients: np.ndarray
    visibility: np.ndarray


@dataclass(frozen=True)
class AngularHarmonicProfiles:
    """Soft-annulus Fourier coefficients of a Cartesian scalar field."""

    radius_um: np.ndarray
    modes: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class CenterHarmonicDiagnostics:
    radius_um: np.ndarray
    first_harmonic_power: np.ndarray
    all_forbidden_power: np.ndarray
    radial_signal_power: np.ndarray
    modes: np.ndarray
    integrated_mode_power: np.ndarray


@dataclass(frozen=True)
class JonesAngularModePower:
    """Radially integrated angular spectrum of a Jones perturbation."""

    modes: np.ndarray
    mode_power: np.ndarray
    total_power: float
    captured_power_fraction: float


def annular_jones_background(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    min_radius_um: float | None = None,
    max_radius_um: float | None = None,
    spec: BackgroundSpec | None = None,
) -> np.ndarray:
    """Return a constant complex Jones background from a physical annulus.

    ``spec`` is the preferred publication interface because it records both
    the physical region and the estimator.  The explicit radius arguments are
    retained for backwards compatibility and use the arithmetic mean.
    """

    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(
            f"Expected Jones field with shape (x, y, 2, 2), got {matrix.shape!r}"
    )
    _validate_positive("pixel_size_um", pixel_size_um)
    if spec is not None:
        if min_radius_um is not None or max_radius_um is not None:
            raise ValueError("Pass either spec or explicit annulus radii, not both")
        min_radius_um = float(spec.min_radius_um)
        max_radius_um = float(spec.max_radius_um)
        estimator = spec.estimator
    else:
        if min_radius_um is None:
            raise ValueError("min_radius_um is required when spec is not supplied")
        estimator = "mean"
    _validate_nonnegative("min_radius_um", min_radius_um)
    if max_radius_um is not None and float(max_radius_um) <= float(min_radius_um):
        raise ValueError("max_radius_um must exceed min_radius_um")
    radius_um, _phi = polar_coordinate_grids(
        matrix.shape[:2],
        center_px=center_px,
        pixel_size_um=float(pixel_size_um),
    )
    selected = radius_um >= float(min_radius_um)
    if max_radius_um is not None:
        selected &= radius_um <= float(max_radius_um)
    if np.count_nonzero(selected) < 16:
        raise ValueError("Background annulus contains fewer than 16 pixels")
    values = np.asarray(matrix[selected], dtype=np.complex128)
    if estimator == "mean":
        return np.mean(values, axis=0, dtype=np.complex128)
    if estimator == "huber":
        assert spec is not None
        return _huber_jones_location(values, spec=spec)
    raise AssertionError(f"Unhandled background estimator: {estimator!r}")


def _huber_jones_location(
    values: np.ndarray,
    *,
    spec: BackgroundSpec,
) -> np.ndarray:
    """Robust unitary-basis-invariant location of complex Jones matrices."""

    matrices = np.asarray(values, dtype=np.complex128)
    if matrices.ndim != 3 or matrices.shape[-2:] != (2, 2):
        raise ValueError("values must have shape (sample, 2, 2)")
    # The componentwise complex median is used only as a robust starting
    # point.  Iteration weights use the Frobenius distance of the complete
    # matrix, so the final M-estimator is invariant under unitary changes of
    # input and output polarization basis.
    location = np.median(matrices.real, axis=0) + 1j * np.median(
        matrices.imag, axis=0
    )
    tiny = np.finfo(np.float64).tiny
    for _ in range(int(spec.maximum_iterations)):
        distance = np.sqrt(np.sum(np.abs(matrices - location[None]) ** 2, axis=(1, 2)))
        median_distance = float(np.median(distance))
        scale = max(1.4826 * median_distance, tiny)
        cutoff = float(spec.huber_threshold) * scale
        weight = np.minimum(1.0, cutoff / np.maximum(distance, tiny))
        updated = np.sum(matrices * weight[:, None, None], axis=0) / np.sum(weight)
        relative_change = np.linalg.norm(updated - location) / max(
            np.linalg.norm(location), tiny
        )
        location = updated
        if relative_change <= float(spec.relative_tolerance):
            break
    return np.asarray(location, dtype=np.complex128)


def jones_angular_mode_power(
    jones_perturbation: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
    modes: np.ndarray,
    angular_samples: int = 720,
    interpolation: str = "cubic",
) -> JonesAngularModePower:
    r"""Return the basis-invariant radial power of signed Jones-field modes.

    The supplied field must already have its constant outer Jones response
    subtracted.  For
    ``J_m(rho)=mean_phi[J(rho,phi) exp(-i m phi)]``, the reported mode power is
    ``2*pi*integral ||J_m||_F^2 rho d rho``.  The denominator is the angularly
    averaged real-space Frobenius power over the same disk, so the individual
    fractions obey Parseval's identity when all angular orders are retained.
    """

    matrix = np.asarray(jones_perturbation)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(
            f"Expected Jones field with shape (x, y, 2, 2), got {matrix.shape!r}"
        )
    _validate_positive("pixel_size_um", pixel_size_um)
    radius = np.asarray(radii_um, dtype=np.float64)
    if (
        radius.ndim != 1
        or radius.size < 2
        or np.any(radius < 0.0)
        or np.any(np.diff(radius) <= 0.0)
    ):
        raise ValueError("radii_um must be nonnegative and strictly increasing")
    signed_modes = np.asarray(modes, dtype=np.int64)
    if signed_modes.ndim != 1 or signed_modes.size == 0:
        raise ValueError("modes must be a nonempty one-dimensional array")
    if np.unique(signed_modes).size != signed_modes.size:
        raise ValueError("modes must not contain duplicates")
    if int(angular_samples) < 2 * int(np.max(np.abs(signed_modes))) + 1:
        raise ValueError("angular_samples is too small for the requested modes")

    grid = PolarSamplingGrid.uniform(
        center_xy_px=center_px,
        pixel_size_xy_um=float(pixel_size_um),
        radius_um=radius,
        angular_samples=int(angular_samples),
        axis_order="xy",
        boundary="raise",
    )
    sampled = sample_polar(matrix, grid, interpolation=interpolation)
    coefficients = angular_fourier(
        sampled,
        signed_modes,
        phi_rad=grid.phi_rad,
        angular_axis=1,
    )
    mode_density = np.sum(np.abs(coefficients) ** 2, axis=(2, 3))
    mode_power = 2.0 * np.pi * np.trapezoid(
        mode_density * radius[None, :],
        radius,
        axis=1,
    )
    total_density = np.mean(
        np.sum(np.abs(sampled) ** 2, axis=(2, 3)),
        axis=1,
    )
    total_power = float(
        2.0 * np.pi * np.trapezoid(total_density * radius, radius)
    )
    if total_power <= np.finfo(float).tiny:
        raise ValueError("Jones perturbation has zero power in the analysis disk")
    return JonesAngularModePower(
        modes=signed_modes,
        mode_power=np.asarray(mode_power, dtype=np.float64),
        total_power=total_power,
        captured_power_fraction=float(np.sum(mode_power) / total_power),
    )


def refine_axisymmetric_center(
    jones_stack: np.ndarray,
    *,
    initial_center_px: tuple[float, float],
    pixel_size_um: float,
    min_radius_um: float = 4.0,
    max_radius_um: float = 30.0,
    radial_step_um: float = 0.75,
    angular_samples: int = 360,
    broad_radius_px: float = 16.0,
    broad_step_px: float = 2.0,
    coarse_radius_px: float = 2.0,
    coarse_step_px: float = 0.25,
    fine_radius_px: float = 0.3,
    fine_step_px: float = 0.05,
) -> SubpixelCenterResult:
    """Refine a common center by suppressing the corotating m=+/-1 sectors."""

    stack = _as_jones_stack(jones_stack)
    _validate_positive("pixel_size_um", pixel_size_um)
    _validate_positive("radial_step_um", radial_step_um)
    _validate_positive("broad_step_px", broad_step_px)
    _validate_positive("coarse_step_px", coarse_step_px)
    _validate_positive("fine_step_px", fine_step_px)
    if max_radius_um <= min_radius_um:
        raise ValueError("max_radius_um must exceed min_radius_um")
    if angular_samples < 16:
        raise ValueError("angular_samples must be at least 16")

    radii_um = np.arange(
        float(min_radius_um),
        float(max_radius_um) + 0.5 * float(radial_step_um),
        float(radial_step_um),
        dtype=np.float64,
    )
    phi = np.linspace(0.0, 2.0 * np.pi, int(angular_samples), endpoint=False)
    broad_phi = phi[:: max(1, phi.size // 90)]
    coarse_phi = phi[:: max(1, phi.size // 180)]
    broad_offsets = _symmetric_grid(broad_radius_px, broad_step_px)
    broad_x = float(initial_center_px[0]) + broad_offsets
    broad_y = float(initial_center_px[1]) + broad_offsets
    broad_objective, _broad_record, broad_all_objective, _broad_all_record = (
        _center_objective_grid(
            stack,
            broad_x,
            broad_y,
            pixel_size_um=float(pixel_size_um),
            radii_um=radii_um,
            phi=broad_phi,
        )
    )
    broad_index = np.unravel_index(
        int(np.nanargmin(broad_objective)), broad_objective.shape
    )
    broad_center = np.asarray(
        (broad_x[broad_index[0]], broad_y[broad_index[1]]), dtype=np.float64
    )
    broad_converged = not _index_on_boundary(broad_index, broad_objective.shape)

    coarse_offsets = _symmetric_grid(coarse_radius_px, coarse_step_px)
    coarse_x = broad_center[0] + coarse_offsets
    coarse_y = broad_center[1] + coarse_offsets
    coarse_objective, coarse_record, coarse_all_objective, _coarse_all_record = (
        _center_objective_grid(
            stack,
            coarse_x,
            coarse_y,
            pixel_size_um=float(pixel_size_um),
            radii_um=radii_um,
            phi=coarse_phi,
        )
    )
    coarse_index = np.unravel_index(
        int(np.nanargmin(coarse_objective)), coarse_objective.shape
    )
    coarse_center = np.asarray(
        (coarse_x[coarse_index[0]], coarse_y[coarse_index[1]]), dtype=np.float64
    )
    coarse_converged = not _index_on_boundary(coarse_index, coarse_objective.shape)

    fine_offsets = _symmetric_grid(fine_radius_px, fine_step_px)
    fine_x = coarse_center[0] + fine_offsets
    fine_y = coarse_center[1] + fine_offsets
    fine_objective, _fine_record, fine_all_objective, _fine_all_record = _center_objective_grid(
        stack,
        fine_x,
        fine_y,
        pixel_size_um=float(pixel_size_um),
        radii_um=radii_um,
        phi=phi,
    )
    fine_index = np.unravel_index(
        int(np.nanargmin(fine_objective)), fine_objective.shape
    )
    center = (float(fine_x[fine_index[0]]), float(fine_y[fine_index[1]]))

    record_indices = np.argmin(
        coarse_record.reshape((stack.shape[0], -1)), axis=1
    )
    record_ix, record_iy = np.unravel_index(
        record_indices, coarse_objective.shape
    )
    record_centers = np.stack(
        (coarse_x[record_ix], coarse_y[record_iy]), axis=-1
    )
    record_objective = coarse_record[
        np.arange(stack.shape[0]), record_ix, record_iy
    ]
    return SubpixelCenterResult(
        center_px=center,
        objective=float(fine_objective[fine_index]),
        initial_center_px=(
            float(initial_center_px[0]),
            float(initial_center_px[1]),
        ),
        broad_x_px=broad_x,
        broad_y_px=broad_y,
        broad_objective=broad_objective,
        broad_all_objective=broad_all_objective,
        coarse_x_px=coarse_x,
        coarse_y_px=coarse_y,
        coarse_objective=coarse_objective,
        coarse_all_objective=coarse_all_objective,
        coarse_record_objective=coarse_record,
        fine_x_px=fine_x,
        fine_y_px=fine_y,
        fine_objective=fine_objective,
        fine_all_objective=fine_all_objective,
        broad_converged=broad_converged,
        coarse_converged=coarse_converged,
        record_centers_px=record_centers,
        record_objective=record_objective,
    )


def axisymmetric_center_objective(
    jones_stack: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
    phi: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return the equally weighted per-record m=+/-1 center objective."""

    first, record, _all, _all_record = _axisymmetric_center_metrics(
        jones_stack,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        radii_um=radii_um,
        phi=phi,
    )
    return first, record


def center_harmonic_diagnostics(
    jones_stack: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
    angular_samples: int = 360,
    max_mode: int = 6,
) -> CenterHarmonicDiagnostics:
    stack = _as_jones_stack(jones_stack)
    radius = np.asarray(radii_um, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, int(angular_samples), endpoint=False)
    sampled = _bilinear_sample_jones_stack(
        stack,
        center_px=center_px,
        pixel_size_um=float(pixel_size_um),
        radii_um=radius,
        phi=phi,
    )
    corotating = _to_corotating(sampled, phi)
    radial_mean = np.mean(corotating, axis=2)
    residual = corotating - radial_mean[:, :, None]
    phase = np.exp(-1j * phi)
    plus = np.mean(corotating * phase[None, None, :, None, None], axis=2)
    minus = np.mean(corotating * np.conj(phase)[None, None, :, None, None], axis=2)
    radial_weight = radius * _quadrature_step(radius)
    radial_average = np.sum(
        radial_mean * radial_weight[None, :, None, None], axis=1
    ) / np.sum(radial_weight)
    radial_signal = radial_mean - radial_average[:, None]
    first_power = np.mean(np.sum(np.abs(plus) ** 2 + np.abs(minus) ** 2, axis=(2, 3)), axis=0)
    all_power = np.mean(np.mean(np.sum(np.abs(residual) ** 2, axis=(3, 4)), axis=2), axis=0)
    signal_power = np.mean(np.sum(np.abs(radial_signal) ** 2, axis=(2, 3)), axis=0)
    modes = np.arange(-int(max_mode), int(max_mode) + 1, dtype=np.int64)
    coefficients = np.asarray(
        [
            np.mean(
                corotating
                * np.exp(-1j * int(mode) * phi)[None, None, :, None, None],
                axis=2,
            )
            for mode in modes
        ]
    )
    mode_power = np.sum(
        np.mean(np.sum(np.abs(coefficients) ** 2, axis=(3, 4)), axis=1)
        * radial_weight[None, :],
        axis=1,
    )
    return CenterHarmonicDiagnostics(
        radius_um=radius,
        first_harmonic_power=first_power,
        all_forbidden_power=all_power,
        radial_signal_power=signal_power,
        modes=modes,
        integrated_mode_power=mode_power,
    )


def _axisymmetric_center_metrics(
    jones_stack: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
    phi: np.ndarray,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Return first-harmonic and complete angular residuals."""

    stack = _as_jones_stack(jones_stack)
    sampled = _bilinear_sample_jones_stack(
        stack,
        center_px=center_px,
        pixel_size_um=float(pixel_size_um),
        radii_um=np.asarray(radii_um, dtype=np.float64),
        phi=np.asarray(phi, dtype=np.float64),
    )
    corotating = _to_corotating(sampled, np.asarray(phi, dtype=np.float64))
    phase = np.exp(-1j * np.asarray(phi, dtype=np.float64))
    plus = np.mean(corotating * phase[None, None, :, None, None], axis=2)
    minus = np.mean(corotating * np.conj(phase)[None, None, :, None, None], axis=2)
    radial_mean = np.mean(corotating, axis=2)
    angular_residual = corotating - radial_mean[:, :, None]

    radius = np.asarray(radii_um, dtype=np.float64)
    radial_weight = radius * _quadrature_step(radius)
    numerator = np.sum(
        (np.abs(plus) ** 2 + np.abs(minus) ** 2)
        * radial_weight[None, :, None, None],
        axis=(1, 2, 3),
    )
    radial_average = np.sum(
        radial_mean * radial_weight[None, :, None, None], axis=1
    ) / np.sum(radial_weight)
    radial_signal = radial_mean - radial_average[:, None, :, :]
    denominator = np.sum(
        np.abs(radial_signal) ** 2 * radial_weight[None, :, None, None],
        axis=(1, 2, 3),
    )
    record_objective = numerator / np.maximum(denominator, 1.0e-300)
    all_numerator = np.sum(
        np.mean(np.abs(angular_residual) ** 2, axis=2)
        * radial_weight[None, :, None, None],
        axis=(1, 2, 3),
    )
    all_record_objective = all_numerator / np.maximum(denominator, 1.0e-300)
    return (
        float(np.mean(record_objective)),
        record_objective,
        float(np.mean(all_record_objective)),
        all_record_objective,
    )


def crossed_intensity_harmonics(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radial_centers_um: np.ndarray,
    radial_sigma_um: float,
    modes: tuple[int, ...] = (0, 2, 4),
) -> IntensityHarmonicProfiles:
    """Return soft-annulus harmonics of |J_yx|^2 for x input and y analysis."""

    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(
            f"Expected Jones field with shape (x, y, 2, 2), got {matrix.shape!r}"
        )
    intensity = np.abs(matrix[..., 1, 0]) ** 2
    signed_modes = np.asarray(modes, dtype=np.int64)
    angular = soft_annular_harmonics(
        intensity,
        center_xy_px=center_px,
        pixel_size_xy_um=pixel_size_um,
        radial_centers_um=np.asarray(radial_centers_um, dtype=np.float64),
        radial_sigma_um=float(radial_sigma_um),
        modes=tuple(int(mode) for mode in signed_modes),
        axis_order="xy",
    )
    coefficients = angular.coefficients
    zero_indices = np.flatnonzero(signed_modes == 0)
    if zero_indices.size != 1:
        raise ValueError("modes must contain zero exactly once")
    mean_intensity = np.real(coefficients[int(zero_indices[0])])
    visibility = 2.0 * np.abs(coefficients) / np.maximum(
        mean_intensity[None, :], 1.0e-300
    )
    visibility[int(zero_indices[0])] = 1.0
    return IntensityHarmonicProfiles(
        radius_um=np.asarray(radial_centers_um, dtype=np.float64),
        modes=signed_modes,
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        visibility=np.asarray(visibility, dtype=np.float64),
    )


def soft_annular_harmonics(
    field: np.ndarray,
    *,
    center_xy_px: tuple[float, float],
    pixel_size_xy_um: float | tuple[float, float],
    radial_centers_um: np.ndarray,
    radial_sigma_um: float,
    modes: tuple[int, ...] = (0, 2, 4),
    axis_order: str = "xy",
) -> AngularHarmonicProfiles:
    """Return the canonical Gaussian-annulus harmonics of a scalar field."""

    values = np.asarray(field)
    if values.ndim != 2:
        raise ValueError("field must be a two-dimensional scalar array")
    signed_modes = np.asarray(modes, dtype=np.int64)
    if signed_modes.ndim != 1 or signed_modes.size == 0:
        raise ValueError("modes must be a nonempty one-dimensional sequence")
    if np.unique(signed_modes).size != signed_modes.size:
        raise ValueError("modes must not contain duplicates")
    centers = np.asarray(radial_centers_um, dtype=np.float64)
    if (
        centers.ndim != 1
        or centers.size == 0
        or np.any(centers < 0.0)
        or (centers.size > 1 and np.any(np.diff(centers) <= 0.0))
    ):
        raise ValueError("radial_centers_um must be nonnegative and increasing")
    _validate_positive("radial_sigma_um", radial_sigma_um)
    x_um, y_um = cartesian_coordinate_grids(
        values.shape,
        center_xy_px=center_xy_px,
        pixel_size_xy_um=pixel_size_xy_um,
        axis_order=axis_order,
    )
    coefficients = soft_annular_angular_profiles(
        values,
        np.hypot(x_um, y_um),
        np.arctan2(y_um, x_um),
        signed_modes=signed_modes,
        radial_centers_um=centers,
        radial_sigma_um=float(radial_sigma_um),
    )
    return AngularHarmonicProfiles(
        radius_um=centers,
        modes=signed_modes,
        coefficients=np.asarray(coefficients, dtype=np.complex128),
    )


def intensity_harmonics_on_circles(
    intensity_yx: np.ndarray,
    *,
    center_xy_px: tuple[float, float],
    pixel_size_um: float,
    radius_um: np.ndarray,
    modes: tuple[int, ...] = (0, 2, 4),
    angular_samples: int = 720,
    interpolation: str = "cubic",
) -> IntensityHarmonicProfiles:
    """Sample exact-radius angular harmonics of a scalar image.

    Unlike :func:`crossed_intensity_harmonics`, this diagnostic applies no
    radial window or smoothing.  Its coefficient convention is
    ``mean(I(phi) * exp(-1j * m * phi))``.
    """

    image = np.asarray(intensity_yx, dtype=np.float64)
    radius = np.asarray(radius_um, dtype=np.float64)
    signed_modes = np.asarray(modes, dtype=np.int64)
    if image.ndim != 2:
        raise ValueError("intensity_yx must be a two-dimensional array")
    if radius.ndim != 1 or np.any(radius < 0.0) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius_um must be nonnegative and strictly increasing")
    if pixel_size_um <= 0.0:
        raise ValueError("pixel_size_um must be positive")
    if angular_samples < 32:
        raise ValueError("angular_samples must be at least 32")

    grid = PolarSamplingGrid.uniform(
        center_xy_px=center_xy_px,
        pixel_size_xy_um=float(pixel_size_um),
        radius_um=radius,
        angular_samples=int(angular_samples),
        axis_order="yx",
        boundary="raise",
    )
    samples = sample_polar(image, grid, interpolation=interpolation)
    coefficients = angular_fourier(
        samples,
        signed_modes,
        phi_rad=grid.phi_rad,
        angular_axis=1,
    )
    zero_indices = np.flatnonzero(signed_modes == 0)
    if zero_indices.size != 1:
        raise ValueError("modes must contain zero exactly once")
    mean_intensity = np.real(coefficients[int(zero_indices[0])])
    visibility = 2.0 * np.abs(coefficients) / np.maximum(
        mean_intensity[None, :], 1.0e-300
    )
    visibility[int(zero_indices[0])] = 1.0
    return IntensityHarmonicProfiles(
        radius_um=radius,
        modes=signed_modes,
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        visibility=np.asarray(visibility, dtype=np.float64),
    )


def integrated_harmonic_visibility(
    profiles: IntensityHarmonicProfiles,
    *,
    mode: int,
    min_radius_um: float,
    max_radius_um: float,
) -> float:
    """Return the I0^2 rho-weighted RMS normalized angular harmonic."""

    indices = np.flatnonzero(profiles.modes == int(mode))
    zero_indices = np.flatnonzero(profiles.modes == 0)
    if indices.size != 1 or zero_indices.size != 1:
        raise ValueError("Requested mode and zero must each occur exactly once")
    selected = (profiles.radius_um >= float(min_radius_um)) & (
        profiles.radius_um <= float(max_radius_um)
    )
    if np.count_nonzero(selected) < 2:
        raise ValueError("Integration interval contains fewer than two samples")
    radius = profiles.radius_um[selected]
    coefficient = profiles.coefficients[int(indices[0]), selected]
    mean = profiles.coefficients[int(zero_indices[0]), selected]
    numerator = np.trapezoid(np.abs(coefficient) ** 2 * radius, radius)
    denominator = np.trapezoid(np.abs(mean) ** 2 * radius, radius)
    return float(2.0 * np.sqrt(numerator / max(denominator, 1.0e-300)))


def polar_coordinate_grids(
    shape: tuple[int, int],
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(int(shape[0]), dtype=np.float64) - float(center_px[0])) * float(
        pixel_size_um
    )
    y = (np.arange(int(shape[1]), dtype=np.float64) - float(center_px[1])) * float(
        pixel_size_um
    )
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.hypot(xx, yy), np.arctan2(yy, xx)


def _center_objective_grid(
    stack: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    pixel_size_um: float,
    radii_um: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    centers = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    objective = np.empty(centers.shape[0], dtype=np.float64)
    all_objective = np.empty_like(objective)
    record = np.empty((stack.shape[0], centers.shape[0]), dtype=np.float64)
    all_record = np.empty_like(record)
    batch_size = 16
    for start in range(0, centers.shape[0], batch_size):
        stop = min(start + batch_size, centers.shape[0])
        first, per_record, complete, complete_per_record = _center_objective_batch(
            stack,
            centers_px=centers[start:stop],
            pixel_size_um=pixel_size_um,
            radii_um=radii_um,
            phi=phi,
        )
        objective[start:stop] = first
        all_objective[start:stop] = complete
        record[:, start:stop] = per_record
        all_record[:, start:stop] = complete_per_record
    grid_shape = (x_values.size, y_values.size)
    return (
        objective.reshape(grid_shape),
        record.reshape((stack.shape[0], *grid_shape)),
        all_objective.reshape(grid_shape),
        all_record.reshape((stack.shape[0], *grid_shape)),
    )


def _center_objective_batch(
    stack: np.ndarray,
    *,
    centers_px: np.ndarray,
    pixel_size_um: float,
    radii_um: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radial_px = radii_um[None, :, None] / float(pixel_size_um)
    x = centers_px[:, 0, None, None] + radial_px * np.cos(phi)[None, None, :]
    y = centers_px[:, 1, None, None] + radial_px * np.sin(phi)[None, None, :]
    if (
        np.min(x) < 0.0
        or np.max(x) > stack.shape[1] - 1
        or np.min(y) < 0.0
        or np.max(y) > stack.shape[2] - 1
    ):
        raise ValueError("Polar sampling domain extends outside the Jones field")
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, stack.shape[1] - 1)
    y1 = np.minimum(y0 + 1, stack.shape[2] - 1)
    wx = (x - x0)[None, ..., None, None]
    wy = (y - y0)[None, ..., None, None]
    sampled = (
        stack[:, x0, y0] * (1.0 - wx) * (1.0 - wy)
        + stack[:, x1, y0] * wx * (1.0 - wy)
        + stack[:, x0, y1] * (1.0 - wx) * wy
        + stack[:, x1, y1] * wx * wy
    )
    corotating = _to_corotating_batch(sampled, phi)
    phase = np.exp(-1j * phi)[None, None, None, :, None, None]
    plus = np.mean(corotating * phase, axis=3)
    minus = np.mean(corotating * np.conj(phase), axis=3)
    radial_mean = np.mean(corotating, axis=3)
    angular_residual = corotating - radial_mean[:, :, :, None]
    radial_weight = radii_um * _quadrature_step(radii_um)
    weight = radial_weight[None, None, :, None, None]
    numerator = np.sum((np.abs(plus) ** 2 + np.abs(minus) ** 2) * weight, axis=(2, 3, 4))
    radial_average = np.sum(radial_mean * weight, axis=2) / np.sum(radial_weight)
    radial_signal = radial_mean - radial_average[:, :, None]
    denominator = np.sum(np.abs(radial_signal) ** 2 * weight, axis=(2, 3, 4))
    first_record = numerator / np.maximum(denominator, 1.0e-300)
    all_numerator = np.sum(
        np.mean(np.abs(angular_residual) ** 2, axis=3) * weight,
        axis=(2, 3, 4),
    )
    all_record = all_numerator / np.maximum(denominator, 1.0e-300)
    return np.mean(first_record, axis=0), first_record, np.mean(all_record, axis=0), all_record


def _bilinear_sample_jones_stack(
    stack: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    radial_px = radii_um[:, None] / float(pixel_size_um)
    x = float(center_px[0]) + radial_px * np.cos(phi)[None, :]
    y = float(center_px[1]) + radial_px * np.sin(phi)[None, :]
    if (
        np.min(x) < 0.0
        or np.max(x) > stack.shape[1] - 1
        or np.min(y) < 0.0
        or np.max(y) > stack.shape[2] - 1
    ):
        raise ValueError("Polar sampling domain extends outside the Jones field")
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, stack.shape[1] - 1)
    y1 = np.minimum(y0 + 1, stack.shape[2] - 1)
    wx = (x - x0)[None, :, :, None, None]
    wy = (y - y0)[None, :, :, None, None]
    return (
        stack[:, x0, y0] * (1.0 - wx) * (1.0 - wy)
        + stack[:, x1, y0] * wx * (1.0 - wy)
        + stack[:, x0, y1] * (1.0 - wx) * wy
        + stack[:, x1, y1] * wx * wy
    )


def _to_corotating(samples: np.ndarray, phi: np.ndarray) -> np.ndarray:
    c = np.cos(phi)[None, None, :, None]
    s = np.sin(phi)[None, None, :, None]
    j00 = samples[..., 0, 0]
    j01 = samples[..., 0, 1]
    j10 = samples[..., 1, 0]
    j11 = samples[..., 1, 1]
    t00 = c[..., 0] * j00 + s[..., 0] * j10
    t01 = c[..., 0] * j01 + s[..., 0] * j11
    t10 = -s[..., 0] * j00 + c[..., 0] * j10
    t11 = -s[..., 0] * j01 + c[..., 0] * j11
    result = np.empty_like(samples, dtype=np.complex128)
    result[..., 0, 0] = t00 * c[..., 0] + t01 * s[..., 0]
    result[..., 0, 1] = -t00 * s[..., 0] + t01 * c[..., 0]
    result[..., 1, 0] = t10 * c[..., 0] + t11 * s[..., 0]
    result[..., 1, 1] = -t10 * s[..., 0] + t11 * c[..., 0]
    return result


def _to_corotating_batch(samples: np.ndarray, phi: np.ndarray) -> np.ndarray:
    c = np.cos(phi)[None, None, None, :, None, None]
    s = np.sin(phi)[None, None, None, :, None, None]
    j00 = samples[..., 0, 0][..., None, None]
    j01 = samples[..., 0, 1][..., None, None]
    j10 = samples[..., 1, 0][..., None, None]
    j11 = samples[..., 1, 1][..., None, None]
    t00 = c * j00 + s * j10
    t01 = c * j01 + s * j11
    t10 = -s * j00 + c * j10
    t11 = -s * j01 + c * j11
    result = np.empty(samples.shape, dtype=np.complex128)
    result[..., 0, 0] = (t00 * c + t01 * s)[..., 0, 0]
    result[..., 0, 1] = (-t00 * s + t01 * c)[..., 0, 0]
    result[..., 1, 0] = (t10 * c + t11 * s)[..., 0, 0]
    result[..., 1, 1] = (-t10 * s + t11 * c)[..., 0, 0]
    return result


def _as_jones_stack(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 4 and array.shape[-2:] == (2, 2):
        array = array[None, ...]
    if array.ndim != 5 or array.shape[-2:] != (2, 2):
        raise ValueError(
            f"Expected Jones stack with shape (record, x, y, 2, 2), got {array.shape!r}"
        )
    if np.issubdtype(array.dtype, np.complexfloating):
        return array
    return np.asarray(array, dtype=np.complex128)


def _symmetric_grid(radius: float, step: float) -> np.ndarray:
    _validate_nonnegative("radius", radius)
    _validate_positive("step", step)
    count = int(np.floor(float(radius) / float(step) + 1.0e-12))
    return np.arange(-count, count + 1, dtype=np.float64) * float(step)


def _index_on_boundary(index: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return any(value in (0, size - 1) for value, size in zip(index, shape, strict=True))


def _quadrature_step(radius: np.ndarray) -> float:
    if radius.size < 2:
        raise ValueError("At least two radial samples are required")
    return float(np.median(np.diff(radius)))


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def _validate_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value!r}")
