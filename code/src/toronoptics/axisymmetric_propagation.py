"""Radial Jones channels and homogeneous one-way propagation.

The rotationally covariant Jones field is represented either by the generator
coefficients ``(a, b, c, d)`` or by the circular-polarization channels
``(A_plus, A_minus, B_plus, B_minus)``.  The latter carry angular orders
``(0, 0, +2, -2)`` and are the primary propagation variables used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import splu
from scipy.special import jv

from toronoptics.axisymmetric_basis import (
    coefficients_from_jones,
    jones_from_coefficients,
)
from toronoptics.polar_fourier import PolarSamplingGrid, sample_polar


ABCD_LABELS = ("a", "b", "c", "d")
CIRCULAR_LABELS = ("A_plus", "A_minus", "B_plus", "B_minus")
CIRCULAR_ORDERS = np.asarray((0, 0, 2, -2), dtype=np.int16)

_CIRCULAR_TO_LINEAR = np.asarray(
    ((1.0, 1.0), (1.0j, -1.0j)), dtype=np.complex128
) / np.sqrt(2.0)


@dataclass(frozen=True)
class AxisymmetricProfile:
    radius_um: np.ndarray
    abcd: np.ndarray
    circular: np.ndarray
    circular_direct: np.ndarray
    angular_rms: np.ndarray
    reconstruction_rms: np.ndarray
    signal_rms: np.ndarray
    circular_consistency_rms: np.ndarray
    background_abcd: np.ndarray
    background_jones: np.ndarray
    center_px: tuple[float, float]
    angular_samples: int
    interpolation: str = "cubic"


@dataclass(frozen=True)
class HomogeneousPropagation:
    z_um: np.ndarray
    radius_um: np.ndarray
    circular: np.ndarray
    method: str
    wavelength_um: float
    medium_index: float
    numerical_aperture: float | None

    @property
    def abcd(self) -> np.ndarray:
        return abcd_from_circular(self.circular)


def circular_jones(jones: np.ndarray) -> np.ndarray:
    """Return a Jones field in the ``(e_plus, e_minus)`` basis."""

    matrix = np.asarray(jones, dtype=np.complex128)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("jones must end in a 2 by 2 matrix")
    return np.einsum(
        "ia,...ij,jb->...ab",
        _CIRCULAR_TO_LINEAR.conj(),
        matrix,
        _CIRCULAR_TO_LINEAR,
        optimize=True,
    )


def circular_from_abcd(abcd: np.ndarray) -> np.ndarray:
    """Convert ``(a,b,c,d)`` to ``(A_plus,A_minus,B_plus,B_minus)``."""

    values = np.asarray(abcd, dtype=np.complex128)
    if values.shape[-1] != 4:
        raise ValueError("abcd must end in four coefficients")
    a, b, c, d = np.moveaxis(values, -1, 0)
    return np.stack((a - 1j * b, a + 1j * b, c + 1j * d, c - 1j * d), axis=-1)


def abcd_from_circular(circular: np.ndarray) -> np.ndarray:
    """Convert ``(A_plus,A_minus,B_plus,B_minus)`` to ``(a,b,c,d)``."""

    values = np.asarray(circular, dtype=np.complex128)
    if values.shape[-1] != 4:
        raise ValueError("circular must end in four coefficients")
    a_plus, a_minus, b_plus, b_minus = np.moveaxis(values, -1, 0)
    return np.stack(
        (
            0.5 * (a_plus + a_minus),
            (a_minus - a_plus) / (2j),
            0.5 * (b_plus + b_minus),
            (b_plus - b_minus) / (2j),
        ),
        axis=-1,
    )


def extract_axisymmetric_profile(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radius_um: np.ndarray,
    angular_samples: int = 720,
    background_interval_um: tuple[float, float] | None = None,
    interpolation: str = "cubic",
) -> AxisymmetricProfile:
    """Project a spatial Jones map onto its rotationally covariant part.

    The field is sampled on subpixel rings.  The returned angular RMS is the
    pointwise scatter of the co-rotating coefficients, while the reconstruction
    RMS measures the Jones content discarded by the radial projection.
    """

    matrix = np.asarray(jones, dtype=np.complex128)
    radius = np.asarray(radius_um, dtype=np.float64)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError("jones must have shape (x, y, 2, 2)")
    if radius.ndim != 1 or radius.size < 2 or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius_um must be a strictly increasing one-dimensional grid")
    if pixel_size_um <= 0.0:
        raise ValueError("pixel_size_um must be positive")
    if angular_samples < 32:
        raise ValueError("angular_samples must be at least 32")

    grid = PolarSamplingGrid.uniform(
        center_xy_px=center_px,
        pixel_size_xy_um=float(pixel_size_um),
        radius_um=radius,
        angular_samples=int(angular_samples),
        axis_order="xy",
        boundary="raise",
    )
    phi = grid.phi_rad
    sampled = sample_polar(matrix, grid, interpolation=interpolation)

    angle = np.broadcast_to(phi[None, :], sampled.shape[:2])
    pointwise_abcd = coefficients_from_jones(sampled, angle)
    abcd = np.mean(pointwise_abcd, axis=1)
    angular_rms = np.sqrt(np.mean(np.abs(pointwise_abcd - abcd[:, None]) ** 2, axis=1))

    reconstructed = jones_from_coefficients(
        np.broadcast_to(abcd[:, None, :], pointwise_abcd.shape), angle
    )
    reconstruction_rms = np.sqrt(
        np.mean(np.sum(np.abs(sampled - reconstructed) ** 2, axis=(-2, -1)), axis=1)
    )

    if background_interval_um is None:
        width = max(0.12 * (radius[-1] - radius[0]), 4.0 * np.median(np.diff(radius)))
        background_interval_um = (float(radius[-1] - width), float(radius[-1]))
    background_mask = (radius >= float(background_interval_um[0])) & (
        radius <= float(background_interval_um[1])
    )
    if np.count_nonzero(background_mask) < 2:
        raise ValueError("background_interval_um contains fewer than two radial samples")
    background_radius = radius[background_mask]
    background_weight = background_radius * np.gradient(background_radius)
    outer_abcd = np.sum(
        abcd[background_mask] * background_weight[:, None],
        axis=0,
    ) / np.sum(background_weight)
    # A spatially constant background belongs only to the order-zero A
    # channels.  Nonzero outer c,d values multiply exp(±2 i phi); retaining
    # them as a static Jones matrix is the same category error corrected in
    # the radial propagators below.
    background_abcd = np.asarray(outer_abcd, dtype=np.complex128)
    background_abcd[2:] = 0.0
    background_jones = jones_from_coefficients(
        background_abcd[None, :], np.asarray([0.0])
    )[0]
    signal_rms = np.sqrt(
        np.mean(
            np.sum(np.abs(sampled - background_jones[None, None]) ** 2, axis=(-2, -1)),
            axis=1,
        )
    )

    sampled_circular = circular_jones(sampled)
    circular_direct = np.stack(
        (
            np.mean(sampled_circular[..., 0, 0], axis=1),
            np.mean(sampled_circular[..., 1, 1], axis=1),
            np.mean(sampled_circular[..., 1, 0] * np.exp(-2j * phi)[None, :], axis=1),
            np.mean(sampled_circular[..., 0, 1] * np.exp(2j * phi)[None, :], axis=1),
        ),
        axis=-1,
    )
    circular = circular_from_abcd(abcd)
    circular_consistency_rms = np.sqrt(
        np.mean(np.abs(circular - circular_direct) ** 2, axis=-1)
    )
    return AxisymmetricProfile(
        radius_um=radius,
        abcd=abcd,
        circular=circular,
        circular_direct=circular_direct,
        angular_rms=angular_rms,
        reconstruction_rms=reconstruction_rms,
        signal_rms=signal_rms,
        circular_consistency_rms=circular_consistency_rms,
        background_abcd=background_abcd,
        background_jones=background_jones,
        center_px=(float(center_px[0]), float(center_px[1])),
        angular_samples=int(angular_samples),
        interpolation=str(interpolation),
    )


def normalize_abcd_by_background(
    abcd: np.ndarray, background_abcd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Right-normalize a radial Jones profile by its constant far field."""

    values = np.asarray(abcd, dtype=np.complex128)
    background = np.asarray(background_abcd, dtype=np.complex128)
    if values.shape[-1] != 4 or background.shape != (4,):
        raise ValueError("Expected abcd[...,4] and background_abcd[4]")
    phi = np.zeros(values.shape[:-1], dtype=np.float64)
    matrices = jones_from_coefficients(values, phi)
    background_matrix = jones_from_coefficients(background[None], np.asarray([0.0]))[0]
    condition = np.linalg.cond(background_matrix)
    if not np.isfinite(condition) or condition > 1.0e8:
        raise ValueError("Far-field Jones matrix is singular or ill-conditioned")
    normalized = matrices @ np.linalg.inv(background_matrix)
    return coefficients_from_jones(normalized, phi), background_matrix


def derived_radial_observables(
    abcd: np.ndarray,
    *,
    background_abcd: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return basis-invariant defects and crossed-channel harmonics."""

    values = np.asarray(abcd, dtype=np.complex128)
    if values.shape[-1] != 4:
        raise ValueError("abcd must end in four coefficients")
    if background_abcd is not None:
        values, _ = normalize_abcd_by_background(values, background_abcd)
    phi = np.zeros(values.shape[:-1], dtype=np.float64)
    matrices = jones_from_coefficients(values, phi)
    gram = np.swapaxes(matrices.conj(), -2, -1) @ matrices
    scalar_power = 0.5 * np.real(np.trace(gram, axis1=-2, axis2=-1))
    identity = np.eye(2, dtype=np.complex128)
    traceless = gram - scalar_power[..., None, None] * identity
    intensity_defect = (scalar_power - 1.0) ** 2
    polarization_defect = 0.5 * np.sum(np.abs(traceless) ** 2, axis=(-2, -1))
    output_gram = matrices @ np.swapaxes(matrices.conj(), -2, -1)
    normality_defect = 0.5 * np.sum(np.abs(gram - output_gram) ** 2, axis=(-2, -1))

    _a, b, c, d = np.moveaxis(values, -1, 0)
    h2 = np.real(np.conj(b) * d) - 1j * np.real(np.conj(b) * c)
    h4 = -0.25 * (c + 1j * d) * np.conj(c - 1j * d)
    crossed_mean = np.abs(b) ** 2 + 0.5 * (np.abs(c) ** 2 + np.abs(d) ** 2)
    return {
        "normalized_abcd": values,
        "scalar_power": scalar_power,
        "D_I": intensity_defect,
        "D_P": polarization_defect,
        "D_N": normality_defect,
        "crossed_H2": h2,
        "crossed_H4": h4,
        "crossed_mean_intensity": crossed_mean,
        "crossed_V2": 2.0 * np.abs(h2) / np.maximum(crossed_mean, 1.0e-300),
        "crossed_V4": 2.0 * np.abs(h4) / np.maximum(crossed_mean, 1.0e-300),
    }


def reflection_axis_observables(
    h2: np.ndarray,
    h4: np.ndarray,
    *,
    pure_mode_fraction: float = 1.0e-8,
    root_unit_tolerance: float = 2.0e-5,
    chunk_size: int = 8192,
) -> dict[str, np.ndarray]:
    """Return analytic mirror-line and rotational-C4 observables.

    The intensity is assumed to have angular part
    2 Re(H2 exp(2 i phi) + H4 exp(4 i phi)).

    axis_quality is the maximum normalized correlation with a reflected copy.
    fourfold_quality is one minus the squared normalized quarter-turn defect,
    equal to the H4 fraction abs(H4)**2 / (abs(H2)**2 + abs(H4)**2).  The
    returned defects and fidelities distinguish rotational C4 from mirror
    symmetry without radial averaging.
    """

    h2_values, h4_values = np.broadcast_arrays(
        np.asarray(h2, dtype=np.complex128),
        np.asarray(h4, dtype=np.complex128),
    )
    shape = h2_values.shape
    h2_flat = h2_values.ravel()
    h4_flat = h4_values.ravel()
    power2 = np.abs(h2_flat) ** 2
    power4 = np.abs(h4_flat) ** 2
    angular_power = power2 + power4
    active = angular_power > np.finfo(np.float64).tiny
    fraction2 = np.divide(
        power2,
        angular_power,
        out=np.zeros_like(power2),
        where=active,
    )
    fraction4 = np.divide(
        power4,
        angular_power,
        out=np.zeros_like(power4),
        where=active,
    )
    alpha2 = np.angle(h2_flat)
    alpha4 = np.angle(h4_flat)

    q_axis = np.zeros_like(angular_power)
    best_axis = np.zeros_like(angular_power)
    q_fourfold = fraction4.copy()
    fourfold_axis = -0.25 * alpha4

    pure2 = active & (fraction4 <= pure_mode_fraction)
    pure4 = active & (fraction2 <= pure_mode_fraction)
    q_axis[pure2 | pure4] = 1.0
    best_axis[pure2] = -0.5 * alpha2[pure2]
    best_axis[pure4] = -0.25 * alpha4[pure4]

    mixed_indices = np.flatnonzero(active & ~pure2 & ~pure4)
    for start in range(0, mixed_indices.size, chunk_size):
        indices = mixed_indices[start : start + chunk_size]
        normalized2 = fraction2[indices]
        normalized4 = fraction4[indices]
        phase_mismatch = alpha4[indices] - 2.0 * alpha2[indices]
        leading = 2.0 * normalized4 * np.exp(2.0j * phase_mismatch)

        companion = np.zeros((indices.size, 4, 4), dtype=np.complex128)
        companion[:, 0, 0] = -normalized2 / leading
        companion[:, 0, 2] = normalized2 / leading
        companion[:, 0, 3] = np.exp(-4.0j * phase_mismatch)
        companion[:, 1, 0] = 1.0
        companion[:, 2, 1] = 1.0
        companion[:, 3, 2] = 1.0
        roots = np.linalg.eigvals(companion)
        unit_error = np.abs(np.abs(roots) - 1.0)
        valid = unit_error <= root_unit_tolerance
        missing = ~np.any(valid, axis=1)
        if np.any(missing):
            nearest = np.argmin(unit_error[missing], axis=1)
            valid[missing] = False
            valid[np.flatnonzero(missing), nearest] = True

        root_angle = np.angle(roots)
        correlation = (
            normalized2[:, None] * np.cos(root_angle)
            + normalized4[:, None]
            * np.cos(2.0 * root_angle + 2.0 * phase_mismatch[:, None])
        )
        correlation[~valid] = -np.inf
        maximum = np.max(correlation, axis=1)

        candidate_axis = (
            0.25 * root_angle - 0.5 * alpha2[indices, None]
        )
        candidate_brightness = np.real(
            h2_flat[indices, None] * np.exp(2.0j * candidate_axis)
            + h4_flat[indices, None] * np.exp(4.0j * candidate_axis)
        )
        tied = valid & (
            correlation >= maximum[:, None] - 1.0e-10
        )
        candidate_brightness[~tied] = -np.inf
        selected = np.argmax(candidate_brightness, axis=1)
        selected_axis = candidate_axis[np.arange(indices.size), selected]

        perpendicular_axis = selected_axis + 0.5 * np.pi
        selected_brightness = np.real(
            h2_flat[indices] * np.exp(2.0j * selected_axis)
            + h4_flat[indices] * np.exp(4.0j * selected_axis)
        )
        perpendicular_brightness = np.real(
            h2_flat[indices] * np.exp(2.0j * perpendicular_axis)
            + h4_flat[indices] * np.exp(4.0j * perpendicular_axis)
        )
        selected_axis = np.where(
            perpendicular_brightness > selected_brightness,
            perpendicular_axis,
            selected_axis,
        )
        q_axis[indices] = np.clip(maximum, 0.0, 1.0)
        best_axis[indices] = selected_axis

    best_axis = 0.5 * np.angle(np.exp(2.0j * best_axis))
    fourfold_axis = 0.25 * np.angle(np.exp(4.0j * fourfold_axis))
    reduced_best_axis = np.angle(np.exp(2.0j * best_axis))
    reduced_fourfold_axis = np.angle(np.exp(4.0j * fourfold_axis))
    mirror_defect = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - q_axis)))
    c4_rotation_defect = np.sqrt(np.maximum(0.0, 1.0 - q_fourfold))
    mirror_fidelity = np.clip(1.0 - mirror_defect, 0.0, 1.0)
    c4_mirror_fidelity = np.minimum(
        mirror_fidelity,
        np.clip(1.0 - c4_rotation_defect, 0.0, 1.0),
    )
    reduced_mirror_family = np.angle(np.exp(4.0j * best_axis))
    reduced_c4_mirror_family = np.angle(np.exp(8.0j * fourfold_axis))
    return {
        "angular_power": angular_power.reshape(shape),
        "axis_quality": q_axis.reshape(shape),
        "fourfold_quality": q_fourfold.reshape(shape),
        "best_axis_rad": best_axis.reshape(shape),
        "fourfold_axis_rad": fourfold_axis.reshape(shape),
        "reduced_best_axis": reduced_best_axis.reshape(shape),
        "reduced_fourfold_axis": reduced_fourfold_axis.reshape(shape),
        "mirror_defect": mirror_defect.reshape(shape),
        "c4_rotation_defect": c4_rotation_defect.reshape(shape),
        "mirror_fidelity": mirror_fidelity.reshape(shape),
        "c4_mirror_fidelity": c4_mirror_fidelity.reshape(shape),
        "reduced_mirror_family": reduced_mirror_family.reshape(shape),
        "reduced_c4_mirror_family": reduced_c4_mirror_family.reshape(shape),
    }


class RadialCrankNicolsonPropagator:
    """Local finite-difference propagator for one cylindrical angular order."""

    def __init__(
        self,
        radius_um: np.ndarray,
        *,
        order: int,
        wavelength_um: float,
        medium_index: float = 1.0,
        approximation: str = "pade11",
    ) -> None:
        radius = np.asarray(radius_um, dtype=np.float64)
        if radius.ndim != 1 or radius.size < 5:
            raise ValueError("radius_um must contain at least five samples")
        spacing = np.diff(radius)
        if abs(radius[0]) > 1.0e-12 or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
            raise ValueError("radius_um must be a uniform grid starting at zero")
        if order not in {0, 2, -2}:
            raise ValueError("Only the axisymmetric orders 0 and +/-2 are supported")
        if approximation not in {"paraxial", "pade11"}:
            raise ValueError("approximation must be 'paraxial' or 'pade11'")
        if wavelength_um <= 0.0 or medium_index <= 0.0:
            raise ValueError("wavelength_um and medium_index must be positive")
        self.radius_um = radius
        self.order = abs(int(order))
        self.wavelength_um = float(wavelength_um)
        self.medium_index = float(medium_index)
        self.approximation = approximation
        self.k = 2.0 * np.pi * self.medium_index / self.wavelength_um
        self._indices, self._laplacian = _radial_laplacian(radius, self.order)

    def propagate(
        self,
        profiles: np.ndarray,
        z_um: np.ndarray,
        *,
        max_step_um: float = 0.25,
    ) -> np.ndarray:
        values = np.asarray(profiles, dtype=np.complex128)
        if values.shape[-1] != self.radius_um.size:
            raise ValueError("profiles must use the propagator radial grid")
        z = np.asarray(z_um, dtype=np.float64)
        if z.ndim != 1 or np.any(~np.isfinite(z)):
            raise ValueError("z_um must be a finite one-dimensional array")
        if max_step_um <= 0.0:
            raise ValueError("max_step_um must be positive")
        flat = values.reshape((-1, values.shape[-1]))
        result = np.empty((z.size, *flat.shape), dtype=np.complex128)
        for index, distance in enumerate(z):
            result[index] = self._propagate_distance(flat, float(distance), max_step_um)
        return result.reshape((z.size, *values.shape))

    def _propagate_distance(
        self, profiles: np.ndarray, distance_um: float, max_step_um: float
    ) -> np.ndarray:
        if abs(distance_um) <= 1.0e-15:
            return profiles.copy()
        steps = max(1, int(np.ceil(abs(distance_um) / max_step_um)))
        step = distance_um / steps
        laplacian = self._laplacian
        unit = eye(laplacian.shape[0], format="csc", dtype=np.complex128)
        if self.approximation == "paraxial":
            base = unit
        else:
            base = unit + laplacian / (4.0 * self.k**2)
        left = csc_matrix(base - 1j * step * laplacian / (4.0 * self.k))
        right = csc_matrix(base + 1j * step * laplacian / (4.0 * self.k))
        solve = splu(left)
        state = profiles[:, self._indices].T.copy()
        for _ in range(steps):
            state = solve.solve(right @ state)
        output = np.zeros_like(profiles)
        output[:, self._indices] = state.T
        if self.order == 0:
            output[:, -1] = 0.0
        return output


class HankelAngularSpectrumPropagator:
    """Exact scalar angular-spectrum propagation by continuous Hankel quadrature."""

    def __init__(
        self,
        *,
        order: int,
        wavelength_um: float,
        medium_index: float = 1.0,
        numerical_aperture: float | None = None,
        spectral_oversampling: float = 2.0,
    ) -> None:
        if order not in {0, 2, -2}:
            raise ValueError("Only the axisymmetric orders 0 and +/-2 are supported")
        if wavelength_um <= 0.0 or medium_index <= 0.0:
            raise ValueError("wavelength_um and medium_index must be positive")
        if numerical_aperture is not None and not 0.0 < numerical_aperture <= medium_index:
            raise ValueError("numerical_aperture must lie in (0, medium_index]")
        if spectral_oversampling < 1.0:
            raise ValueError("spectral_oversampling must be at least one")
        self.order = abs(int(order))
        self.wavelength_um = float(wavelength_um)
        self.medium_index = float(medium_index)
        self.numerical_aperture = (
            None if numerical_aperture is None else float(numerical_aperture)
        )
        self.spectral_oversampling = float(spectral_oversampling)
        self.k = 2.0 * np.pi * self.medium_index / self.wavelength_um

    def propagate(
        self,
        profiles: np.ndarray,
        radius_um: np.ndarray,
        z_um: np.ndarray,
    ) -> np.ndarray:
        radius = np.asarray(radius_um, dtype=np.float64)
        values = np.asarray(profiles, dtype=np.complex128)
        z = np.asarray(z_um, dtype=np.float64)
        if radius.ndim != 1 or values.shape[-1] != radius.size:
            raise ValueError("profiles must end on the supplied one-dimensional radial grid")
        if np.any(np.diff(radius) <= 0.0) or radius[0] < 0.0:
            raise ValueError("radius_um must be nonnegative and strictly increasing")
        flat = values.reshape((-1, radius.size))
        radial_weights = _quadrature_weights(radius)
        kappa_max = self.k
        if self.numerical_aperture is not None:
            kappa_max = 2.0 * np.pi * self.numerical_aperture / self.wavelength_um
        spectral_step = np.pi / (self.spectral_oversampling * radius[-1])
        spectral_count = max(65, int(np.ceil(kappa_max / spectral_step)) + 1)
        if spectral_count % 2 == 0:
            spectral_count += 1
        kappa = np.linspace(0.0, kappa_max, spectral_count)
        spectral_weights = _quadrature_weights(kappa)
        bessel = jv(self.order, np.outer(kappa, radius))
        transformed = (bessel * (radius * radial_weights)[None, :]) @ flat.T
        delta_kz = np.sqrt(np.maximum(self.k**2 - kappa**2, 0.0)) - self.k
        result = np.empty((z.size, flat.shape[0], radius.size), dtype=np.complex128)
        for z_index, distance in enumerate(z):
            if (
                abs(float(distance)) <= 1.0e-15
                and self.numerical_aperture is None
            ):
                result[z_index] = flat
                continue
            phased = transformed * np.exp(1j * float(distance) * delta_kz)[:, None]
            result[z_index] = (
                bessel.T @ (phased * (kappa * spectral_weights)[:, None])
            ).T
            if self.order:
                result[z_index, :, 0] = 0.0
        return result.reshape((z.size, *values.shape))


def propagate_axisymmetric_circular(
    circular: np.ndarray,
    radius_um: np.ndarray,
    z_um: np.ndarray,
    *,
    wavelength_um: float,
    medium_index: float = 1.0,
    background_circular: np.ndarray | None = None,
    method: str = "pade11",
    max_step_um: float = 0.25,
    extension_radius_um: float | None = None,
) -> HomogeneousPropagation:
    """Propagate all four circular channels with a local radial integrator.

    Only the two order-zero ``A`` channels can contain a spatially constant
    plane-wave background.  The order-two ``B`` channels are propagated in
    full: a constant radial ``B`` coefficient still multiplies
    ``exp(+/-2j*phi)`` and therefore is not a plane wave.
    """

    radius = np.asarray(radius_um, dtype=np.float64)
    values = np.asarray(circular, dtype=np.complex128)
    if values.shape != (radius.size, 4):
        raise ValueError("circular must have shape (radius, 4)")
    spacing = np.diff(radius)
    if radius[0] != 0.0 or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
        raise ValueError("radius_um must be a uniform grid starting at zero")
    if background_circular is None:
        background = np.mean(values[-max(3, radius.size // 20) :], axis=0)
    else:
        background = np.asarray(background_circular, dtype=np.complex128)
        if background.shape != (4,):
            raise ValueError("background_circular must contain four coefficients")
    if extension_radius_um is None:
        extension_radius_um = max(2.0 * radius[-1], radius[-1] + 40.0)
    extended_radius = np.arange(
        0.0,
        float(extension_radius_um) + 0.5 * spacing[0],
        spacing[0],
        dtype=np.float64,
    )
    extended, static_background = _extend_circular_profiles(
        values,
        radius,
        extended_radius,
        background,
    )
    propagated = np.empty((len(z_um), extended_radius.size, 4), dtype=np.complex128)
    for order, channels in ((0, (0, 1)), (2, (2, 3))):
        integrator = RadialCrankNicolsonPropagator(
            extended_radius,
            order=order,
            wavelength_um=wavelength_um,
            medium_index=medium_index,
            approximation=method,
        )
        channel_result = integrator.propagate(
            extended[:, channels].T, np.asarray(z_um), max_step_um=max_step_um
        )
        propagated[:, :, channels] = np.moveaxis(channel_result, 1, -1)
    propagated += static_background[None, None, :]
    propagated[:, 0, 2:] = 0.0
    return HomogeneousPropagation(
        z_um=np.asarray(z_um, dtype=np.float64),
        radius_um=extended_radius,
        circular=propagated,
        method=f"radial-crank-nicolson-{method}",
        wavelength_um=float(wavelength_um),
        medium_index=float(medium_index),
        numerical_aperture=None,
    )


def propagate_axisymmetric_circular_hankel(
    circular: np.ndarray,
    radius_um: np.ndarray,
    z_um: np.ndarray,
    *,
    wavelength_um: float,
    medium_index: float = 1.0,
    numerical_aperture: float | None = None,
    background_circular: np.ndarray | None = None,
    extension_radius_um: float | None = None,
    spectral_oversampling: float = 2.0,
) -> HomogeneousPropagation:
    """Propagate all circular channels with the continuous Hankel law.

    Only the two order-zero ``A`` channels can contain a spatially constant
    plane-wave background.  The order-two ``B`` channels are propagated in
    full: a constant radial ``B`` coefficient still multiplies
    ``exp(+/-2j*phi)`` and therefore is not a plane wave.
    """

    radius = np.asarray(radius_um, dtype=np.float64)
    values = np.asarray(circular, dtype=np.complex128)
    if values.shape != (radius.size, 4):
        raise ValueError("circular must have shape (radius, 4)")
    spacing = np.diff(radius)
    if radius[0] != 0.0 or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
        raise ValueError("radius_um must be a uniform grid starting at zero")
    if background_circular is None:
        background = np.mean(values[-max(3, radius.size // 20) :], axis=0)
    else:
        background = np.asarray(background_circular, dtype=np.complex128)
        if background.shape != (4,):
            raise ValueError("background_circular must contain four coefficients")
    if extension_radius_um is None:
        extension_radius_um = max(2.0 * radius[-1], radius[-1] + 40.0)
    extended_radius = np.arange(
        0.0,
        float(extension_radius_um) + 0.5 * spacing[0],
        spacing[0],
        dtype=np.float64,
    )
    extended, static_background = _extend_circular_profiles(
        values,
        radius,
        extended_radius,
        background,
    )
    propagated = np.empty((len(z_um), extended_radius.size, 4), dtype=np.complex128)
    for order, channels in ((0, (0, 1)), (2, (2, 3))):
        propagator = HankelAngularSpectrumPropagator(
            order=order,
            wavelength_um=wavelength_um,
            medium_index=medium_index,
            numerical_aperture=numerical_aperture,
            spectral_oversampling=spectral_oversampling,
        )
        channel_result = propagator.propagate(
            extended[:, channels].T, extended_radius, np.asarray(z_um)
        )
        propagated[:, :, channels] = np.moveaxis(channel_result, 1, -1)
    propagated += static_background[None, None, :]
    propagated[:, 0, 2:] = 0.0
    return HomogeneousPropagation(
        z_um=np.asarray(z_um, dtype=np.float64),
        radius_um=extended_radius,
        circular=propagated,
        method="continuous-hankel-angular-spectrum",
        wavelength_um=float(wavelength_um),
        medium_index=float(medium_index),
        numerical_aperture=(
            None if numerical_aperture is None else float(numerical_aperture)
        ),
    )


def _extend_circular_profiles(
    circular: np.ndarray,
    radius_um: np.ndarray,
    extended_radius_um: np.ndarray,
    background_circular: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return localized radial profiles and their true plane-wave background.

    The order-zero channels are represented as a localized perturbation plus
    their constant far field.  Order-two channels have no constant plane-wave
    part, irrespective of their outer-annulus mean, and are consequently kept
    intact.  All localized channels are continued from the measured boundary
    to zero with a raised-cosine taper.  This preserves every supplied sample
    while avoiding a discontinuous hard cut at the start of the padding.
    """

    values = np.asarray(circular, dtype=np.complex128)
    radius = np.asarray(radius_um, dtype=np.float64)
    extended_radius = np.asarray(extended_radius_um, dtype=np.float64)
    background = np.asarray(background_circular, dtype=np.complex128)
    if values.shape != (radius.size, 4):
        raise ValueError("circular must have shape (radius, 4)")
    if background.shape != (4,):
        raise ValueError("background_circular must contain four coefficients")
    if extended_radius.ndim != 1 or extended_radius.size < radius.size:
        raise ValueError("extended_radius_um must contain the source radial grid")
    if not np.allclose(
        extended_radius[: radius.size],
        radius,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("extended_radius_um must start with the source radial grid")
    if extended_radius[-1] < radius[-1]:
        raise ValueError("extended_radius_um must not truncate the source radial grid")

    static_background = np.zeros(4, dtype=np.complex128)
    static_background[:2] = background[:2]
    localized = values - static_background[None, :]

    # Regular cylindrical harmonics of nonzero order vanish at the origin.
    # Projection and interpolation may leave a tiny numerical remnant there.
    localized = localized.copy()
    localized[0, 2:] = 0.0

    extended = np.zeros((extended_radius.size, 4), dtype=np.complex128)
    extended[: radius.size] = localized
    padding = extended_radius[radius.size :]
    if padding.size:
        taper_coordinate = (padding - radius[-1]) / (
            extended_radius[-1] - radius[-1]
        )
        taper = 0.5 * (1.0 + np.cos(np.pi * taper_coordinate))
        extended[radius.size :] = taper[:, None] * localized[-1][None, :]
        extended[-1] = 0.0
    return extended, static_background


def _radial_laplacian(radius_um: np.ndarray, order: int) -> tuple[np.ndarray, csc_matrix]:
    radius = np.asarray(radius_um, dtype=np.float64)
    dr = float(radius[1] - radius[0])
    if order == 0:
        indices = np.arange(0, radius.size - 1, dtype=np.int64)
        size = indices.size
        lower = np.zeros(size - 1, dtype=np.float64)
        diagonal = np.zeros(size, dtype=np.float64)
        upper = np.zeros(size - 1, dtype=np.float64)
        diagonal[0] = -4.0 / dr**2
        upper[0] = 4.0 / dr**2
        start = 1
    else:
        indices = np.arange(1, radius.size - 1, dtype=np.int64)
        size = indices.size
        lower = np.zeros(size - 1, dtype=np.float64)
        diagonal = np.zeros(size, dtype=np.float64)
        upper = np.zeros(size - 1, dtype=np.float64)
        start = 0
    for local in range(start, size):
        global_index = int(indices[local])
        r = radius[global_index]
        diagonal[local] = -2.0 / dr**2 - order**2 / r**2
        if local > 0:
            lower[local - 1] = 1.0 / dr**2 - 1.0 / (2.0 * r * dr)
        if local < size - 1:
            upper[local] = 1.0 / dr**2 + 1.0 / (2.0 * r * dr)
    return indices, diags((lower, diagonal, upper), offsets=(-1, 0, 1), format="csc")


def _quadrature_weights(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    spacing = np.diff(values)
    if values.size % 2 == 1 and np.allclose(
        spacing, spacing[0], rtol=1.0e-10, atol=1.0e-14
    ):
        weights = np.full_like(values, 2.0)
        weights[1:-1:2] = 4.0
        weights[[0, -1]] = 1.0
        return weights * spacing[0] / 3.0
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights
