"""Spin-resolved projection of Jones fields onto rotational covariance.

For the circular basis used here,

``e_plus = (e_x + i e_y) / sqrt(2)`` and
``e_minus = (e_x - i e_y) / sqrt(2)``,

rotational covariance locks one angular order to each Jones channel:

``J_plus,plus -> 0``, ``J_minus,minus -> 0``,
``J_minus,plus -> +2``, and ``J_plus,minus -> -2``.

Keeping all four allowed orders somewhere in the matrix is not sufficient:
the order must occur in its corresponding spin channel.  The projector below
enforces this locking directly.  It is linear and contains no background
subtraction, radial taper, normalization, or intensity operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

from toronoptics.polar_fourier import (
    AxisOrder,
    InterpolationPolicy,
    PolarSamplingGrid,
    angular_fourier,
    cartesian_coordinate_grids,
    sample_polar,
)


SPIN_CHANNEL_LABELS = (
    "J_plus_plus",
    "J_minus_minus",
    "J_minus_plus",
    "J_plus_minus",
)
SPIN_ALLOWED_ORDERS = np.asarray((0, 0, 2, -2), dtype=np.int16)
_SPIN_CHANNEL_INDICES = ((0, 0), (1, 1), (1, 0), (0, 1))

_CIRCULAR_TO_LINEAR = np.asarray(
    ((1.0, 1.0), (1.0j, -1.0j)), dtype=np.complex128
) / np.sqrt(2.0)

RadialInterpolation = Literal["nearest", "linear", "cubic"]
RadialExtrapolation = Literal["nearest", "zero", "nan"]


@dataclass(frozen=True)
class CovariantProjectionDiagnostics:
    """Norm and closure diagnostics for an orthogonal polar projection."""

    radius_um: np.ndarray | None
    total_power_density: np.ndarray
    covariant_power_density: np.ndarray
    residual_power_density: np.ndarray
    orthogonality_density: np.ndarray
    channel_total_power: np.ndarray
    channel_covariant_power: np.ndarray
    channel_residual_power: np.ndarray
    channel_residual_fraction: np.ndarray
    total_power: float
    covariant_power: float
    residual_power: float
    orthogonality: complex
    relative_residual_norm: float
    covariant_power_fraction: float
    relative_closure_error: float
    relative_orthogonality: float


@dataclass(frozen=True)
class SpinResolvedAngularSpectrum:
    """Angular coefficients for the four circular Jones channels."""

    modes: np.ndarray
    coefficients: np.ndarray
    channel_labels: tuple[str, ...] = SPIN_CHANNEL_LABELS
    allowed_orders: np.ndarray = field(
        default_factory=lambda: SPIN_ALLOWED_ORDERS.copy()
    )

    def coefficient(self, channel: str | int, mode: int) -> np.ndarray:
        """Return one radial coefficient by channel label/index and order."""

        channel_index = (
            self.channel_labels.index(channel) if isinstance(channel, str) else int(channel)
        )
        mode_indices = np.flatnonzero(self.modes == int(mode))
        if mode_indices.size != 1:
            raise ValueError(f"mode {mode} does not occur exactly once")
        return self.coefficients[int(mode_indices[0]), :, channel_index]


@dataclass(frozen=True)
class PolarCovariantJonesProjection:
    """Exact discrete projection on a uniformly sampled polar grid."""

    sampled_jones: np.ndarray
    covariant_jones: np.ndarray
    residual_jones: np.ndarray
    sampled_circular: np.ndarray
    covariant_circular: np.ndarray
    residual_circular: np.ndarray
    radial_channels: np.ndarray
    phi_rad: np.ndarray
    radius_um: np.ndarray | None
    diagnostics: CovariantProjectionDiagnostics
    channel_labels: tuple[str, ...] = SPIN_CHANNEL_LABELS
    allowed_orders: np.ndarray = field(
        default_factory=lambda: SPIN_ALLOWED_ORDERS.copy()
    )


@dataclass(frozen=True)
class CartesianCovariantJonesProjection:
    """Polar projection reconstructed on the input Cartesian grid."""

    covariant_jones: np.ndarray
    residual_jones: np.ndarray
    valid_disk: np.ndarray
    polar: PolarCovariantJonesProjection
    grid: PolarSamplingGrid
    radial_interpolation: str
    radial_extrapolation: str

    @property
    def diagnostics(self) -> CovariantProjectionDiagnostics:
        return self.polar.diagnostics


@dataclass(frozen=True)
class IntensityHarmonicDecomposition:
    r"""Exact nonlinear decomposition for ``u = u_c + u_r``.

    The four coefficient arrays satisfy, to rounding precision,

    ``total = cc + cr + rr``,

    where ``cc`` is the transform of ``|u_c|^2``, ``cr`` that of
    ``2 Re(u_c u_r*)``, and ``rr`` that of ``|u_r|^2``.
    """

    modes: np.ndarray
    total: np.ndarray
    cc: np.ndarray
    cr: np.ndarray
    rr: np.ndarray
    total_intensity: np.ndarray
    cc_intensity: np.ndarray
    cr_intensity: np.ndarray
    rr_intensity: np.ndarray
    angular_axis: int

    @property
    def closure(self) -> np.ndarray:
        """Return coefficient-level ``total - cc - cr - rr``."""

        return self.total - self.cc - self.cr - self.rr

    @property
    def relative_closure_error(self) -> float:
        scale = max(float(np.linalg.norm(self.total.ravel())), np.finfo(float).tiny)
        return float(np.linalg.norm(self.closure.ravel()) / scale)


def linear_to_circular_jones(jones: np.ndarray) -> np.ndarray:
    """Transform Jones matrices from the linear to the circular basis."""

    matrix = _as_jones(jones)
    return np.einsum(
        "ia,...ij,jb->...ab",
        _CIRCULAR_TO_LINEAR.conj(),
        matrix,
        _CIRCULAR_TO_LINEAR,
        optimize=True,
    )


def circular_to_linear_jones(jones_circular: np.ndarray) -> np.ndarray:
    """Transform Jones matrices from the circular to the linear basis."""

    matrix = _as_jones(jones_circular)
    return np.einsum(
        "ia,...ab,jb->...ij",
        _CIRCULAR_TO_LINEAR,
        matrix,
        _CIRCULAR_TO_LINEAR.conj(),
        optimize=True,
    )


def covariance_rotation_difference(
    jones: np.ndarray,
    *,
    quarter_turns: int = 1,
) -> np.ndarray:
    r"""Return the exact-grid residual of a discrete covariance rotation.

    Jones arrays in this project use spatial-axis order ``(x, y)``.  On a
    square grid centered on its middle pixel, ``np.rot90(..., k=-q)`` samples
    ``J(R_q r)`` exactly, without interpolation, for an integer number ``q``
    of positive physical quarter turns.  Rotational covariance requires

    ``J(R_q r) = R_q J(r) R_q^{-1}``.

    The returned array is the left-hand side minus the right-hand side.
    """

    matrix = _as_jones(jones)
    if matrix.ndim != 4:
        raise ValueError("jones must have shape (x, y, 2, 2)")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("exact quarter-turn sampling requires a square grid")
    turns = int(quarter_turns)
    if turns != quarter_turns:
        raise ValueError("quarter_turns must be an integer")
    reduced_turns = turns % 4
    spatially_rotated = np.rot90(
        matrix,
        k=-reduced_turns,
        axes=(0, 1),
    )
    angle = 0.5 * np.pi * reduced_turns
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        ),
        dtype=np.float64,
    )
    polarization_rotated = np.einsum(
        "ia,...ab,jb->...ij",
        rotation,
        matrix,
        rotation,
        optimize=True,
    )
    return np.asarray(
        spatially_rotated - polarization_rotated,
        dtype=np.complex128,
    )


def normality_defect(jones: np.ndarray) -> np.ndarray:
    r"""Return ``J^\dagger J-JJ^\dagger`` pointwise."""

    matrix = _as_jones(jones)
    adjoint = np.swapaxes(matrix.conj(), -2, -1)
    return np.asarray(
        adjoint @ matrix - matrix @ adjoint,
        dtype=np.complex128,
    )


def normality_defect_density(jones: np.ndarray) -> np.ndarray:
    r"""Return ``||J^\dagger J-JJ^\dagger||_F^2/2`` pointwise."""

    defect = normality_defect(jones)
    return np.asarray(
        0.5 * np.sum(np.abs(defect) ** 2, axis=(-2, -1)),
        dtype=np.float64,
    )


def normality_order_two_field(
    jones: np.ndarray,
    phi_rad: np.ndarray,
) -> np.ndarray:
    r"""Contract the pointwise normality defect with its order-two basis.

    For an exactly rotationally covariant response the returned field is
    independent of azimuth and equals the complex crossed-intensity
    coefficient ``H2``:

    ``tr[(J^\dagger J-JJ^\dagger)(Sigma1+i Sigma2)]/8``.
    """

    matrix = _as_jones(jones)
    angle = np.broadcast_to(
        np.asarray(phi_rad, dtype=np.float64),
        matrix.shape[:-2],
    )
    c2 = np.cos(2.0 * angle)
    s2 = np.sin(2.0 * angle)
    sigma = np.empty(angle.shape + (2, 2), dtype=np.complex128)
    sigma[..., 0, 0] = c2 - 1j * s2
    sigma[..., 0, 1] = s2 + 1j * c2
    sigma[..., 1, 0] = s2 + 1j * c2
    sigma[..., 1, 1] = -c2 + 1j * s2
    defect = normality_defect(matrix)
    return np.asarray(
        np.einsum("...ij,...ji->...", defect, sigma, optimize=True) / 8.0,
        dtype=np.complex128,
    )


def rotationally_invariant_matrix_part(jones: np.ndarray) -> np.ndarray:
    r"""Return the ``a I+b K`` part invariant under in-plane conjugation."""

    matrix = _as_jones(jones)
    a = 0.5 * (matrix[..., 0, 0] + matrix[..., 1, 1])
    b = 0.5 * (matrix[..., 1, 0] - matrix[..., 0, 1])
    invariant = np.empty_like(matrix, dtype=np.complex128)
    invariant[..., 0, 0] = a
    invariant[..., 0, 1] = -b
    invariant[..., 1, 0] = b
    invariant[..., 1, 1] = a
    return invariant


def spin_resolved_angular_spectrum(
    sampled_jones: np.ndarray,
    modes: np.ndarray | tuple[int, ...],
    *,
    phi_rad: np.ndarray,
) -> SpinResolvedAngularSpectrum:
    """Return angular spectra without conflating the four spin channels.

    ``sampled_jones`` must have shape ``(radius, phi, 2, 2)``.  Coefficients
    have shape ``(mode, radius, channel)``.
    """

    matrix = _as_polar_jones(sampled_jones, phi_rad)
    circular = linear_to_circular_jones(matrix)
    channels = _circular_channel_stack(circular)
    signed_modes = np.asarray(modes, dtype=np.int64)
    coefficients = angular_fourier(
        channels,
        signed_modes,
        phi_rad=np.asarray(phi_rad, dtype=np.float64),
        angular_axis=1,
    )
    return SpinResolvedAngularSpectrum(
        modes=signed_modes,
        coefficients=np.asarray(coefficients, dtype=np.complex128),
    )


def project_covariant_polar(
    sampled_jones: np.ndarray,
    *,
    phi_rad: np.ndarray,
    radius_um: np.ndarray | None = None,
) -> PolarCovariantJonesProjection:
    """Apply the exact spin-order projector on a sampled polar field."""

    phi = np.asarray(phi_rad, dtype=np.float64)
    matrix = _as_polar_jones(sampled_jones, phi)
    radius = _optional_radius(radius_um, matrix.shape[0])
    circular = linear_to_circular_jones(matrix)
    channels = _circular_channel_stack(circular)
    phase = np.exp(
        -1j * SPIN_ALLOWED_ORDERS[None, None, :] * phi[None, :, None]
    )
    radial_channels = np.mean(channels * phase, axis=1)
    reconstructed_channels = radial_channels[:, None, :] * np.exp(
        1j * SPIN_ALLOWED_ORDERS[None, None, :] * phi[None, :, None]
    )
    covariant_circular = _circular_matrix_from_channels(reconstructed_channels)
    covariant = circular_to_linear_jones(covariant_circular)
    residual = matrix - covariant
    residual_circular = circular - covariant_circular
    diagnostics = _projection_diagnostics(
        matrix,
        covariant,
        residual,
        radius_um=radius,
    )
    return PolarCovariantJonesProjection(
        sampled_jones=np.asarray(matrix, dtype=np.complex128),
        covariant_jones=np.asarray(covariant, dtype=np.complex128),
        residual_jones=np.asarray(residual, dtype=np.complex128),
        sampled_circular=np.asarray(circular, dtype=np.complex128),
        covariant_circular=np.asarray(covariant_circular, dtype=np.complex128),
        residual_circular=np.asarray(residual_circular, dtype=np.complex128),
        radial_channels=np.asarray(radial_channels, dtype=np.complex128),
        phi_rad=phi,
        radius_um=radius,
        diagnostics=diagnostics,
    )


def project_covariant_jones(
    jones: np.ndarray,
    *,
    center_xy_px: tuple[float, float],
    pixel_size_xy_um: float | tuple[float, float],
    radius_um: np.ndarray,
    angular_samples: int = 720,
    interpolation: InterpolationPolicy | int = "cubic",
    radial_interpolation: RadialInterpolation = "cubic",
    radial_extrapolation: RadialExtrapolation = "nearest",
    axis_order: AxisOrder = "xy",
) -> CartesianCovariantJonesProjection:
    """Project a Cartesian Jones field and reconstruct it on the same grid.

    The projection is a composition of fixed linear sampling, angular
    projection, and radial interpolation operators.  No background or taper
    is inferred.  If a physical constant background is to be excluded from
    the reported norm, callers should subtract it before this function and
    add it to ``covariant_jones`` afterwards.

    ``radial_extrapolation='nearest'`` extends the outermost projected radial
    coefficient to Cartesian corners.  Diagnostics and ``valid_disk`` refer
    only to the genuinely sampled disk.  Use ``'nan'`` to make extrapolated
    pixels impossible to consume accidentally.
    """

    matrix = _as_jones(jones)
    if matrix.ndim != 4:
        raise ValueError("jones must have shape (axis0, axis1, 2, 2)")
    grid = PolarSamplingGrid.uniform(
        center_xy_px=center_xy_px,
        pixel_size_xy_um=pixel_size_xy_um,
        radius_um=np.asarray(radius_um, dtype=np.float64),
        angular_samples=int(angular_samples),
        axis_order=axis_order,
        boundary="raise",
    )
    sampled = sample_polar(matrix, grid, interpolation=interpolation)
    polar = project_covariant_polar(
        sampled,
        phi_rad=grid.phi_rad,
        radius_um=grid.radius_um,
    )
    covariant, valid_disk = reconstruct_covariant_jones(
        polar.radial_channels,
        radius_um=grid.radius_um,
        shape=matrix.shape[:2],
        center_xy_px=grid.center_xy_px,
        pixel_size_xy_um=grid.pixel_size_xy_um,
        axis_order=grid.axis_order,
        radial_interpolation=radial_interpolation,
        radial_extrapolation=radial_extrapolation,
    )
    residual = matrix - covariant
    return CartesianCovariantJonesProjection(
        covariant_jones=np.asarray(covariant, dtype=np.complex128),
        residual_jones=np.asarray(residual, dtype=np.complex128),
        valid_disk=valid_disk,
        polar=polar,
        grid=grid,
        radial_interpolation=str(radial_interpolation),
        radial_extrapolation=str(radial_extrapolation),
    )


def reconstruct_covariant_jones(
    radial_channels: np.ndarray,
    *,
    radius_um: np.ndarray,
    shape: tuple[int, int],
    center_xy_px: tuple[float, float],
    pixel_size_xy_um: float | tuple[float, float],
    axis_order: AxisOrder = "xy",
    radial_interpolation: RadialInterpolation = "cubic",
    radial_extrapolation: RadialExtrapolation = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct spin-locked radial channels on a Cartesian grid.

    This function is deliberately background agnostic.  It also returns the
    disk in which no radial extrapolation was needed.
    """

    channels = np.asarray(radial_channels, dtype=np.complex128)
    radius = np.asarray(radius_um, dtype=np.float64)
    if channels.shape != (radius.size, 4):
        raise ValueError("radial_channels must have shape (len(radius_um), 4)")
    if radius.ndim != 1 or radius.size < 2 or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius_um must be a strictly increasing 1-D grid")
    if radial_interpolation not in ("nearest", "linear", "cubic"):
        raise ValueError(f"Unsupported radial interpolation {radial_interpolation!r}")
    if radial_interpolation == "cubic" and radius.size < 4:
        raise ValueError("cubic radial interpolation requires at least four radii")
    if radial_extrapolation not in ("nearest", "zero", "nan"):
        raise ValueError(f"Unsupported radial extrapolation {radial_extrapolation!r}")

    x_um, y_um = cartesian_coordinate_grids(
        shape,
        center_xy_px=center_xy_px,
        pixel_size_xy_um=pixel_size_xy_um,
        axis_order=axis_order,
    )
    cartesian_radius = np.hypot(x_um, y_um)
    phi = np.arctan2(y_um, x_um)
    valid = (cartesian_radius >= radius[0]) & (cartesian_radius <= radius[-1])
    interpolation_kind = (
        "nearest" if radial_interpolation == "nearest" else radial_interpolation
    )
    if radial_extrapolation == "nearest":
        fill_value: tuple[np.ndarray, np.ndarray] | complex = (
            channels[0],
            channels[-1],
        )
    elif radial_extrapolation == "zero":
        fill_value = 0.0j
    else:
        fill_value = np.nan + 1j * np.nan
    interpolator = interp1d(
        radius,
        channels,
        axis=0,
        kind=interpolation_kind,
        bounds_error=False,
        fill_value=fill_value,
        assume_sorted=True,
    )
    radial = np.asarray(interpolator(cartesian_radius), dtype=np.complex128)
    angular = np.exp(
        1j * phi[..., None] * SPIN_ALLOWED_ORDERS[None, None, :]
    )
    circular_channels = radial * angular
    circular = _circular_matrix_from_channels(circular_channels)
    return circular_to_linear_jones(circular), valid


def jones_channel_amplitude(
    jones: np.ndarray,
    *,
    input_state: np.ndarray = np.asarray((1.0, 0.0)),
    analyzer_state: np.ndarray = np.asarray((0.0, 1.0)),
) -> np.ndarray:
    r"""Return ``analyzer_state^\dagger J input_state``."""

    matrix = _as_jones(jones)
    input_vector = _polarization_state("input_state", input_state)
    analyzer_vector = _polarization_state("analyzer_state", analyzer_state)
    return np.einsum(
        "i,...ij,j->...",
        analyzer_vector.conj(),
        matrix,
        input_vector,
        optimize=True,
    )


def decompose_intensity_harmonics(
    covariant_amplitude: np.ndarray,
    residual_amplitude: np.ndarray,
    *,
    modes: np.ndarray | tuple[int, ...],
    phi_rad: np.ndarray,
    angular_axis: int = 1,
) -> IntensityHarmonicDecomposition:
    """Fourier-decompose all terms in ``|u_c + u_r|^2`` exactly."""

    covariant = np.asarray(covariant_amplitude, dtype=np.complex128)
    residual = np.asarray(residual_amplitude, dtype=np.complex128)
    if covariant.shape != residual.shape:
        raise ValueError("covariant_amplitude and residual_amplitude must match")
    cc_intensity = np.abs(covariant) ** 2
    cr_intensity = 2.0 * np.real(covariant * residual.conj())
    rr_intensity = np.abs(residual) ** 2
    total_intensity = np.abs(covariant + residual) ** 2
    signed_modes = np.asarray(modes, dtype=np.int64)

    def transform(values: np.ndarray) -> np.ndarray:
        return angular_fourier(
            values,
            signed_modes,
            phi_rad=np.asarray(phi_rad, dtype=np.float64),
            angular_axis=angular_axis,
        )

    return IntensityHarmonicDecomposition(
        modes=signed_modes,
        total=np.asarray(transform(total_intensity), dtype=np.complex128),
        cc=np.asarray(transform(cc_intensity), dtype=np.complex128),
        cr=np.asarray(transform(cr_intensity), dtype=np.complex128),
        rr=np.asarray(transform(rr_intensity), dtype=np.complex128),
        total_intensity=np.asarray(total_intensity, dtype=np.float64),
        cc_intensity=np.asarray(cc_intensity, dtype=np.float64),
        cr_intensity=np.asarray(cr_intensity, dtype=np.float64),
        rr_intensity=np.asarray(rr_intensity, dtype=np.float64),
        angular_axis=int(angular_axis),
    )


def decompose_crossed_intensity_harmonics(
    projection: PolarCovariantJonesProjection,
    *,
    modes: np.ndarray | tuple[int, ...] = (0, 2, 4),
    input_state: np.ndarray = np.asarray((1.0, 0.0)),
    analyzer_state: np.ndarray = np.asarray((0.0, 1.0)),
) -> IntensityHarmonicDecomposition:
    """Decompose crossed-channel harmonics of a polar Jones projection."""

    covariant = jones_channel_amplitude(
        projection.covariant_jones,
        input_state=input_state,
        analyzer_state=analyzer_state,
    )
    residual = jones_channel_amplitude(
        projection.residual_jones,
        input_state=input_state,
        analyzer_state=analyzer_state,
    )
    return decompose_intensity_harmonics(
        covariant,
        residual,
        modes=modes,
        phi_rad=projection.phi_rad,
        angular_axis=1,
    )


def _projection_diagnostics(
    total: np.ndarray,
    covariant: np.ndarray,
    residual: np.ndarray,
    *,
    radius_um: np.ndarray | None,
) -> CovariantProjectionDiagnostics:
    total_density = np.mean(np.sum(np.abs(total) ** 2, axis=(-2, -1)), axis=1)
    covariant_density = np.mean(
        np.sum(np.abs(covariant) ** 2, axis=(-2, -1)), axis=1
    )
    residual_density = np.mean(
        np.sum(np.abs(residual) ** 2, axis=(-2, -1)), axis=1
    )
    orthogonality_density = np.mean(
        np.sum(covariant.conj() * residual, axis=(-2, -1)), axis=1
    )
    total_channels = _circular_channel_stack(linear_to_circular_jones(total))
    covariant_channels = _circular_channel_stack(
        linear_to_circular_jones(covariant)
    )
    residual_channels = _circular_channel_stack(linear_to_circular_jones(residual))
    channel_total_density = np.mean(np.abs(total_channels) ** 2, axis=1)
    channel_covariant_density = np.mean(np.abs(covariant_channels) ** 2, axis=1)
    channel_residual_density = np.mean(np.abs(residual_channels) ** 2, axis=1)
    total_power = _integrate_density(total_density, radius_um)
    covariant_power = _integrate_density(covariant_density, radius_um)
    residual_power = _integrate_density(residual_density, radius_um)
    orthogonality = _integrate_density(orthogonality_density, radius_um)
    channel_total_power = np.asarray(
        _integrate_density(channel_total_density, radius_um), dtype=np.float64
    )
    channel_covariant_power = np.asarray(
        _integrate_density(channel_covariant_density, radius_um), dtype=np.float64
    )
    channel_residual_power = np.asarray(
        _integrate_density(channel_residual_density, radius_um), dtype=np.float64
    )
    closure = total - covariant - residual
    closure_scale = max(float(np.linalg.norm(total.ravel())), np.finfo(float).tiny)
    relative_closure = float(np.linalg.norm(closure.ravel()) / closure_scale)
    normalization = max(
        np.sqrt(max(covariant_power, 0.0) * max(residual_power, 0.0)),
        np.finfo(float).tiny,
    )
    return CovariantProjectionDiagnostics(
        radius_um=radius_um,
        total_power_density=np.asarray(total_density, dtype=np.float64),
        covariant_power_density=np.asarray(covariant_density, dtype=np.float64),
        residual_power_density=np.asarray(residual_density, dtype=np.float64),
        orthogonality_density=np.asarray(
            orthogonality_density, dtype=np.complex128
        ),
        channel_total_power=channel_total_power,
        channel_covariant_power=channel_covariant_power,
        channel_residual_power=channel_residual_power,
        channel_residual_fraction=np.divide(
            channel_residual_power,
            channel_total_power,
            out=np.zeros_like(channel_residual_power),
            where=channel_total_power > np.finfo(float).tiny,
        ),
        total_power=float(total_power),
        covariant_power=float(covariant_power),
        residual_power=float(residual_power),
        orthogonality=complex(orthogonality),
        relative_residual_norm=float(
            np.sqrt(max(residual_power, 0.0) / max(total_power, np.finfo(float).tiny))
        ),
        covariant_power_fraction=float(
            covariant_power / max(total_power, np.finfo(float).tiny)
        ),
        relative_closure_error=relative_closure,
        relative_orthogonality=float(abs(orthogonality) / normalization),
    )


def _integrate_density(
    density: np.ndarray, radius_um: np.ndarray | None
) -> float | complex | np.ndarray:
    values = np.asarray(density)
    if radius_um is None:
        result = np.mean(values, axis=0)
        return result.item() if result.ndim == 0 else result
    radius = np.asarray(radius_um, dtype=np.float64)
    if radius.size == 1:
        result = values[0]
        return result.item() if result.ndim == 0 else result
    radial_weight_shape = (radius.size,) + (1,) * (values.ndim - 1)
    result = 2.0 * np.pi * np.trapezoid(
        values * radius.reshape(radial_weight_shape),
        radius,
        axis=0,
    )
    return result.item() if result.ndim == 0 else result


def _circular_channel_stack(circular: np.ndarray) -> np.ndarray:
    return np.stack(
        tuple(circular[..., row, column] for row, column in _SPIN_CHANNEL_INDICES),
        axis=-1,
    )


def _circular_matrix_from_channels(channels: np.ndarray) -> np.ndarray:
    values = np.asarray(channels, dtype=np.complex128)
    if values.shape[-1] != 4:
        raise ValueError("circular channel array must end in four components")
    matrix = np.zeros(values.shape[:-1] + (2, 2), dtype=np.complex128)
    for index, (row, column) in enumerate(_SPIN_CHANNEL_INDICES):
        matrix[..., row, column] = values[..., index]
    return matrix


def _as_jones(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("Jones array must end in a 2 by 2 matrix")
    return matrix


def _as_polar_jones(values: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    matrix = _as_jones(values)
    phi = np.asarray(phi_rad, dtype=np.float64)
    if matrix.ndim != 4 or matrix.shape[1] != phi.size:
        raise ValueError("sampled_jones must have shape (radius, phi, 2, 2)")
    return matrix


def _optional_radius(
    radius_um: np.ndarray | None, radial_samples: int
) -> np.ndarray | None:
    if radius_um is None:
        return None
    radius = np.array(radius_um, dtype=np.float64, copy=True)
    if (
        radius.ndim != 1
        or radius.size != radial_samples
        or not np.all(np.isfinite(radius))
        or np.any(radius < 0.0)
        or (radius.size > 1 and np.any(np.diff(radius) <= 0.0))
    ):
        raise ValueError(
            "radius_um must be finite, nonnegative, strictly increasing, and "
            "match the radial sample count"
        )
    radius.setflags(write=False)
    return radius


def _polarization_state(name: str, state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite two-component vector")
    if np.linalg.norm(vector) <= np.finfo(float).tiny:
        raise ValueError(f"{name} must be nonzero")
    return vector
