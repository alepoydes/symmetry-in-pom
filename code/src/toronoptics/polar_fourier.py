r"""Canonical polar sampling and angular Fourier transforms.

The project contains both numerical Jones arrays whose spatial axes are
``(x, y)`` and camera-like scalar images whose axes are ``(y, x)``.  This
module keeps the physical coordinate convention fixed -- centers are always
specified as ``(x, y)`` -- and makes the array-axis convention explicit.

Angular coefficients use

.. math::

   f_m(\rho) = \frac{1}{2\pi}\int_0^{2\pi}
       f(\rho,\phi)\exp(-i m\phi)\,d\phi.

For the uniformly spaced full circles required here, the integral is the
corresponding discrete angular mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import map_coordinates


AxisOrder = Literal["xy", "yx"]
BoundaryPolicy = Literal["raise", "nearest", "reflect", "wrap", "constant"]
InterpolationPolicy = Literal["nearest", "linear", "cubic", "quintic"]

_INTERPOLATION_ORDERS: dict[str, int] = {
    "nearest": 0,
    "linear": 1,
    "cubic": 3,
    "quintic": 5,
}
_SCIPY_BOUNDARY_MODES: dict[str, str] = {
    "nearest": "nearest",
    "reflect": "reflect",
    "wrap": "grid-wrap",
    "constant": "grid-constant",
}


@dataclass(frozen=True)
class PolarSamplingGrid:
    """A full-circle polar sampling grid.

    ``center_xy_px`` is always in physical ``(x, y)`` order, independently of
    ``axis_order``.  ``pixel_size_xy_um`` may be a scalar or an ``(x, y)``
    pair.  The sampled array must have its two spatial axes first; any
    remaining scalar, vector, or matrix axes are preserved.
    """

    center_xy_px: tuple[float, float]
    pixel_size_xy_um: float | tuple[float, float]
    radius_um: np.ndarray
    phi_rad: np.ndarray
    axis_order: AxisOrder = "xy"
    boundary: BoundaryPolicy = "raise"

    def __post_init__(self) -> None:
        center = _finite_pair("center_xy_px", self.center_xy_px)
        pixel_size = _positive_pair("pixel_size_xy_um", self.pixel_size_xy_um)
        radius = np.array(self.radius_um, dtype=np.float64, copy=True)
        phi = np.array(self.phi_rad, dtype=np.float64, copy=True)
        if radius.ndim != 1 or radius.size == 0:
            raise ValueError("radius_um must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
            raise ValueError("radius_um must be finite and nonnegative")
        if radius.size > 1 and np.any(np.diff(radius) <= 0.0):
            raise ValueError("radius_um must be strictly increasing")
        _validate_full_uniform_angles(phi)
        if self.axis_order not in ("xy", "yx"):
            raise ValueError("axis_order must be either 'xy' or 'yx'")
        if self.boundary not in ("raise", *_SCIPY_BOUNDARY_MODES):
            raise ValueError(f"Unsupported boundary policy {self.boundary!r}")
        radius.setflags(write=False)
        phi.setflags(write=False)
        object.__setattr__(self, "center_xy_px", center)
        object.__setattr__(self, "pixel_size_xy_um", pixel_size)
        object.__setattr__(self, "radius_um", radius)
        object.__setattr__(self, "phi_rad", phi)

    @classmethod
    def uniform(
        cls,
        *,
        center_xy_px: tuple[float, float],
        pixel_size_xy_um: float | tuple[float, float],
        radius_um: np.ndarray,
        angular_samples: int,
        phi_offset_rad: float = 0.0,
        axis_order: AxisOrder = "xy",
        boundary: BoundaryPolicy = "raise",
    ) -> PolarSamplingGrid:
        """Construct a uniformly sampled full circle."""

        count = int(angular_samples)
        if count < 4:
            raise ValueError("angular_samples must be at least 4")
        phi = float(phi_offset_rad) + 2.0 * np.pi * np.arange(count) / count
        return cls(
            center_xy_px=center_xy_px,
            pixel_size_xy_um=pixel_size_xy_um,
            radius_um=np.asarray(radius_um, dtype=np.float64),
            phi_rad=phi,
            axis_order=axis_order,
            boundary=boundary,
        )

    @property
    def angular_samples(self) -> int:
        return int(self.phi_rad.size)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.radius_um.size), self.angular_samples)

    def cartesian_coordinates_px(self) -> tuple[np.ndarray, np.ndarray]:
        """Return physical ``(x, y)`` sampling coordinates in pixels."""

        radial_x = self.radius_um[:, None] / self.pixel_size_xy_um[0]
        radial_y = self.radius_um[:, None] / self.pixel_size_xy_um[1]
        x = self.center_xy_px[0] + radial_x * np.cos(self.phi_rad)[None, :]
        y = self.center_xy_px[1] + radial_y * np.sin(self.phi_rad)[None, :]
        return x, y

    def array_coordinates_px(self) -> np.ndarray:
        """Return coordinates ordered for :func:`scipy.ndimage.map_coordinates`."""

        x, y = self.cartesian_coordinates_px()
        return np.stack((x, y) if self.axis_order == "xy" else (y, x), axis=0)

    def valid_disk_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Return the Cartesian pixels lying in the sampled radial interval."""

        x, y = cartesian_coordinate_grids(
            shape,
            center_xy_px=self.center_xy_px,
            pixel_size_xy_um=self.pixel_size_xy_um,
            axis_order=self.axis_order,
        )
        radius = np.hypot(x, y)
        return (radius >= self.radius_um[0]) & (radius <= self.radius_um[-1])


def cartesian_coordinate_grids(
    shape: tuple[int, int],
    *,
    center_xy_px: tuple[float, float],
    pixel_size_xy_um: float | tuple[float, float],
    axis_order: AxisOrder = "xy",
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical coordinates relative to a center.

    The result is always ``(x_um, y_um)`` while each array follows
    ``axis_order`` and therefore has ``shape``.
    """

    if len(shape) != 2 or min(map(int, shape)) <= 0:
        raise ValueError("shape must contain two positive spatial dimensions")
    center = _finite_pair("center_xy_px", center_xy_px)
    pixel_size = _positive_pair("pixel_size_xy_um", pixel_size_xy_um)
    if axis_order == "xy":
        x = (np.arange(shape[0], dtype=np.float64) - center[0]) * pixel_size[0]
        y = (np.arange(shape[1], dtype=np.float64) - center[1]) * pixel_size[1]
        return np.meshgrid(x, y, indexing="ij")
    if axis_order == "yx":
        y = (np.arange(shape[0], dtype=np.float64) - center[1]) * pixel_size[1]
        x = (np.arange(shape[1], dtype=np.float64) - center[0]) * pixel_size[0]
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return xx, yy
    raise ValueError("axis_order must be either 'xy' or 'yx'")


def sample_polar(
    field: np.ndarray,
    grid: PolarSamplingGrid,
    *,
    interpolation: InterpolationPolicy | int = "cubic",
    constant_value: complex | float = 0.0,
) -> np.ndarray:
    """Sample a scalar, vector, or matrix field on ``grid``.

    The first two axes of ``field`` are spatial.  The output shape is
    ``(radius, phi, *field.shape[2:])``.  Complex coherent fields are
    interpolated componentwise without converting them to intensities.
    """

    values = np.asarray(field)
    if values.ndim < 2:
        raise ValueError("field must have at least two spatial dimensions")
    order = interpolation_order(interpolation)
    coordinates = grid.array_coordinates_px()
    _validate_coordinates(values.shape[:2], coordinates, grid.boundary)
    scipy_mode = (
        "nearest"
        if grid.boundary == "raise"
        else _SCIPY_BOUNDARY_MODES[grid.boundary]
    )
    trailing_shape = values.shape[2:]
    flat = values.reshape(values.shape[:2] + (-1,))
    output_dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    sampled = np.empty(grid.shape + (flat.shape[-1],), dtype=output_dtype)
    cval = complex(constant_value)
    for index in range(flat.shape[-1]):
        component = flat[..., index]
        if np.iscomplexobj(component):
            real = map_coordinates(
                component.real,
                coordinates,
                order=order,
                mode=scipy_mode,
                cval=float(cval.real),
                prefilter=order > 1,
            )
            imaginary = map_coordinates(
                component.imag,
                coordinates,
                order=order,
                mode=scipy_mode,
                cval=float(cval.imag),
                prefilter=order > 1,
            )
            sampled[..., index] = real + 1j * imaginary
        else:
            sampled[..., index] = map_coordinates(
                component,
                coordinates,
                order=order,
                mode=scipy_mode,
                cval=float(cval.real),
                prefilter=order > 1,
            )
    return sampled.reshape(grid.shape + trailing_shape)


def angular_fourier(
    samples: np.ndarray,
    modes: np.ndarray | tuple[int, ...],
    *,
    phi_rad: np.ndarray,
    angular_axis: int = 1,
) -> np.ndarray:
    """Return signed angular Fourier coefficients.

    The mode axis is placed first.  For samples shaped
    ``(radius, phi, channels...)`` the result is shaped
    ``(mode, radius, channels...)``.
    """

    values = np.asarray(samples)
    phi = np.asarray(phi_rad, dtype=np.float64)
    _validate_full_uniform_angles(phi)
    axis = np.lib.array_utils.normalize_axis_index(angular_axis, values.ndim)
    if values.shape[axis] != phi.size:
        raise ValueError(
            f"angular axis has length {values.shape[axis]}, expected {phi.size}"
        )
    signed_modes = _validate_modes(modes, phi.size)
    moved = np.moveaxis(values, axis, -1)
    phase = np.exp(-1j * signed_modes[:, None] * phi[None, :])
    return np.einsum("...p,mp->m...", moved, phase, optimize=True) / phi.size


def angular_synthesis(
    coefficients: np.ndarray,
    modes: np.ndarray | tuple[int, ...],
    *,
    phi_rad: np.ndarray,
    mode_axis: int = 0,
    angular_axis: int = 1,
) -> np.ndarray:
    """Synthesize angular samples from signed Fourier coefficients.

    ``angular_axis`` is interpreted in the result after the mode axis has
    been removed.
    """

    values = np.asarray(coefficients)
    phi = np.asarray(phi_rad, dtype=np.float64)
    _validate_full_uniform_angles(phi)
    axis = np.lib.array_utils.normalize_axis_index(mode_axis, values.ndim)
    signed_modes = _validate_modes(modes, phi.size)
    if values.shape[axis] != signed_modes.size:
        raise ValueError(
            f"mode axis has length {values.shape[axis]}, expected {signed_modes.size}"
        )
    moved = np.moveaxis(values, axis, 0)
    synthesized = np.einsum(
        "m...,mp->...p",
        moved,
        np.exp(1j * signed_modes[:, None] * phi[None, :]),
        optimize=True,
    )
    result_ndim = synthesized.ndim
    destination = np.lib.array_utils.normalize_axis_index(angular_axis, result_ndim)
    return np.moveaxis(synthesized, -1, destination)


def interpolation_order(interpolation: InterpolationPolicy | int) -> int:
    """Return the spline order used by a named interpolation policy."""

    if isinstance(interpolation, str):
        try:
            return _INTERPOLATION_ORDERS[interpolation]
        except KeyError as error:
            raise ValueError(
                f"Unsupported interpolation policy {interpolation!r}"
            ) from error
    order = int(interpolation)
    if order != interpolation or order < 0 or order > 5:
        raise ValueError("interpolation order must be an integer from 0 to 5")
    return order


def _validate_full_uniform_angles(phi: np.ndarray) -> None:
    if phi.ndim != 1 or phi.size < 4:
        raise ValueError("phi_rad must be a one-dimensional array of length >= 4")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi_rad must contain only finite values")
    unwrapped = np.unwrap(phi)
    steps = np.diff(np.concatenate((unwrapped, [unwrapped[0] + 2.0 * np.pi])))
    expected = 2.0 * np.pi / phi.size
    if not np.allclose(steps, expected, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("phi_rad must sample one complete circle uniformly")


def _validate_modes(
    modes: np.ndarray | tuple[int, ...], angular_samples: int
) -> np.ndarray:
    raw = np.asarray(modes)
    signed = np.asarray(raw, dtype=np.int64)
    if signed.ndim != 1 or signed.size == 0:
        raise ValueError("modes must be a nonempty one-dimensional array")
    if not np.array_equal(raw, signed):
        raise ValueError("modes must contain integers")
    if np.unique(signed).size != signed.size:
        raise ValueError("modes must not contain duplicates")
    if np.any(np.abs(signed) > angular_samples // 2):
        raise ValueError("requested mode exceeds the angular Nyquist order")
    return signed


def _validate_coordinates(
    shape: tuple[int, int],
    coordinates: np.ndarray,
    boundary: BoundaryPolicy,
) -> None:
    if boundary != "raise":
        return
    for axis, size in enumerate(shape):
        coordinate = coordinates[axis]
        tolerance = 32.0 * np.finfo(np.float64).eps * max(1, int(size))
        if np.min(coordinate) < -tolerance or np.max(coordinate) > size - 1 + tolerance:
            raise ValueError(
                "polar sampling domain extends outside the field; choose a "
                "smaller radius or an explicit non-'raise' boundary policy"
            )


def _finite_pair(name: str, value: tuple[float, float]) -> tuple[float, float]:
    try:
        pair = (float(value[0]), float(value[1]))
    except (IndexError, TypeError) as error:
        raise ValueError(f"{name} must be a two-element pair") from error
    if not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain finite values")
    return pair


def _positive_pair(
    name: str, value: float | tuple[float, float]
) -> tuple[float, float]:
    if np.isscalar(value):
        pair = (float(value), float(value))
    else:
        pair = _finite_pair(name, value)
    if not np.all(np.isfinite(pair)) or min(pair) <= 0.0:
        raise ValueError(f"{name} must contain positive finite values")
    return pair
