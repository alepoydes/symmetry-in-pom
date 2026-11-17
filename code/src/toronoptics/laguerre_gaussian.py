"""Laguerre--Gaussian analysis of localized transverse Jones fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


@dataclass(frozen=True)
class ComplexLeastSquaresFit:
    coefficients: np.ndarray
    residual_power: np.ndarray
    total_power: np.ndarray
    condition_number: float

    @property
    def relative_residual(self) -> np.ndarray:
        return self.residual_power / np.maximum(self.total_power, 1.0e-300)


def laguerre_gaussian_radial(
    radius_um: np.ndarray,
    *,
    mode: int,
    radial_order: int,
    waist_um: float,
) -> np.ndarray:
    r"""Return a radial LG function normalized in ``L2(rho d rho)``.

    The convention is

    .. math::

       R_{pm}(\rho;w)=\frac{1}{w}
       \sqrt{\frac{2p!}{(p+|m|)!}}
       (\rho/w)^{|m|}e^{-\rho^2/(2w^2)}
       L_p^{|m|}(\rho^2/w^2).
    """

    if radial_order < 0:
        raise ValueError("radial_order must be nonnegative")
    if waist_um <= 0.0:
        raise ValueError("waist_um must be positive")

    abs_mode = abs(int(mode))
    order = int(radial_order)
    radius = np.asarray(radius_um, dtype=np.float64)
    scaled_radius = radius / float(waist_um)
    x = scaled_radius * scaled_radius
    log_normalization = 0.5 * (
        np.log(2.0)
        + gammaln(order + 1.0)
        - gammaln(order + abs_mode + 1.0)
    ) - np.log(float(waist_um))
    return (
        np.exp(log_normalization)
        * np.power(scaled_radius, abs_mode)
        * np.exp(-0.5 * x)
        * eval_genlaguerre(order, abs_mode, x)
    )


def laguerre_gaussian_mode(
    radius_um: np.ndarray,
    phi: np.ndarray,
    *,
    mode: int,
    radial_order: int,
    waist_um: float,
) -> np.ndarray:
    r"""Return ``R_pm exp(i m phi) / sqrt(2 pi)``."""

    radial = laguerre_gaussian_radial(
        radius_um,
        mode=mode,
        radial_order=radial_order,
        waist_um=waist_um,
    )
    return radial * np.exp(1j * int(mode) * np.asarray(phi)) / np.sqrt(2.0 * np.pi)


def raised_cosine_radial_window(
    radius_um: np.ndarray,
    *,
    full_weight_radius_um: float,
    zero_weight_radius_um: float,
) -> np.ndarray:
    """Return a C1 radial window with a cosine transition to zero."""

    inner = float(full_weight_radius_um)
    outer = float(zero_weight_radius_um)
    if inner < 0.0 or outer <= inner:
        raise ValueError("Require 0 <= full_weight_radius_um < zero_weight_radius_um")
    radius = np.asarray(radius_um, dtype=np.float64)
    result = np.ones_like(radius)
    result[radius >= outer] = 0.0
    transition = (radius > inner) & (radius < outer)
    coordinate = (radius[transition] - inner) / (outer - inner)
    result[transition] = 0.5 * (1.0 + np.cos(np.pi * coordinate))
    return result


def renyi_effective_mode_count(
    modal_power: np.ndarray,
    *,
    alpha: float,
    axis: int = -1,
) -> np.ndarray:
    r"""Return the Renyi effective number of populated modes.

    The input is normalized internally along ``axis``.  At ``alpha=1`` this
    is the exponential of the Shannon entropy; ``alpha=2`` gives the inverse
    participation ratio.  A one-mode distribution has effective count one,
    while ``K`` equally populated modes have effective count ``K`` for every
    positive order.
    """

    order = float(alpha)
    if order <= 0.0:
        raise ValueError("alpha must be positive")
    power = np.asarray(modal_power, dtype=np.float64)
    if np.any(power < 0.0) or not np.all(np.isfinite(power)):
        raise ValueError("modal_power must be finite and nonnegative")
    total = np.sum(power, axis=axis, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("modal_power must have positive sum along axis")
    probability = power / total
    if np.isclose(order, 1.0, rtol=0.0, atol=1.0e-12):
        terms = np.zeros_like(probability)
        positive = probability > 0.0
        terms[positive] = probability[positive] * np.log(probability[positive])
        return np.exp(-np.sum(terms, axis=axis))
    moment = np.sum(np.power(probability, order), axis=axis)
    return np.power(moment, 1.0 / (1.0 - order))


def modal_concentration_curve(
    modal_power: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Return cumulative power after sorting modes from strongest to weakest."""

    power = np.asarray(modal_power, dtype=np.float64)
    if np.any(power < 0.0) or not np.all(np.isfinite(power)):
        raise ValueError("modal_power must be finite and nonnegative")
    total = np.sum(power, axis=axis, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("modal_power must have positive sum along axis")
    probability = power / total
    ordered = np.flip(np.sort(probability, axis=axis), axis=axis)
    return np.cumsum(ordered, axis=axis)


def soft_annular_angular_profiles(
    fields: np.ndarray,
    radius_um: np.ndarray,
    phi: np.ndarray,
    *,
    signed_modes: np.ndarray,
    radial_centers_um: np.ndarray,
    radial_sigma_um: float,
    spatial_window: np.ndarray | None = None,
    radial_chunk_size: int = 24,
) -> np.ndarray:
    r"""Estimate angular Fourier profiles with overlapping Gaussian annuli.

    ``fields`` has shape ``(..., x, y)``.  Every profile sample is an area
    average over all Cartesian pixels with weight
    ``exp(-(rho-rho_0)^2/(2 sigma^2))``; no pixel is assigned to a radial bin.
    An optional spatial window multiplies the field but not the averaging
    measure.
    """

    if radial_sigma_um <= 0.0:
        raise ValueError("radial_sigma_um must be positive")
    field_array = np.asarray(fields)
    radius = np.asarray(radius_um, dtype=np.float64)
    angle = np.asarray(phi, dtype=np.float64)
    if field_array.shape[-2:] != radius.shape or radius.shape != angle.shape:
        raise ValueError("fields, radius_um, and phi must share spatial axes")

    leading_shape = field_array.shape[:-2]
    flat_fields = field_array.reshape((-1, radius.size)).astype(
        np.complex128,
        copy=False,
    )
    if spatial_window is not None:
        window = np.asarray(spatial_window, dtype=np.float64)
        if window.shape != radius.shape:
            raise ValueError("spatial_window must match radius_um")
        flat_fields = flat_fields * window.ravel()[None, :]

    flat_radius = radius.ravel()
    flat_phi = angle.ravel()
    modes = np.asarray(signed_modes, dtype=np.int64)
    centers = np.asarray(radial_centers_um, dtype=np.float64)
    profiles = np.empty(
        (flat_fields.shape[0], modes.size, centers.size),
        dtype=np.complex128,
    )

    phased_fields = [
        flat_fields * np.exp(-1j * int(mode) * flat_phi)[None, :]
        for mode in modes
    ]
    for start in range(0, centers.size, int(radial_chunk_size)):
        stop = min(centers.size, start + int(radial_chunk_size))
        offsets = (
            flat_radius[None, :] - centers[start:stop, None]
        ) / float(radial_sigma_um)
        weights = np.exp(-0.5 * offsets * offsets)
        denominator = np.maximum(np.sum(weights, axis=1), 1.0e-300)
        for mode_index, phased in enumerate(phased_fields):
            profiles[:, mode_index, start:stop] = (
                phased @ weights.T
            ) / denominator[None, :]

    return profiles.reshape((*leading_shape, modes.size, centers.size))


def soft_annular_angular_profiles_by_mode(
    fields: np.ndarray,
    radius_um: np.ndarray,
    phi: np.ndarray,
    *,
    signed_modes: np.ndarray,
    radial_centers_um: np.ndarray,
    radial_sigma_um: float,
    spatial_window: np.ndarray | None = None,
    radial_chunk_size: int = 24,
) -> np.ndarray:
    r"""Estimate one selected angular order for each Cartesian field.

    This is the mixed-order counterpart of
    :func:`soft_annular_angular_profiles`.  ``signed_modes`` must match the
    leading shape of ``fields``; field ``k`` is multiplied by
    ``exp(-i m_k phi)`` before the common Gaussian-annulus operator is
    applied.  Sharing one radial-weight pass is substantially cheaper than
    calling the scalar estimator independently for many diagnostics.
    """

    if radial_sigma_um <= 0.0:
        raise ValueError("radial_sigma_um must be positive")
    field_array = np.asarray(fields)
    radius = np.asarray(radius_um, dtype=np.float64)
    angle = np.asarray(phi, dtype=np.float64)
    if field_array.shape[-2:] != radius.shape or radius.shape != angle.shape:
        raise ValueError("fields, radius_um, and phi must share spatial axes")

    leading_shape = field_array.shape[:-2]
    modes = np.asarray(signed_modes, dtype=np.int64)
    if modes.shape != leading_shape:
        raise ValueError("signed_modes must match the leading shape of fields")
    centers = np.asarray(radial_centers_um, dtype=np.float64)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("radial_centers_um must be a nonempty 1-D array")
    if centers.size > 1 and np.any(np.diff(centers) <= 0.0):
        raise ValueError("radial_centers_um must be strictly increasing")

    flat_fields = field_array.reshape((-1, radius.size)).astype(
        np.complex128,
        copy=False,
    )
    if spatial_window is not None:
        window = np.asarray(spatial_window, dtype=np.float64)
        if window.shape != radius.shape:
            raise ValueError("spatial_window must match radius_um")
        flat_fields = flat_fields * window.ravel()[None, :]

    flat_radius = radius.ravel()
    flat_phi = angle.ravel()
    phased = flat_fields * np.exp(
        -1j * modes.reshape(-1, 1) * flat_phi[None, :]
    )
    profiles = np.empty(
        (flat_fields.shape[0], centers.size),
        dtype=np.complex128,
    )
    for start in range(0, centers.size, int(radial_chunk_size)):
        stop = min(centers.size, start + int(radial_chunk_size))
        offsets = (
            flat_radius[None, :] - centers[start:stop, None]
        ) / float(radial_sigma_um)
        weights = np.exp(-0.5 * offsets * offsets)
        denominator = np.maximum(np.sum(weights, axis=1), 1.0e-300)
        profiles[:, start:stop] = (
            phased @ weights.T
        ) / denominator[None, :]

    return profiles.reshape((*leading_shape, centers.size))


def soft_annular_radial_integral_weights(
    radius_um: np.ndarray,
    *,
    radial_centers_um: np.ndarray,
    radial_sigma_um: float,
    integration_interval_um: tuple[float, float],
    radial_chunk_size: int = 24,
) -> np.ndarray:
    r"""Return pixel weights for an integrated soft-annulus radial profile.

    For any Cartesian scalar field ``f``, the returned array ``w`` satisfies

    ``sum(f * w) == trapz(<f>_sigma(rho) * rho, rho)``

    over the selected radial-center interval, up to floating-point summation
    order.  It collapses a repeatedly used Gaussian-annulus profile followed
    by radial integration into one reusable weighted sum.
    """

    if radial_sigma_um <= 0.0:
        raise ValueError("radial_sigma_um must be positive")
    radius = np.asarray(radius_um, dtype=np.float64)
    centers = np.asarray(radial_centers_um, dtype=np.float64)
    if radius.ndim != 2:
        raise ValueError("radius_um must be a two-dimensional Cartesian grid")
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("radial_centers_um must contain at least two samples")
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("radial_centers_um must be strictly increasing")
    low, high = map(float, integration_interval_um)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("integration_interval_um must be finite and increasing")
    selected = (centers >= low) & (centers <= high)
    selected_centers = centers[selected]
    if selected_centers.size < 2:
        raise ValueError("integration interval contains fewer than two radii")

    trapezoid_weights = np.zeros_like(selected_centers)
    spacing = np.diff(selected_centers)
    trapezoid_weights[:-1] += 0.5 * spacing
    trapezoid_weights[1:] += 0.5 * spacing
    radial_quadrature = trapezoid_weights * selected_centers

    flat_radius = radius.ravel()
    integrated = np.zeros_like(flat_radius)
    for start in range(0, selected_centers.size, int(radial_chunk_size)):
        stop = min(
            selected_centers.size,
            start + int(radial_chunk_size),
        )
        offsets = (
            flat_radius[None, :] - selected_centers[start:stop, None]
        ) / float(radial_sigma_um)
        weights = np.exp(-0.5 * offsets * offsets)
        denominator = np.maximum(np.sum(weights, axis=1), 1.0e-300)
        coefficients = (
            radial_quadrature[start:stop] / denominator
        )
        integrated += coefficients @ weights
    return integrated.reshape(radius.shape)


def soft_annular_lg_design(
    radius_um: np.ndarray,
    phi: np.ndarray,
    *,
    mode: int,
    radial_order_count: int,
    waist_um: float,
    radial_centers_um: np.ndarray,
    radial_sigma_um: float,
    spatial_window: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the soft-annulus estimator to LG basis functions."""

    basis = np.stack(
        [
            laguerre_gaussian_mode(
                radius_um,
                phi,
                mode=mode,
                radial_order=order,
                waist_um=waist_um,
            )
            for order in range(int(radial_order_count))
        ],
        axis=0,
    )
    profiles = soft_annular_angular_profiles(
        basis,
        radius_um,
        phi,
        signed_modes=np.asarray([mode], dtype=np.int64),
        radial_centers_um=radial_centers_um,
        radial_sigma_um=radial_sigma_um,
        spatial_window=spatial_window,
    )
    return np.asarray(profiles[:, 0, :].T, dtype=np.complex128)


def fit_complex_profiles(
    profiles: np.ndarray,
    design: np.ndarray,
    *,
    radial_centers_um: np.ndarray,
) -> ComplexLeastSquaresFit:
    """Fit one or more complex radial profiles in the ``rho d rho`` metric."""

    values = np.asarray(profiles, dtype=np.complex128)
    matrix = np.asarray(design, dtype=np.complex128)
    radius = np.asarray(radial_centers_um, dtype=np.float64)
    if values.shape[-1] != radius.size or matrix.shape[0] != radius.size:
        raise ValueError("profiles and design must use radial_centers_um as samples")
    if radius.size < 2:
        raise ValueError("At least two radial centers are required")

    radial_step = float(np.median(np.diff(radius)))
    quadrature = np.maximum(radius, 0.5 * radial_step) * radial_step
    sqrt_weight = np.sqrt(quadrature)
    weighted_design = matrix * sqrt_weight[:, None]
    flat_values = values.reshape((-1, radius.size))
    weighted_targets = flat_values * sqrt_weight[None, :]
    coefficients = np.linalg.lstsq(
        weighted_design,
        weighted_targets.T,
        rcond=None,
    )[0].T
    fitted = coefficients @ matrix.T
    residual_power = np.sum(
        np.abs(flat_values - fitted) ** 2 * quadrature[None, :],
        axis=-1,
    )
    total_power = np.sum(
        np.abs(flat_values) ** 2 * quadrature[None, :],
        axis=-1,
    )
    return ComplexLeastSquaresFit(
        coefficients=coefficients.reshape((*values.shape[:-1], matrix.shape[1])),
        residual_power=residual_power.reshape(values.shape[:-1]),
        total_power=total_power.reshape(values.shape[:-1]),
        condition_number=float(np.linalg.cond(weighted_design)),
    )


def project_lg_coefficients_cartesian(
    fields: np.ndarray,
    radius_um: np.ndarray,
    phi: np.ndarray,
    *,
    signed_modes: np.ndarray,
    radial_order_count: int,
    waist_um: float,
    pixel_area_um2: float,
    spatial_window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project windowed Cartesian fields onto normalized LG modes.

    Returns ``(coefficients, total_power, gram_matrices)``.  Coefficients have
    shape ``(..., mode, radial_order)`` and each Gram matrix describes the
    discretized basis on the available Cartesian field of view.
    """

    field_array = np.asarray(fields)
    radius = np.asarray(radius_um, dtype=np.float64)
    angle = np.asarray(phi, dtype=np.float64)
    if field_array.shape[-2:] != radius.shape or radius.shape != angle.shape:
        raise ValueError("fields, radius_um, and phi must share spatial axes")
    if spatial_window is None:
        window = np.ones_like(radius)
    else:
        window = np.asarray(spatial_window, dtype=np.float64)
        if window.shape != radius.shape:
            raise ValueError("spatial_window must match radius_um")

    leading_shape = field_array.shape[:-2]
    flat_fields = (
        field_array.reshape((-1, radius.size)).astype(np.complex128, copy=False)
        * window.ravel()[None, :]
    )
    modes = np.asarray(signed_modes, dtype=np.int64)
    coefficients = np.empty(
        (flat_fields.shape[0], modes.size, int(radial_order_count)),
        dtype=np.complex128,
    )
    gram = np.empty(
        (modes.size, int(radial_order_count), int(radial_order_count)),
        dtype=np.complex128,
    )
    area = float(pixel_area_um2)

    for mode_index, mode in enumerate(modes):
        basis = np.stack(
            [
                laguerre_gaussian_mode(
                    radius,
                    angle,
                    mode=int(mode),
                    radial_order=order,
                    waist_um=waist_um,
                ).ravel()
                for order in range(int(radial_order_count))
            ],
            axis=0,
        )
        coefficients[:, mode_index, :] = area * flat_fields @ basis.conj().T
        gram[mode_index] = area * basis.conj() @ basis.T

    total_power = area * np.sum(np.abs(flat_fields) ** 2, axis=-1)
    return (
        coefficients.reshape((*leading_shape, modes.size, int(radial_order_count))),
        total_power.reshape(leading_shape),
        gram,
    )
