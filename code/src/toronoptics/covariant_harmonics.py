"""Crossed-intensity harmonics of covariant and residual Jones fields."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from toronoptics.axisymmetric_jones import soft_annular_harmonics


COVARIANT_RESIDUAL_TERMS = ("total", "cc", "cr", "rr")


def crossed_covariant_residual_harmonics(
    total_jones: np.ndarray,
    covariant_jones: np.ndarray,
    residual_jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radial_centers_um: np.ndarray,
    radial_sigma_um: float,
    modes: Sequence[int],
    axis_order: str = "xy",
) -> np.ndarray:
    """Return harmonics of total, covariant, cross, and residual intensities.

    The returned leading axis follows ``COVARIANT_RESIDUAL_TERMS``.  Complex
    amplitudes are promoted before forming quadratic terms so that the
    identity ``total = cc + cr + rr`` is not limited by mixed precision.
    """

    total_amplitude = np.asarray(total_jones, dtype=np.complex128)[..., 1, 0]
    covariant_amplitude = np.asarray(
        covariant_jones, dtype=np.complex128
    )[..., 1, 0]
    residual_amplitude = np.asarray(
        residual_jones, dtype=np.complex128
    )[..., 1, 0]
    intensities = (
        np.abs(total_amplitude) ** 2,
        np.abs(covariant_amplitude) ** 2,
        2.0 * np.real(covariant_amplitude * residual_amplitude.conj()),
        np.abs(residual_amplitude) ** 2,
    )
    return np.stack(
        [
            soft_annular_harmonics(
                intensity,
                center_xy_px=center_px,
                pixel_size_xy_um=float(pixel_size_um),
                radial_centers_um=radial_centers_um,
                radial_sigma_um=float(radial_sigma_um),
                modes=modes,
                axis_order=axis_order,
            ).coefficients
            for intensity in intensities
        ],
        axis=0,
    )


def summarize_covariant_residual_harmonics(
    coefficients: np.ndarray,
    *,
    radius_um: np.ndarray,
    modes: Sequence[int],
    integration_interval_um: tuple[float, float],
    terms: Sequence[str] = COVARIANT_RESIDUAL_TERMS,
) -> dict[str, object]:
    """Summarize radial-L2 norms and signed overlaps for H2 and H4."""

    coefficients = np.asarray(coefficients, dtype=np.complex128)
    radius = np.asarray(radius_um, dtype=np.float64)
    mode_array = np.asarray(tuple(modes), dtype=np.int64)
    term_names = tuple(terms)
    if coefficients.shape[0] != len(term_names):
        raise ValueError("coefficient term axis disagrees with term labels")
    if coefficients.shape[1] != mode_array.size:
        raise ValueError("coefficient mode axis disagrees with modes")
    if coefficients.shape[2] != radius.size:
        raise ValueError("coefficient radial axis disagrees with radius")

    low, high = map(float, integration_interval_um)
    selected = (radius >= low) & (radius <= high)
    selected_radius = radius[selected]
    if selected_radius.size < 2:
        raise ValueError("integration interval contains fewer than two radii")
    zero_matches = np.flatnonzero(mode_array == 0)
    if zero_matches.size != 1:
        raise ValueError("modes must contain one zero-order entry")
    zero_index = int(zero_matches[0])
    total_h0 = coefficients[0, zero_index, selected]
    denominator_h0 = float(
        np.trapezoid(np.abs(total_h0) ** 2 * selected_radius, selected_radius)
    )

    summary: dict[str, object] = {}
    for mode in (2, 4):
        matches = np.flatnonzero(mode_array == mode)
        if matches.size != 1:
            raise ValueError(f"modes must contain one order-{mode} entry")
        mode_index = int(matches[0])
        total = coefficients[0, mode_index, selected]
        norm2 = float(
            np.trapezoid(np.abs(total) ** 2 * selected_radius, selected_radius)
        )
        denominator = max(norm2, np.finfo(float).tiny)
        term_summary: dict[str, object] = {}
        for term_index, term in enumerate(term_names):
            value = coefficients[term_index, mode_index, selected]
            value_norm2 = float(
                np.trapezoid(
                    np.abs(value) ** 2 * selected_radius,
                    selected_radius,
                )
            )
            difference_norm2 = float(
                np.trapezoid(
                    np.abs(value - total) ** 2 * selected_radius,
                    selected_radius,
                )
            )
            overlap = np.trapezoid(
                np.conj(total) * value * selected_radius,
                selected_radius,
            ) / denominator
            term_summary[str(term)] = {
                "norm_ratio_to_total": float(
                    np.sqrt(value_norm2 / denominator)
                ),
                "difference_norm_ratio_to_total": float(
                    np.sqrt(difference_norm2 / denominator)
                ),
                "complex_projection_onto_total": [
                    float(np.real(overlap)),
                    float(np.imag(overlap)),
                ],
                "signed_projection_onto_total": float(np.real(overlap)),
            }
        summary[f"H{mode}"] = {
            "integrated_visibility": float(
                2.0
                * np.sqrt(
                    norm2 / max(denominator_h0, np.finfo(float).tiny)
                )
            ),
            "terms": term_summary,
        }
    return summary
