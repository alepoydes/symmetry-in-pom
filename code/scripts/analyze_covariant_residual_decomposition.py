#!/usr/bin/env python3
"""Decompose the measured twofold harmonic into covariant and residual terms.

This is a derived-data calculation.  It reads the saved 520-nm holographic
Jones fields and the current surface-alignment report, performs only
homogeneous free-space propagation, and never invokes an in-layer optical
solver.

For the crossed-channel amplitude ``u = u_c + u_r`` produced by the exact
spin-order Jones projector, the script stores the Gaussian-annulus harmonics
of ``|u_c|^2``, ``2 Re(u_c u_r*)``, and ``|u_r|^2``.  Their sum is checked
against the directly calculated harmonic of ``|u|^2``.  The same calculation
is repeated for the nine individual 520-nm acquisitions to distinguish a
stable effect from coherent-averaging artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from toronoptics.analysis_specs import (
    PUBLICATION_BACKGROUND,
    PUBLICATION_HARMONICS,
    CenterSpec,
)
from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    ARTICLE_VARIANT_SUFFIX,
    decimate_centered_jones,
    experimental_array_center,
    experimental_average_path,
    experimental_jones_propagation_wavelength_um,
)
from toronoptics.axisymmetric_jones import annular_jones_background
from toronoptics.axisymmetric_jones import (
    polar_coordinate_grids,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.covariant_jones import (
    SPIN_ALLOWED_ORDERS,
    SPIN_CHANNEL_LABELS,
    covariance_rotation_difference,
    normality_defect,
    project_covariant_jones,
    rotationally_invariant_matrix_part,
)
from toronoptics.covariant_harmonics import (
    COVARIANT_RESIDUAL_TERMS,
    summarize_covariant_residual_harmonics,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.focus_stack import AngularSpectrumJonesPropagator
from toronoptics.jones_ensemble_average import extract_centered_jones_physical
from toronoptics.laguerre_gaussian import (
    soft_annular_angular_profiles_by_mode,
    soft_annular_radial_integral_weights,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELENGTH_NM = ARTICLE_GREEN_WAVELENGTH_NM
WAVELENGTH_UM = experimental_jones_propagation_wavelength_um(WAVELENGTH_NM)
EXPERIMENT_PATH = experimental_average_path(PROJECT_ROOT, WAVELENGTH_NM)
EXPERIMENT_MANIFEST = PROJECT_ROOT / "data_src" / "experiment.toml"
EVOLUTION_REPORT = (
    PROJECT_ROOT
    / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}"
    / "report.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / f"data/analysis/covariant_residual_520{ARTICLE_VARIANT_SUFFIX}"
)
ARCHIVE_PATH = OUTPUT_DIR / "covariant_residual_decomposition.npz"
REPORT_PATH = OUTPUT_DIR / "report.json"

PROJECTION_RADIUS_UM = np.arange(0.0, 45.01, 0.125, dtype=np.float64)
# On the averaged field, doubling 360 to 720 changes the projected normalized
# normality defect by 2.5e-8 and the covariant residual by 5.4e-6.
ANGULAR_SAMPLES = 360
POLAR_INTERPOLATION = "cubic"
RADIAL_INTERPOLATION = "cubic"
MODES = np.asarray(PUBLICATION_HARMONICS.modes, dtype=np.int16)
TERMS = COVARIANT_RESIDUAL_TERMS
MAXIMUM_CLOSURE_ERROR = 5.0e-12


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _surface_distance_um() -> float:
    report = json.loads(EVOLUTION_REPORT.read_text(encoding="utf-8"))
    return float(
        report["annular_focus_alignment"][str(WAVELENGTH_NM)][
            "inferred_experimental_surface_z_um"
        ]
    )


def _propagate(
    jones: np.ndarray,
    *,
    pixel_size_um: float,
    distance_um: float,
) -> np.ndarray:
    propagator = AngularSpectrumJonesPropagator(
        np.asarray(jones, dtype=np.complex64),
        pixel_size_um=float(pixel_size_um),
        wavelength_um=WAVELENGTH_UM,
        numerical_aperture=None,
        padding_factor=2.0,
    )
    return np.asarray(
        propagator.propagate_jones(float(distance_um)),
        dtype=np.complex64,
    )


def _complex_mode_summary(coefficients: np.ndarray) -> dict[str, object]:
    return summarize_covariant_residual_harmonics(
        coefficients,
        radius_um=PUBLICATION_HARMONICS.radial_centers_um,
        modes=MODES,
        integration_interval_um=PUBLICATION_HARMONICS.integration_interval_um,
        terms=TERMS,
    )


def _integrated_radial_value(profile: np.ndarray) -> float:
    radius = PUBLICATION_HARMONICS.radial_centers_um
    low, high = PUBLICATION_HARMONICS.integration_interval_um
    selected = (radius >= low) & (radius <= high)
    return float(
        np.trapezoid(
            np.real(np.asarray(profile)[selected]) * radius[selected],
            radius[selected],
        )
    )


def _integrated_complex_power(profile: np.ndarray) -> float:
    return _integrated_radial_value(np.abs(np.asarray(profile)) ** 2)


def _integrated_cartesian_value(
    field: np.ndarray,
    radial_integral_weights: np.ndarray,
) -> float:
    values = np.asarray(field)
    weights = np.asarray(radial_integral_weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("field and radial_integral_weights must have equal shape")
    return float(np.real(np.sum(values * weights)))


def _integrated_matrix_power(
    matrix: np.ndarray,
    radial_integral_weights: np.ndarray,
) -> float:
    values = np.asarray(matrix)
    if values.ndim < 3:
        raise ValueError("a sampled field must retain at least one component axis")
    density = np.sum(
        np.abs(values) ** 2,
        axis=tuple(range(2, values.ndim)),
    )
    return _integrated_cartesian_value(density, radial_integral_weights)


def _relative_matrix_norm(
    difference: np.ndarray,
    reference: np.ndarray,
    radial_integral_weights: np.ndarray,
) -> float:
    numerator = _integrated_matrix_power(
        difference,
        radial_integral_weights,
    )
    denominator = _integrated_matrix_power(
        reference,
        radial_integral_weights,
    )
    return float(np.sqrt(numerator / max(denominator, 1.0e-300)))


def _rotation_summary(
    signal: np.ndarray,
    *,
    center_px: tuple[float, float],
    radial_integral_weights: np.ndarray,
    quarter_turns: int,
) -> dict[str, object]:
    expected_center = (
        0.5 * (signal.shape[0] - 1),
        0.5 * (signal.shape[1] - 1),
    )
    if not np.allclose(center_px, expected_center, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "exact discrete rotation requires the Jones field to be centered "
            f"on the grid: got {center_px}, expected {expected_center}"
        )
    difference = covariance_rotation_difference(
        signal,
        quarter_turns=quarter_turns,
    )
    spatially_rotated = np.rot90(
        signal,
        k=-(int(quarter_turns) % 4),
        axes=(0, 1),
    )
    polarization_rotated = spatially_rotated - difference

    def relative(
        values: np.ndarray,
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        numerator = _integrated_matrix_power(
            values,
            radial_integral_weights,
        )
        reference_power = 0.5 * (
            _integrated_matrix_power(
                observed,
                radial_integral_weights,
            )
            + _integrated_matrix_power(
                predicted,
                radial_integral_weights,
            )
        )
        return float(
            np.sqrt(numerator / max(reference_power, 1.0e-300))
        )

    result: dict[str, object] = {
        "relative_l2_residual": relative(
            difference,
            spatially_rotated,
            polarization_rotated,
        ),
        "input_column_relative_l2_residual": {},
    }
    column_labels = ("x", "y")
    for column, label in enumerate(column_labels):
        observed = spatially_rotated[..., :, column]
        predicted = polarization_rotated[..., :, column]
        column_difference = difference[..., :, column]
        result["input_column_relative_l2_residual"][label] = relative(
            column_difference,
            observed,
            predicted,
        )
    return result


def _normality_order_two_from_defect(
    defect: np.ndarray,
    phi_rad: np.ndarray,
) -> np.ndarray:
    angle = np.asarray(phi_rad, dtype=np.float64)
    c2 = np.cos(2.0 * angle)
    s2 = np.sin(2.0 * angle)
    sigma = np.empty(angle.shape + (2, 2), dtype=np.complex128)
    sigma[..., 0, 0] = c2 - 1j * s2
    sigma[..., 0, 1] = s2 + 1j * c2
    sigma[..., 1, 0] = s2 + 1j * c2
    sigma[..., 1, 1] = -c2 + 1j * s2
    return np.asarray(
        np.einsum("...ij,...ji->...", defect, sigma, optimize=True) / 8.0,
        dtype=np.complex128,
    )


def _complex_profile_relative_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> float:
    numerator = _integrated_complex_power(
        np.asarray(candidate) - np.asarray(reference)
    )
    denominator = _integrated_complex_power(reference)
    return float(np.sqrt(numerator / max(denominator, 1.0e-300)))


def _surface_profiles_and_normality(
    total_jones: np.ndarray,
    covariant_jones: np.ndarray,
    residual_jones: np.ndarray,
    *,
    radius_um: np.ndarray,
    phi_rad: np.ndarray,
    radial_integral_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    fields = {
        "measured": np.asarray(total_jones, dtype=np.complex128),
        "covariant_projection": np.asarray(
            covariant_jones,
            dtype=np.complex128,
        ),
    }
    total_amplitude = fields["measured"][..., 1, 0]
    covariant_amplitude = fields["covariant_projection"][..., 1, 0]
    residual_amplitude = np.asarray(
        residual_jones,
        dtype=np.complex128,
    )[..., 1, 0]
    intensity_terms = (
        np.abs(total_amplitude) ** 2,
        np.abs(covariant_amplitude) ** 2,
        2.0 * np.real(covariant_amplitude * residual_amplitude.conj()),
        np.abs(residual_amplitude) ** 2,
    )

    profile_fields: list[np.ndarray] = []
    profile_modes: list[int] = []
    # Encode every requested (field, angular order) pair in one shared
    # Gaussian-annulus pass.  Rebuilding the same radial weights for each
    # diagnostic dominated the original run time.
    for intensity in intensity_terms:
        for mode in MODES:
            profile_fields.append(intensity)
            profile_modes.append(int(mode))

    defects: dict[str, np.ndarray] = {}
    normality_fields: dict[str, dict[str, np.ndarray]] = {}
    for label, field in fields.items():
        defect = normality_defect(field)
        defects[label] = defect
        density = 0.5 * np.sum(np.abs(defect) ** 2, axis=(-2, -1))
        order_two = _normality_order_two_from_defect(
            defect,
            phi_rad,
        )
        crossed_intensity = np.abs(field[..., 1, 0]) ** 2
        adjoint = np.swapaxes(field.conj(), -2, -1)
        scalar_power = 0.5 * np.real(
            np.trace(adjoint @ field, axis1=-2, axis2=-1)
        )
        normality_fields[label] = {
            "density": density,
            "order_two": order_two,
            "crossed_intensity": crossed_intensity,
            "scale": scalar_power**2,
        }
        profile_fields.extend(
            (density, order_two, crossed_intensity)
        )
        profile_modes.extend((0, 0, 2))

    mixed_profiles = soft_annular_angular_profiles_by_mode(
        np.stack(profile_fields, axis=0),
        radius_um,
        phi_rad,
        signed_modes=np.asarray(profile_modes, dtype=np.int64),
        radial_centers_um=PUBLICATION_HARMONICS.radial_centers_um,
        radial_sigma_um=PUBLICATION_HARMONICS.radial_sigma_um,
    )
    coefficient_count = len(TERMS) * MODES.size
    coefficients = np.asarray(
        mixed_profiles[:coefficient_count].reshape(
            len(TERMS),
            MODES.size,
            PUBLICATION_HARMONICS.radial_centers_um.size,
        ),
        dtype=np.complex128,
    )

    profiles: dict[str, np.ndarray] = {}
    summaries: dict[str, object] = {}
    profile_index = coefficient_count
    for label in fields:
        density_profile = np.real(mixed_profiles[profile_index])
        order_two_profile = np.asarray(
            mixed_profiles[profile_index + 1],
            dtype=np.complex128,
        )
        crossed_h2 = np.asarray(
            mixed_profiles[profile_index + 2],
            dtype=np.complex128,
        )
        profile_index += 3
        integrated_density = _integrated_cartesian_value(
            normality_fields[label]["density"],
            radial_integral_weights,
        )
        integrated_scale = _integrated_cartesian_value(
            normality_fields[label]["scale"],
            radial_integral_weights,
        )
        order_two_power = _integrated_complex_power(order_two_profile)
        h2_power = _integrated_complex_power(crossed_h2)
        profiles[f"{label}_D_N"] = density_profile
        profiles[f"{label}_normality_order_two"] = order_two_profile
        profiles[f"{label}_crossed_H2"] = crossed_h2
        summaries[label] = {
            "normalized_rms_normality_defect": float(
                np.sqrt(
                    integrated_density
                    / max(integrated_scale, 1.0e-300)
                )
            ),
            "integrated_D_N": integrated_density,
            "normality_order_two_l2": float(np.sqrt(order_two_power)),
            "crossed_H2_l2": float(np.sqrt(h2_power)),
            "order_two_to_crossed_H2_relative_profile_error": (
                _complex_profile_relative_difference(
                    order_two_profile,
                    crossed_h2,
                )
            ),
        }
    summaries["measured_minus_covariant_projection"] = {
        "normality_matrix_relative_l2_difference": _relative_matrix_norm(
            defects["measured"] - defects["covariant_projection"],
            defects["covariant_projection"],
            radial_integral_weights,
        ),
        "normality_order_two_relative_profile_difference": (
            _complex_profile_relative_difference(
                profiles["measured_normality_order_two"],
                profiles[
                    "covariant_projection_normality_order_two"
                ],
            )
        ),
    }
    return coefficients, profiles, summaries


def _analyze_surface(
    surface: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radius_um: np.ndarray,
    phi_rad: np.ndarray,
    radial_integral_weights: np.ndarray,
    angular_samples: int = ANGULAR_SAMPLES,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    background = annular_jones_background(
        surface,
        center_px=center_px,
        pixel_size_um=float(pixel_size_um),
        spec=PUBLICATION_BACKGROUND,
    )
    signal = np.asarray(surface, dtype=np.complex128) - background[None, None]
    signal_projection = project_covariant_jones(
        signal,
        center_xy_px=center_px,
        pixel_size_xy_um=float(pixel_size_um),
        radius_um=PROJECTION_RADIUS_UM,
        angular_samples=int(angular_samples),
        interpolation=POLAR_INTERPOLATION,
        radial_interpolation=RADIAL_INTERPOLATION,
        radial_extrapolation="nearest",
        axis_order="xy",
    )
    invariant_background = rotationally_invariant_matrix_part(background)
    covariant_jones = (
        signal_projection.covariant_jones
        + invariant_background[None, None]
    )
    residual_jones = np.asarray(surface, dtype=np.complex128) - covariant_jones
    coefficients, normality_profiles, normality = (
        _surface_profiles_and_normality(
            surface,
            covariant_jones,
            residual_jones,
            radius_um=radius_um,
            phi_rad=phi_rad,
            radial_integral_weights=radial_integral_weights,
        )
    )
    closure = coefficients[0] - np.sum(coefficients[1:], axis=0)
    closure_error = float(
        np.linalg.norm(closure.ravel())
        / max(np.linalg.norm(coefficients[0].ravel()), np.finfo(float).tiny)
    )
    if closure_error > MAXIMUM_CLOSURE_ERROR:
        raise RuntimeError(
            f"Nonlinear harmonic decomposition failed to close: {closure_error:.3e}"
        )
    diagnostics = signal_projection.diagnostics
    rotation = {
        "quarter_turn_90_deg": _rotation_summary(
            signal,
            center_px=center_px,
            radial_integral_weights=radial_integral_weights,
            quarter_turns=1,
        ),
        "half_turn_180_deg": _rotation_summary(
            signal,
            center_px=center_px,
            radial_integral_weights=radial_integral_weights,
            quarter_turns=2,
        ),
    }
    summary = {
        "background_jones": [
            [[float(value.real), float(value.imag)] for value in row]
            for row in background
        ],
        "projection": {
            "relative_residual_norm_of_background_subtracted_jones": float(
                diagnostics.relative_residual_norm
            ),
            "covariant_power_fraction": float(
                diagnostics.covariant_power_fraction
            ),
            "relative_orthogonality": float(
                diagnostics.relative_orthogonality
            ),
            "relative_closure_error": float(
                diagnostics.relative_closure_error
            ),
            "spin_channels": {
                label: {
                    "allowed_order": int(order),
                    "residual_power_fraction": float(fraction),
                }
                for label, order, fraction in zip(
                    SPIN_CHANNEL_LABELS,
                    SPIN_ALLOWED_ORDERS,
                    diagnostics.channel_residual_fraction,
                    strict=True,
                )
            },
        },
        "discrete_rotation_covariance": rotation,
        "normality": normality,
        "harmonic_decomposition_closure_error": closure_error,
        "harmonics": _complex_mode_summary(coefficients),
    }
    return coefficients, normality_profiles, summary


def _average_surface(
    distance_um: float,
) -> tuple[np.ndarray, tuple[float, float], float]:
    archive = load_jones_archive(EXPERIMENT_PATH, dtype=np.complex64)
    center = experimental_array_center(EXPERIMENT_PATH)
    reduced, reduced_center = decimate_centered_jones(
        archive.jones,
        center,
        factor=2,
    )
    pixel_size_um = 2.0 * float(archive.pixel_size_um)
    return (
        _propagate(
            reduced,
            pixel_size_um=pixel_size_um,
            distance_um=distance_um,
        ),
        tuple(map(float, reduced_center)),
        pixel_size_um,
    )


def _repeatability(
    distance_um: float,
    *,
    skip: bool,
    radius_um: np.ndarray,
    phi_rad: np.ndarray,
    radial_integral_weights: np.ndarray,
) -> list[dict[str, object]]:
    if skip:
        return []
    with np.load(EXPERIMENT_PATH, allow_pickle=False) as average:
        source_paths = tuple(Path(str(value)) for value in average["source_paths"])
        source_centers = np.asarray(average["source_centers_px"], dtype=np.float64)
        output_side = (int(average["Jxx"].shape[0]) + 1) // 2
    center = (0.5 * (output_side - 1), 0.5 * (output_side - 1))
    records: list[dict[str, object]] = []
    for index, (source, source_center) in enumerate(
        zip(source_paths, source_centers, strict=True),
        start=1,
    ):
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        archive = load_jones_archive(source, dtype=np.complex64)
        pixel_size_um = float(archive.pixel_size_um)
        centered, valid = extract_centered_jones_physical(
            archive.jones,
            center=tuple(map(float, source_center)),
            source_pixel_size_um=pixel_size_um,
            output_pixel_size_um=pixel_size_um,
            crop_size_px=output_side,
        )
        if not np.all(valid) or not np.isfinite(centered).all():
            raise ValueError(f"Centered repeatability crop is incomplete: {source}")
        surface = _propagate(
            centered,
            pixel_size_um=pixel_size_um,
            distance_um=distance_um,
        )
        _coefficients, _normality_profiles, summary = _analyze_surface(
            surface,
            center_px=center,
            pixel_size_um=pixel_size_um,
            radius_um=radius_um,
            phi_rad=phi_rad,
            radial_integral_weights=radial_integral_weights,
        )
        records.append(
            {
                "index": index,
                "source": _relative(source),
                "source_sha256": _sha256(source),
                **summary,
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    parser.add_argument(
        "--skip-repeatability",
        action="store_true",
        help="Analyze only the coherently averaged 520-nm field",
    )
    args = parser.parse_args(argv)
    refuse_existing_outputs(
        (ARCHIVE_PATH, REPORT_PATH),
        force=args.force,
        project_root=PROJECT_ROOT,
    )

    distance_um = _surface_distance_um()
    surface, center, pixel_size_um = _average_surface(distance_um)
    radius_um, phi_rad = polar_coordinate_grids(
        surface.shape[:2],
        center_px=center,
        pixel_size_um=float(pixel_size_um),
    )
    # Radial integration is a linear functional of a soft-annulus profile.
    # Contract it once to Cartesian pixel weights and reuse it for all ten
    # fields and every matrix norm.
    radial_integral_weights = soft_annular_radial_integral_weights(
        radius_um,
        radial_centers_um=PUBLICATION_HARMONICS.radial_centers_um,
        radial_sigma_um=PUBLICATION_HARMONICS.radial_sigma_um,
        integration_interval_um=(
            PUBLICATION_HARMONICS.integration_interval_um
        ),
    )
    coefficients, normality_profiles, summary = _analyze_surface(
        surface,
        center_px=center,
        pixel_size_um=pixel_size_um,
        radius_um=radius_um,
        phi_rad=phi_rad,
        radial_integral_weights=radial_integral_weights,
    )
    repeatability = _repeatability(
        distance_um,
        skip=bool(args.skip_repeatability),
        radius_um=radius_um,
        phi_rad=phi_rad,
        radial_integral_weights=radial_integral_weights,
    )
    center_spec = CenterSpec(
        center[0],
        center[1],
        method="nine-acquisition registration anchor, decimated with the field",
        fitted=False,
    )
    metadata = {
        "schema_version": "covariant-residual-decomposition-v4",
        "source": _relative(EXPERIMENT_PATH),
        "source_sha256": _sha256(EXPERIMENT_PATH),
        "experiment_manifest": _relative(EXPERIMENT_MANIFEST),
        "experiment_manifest_sha256": _sha256(EXPERIMENT_MANIFEST),
        "surface_alignment_report": _relative(EVOLUTION_REPORT),
        "surface_alignment_report_sha256": _sha256(EVOLUTION_REPORT),
        "wavelength_nm": WAVELENGTH_NM,
        "wavelength_um": WAVELENGTH_UM,
        "plane": "MW-referenced liquid-crystal exit surface",
        "homogeneous_propagation_distance_um": distance_um,
        "propagation": {
            "medium": "air",
            "pupil": None,
            "padding_factor": 2.0,
            "input_decimation_factor": 2,
            "note": (
                "The measured field is propagated homogeneously from its "
                "holographic plane; no measured or simulated pupil is applied."
            ),
        },
        "pixel_size_um": pixel_size_um,
        "center": center_spec.metadata(),
        "background": PUBLICATION_BACKGROUND.metadata(),
        "harmonic_analysis": PUBLICATION_HARMONICS.metadata(),
        "projection": {
            "definition": (
                "spin-order locking: J++ and J-- retain m=0, "
                "J-+ retains m=+2, J+- retains m=-2"
            ),
            "radius_um": [
                float(PROJECTION_RADIUS_UM[0]),
                float(PROJECTION_RADIUS_UM[-1]),
            ],
            "radial_step_um": float(np.diff(PROJECTION_RADIUS_UM)[0]),
            "angular_samples": ANGULAR_SAMPLES,
            "complex_field_interpolation": POLAR_INTERPOLATION,
            "radial_reconstruction": RADIAL_INTERPOLATION,
            "background_subtraction_inside_projector": False,
            "background_subtracted_before_projection": True,
            "background_restoration": (
                "analytic rotationally invariant a I + b K part"
            ),
        },
        "terms": {
            "total": "|u_c+u_r|^2",
            "cc": "|u_c|^2",
            "cr": "2 Re(u_c u_r*)",
            "rr": "|u_r|^2",
        },
        "discrete_rotation_covariance": {
            "definition": (
                "exact-grid J(R_beta r)-R_beta J(r) R_beta^{-1} "
                "on the background-subtracted 705x705 Jones field"
            ),
            "angles_deg": [90, 180],
            "spatial_interpolation": None,
            "radial_window": PUBLICATION_HARMONICS.metadata(),
        },
        "normality": {
            "defect": "N=J^dagger J-J J^dagger",
            "density": "D_N=||N||_F^2/2, evaluated pixelwise before averaging",
            "order_two_projection": (
                "tr[N(Sigma_1+i Sigma_2)]/8, evaluated pixelwise "
                "before Gaussian-annulus averaging"
            ),
            "radial_window": PUBLICATION_HARMONICS.metadata(),
        },
        "repeatability_records": len(repeatability),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARCHIVE_PATH,
        schema_version=np.asarray(metadata["schema_version"]),
        radius_um=PUBLICATION_HARMONICS.radial_centers_um,
        modes=MODES,
        terms=np.asarray(TERMS),
        coefficients=np.asarray(coefficients, dtype=np.complex64),
        normality_radius_um=PUBLICATION_HARMONICS.radial_centers_um,
        **{
            f"normality_{key}": np.asarray(value, dtype=np.complex64)
            for key, value in normality_profiles.items()
        },
        projection_radius_um=PROJECTION_RADIUS_UM,
        center_px=np.asarray(center, dtype=np.float64),
        pixel_size_um=np.asarray(pixel_size_um, dtype=np.float64),
        surface_distance_um=np.asarray(distance_um, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
        repeatability_json=np.asarray(json.dumps(repeatability, sort_keys=True)),
    )
    report = {
        **metadata,
        "analysis_archive": _relative(ARCHIVE_PATH),
        "average": summary,
        "repeatability": repeatability,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[covariant-residual] archive: {ARCHIVE_PATH}", flush=True)
    print(f"[covariant-residual] report: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
