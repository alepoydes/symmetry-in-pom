#!/usr/bin/env python3
"""Audit sensitivity of the full 520-nm Jones analysis to source mismatch.

The publication calculation treats the reconstructed Jones field as one
monochromatic operator and propagates both input columns at 520 nm.  This
lightweight control repeats only the post-reconstruction homogeneous
propagation, assigning the Cartesian input columns the fixed control values
518.2 and 522 nm before any polarization-basis change or covariant projection.  It
then repeats the unprojected and projected H2/H4 amplitudes, the exact
covariance residual, and the covariant/cross/residual intensity decomposition.

No director or in-layer optical field is recalculated. The measured source
centers and their uncertainties are recorded separately from these numerical
inputs in data_src/experiment.toml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toronoptics.analysis_specs import (
    PUBLICATION_BACKGROUND,
    PUBLICATION_HARMONICS,
)
from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    ARTICLE_VARIANT_SUFFIX,
    decimate_centered_jones,
    experimental_array_center,
    experimental_average_path,
    experimental_jones_propagation_wavelength_um,
    publication_focus_grid,
)
from toronoptics.axisymmetric_jones import (
    annular_jones_background,
    crossed_intensity_harmonics,
    integrated_harmonic_visibility,
    polar_coordinate_grids,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.covariant_jones import project_covariant_jones
from toronoptics.covariant_harmonics import (
    COVARIANT_RESIDUAL_TERMS,
    crossed_covariant_residual_harmonics,
    summarize_covariant_residual_harmonics,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.focus_stack import InputColumnWavelengthJonesPropagator
from toronoptics.source_config import (
    HolographyColumnWavelengthConfig,
    load_holography_column_wavelength_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "experimental-column-wavelength-sensitivity-520-v2"
DEFAULT_MANIFEST = PROJECT_ROOT / "data_src/experiment.toml"
DEFAULT_SOURCE = experimental_average_path(
    PROJECT_ROOT, ARTICLE_GREEN_WAVELENGTH_NM
)
DEFAULT_EVOLUTION_REPORT = (
    PROJECT_ROOT
    / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}"
    / "report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/analysis/numerical_validation"
    / "experimental_520_column_wavelength_sensitivity.json"
)

PROJECTION_RADIUS_UM = np.arange(0.0, 45.01, 0.125, dtype=np.float64)
PROJECTION_ANGULAR_SAMPLES = 720
ANALYSIS_RADIUS_UM = PUBLICATION_HARMONICS.integration_max_um
MAXIMUM_DECOMPOSITION_CLOSURE_ERROR = 5.0e-12


def _portable_path(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path) -> dict[str, str]:
    return {
        "path": _portable_path(path),
        "sha256": _sha256(path),
    }


def _wavelength_metadata(
    config: HolographyColumnWavelengthConfig,
) -> dict[str, Any]:
    return {
        "primary_common_um": float(config.primary_wavelength_um),
        "split_horizontal_um": float(config.horizontal_wavelength_um),
        "split_vertical_um": float(config.vertical_wavelength_um),
        "control_manifest_section": "holography.column_wavelength_control",
        "measured_sources": {
            "horizontal_um": config.measured_horizontal_wavelength_um,
            "horizontal_uncertainty_um": config.measured_horizontal_uncertainty_um,
            "vertical_um": config.measured_vertical_wavelength_um,
            "vertical_uncertainty_um": config.measured_vertical_uncertainty_um,
            "instrument": config.measurement_instrument,
        },
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _integrated_visibilities(profiles: object) -> dict[str, float]:
    low, high = PUBLICATION_HARMONICS.integration_interval_um
    return {
        f"V{mode}": integrated_harmonic_visibility(
            profiles,
            mode=mode,
            min_radius_um=low,
            max_radius_um=high,
        )
        for mode in (2, 4)
    }


def _covariant_residual_summary(
    total_jones: np.ndarray,
    covariant_jones: np.ndarray,
    residual_jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> dict[str, Any]:
    coefficients = crossed_covariant_residual_harmonics(
        total_jones,
        covariant_jones,
        residual_jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        radial_centers_um=PUBLICATION_HARMONICS.radial_centers_um,
        radial_sigma_um=PUBLICATION_HARMONICS.radial_sigma_um,
        modes=PUBLICATION_HARMONICS.modes,
        axis_order="xy",
    )
    closure = coefficients[0] - np.sum(coefficients[1:], axis=0)
    closure_error = float(
        np.linalg.norm(closure.ravel())
        / max(np.linalg.norm(coefficients[0].ravel()), np.finfo(float).tiny)
    )
    if closure_error > MAXIMUM_DECOMPOSITION_CLOSURE_ERROR:
        raise RuntimeError(
            "Covariant/residual harmonic decomposition failed to close: "
            f"{closure_error:.3e}"
        )
    return {
        "terms": {
            "total": "|u_c+u_r|^2",
            "cc": "|u_c|^2",
            "cr": "2 Re(u_c u_r*)",
            "rr": "|u_r|^2",
        },
        "closure_error": closure_error,
        "harmonics": summarize_covariant_residual_harmonics(
            coefficients,
            radius_um=PUBLICATION_HARMONICS.radial_centers_um,
            modes=PUBLICATION_HARMONICS.modes,
            integration_interval_um=(
                PUBLICATION_HARMONICS.integration_interval_um
            ),
            terms=COVARIANT_RESIDUAL_TERMS,
        ),
    }


def _covariant_residual_difference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for harmonic in ("H2", "H4"):
        candidate_harmonic = candidate["harmonics"][harmonic]
        reference_harmonic = reference["harmonics"][harmonic]
        term_changes: dict[str, Any] = {}
        for term in ("cc", "cr", "rr"):
            candidate_term = candidate_harmonic["terms"][term]
            reference_term = reference_harmonic["terms"][term]
            term_changes[term] = {
                key: float(candidate_term[key] - reference_term[key])
                for key in (
                    "norm_ratio_to_total",
                    "difference_norm_ratio_to_total",
                    "signed_projection_onto_total",
                )
            }
        result[harmonic] = {
            "integrated_visibility": float(
                candidate_harmonic["integrated_visibility"]
                - reference_harmonic["integrated_visibility"]
            ),
            "terms": term_changes,
        }
    return result


def _analyze_field(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    background = annular_jones_background(
        jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        spec=PUBLICATION_BACKGROUND,
    )
    full_projection = project_covariant_jones(
        jones,
        center_xy_px=center_px,
        pixel_size_xy_um=pixel_size_um,
        radius_um=PROJECTION_RADIUS_UM,
        angular_samples=PROJECTION_ANGULAR_SAMPLES,
        interpolation="cubic",
        radial_interpolation="cubic",
        radial_extrapolation="nearest",
        axis_order="xy",
    )
    signal_projection = project_covariant_jones(
        jones - background[None, None],
        center_xy_px=center_px,
        pixel_size_xy_um=pixel_size_um,
        radius_um=PROJECTION_RADIUS_UM,
        angular_samples=PROJECTION_ANGULAR_SAMPLES,
        interpolation="cubic",
        radial_interpolation="cubic",
        radial_extrapolation="nearest",
        axis_order="xy",
    )
    harmonic_arguments = {
        "center_px": center_px,
        "pixel_size_um": pixel_size_um,
        "radial_centers_um": PUBLICATION_HARMONICS.radial_centers_um,
        "radial_sigma_um": PUBLICATION_HARMONICS.radial_sigma_um,
        "modes": PUBLICATION_HARMONICS.modes,
    }
    measured = crossed_intensity_harmonics(jones, **harmonic_arguments)
    projected = crossed_intensity_harmonics(
        full_projection.covariant_jones,
        **harmonic_arguments,
    )
    decomposition = _covariant_residual_summary(
        jones,
        full_projection.covariant_jones,
        full_projection.residual_jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
    )
    return (
        {
            "covariant_residual_norm": float(
                signal_projection.polar.diagnostics.relative_residual_norm
            ),
            "unprojected_crossed": _integrated_visibilities(measured),
            "covariant_projection_crossed": _integrated_visibilities(projected),
            "covariant_residual_crossed_decomposition": decomposition,
        },
        {
            "unprojected": np.asarray(
                measured.coefficients, dtype=np.complex128
            ),
            "projected": np.asarray(
                projected.coefficients, dtype=np.complex128
            ),
        },
    )


def _profile_relative_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    mode: int,
) -> float:
    modes = np.asarray(PUBLICATION_HARMONICS.modes, dtype=np.int64)
    mode_index = int(np.flatnonzero(modes == int(mode))[0])
    radius = PUBLICATION_HARMONICS.radial_centers_um
    low, high = PUBLICATION_HARMONICS.integration_interval_um
    selected = (radius >= low) & (radius <= high)
    radial = radius[selected]
    difference = candidate[mode_index, selected] - reference[
        mode_index, selected
    ]
    denominator = np.trapezoid(
        np.abs(reference[mode_index, selected]) ** 2 * radial,
        radial,
    )
    numerator = np.trapezoid(
        np.abs(difference) ** 2 * radial,
        radial,
    )
    return float(np.sqrt(numerator / max(denominator, 1.0e-300)))


def _field_relative_differences(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> dict[str, float]:
    candidate_background = annular_jones_background(
        candidate,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        spec=PUBLICATION_BACKGROUND,
    )
    reference_background = annular_jones_background(
        reference,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        spec=PUBLICATION_BACKGROUND,
    )
    candidate_signal = candidate - candidate_background[None, None]
    reference_signal = reference - reference_background[None, None]
    radius_um, _phi = polar_coordinate_grids(
        reference.shape[:2],
        center_px=center_px,
        pixel_size_um=pixel_size_um,
    )
    selected = radius_um <= ANALYSIS_RADIUS_UM

    def relative(input_index: int | None) -> float:
        if input_index is None:
            candidate_values = candidate_signal[selected]
            reference_values = reference_signal[selected]
        else:
            candidate_values = candidate_signal[selected, :, input_index]
            reference_values = reference_signal[selected, :, input_index]
        numerator = float(
            np.sum(np.abs(candidate_values - reference_values) ** 2)
        )
        denominator = float(np.sum(np.abs(reference_values) ** 2))
        return float(np.sqrt(numerator / max(denominator, 1.0e-300)))

    return {
        "full_jones": relative(None),
        "horizontal_input_column": relative(0),
        "vertical_input_column": relative(1),
    }


def _completed_report(
    *,
    manifest: Path,
    source: Path,
    evolution_report_path: Path,
    config: HolographyColumnWavelengthConfig,
    workers: int,
) -> dict[str, Any]:
    expected_primary = experimental_jones_propagation_wavelength_um(
        ARTICLE_GREEN_WAVELENGTH_NM
    )
    if not np.isclose(
        config.primary_wavelength_um,
        expected_primary,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "The manifest 520-nm calculation wavelength disagrees with the "
            f"article configuration: {config.primary_wavelength_um:g} versus "
            f"{expected_primary:g} um"
        )

    evolution_report = json.loads(
        evolution_report_path.read_text(encoding="utf-8")
    )
    z_um, surface_um, annular_focus_um = publication_focus_grid(
        evolution_report,
        wavelength_nm=ARTICLE_GREEN_WAVELENGTH_NM,
    )
    archive = load_jones_archive(source, dtype=np.complex64)
    center = experimental_array_center(source)
    jones, center = decimate_centered_jones(
        archive.jones,
        center,
        factor=2,
    )
    pixel_size_um = 2.0 * float(archive.pixel_size_um)

    common = InputColumnWavelengthJonesPropagator(
        jones,
        pixel_size_um=pixel_size_um,
        wavelength_um_by_input=(
            config.primary_wavelength_um,
            config.primary_wavelength_um,
        ),
        numerical_aperture=None,
        padding_factor=2.0,
        workers=workers,
    )
    split = InputColumnWavelengthJonesPropagator(
        jones,
        pixel_size_um=pixel_size_um,
        wavelength_um_by_input=(
            config.horizontal_wavelength_um,
            config.vertical_wavelength_um,
        ),
        numerical_aperture=None,
        padding_factor=2.0,
        workers=workers,
    )

    planes: list[dict[str, Any]] = []
    for distance_um in z_um:
        primary_field = common.propagate_jones(float(distance_um))
        split_field = split.propagate_jones(float(distance_um))
        primary_summary, primary_profiles = _analyze_field(
            primary_field,
            center_px=center,
            pixel_size_um=pixel_size_um,
        )
        split_summary, split_profiles = _analyze_field(
            split_field,
            center_px=center,
            pixel_size_um=pixel_size_um,
        )
        profile_changes = {
            sector: {
                f"H{mode}": _profile_relative_difference(
                    split_profiles[sector],
                    primary_profiles[sector],
                    mode=mode,
                )
                for mode in (2, 4)
            }
            for sector in ("unprojected", "projected")
        }
        planes.append(
            {
                "distance_from_holographic_plane_um": float(distance_um),
                "surface_referenced_z_um": float(distance_um - surface_um),
                "primary_common_520": primary_summary,
                "split_input_columns": split_summary,
                "split_minus_primary": {
                    "background_subtracted_complex_field_relative_difference": (
                        _field_relative_differences(
                            split_field,
                            primary_field,
                            center_px=center,
                            pixel_size_um=pixel_size_um,
                        )
                    ),
                    "complex_harmonic_profile_relative_difference": (
                        profile_changes
                    ),
                    "covariant_residual_norm_change": float(
                        split_summary["covariant_residual_norm"]
                        - primary_summary["covariant_residual_norm"]
                    ),
                    "covariant_residual_crossed_decomposition": (
                        _covariant_residual_difference(
                            split_summary[
                                "covariant_residual_crossed_decomposition"
                            ],
                            primary_summary[
                                "covariant_residual_crossed_decomposition"
                            ],
                        )
                    ),
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "inputs": [
            _input_record(manifest),
            _input_record(source),
            _input_record(evolution_report_path),
        ],
        "wavelengths": _wavelength_metadata(config),
        "field": (
            "factor-two-decimated coherently averaged experimental Jones field"
        ),
        "control_definition": (
            "the two Cartesian input columns are propagated independently at "
            "their source wavelengths before any input-basis change; all "
            "planes, centers, backgrounds, and estimators are fixed by the "
            "primary common-520-nm analysis"
        ),
        "interpretation": (
            "the split-column field is a source-mismatch sensitivity surrogate, "
            "not a monochromatic Jones operator"
        ),
        "propagation": {
            "method": (
                "band-limited Cartesian angular spectrum with analytic "
                "plane-wave background"
            ),
            "medium_index": 1.0,
            "numerical_aperture": None,
            "padding_factor": 2.0,
            "surface_distance_um": float(surface_um),
            "annular_focus_distance_um": float(annular_focus_um),
        },
        "center_px_after_decimation": [float(value) for value in center],
        "pixel_size_um_after_decimation": pixel_size_um,
        "analysis_radius_um": ANALYSIS_RADIUS_UM,
        "background": PUBLICATION_BACKGROUND.metadata(),
        "harmonics": PUBLICATION_HARMONICS.metadata(),
        "planes": planes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    source = args.source.expanduser().resolve()
    evolution_report = args.evolution_report.expanduser().resolve()
    config = load_holography_column_wavelength_config(manifest)
    return _completed_report(
        manifest=manifest,
        source=source,
        evolution_report_path=evolution_report,
        config=config,
        workers=int(args.workers),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--evolution-report",
        type=Path,
        default=DEFAULT_EVOLUTION_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=-1)
    add_force_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    refuse_existing_outputs(
        (output,),
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    report = run(args)
    _write_report(output, report)
    print(
        "[column-wavelength] completed at "
        f"{len(report['planes'])} fixed publication planes",
        flush=True,
    )
    print(f"[column-wavelength] report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
