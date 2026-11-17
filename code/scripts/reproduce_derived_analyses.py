#!/usr/bin/env python3
"""Build angular-mode analyses, propagation controls, and display caches.

Jones-field simulations belong to reproduce_simulated_optics.py; publication
figures are generated separately by make figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from toronoptics.article_data import (
    ARTICLE_WAVELENGTHS_NM,
    effective_coherent_na,
    experimental_jones_propagation_wavelength_um,
    optical_parameters,
)
from toronoptics.cli import add_force_argument
from toronoptics.reproduction import ReproductionStep, run_steps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = ("uv", "run", "python")
EXPECTED_EXPERIMENTAL_ACQUISITIONS = 9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz_schema(path: Path) -> object:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["schema_version"]).item()


def _output_metadata(path: Path) -> dict[str, object]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs_are_current(metadata: dict[str, object]) -> bool:
    inputs = metadata["inputs"]
    if not isinstance(inputs, list) or not inputs:
        return False
    return all(
        _sha256(PROJECT_ROOT / item["path"]) == item["sha256"]
        for item in inputs
    )


def _metadata_inputs_output_is_current(expected_schema: str):
    def check(path: Path) -> bool:
        try:
            metadata = _output_metadata(path)
            return bool(
                metadata.get("schema_version") == expected_schema
                and _inputs_are_current(metadata)
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False

    return check


def _director_helicity_output_is_current(path: Path) -> bool:
    expected_schema = "director-layer-helicity-v1_soft"
    try:
        report = _output_metadata(path.parent / "director_helicity_analysis.json")
        with np.load(path.parent / "director_helicity_analysis.npz", allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            schema = str(archive["schema_version"].item())
            z = archive["z_um"]
            weights = archive["layer_weight"]
            means = archive["layer_tau_rad"]
            fits = archive["fit_linear_tau_rad"]
            coefficients = archive["fit_linear_coefficients_rad"]
            active = weights > np.finfo(float).tiny
        return bool(
            schema == metadata.get("schema_version") == report.get("schema_version") == expected_schema
            and metadata["inputs"] == report["inputs"]
            and len(metadata["inputs"]) == 1
            and _inputs_are_current(metadata)
            and z.ndim == 1 and z.size > 2 and np.all(np.diff(z) > 0)
            and weights.shape == means.shape == fits.shape == (2, z.size)
            and np.all(np.count_nonzero(active, axis=1) >= 2)
            and np.all(np.isfinite(means[active]))
            and np.all(np.isfinite(fits))
            and coefficients.shape == (2, 2)
            and np.all(np.isfinite(coefficients))
            and all(np.isfinite(report["weightings"][name]["linear_rms_deg"])
                    for name in ("geometric", "optical"))
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _single_source_json_is_current(expected_schema: str):
    def check(path: Path) -> bool:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            return bool(
                metadata.get("schema_version") == expected_schema
                and _sha256(PROJECT_ROOT / metadata["source"])
                == metadata["source_sha256"]
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False

    return check


def _axisymmetric_output_is_current(path: Path) -> bool:
    expected = {
        "axisymmetric_jones_coefficients.npz": 3,
        "axisymmetric_jones_evolution.npz": 5,
        "axisymmetric_symmetry_axes_520.npz": 6,
    }
    try:
        report_path = path if path.name == "report.json" else path.parent / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        datasets = report["datasets"]
        if not isinstance(datasets, dict) or not datasets:
            return False
        inputs_current = all(
            _sha256(PROJECT_ROOT / item["source"]) == item["source_sha256"]
            for item in datasets.values()
        )
        pupil_current = all(
            np.isclose(
                float(item["effective_coherent_na"]),
                effective_coherent_na(PROJECT_ROOT, int(wavelength_nm)),
                rtol=0.0,
                atol=1.0e-12,
            )
            for wavelength_nm in ARTICLE_WAVELENGTHS_NM
            for key, item in report["datasets"].items()
            if key.startswith(f"{wavelength_nm}_")
        )
        propagation_wavelengths_current = True
        for key, item in datasets.items():
            wavelength_text, separator, dataset_key = key.partition("_")
            if not separator:
                propagation_wavelengths_current = False
                break
            wavelength_nm = int(wavelength_text)
            expected_wavelength_um = (
                experimental_jones_propagation_wavelength_um(wavelength_nm)
                if dataset_key == "experiment"
                else optical_parameters(wavelength_nm).wavelength_um
            )
            if not np.isclose(
                float(item["propagation_wavelength_um"]),
                expected_wavelength_um,
                rtol=0.0,
                atol=1.0e-12,
            ):
                propagation_wavelengths_current = False
                break
        if (
            not inputs_current
            or not pupil_current
            or not propagation_wavelengths_current
        ):
            return False
        if path.name == "report.json":
            return (
                report.get("schema_version")
                == "axisymmetric-jones-coefficients-v3"
            )
        return _npz_schema(path) == expected[path.name]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _experimental_modes_output_is_current(path: Path) -> bool:
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                schema = np.asarray(archive["schema_version"]).item()
                sources = [str(value) for value in archive["source_paths"]]
                digests = [str(value) for value in archive["source_sha256"]]
        else:
            report = json.loads(path.read_text(encoding="utf-8"))
            schema = report["schema_version"]
            sources = report["source_paths"]
            digests = report["source_sha256"]
        return bool(
            schema == "experimental-angular-modes-v3"
            and len(sources) == len(digests)
            and all(
                _sha256(PROJECT_ROOT / source) == digest
                for source, digest in zip(sources, digests, strict=True)
            )
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _projection_convergence_output_is_current(path: Path) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        sources = report["sources"]
        source_digests = report["source_sha256"]
        implementations = report["implementation_sha256"]
        configuration = json.dumps(
            report["configuration"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return bool(
            report.get("schema_version")
            == "covariant-projection-interpolation-convergence-520-v1"
            and hashlib.sha256(configuration).hexdigest()
            == report["configuration_sha256"]
            and all(
                source_digests[key] == item["sha256"]
                and item["sha256"] == _sha256(PROJECT_ROOT / item["path"])
                for key, item in sources.items()
            )
            and bool(implementations)
            and all(
                digest == _sha256(PROJECT_ROOT / implementation)
                for implementation, digest in implementations.items()
            )
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _covariant_residual_output_is_current(path: Path) -> bool:
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(
                    str(np.asarray(archive["metadata_json"]).item())
                )
                repeatability = json.loads(
                    str(np.asarray(archive["repeatability_json"]).item())
                )
        else:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            repeatability = metadata["repeatability"]
        alignment = PROJECT_ROOT / metadata["surface_alignment_report"]
        experiment_manifest = PROJECT_ROOT / metadata["experiment_manifest"]
        repeatability_indices = {
            int(item["index"]) for item in repeatability
        }
        repeatability_sources = {
            str(item["source"]) for item in repeatability
        }
        return bool(
            metadata.get("schema_version")
            == "covariant-residual-decomposition-v4"
            and int(metadata["repeatability_records"])
            == EXPECTED_EXPERIMENTAL_ACQUISITIONS
            and len(repeatability)
            == EXPECTED_EXPERIMENTAL_ACQUISITIONS
            and repeatability_indices
            == set(range(1, EXPECTED_EXPERIMENTAL_ACQUISITIONS + 1))
            and len(repeatability_sources)
            == EXPECTED_EXPERIMENTAL_ACQUISITIONS
            and metadata.get("source_sha256")
            == _sha256(PROJECT_ROOT / metadata["source"])
            and metadata.get("experiment_manifest_sha256")
            == _sha256(experiment_manifest)
            and metadata.get("surface_alignment_report_sha256")
            == _sha256(alignment)
            and all(
                item["source_sha256"]
                == _sha256(PROJECT_ROOT / item["source"])
                for item in repeatability
            )
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _tilted_output_is_current(path: Path) -> bool:
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                schema = np.asarray(archive["schema_version"]).item()
                sources = [str(value) for value in archive["source_paths"]]
                digests = [str(value) for value in archive["source_sha256"]]
                director_source = str(
                    np.asarray(archive["reversed_core_source"]).item()
                )
                director_digest = str(
                    np.asarray(
                        archive["reversed_core_source_sha256"]
                    ).item()
                )
        else:
            report = json.loads(path.read_text(encoding="utf-8"))
            schema = report["schema_version"]
            sources = report["source_paths"]
            digests = report["source_sha256"]
            director_source = report["reversed_core_axis"]["source"]
            director_digest = report["reversed_core_axis"][
                "source_sha256"
            ]
        return bool(
            schema == "tilted-angular-modes-520-v6"
            and len(sources) == len(digests)
            and all(
                _sha256(PROJECT_ROOT / source) == digest
                for source, digest in zip(sources, digests, strict=True)
            )
            and _sha256(PROJECT_ROOT / director_source) == director_digest
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _main_analysis_output_is_current(path: Path) -> bool:
    return _metadata_inputs_output_is_current("main-symmetry-520-v6")(path)


FOCUS_DISPLAY_REPORT = (
    PROJECT_ROOT
    / "data/analysis/results_toron25um_50X_focus_recovery_soft"
    / "focus_i90_comparison_report.json"
)


def _focus_display_output_is_current(path: Path) -> bool:
    try:
        metadata = json.loads(FOCUS_DISPLAY_REPORT.read_text(encoding="utf-8"))
        audit_report_path = FOCUS_DISPLAY_REPORT.with_name(
            "figure1_harmonic_audit.json"
        )
        audit_archive_path = FOCUS_DISPLAY_REPORT.with_name(
            "figure1_harmonic_audit.npz"
        )
        audit_figure_path = FOCUS_DISPLAY_REPORT.with_name(
            "figure1_harmonic_audit.pdf"
        )
        audit_report = json.loads(
            audit_report_path.read_text(encoding="utf-8")
        )
        with np.load(audit_archive_path, allow_pickle=False) as archive:
            audit_schema = str(
                np.asarray(archive["schema_version"]).item()
            )
            audit_archive_metadata = json.loads(
                str(np.asarray(archive["metadata_json"]).item())
            )
            radius_um = np.asarray(archive["radius_um"], dtype=np.float64)
            modes = np.asarray(archive["modes"], dtype=np.int64)
            direct = np.asarray(archive["direct_2d_coefficients"])
            axisymmetric = np.asarray(archive["axisymmetric_coefficients"])
        expected_audit_outputs = {
            "archive": str(
                audit_archive_path.relative_to(PROJECT_ROOT).as_posix()
            ),
            "figure": str(
                audit_figure_path.relative_to(PROJECT_ROOT).as_posix()
            ),
            "report": str(
                audit_report_path.relative_to(PROJECT_ROOT).as_posix()
            ),
        }
        return bool(
            path.exists()
            and metadata.get("schema_version")
            == "hdr-holography-focus-i90-comparison-v6"
            and _inputs_are_current(metadata)
            and audit_report.get("schema_version")
            == "figure1-harmonic-audit-v2"
            and _inputs_are_current(audit_report)
            and audit_schema == "figure1-harmonic-audit-v2"
            and audit_archive_metadata == audit_report
            and _inputs_are_current(audit_archive_metadata)
            and metadata["outputs"]["harmonic_audit"]
            == expected_audit_outputs
            and metadata["holography_harmonics"][
                "two_dimensional_audit"
            ]
            == audit_report
            and radius_um.ndim == 1
            and radius_um.size > 1
            and np.all(np.isfinite(radius_um))
            and np.array_equal(modes, np.asarray((0, 2, 4)))
            and direct.shape == axisymmetric.shape
            and direct.ndim == 3
            and direct.shape[1:] == (3, radius_um.size)
            and np.all(np.isfinite(direct))
            and np.all(np.isfinite(axisymmetric))
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


STEPS = (
    # Test the rotational selection rule directly on the three coherently
    # averaged experimental Jones fields.  This reads the unprojected fields
    # and also records the wavelength-resolved crossed-intensity harmonics.
    ReproductionStep(
        "Measure experimental angular modes",
        "Decompose the complete unprojected 450-, 520-, and 660-nm Jones "
        "fields and quantify their covariance-forbidden residuals.",
        (*PYTHON, "code/scripts/analyze_experimental_angular_modes.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/experimental_angular_modes/experimental_angular_modes.npz",
            PROJECT_ROOT
            / "data/analysis/experimental_angular_modes/report.json",
        ),
        output_is_current=_experimental_modes_output_is_current,
    ),
    ReproductionStep(
        "Audit background and center sensitivity",
        "Compare Huber and arithmetic-mean Jones backgrounds and repeat the "
        "520-nm mode measurement at fixed ±0.10 and ±0.25 um center offsets.",
        (*PYTHON, "code/scripts/audit_analysis_sensitivity.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/numerical_validation"
            / "experimental_520_background_center_sensitivity.json",
        ),
        output_is_current=_single_source_json_is_current(
            "experimental-analysis-sensitivity-520-v2"
        ),
    ),
    ReproductionStep(
        "Audit covariant-projector interpolation",
        "Compare linear, cubic, and quintic complex-field sampling for the "
        "saved 520-nm experiment and both optical models; no field is "
        "propagated or simulated.",
        (
            *PYTHON,
            "code/scripts/audit_covariant_projection_convergence.py",
        ),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/numerical_validation"
            / "covariant_projection_interpolation_520.json",
        ),
        output_is_current=_projection_convergence_output_is_current,
    ),
    # Measure the geometric and optically weighted layer helicity of the director.
    ReproductionStep(
        "Analyze director helicity",
        "Calculate layer-resolved helicity and its longitudinal gradient.",
        (*PYTHON, "code/scripts/director_helicity_analysis.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/director_helicity_soft/director_helicity_analysis.npz",
            PROJECT_ROOT
            / "data/analysis/director_helicity_soft/director_helicity_analysis.json",
        ),
        output_is_current=_director_helicity_output_is_current,
    ),
    # Project experiment and both models onto the exactly covariant radial
    # sector, store its four complex channels, propagate them, and append the
    # exact five planes used by the publication figures.
    ReproductionStep(
        "Compute radial Jones profiles and propagation",
        "Extract the axisymmetric coefficients, propagate them over focus, "
        "evaluate symmetry observables, and store the five publication planes.",
        (
            *PYTHON,
            "code/scripts/analyze_axisymmetric_jones_evolution.py",
            "--mode",
            "compute",
        ),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/axisymmetric_jones_evolution_soft"
            / "axisymmetric_jones_coefficients.npz",
            PROJECT_ROOT
            / "data/analysis/axisymmetric_jones_evolution_soft"
            / "axisymmetric_jones_evolution.npz",
            PROJECT_ROOT
            / "data/analysis/axisymmetric_jones_evolution_soft"
            / "axisymmetric_symmetry_axes_520.npz",
            PROJECT_ROOT
            / "data/analysis/axisymmetric_jones_evolution_soft/report.json",
        ),
        output_is_current=_axisymmetric_output_is_current,
    ),
    # Compare common-520-nm propagation with the fixed 518.2/522-nm control.
    ReproductionStep(
        "Audit input-column wavelength sensitivity",
        "Compare common-520-nm full-Jones propagation with independently "
        "propagated input columns at 518.2 and 522 nm.",
        (
            *PYTHON,
            "code/scripts/audit_jones_column_wavelength_sensitivity.py",
        ),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/numerical_validation"
            / "experimental_520_column_wavelength_sensitivity.json",
        ),
        output_is_current=_metadata_inputs_output_is_current(
            "experimental-column-wavelength-sensitivity-520-v2"
        ),
    ),
    # Split the measured crossed intensity into the covariant contribution,
    # its interference with the discarded Jones field, and the residual
    # intensity.  This directly tests the claim that the measured H2 is not an
    # artifact of the noncovariant remainder.
    ReproductionStep(
        "Decompose covariant and residual intensity harmonics",
        "Project the surface-aligned 520-nm Jones field with exact spin-order "
        "locking and calculate the covariant, cross, and residual H2/H4 terms "
        "for the average and all nine acquisitions.",
        (
            *PYTHON,
            "code/scripts/analyze_covariant_residual_decomposition.py",
        ),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/covariant_residual_520_soft"
            / "covariant_residual_decomposition.npz",
            PROJECT_ROOT
            / "data/analysis/covariant_residual_520_soft"
            / "report.json",
        ),
        output_is_current=_covariant_residual_output_is_current,
    ),
    # Quantify the positive control without rerunning either optical solver:
    # decompose the four saved 520-nm Jones maps at the exit surface and after
    # homogeneous propagation to the common annular-focus plane.
    ReproductionStep(
        "Compare symmetric and inclined angular modes",
        "Measure signed exit-surface Jones-field orders for both optical "
        "models and both supplied directors.",
        (*PYTHON, "code/scripts/analyze_tilted_angular_modes.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/tilted_angular_modes_520_soft/tilted_angular_modes.npz",
            PROJECT_ROOT
            / "data/analysis/tilted_angular_modes_520_soft/report.json",
        ),
        output_is_current=_tilted_output_is_current,
    ),
    ReproductionStep(
        "Measure 520-nm phase-step sensitivity",
        "Compare the saved production and half-phase-step marching-wave "
        "fields without invoking the optical solver.",
        (*PYTHON, "code/scripts/analyze_phase_step_convergence.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/phase_step_convergence_520_soft/report.json",
        ),
        output_is_current=_metadata_inputs_output_is_current(
            "phase-step-convergence-520-v3"
        ),
    ),
    # Assemble the compact numerical archive consumed by the main figure
    # builders.  This also measures repeatability from the 27 original fields.
    ReproductionStep(
        "Build the compact 520-nm symmetry analysis",
        "Combine experimental, modeled, pupil, propagation, helicity-control, "
        "and repeatability metrics without rendering publication figures.",
        (
            *PYTHON,
            "code/scripts/build_letter_figures.py",
            "--analyze",
            "--analysis-only",
        ),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/main_symmetry_520_soft/main_symmetry_520.npz",
            PROJECT_ROOT / "data/analysis/main_symmetry_520_soft/report.json",
        ),
        output_is_current=_main_analysis_output_is_current,
    ),
    # Rendered TopoVec panels are a cache used by the final compositor.  The
    # physical HDR arrays are never filled; only the display copy repairs
    # narrow zero-confidence gaps to avoid black NaN arcs.
    ReproductionStep(
        "Build the matched direct-POM display cache",
        "Render the five measured and propagated I90 planes and record every "
        "display-only invalid-pixel repair.",
        (*PYTHON, "code/scripts/render_hdr_holography_focus_comparison.py"),
        outputs=(
            FOCUS_DISPLAY_REPORT,
            PROJECT_ROOT
            / "data/analysis/results_toron25um_50X_focus_recovery_soft"
            / "figure1_harmonic_audit.json",
            PROJECT_ROOT
            / "data/analysis/results_toron25um_50X_focus_recovery_soft"
            / "figure1_harmonic_audit.npz",
            *tuple(
                PROJECT_ROOT
                / "data/analysis/results_toron25um_50X_focus_recovery_soft"
                / "topovec_i90_focus_panels"
                / f"frame_{frame:04d}_{kind}.png"
                for frame in range(45, 50)
                for kind in ("direct", "propagated")
            ),
            *tuple(
                PROJECT_ROOT
                / "data/analysis/results_toron25um_50X_focus_recovery_soft"
                / "topovec_i90_holography_propagation"
                / f"plane_{index}_propagated.png"
                for index in range(1, 6)
            ),
        ),
        output_is_current=_focus_display_output_is_current,
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    parser.add_argument("--dry-run", action="store_true", help="Check existing outputs without running calculations")
    args = parser.parse_args(argv)
    run_steps(STEPS, project_root=PROJECT_ROOT, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
