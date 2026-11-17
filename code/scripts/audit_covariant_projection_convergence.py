#!/usr/bin/env python3
"""Audit interpolation convergence of the exact 520-nm Jones projector.

This standalone diagnostic reads the saved experimental, local-Jones, and
marching-wave Jones archives.  It performs no optical simulation,
propagation, pupil filtering, center fitting, or source-data modification.
For each field it changes only the interpolation of the complex Cartesian
Jones field onto the common polar grid: linear, cubic, and quintic.

The JSON report records source hashes, physical centers, sampling grids, the
background-subtracted exact-projection residual, and the projected crossed
intensity V2 and V4.  Cubic and linear results are compared with quintic as
the highest-order reference used by this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from toronoptics.analysis_specs import (
    PUBLICATION_BACKGROUND,
    PUBLICATION_HARMONICS,
    CenterSpec,
)
from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    decimate_centered_jones,
    experimental_array_center,
    experimental_average_path,
    simulated_array_center,
    simulated_jones_path,
)
from toronoptics.axisymmetric_jones import (
    annular_jones_background,
    crossed_intensity_harmonics,
    integrated_harmonic_visibility,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.covariant_jones import (
    SPIN_ALLOWED_ORDERS,
    SPIN_CHANNEL_LABELS,
    project_covariant_jones,
    rotationally_invariant_matrix_part,
)
from toronoptics.experimental_jones import load_jones_archive


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELENGTH_NM = ARTICLE_GREEN_WAVELENGTH_NM
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/analysis/numerical_validation"
    / "covariant_projection_interpolation_520.json"
)

SCHEMA_VERSION = "covariant-projection-interpolation-convergence-520-v1"
INTERPOLATIONS = ("linear", "cubic", "quintic")
REFERENCE_INTERPOLATION = "quintic"
PROJECTION_RADIUS_UM = np.arange(0.0, 45.01, 0.125, dtype=np.float64)
ANGULAR_SAMPLES = 720
RADIAL_INTERPOLATION = "cubic"
RADIAL_EXTRAPOLATION = "nearest"
COMPARISON_METRICS = (
    "relative_residual_norm",
    "projected_V2",
    "projected_V4",
)

SOURCE_PATHS = {
    "experiment": experimental_average_path(PROJECT_ROOT, WAVELENGTH_NM),
    "local_jones": simulated_jones_path(
        PROJECT_ROOT,
        WAVELENGTH_NM,
        "spectral-jones",
    ),
    "marching_wave": simulated_jones_path(
        PROJECT_ROOT,
        WAVELENGTH_NM,
        "marching-wave",
    ),
}


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration() -> dict[str, object]:
    return {
        "wavelength_nm": WAVELENGTH_NM,
        "field_plane": "native plane stored in each Jones archive",
        "propagation": "none",
        "pupil_filter": "none",
        "center_fitting": "none",
        "source_preprocessing": {
            "experiment": (
                "separable [0.25, 0.5, 0.25] low-pass and centered "
                "factor-two decimation"
            ),
            "local_jones": "none",
            "marching_wave": "none",
        },
        "background": PUBLICATION_BACKGROUND.metadata(),
        "background_handling": {
            "residual_norm": (
                "exact projection of J minus the constant Huber background"
            ),
            "projected_visibility": (
                "P[J-background] plus the analytic rotationally invariant "
                "a I + b K part of the constant background"
            ),
            "background_subtracted_before_projection": True,
        },
        "projection_grid": {
            "radius_start_um": float(PROJECTION_RADIUS_UM[0]),
            "radius_stop_um": float(PROJECTION_RADIUS_UM[-1]),
            "radius_step_um": float(np.diff(PROJECTION_RADIUS_UM)[0]),
            "radial_samples": int(PROJECTION_RADIUS_UM.size),
            "angular_samples": ANGULAR_SAMPLES,
            "axis_order": "xy",
        },
        "interpolations": list(INTERPOLATIONS),
        "reference_interpolation": REFERENCE_INTERPOLATION,
        "radial_reconstruction": RADIAL_INTERPOLATION,
        "radial_extrapolation": RADIAL_EXTRAPOLATION,
        "harmonic_analysis": PUBLICATION_HARMONICS.metadata(),
    }


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PROJECT_ROOT / "code/src/toronoptics/analysis_specs.py",
        PROJECT_ROOT / "code/src/toronoptics/article_data.py",
        PROJECT_ROOT / "code/src/toronoptics/polar_fourier.py",
        PROJECT_ROOT / "code/src/toronoptics/covariant_jones.py",
        PROJECT_ROOT / "code/src/toronoptics/axisymmetric_jones.py",
    )


def _state_hashes() -> tuple[dict[str, str], dict[str, str]]:
    source_sha256 = {
        key: _sha256(path) for key, path in SOURCE_PATHS.items()
    }
    implementation_sha256 = {
        _relative(path): _sha256(path) for path in _implementation_paths()
    }
    return source_sha256, implementation_sha256


def _report_matches_state(
    report: Mapping[str, object],
    *,
    source_sha256: Mapping[str, str],
    implementation_sha256: Mapping[str, str],
    configuration_sha256: str,
) -> bool:
    """Return whether a parsed report matches every deterministic input."""

    try:
        return bool(
            report.get("schema_version") == SCHEMA_VERSION
            and report.get("source_sha256") == dict(source_sha256)
            and report.get("implementation_sha256")
            == dict(implementation_sha256)
            and report.get("configuration_sha256")
            == str(configuration_sha256)
        )
    except (TypeError, ValueError):
        return False


def _output_is_current(
    path: Path,
    *,
    source_sha256: Mapping[str, str],
    implementation_sha256: Mapping[str, str],
    configuration_sha256: str,
) -> bool:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(report, dict):
        return False
    return _report_matches_state(
        report,
        source_sha256=source_sha256,
        implementation_sha256=implementation_sha256,
        configuration_sha256=configuration_sha256,
    )


def _metric_comparisons(
    metrics_by_interpolation: Mapping[str, Mapping[str, float]],
    *,
    reference: str = REFERENCE_INTERPOLATION,
) -> dict[str, object]:
    """Compare residual/V2/V4 values with one named reference method."""

    if reference not in metrics_by_interpolation:
        raise ValueError(f"Missing reference interpolation {reference!r}")
    methods = tuple(metrics_by_interpolation)
    if set(methods) != set(INTERPOLATIONS):
        raise ValueError(
            f"Expected interpolation results for {INTERPOLATIONS!r}, got {methods!r}"
        )
    result: dict[str, object] = {
        "reference_interpolation": reference,
        "metrics": {},
    }
    metric_result = result["metrics"]
    assert isinstance(metric_result, dict)
    for metric in COMPARISON_METRICS:
        values = {
            method: float(metrics_by_interpolation[method][metric])
            for method in methods
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite {metric} comparison values: {values!r}")
        reference_value = values[reference]
        scale = max(abs(reference_value), 1.0e-15)
        metric_result[metric] = {
            "reference_value": reference_value,
            "by_interpolation": {
                method: {
                    "value": value,
                    "difference_from_reference": value - reference_value,
                    "relative_difference_from_reference": (
                        (value - reference_value) / scale
                    ),
                }
                for method, value in values.items()
            },
            "minimum": min(values.values()),
            "maximum": max(values.values()),
            "absolute_spread": max(values.values()) - min(values.values()),
            "relative_spread_to_reference": (
                (max(values.values()) - min(values.values())) / scale
            ),
        }
    return result


def _maximum_inscribed_radius_um(
    shape: tuple[int, int],
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> float:
    distances_px = (
        float(center_px[0]),
        float(shape[0] - 1) - float(center_px[0]),
        float(center_px[1]),
        float(shape[1] - 1) - float(center_px[1]),
    )
    return float(min(distances_px) * float(pixel_size_um))


def _visibilities(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> dict[str, float]:
    profiles = crossed_intensity_harmonics(
        jones,
        center_px=center_px,
        pixel_size_um=float(pixel_size_um),
        radial_centers_um=PUBLICATION_HARMONICS.radial_centers_um,
        radial_sigma_um=PUBLICATION_HARMONICS.radial_sigma_um,
        modes=PUBLICATION_HARMONICS.modes,
    )
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


def _center_for_source(
    key: str,
    path: Path,
) -> CenterSpec:
    if key == "experiment":
        center = experimental_array_center(path)
        method = (
            "array_center_px from the registered nine-position experimental "
            "average archive"
        )
    else:
        center = simulated_array_center(path)
        method = (
            "array_center_px from simulation metadata, or the documented "
            "director-axis mapping from source and output grid dimensions"
        )
    return CenterSpec(*center, method=method, fitted=False)


def _analyze_source(
    key: str,
    path: Path,
    *,
    source_sha256: str,
) -> dict[str, object]:
    archive = load_jones_archive(path, dtype=np.complex64)
    center_spec = _center_for_source(key, path)
    center = center_spec.center_px
    pixel_size_um = float(archive.pixel_size_um)
    source_shape = archive.jones.shape[:2]
    if key == "experiment":
        jones, center = decimate_centered_jones(
            archive.jones,
            center,
            factor=2,
        )
        pixel_size_um *= 2.0
        center_spec = CenterSpec(
            *center,
            method=(
                "registered nine-position experimental center, transformed "
                "with the documented centered factor-two decimation"
            ),
            fitted=False,
        )
        preprocessing = "centered low-pass decimation by factor 2"
    else:
        jones = archive.jones
        preprocessing = "none"
    maximum_radius_um = _maximum_inscribed_radius_um(
        jones.shape[:2],
        center_px=center,
        pixel_size_um=pixel_size_um,
    )
    if PROJECTION_RADIUS_UM[-1] > maximum_radius_um + 1.0e-10:
        raise ValueError(
            f"{path} supports circles only to {maximum_radius_um:.6g} um, "
            f"less than the requested {PROJECTION_RADIUS_UM[-1]:.6g} um"
        )
    background = annular_jones_background(
        jones,
        center_px=center,
        pixel_size_um=pixel_size_um,
        spec=PUBLICATION_BACKGROUND,
    )
    signal = np.empty_like(jones, dtype=np.complex64)
    np.subtract(
        jones,
        np.asarray(background, dtype=np.complex64),
        out=signal,
    )
    invariant_background = rotationally_invariant_matrix_part(background)
    raw_visibility = _visibilities(
        jones,
        center_px=center,
        pixel_size_um=pixel_size_um,
    )

    metrics: dict[str, dict[str, object]] = {}
    comparison_input: dict[str, dict[str, float]] = {}
    for interpolation in INTERPOLATIONS:
        projection = project_covariant_jones(
            signal,
            center_xy_px=center,
            pixel_size_xy_um=pixel_size_um,
            radius_um=PROJECTION_RADIUS_UM,
            angular_samples=ANGULAR_SAMPLES,
            interpolation=interpolation,
            radial_interpolation=RADIAL_INTERPOLATION,
            radial_extrapolation=RADIAL_EXTRAPOLATION,
            axis_order="xy",
        )
        projected_total = (
            projection.covariant_jones
            + invariant_background[None, None, :, :]
        )
        diagnostics = projection.diagnostics
        scalar_metrics = {
            "relative_residual_norm": float(
                diagnostics.relative_residual_norm
            ),
        }
        projected_visibility = _visibilities(
            projected_total,
            center_px=center,
            pixel_size_um=pixel_size_um,
        )
        scalar_metrics["projected_V2"] = projected_visibility["V2"]
        scalar_metrics["projected_V4"] = projected_visibility["V4"]
        comparison_input[interpolation] = scalar_metrics
        metrics[interpolation] = {
            **scalar_metrics,
            "covariant_power_fraction": float(
                diagnostics.covariant_power_fraction
            ),
            "relative_orthogonality": float(
                diagnostics.relative_orthogonality
            ),
            "relative_projection_closure_error": float(
                diagnostics.relative_closure_error
            ),
            "spin_channel_residual_power_fraction": {
                label: {
                    "allowed_order": int(order),
                    "fraction": float(fraction),
                }
                for label, order, fraction in zip(
                    SPIN_CHANNEL_LABELS,
                    SPIN_ALLOWED_ORDERS,
                    diagnostics.channel_residual_fraction,
                    strict=True,
                )
            },
        }

    return {
        "path": _relative(path),
        "sha256": source_sha256,
        "source_field_shape_px": [int(value) for value in source_shape],
        "analyzed_field_shape_px": [int(value) for value in jones.shape[:2]],
        "preprocessing": preprocessing,
        "pixel_size_um": pixel_size_um,
        "center": center_spec.metadata(),
        "maximum_inscribed_radius_um": maximum_radius_um,
        "analyzed_grid": {
            "shape_px": [int(value) for value in jones.shape[:2]],
            "pixel_size_um": pixel_size_um,
            "axis_order": "xy",
            "center_px": [float(value) for value in center],
            "maximum_inscribed_radius_um": maximum_radius_um,
        },
        "background_jones": {
            "real": np.real(background).tolist(),
            "imag": np.imag(background).tolist(),
        },
        "raw_crossed_intensity": raw_visibility,
        "interpolation_results": metrics,
        "comparisons": _metric_comparisons(comparison_input),
    }


def _write_json_atomic(path: Path, report: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    args = parser.parse_args(argv)

    configuration = _configuration()
    configuration_sha256 = _json_sha256(configuration)
    source_sha256, implementation_sha256 = _state_hashes()
    if OUTPUT_PATH.exists() and not args.force:
        if _output_is_current(
            OUTPUT_PATH,
            source_sha256=source_sha256,
            implementation_sha256=implementation_sha256,
            configuration_sha256=configuration_sha256,
        ):
            print(
                f"[projection-convergence] current: {OUTPUT_PATH}",
                flush=True,
            )
            return 0
        refuse_existing_outputs(
            (OUTPUT_PATH,),
            force=False,
            project_root=PROJECT_ROOT,
        )

    sources = {
        key: _analyze_source(
            key,
            path,
            source_sha256=source_sha256[key],
        )
        for key, path in SOURCE_PATHS.items()
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "implementation_sha256": implementation_sha256,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "sources": sources,
        "interpretation": (
            "Differences among rows isolate interpolation of the saved "
            "complex Jones field. All centers, backgrounds, polar grids, "
            "radial reconstruction, and harmonic estimators are fixed."
        ),
    }
    _write_json_atomic(OUTPUT_PATH, report)
    print(f"[projection-convergence] report: {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
