#!/usr/bin/env python3
"""Audit background and fixed-center sensitivity of the 520-nm experiment.

This diagnostic reads the already averaged experimental Jones archive and
does not propagate a field or invoke an optical solver.  It compares the
publication Huber background with the arithmetic mean and repeats the
signed Jones-mode and exact spin-resolved covariance measurements at
prescribed physical offsets from the registered center.  The center is never
fitted inside this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from toronoptics.analysis_specs import (
    BackgroundSpec,
    CenterSpec,
    PUBLICATION_BACKGROUND,
)
from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    experimental_array_center,
    experimental_average_path,
)
from toronoptics.axisymmetric_jones import (
    annular_jones_background,
    jones_angular_mode_power,
    polar_coordinate_grids,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.covariant_jones import project_covariant_polar
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.polar_fourier import PolarSamplingGrid, sample_polar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = experimental_average_path(
    PROJECT_ROOT, ARTICLE_GREEN_WAVELENGTH_NM
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/analysis/numerical_validation"
    / "experimental_520_background_center_sensitivity.json"
)

ANALYSIS_RADIUS_UM = 28.0
RADIAL_STEP_UM = 0.25
ANGULAR_SAMPLES = 720
MODES = np.arange(-5, 6, dtype=np.int16)
ALLOWED_MODES = np.asarray((-2, 0, 2), dtype=np.int16)
COVARIANCE_RADII_UM = np.arange(0.0, 45.01, 0.125, dtype=np.float64)
COVARIANCE_ANGULAR_SAMPLES = 720

PRIMARY_BACKGROUND = PUBLICATION_BACKGROUND
MEAN_BACKGROUND = BackgroundSpec(
    PUBLICATION_BACKGROUND.min_radius_um,
    PUBLICATION_BACKGROUND.max_radius_um,
    estimator="mean",
)

# Cardinal offsets isolate sensitivity to each image axis.  The two magnitudes
# are those used in the numerical review of experimental centering.
CENTER_OFFSETS_UM = (
    (0.0, 0.0),
    (-0.10, 0.0),
    (+0.10, 0.0),
    (0.0, -0.10),
    (0.0, +0.10),
    (-0.25, 0.0),
    (+0.25, 0.0),
    (0.0, -0.25),
    (0.0, +0.25),
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_matrix_json(matrix: np.ndarray) -> dict[str, list[list[float]]]:
    value = np.asarray(matrix, dtype=np.complex128)
    return {
        "real": np.real(value).tolist(),
        "imag": np.imag(value).tolist(),
    }


def _spectrum_metrics(
    jones: np.ndarray,
    *,
    background: np.ndarray,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
) -> dict[str, object]:
    spectrum = jones_angular_mode_power(
        jones - background,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        radii_um=radii_um,
        modes=MODES,
        angular_samples=ANGULAR_SAMPLES,
    )
    fractions = np.asarray(
        spectrum.mode_power / spectrum.total_power, dtype=np.float64
    )
    allowed_fraction = float(
        np.sum(fractions[np.isin(spectrum.modes, ALLOWED_MODES)])
    )
    forbidden_fraction = max(0.0, 1.0 - allowed_fraction)
    return {
        "background_jones": _complex_matrix_json(background),
        "mode_power_fraction": fractions.tolist(),
        "captured_power_fraction": float(spectrum.captured_power_fraction),
        "allowed_power_fraction": allowed_fraction,
        "forbidden_power_fraction": forbidden_fraction,
        "forbidden_amplitude_fraction": float(np.sqrt(forbidden_fraction)),
    }


def _center_with_offset(
    center_px: tuple[float, float],
    offset_um: tuple[float, float],
    *,
    pixel_size_um: float,
) -> tuple[float, float]:
    return (
        float(center_px[0]) + float(offset_um[0]) / float(pixel_size_um),
        float(center_px[1]) + float(offset_um[1]) / float(pixel_size_um),
    )


def _covariance_residual_norm(
    jones: np.ndarray,
    *,
    background: np.ndarray,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> float:
    signal = np.asarray(jones, dtype=np.complex64) - np.asarray(
        background, dtype=np.complex64
    )
    grid = PolarSamplingGrid.uniform(
        center_xy_px=center_px,
        pixel_size_xy_um=pixel_size_um,
        radius_um=COVARIANCE_RADII_UM,
        angular_samples=COVARIANCE_ANGULAR_SAMPLES,
        axis_order="xy",
        boundary="raise",
    )
    sampled = sample_polar(signal, grid, interpolation="cubic")
    return float(
        project_covariant_polar(
            sampled,
            phi_rad=grid.phi_rad,
            radius_um=grid.radius_um,
        ).diagnostics.relative_residual_norm
    )


def _background_difference_scale(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    primary_background: np.ndarray,
    arithmetic_mean_background: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(arithmetic_mean_background) - np.asarray(primary_background)
    radius_um, _phi = polar_coordinate_grids(
        jones.shape[:2],
        center_px=center_px,
        pixel_size_um=pixel_size_um,
    )
    selected = radius_um <= ANALYSIS_RADIUS_UM
    signal = np.asarray(jones[selected], dtype=np.complex128) - primary_background
    signal_rms = float(
        np.sqrt(np.mean(np.sum(np.abs(signal) ** 2, axis=(1, 2))))
    )
    absolute = float(np.linalg.norm(difference))
    return {
        "frobenius": absolute,
        "relative_to_primary_background": (
            absolute / max(float(np.linalg.norm(primary_background)), 1.0e-300)
        ),
        "relative_to_primary_delta_j_rms_in_analysis_disk": (
            absolute / max(signal_rms, 1.0e-300)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    args = parser.parse_args(argv)
    refuse_existing_outputs(
        (OUTPUT_PATH,),
        force=args.force,
        project_root=PROJECT_ROOT,
    )

    archive = load_jones_archive(SOURCE_PATH, dtype=np.complex64)
    pixel_size_um = float(archive.pixel_size_um)
    registered_center = experimental_array_center(SOURCE_PATH)
    center_spec = CenterSpec(
        *registered_center,
        method=(
            "fixed array center from the nine-position experimental "
            "registration archive"
        ),
        fitted=False,
    )
    radii_um = np.arange(
        0.0,
        ANALYSIS_RADIUS_UM + 0.5 * RADIAL_STEP_UM,
        RADIAL_STEP_UM,
        dtype=np.float64,
    )

    cases: list[dict[str, object]] = []
    for offset_um in CENTER_OFFSETS_UM:
        center_px = _center_with_offset(
            registered_center,
            offset_um,
            pixel_size_um=pixel_size_um,
        )
        primary_background = annular_jones_background(
            archive.jones,
            center_px=center_px,
            pixel_size_um=pixel_size_um,
            spec=PRIMARY_BACKGROUND,
        )
        arithmetic_mean_background = annular_jones_background(
            archive.jones,
            center_px=center_px,
            pixel_size_um=pixel_size_um,
            spec=MEAN_BACKGROUND,
        )
        primary = _spectrum_metrics(
            archive.jones,
            background=primary_background,
            center_px=center_px,
            pixel_size_um=pixel_size_um,
            radii_um=radii_um,
        )
        primary["covariant_residual_norm"] = _covariance_residual_norm(
            archive.jones,
            background=primary_background,
            center_px=center_px,
            pixel_size_um=pixel_size_um,
        )
        arithmetic_mean = _spectrum_metrics(
            archive.jones,
            background=arithmetic_mean_background,
            center_px=center_px,
            pixel_size_um=pixel_size_um,
            radii_um=radii_um,
        )
        cases.append(
            {
                "offset_um": [float(value) for value in offset_um],
                "center_px": [float(value) for value in center_px],
                "primary_huber": primary,
                "arithmetic_mean": arithmetic_mean,
                "mean_minus_huber": {
                    "background": _background_difference_scale(
                        archive.jones,
                        center_px=center_px,
                        pixel_size_um=pixel_size_um,
                        primary_background=primary_background,
                        arithmetic_mean_background=arithmetic_mean_background,
                    ),
                    "forbidden_amplitude_fraction": (
                        float(arithmetic_mean["forbidden_amplitude_fraction"])
                        - float(primary["forbidden_amplitude_fraction"])
                    ),
                },
            }
        )

    nominal_primary = float(
        cases[0]["primary_huber"]["forbidden_amplitude_fraction"]
    )
    nominal_mean = float(
        cases[0]["arithmetic_mean"]["forbidden_amplitude_fraction"]
    )
    report = {
        "schema_version": "experimental-analysis-sensitivity-520-v2",
        "source": _relative(SOURCE_PATH),
        "source_sha256": _sha256(SOURCE_PATH),
        "wavelength_nm": ARTICLE_GREEN_WAVELENGTH_NM,
        "field": "coherently averaged experimental Jones field",
        "projection": {
            "signed_order_spectrum": "none",
            "covariance_sensitivity": (
                "exact circular-basis spin--order projector used only to "
                "measure epsilon_cov"
            ),
        },
        "propagation": "none",
        "registered_center": center_spec.metadata(),
        "center_policy": (
            "registered center plus prescribed Cartesian offsets; no center "
            "is fitted to optical modes in this audit"
        ),
        "pixel_size_um": pixel_size_um,
        "analysis_radius_um": ANALYSIS_RADIUS_UM,
        "mode_sampling": {
            "radial_step_um": RADIAL_STEP_UM,
            "angular_samples": ANGULAR_SAMPLES,
            "modes": MODES.tolist(),
            "allowed_modes": ALLOWED_MODES.tolist(),
        },
        "covariance_sampling": {
            "radius_um": [
                float(COVARIANCE_RADII_UM[0]),
                float(COVARIANCE_RADII_UM[-1]),
            ],
            "radial_step_um": float(np.diff(COVARIANCE_RADII_UM)[0]),
            "angular_samples": COVARIANCE_ANGULAR_SAMPLES,
            "complex_field_interpolation": "cubic",
        },
        "background": {
            "primary": PRIMARY_BACKGROUND.metadata(),
            "arithmetic_mean": MEAN_BACKGROUND.metadata(),
        },
        "cases": cases,
        "summary": {
            "nominal_forbidden_amplitude_fraction": {
                "primary_huber": nominal_primary,
                "arithmetic_mean": nominal_mean,
                "mean_minus_huber": nominal_mean - nominal_primary,
            },
            "nominal_covariant_residual_norm": float(
                cases[0]["primary_huber"]["covariant_residual_norm"]
            ),
            "maximum_absolute_center_induced_change_from_nominal": {
                "primary_huber": max(
                    abs(
                        float(case["primary_huber"][
                            "forbidden_amplitude_fraction"
                        ])
                        - nominal_primary
                    )
                    for case in cases[1:]
                ),
                "arithmetic_mean": max(
                    abs(
                        float(case["arithmetic_mean"][
                            "forbidden_amplitude_fraction"
                        ])
                        - nominal_mean
                    )
                    for case in cases[1:]
                ),
                "covariant_residual_norm": max(
                    abs(
                        float(
                            case["primary_huber"][
                                "covariant_residual_norm"
                            ]
                        )
                        - float(
                            cases[0]["primary_huber"][
                                "covariant_residual_norm"
                            ]
                        )
                    )
                    for case in cases[1:]
                ),
            },
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[analysis-sensitivity] report: {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
