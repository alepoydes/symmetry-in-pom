#!/usr/bin/env python3
"""Measure signed angular modes of the three experimental Jones fields.

This lightweight post-processing step reads only the coherently averaged
experimental Jones archives.  It does not propagate a field or invoke either
optical solver.  Projection is used only as a diagnostic norm: the displayed
signed spectra themselves remain the unprojected measured fields.

For each wavelength the script:

1. subtracts the complex Jones background measured in the outer annulus;
2. decomposes the complete laboratory-basis Jones perturbation into signed
   angular orders and integrates every matrix component over radius;
3. reports the amplitude outside the covariance-allowed orders ``0,+/-2``;
4. separately reports the exact spin-resolved distance to rotational
   covariance, which also tests which Jones channel contains each order;
5. evaluates the already defined crossed-intensity harmonic amplitudes over
   the same 4--28 um interval used for the main 520-nm comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from toronoptics.article_data import (
    ARTICLE_WAVELENGTHS_NM,
    experimental_array_center,
    experimental_average_path,
)
from toronoptics.analysis_specs import (
    BackgroundSpec,
    CenterSpec,
    PUBLICATION_BACKGROUND,
    PUBLICATION_HARMONICS,
)
from toronoptics.axisymmetric_jones import (
    annular_jones_background,
    crossed_intensity_harmonics,
    integrated_harmonic_visibility,
    jones_angular_mode_power,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.covariant_jones import (
    SPIN_ALLOWED_ORDERS,
    SPIN_CHANNEL_LABELS,
    project_covariant_polar,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.polar_fourier import PolarSamplingGrid, sample_polar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT / "data/analysis/experimental_angular_modes"
)
ANALYSIS_PATH = OUTPUT_DIR / "experimental_angular_modes.npz"
REPORT_PATH = OUTPUT_DIR / "report.json"

ANALYSIS_RADIUS_UM = 28.0
MODE_RADIAL_STEP_UM = 0.25
MODE_ANGULAR_SAMPLES = 720
DISPLAY_MODES = np.arange(-5, 6, dtype=np.int16)
ALLOWED_MODES = np.asarray((-2, 0, 2), dtype=np.int16)
MINIMUM_CAPTURED_POWER_FRACTION = 0.995
COVARIANCE_RADII_UM = np.arange(0.0, 45.01, 0.125, dtype=np.float64)
COVARIANCE_ANGULAR_SAMPLES = 720
COVARIANCE_INTERPOLATION = "cubic"

BACKGROUND_SPEC = PUBLICATION_BACKGROUND
MEAN_BACKGROUND_SPEC = BackgroundSpec(
    PUBLICATION_BACKGROUND.min_radius_um,
    PUBLICATION_BACKGROUND.max_radius_um,
    estimator="mean",
)
HARMONIC_SPEC = PUBLICATION_HARMONICS


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


def _mode_metrics(
    jones: np.ndarray,
    *,
    background: np.ndarray,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
) -> dict[str, np.ndarray | float]:
    spectrum = jones_angular_mode_power(
        jones - background,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        radii_um=radii_um,
        modes=DISPLAY_MODES,
        angular_samples=MODE_ANGULAR_SAMPLES,
    )
    fractions = spectrum.mode_power / spectrum.total_power
    amplitudes = np.sqrt(np.maximum(fractions, 0.0))
    allowed = np.isin(spectrum.modes, ALLOWED_MODES)
    allowed_fraction = float(np.sum(fractions[allowed]))
    forbidden_fraction = max(0.0, 1.0 - allowed_fraction)
    return {
        "mode_power_fraction": fractions,
        "mode_amplitude_fraction": amplitudes,
        "total_power": float(spectrum.total_power),
        "captured_power_fraction": float(spectrum.captured_power_fraction),
        "allowed_power_fraction": allowed_fraction,
        "forbidden_power_fraction": forbidden_fraction,
        "forbidden_amplitude_fraction": float(np.sqrt(forbidden_fraction)),
    }


def _covariance_metrics(
    jones: np.ndarray,
    *,
    background: np.ndarray,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> dict[str, np.ndarray | float]:
    """Return the exact polar spin--order projection diagnostics.

    The constant Jones background is removed before measuring the norm.  No
    projected Cartesian image is reconstructed, because this diagnostic needs
    only the orthogonal angular projection on the sampled polar disk.
    """

    signal = np.asarray(jones, dtype=np.complex64) - np.asarray(
        background, dtype=np.complex64
    )
    grid = PolarSamplingGrid.uniform(
        center_xy_px=center_px,
        pixel_size_xy_um=float(pixel_size_um),
        radius_um=COVARIANCE_RADII_UM,
        angular_samples=COVARIANCE_ANGULAR_SAMPLES,
        axis_order="xy",
        boundary="raise",
    )
    sampled = sample_polar(
        signal,
        grid,
        interpolation=COVARIANCE_INTERPOLATION,
    )
    diagnostics = project_covariant_polar(
        sampled,
        phi_rad=grid.phi_rad,
        radius_um=grid.radius_um,
    ).diagnostics
    return {
        "relative_residual_norm": float(diagnostics.relative_residual_norm),
        "residual_power_fraction": float(
            diagnostics.relative_residual_norm**2
        ),
        "covariant_power_fraction": float(
            diagnostics.covariant_power_fraction
        ),
        "channel_residual_power_fraction": np.asarray(
            diagnostics.channel_residual_fraction,
            dtype=np.float64,
        ),
        "relative_orthogonality": float(
            diagnostics.relative_orthogonality
        ),
        "relative_closure_error": float(
            diagnostics.relative_closure_error
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    args = parser.parse_args(argv)
    refuse_existing_outputs(
        (ANALYSIS_PATH, REPORT_PATH),
        force=args.force,
        project_root=PROJECT_ROOT,
    )

    radii_um = np.arange(
        0.0,
        ANALYSIS_RADIUS_UM + 0.5 * MODE_RADIAL_STEP_UM,
        MODE_RADIAL_STEP_UM,
        dtype=np.float64,
    )
    wavelength_nm = np.asarray(ARTICLE_WAVELENGTHS_NM, dtype=np.int16)
    mode_power_fraction = np.empty(
        (wavelength_nm.size, DISPLAY_MODES.size), dtype=np.float64
    )
    mode_amplitude_fraction = np.empty_like(mode_power_fraction)
    total_power = np.empty(wavelength_nm.size, dtype=np.float64)
    captured_power_fraction = np.empty(wavelength_nm.size, dtype=np.float64)
    allowed_power_fraction = np.empty(wavelength_nm.size, dtype=np.float64)
    forbidden_power_fraction = np.empty(wavelength_nm.size, dtype=np.float64)
    forbidden_amplitude_fraction = np.empty(
        wavelength_nm.size, dtype=np.float64
    )
    mean_mode_power_fraction = np.empty_like(mode_power_fraction)
    mean_mode_amplitude_fraction = np.empty_like(mode_amplitude_fraction)
    mean_total_power = np.empty_like(total_power)
    mean_captured_power_fraction = np.empty_like(captured_power_fraction)
    mean_allowed_power_fraction = np.empty_like(allowed_power_fraction)
    mean_forbidden_power_fraction = np.empty_like(forbidden_power_fraction)
    mean_forbidden_amplitude_fraction = np.empty_like(
        forbidden_amplitude_fraction
    )
    integrated_visibility = np.empty(
        (wavelength_nm.size, 2), dtype=np.float64
    )
    covariant_residual_norm = np.empty(
        wavelength_nm.size, dtype=np.float64
    )
    covariant_residual_power_fraction = np.empty_like(
        covariant_residual_norm
    )
    covariant_power_fraction = np.empty_like(covariant_residual_norm)
    covariant_channel_residual_power_fraction = np.empty(
        (wavelength_nm.size, len(SPIN_CHANNEL_LABELS)),
        dtype=np.float64,
    )
    covariant_relative_orthogonality = np.empty_like(
        covariant_residual_norm
    )
    covariant_relative_closure_error = np.empty_like(
        covariant_residual_norm
    )
    centers_px = np.empty((wavelength_nm.size, 2), dtype=np.float64)
    pixel_size_um = np.empty(wavelength_nm.size, dtype=np.float64)
    background_jones = np.empty(
        (wavelength_nm.size, 2, 2), dtype=np.complex128
    )
    mean_background_jones = np.empty_like(background_jones)
    source_paths: list[str] = []
    source_sha256: list[str] = []
    center_specs: list[CenterSpec] = []

    for wavelength_index, wavelength in enumerate(wavelength_nm):
        source = experimental_average_path(PROJECT_ROOT, int(wavelength))
        print(f"[experimental-modes] load {source}", flush=True)
        archive = load_jones_archive(source, dtype=np.complex64)
        center = experimental_array_center(source)
        center_spec = CenterSpec(
            *center,
            method=(
                "fixed array center from the nine-position experimental "
                "registration archive"
            ),
            fitted=False,
        )
        center_specs.append(center_spec)
        pixel_size = float(archive.pixel_size_um)
        background = annular_jones_background(
            archive.jones,
            center_px=center,
            pixel_size_um=pixel_size,
            spec=BACKGROUND_SPEC,
        )
        mean_background = annular_jones_background(
            archive.jones,
            center_px=center,
            pixel_size_um=pixel_size,
            spec=MEAN_BACKGROUND_SPEC,
        )

        # This is the direct laboratory-basis spectrum of the measured field.
        # In particular, no co-rotating average or covariant reconstruction is
        # used before measuring the forbidden orders.
        primary = _mode_metrics(
            archive.jones,
            background=background,
            center_px=center,
            pixel_size_um=pixel_size,
            radii_um=radii_um,
        )
        mean_sensitivity = _mode_metrics(
            archive.jones,
            background=mean_background,
            center_px=center,
            pixel_size_um=pixel_size,
            radii_um=radii_um,
        )
        covariance = _covariance_metrics(
            archive.jones,
            background=background,
            center_px=center,
            pixel_size_um=pixel_size,
        )
        for estimator, metrics in (
            ("Huber", primary),
            ("arithmetic-mean", mean_sensitivity),
        ):
            captured = float(metrics["captured_power_fraction"])
            if captured < MINIMUM_CAPTURED_POWER_FRACTION:
                raise RuntimeError(
                    f"{int(wavelength)} nm ({estimator} background): displayed "
                    f"modes capture only {captured:.6%} of the Jones power"
                )

        # These are the same radial-L2 crossed-intensity amplitudes used in
        # Eq. (integrated-visibility) and in the primary 520-nm comparison.
        harmonic_profiles = crossed_intensity_harmonics(
            archive.jones,
            center_px=center,
            pixel_size_um=pixel_size,
            radial_centers_um=HARMONIC_SPEC.radial_centers_um,
            radial_sigma_um=HARMONIC_SPEC.radial_sigma_um,
            modes=HARMONIC_SPEC.modes,
        )
        integrated_visibility[wavelength_index] = [
            integrated_harmonic_visibility(
                harmonic_profiles,
                mode=mode,
                min_radius_um=HARMONIC_SPEC.integration_min_um,
                max_radius_um=HARMONIC_SPEC.integration_max_um,
            )
            for mode in (2, 4)
        ]
        covariant_residual_norm[wavelength_index] = covariance[
            "relative_residual_norm"
        ]
        covariant_residual_power_fraction[wavelength_index] = covariance[
            "residual_power_fraction"
        ]
        covariant_power_fraction[wavelength_index] = covariance[
            "covariant_power_fraction"
        ]
        covariant_channel_residual_power_fraction[wavelength_index] = (
            covariance["channel_residual_power_fraction"]
        )
        covariant_relative_orthogonality[wavelength_index] = covariance[
            "relative_orthogonality"
        ]
        covariant_relative_closure_error[wavelength_index] = covariance[
            "relative_closure_error"
        ]

        mode_power_fraction[wavelength_index] = primary["mode_power_fraction"]
        mode_amplitude_fraction[wavelength_index] = primary[
            "mode_amplitude_fraction"
        ]
        total_power[wavelength_index] = primary["total_power"]
        captured_power_fraction[wavelength_index] = primary[
            "captured_power_fraction"
        ]
        allowed_power_fraction[wavelength_index] = primary[
            "allowed_power_fraction"
        ]
        forbidden_power_fraction[wavelength_index] = primary[
            "forbidden_power_fraction"
        ]
        forbidden_amplitude_fraction[wavelength_index] = primary[
            "forbidden_amplitude_fraction"
        ]
        mean_mode_power_fraction[wavelength_index] = mean_sensitivity[
            "mode_power_fraction"
        ]
        mean_mode_amplitude_fraction[wavelength_index] = mean_sensitivity[
            "mode_amplitude_fraction"
        ]
        mean_total_power[wavelength_index] = mean_sensitivity["total_power"]
        mean_captured_power_fraction[wavelength_index] = mean_sensitivity[
            "captured_power_fraction"
        ]
        mean_allowed_power_fraction[wavelength_index] = mean_sensitivity[
            "allowed_power_fraction"
        ]
        mean_forbidden_power_fraction[wavelength_index] = mean_sensitivity[
            "forbidden_power_fraction"
        ]
        mean_forbidden_amplitude_fraction[wavelength_index] = mean_sensitivity[
            "forbidden_amplitude_fraction"
        ]
        centers_px[wavelength_index] = center
        pixel_size_um[wavelength_index] = pixel_size
        background_jones[wavelength_index] = background
        mean_background_jones[wavelength_index] = mean_background
        source_paths.append(_relative(source))
        source_sha256.append(_sha256(source))
        print(
            f"[experimental-modes] {int(wavelength)} nm: "
            f"captured={float(primary['captured_power_fraction']):.6%}, "
            "out-of-set angular-order leakage="
            f"{float(primary['forbidden_amplitude_fraction']):.6%} "
            "(Huber), "
            f"{float(mean_sensitivity['forbidden_amplitude_fraction']):.6%} "
            "(mean), "
            f"exact covariant residual="
            f"{covariant_residual_norm[wavelength_index]:.6%}, "
            f"V2={integrated_visibility[wavelength_index, 0]:.6g}, "
            f"V4={integrated_visibility[wavelength_index, 1]:.6g}",
            flush=True,
        )

    metadata = {
        "schema_version": "experimental-angular-modes-v3",
        "field": (
            "complete coherently averaged experimental Jones perturbation "
            "Delta J=J-J_infinity in the laboratory polarization basis"
        ),
        "projection": {
            "displayed_signed_spectra": "none",
            "exact_covariance_diagnostic": (
                "orthogonal circular-basis spin--order projection of "
                "background-subtracted Delta J; J++ and J-- retain m=0, "
                "J-+ retains m=+2, and J+- retains m=-2"
            ),
            "radius_um": [
                float(COVARIANCE_RADII_UM[0]),
                float(COVARIANCE_RADII_UM[-1]),
            ],
            "radial_step_um": float(np.diff(COVARIANCE_RADII_UM)[0]),
            "angular_samples": COVARIANCE_ANGULAR_SAMPLES,
            "complex_field_interpolation": COVARIANCE_INTERPOLATION,
            "cartesian_reconstruction": "none",
        },
        "propagation": "none",
        "mode_amplitude_definition": (
            "q_m = sqrt(2*pi*integral ||Delta J_m(rho)||_F^2 rho dr / "
            "integral ||Delta J||_F^2 d^2r)"
        ),
        "allowed_order_leakage_definition": (
            "epsilon_orders = sqrt(1 - sum_{m in {-2,0,2}} q_m^2); "
            "this tests only the set of angular orders and is not the distance "
            "to the covariant spin-locked sector; the complement includes "
            "orders outside the displayed range"
        ),
        "covariant_residual_definition": (
            "epsilon_cov = ||(1-P_cov)(J-J_infinity)||_2 / "
            "||J-J_infinity||_2 on the sampled polar disk, with radial "
            "measure rho dr and exact spin--order locking"
        ),
        "analysis_radius_um": ANALYSIS_RADIUS_UM,
        "background": {
            "primary": BACKGROUND_SPEC.metadata(),
            "sensitivity": MEAN_BACKGROUND_SPEC.metadata(),
            "note": (
                "All unsuffixed Jones-mode quantities use the primary Huber "
                "background. Arrays and report entries labelled mean_background "
                "preserve the previous arithmetic-mean analysis."
            ),
        },
        "mode_radial_step_um": MODE_RADIAL_STEP_UM,
        "mode_angular_samples": MODE_ANGULAR_SAMPLES,
        "minimum_captured_power_fraction": MINIMUM_CAPTURED_POWER_FRACTION,
        "centers": [spec.metadata() for spec in center_specs],
        "harmonic_analysis": HARMONIC_SPEC.metadata(),
        "integrated_visibility_definition": {
            "definition": (
                "radial-L2 crossed-intensity amplitudes V2,V4"
            ),
            "background_dependence": (
                "none: harmonics are formed directly from the measured "
                "crossed-polarizer intensity; the Jones background enters only "
                "the Jones-field angular spectrum"
            ),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ANALYSIS_PATH,
        schema_version=np.asarray(metadata["schema_version"]),
        wavelength_nm=wavelength_nm,
        source_paths=np.asarray(source_paths),
        source_sha256=np.asarray(source_sha256),
        centers_px=centers_px,
        pixel_size_um=pixel_size_um,
        background_jones=background_jones.astype(np.complex64),
        radii_um=radii_um,
        modes=DISPLAY_MODES,
        allowed_modes=ALLOWED_MODES,
        mode_power_fraction=mode_power_fraction,
        mode_amplitude_fraction=mode_amplitude_fraction,
        total_power=total_power,
        captured_power_fraction=captured_power_fraction,
        allowed_power_fraction=allowed_power_fraction,
        forbidden_power_fraction=forbidden_power_fraction,
        forbidden_amplitude_fraction=forbidden_amplitude_fraction,
        spin_channel_labels=np.asarray(SPIN_CHANNEL_LABELS),
        spin_allowed_orders=SPIN_ALLOWED_ORDERS,
        covariant_residual_norm=covariant_residual_norm,
        covariant_residual_power_fraction=(
            covariant_residual_power_fraction
        ),
        covariant_power_fraction=covariant_power_fraction,
        covariant_channel_residual_power_fraction=(
            covariant_channel_residual_power_fraction
        ),
        covariant_relative_orthogonality=(
            covariant_relative_orthogonality
        ),
        covariant_relative_closure_error=(
            covariant_relative_closure_error
        ),
        mean_background_jones=mean_background_jones.astype(np.complex64),
        mean_background_mode_power_fraction=mean_mode_power_fraction,
        mean_background_mode_amplitude_fraction=mean_mode_amplitude_fraction,
        mean_background_total_power=mean_total_power,
        mean_background_captured_power_fraction=(
            mean_captured_power_fraction
        ),
        mean_background_allowed_power_fraction=mean_allowed_power_fraction,
        mean_background_forbidden_power_fraction=(
            mean_forbidden_power_fraction
        ),
        mean_background_forbidden_amplitude_fraction=(
            mean_forbidden_amplitude_fraction
        ),
        harmonic_modes=np.asarray((2, 4), dtype=np.int16),
        integrated_visibility=integrated_visibility,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        **metadata,
        "analysis_archive": _relative(ANALYSIS_PATH),
        "wavelength_nm": wavelength_nm.tolist(),
        "source_paths": source_paths,
        "source_sha256": source_sha256,
        "centers_px": centers_px.tolist(),
        "pixel_size_um": pixel_size_um.tolist(),
        "modes": DISPLAY_MODES.tolist(),
        "allowed_modes": ALLOWED_MODES.tolist(),
        "mode_power_fraction": mode_power_fraction.tolist(),
        "mode_amplitude_fraction": mode_amplitude_fraction.tolist(),
        "captured_power_fraction": captured_power_fraction.tolist(),
        "allowed_power_fraction": allowed_power_fraction.tolist(),
        "forbidden_power_fraction": forbidden_power_fraction.tolist(),
        "forbidden_amplitude_fraction": forbidden_amplitude_fraction.tolist(),
        "exact_covariance_diagnostic": {
            "spin_channel_labels": list(SPIN_CHANNEL_LABELS),
            "spin_allowed_orders": SPIN_ALLOWED_ORDERS.tolist(),
            "relative_residual_norm": covariant_residual_norm.tolist(),
            "residual_power_fraction": (
                covariant_residual_power_fraction.tolist()
            ),
            "covariant_power_fraction": covariant_power_fraction.tolist(),
            "channel_residual_power_fraction": (
                covariant_channel_residual_power_fraction.tolist()
            ),
            "relative_orthogonality": (
                covariant_relative_orthogonality.tolist()
            ),
            "relative_closure_error": (
                covariant_relative_closure_error.tolist()
            ),
        },
        "background_jones": [
            _complex_matrix_json(value) for value in background_jones
        ],
        "mean_background_sensitivity": {
            "background_jones": [
                _complex_matrix_json(value) for value in mean_background_jones
            ],
            "background_frobenius_difference": [
                float(np.linalg.norm(mean - robust))
                for mean, robust in zip(
                    mean_background_jones, background_jones, strict=True
                )
            ],
            "mode_power_fraction": mean_mode_power_fraction.tolist(),
            "mode_amplitude_fraction": mean_mode_amplitude_fraction.tolist(),
            "captured_power_fraction": (
                mean_captured_power_fraction.tolist()
            ),
            "allowed_power_fraction": mean_allowed_power_fraction.tolist(),
            "forbidden_power_fraction": (
                mean_forbidden_power_fraction.tolist()
            ),
            "forbidden_amplitude_fraction": (
                mean_forbidden_amplitude_fraction.tolist()
            ),
            "forbidden_amplitude_signed_difference_mean_minus_huber": (
                mean_forbidden_amplitude_fraction
                - forbidden_amplitude_fraction
            ).tolist(),
        },
        "integrated_visibility": {
            str(int(wavelength)): {
                "V2": float(values[0]),
                "V4": float(values[1]),
            }
            for wavelength, values in zip(
                wavelength_nm, integrated_visibility, strict=True
            )
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[experimental-modes] archive: {ANALYSIS_PATH}", flush=True)
    print(f"[experimental-modes] report: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
