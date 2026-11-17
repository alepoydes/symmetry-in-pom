#!/usr/bin/env python3
"""Compare the two saved 520-nm marching-wave phase increments.

This is a lightweight analysis of already computed Jones fields.  It does not
invoke an optical solver.  The three reported diagnostics are the quantities
quoted in the Letter and Supplemental Material:

1. the normalized background-subtracted antisymmetric ``b`` amplitude;
2. the crossed-channel integrated twofold intensity amplitude;
3. the relative Jones-field residual after one optimal complex scalar
   aligns the production field with the half-step field.
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
    PUBLICATION_HARMONICS,
)
from toronoptics.axisymmetric_jones import (
    annular_jones_background,
    crossed_intensity_harmonics,
    integrated_harmonic_visibility,
    polar_coordinate_grids,
)
from toronoptics.article_data import simulated_array_center
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.experimental_jones import load_jones_archive


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_PATH = (
    PROJECT_ROOT
    / "data/simulated/toron_27um_5cb_soft/520nm_marching-wave.npz"
)
HALF_STEP_PATH = (
    PROJECT_ROOT
    / "data/simulated/toron_27um_5cb_soft"
    / "520nm_marching-wave_phase-step-0p0125.npz"
)
OUTPUT_DIR = PROJECT_ROOT / "data/analysis/phase_step_convergence_520_soft"
REPORT_PATH = OUTPUT_DIR / "report.json"

ANALYSIS_RADIUS_UM = 28.0
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
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_matrix_json(matrix: np.ndarray) -> dict[str, list[list[float]]]:
    value = np.asarray(matrix, dtype=np.complex128)
    return {
        "real": np.real(value).tolist(),
        "imag": np.imag(value).tolist(),
    }


def _field_diagnostics(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    background_spec: BackgroundSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    background = annular_jones_background(
        jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        spec=background_spec,
    )
    delta = np.asarray(jones, dtype=np.complex128) - background
    radius_um, _phi = polar_coordinate_grids(
        jones.shape[:2],
        center_px=center_px,
        pixel_size_um=pixel_size_um,
    )
    analysis = radius_um <= ANALYSIS_RADIUS_UM
    b = 0.5 * (delta[..., 1, 0] - delta[..., 0, 1])
    b_amplitude = float(
        np.sqrt(
            2.0
            * np.sum(np.abs(b[analysis]) ** 2)
            / np.sum(np.abs(delta[analysis]) ** 2)
        )
    )
    return background, delta, analysis, b_amplitude


def _crossed_twofold(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> float:
    harmonics = crossed_intensity_harmonics(
        jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        radial_centers_um=HARMONIC_SPEC.radial_centers_um,
        radial_sigma_um=HARMONIC_SPEC.radial_sigma_um,
        modes=HARMONIC_SPEC.modes,
    )
    return integrated_harmonic_visibility(
        harmonics,
        mode=2,
        min_radius_um=HARMONIC_SPEC.integration_min_um,
        max_radius_um=HARMONIC_SPEC.integration_max_um,
    )


def _aligned_residual(
    reference_delta: np.ndarray,
    target_delta: np.ndarray,
    mask: np.ndarray,
) -> tuple[complex, float]:
    reference = reference_delta[mask].reshape(-1)
    target = target_delta[mask].reshape(-1)
    aligned_scale = np.vdot(reference, target) / np.vdot(reference, reference)
    aligned_residual = float(
        np.linalg.norm(target - aligned_scale * reference)
        / np.linalg.norm(target)
    )
    return complex(aligned_scale), aligned_residual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    args = parser.parse_args(argv)
    refuse_existing_outputs(
        (REPORT_PATH,),
        force=args.force,
        project_root=PROJECT_ROOT,
    )

    production = load_jones_archive(PRODUCTION_PATH, dtype=np.complex64)
    half_step = load_jones_archive(HALF_STEP_PATH, dtype=np.complex64)
    if production.jones.shape != half_step.jones.shape:
        raise ValueError(
            "Phase-step fields have different shapes: "
            f"{production.jones.shape} and {half_step.jones.shape}"
        )
    if not np.isclose(production.pixel_size_um, half_step.pixel_size_um):
        raise ValueError(
            "Phase-step fields have different pixel sizes: "
            f"{production.pixel_size_um} and {half_step.pixel_size_um}"
        )
    pixel_size_um = float(production.pixel_size_um)
    production_center = simulated_array_center(PRODUCTION_PATH)
    half_step_center = simulated_array_center(HALF_STEP_PATH)
    if not np.allclose(
        production_center, half_step_center, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            "Phase-step fields have different physical centers: "
            f"{production_center} and {half_step_center}"
        )
    center_spec = CenterSpec(
        *production_center,
        method="physical director-axis center stored in simulation metadata",
        fitted=False,
    )
    (
        production_background,
        production_delta,
        production_mask,
        production_b,
    ) = _field_diagnostics(
        production.jones,
        center_px=production_center,
        pixel_size_um=pixel_size_um,
        background_spec=BACKGROUND_SPEC,
    )
    half_background, half_delta, half_mask, half_b = _field_diagnostics(
        half_step.jones,
        center_px=half_step_center,
        pixel_size_um=pixel_size_um,
        background_spec=BACKGROUND_SPEC,
    )
    (
        mean_production_background,
        mean_production_delta,
        mean_production_mask,
        mean_production_b,
    ) = _field_diagnostics(
        production.jones,
        center_px=production_center,
        pixel_size_um=pixel_size_um,
        background_spec=MEAN_BACKGROUND_SPEC,
    )
    (
        mean_half_background,
        mean_half_delta,
        mean_half_mask,
        mean_half_b,
    ) = _field_diagnostics(
        half_step.jones,
        center_px=half_step_center,
        pixel_size_um=pixel_size_um,
        background_spec=MEAN_BACKGROUND_SPEC,
    )
    if not (
        np.array_equal(production_mask, half_mask)
        and np.array_equal(production_mask, mean_production_mask)
        and np.array_equal(production_mask, mean_half_mask)
    ):
        raise RuntimeError("Phase-step analysis masks differ")

    production_v2 = _crossed_twofold(
        production.jones,
        center_px=production_center,
        pixel_size_um=pixel_size_um,
    )
    half_v2 = _crossed_twofold(
        half_step.jones,
        center_px=half_step_center,
        pixel_size_um=pixel_size_um,
    )
    aligned_scale, aligned_residual = _aligned_residual(
        production_delta,
        half_delta,
        production_mask,
    )
    mean_aligned_scale, mean_aligned_residual = _aligned_residual(
        mean_production_delta,
        mean_half_delta,
        production_mask,
    )

    def signed_percent_change(production_value: float, half_value: float) -> float:
        return 100.0 * (half_value / production_value - 1.0)

    b_change = signed_percent_change(production_b, half_b)
    mean_b_change = signed_percent_change(mean_production_b, mean_half_b)
    v2_change = signed_percent_change(production_v2, half_v2)
    input_paths = (PRODUCTION_PATH, HALF_STEP_PATH)
    report = {
        "schema_version": "phase-step-convergence-520-v3",
        "inputs": [
            {
                "path": _relative(path),
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "production_source": _relative(PRODUCTION_PATH),
        "half_step_source": _relative(HALF_STEP_PATH),
        "production_phase_step_rad": 0.025,
        "half_phase_step_rad": 0.0125,
        "center": center_spec.metadata(),
        "pixel_size_um": pixel_size_um,
        "analysis_radius_um": ANALYSIS_RADIUS_UM,
        "background": {
            "primary": BACKGROUND_SPEC.metadata(),
            "sensitivity": MEAN_BACKGROUND_SPEC.metadata(),
            "note": (
                "Unsuffixed diagnostics use the primary Huber background. "
                "arithmetic_mean_background uses an arithmetic mean "
                "calculation on the same physical annulus."
            ),
        },
        "harmonic_analysis": HARMONIC_SPEC.metadata(),
        "b_amplitude_definition": (
            "sqrt(2 integral |b-b_infinity|^2 / "
            "integral ||J-J_infinity||_F^2)"
        ),
        "crossed_twofold_definition": (
            "I0^2 rho-weighted RMS amplitude 2|H2|/H0; formed from the "
            "unsubtracted crossed-channel intensity and therefore independent "
            "of the Jones-background estimator"
        ),
        "aligned_jones_residual_definition": (
            "min_alpha ||Delta J_half-alpha Delta J_production||_F / "
            "||Delta J_half||_F over rho<=28 um, alpha complex"
        ),
        "production": {
            "background_jones": _complex_matrix_json(production_background),
            "normalized_b_amplitude": production_b,
            "integrated_crossed_twofold_amplitude": production_v2,
        },
        "half_step": {
            "background_jones": _complex_matrix_json(half_background),
            "normalized_b_amplitude": half_b,
            "integrated_crossed_twofold_amplitude": half_v2,
        },
        "signed_percent_change": {
            "normalized_b_amplitude": b_change,
            "integrated_crossed_twofold_amplitude": v2_change,
        },
        "absolute_percent_change": {
            "normalized_b_amplitude": abs(b_change),
            "integrated_crossed_twofold_amplitude": abs(v2_change),
        },
        "optimal_complex_scale": {
            "real": float(np.real(aligned_scale)),
            "imag": float(np.imag(aligned_scale)),
        },
        "aligned_jones_rms_percent": 100.0 * aligned_residual,
        "arithmetic_mean_background": {
            "production": {
                "background_jones": _complex_matrix_json(
                    mean_production_background
                ),
                "normalized_b_amplitude": mean_production_b,
                "integrated_crossed_twofold_amplitude": production_v2,
            },
            "half_step": {
                "background_jones": _complex_matrix_json(mean_half_background),
                "normalized_b_amplitude": mean_half_b,
                "integrated_crossed_twofold_amplitude": half_v2,
            },
            "signed_percent_change": {
                "normalized_b_amplitude": mean_b_change,
                "integrated_crossed_twofold_amplitude": v2_change,
            },
            "optimal_complex_scale": {
                "real": float(np.real(mean_aligned_scale)),
                "imag": float(np.imag(mean_aligned_scale)),
            },
            "aligned_jones_rms_percent": 100.0 * mean_aligned_residual,
            "difference_from_primary": {
                "production_normalized_b_amplitude": (
                    mean_production_b - production_b
                ),
                "half_step_normalized_b_amplitude": mean_half_b - half_b,
                "aligned_jones_rms_percent": (
                    100.0 * (mean_aligned_residual - aligned_residual)
                ),
            },
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[phase-step] "
        f"|delta b|={abs(b_change):.6g}%, "
        f"|delta V2|={abs(v2_change):.6g}%, "
        f"aligned rms={100.0 * aligned_residual:.6g}%",
        flush=True,
    )
    print(f"[phase-step] report: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
