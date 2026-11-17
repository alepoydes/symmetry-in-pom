#!/usr/bin/env python3
"""Compare exit-surface Jones-field modes of symmetric and inclined torons.

This is a post-processing step: it consumes the four saved 520-nm Jones
fields and never invokes an in-layer optical solver.  Primary spectra use the
same director-defined structural center for both optical models.  A separately
stored fitted-center sensitivity gives the lower leakage obtainable by
optimizing the origin against the optical field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toronoptics.analysis_specs import BackgroundSpec
from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    TILTED_DIRECTOR_STRUCTURAL_CENTER_PX,
    optical_parameters,
    simulated_jones_path,
    simulated_array_center,
    tilted_director_path,
    tilted_simulated_jones_path,
)
from toronoptics.axisymmetric_jones import (
    annular_jones_background,
    jones_angular_mode_power,
    refine_axisymmetric_center,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.covariant_jones import project_covariant_jones
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.simulate_jones import load_director_archive


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELENGTH_NM = ARTICLE_GREEN_WAVELENGTH_NM
WAVELENGTH_UM = optical_parameters(WAVELENGTH_NM).wavelength_um
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/analysis/tilted_angular_modes_520_soft"
)
ANALYSIS_PATH = OUTPUT_DIR / "tilted_angular_modes.npz"
REPORT_PATH = OUTPUT_DIR / "report.json"

ANALYSIS_RADIUS_UM = 28.0
BACKGROUND_MIN_RADIUS_UM = 31.0
BACKGROUND_MAX_RADIUS_UM = 45.0
RADIAL_STEP_UM = 0.25
ANGULAR_SAMPLES = 720
MODES = np.arange(-5, 6, dtype=np.int16)
ALLOWED_MODES = np.asarray((-2, 0, 2), dtype=np.int16)
LEADING_ODD_MODES = np.asarray((-3, -1, 1, 3), dtype=np.int16)
BACKGROUND_SPEC = BackgroundSpec(
    BACKGROUND_MIN_RADIUS_UM,
    BACKGROUND_MAX_RADIUS_UM,
    estimator="huber",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    toron: str
    model: str
    source: Path
    initial_center_px: tuple[float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reversed_core_axis_geometry() -> dict[str, object]:
    """Measure the resolved core-axis shift directly from the tilted director."""
    source = tilted_director_path(PROJECT_ROOT)
    archive = load_director_archive(source)
    field = np.asarray(archive.director, dtype=np.float64)
    if archive.thickness_um is None:
        raise ValueError(f"{source} does not specify the cell thickness")
    spacing_um = float(archive.thickness_um) / float(field.shape[2] - 1)
    x_px, y_px = np.indices(field.shape[:2], dtype=np.float64)
    transverse_weight = np.sum(field[..., :2] ** 2, axis=-1)
    transverse_weight_sum = float(np.sum(transverse_weight))
    if transverse_weight_sum <= np.finfo(float).tiny:
        raise RuntimeError("Inclined director has no transverse structure")
    transverse_weight_centroid = (
        float(
            np.sum(x_px[..., None] * transverse_weight)
            / transverse_weight_sum
        ),
        float(
            np.sum(y_px[..., None] * transverse_weight)
            / transverse_weight_sum
        ),
    )
    layer_centers = np.full((field.shape[2], 2), np.nan, dtype=np.float64)
    reversed_count = np.zeros(field.shape[2], dtype=np.int64)
    for z_index in range(field.shape[2]):
        reversed_core = field[..., z_index, 2] < 0.0
        reversed_count[z_index] = int(np.count_nonzero(reversed_core))
        if reversed_count[z_index]:
            layer_centers[z_index] = (
                float(np.mean(x_px[reversed_core])),
                float(np.mean(y_px[reversed_core])),
            )
    active = np.flatnonzero(reversed_count)
    if active.size < 2:
        raise RuntimeError("Inclined director has fewer than two reversed-core layers")
    first_layer = int(active[0])
    last_layer = int(active[-1])
    endpoint_centers = layer_centers[[first_layer, last_layer]]
    transverse_displacement_um = float(
        np.linalg.norm(endpoint_centers[1] - endpoint_centers[0]) * spacing_um
    )
    axial_span_um = float((last_layer - first_layer) * spacing_um)
    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "source_sha256": _sha256(source),
        "definition": (
            "area centroid of the n_z<0 reversed core on each grid plane; "
            "endpoints are the first and last planes containing that core"
        ),
        "grid_spacing_um": spacing_um,
        "transverse_weight": "n_x^2+n_y^2 integrated over the full volume",
        "transverse_weight_centroid_px": list(transverse_weight_centroid),
        "active_layer_indices": [first_layer, last_layer],
        "endpoint_centers_px": endpoint_centers.tolist(),
        "transverse_displacement_um": transverse_displacement_um,
        "axial_span_um": axial_span_um,
        "three_dimensional_endpoint_separation_um": float(
            np.hypot(axial_span_um, transverse_displacement_um)
        ),
    }


def _dataset_specs() -> tuple[DatasetSpec, ...]:
    return (
        DatasetSpec(
            "symmetric_local",
            "symmetric",
            "spectral-jones",
            simulated_jones_path(PROJECT_ROOT, WAVELENGTH_NM, "spectral-jones"),
            simulated_array_center(
                simulated_jones_path(
                    PROJECT_ROOT, WAVELENGTH_NM, "spectral-jones"
                )
            ),
        ),
        DatasetSpec(
            "inclined_local",
            "inclined",
            "spectral-jones",
            tilted_simulated_jones_path(
                PROJECT_ROOT, WAVELENGTH_NM, "spectral-jones"
            ),
            simulated_array_center(
                tilted_simulated_jones_path(
                    PROJECT_ROOT, WAVELENGTH_NM, "spectral-jones"
                )
            ),
        ),
        DatasetSpec(
            "symmetric_marching",
            "symmetric",
            "marching-wave",
            simulated_jones_path(PROJECT_ROOT, WAVELENGTH_NM, "marching-wave"),
            simulated_array_center(
                simulated_jones_path(
                    PROJECT_ROOT, WAVELENGTH_NM, "marching-wave"
                )
            ),
        ),
        DatasetSpec(
            "inclined_marching",
            "inclined",
            "marching-wave",
            tilted_simulated_jones_path(
                PROJECT_ROOT, WAVELENGTH_NM, "marching-wave"
            ),
            simulated_array_center(
                tilted_simulated_jones_path(
                    PROJECT_ROOT, WAVELENGTH_NM, "marching-wave"
                )
            ),
        ),
    )


def _fit_center(
    jones: np.ndarray,
    *,
    initial_center_px: tuple[float, float],
    pixel_size_um: float,
) -> tuple[float, float]:
    # All search widths are specified physically and converted to pixels so
    # the local and subdivided marching-wave grids receive the same freedom.
    result = refine_axisymmetric_center(
        jones,
        initial_center_px=initial_center_px,
        pixel_size_um=pixel_size_um,
        min_radius_um=4.0,
        max_radius_um=ANALYSIS_RADIUS_UM,
        radial_step_um=0.75,
        angular_samples=360,
        broad_radius_px=4.0 / pixel_size_um,
        broad_step_px=0.5 / pixel_size_um,
        coarse_radius_px=0.75 / pixel_size_um,
        coarse_step_px=0.125 / pixel_size_um,
        fine_radius_px=0.15 / pixel_size_um,
        fine_step_px=0.025 / pixel_size_um,
    )
    if not result.broad_converged or not result.coarse_converged:
        raise RuntimeError(
            "Inclined-control center search reached a grid boundary: "
            f"initial={initial_center_px}, fitted={result.center_px}"
        )
    return result.center_px


def _plane_spectrum(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    background = annular_jones_background(
        jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        spec=BACKGROUND_SPEC,
    )
    spectrum = jones_angular_mode_power(
        jones - background,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        radii_um=radii_um,
        modes=MODES,
        angular_samples=ANGULAR_SAMPLES,
    )
    fractions = spectrum.mode_power / spectrum.total_power
    amplitudes = np.sqrt(np.maximum(fractions, 0.0))
    allowed = np.isin(spectrum.modes, ALLOWED_MODES)
    forbidden_fraction = max(
        0.0,
        1.0 - float(np.sum(fractions[allowed])),
    )
    forbidden_amplitude = float(np.sqrt(forbidden_fraction))
    leading_odd = np.isin(spectrum.modes, LEADING_ODD_MODES)
    leading_odd_share = float(
        np.clip(
            np.sum(fractions[leading_odd])
            / max(forbidden_fraction, 1.0e-300),
            0.0,
            1.0,
        )
    )
    return (
        fractions,
        amplitudes,
        spectrum.total_power,
        spectrum.captured_power_fraction,
        forbidden_fraction,
        forbidden_amplitude,
        leading_odd_share,
    )


def _spin_covariance_residual(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    radii_um: np.ndarray,
) -> tuple[float, np.ndarray]:
    background = annular_jones_background(
        jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        spec=BACKGROUND_SPEC,
    )
    projection = project_covariant_jones(
        jones - background[None, None],
        center_xy_px=center_px,
        pixel_size_xy_um=pixel_size_um,
        radius_um=radii_um,
        angular_samples=ANGULAR_SAMPLES,
        interpolation="cubic",
        radial_interpolation="cubic",
        radial_extrapolation="nearest",
        axis_order="xy",
    )
    return (
        float(projection.diagnostics.relative_residual_norm),
        np.asarray(
            projection.diagnostics.channel_residual_fraction,
            dtype=np.float64,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    args = parser.parse_args(argv)
    refuse_existing_outputs(
        (ANALYSIS_PATH, REPORT_PATH),
        force=args.force,
        project_root=PROJECT_ROOT,
    )

    plane_z_um = np.asarray((0.0,), dtype=np.float64)
    radii_um = np.arange(
        0.0,
        ANALYSIS_RADIUS_UM + 0.5 * RADIAL_STEP_UM,
        RADIAL_STEP_UM,
        dtype=np.float64,
    )
    specs = _dataset_specs()
    centers_px = np.empty((len(specs), 2), dtype=np.float64)
    fitted_centers_px = np.empty_like(centers_px)
    pixel_size_um = np.empty(len(specs), dtype=np.float64)
    mode_power_fraction = np.empty(
        (plane_z_um.size, len(specs), MODES.size), dtype=np.float64
    )
    mode_amplitude_fraction = np.empty_like(mode_power_fraction)
    total_power = np.empty((plane_z_um.size, len(specs)), dtype=np.float64)
    captured_power_fraction = np.empty_like(total_power)
    forbidden_power_fraction = np.empty_like(total_power)
    forbidden_amplitude_fraction = np.empty_like(total_power)
    leading_odd_share_of_forbidden = np.empty_like(total_power)
    fitted_mode_power_fraction = np.empty_like(mode_power_fraction)
    fitted_mode_amplitude_fraction = np.empty_like(mode_amplitude_fraction)
    fitted_total_power = np.empty_like(total_power)
    fitted_captured_power_fraction = np.empty_like(total_power)
    fitted_forbidden_power_fraction = np.empty_like(total_power)
    fitted_forbidden_amplitude_fraction = np.empty_like(total_power)
    fitted_leading_odd_share_of_forbidden = np.empty_like(total_power)
    spin_covariance_residual_amplitude = np.empty_like(total_power)
    fitted_spin_covariance_residual_amplitude = np.empty_like(total_power)
    spin_channel_residual_power_fraction = np.empty(
        (plane_z_um.size, len(specs), 4), dtype=np.float64
    )
    fitted_spin_channel_residual_power_fraction = np.empty_like(
        spin_channel_residual_power_fraction
    )
    source_paths: list[str] = []
    source_sha256: list[str] = []
    reversed_core_axis = _reversed_core_axis_geometry()
    if not np.allclose(
        reversed_core_axis["transverse_weight_centroid_px"],
        TILTED_DIRECTOR_STRUCTURAL_CENTER_PX,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Declared inclined structural origin no longer matches the "
            "supplied director field"
        )

    for dataset_index, spec in enumerate(specs):
        print(f"[tilted-modes] load {spec.source}", flush=True)
        archive = load_jones_archive(spec.source, dtype=np.complex64)
        source_paths.append(str(spec.source.relative_to(PROJECT_ROOT)))
        source_sha256.append(_sha256(spec.source))
        pixel_size = float(archive.pixel_size_um)
        pixel_size_um[dataset_index] = pixel_size
        center = tuple(map(float, spec.initial_center_px))
        fitted_center = _fit_center(
            archive.jones,
            initial_center_px=spec.initial_center_px,
            pixel_size_um=pixel_size,
        )
        centers_px[dataset_index] = center
        fitted_centers_px[dataset_index] = fitted_center
        (
            fractions,
            amplitudes,
            power,
            captured,
            forbidden_power,
            forbidden_amplitude,
            leading_odd_share,
        ) = _plane_spectrum(
            archive.jones,
            center_px=center,
            pixel_size_um=pixel_size,
            radii_um=radii_um,
        )
        mode_power_fraction[0, dataset_index] = fractions
        mode_amplitude_fraction[0, dataset_index] = amplitudes
        total_power[0, dataset_index] = power
        captured_power_fraction[0, dataset_index] = captured
        forbidden_power_fraction[0, dataset_index] = forbidden_power
        forbidden_amplitude_fraction[0, dataset_index] = forbidden_amplitude
        leading_odd_share_of_forbidden[0, dataset_index] = leading_odd_share
        (
            fitted_fractions,
            fitted_amplitudes,
            fitted_power,
            fitted_captured,
            fitted_forbidden_power,
            fitted_forbidden_amplitude,
            fitted_leading_odd_share,
        ) = _plane_spectrum(
            archive.jones,
            center_px=fitted_center,
            pixel_size_um=pixel_size,
            radii_um=radii_um,
        )
        fitted_mode_power_fraction[0, dataset_index] = fitted_fractions
        fitted_mode_amplitude_fraction[0, dataset_index] = fitted_amplitudes
        fitted_total_power[0, dataset_index] = fitted_power
        fitted_captured_power_fraction[0, dataset_index] = fitted_captured
        fitted_forbidden_power_fraction[0, dataset_index] = (
            fitted_forbidden_power
        )
        fitted_forbidden_amplitude_fraction[0, dataset_index] = (
            fitted_forbidden_amplitude
        )
        fitted_leading_odd_share_of_forbidden[0, dataset_index] = (
            fitted_leading_odd_share
        )
        (
            spin_covariance_residual_amplitude[0, dataset_index],
            spin_channel_residual_power_fraction[0, dataset_index],
        ) = _spin_covariance_residual(
            archive.jones,
            center_px=center,
            pixel_size_um=pixel_size,
            radii_um=radii_um,
        )
        (
            fitted_spin_covariance_residual_amplitude[0, dataset_index],
            fitted_spin_channel_residual_power_fraction[0, dataset_index],
        ) = _spin_covariance_residual(
            archive.jones,
            center_px=fitted_center,
            pixel_size_um=pixel_size,
            radii_um=radii_um,
        )
        print(
            f"[tilted-modes] {spec.key}, z=0 um, structural center: "
            "out-of-set angular-order leakage="
            f"{100.0 * forbidden_amplitude:.6g}%, "
            f"spin-covariance residual="
            f"{100.0 * spin_covariance_residual_amplitude[0, dataset_index]:.6g}%; "
            "fitted-center out-of-set angular-order leakage="
            f"{100.0 * fitted_forbidden_amplitude:.6g}%",
            flush=True,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "tilted-angular-modes-520-v6",
        "power_definition": (
            "p_m = 2 pi integral ||Delta J_m(rho)||_F^2 rho d rho / "
            "integral ||Delta J||_F^2 d^2r"
        ),
        "amplitude_definition": "q_m = sqrt(p_m)",
        "allowed_order_leakage_definition": (
            "epsilon_orders = sqrt(1 - sum_{m in {-2,0,2}} p_m); "
            "this tests only the set of angular orders and is not the distance "
            "to the covariant sector"
        ),
        "spin_covariance_residual_definition": (
            "orthogonal circular-basis projection with J++ and J-- locked to "
            "m=0, J-+ to m=+2, and J+- to m=-2"
        ),
        "wavelength_nm": WAVELENGTH_NM,
        "wavelength_um": WAVELENGTH_UM,
        "analysis_radius_um": ANALYSIS_RADIUS_UM,
        "background": BACKGROUND_SPEC.metadata(),
        "angular_samples": ANGULAR_SAMPLES,
        "plane": "liquid-crystal exit surface z=0; no homogeneous propagation",
        "center": {
            "primary": (
                "source-director structural origin declared independently of "
                "the optical fields and mapped with the model-specific "
                "sampling convention; for the inclined state this is the volume "
                "centroid of n_x^2+n_y^2 and is not fitted to optical modes"
            ),
            "sensitivity": (
                "one independently optimized transverse center per field; "
                "reported only as a lower-leakage sensitivity bound"
            ),
        },
    }
    np.savez_compressed(
        ANALYSIS_PATH,
        schema_version=np.asarray(metadata["schema_version"]),
        dataset_keys=np.asarray([spec.key for spec in specs]),
        toron_labels=np.asarray([spec.toron for spec in specs]),
        model_labels=np.asarray([spec.model for spec in specs]),
        source_paths=np.asarray(source_paths),
        source_sha256=np.asarray(source_sha256),
        wavelength_nm=np.asarray(WAVELENGTH_NM, dtype=np.int16),
        plane_z_um=plane_z_um,
        radii_um=radii_um,
        modes=MODES,
        allowed_modes=ALLOWED_MODES,
        leading_odd_modes=LEADING_ODD_MODES,
        centers_px=centers_px,
        fitted_centers_px=fitted_centers_px,
        pixel_size_um=pixel_size_um,
        mode_power_fraction=mode_power_fraction,
        mode_amplitude_fraction=mode_amplitude_fraction,
        total_power=total_power,
        captured_power_fraction=captured_power_fraction,
        forbidden_power_fraction=forbidden_power_fraction,
        forbidden_amplitude_fraction=forbidden_amplitude_fraction,
        leading_odd_share_of_forbidden=leading_odd_share_of_forbidden,
        spin_covariance_residual_amplitude=(
            spin_covariance_residual_amplitude
        ),
        spin_channel_residual_power_fraction=(
            spin_channel_residual_power_fraction
        ),
        fitted_mode_power_fraction=fitted_mode_power_fraction,
        fitted_mode_amplitude_fraction=fitted_mode_amplitude_fraction,
        fitted_total_power=fitted_total_power,
        fitted_captured_power_fraction=fitted_captured_power_fraction,
        fitted_forbidden_power_fraction=fitted_forbidden_power_fraction,
        fitted_forbidden_amplitude_fraction=(
            fitted_forbidden_amplitude_fraction
        ),
        fitted_leading_odd_share_of_forbidden=(
            fitted_leading_odd_share_of_forbidden
        ),
        fitted_spin_covariance_residual_amplitude=(
            fitted_spin_covariance_residual_amplitude
        ),
        fitted_spin_channel_residual_power_fraction=(
            fitted_spin_channel_residual_power_fraction
        ),
        reversed_core_source=np.asarray(reversed_core_axis["source"]),
        reversed_core_source_sha256=np.asarray(
            reversed_core_axis["source_sha256"]
        ),
        reversed_core_endpoint_centers_px=np.asarray(
            reversed_core_axis["endpoint_centers_px"],
            dtype=np.float64,
        ),
        reversed_core_transverse_displacement_um=np.asarray(
            reversed_core_axis["transverse_displacement_um"],
            dtype=np.float64,
        ),
        reversed_core_axial_span_um=np.asarray(
            reversed_core_axis["axial_span_um"],
            dtype=np.float64,
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        **metadata,
        "analysis_archive": str(ANALYSIS_PATH.relative_to(PROJECT_ROOT)),
        "dataset_keys": [spec.key for spec in specs],
        "source_paths": source_paths,
        "source_sha256": source_sha256,
        "plane_z_um": plane_z_um.tolist(),
        "modes": MODES.tolist(),
        "allowed_modes": ALLOWED_MODES.tolist(),
        "leading_odd_modes": LEADING_ODD_MODES.tolist(),
        "centers_px": centers_px.tolist(),
        "fitted_centers_px": fitted_centers_px.tolist(),
        "pixel_size_um": pixel_size_um.tolist(),
        "mode_power_fraction": mode_power_fraction.tolist(),
        "mode_amplitude_fraction": mode_amplitude_fraction.tolist(),
        "total_power": total_power.tolist(),
        "captured_power_fraction": captured_power_fraction.tolist(),
        "allowed_order_leakage_power_fraction": (
            forbidden_power_fraction.tolist()
        ),
        "allowed_order_leakage_amplitude_fraction": (
            forbidden_amplitude_fraction.tolist()
        ),
        "leading_odd_share_of_leakage": (
            leading_odd_share_of_forbidden.tolist()
        ),
        "spin_covariance_residual_amplitude": (
            spin_covariance_residual_amplitude.tolist()
        ),
        "spin_channel_residual_power_fraction": (
            spin_channel_residual_power_fraction.tolist()
        ),
        "fitted_center_sensitivity": {
            "mode_power_fraction": fitted_mode_power_fraction.tolist(),
            "mode_amplitude_fraction": (
                fitted_mode_amplitude_fraction.tolist()
            ),
            "total_power": fitted_total_power.tolist(),
            "captured_power_fraction": (
                fitted_captured_power_fraction.tolist()
            ),
            "allowed_order_leakage_power_fraction": (
                fitted_forbidden_power_fraction.tolist()
            ),
            "allowed_order_leakage_amplitude_fraction": (
                fitted_forbidden_amplitude_fraction.tolist()
            ),
            "leading_odd_share_of_leakage": (
                fitted_leading_odd_share_of_forbidden.tolist()
            ),
            "spin_covariance_residual_amplitude": (
                fitted_spin_covariance_residual_amplitude.tolist()
            ),
            "spin_channel_residual_power_fraction": (
                fitted_spin_channel_residual_power_fraction.tolist()
            ),
        },
        "reversed_core_axis": reversed_core_axis,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[tilted-modes] archive: {ANALYSIS_PATH}", flush=True)
    print(f"[tilted-modes] report: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
