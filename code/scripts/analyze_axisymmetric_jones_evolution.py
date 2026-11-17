#!/usr/bin/env python3
"""Extract, propagate, and compare the four radial Jones channels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.ndimage import gaussian_filter
from topovec.optics import apply_coherent_pupil

from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    ARTICLE_VARIANT_SUFFIX,
    ARTICLE_WAVELENGTHS_NM,
    effective_coherent_na,
    experimental_array_center,
    experimental_average_path,
    experimental_jones_propagation_wavelength_um,
    optical_parameters,
    publication_focus_grid,
    simulated_array_center,
    simulated_jones_path,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.axisymmetric_propagation import (
    ABCD_LABELS,
    CIRCULAR_LABELS,
    AxisymmetricProfile,
    HomogeneousPropagation,
    abcd_from_circular,
    circular_from_abcd,
    derived_radial_observables,
    extract_axisymmetric_profile,
    propagate_axisymmetric_circular,
    propagate_axisymmetric_circular_hankel,
    reflection_axis_observables,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.figure_output import figure_output_dir
from toronoptics.focus_stack import AngularSpectrumJonesPropagator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = figure_output_dir(PROJECT_ROOT)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}"
)
DEFAULT_COEFFICIENT_FIGURE = (
    FIGURE_DIR
    / f"supplement_axisymmetric_coefficients_focus_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)
DEFAULT_OBSERVABLE_FIGURE = (
    FIGURE_DIR
    / f"supplement_axisymmetric_observables_focus_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)
DEFAULT_AXIS_ORIENTATION_FIGURE = (
    FIGURE_DIR
    / f"supplement_symmetry_axis_focus_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)
DEFAULT_C4_BREAKING_FIGURE = (
    FIGURE_DIR
    / f"supplement_c4_breaking_focus_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)
DEFAULT_CIRCULAR_FIGURE = (
    DEFAULT_OUTPUT_DIR / f"axisymmetric_circular_focus_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)

DATASET_KEYS = (
    "experiment",
    "marching_wave",
    "marching_wave_pupil_matched",
    "local_jones_pupil_matched",
)
PRIMARY_DATASET_KEYS = (
    "experiment",
    "marching_wave_pupil_matched",
    "local_jones_pupil_matched",
)
DATASET_LABELS = {
    "experiment": "Experiment",
    "marching_wave": "Marching waves, exit field",
    "marching_wave_pupil_matched": "Marching waves",
    "local_jones_pupil_matched": "Local Jones",
}
RADIUS_STEP_UM = 0.125
MAX_SOURCE_RADIUS_UM = 45.0
BACKGROUND_INTERVAL_UM = (41.0, 45.0)
DISPLAY_MAX_RADIUS_UM = 40.0
EXTENSION_RADIUS_UM = 96.0
PROPAGATION_HALF_WIDTH_UM = 40.0
EXPERIMENT_PROPAGATION_CENTER_UM = 7.0
PROPAGATION_Z_UM_BY_DATASET = {
    # The experimental coordinate is measured from the holographic plane.
    # Its fixed range is centered on the rounded current surface estimate.
    "experiment": np.arange(
        EXPERIMENT_PROPAGATION_CENTER_UM - PROPAGATION_HALF_WIDTH_UM,
        EXPERIMENT_PROPAGATION_CENTER_UM + PROPAGATION_HALF_WIDTH_UM + 0.01,
        1.0,
    ),
    # Model coordinates are measured from the known liquid-crystal exit.
    "marching_wave_pupil_matched": np.arange(
        -PROPAGATION_HALF_WIDTH_UM,
        PROPAGATION_HALF_WIDTH_UM + 0.01,
        1.0,
    ),
    "local_jones_pupil_matched": np.arange(
        -PROPAGATION_HALF_WIDTH_UM,
        PROPAGATION_HALF_WIDTH_UM + 0.01,
        1.0,
    ),
}


def _dataset_propagation_wavelength_um(
    wavelength_nm: int,
    dataset_key: str,
) -> float:
    if dataset_key == "experiment":
        return experimental_jones_propagation_wavelength_um(wavelength_nm)
    return float(optical_parameters(wavelength_nm).wavelength_um)
FOCUS_Z_INTERVAL_UM_BY_DATASET = {
    "experiment": (20.0, 30.0),
    "marching_wave_pupil_matched": (12.0, 22.0),
    "local_jones_pupil_matched": (20.0, 32.0),
}
ANGULAR_SAMPLES = 720
POLAR_INTERPOLATION = "cubic"
MEDIUM_INDEX = 1.0
LOCAL_MAX_STEP_UM = 0.20
FINAL_DPI = 600
LCSIM_AZIMUTH_HUE_SHIFT = 0.81
FOCUS_RADIUS_INTERVAL_UM = (2.0, 25.0)
FOCUS_REFINEMENT_HALF_WIDTH_UM = 2.0
FOCUS_REFINEMENT_STEP_UM = 0.025
FIGURE_FONT_SIZE = 11.0
EVOLUTION_OBSERVABLE_NAMES = (
    "scalar_power",
    "D_I",
    "D_P",
    "D_N",
    "crossed_H2",
    "crossed_H4",
    "crossed_mean_intensity",
    "crossed_V2",
    "crossed_V4",
)
EXACT_SAMPLE_SET = "figure1_through_focus"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integrated_relative_residual(
    radius_um: np.ndarray,
    residual_rms: np.ndarray,
    signal_rms: np.ndarray,
) -> float:
    radius = np.asarray(radius_um)
    numerator = np.trapezoid(residual_rms**2 * radius, radius)
    denominator = np.trapezoid(signal_rms**2 * radius, radius)
    return float(np.sqrt(numerator / max(denominator, 1.0e-300)))


def _extract_all() -> tuple[dict[tuple[int, str], object], dict]:
    radius_um = np.arange(
        0.0, MAX_SOURCE_RADIUS_UM + 0.5 * RADIUS_STEP_UM, RADIUS_STEP_UM
    )
    profiles: dict[tuple[int, str], object] = {}
    report: dict[str, object] = {
        "schema_version": "axisymmetric-jones-coefficients-v3",
        "radius_step_um": RADIUS_STEP_UM,
        "maximum_source_radius_um": MAX_SOURCE_RADIUS_UM,
        "background_interval_um": list(BACKGROUND_INTERVAL_UM),
        "angular_samples": ANGULAR_SAMPLES,
        "polar_interpolation": POLAR_INTERPOLATION,
        "datasets": {},
    }
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        field_wavelength_um = optical_parameters(wavelength_nm).wavelength_um
        experiment_path = experimental_average_path(PROJECT_ROOT, wavelength_nm)
        model_path = simulated_jones_path(PROJECT_ROOT, wavelength_nm, "marching-wave")
        local_path = simulated_jones_path(PROJECT_ROOT, wavelength_nm, "spectral-jones")
        pupil_na = effective_coherent_na(PROJECT_ROOT, wavelength_nm)
        sources = (
            (
                "experiment",
                experiment_path,
                experimental_array_center(experiment_path),
                False,
            ),
            (
                "marching_wave",
                model_path,
                simulated_array_center(model_path),
                False,
            ),
            (
                "marching_wave_pupil_matched",
                model_path,
                simulated_array_center(model_path),
                True,
            ),
            (
                "local_jones_pupil_matched",
                local_path,
                simulated_array_center(local_path),
                True,
            ),
        )
        for key, path, center, apply_pupil in sources:
            propagation_wavelength_um = _dataset_propagation_wavelength_um(
                wavelength_nm,
                key,
            )
            loaded = load_jones_archive(path, dtype=np.complex64)
            jones = loaded.jones
            if apply_pupil:
                jones = apply_coherent_pupil(
                    jones,
                    pixel_size_um=float(loaded.pixel_size_um),
                    wavelength_um=float(field_wavelength_um),
                    numerical_aperture=float(pupil_na),
                )
            profile = extract_axisymmetric_profile(
                jones,
                center_px=center,
                pixel_size_um=float(loaded.pixel_size_um),
                radius_um=radius_um,
                angular_samples=ANGULAR_SAMPLES,
                background_interval_um=BACKGROUND_INTERVAL_UM,
                interpolation=POLAR_INTERPOLATION,
            )
            profiles[(wavelength_nm, key)] = profile
            condition = float(np.linalg.cond(profile.background_jones))
            report["datasets"][f"{wavelength_nm}_{key}"] = {
                "source": _relative(path),
                "source_sha256": _sha256(path),
                "pixel_size_um": float(loaded.pixel_size_um),
                "center_px": list(center),
                "field_calculation_wavelength_um": float(field_wavelength_um),
                "propagation_wavelength_um": float(
                    propagation_wavelength_um
                ),
                "effective_coherent_na": float(pupil_na),
                "pupil_applied_by_this_analysis": bool(apply_pupil),
                "background_condition_number": condition,
                "outer_order_two_rms": float(
                    np.sqrt(
                        np.mean(
                            np.abs(
                                profile.circular[
                                    (profile.radius_um >= BACKGROUND_INTERVAL_UM[0])
                                    & (
                                        profile.radius_um
                                        <= BACKGROUND_INTERVAL_UM[1]
                                    ),
                                    2:,
                                ]
                            )
                            ** 2
                        )
                    )
                ),
                "integrated_axisymmetric_projection_residual": (
                    _integrated_relative_residual(
                        radius_um,
                        profile.reconstruction_rms,
                        profile.signal_rms,
                    )
                ),
                "maximum_circular_consistency_rms": float(
                    np.max(profile.circular_consistency_rms)
                ),
            }
            del jones, loaded
    return profiles, report


def _save_coefficient_archive(
    output_dir: Path,
    profiles: dict[tuple[int, str], object],
    report: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "axisymmetric_jones_coefficients.npz"
    ordered = [
        profiles[(wavelength_nm, key)]
        for wavelength_nm in ARTICLE_WAVELENGTHS_NM
        for key in DATASET_KEYS
    ]
    shape = (len(ARTICLE_WAVELENGTHS_NM), len(DATASET_KEYS))
    metadata = {
        "schema_version": "axisymmetric-jones-coefficients-v3",
        "component_axis_order": "J[x,y,output,input]",
        "circular_basis": "e_plus=(e_x+i e_y)/sqrt(2), e_minus=(e_x-i e_y)/sqrt(2)",
        "circular_channel_order": list(CIRCULAR_LABELS),
        "circular_angular_orders": [0, 0, 2, -2],
        "generator_channel_order": list(ABCD_LABELS),
        "projection": (
            "cubic subpixel sampling of the complex Jones field followed by "
            "the spin-locked co-rotating angular mean"
        ),
        "polar_interpolation": POLAR_INTERPOLATION,
        "normalization": "none; stored coefficients retain the physical source Jones normalization and global phase",
        "model_matching": "the pupil-matched model is filtered once with the source-ensemble coherent hard pupil; the experimental field is not filtered again",
        "background": (
            "rho d rho-weighted outer-annulus mean of the order-zero A "
            "channels; order-two B channels have no static plane-wave "
            "background and are retained in full"
        ),
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray(3, dtype=np.int16),
        wavelength_nm=np.asarray(ARTICLE_WAVELENGTHS_NM, dtype=np.int16),
        propagation_wavelength_um=np.asarray(
            [
                _dataset_propagation_wavelength_um(wavelength_nm, key)
                for wavelength_nm in ARTICLE_WAVELENGTHS_NM
                for key in DATASET_KEYS
            ],
            dtype=np.float64,
        ).reshape(shape),
        field_calculation_wavelength_um=np.asarray(
            [
                optical_parameters(wavelength_nm).wavelength_um
                for wavelength_nm in ARTICLE_WAVELENGTHS_NM
            ],
            dtype=np.float64,
        ),
        dataset_keys=np.asarray(DATASET_KEYS),
        dataset_labels=np.asarray([DATASET_LABELS[key] for key in DATASET_KEYS]),
        radius_um=ordered[0].radius_um.astype(np.float64),
        abcd=np.stack([profile.abcd for profile in ordered])
        .reshape((*shape, ordered[0].radius_um.size, 4))
        .astype(np.complex64),
        circular=np.stack([profile.circular for profile in ordered])
        .reshape((*shape, ordered[0].radius_um.size, 4))
        .astype(np.complex64),
        circular_direct=np.stack([profile.circular_direct for profile in ordered])
        .reshape((*shape, ordered[0].radius_um.size, 4))
        .astype(np.complex64),
        angular_rms=np.stack([profile.angular_rms for profile in ordered])
        .reshape((*shape, ordered[0].radius_um.size, 4))
        .astype(np.float32),
        reconstruction_rms=np.stack([profile.reconstruction_rms for profile in ordered])
        .reshape((*shape, ordered[0].radius_um.size))
        .astype(np.float32),
        signal_rms=np.stack([profile.signal_rms for profile in ordered])
        .reshape((*shape, ordered[0].radius_um.size))
        .astype(np.float32),
        circular_consistency_rms=np.stack(
            [profile.circular_consistency_rms for profile in ordered]
        )
        .reshape((*shape, ordered[0].radius_um.size))
        .astype(np.float32),
        background_abcd=np.stack([profile.background_abcd for profile in ordered])
        .reshape((*shape, 4))
        .astype(np.complex64),
        background_jones=np.stack([profile.background_jones for profile in ordered])
        .reshape((*shape, 2, 2))
        .astype(np.complex64),
        centers_px=np.asarray(
            [profile.center_px for profile in ordered], dtype=np.float64
        ).reshape((*shape, 2)),
        angular_samples=np.asarray(ANGULAR_SAMPLES, dtype=np.int16),
        polar_interpolation=np.asarray(POLAR_INTERPOLATION),
        background_interval_um=np.asarray(BACKGROUND_INTERVAL_UM, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
        report_json=np.asarray(json.dumps(report, indent=2, sort_keys=True)),
    )
    return path


def _propagate_all(
    profiles: dict[tuple[int, str], object], report: dict
) -> tuple[dict[tuple[int, str], dict[str, object]], dict]:
    evolutions: dict[tuple[int, str], dict[str, object]] = {}
    validation: dict[str, object] = {}
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        for key in PRIMARY_DATASET_KEYS:
            wavelength_um = _dataset_propagation_wavelength_um(
                wavelength_nm,
                key,
            )
            profile = profiles[(wavelength_nm, key)]
            z_um = PROPAGATION_Z_UM_BY_DATASET[key]
            background_circular = circular_from_abcd(profile.background_abcd)
            hankel = propagate_axisymmetric_circular_hankel(
                profile.circular,
                profile.radius_um,
                z_um,
                wavelength_um=wavelength_um,
                medium_index=MEDIUM_INDEX,
                background_circular=background_circular,
                extension_radius_um=EXTENSION_RADIUS_UM,
                spectral_oversampling=2.5,
            )
            pade = propagate_axisymmetric_circular(
                profile.circular,
                profile.radius_um,
                z_um,
                wavelength_um=wavelength_um,
                medium_index=MEDIUM_INDEX,
                background_circular=background_circular,
                method="pade11",
                max_step_um=LOCAL_MAX_STEP_UM,
                extension_radius_um=EXTENSION_RADIUS_UM,
            )
            if not np.allclose(hankel.radius_um, pade.radius_um):
                raise RuntimeError("Hankel and local propagators returned different grids")
            abcd_hankel = hankel.abcd
            abcd_pade = pade.abcd
            observables = derived_radial_observables(
                abcd_hankel, background_abcd=profile.background_abcd
            )
            display = hankel.radius_um <= DISPLAY_MAX_RADIUS_UM
            radial_weight = hankel.radius_um[display]
            difference = abcd_pade[:, display] - abcd_hankel[:, display]
            reference = abcd_hankel[:, display] - profile.background_abcd[None, None, :]
            numerator = np.trapezoid(
                np.sum(np.abs(difference) ** 2, axis=-1) * radial_weight[None, :],
                radial_weight,
                axis=1,
            )
            denominator = np.trapezoid(
                np.sum(np.abs(reference) ** 2, axis=-1) * radial_weight[None, :],
                radial_weight,
                axis=1,
            )
            relative_error = np.sqrt(numerator / np.maximum(denominator, 1.0e-300))
            validation[f"{wavelength_nm}_{key}"] = {
                "maximum_pade11_to_hankel_relative_error": float(
                    np.max(relative_error)
                ),
                "median_pade11_to_hankel_relative_error": float(
                    np.median(relative_error)
                ),
                "error_at_z_um": {
                    f"{float(z):g}": float(error)
                    for z, error in zip(z_um, relative_error, strict=True)
                },
            }
            focus = _refine_annular_focus(
                profile,
                hankel,
                dataset_key=key,
                wavelength_um=wavelength_um,
                background_circular=background_circular,
            )
            report.setdefault("annular_focus", {})[
                f"{wavelength_nm}_{key}"
            ] = focus
            evolutions[(wavelength_nm, key)] = {
                "hankel": hankel,
                "pade": pade,
                "observables": observables,
                "relative_error": relative_error,
                "annular_focus": focus,
            }
    focus_alignment = {}
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        experiment = evolutions[(wavelength_nm, "experiment")]["annular_focus"]
        model = evolutions[
            (wavelength_nm, "marching_wave_pupil_matched")
        ]["annular_focus"]
        focus_alignment[str(wavelength_nm)] = {
            "experiment_minus_model_z_um": float(
                experiment["z_um"] - model["z_um"]
            ),
            "experiment_minus_model_radius_um": float(
                experiment["radius_um"] - model["radius_um"]
            ),
            "aligned_coordinate_definition": (
                "z_experiment = zeta_experiment - experiment_minus_model_z_um"
            ),
            "inferred_experimental_surface_z_um": float(
                experiment["z_um"] - model["z_um"]
            ),
            "surface_definition": (
                "known marching-wave exit surface z=0 transferred by annular-focus matching"
            ),
        }
    report["annular_focus_alignment"] = focus_alignment
    report["homogeneous_propagation_validation"] = validation
    return evolutions, report


def _quadratic_peak(
    z_um: np.ndarray,
    radius_um: np.ndarray,
    values: np.ndarray,
    coarse_index: tuple[int, int],
) -> tuple[float, float, float, bool]:
    """Refine an interior grid maximum by two local parabolic sections."""

    iz, ir = coarse_index
    if not (1 <= iz < z_um.size - 1 and 1 <= ir < radius_um.size - 1):
        return (
            float(z_um[iz]),
            float(radius_um[ir]),
            float(values[iz, ir]),
            False,
        )

    def section_offset(
        minus: float, center: float, plus: float, spacing: float
    ) -> tuple[float, float, bool]:
        curvature = minus - 2.0 * center + plus
        if not curvature < 0.0:
            return 0.0, 0.0, False
        offset = 0.5 * (minus - plus) / curvature * spacing
        if abs(offset) > spacing:
            return 0.0, 0.0, False
        slope = (plus - minus) / (2.0 * spacing)
        quadratic = curvature / (2.0 * spacing**2)
        gain = slope * offset + quadratic * offset**2
        return float(offset), float(gain), True

    dz, gain_z, accepted_z = section_offset(
        float(values[iz - 1, ir]),
        float(values[iz, ir]),
        float(values[iz + 1, ir]),
        float(z_um[iz + 1] - z_um[iz]),
    )
    dr, gain_r, accepted_r = section_offset(
        float(values[iz, ir - 1]),
        float(values[iz, ir]),
        float(values[iz, ir + 1]),
        float(radius_um[ir + 1] - radius_um[ir]),
    )
    accepted = accepted_z and accepted_r
    if not accepted:
        dz = dr = gain_z = gain_r = 0.0
    return (
        float(z_um[iz] + dz),
        float(radius_um[ir] + dr),
        float(values[iz, ir] + gain_z + gain_r),
        accepted,
    )


def _refine_annular_focus(
    profile: object,
    coarse_evolution: object,
    *,
    dataset_key: str,
    wavelength_um: float,
    background_circular: np.ndarray,
) -> dict[str, object]:
    """Locate the basis-invariant annular focus from unpolarized power gain."""

    coarse_observables = derived_radial_observables(
        coarse_evolution.abcd, background_abcd=profile.background_abcd
    )
    radial_mask = (
        (coarse_evolution.radius_um >= FOCUS_RADIUS_INTERVAL_UM[0])
        & (coarse_evolution.radius_um <= FOCUS_RADIUS_INTERVAL_UM[1])
    )
    focus_z_interval = FOCUS_Z_INTERVAL_UM_BY_DATASET[dataset_key]
    axial_mask = (
        (coarse_evolution.z_um >= focus_z_interval[0])
        & (coarse_evolution.z_um <= focus_z_interval[1])
    )
    if np.count_nonzero(axial_mask) < 3:
        raise ValueError(
            f"Focus interval {focus_z_interval} contains fewer than three "
            f"samples for {dataset_key}"
        )
    axial_indices = np.flatnonzero(axial_mask)
    coarse_values = coarse_observables["scalar_power"][axial_mask][:, radial_mask]
    coarse_local = np.unravel_index(np.argmax(coarse_values), coarse_values.shape)
    coarse_z_index = int(axial_indices[coarse_local[0]])
    coarse_z = float(coarse_evolution.z_um[coarse_z_index])
    lower = max(
        float(focus_z_interval[0]),
        coarse_z - FOCUS_REFINEMENT_HALF_WIDTH_UM,
    )
    upper = min(
        float(focus_z_interval[1]),
        coarse_z + FOCUS_REFINEMENT_HALF_WIDTH_UM,
    )
    fine_z = np.arange(
        lower,
        upper + 0.5 * FOCUS_REFINEMENT_STEP_UM,
        FOCUS_REFINEMENT_STEP_UM,
    )
    fine_evolution = propagate_axisymmetric_circular_hankel(
        profile.circular,
        profile.radius_um,
        fine_z,
        wavelength_um=wavelength_um,
        medium_index=MEDIUM_INDEX,
        background_circular=background_circular,
        extension_radius_um=EXTENSION_RADIUS_UM,
        spectral_oversampling=2.5,
    )
    fine_observables = derived_radial_observables(
        fine_evolution.abcd, background_abcd=profile.background_abcd
    )
    fine_radius_mask = (
        (fine_evolution.radius_um >= FOCUS_RADIUS_INTERVAL_UM[0])
        & (fine_evolution.radius_um <= FOCUS_RADIUS_INTERVAL_UM[1])
    )
    fine_radius = fine_evolution.radius_um[fine_radius_mask]
    fine_values = fine_observables["scalar_power"][:, fine_radius_mask]
    fine_index = np.unravel_index(np.argmax(fine_values), fine_values.shape)
    peak_z, peak_radius, peak_power, accepted = _quadratic_peak(
        fine_z, fine_radius, fine_values, fine_index
    )
    boundary = bool(
        fine_index[0] in {0, fine_z.size - 1}
        or fine_index[1] in {0, fine_radius.size - 1}
    )
    return {
        "definition": "maximum of s=tr(T^dagger T)/2 within the stated radial interval",
        "z_um": peak_z,
        "radius_um": peak_radius,
        "scalar_power": peak_power,
        "coarse_z_um": coarse_z,
        "fine_z_step_um": FOCUS_REFINEMENT_STEP_UM,
        "radial_step_um": float(np.median(np.diff(fine_evolution.radius_um))),
        "search_z_interval_um": list(focus_z_interval),
        "refinement_z_interval_um": [float(fine_z[0]), float(fine_z[-1])],
        "search_radius_interval_um": list(FOCUS_RADIUS_INTERVAL_UM),
        "quadratic_refinement_accepted": accepted,
        "maximum_on_search_boundary": boundary,
    }


def _save_evolution_archive(
    output_dir: Path,
    evolutions: dict[tuple[int, str], dict[str, object]],
    profiles: dict[tuple[int, str], object],
    report: dict,
) -> Path:
    path = output_dir / "axisymmetric_jones_evolution.npz"
    ordered = [
        evolutions[(wavelength_nm, key)]
        for wavelength_nm in ARTICLE_WAVELENGTHS_NM
        for key in PRIMARY_DATASET_KEYS
    ]
    shape = (len(ARTICLE_WAVELENGTHS_NM), len(PRIMARY_DATASET_KEYS))
    radius_um = ordered[0]["hankel"].radius_um
    z_count = ordered[0]["hankel"].z_um.size
    if any(entry["hankel"].z_um.size != z_count for entry in ordered):
        raise ValueError("All publication z grids must contain the same number of samples")
    exact_samples = _exact_publication_samples(profiles, report)
    metadata = {
        "schema_version": "axisymmetric-jones-evolution-v5",
        "z_axis_layout": "z_um[dataset,z]",
        "primary_variables": list(CIRCULAR_LABELS),
        "primary_angular_orders": [0, 0, 2, -2],
        "reference_propagator": "continuous Fourier-Hankel angular spectrum with exact sqrt(k^2-kappa^2)-k phase",
        "local_propagator": "radial Crank-Nicolson with [1/1] Pade wide-angle operator",
        "reported_z_origin": "liquid-crystal exit surface for both datasets",
        "internal_experiment_origin": "holographic reconstruction plane",
        "z_ranges_um": {
            key: [
                float(PROPAGATION_Z_UM_BY_DATASET[key][0]),
                float(PROPAGATION_Z_UM_BY_DATASET[key][-1]),
            ]
            for key in PRIMARY_DATASET_KEYS
        },
        "propagation_range_definition": (
            "fixed native-coordinate grids chosen before focus extraction; "
            "the full stored range is plotted"
        ),
        "focus_search_z_intervals_um": {
            key: list(FOCUS_Z_INTERVAL_UM_BY_DATASET[key])
            for key in PRIMARY_DATASET_KEYS
        },
        "surface_alignment": (
            "the experimental origin is shifted by annular-focus matching to "
            "the known marching-wave exit surface"
        ),
        "medium_index": MEDIUM_INDEX,
        "z_coordinate": (
            "air-equivalent homogeneous propagation distance; both experiment "
            "and model use n_out=1"
        ),
        "pupil": "no second pupil is applied during continuation; the experiment is already measured through the pupil and the comparison model was filtered before projection",
        "D_I": "(tr(T^dagger T)/2 - 1)^2 after explicit right normalization T=J J_infinity^{-1}",
        "D_P": "0.5 ||T^dagger T - tr(T^dagger T) I/2||_F^2",
        "D_N": "0.5 ||T^dagger T - T T^dagger||_F^2",
        "H2_H4": "complex m=2 and m=4 harmonics of crossed linear-polarizer intensity",
        "exact_samples": (
            "named off-grid planes propagated directly from the extracted "
            "radial source profiles; no interpolation along z"
        ),
        "outer_boundary": (
            "all localized channels use a raised-cosine continuation from "
            "the last measured radius to zero at the extension radius"
        ),
    }
    save = {
        "schema_version": np.asarray(5, dtype=np.int16),
        "wavelength_nm": np.asarray(ARTICLE_WAVELENGTHS_NM, dtype=np.int16),
        "propagation_wavelength_um": np.asarray(
            [
                _dataset_propagation_wavelength_um(wavelength_nm, key)
                for wavelength_nm in ARTICLE_WAVELENGTHS_NM
                for key in PRIMARY_DATASET_KEYS
            ],
            dtype=np.float64,
        ).reshape(shape),
        "dataset_keys": np.asarray(PRIMARY_DATASET_KEYS),
        "dataset_labels": np.asarray(
            [DATASET_LABELS[key] for key in PRIMARY_DATASET_KEYS]
        ),
        "z_um": np.stack(
            [
                evolutions[(ARTICLE_WAVELENGTHS_NM[0], key)]["hankel"].z_um
                for key in PRIMARY_DATASET_KEYS
            ]
        ).astype(np.float64),
        "surface_referenced_z_um": np.stack(
            [
                _surface_aligned_z_um(evolutions, wavelength_nm, key)
                for wavelength_nm in ARTICLE_WAVELENGTHS_NM
                for key in PRIMARY_DATASET_KEYS
            ]
        ).reshape((*shape, z_count)).astype(np.float64),
        "radius_um": radius_um.astype(np.float64),
        "circular_hankel": np.stack(
            [entry["hankel"].circular for entry in ordered]
        )
        .reshape((*shape, z_count, radius_um.size, 4))
        .astype(np.complex64),
        "abcd_hankel": np.stack([entry["hankel"].abcd for entry in ordered])
        .reshape((*shape, z_count, radius_um.size, 4))
        .astype(np.complex64),
        "circular_pade11": np.stack([entry["pade"].circular for entry in ordered])
        .reshape((*shape, z_count, radius_um.size, 4))
        .astype(np.complex64),
        "pade11_relative_error": np.stack(
            [entry["relative_error"] for entry in ordered]
        ).reshape((*shape, z_count)),
        "annular_focus_z_um": np.asarray(
            [entry["annular_focus"]["z_um"] for entry in ordered], dtype=np.float64
        ).reshape(shape),
        "annular_focus_radius_um": np.asarray(
            [entry["annular_focus"]["radius_um"] for entry in ordered],
            dtype=np.float64,
        ).reshape(shape),
        "annular_focus_scalar_power": np.asarray(
            [entry["annular_focus"]["scalar_power"] for entry in ordered],
            dtype=np.float64,
        ).reshape(shape),
        "surface_z_um": np.asarray(
            [
                (
                    evolutions[(wavelength_nm, "experiment")]["annular_focus"][
                        "z_um"
                    ]
                    - evolutions[
                        (wavelength_nm, "marching_wave_pupil_matched")
                    ]["annular_focus"]["z_um"]
                    if key == "experiment"
                    else 0.0
                )
                for wavelength_nm in ARTICLE_WAVELENGTHS_NM
                for key in PRIMARY_DATASET_KEYS
            ],
            dtype=np.float64,
        ).reshape(shape),
        "background_abcd": np.stack(
            [
                profiles[(wavelength_nm, key)].background_abcd
                for wavelength_nm in ARTICLE_WAVELENGTHS_NM
                for key in PRIMARY_DATASET_KEYS
            ]
        ).reshape((*shape, 4)).astype(np.complex64),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
        "report_json": np.asarray(json.dumps(report, indent=2, sort_keys=True)),
    }
    for name in EVOLUTION_OBSERVABLE_NAMES:
        values = np.stack([entry["observables"][name] for entry in ordered]).reshape(
            (*shape, z_count, radius_um.size)
        )
        save[name] = values.astype(
            np.complex64 if np.iscomplexobj(values) else np.float32
        )
    save.update(exact_samples)
    np.savez_compressed(path, **save)
    return path


def _exact_publication_samples(
    profiles: dict[tuple[int, str], object],
    report: dict,
) -> dict[str, np.ndarray]:
    wavelength_nm = ARTICLE_GREEN_WAVELENGTH_NM
    key = "experiment"
    profile = profiles[(wavelength_nm, key)]
    z_um, surface_um, annular_focus_um = publication_focus_grid(
        report, wavelength_nm=wavelength_nm
    )
    propagation = propagate_axisymmetric_circular_hankel(
        profile.circular,
        profile.radius_um,
        z_um,
        wavelength_um=_dataset_propagation_wavelength_um(
            wavelength_nm,
            key,
        ),
        medium_index=MEDIUM_INDEX,
        background_circular=circular_from_abcd(profile.background_abcd),
        extension_radius_um=EXTENSION_RADIUS_UM,
        spectral_oversampling=2.5,
    )
    observables = derived_radial_observables(
        propagation.abcd,
        background_abcd=profile.background_abcd,
    )
    physical_observables = derived_radial_observables(propagation.abcd)
    report.setdefault("exact_sample_sets", {})[EXACT_SAMPLE_SET] = {
        "wavelength_nm": wavelength_nm,
        "propagation_wavelength_um": _dataset_propagation_wavelength_um(
            wavelength_nm,
            key,
        ),
        "dataset_key": key,
        "z_um": z_um.tolist(),
        "surface_referenced_z_um": (z_um - surface_um).tolist(),
        "surface_um": surface_um,
        "annular_focus_um": annular_focus_um,
        "propagator": propagation.method,
        "z_interpolation": "none",
    }
    payload: dict[str, np.ndarray] = {
        "exact_sample_set_keys": np.asarray([EXACT_SAMPLE_SET]),
        "exact_sample_wavelength_nm": np.asarray(wavelength_nm, dtype=np.int16),
        "exact_sample_dataset_key": np.asarray(key),
        "exact_sample_z_um": z_um.astype(np.float64),
        "exact_sample_surface_referenced_z_um": (z_um - surface_um).astype(
            np.float64
        ),
        "exact_sample_radius_um": propagation.radius_um.astype(np.float64),
        "exact_sample_circular_hankel": propagation.circular.astype(np.complex64),
        "exact_sample_abcd_hankel": propagation.abcd.astype(np.complex64),
        "exact_sample_physical_crossed_H0": physical_observables[
            "crossed_mean_intensity"
        ].astype(np.float32),
        "exact_sample_physical_crossed_H2": physical_observables[
            "crossed_H2"
        ].astype(np.complex64),
        "exact_sample_physical_crossed_H4": physical_observables[
            "crossed_H4"
        ].astype(np.complex64),
    }
    for name in EVOLUTION_OBSERVABLE_NAMES:
        values = observables[name]
        payload[f"exact_sample_{name}"] = values.astype(
            np.complex64 if np.iscomplexobj(values) else np.float32
        )
    return payload


def _upgrade_exact_publication_samples(output_dir: Path) -> tuple[Path, Path]:
    profiles, _, _ = _load_coefficient_archive(output_dir)
    archive_path = output_dir / "axisymmetric_jones_evolution.npz"
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exact_samples = _exact_publication_samples(profiles, report)
    with np.load(archive_path, allow_pickle=False) as archive:
        saved = {name: np.asarray(archive[name]).copy() for name in archive.files}
    schema_version = int(np.asarray(saved["schema_version"]).item())
    if schema_version != 5:
        raise ValueError(
            f"Cached evolution schema {schema_version} is stale; "
            "rerun with --mode compute --force"
        )
    metadata = json.loads(str(np.asarray(saved["metadata_json"]).item()))
    metadata["exact_samples"] = (
        "named off-grid planes propagated directly from the extracted radial "
        "source profiles; no interpolation along z"
    )
    saved.update(exact_samples)
    saved["metadata_json"] = np.asarray(json.dumps(metadata, indent=2, sort_keys=True))
    saved["report_json"] = np.asarray(json.dumps(report, indent=2, sort_keys=True))
    temporary = archive_path.with_name(archive_path.stem + ".updating.npz")
    np.savez_compressed(temporary, **saved)
    temporary.replace(archive_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive_path, report_path


def _archive_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(values).tolist())


def _load_coefficient_archive(
    output_dir: Path,
) -> tuple[dict[tuple[int, str], AxisymmetricProfile], dict[str, object], Path]:
    path = output_dir / "axisymmetric_jones_coefficients.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing coefficient archive {path}; run this script with "
            "--mode compute first"
        )
    with np.load(path, allow_pickle=False) as archive:
        schema_version = int(np.asarray(archive["schema_version"]).item())
        if schema_version != 3:
            raise ValueError(
                f"Cached coefficient schema {schema_version} is stale; "
                "rerun with --mode compute --force"
            )
        wavelength_nm = tuple(int(value) for value in archive["wavelength_nm"])
        dataset_keys = _archive_strings(archive["dataset_keys"])
        if wavelength_nm != tuple(ARTICLE_WAVELENGTHS_NM):
            raise ValueError(
                f"Cached wavelengths {wavelength_nm} do not match "
                f"{tuple(ARTICLE_WAVELENGTHS_NM)}; rerun with --mode compute"
            )
        if dataset_keys != DATASET_KEYS:
            raise ValueError(
                f"Cached datasets {dataset_keys} do not match {DATASET_KEYS}; "
                "rerun with --mode compute"
            )
        radius_um = np.asarray(archive["radius_um"], dtype=np.float64).copy()
        angular_samples = int(np.asarray(archive["angular_samples"]).item())
        interpolation = str(np.asarray(archive["polar_interpolation"]).item())
        if interpolation != POLAR_INTERPOLATION:
            raise ValueError(
                f"Cached interpolation {interpolation!r} does not match "
                f"{POLAR_INTERPOLATION!r}; rerun with --mode compute --force"
            )
        report = json.loads(str(np.asarray(archive["report_json"]).item()))
        profiles: dict[tuple[int, str], AxisymmetricProfile] = {}
        for wavelength_index, wavelength in enumerate(wavelength_nm):
            for dataset_index, key in enumerate(dataset_keys):
                index = (wavelength_index, dataset_index)
                center = np.asarray(archive["centers_px"][index], dtype=np.float64)
                profiles[(wavelength, key)] = AxisymmetricProfile(
                    radius_um=radius_um,
                    abcd=np.asarray(archive["abcd"][index]).copy(),
                    circular=np.asarray(archive["circular"][index]).copy(),
                    circular_direct=np.asarray(
                        archive["circular_direct"][index]
                    ).copy(),
                    angular_rms=np.asarray(archive["angular_rms"][index]).copy(),
                    reconstruction_rms=np.asarray(
                        archive["reconstruction_rms"][index]
                    ).copy(),
                    signal_rms=np.asarray(archive["signal_rms"][index]).copy(),
                    circular_consistency_rms=np.asarray(
                        archive["circular_consistency_rms"][index]
                    ).copy(),
                    background_abcd=np.asarray(
                        archive["background_abcd"][index]
                    ).copy(),
                    background_jones=np.asarray(
                        archive["background_jones"][index]
                    ).copy(),
                    center_px=(float(center[0]), float(center[1])),
                    angular_samples=angular_samples,
                    interpolation=interpolation,
                )
    return profiles, report, path


def _load_evolution_archive(
    output_dir: Path,
) -> tuple[dict[tuple[int, str], dict[str, object]], dict[str, object], Path]:
    path = output_dir / "axisymmetric_jones_evolution.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing propagation archive {path}; run this script with "
            "--mode compute first"
        )
    with np.load(path, allow_pickle=False) as archive:
        schema_version = int(np.asarray(archive["schema_version"]).item())
        if schema_version != 5:
            raise ValueError(
                f"Cached evolution schema {schema_version} is stale; "
                "rerun with --mode compute --force"
            )
        wavelength_nm = tuple(int(value) for value in archive["wavelength_nm"])
        dataset_keys = _archive_strings(archive["dataset_keys"])
        if wavelength_nm != tuple(ARTICLE_WAVELENGTHS_NM):
            raise ValueError(
                f"Cached wavelengths {wavelength_nm} do not match "
                f"{tuple(ARTICLE_WAVELENGTHS_NM)}; rerun with --mode compute"
            )
        if dataset_keys != PRIMARY_DATASET_KEYS:
            raise ValueError(
                f"Cached datasets {dataset_keys} do not match "
                f"{PRIMARY_DATASET_KEYS}; rerun with --mode compute"
            )
        z_um = np.asarray(archive["z_um"], dtype=np.float64)
        radius_um = np.asarray(archive["radius_um"], dtype=np.float64).copy()
        circular_hankel = np.asarray(archive["circular_hankel"]).copy()
        observable_arrays = {
            name: np.asarray(archive[name]).copy()
            for name in EVOLUTION_OBSERVABLE_NAMES
        }
        report = json.loads(str(np.asarray(archive["report_json"]).item()))
        evolutions: dict[tuple[int, str], dict[str, object]] = {}
        for wavelength_index, wavelength in enumerate(wavelength_nm):
            for dataset_index, key in enumerate(dataset_keys):
                expected_z = PROPAGATION_Z_UM_BY_DATASET[key]
                cached_z = z_um[dataset_index]
                if cached_z.shape != expected_z.shape or not np.allclose(
                    cached_z, expected_z, rtol=0.0, atol=1.0e-12
                ):
                    raise ValueError(
                        f"Cached z grid for {key} does not match the fixed "
                        "propagation range; rerun with --mode compute"
                    )
                index = (wavelength_index, dataset_index)
                hankel = HomogeneousPropagation(
                    z_um=cached_z.copy(),
                    radius_um=radius_um,
                    circular=circular_hankel[index],
                    method="continuous-hankel",
                    wavelength_um=_dataset_propagation_wavelength_um(
                        wavelength,
                        key,
                    ),
                    medium_index=MEDIUM_INDEX,
                    numerical_aperture=None,
                )
                observables = {
                    name: values[index]
                    for name, values in observable_arrays.items()
                }
                focus = report["annular_focus"][f"{wavelength}_{key}"]
                evolutions[(wavelength, key)] = {
                    "hankel": hankel,
                    "observables": observables,
                    "annular_focus": focus,
                }
    return evolutions, report, path


def _validate_green_hankel_against_cartesian(
    evolutions: dict[tuple[int, str], dict[str, object]],
    profiles: dict[tuple[int, str], object],
    report: dict,
) -> None:
    """Check that radial projection commutes with full 2-D propagation."""

    wavelength_nm = 520
    path = simulated_jones_path(PROJECT_ROOT, wavelength_nm, "marching-wave")
    loaded = load_jones_archive(path, dtype=np.complex64)
    pupil_na = effective_coherent_na(PROJECT_ROOT, wavelength_nm)
    wavelength_um = optical_parameters(wavelength_nm).wavelength_um
    matched = apply_coherent_pupil(
        loaded.jones,
        pixel_size_um=float(loaded.pixel_size_um),
        wavelength_um=float(wavelength_um),
        numerical_aperture=float(pupil_na),
    )
    propagator = AngularSpectrumJonesPropagator(
        matched,
        pixel_size_um=float(loaded.pixel_size_um),
        wavelength_um=float(wavelength_um),
        numerical_aperture=None,
        padding_factor=2.0,
        taper_fraction=0.15,
    )
    source_profile = profiles[(wavelength_nm, "marching_wave_pupil_matched")]
    hankel = evolutions[(wavelength_nm, "marching_wave_pupil_matched")]["hankel"]
    selected_radius = source_profile.radius_um <= DISPLAY_MAX_RADIUS_UM
    radius = source_profile.radius_um[selected_radius]
    radial_weight = radius
    errors = {}
    for distance in (-10.0, -5.0, 0.0, 5.0, 10.0):
        propagated = propagator.propagate_jones(distance)
        projected = extract_axisymmetric_profile(
            propagated,
            center_px=simulated_array_center(path),
            pixel_size_um=float(loaded.pixel_size_um),
            radius_um=source_profile.radius_um,
            angular_samples=ANGULAR_SAMPLES,
            background_interval_um=BACKGROUND_INTERVAL_UM,
            interpolation=POLAR_INTERPOLATION,
        )
        z_index = int(np.flatnonzero(np.isclose(hankel.z_um, distance))[0])
        radial_abcd = hankel.abcd[z_index, : source_profile.radius_um.size]
        difference = projected.abcd[selected_radius] - radial_abcd[selected_radius]
        reference = (
            radial_abcd[selected_radius] - source_profile.background_abcd[None, :]
        )
        numerator = np.trapezoid(
            np.sum(np.abs(difference) ** 2, axis=-1) * radial_weight,
            radius,
        )
        denominator = np.trapezoid(
            np.sum(np.abs(reference) ** 2, axis=-1) * radial_weight,
            radius,
        )
        errors[f"{distance:g}"] = float(
            np.sqrt(numerator / max(denominator, 1.0e-300))
        )
    report["green_hankel_to_cartesian_validation"] = {
        "description": "project-then-propagate compared with propagate-then-project for the pupil-matched 520-nm marching-wave field",
        "z_um_relative_error": errors,
        "padding_factor": 2.0,
        "taper_fraction": 0.15,
        "comparison_radius_um": [0.0, DISPLAY_MAX_RADIUS_UM],
        "maximum_allowed_relative_error": 0.02,
    }
    maximum_error = max(errors.values())
    if maximum_error > 0.02:
        raise RuntimeError(
            "Radial and padded Cartesian propagation disagree by "
            f"{maximum_error:.3%}, above the 2% acceptance threshold"
        )


def _common_limit(values: tuple[np.ndarray, ...], percentile: float = 99.7) -> float:
    concatenated = np.concatenate([np.asarray(value).ravel() for value in values])
    finite = concatenated[np.isfinite(concatenated)]
    return float(max(np.percentile(finite, percentile), 1.0e-12))


def _surface_z_um(
    evolutions: dict[tuple[int, str], dict[str, object]],
    wavelength_nm: int,
    key: str,
) -> float:
    if key != "experiment":
        return 0.0
    experiment = evolutions[(wavelength_nm, "experiment")]["annular_focus"]
    model = evolutions[
        (wavelength_nm, "marching_wave_pupil_matched")
    ]["annular_focus"]
    return float(experiment["z_um"] - model["z_um"])


def _draw_axial_references(
    axis: plt.Axes,
    evolutions: dict[tuple[int, str], dict[str, object]],
    wavelength_nm: int,
    key: str,
    *,
    color: str = "white",
) -> None:
    axis.axvline(0.0, color=color, lw=0.8, alpha=0.9)
    registration_focus_z_um = evolutions[
        (wavelength_nm, "marching_wave_pupil_matched")
    ]["annular_focus"]["z_um"]
    axis.axvline(
        registration_focus_z_um,
        color=color,
        lw=0.8,
        ls="-.",
        alpha=0.9,
    )
    if key == "experiment":
        holographic_plane_z_um = -_surface_z_um(
            evolutions, wavelength_nm, key
        )
        axis.axvline(
            holographic_plane_z_um,
            color=color,
            lw=0.75,
            ls="--",
            alpha=0.9,
        )


def _set_uniform_font_size(axis: plt.Axes) -> None:
    axis.title.set_fontsize(FIGURE_FONT_SIZE)
    axis.xaxis.label.set_fontsize(FIGURE_FONT_SIZE)
    axis.yaxis.label.set_fontsize(FIGURE_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=FIGURE_FONT_SIZE)


def _surface_aligned_z_um(
    evolutions: dict[tuple[int, str], dict[str, object]],
    wavelength_nm: int,
    key: str,
) -> np.ndarray:
    z_um = evolutions[(wavelength_nm, key)]["hankel"].z_um
    if key == "experiment":
        return z_um - _surface_z_um(evolutions, wavelength_nm, key)
    return z_um


def _plot_coefficient_maps(
    path: Path,
    evolutions: dict[tuple[int, str], dict[str, object]],
    profiles: dict[tuple[int, str], object],
    *,
    wavelength_nm: int,
    circular: bool = False,
) -> None:
    keys = PRIMARY_DATASET_KEYS
    labels = CIRCULAR_LABELS if circular else ABCD_LABELS
    figure = plt.figure(figsize=(7.15, 6.55), layout="compressed")
    grid = figure.add_gridspec(
        4,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        wspace=0.04,
        hspace=0.06,
    )
    axes = np.empty((4, 3), dtype=object)
    for row in range(4):
        row_values = []
        for key in keys:
            evolution = evolutions[(wavelength_nm, key)]["hankel"]
            if circular:
                values = evolution.circular[..., row]
                background = circular_from_abcd(
                    profiles[(wavelength_nm, key)].background_abcd
                )[row]
            else:
                values = evolution.abcd[..., row]
                background = profiles[(wavelength_nm, key)].background_abcd[row]
            row_values.append(np.abs(values - background))
        limit = _common_limit(tuple(row_values))
        for column, (key, values) in enumerate(zip(keys, row_values, strict=True)):
            axis = figure.add_subplot(grid[row, column])
            axes[row, column] = axis
            evolution = evolutions[(wavelength_nm, key)]["hankel"]
            selected = evolution.radius_um <= DISPLAY_MAX_RADIUS_UM
            image = axis.pcolormesh(
                _surface_aligned_z_um(evolutions, wavelength_nm, key),
                evolution.radius_um[selected],
                values[:, selected].T,
                shading="auto",
                cmap="magma",
                vmin=0.0,
                vmax=limit,
                rasterized=True,
            )
            _draw_axial_references(axis, evolutions, wavelength_nm, key)
            if row == 0:
                axis.set_title(DATASET_LABELS[key])
            if column == 0:
                symbol = labels[row].replace("_plus", "+").replace("_minus", "-")
                axis.set_ylabel(
                    rf"$|\delta {symbol}|$" + "\n" + r"$\rho$ ($\mathregular{\mu}$m)",
                )
            else:
                axis.set_yticklabels([])
            if row < 3:
                axis.set_xticklabels([])
            _set_uniform_font_size(axis)
        color_axis = figure.add_subplot(grid[row, 3])
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE)
    for column in range(3):
        axes[-1, column].set_xlabel(r"$z$ ($\mathregular{\mu}$m)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FINAL_DPI)
    figure.savefig(path.with_suffix(".png"), dpi=FINAL_DPI)
    plt.close(figure)


def _radial_average(values: np.ndarray, radius_um: np.ndarray, selected: np.ndarray) -> np.ndarray:
    radius = radius_um[selected]
    numerator = np.trapezoid(values[:, selected] * radius[None, :], radius, axis=1)
    denominator = np.trapezoid(radius, radius)
    return numerator / denominator


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> list[float]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    coordinate = np.cumsum(sorted_weights)
    coordinate /= coordinate[-1]
    return [float(value) for value in np.interp(quantiles, coordinate, sorted_values)]


def _ridge_slope_report(
    evolutions: dict[tuple[int, str], dict[str, object]],
    *,
    wavelength_nm: int,
) -> dict[str, object]:
    """Estimate outer-fringe slopes from high-pass log-observable maps."""

    radial_interval_um = (15.0, 35.0)
    result: dict[str, object] = {
        "definition": "weighted local contour slope d rho/dz=-partial_z X/partial_rho X",
        "map": "high-pass log(abs(observable))",
        "radial_interval_um": list(radial_interval_um),
        "slow_background_sigma_um": [3.0, 1.0],
        "gradient_selection": "upper quartile",
        "reported_weighted_quantiles": [0.16, 0.5, 0.84],
        "observables": {},
    }
    for name in ("D_I", "D_P", "crossed_H2", "crossed_H4"):
        entries = {}
        for key in PRIMARY_DATASET_KEYS:
            evolution = evolutions[(wavelength_nm, key)]["hankel"]
            values = np.abs(evolutions[(wavelength_nm, key)]["observables"][name])
            radial = (
                (evolution.radius_um >= radial_interval_um[0])
                & (evolution.radius_um <= radial_interval_um[1])
            )
            positive = values[:, radial][values[:, radial] > 0.0]
            floor = max(float(np.percentile(positive, 1.0)) * 0.1, 1.0e-300)
            mapped = np.log(np.maximum(values[:, radial], floor))
            dz = float(np.median(np.diff(evolution.z_um)))
            dr = float(np.median(np.diff(evolution.radius_um[radial])))
            residual = mapped - gaussian_filter(
                mapped,
                sigma=(3.0 / dz, 1.0 / dr),
                mode="nearest",
            )
            gradient_z = np.gradient(residual, evolution.z_um, axis=0)
            gradient_r = np.gradient(
                residual, evolution.radius_um[radial], axis=1
            )
            magnitude = np.hypot(gradient_z, gradient_r)
            slopes = -gradient_z / np.where(
                np.abs(gradient_r) > 1.0e-12, gradient_r, np.nan
            )
            selected = (
                np.isfinite(slopes)
                & (np.abs(slopes) < 1.0)
                & (magnitude >= np.percentile(magnitude, 75.0))
            )
            entries[key] = {
                "slope_quantiles": _weighted_quantiles(
                    slopes[selected],
                    magnitude[selected],
                    (0.16, 0.5, 0.84),
                ),
                "selected_samples": int(np.count_nonzero(selected)),
            }
        result["observables"][name] = entries
    return result


def _plot_observable_maps(
    path: Path,
    evolutions: dict[tuple[int, str], dict[str, object]],
    *,
    wavelength_nm: int,
) -> None:
    rows = (
        ("D_I", r"$D_I$"),
        ("D_P", r"$D_P$"),
        ("crossed_H2", r"$|H_2|$"),
        ("crossed_H4", r"$|H_4|$"),
    )
    keys = PRIMARY_DATASET_KEYS
    figure = plt.figure(figsize=(7.15, 6.55), layout="compressed")
    grid = figure.add_gridspec(
        4,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        wspace=0.04,
        hspace=0.055,
    )
    axes = np.empty((4, 3), dtype=object)
    for row, (name, label) in enumerate(rows):
        row_values = []
        for key in keys:
            values = evolutions[(wavelength_nm, key)]["observables"][name]
            row_values.append(np.abs(values))
        maximum = _common_limit(tuple(row_values), percentile=99.5)
        for column, (key, values) in enumerate(zip(keys, row_values, strict=True)):
            axis = figure.add_subplot(grid[row, column])
            axes[row, column] = axis
            evolution = evolutions[(wavelength_nm, key)]["hankel"]
            selected = evolution.radius_um <= DISPLAY_MAX_RADIUS_UM
            image = axis.pcolormesh(
                _surface_aligned_z_um(evolutions, wavelength_nm, key),
                evolution.radius_um[selected],
                values[:, selected].T,
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=maximum,
                rasterized=True,
            )
            _draw_axial_references(axis, evolutions, wavelength_nm, key)
            focus = evolutions[(wavelength_nm, key)]["annular_focus"]
            axis.plot(
                focus["z_um"] - _surface_z_um(
                    evolutions, wavelength_nm, key
                ),
                focus["radius_um"],
                marker="+",
                color="white",
                ms=5.5,
                mew=0.9,
                linestyle="none",
            )
            if row == 0:
                axis.set_title(DATASET_LABELS[key])
            if column == 0:
                axis.set_ylabel(label + "\n" + r"$\rho$ ($\mathregular{\mu}$m)")
            else:
                axis.set_yticklabels([])
            if row < 3:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(r"$z$ ($\mathregular{\mu}$m)")
            _set_uniform_font_size(axis)
        color_axis = figure.add_subplot(grid[row, 3])
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_ticks(np.linspace(0.0, maximum, 3))
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        colorbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE, pad=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FINAL_DPI)
    figure.savefig(path.with_suffix(".png"), dpi=FINAL_DPI)
    plt.close(figure)


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("At least two one-dimensional centers are required")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _reduced_orientation_rgb(
    reduced_angle: np.ndarray,
    saturation: np.ndarray,
) -> np.ndarray:
    hue = np.mod(
        reduced_angle / (2.0 * np.pi) + 0.5 + LCSIM_AZIMUTH_HUE_SHIFT,
        1.0,
    )
    hsv = np.stack(
        (
            hue,
            np.clip(saturation, 0.0, 1.0),
            np.ones_like(hue),
        ),
        axis=-1,
    )
    return hsv_to_rgb(hsv)


def _orientation_colormap() -> ListedColormap:
    reduced_angle = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    hue = np.mod(
        reduced_angle / (2.0 * np.pi) + 0.5 + LCSIM_AZIMUTH_HUE_SHIFT,
        1.0,
    )
    hsv = np.stack((hue, np.ones_like(hue), np.ones_like(hue)), axis=-1)
    return ListedColormap(hsv_to_rgb(hsv), name="lcsim_azimuth")


def _orientation_saturation_colormap() -> ListedColormap:
    saturation = np.linspace(0.0, 1.0, 256)
    reduced_angle = np.zeros_like(saturation)
    return ListedColormap(
        _reduced_orientation_rgb(reduced_angle, saturation),
        name="lcsim_fixed_hue_saturation",
    )


def _plot_symmetry_axis_maps(
    path: Path | None,
    c4_breaking_path: Path | None,
    archive_path: Path,
    evolutions: dict[tuple[int, str], dict[str, object]],
    *,
    wavelength_nm: int,
    save_archive: bool = True,
) -> None:
    keys = PRIMARY_DATASET_KEYS
    selected_by_key = {
        key: evolutions[(wavelength_nm, key)]["hankel"].radius_um
        <= DISPLAY_MAX_RADIUS_UM
        for key in keys
    }
    harmonic_by_order = {
        2: {
            key: evolutions[(wavelength_nm, key)]["observables"]["crossed_H2"][
                :, selected_by_key[key]
            ]
            for key in keys
        },
        4: {
            key: evolutions[(wavelength_nm, key)]["observables"]["crossed_H4"][
                :, selected_by_key[key]
            ]
            for key in keys
        },
    }
    mean_intensity = {
        key: np.maximum(
            evolutions[(wavelength_nm, key)]["observables"][
                "crossed_mean_intensity"
            ][:, selected_by_key[key]],
            0.0,
        )
        for key in keys
    }
    global_maximum = {
        order: max(
            float(np.max(np.abs(values)))
            for values in harmonic_by_order[order].values()
        )
        for order in (2, 4)
    }
    all_mean_intensity = np.concatenate(
        [values.ravel() for values in mean_intensity.values()]
    )
    intensity_floor = 0.01 * float(np.percentile(all_mean_intensity, 99.0))
    reflection = {
        key: reflection_axis_observables(
            harmonic_by_order[2][key],
            harmonic_by_order[4][key],
        )
        for key in keys
    }
    visibility = {
        key: np.clip(
            np.sqrt(2.0 * reflection[key]["angular_power"])
            / (mean_intensity[key] + intensity_floor),
            0.0,
            1.0,
        )
        for key in keys
    }
    c4_breaking = {
        key: np.clip(
            visibility[key] * (1.0 - reflection[key]["fourfold_quality"]),
            0.0,
            1.0,
        )
        for key in keys
    }
    for key in keys:
        if np.any(
            reflection[key]["c4_mirror_fidelity"]
            > reflection[key]["mirror_fidelity"] + 5.0e-10
        ):
            raise RuntimeError(
                f"Nested mirror fidelity failed for {key}"
            )

    rows = (
        ("H2_absolute", r"twofold harmonic $H_2$", r"$S_2^{\rm abs}$"),
        ("H4_absolute", r"fourfold harmonic $H_4$", r"$S_4^{\rm abs}$"),
        ("mirror_family", r"local mirror family", r"$W_M$"),
        ("c4_mirror_family", r"$C_4$ mirror family", r"$W_{4M}$"),
    )
    figure = plt.figure(figsize=(7.9, 7.45), layout="compressed")
    grid = figure.add_gridspec(
        4,
        5,
        width_ratios=(1.0, 1.0, 1.0, 0.04, 0.045),
        wspace=0.03,
        hspace=0.045,
    )
    saved: dict[str, np.ndarray] = {
        "schema_version": np.asarray(6, dtype=np.int16),
        "wavelength_nm": np.asarray(wavelength_nm, dtype=np.int16),
        "dataset_keys": np.asarray(keys),
        "dataset_labels": np.asarray([DATASET_LABELS[key] for key in keys]),
        "global_maximum_H2": np.asarray(global_maximum[2]),
        "global_maximum_H4": np.asarray(global_maximum[4]),
        "mean_intensity_floor": np.asarray(intensity_floor),
    }

    saturation_mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap=_orientation_saturation_colormap(),
    )
    for row, (quantity, row_label, saturation_label) in enumerate(rows):
        for column, key in enumerate(keys):
            if quantity == "H2_absolute":
                harmonic = harmonic_by_order[2][key]
                magnitude = np.abs(harmonic)
                saturation = np.sqrt(
                    np.clip(magnitude / global_maximum[2], 0.0, 1.0)
                )
                reduced_angle = np.angle(np.conj(harmonic))
            elif quantity == "H4_absolute":
                harmonic = harmonic_by_order[4][key]
                magnitude = np.abs(harmonic)
                saturation = np.sqrt(
                    np.clip(magnitude / global_maximum[4], 0.0, 1.0)
                )
                reduced_angle = np.angle(np.conj(harmonic))
            elif quantity == "mirror_family":
                saturation = (
                    visibility[key] * reflection[key]["mirror_fidelity"]
                )
                reduced_angle = reflection[key]["reduced_mirror_family"]
            else:
                saturation = (
                    visibility[key] * reflection[key]["c4_mirror_fidelity"]
                )
                reduced_angle = reflection[key]["reduced_c4_mirror_family"]
            rgb = _reduced_orientation_rgb(reduced_angle, saturation)
            evolution = evolutions[(wavelength_nm, key)]["hankel"]
            radius_um = evolution.radius_um[selected_by_key[key]]
            display_z_um = _surface_aligned_z_um(evolutions, wavelength_nm, key)
            z_edges = _centers_to_edges(display_z_um)
            radial_edges = _centers_to_edges(radius_um)

            axis = figure.add_subplot(grid[row, column])
            axis.imshow(
                np.transpose(rgb, (1, 0, 2)),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(
                    z_edges[0],
                    z_edges[-1],
                    radial_edges[0],
                    radial_edges[-1],
                ),
                rasterized=True,
            )
            _draw_axial_references(
                axis,
                evolutions,
                wavelength_nm,
                key,
                color="0.25",
            )
            if row == 0:
                axis.set_title(DATASET_LABELS[key])
            if column == 0:
                axis.set_ylabel(row_label + "\n" + r"$\rho$ ($\mathregular{\mu}$m)")
            else:
                axis.set_yticklabels([])
            if row < len(rows) - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(r"$z$ ($\mathregular{\mu}$m)")
            _set_uniform_font_size(axis)

            saved[f"{key}_z_um"] = display_z_um.astype(np.float64)
            saved[f"{key}_radius_um"] = radius_um.astype(np.float64)
            saved[f"{key}_{quantity}_reduced_angle"] = reduced_angle.astype(
                np.float32
            )
            saved[f"{key}_{quantity}_saturation"] = saturation.astype(
                np.float32
            )
            saved[f"{key}_angular_visibility"] = visibility[key].astype(
                np.float32
            )
            saved[f"{key}_axis_quality"] = reflection[key]["axis_quality"].astype(
                np.float32
            )
            saved[f"{key}_fourfold_quality"] = reflection[key][
                "fourfold_quality"
            ].astype(np.float32)
            saved[f"{key}_best_axis_rad"] = reflection[key][
                "best_axis_rad"
            ].astype(np.float32)
            saved[f"{key}_fourfold_axis_rad"] = reflection[key][
                "fourfold_axis_rad"
            ].astype(np.float32)
            for name in (
                "mirror_defect",
                "c4_rotation_defect",
                "mirror_fidelity",
                "c4_mirror_fidelity",
            ):
                saved[f"{key}_{name}"] = reflection[key][name].astype(np.float32)
            saved[f"{key}_c4_breaking"] = c4_breaking[key].astype(np.float32)

        saturation_axis = figure.add_subplot(grid[row, 3])
        saturation_colorbar = figure.colorbar(
            saturation_mappable,
            cax=saturation_axis,
        )
        saturation_colorbar.set_ticks((0.0, 1.0))
        saturation_colorbar.set_label(saturation_label, labelpad=1.5)
        saturation_colorbar.ax.tick_params(
            labelsize=FIGURE_FONT_SIZE,
            pad=1.0,
        )

    color_axis = figure.add_subplot(grid[:, 4])
    orientation_mappable = ScalarMappable(
        norm=Normalize(vmin=-np.pi, vmax=np.pi),
        cmap=_orientation_colormap(),
    )
    colorbar = figure.colorbar(orientation_mappable, cax=color_axis)
    colorbar.set_ticks((-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi))
    colorbar.set_ticklabels(
        (r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$")
    )
    colorbar.set_label(r"reduced orientation phase")
    colorbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE, pad=1.5)

    saved["metadata_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": "symmetry-axis-evolution-v6",
                "radial_averaging": "none; every value is local in (z,rho)",
                "harmonic_reduced_angle": "Theta_m=wrap(-arg(H_m))",
                "mirror_family_reduced_angle": "Phi_M=wrap(4*psi_M)",
                "c4_mirror_family_reduced_angle": "Phi_4M=wrap(8*psi_4)",
                "hue": (
                    "lcsim azimuth convention: "
                    "h=(Theta/(2*pi)+1/2+0.81) mod 1"
                ),
                "absolute_saturation": "sqrt(min(1,abs(H_m)/M_m))",
                "M_m": "joint maximum over all displayed datasets, z, and rho",
                "angular_visibility": (
                    "min(1,sqrt(2*(abs(H2)^2+abs(H4)^2))/"
                    "(I0_crossed+epsilon))"
                ),
                "mirror_defect": "sqrt(2*(1-axis_quality))",
                "c4_rotation_defect": "sqrt(1-fourfold_quality)",
                "mirror_fidelity": "max(0,1-mirror_defect)",
                "c4_mirror_fidelity": (
                    "min(mirror_fidelity,max(0,1-c4_rotation_defect))"
                ),
                "mirror_family_saturation": "angular_visibility*mirror_fidelity",
                "c4_mirror_family_saturation": (
                    "angular_visibility*c4_mirror_fidelity"
                ),
                "visible_c4_breaking": (
                    "angular_visibility*(1-fourfold_quality) = "
                    "angular_visibility*abs(H2)^2/(abs(H2)^2+abs(H4)^2)"
                ),
                "axis_quality": "maximum normalized correlation with a reflected image",
                "fourfold_quality": (
                    "one minus the squared normalized quarter-turn defect; "
                    "abs(H4)^2/(abs(H2)^2+abs(H4)^2)"
                ),
                "epsilon": "0.01 times joint 99th percentile of I0_crossed",
                "display_radius_um": [0.0, DISPLAY_MAX_RADIUS_UM],
            },
            indent=2,
            sort_keys=True,
        )
    )

    if save_archive:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive_path, **saved)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=FINAL_DPI)
        figure.savefig(path.with_suffix(".png"), dpi=FINAL_DPI)
    plt.close(figure)

    if c4_breaking_path is None:
        return

    breaking_figure = plt.figure(figsize=(7.9, 2.45), layout="compressed")
    breaking_grid = breaking_figure.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        wspace=0.04,
    )
    for column, key in enumerate(keys):
        evolution = evolutions[(wavelength_nm, key)]["hankel"]
        radius_um = evolution.radius_um[selected_by_key[key]]
        display_z_um = _surface_aligned_z_um(evolutions, wavelength_nm, key)
        axis = breaking_figure.add_subplot(breaking_grid[0, column])
        axis.imshow(
            np.transpose(c4_breaking[key]),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(
                _centers_to_edges(display_z_um)[0],
                _centers_to_edges(display_z_um)[-1],
                _centers_to_edges(radius_um)[0],
                _centers_to_edges(radius_um)[-1],
            ),
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        _draw_axial_references(
            axis,
            evolutions,
            wavelength_nm,
            key,
            color="0.25",
        )
        axis.set_title(DATASET_LABELS[key])
        axis.set_xlabel(r"$z$ ($\mathregular{\mu}$m)")
        if column == 0:
            axis.set_ylabel(r"$\rho$ ($\mathregular{\mu}$m)")
        else:
            axis.set_yticklabels([])
        _set_uniform_font_size(axis)

    breaking_color_axis = breaking_figure.add_subplot(breaking_grid[0, 3])
    breaking_colorbar = breaking_figure.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=1.0),
            cmap=plt.get_cmap("Greys"),
        ),
        cax=breaking_color_axis,
    )
    breaking_colorbar.set_ticks((0.0, 1.0))
    breaking_colorbar.set_label(r"$B_{4\to2}$")
    breaking_colorbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE, pad=1.5)
    c4_breaking_path.parent.mkdir(parents=True, exist_ok=True)
    breaking_figure.savefig(c4_breaking_path, dpi=FINAL_DPI)
    breaking_figure.savefig(c4_breaking_path.with_suffix(".png"), dpi=FINAL_DPI)
    plt.close(breaking_figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coefficient-figure", type=Path, default=DEFAULT_COEFFICIENT_FIGURE)
    parser.add_argument("--observable-figure", type=Path, default=DEFAULT_OBSERVABLE_FIGURE)
    parser.add_argument(
        "--axis-orientation-figure",
        type=Path,
        default=DEFAULT_AXIS_ORIENTATION_FIGURE,
    )
    parser.add_argument(
        "--c4-breaking-figure",
        type=Path,
        default=DEFAULT_C4_BREAKING_FIGURE,
    )
    parser.add_argument("--circular-figure", type=Path, default=DEFAULT_CIRCULAR_FIGURE)
    parser.add_argument(
        "--publication-figures-only",
        action="store_true",
        help="omit diagnostic figures that are not included in the article",
    )
    parser.add_argument(
        "--mode",
        choices=("compute", "plot", "publication", "all"),
        default="plot",
        help=(
            "compute and save propagation, plot existing archives, add exact "
            "publication planes to an existing archive, or compute then reload "
            "the saved archives and plot (default: plot)"
        ),
    )
    add_force_argument(parser)
    args = parser.parse_args()

    coefficient_archive = args.output_dir / "axisymmetric_jones_coefficients.npz"
    evolution_archive = args.output_dir / "axisymmetric_jones_evolution.npz"
    report_path = args.output_dir / "report.json"
    axis_orientation_archive = (
        args.output_dir / "axisymmetric_symmetry_axes_520.npz"
    )
    outputs = []
    if args.mode in {"compute", "all"}:
        outputs.extend(
            (
                coefficient_archive,
                evolution_archive,
                report_path,
                axis_orientation_archive,
            )
        )
    if args.mode == "publication":
        outputs.extend((evolution_archive, report_path))
    if args.mode in {"plot", "all"}:
        figure_paths = (
            args.coefficient_figure,
            args.observable_figure,
            args.axis_orientation_figure,
        )
        outputs.extend(
            path
            for figure_path in figure_paths
            for path in (figure_path, figure_path.with_suffix(".png"))
        )
        if not args.publication_figures_only:
            outputs.extend(
                (
                    args.c4_breaking_figure,
                    args.c4_breaking_figure.with_suffix(".png"),
                    args.circular_figure,
                    args.circular_figure.with_suffix(".png"),
                    args.output_dir / "axisymmetric_symmetry_axes_520.npz",
                )
            )
    refuse_existing_outputs(
        outputs,
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    if args.mode == "publication":
        evolution_archive, report_path = _upgrade_exact_publication_samples(
            args.output_dir
        )
        print(evolution_archive)
        print(report_path)
        return
    if args.mode in {"compute", "all"}:
        profiles, report = _extract_all()
        coefficient_archive = _save_coefficient_archive(
            args.output_dir, profiles, report
        )
        evolutions, report = _propagate_all(profiles, report)
        _validate_green_hankel_against_cartesian(evolutions, profiles, report)
        report["green_outer_fringe_slopes"] = _ridge_slope_report(
            evolutions, wavelength_nm=520
        )
        evolution_archive = _save_evolution_archive(
            args.output_dir, evolutions, profiles, report
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Store the exact five planes used in the publication immediately,
        # so downstream figure scripts never need a separate upgrade command.
        evolution_archive, report_path = _upgrade_exact_publication_samples(
            args.output_dir
        )
        # Persist the local reflection-axis and C4-breaking maps as numerical
        # data.  Figure 5 consumes this archive, but figure rendering itself
        # remains in the later ``make figures`` stage.
        _plot_symmetry_axis_maps(
            None,
            None,
            axis_orientation_archive,
            evolutions,
            wavelength_nm=520,
            save_archive=True,
        )

    if args.mode == "compute":
        print(coefficient_archive)
        print(evolution_archive)
        print(report_path)
        return

    # Reload even in all mode so figures use the persisted intermediate data.
    profiles, _, coefficient_archive = _load_coefficient_archive(args.output_dir)
    evolutions, _, evolution_archive = _load_evolution_archive(args.output_dir)
    _plot_coefficient_maps(
        args.coefficient_figure, evolutions, profiles, wavelength_nm=520
    )
    # Closed Matplotlib figures retain cyclic references to large raster buffers.
    gc.collect()
    if not args.publication_figures_only:
        _plot_coefficient_maps(
            args.circular_figure, evolutions, profiles, wavelength_nm=520, circular=True
        )
        gc.collect()
    _plot_observable_maps(args.observable_figure, evolutions, wavelength_nm=520)
    gc.collect()
    _plot_symmetry_axis_maps(
        args.axis_orientation_figure,
        None if args.publication_figures_only else args.c4_breaking_figure,
        axis_orientation_archive,
        evolutions,
        wavelength_nm=520,
        save_archive=not args.publication_figures_only,
    )
    print(coefficient_archive)
    print(evolution_archive)
    print(report_path)
    print(args.coefficient_figure)
    print(args.observable_figure)
    print(args.axis_orientation_figure)
    if not args.publication_figures_only:
        print(args.c4_breaking_figure)
    print(axis_orientation_archive)
    if not args.publication_figures_only:
        print(args.circular_figure)


if __name__ == "__main__":
    main()
