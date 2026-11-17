#!/usr/bin/env python3
"""Build the compact 520-nm figures used in the PRL manuscript.

The script deliberately separates analysis from rendering and loads Jones
archives sequentially.  This keeps the peak resident memory small enough to
run alongside an editor while preserving a reproducible path from the saved
Jones fields to the manuscript PDFs.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "build/.matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates

import topovec.mgl as tvmgl
import topovec.matplotlib as tvmpl
from topovec import System
from topovec.core import vector_to_rgb
from topovec.optics import apply_coherent_pupil

from toronoptics.analysis_specs import PUBLICATION_BACKGROUND
from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    ARTICLE_VARIANT_SUFFIX,
    decimate_centered_jones,
    experimental_array_center,
    experimental_average_path,
    experimental_jones_propagation_wavelength_um,
    map_center_between_uniform_grids,
    model_director_path,
    optical_parameters,
    simulated_array_center,
    simulated_jones_path,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.axisymmetric_jones import (
    annular_jones_background,
    crossed_intensity_harmonics,
    integrated_harmonic_visibility,
    refine_axisymmetric_center,
)
from toronoptics.axisymmetric_propagation import abcd_from_circular
from toronoptics.covariant_jones import (
    project_covariant_jones,
    project_covariant_polar,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.figure_output import (
    OPTICAL_ANNOTATION_COLOR,
    figure_output_dir,
    recolor_white_annotations,
    release_render_container,
)
from toronoptics.focus_stack import AngularSpectrumJonesPropagator
from toronoptics.jones_ensemble_average import extract_centered_jones_physical
from toronoptics.simulate_jones import load_director_archive


FIGURE_DIR = figure_output_dir(PROJECT_ROOT)
ANALYSIS_DIR = PROJECT_ROOT / f"data/analysis/main_symmetry_520{ARTICLE_VARIANT_SUFFIX}"
ANALYSIS_ARCHIVE = ANALYSIS_DIR / "main_symmetry_520.npz"
ANALYSIS_REPORT = ANALYSIS_DIR / "report.json"
CONTROL_REPORT = (
    PROJECT_ROOT
    / f"data/analysis/constant_helicity_diffraction_control_520{ARTICLE_VARIANT_SUFFIX}"
    / "constant_helicity_diffraction_control.json"
)
PUPIL_REPORT = PROJECT_ROOT / "data/analysis/averaged_pupil_spectrum/report.json"
AXISYMMETRIC_EVOLUTION_REPORT = (
    PROJECT_ROOT
    / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}/report.json"
)
AXISYMMETRIC_SYMMETRY_ARCHIVE = (
    PROJECT_ROOT
    / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}"
    / "axisymmetric_symmetry_axes_520.npz"
)
COVARIANT_RESIDUAL_REPORT = (
    PROJECT_ROOT
    / f"data/analysis/covariant_residual_520{ARTICLE_VARIANT_SUFFIX}"
    / "report.json"
)

DIRECTOR_SOURCE = model_director_path(PROJECT_ROOT)
HELICITY_ARCHIVE = (
    PROJECT_ROOT
    / f"data/analysis/director_helicity{ARTICLE_VARIANT_SUFFIX}/director_helicity_analysis.npz"
)
WAVELENGTH_NM = ARTICLE_GREEN_WAVELENGTH_NM
WAVELENGTH_UM = optical_parameters(WAVELENGTH_NM).wavelength_um
EXPERIMENTAL_JONES_PROPAGATION_WAVELENGTH_UM = (
    experimental_jones_propagation_wavelength_um(WAVELENGTH_NM)
)
EXPERIMENT_MANIFEST = PROJECT_ROOT / "data_src" / "experiment.toml"
CONTROL_WAVELENGTH_NM = WAVELENGTH_NM
CONTROL_WAVELENGTH_UM = WAVELENGTH_UM
CONTROL_PHASE_STEP = 0.025
FIELD_WIDTH_UM = 90.0
RADIAL_CENTERS_UM = np.arange(1.0, 32.01, 0.25)
RADIAL_SIGMA_UM = 0.75
CONTROL_RADIAL_CENTERS_UM = np.arange(0.0, 32.01, 0.25)
INTEGRATION_INTERVAL_UM = (4.0, 28.0)
REGIONAL_INTERVALS_UM = ((1.0, 10.0), (10.0, 28.0))
HELICITY_QUANTILE_PROBABILITIES = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)
AXISYMMETRIC_RADIUS_UM = np.arange(0.0, 45.01, 0.125)
AXISYMMETRIC_BACKGROUND_UM = (41.0, 45.0)
FINAL_DPI = 600
GOLDEN_RATIO = 0.5 * (1.0 + np.sqrt(5.0))
EXPECTED_EXPERIMENTAL_ACQUISITIONS = 9

MAIN_OPTICAL_FIGURE = FIGURE_DIR / f"toron_symmetry_520{ARTICLE_VARIANT_SUFFIX}.pdf"
MAIN_DIRECTOR_FIGURE = FIGURE_DIR / f"toron_director_structure{ARTICLE_VARIANT_SUFFIX}.pdf"
MAIN_DIRECTOR_TOP_VIEW = (
    FIGURE_DIR / f"toron_director_central_slice{ARTICLE_VARIANT_SUFFIX}.png"
)
MAIN_MECHANISM_FIGURE = FIGURE_DIR / f"symmetry_mechanism_520{ARTICLE_VARIANT_SUFFIX}.pdf"
MAIN_FOCUS_SYMMETRY_FIGURE = (
    FIGURE_DIR / f"focus_symmetry_reduction_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)
MECHANISM_REPORT = PROJECT_ROOT / "build/analysis/mechanism_metrics.json"
SUPPLEMENT_CROSSOVER_FIGURE = (
    FIGURE_DIR / f"supplement_symmetry_crossover_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)
DIRECTOR_ORIENTATION_MAP_FIGURE = (
    FIGURE_DIR / f"toron_director_orientation_map{ARTICLE_VARIANT_SUFFIX}.pdf"
)
SUPPLEMENT_DIRECTOR_VIEWS_FIGURE = (
    FIGURE_DIR / f"supplement_director_views{ARTICLE_VARIANT_SUFFIX}.pdf"
)
DIRECTOR_RENDER_CACHE = PROJECT_ROOT / "data/analysis/director_render_cache"
DIRECTOR_PANEL_PATHS = (
    DIRECTOR_RENDER_CACHE / f"toron_director_cylindrical_slice{ARTICLE_VARIANT_SUFFIX}.png",
    DIRECTOR_RENDER_CACHE / f"toron_director_level_surfaces{ARTICLE_VARIANT_SUFFIX}.png",
    DIRECTOR_RENDER_CACHE / f"toron_director_orientation_isolines{ARTICLE_VARIANT_SUFFIX}.png",
)
DIRECTOR_SPHERE_PATH = (
    DIRECTOR_RENDER_CACHE / f"toron_director_orientation_sphere{ARTICLE_VARIANT_SUFFIX}.png"
)
DIRECTOR_MAP_MAX_RADIUS_UM = 30.0
DIRECTOR_MAP_RADIAL_STEP_UM = 0.10

EXPERIMENT_COLOR = "#222222"
MARCHING_COLOR = "#0072b2"
LOCAL_COLOR = "#d55e00"
GEOMETRY_COLOR = "#2f6f4e"
HELICITY_COLOR = "#b24745"
AXISYMMETRIC_COLOR = "#8c6bb1"
MOUNTAIN_HAZE_CMAP = LinearSegmentedColormap.from_list(
    "mountain_haze",
    (
        (0.00, "#f7f7f3"),
        (0.12, "#e8ebee"),
        (0.30, "#ced8df"),
        (0.50, "#a4b5c1"),
        (0.70, "#758b9d"),
        (0.86, "#485d70"),
        (1.00, "#263746"),
    ),
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10.0,
        "savefig.dpi": 400,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


EXPERIMENT_PATH = experimental_average_path(PROJECT_ROOT, WAVELENGTH_NM)
MODEL_SPECS = (
    (
        "experiment",
        "Measured field",
        EXPERIMENT_COLOR,
        EXPERIMENT_PATH,
        experimental_array_center(EXPERIMENT_PATH),
    ),
    (
        "marching",
        "Marching-wave model",
        MARCHING_COLOR,
        simulated_jones_path(PROJECT_ROOT, WAVELENGTH_NM, "marching-wave"),
        simulated_array_center(
            simulated_jones_path(
                PROJECT_ROOT, WAVELENGTH_NM, "marching-wave"
            )
        ),
    ),
    (
        "local_jones",
        "Local-Jones model",
        LOCAL_COLOR,
        simulated_jones_path(PROJECT_ROOT, WAVELENGTH_NM, "spectral-jones"),
        simulated_array_center(
            simulated_jones_path(
                PROJECT_ROOT, WAVELENGTH_NM, "spectral-jones"
            )
        ),
    ),
)
ANALYSIS_SPECS = (
    ("experiment", "Measured field", EXPERIMENT_COLOR),
    (
        "axisymmetric_experiment",
        "Measured field, covariant projection",
        AXISYMMETRIC_COLOR,
    ),
    ("marching", "Marching-wave model", MARCHING_COLOR),
    ("local_jones", "Local-Jones model", LOCAL_COLOR),
)


def _analysis_display_labels(model_keys) -> tuple[str, ...]:
    label_by_key = {
        key: label
        for key, label, _color in ANALYSIS_SPECS
    }
    return tuple(label_by_key[str(key)] for key in model_keys)


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integrated_visibility(coefficients: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    radius = RADIAL_CENTERS_UM
    selected = (radius >= interval[0]) & (radius <= interval[1])
    denominator = np.trapezoid(
        np.abs(coefficients[..., 0, selected]) ** 2 * radius[selected],
        radius[selected],
        axis=-1,
    )
    output = np.empty(coefficients.shape[:-2] + (2,), dtype=np.float64)
    for output_index, mode_index in enumerate((1, 2)):
        numerator = np.trapezoid(
            np.abs(coefficients[..., mode_index, selected]) ** 2 * radius[selected],
            radius[selected],
            axis=-1,
        )
        output[..., output_index] = 2.0 * np.sqrt(
            numerator / np.maximum(denominator, 1.0e-300)
        )
    return output


def _experimental_surface_distance_um() -> float:
    report = json.loads(
        AXISYMMETRIC_EVOLUTION_REPORT.read_text(encoding="utf-8")
    )
    return float(
        report["annular_focus_alignment"][str(WAVELENGTH_NM)][
            "inferred_experimental_surface_z_um"
        ]
    )


@lru_cache(maxsize=1)
def _experimental_surface_field() -> tuple[
    np.ndarray, tuple[float, float], float
]:
    archive = load_jones_archive(EXPERIMENT_PATH, dtype=np.complex64)
    center = experimental_array_center(EXPERIMENT_PATH)
    reduced, reduced_center = decimate_centered_jones(
        archive.jones,
        center,
        factor=2,
    )
    pixel_size_um = 2.0 * float(archive.pixel_size_um)
    propagator = AngularSpectrumJonesPropagator(
        reduced,
        pixel_size_um=pixel_size_um,
        wavelength_um=EXPERIMENTAL_JONES_PROPAGATION_WAVELENGTH_UM,
        numerical_aperture=None,
        padding_factor=2.0,
    )
    surface = propagator.propagate_jones(_experimental_surface_distance_um())
    del archive, reduced, propagator
    gc.collect()
    return surface, tuple(map(float, reduced_center)), pixel_size_um


def _axisymmetric_surface_projection(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
) -> tuple[np.ndarray, object]:
    projection = project_covariant_jones(
        jones,
        center_xy_px=center_px,
        pixel_size_xy_um=pixel_size_um,
        radius_um=AXISYMMETRIC_RADIUS_UM,
        angular_samples=720,
        interpolation="cubic",
        radial_interpolation="cubic",
        radial_extrapolation="nearest",
        axis_order="xy",
    )
    background = annular_jones_background(
        jones,
        center_px=center_px,
        pixel_size_um=pixel_size_um,
        spec=PUBLICATION_BACKGROUND,
    )
    signal_projection = project_covariant_polar(
        projection.polar.sampled_jones - background[None, None],
        phi_rad=projection.polar.phi_rad,
        radius_um=projection.polar.radius_um,
    )
    diagnostics = {
        "projection": projection,
        "signal_projection": signal_projection,
        "background_jones": background,
    }
    return np.asarray(projection.covariant_jones, dtype=np.complex64), diagnostics


def _axisymmetric_projection_residual(profile: object) -> float:
    return float(
        profile["signal_projection"].diagnostics.relative_residual_norm
    )


def _crossed_h0_order_partition(
    radius_um: np.ndarray,
    abcd: np.ndarray,
) -> dict[str, object]:
    """Partition the projected crossed-channel angular mean by field order."""

    radius = np.asarray(radius_um, dtype=np.float64)
    coefficients = np.asarray(abcd, dtype=np.complex128)
    if coefficients.shape != (radius.size, 4):
        raise ValueError("abcd must have shape (len(radius_um), 4)")
    low, high = INTEGRATION_INTERVAL_UM
    selected = (radius >= low) & (radius <= high)
    if np.count_nonzero(selected) < 2:
        raise ValueError("H0 partition interval contains fewer than two samples")
    radial = radius[selected]
    b = coefficients[selected, 1]
    c = coefficients[selected, 2]
    d = coefficients[selected, 3]
    order_zero = float(np.trapezoid(np.abs(b) ** 2 * radial, radial))
    combined_order_two = float(
        np.trapezoid(0.5 * (np.abs(c) ** 2 + np.abs(d) ** 2) * radial, radial)
    )
    total = order_zero + combined_order_two
    if not total > 0.0:
        raise ValueError("integrated crossed-channel H0 must be positive")
    return {
        "definition": (
            "integral rho dr of |b|^2 and "
            "(|c|^2+|d|^2)/2 in the covariantly projected measured field"
        ),
        "integration_interval_um": [float(low), float(high)],
        "order_zero_fraction": order_zero / total,
        "combined_order_plus_minus_two_fraction": combined_order_two / total,
    }


def _robust_summary(values: np.ndarray) -> dict[str, float]:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("Cannot summarize an empty finite sample")
    median = float(np.median(sample))
    robust_sd = float(1.4826 * np.median(np.abs(sample - median)))
    return {
        "median": median,
        "robust_sd": robust_sd,
        "minimum": float(np.min(sample)),
        "maximum": float(np.max(sample)),
    }


def _experimental_source_paths() -> tuple[Path, ...]:
    """Return the individual holographic acquisitions used by the average."""

    with np.load(EXPERIMENT_PATH, allow_pickle=False) as average:
        raw_paths = tuple(Path(str(value)) for value in average["source_paths"])
    source_paths = tuple(
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in raw_paths
    )
    if len(source_paths) != EXPECTED_EXPERIMENTAL_ACQUISITIONS:
        raise ValueError(
            "The publication average must reference exactly "
            f"{EXPECTED_EXPERIMENTAL_ACQUISITIONS} acquisitions, got "
            f"{len(source_paths)}"
        )
    if len({path.resolve() for path in source_paths}) != len(source_paths):
        raise ValueError("The publication average contains duplicate acquisitions")
    return source_paths


def _experimental_repeatability(
    *,
    surface_distance_um: float,
) -> dict[str, object]:
    """Measure acquisition repeatability before coherent ensemble averaging."""

    with np.load(EXPERIMENT_PATH, allow_pickle=False) as average:
        source_centers = np.asarray(average["source_centers_px"], dtype=np.float64)
        reduced_side = (int(average["Jxx"].shape[0]) + 1) // 2
    source_paths = _experimental_source_paths()
    if len(source_paths) != source_centers.shape[0]:
        raise ValueError("Source paths and centers have inconsistent lengths")

    records: list[dict[str, object]] = []
    center = (0.5 * (reduced_side - 1), 0.5 * (reduced_side - 1))
    for source_index, (source, source_center) in enumerate(
        zip(source_paths, source_centers, strict=True), start=1
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
            crop_size_px=reduced_side,
        )
        if not np.all(valid) or not np.isfinite(centered).all():
            raise ValueError(f"The repeatability crop extends beyond {source}")
        propagator = AngularSpectrumJonesPropagator(
            np.asarray(centered, dtype=np.complex64),
            pixel_size_um=pixel_size_um,
            wavelength_um=EXPERIMENTAL_JONES_PROPAGATION_WAVELENGTH_UM,
            numerical_aperture=None,
            padding_factor=2.0,
        )
        surface = propagator.propagate_jones(float(surface_distance_um))
        projected, radial_profile = _axisymmetric_surface_projection(
            surface,
            center_px=center,
            pixel_size_um=pixel_size_um,
        )
        measured_harmonics = crossed_intensity_harmonics(
            surface,
            center_px=center,
            pixel_size_um=pixel_size_um,
            radial_centers_um=RADIAL_CENTERS_UM,
            radial_sigma_um=RADIAL_SIGMA_UM,
        )
        projected_harmonics = crossed_intensity_harmonics(
            projected,
            center_px=center,
            pixel_size_um=pixel_size_um,
            radial_centers_um=RADIAL_CENTERS_UM,
            radial_sigma_um=RADIAL_SIGMA_UM,
        )

        def visibility(profiles: object, mode: int) -> float:
            return integrated_harmonic_visibility(
                profiles,
                mode=mode,
                min_radius_um=INTEGRATION_INTERVAL_UM[0],
                max_radius_um=INTEGRATION_INTERVAL_UM[1],
            )

        records.append(
            {
                "index": source_index,
                "source": _relative(source),
                "center_px": source_center.tolist(),
                "axisymmetric_projection_residual": _axisymmetric_projection_residual(
                    radial_profile
                ),
                "measured_V2": visibility(measured_harmonics, 2),
                "measured_V4": visibility(measured_harmonics, 4),
                "projected_V2": visibility(projected_harmonics, 2),
                "projected_V4": visibility(projected_harmonics, 4),
            }
        )
        del (
            archive,
            centered,
            propagator,
            surface,
            projected,
            radial_profile,
            measured_harmonics,
            projected_harmonics,
        )
        gc.collect()

    metric_names = (
        "axisymmetric_projection_residual",
        "measured_V2",
        "measured_V4",
        "projected_V2",
        "projected_V4",
    )
    summary = {
        name: _robust_summary(
            np.asarray([float(record[name]) for record in records], dtype=np.float64)
        )
        for name in metric_names
    }
    return {
        "method": (
            "each centered source reconstruction is independently propagated to "
            "the common MW-referenced surface; values describe variation among "
            "nine distinct neighboring torons under common conditions, including "
            "measurement noise and differences between torons"
        ),
        "records": records,
        "summary": summary,
    }


def _model_profiles(
    *,
    pupil_na: float,
) -> dict[str, np.ndarray | list[str]]:
    experiment, experiment_center, experiment_pixel_size = (
        _experimental_surface_field()
    )
    axisymmetric, axisymmetric_profile = _axisymmetric_surface_projection(
        experiment,
        center_px=experiment_center,
        pixel_size_um=experiment_pixel_size,
    )
    sources: list[
        tuple[str, np.ndarray, tuple[float, float], float, str]
    ] = [
        (
            "experiment",
            experiment,
            experiment_center,
            experiment_pixel_size,
            _relative(EXPERIMENT_PATH),
        ),
        (
            "axisymmetric_experiment",
            axisymmetric,
            experiment_center,
            experiment_pixel_size,
            "exact covariant projection of the surface-propagated experiment",
        ),
    ]
    for key, _label, _color, path, stored_center in MODEL_SPECS:
        if key == "experiment":
            continue
        archive = load_jones_archive(path)
        jones = apply_coherent_pupil(
            np.asarray(archive.jones),
            pixel_size_um=float(archive.pixel_size_um),
            wavelength_um=WAVELENGTH_UM,
            numerical_aperture=float(pupil_na),
        )
        sources.append(
            (
                key,
                jones,
                tuple(map(float, stored_center)),
                float(archive.pixel_size_um),
                _relative(path),
            )
        )
        del archive

    coefficients = []
    visibility = []
    integrated = []
    regional = []
    paths = []
    centers = []
    pixel_sizes = []
    for key, jones, center, pixel_size_um, source in sources:
        profiles = crossed_intensity_harmonics(
            jones,
            center_px=center,
            pixel_size_um=pixel_size_um,
            radial_centers_um=RADIAL_CENTERS_UM,
            radial_sigma_um=RADIAL_SIGMA_UM,
        )
        coefficients.append(profiles.coefficients)
        visibility.append(profiles.visibility)
        integrated.append(
            [
                integrated_harmonic_visibility(
                    profiles,
                    mode=mode,
                    min_radius_um=INTEGRATION_INTERVAL_UM[0],
                    max_radius_um=INTEGRATION_INTERVAL_UM[1],
                )
                for mode in (2, 4)
            ]
        )
        regional.append(
            [_integrated_visibility(profiles.coefficients[None, ...], interval)[0] for interval in REGIONAL_INTERVALS_UM]
        )
        paths.append(source)
        centers.append(center)
        pixel_sizes.append(pixel_size_um)
        del profiles
        gc.collect()
    return {
        "coefficients": np.asarray(coefficients, dtype=np.complex64),
        "visibility": np.asarray(visibility, dtype=np.float32),
        "integrated": np.asarray(integrated, dtype=np.float32),
        "regional": np.asarray(regional, dtype=np.float32),
        "paths": paths,
        "centers": np.asarray(centers, dtype=np.float32),
        "pixel_sizes": np.asarray(pixel_sizes, dtype=np.float64),
        "axisymmetric_radius_um": np.asarray(
            axisymmetric_profile["projection"].polar.radius_um,
            dtype=np.float32,
        ),
        "axisymmetric_abcd": np.asarray(
            abcd_from_circular(
                axisymmetric_profile["projection"].polar.radial_channels
            ),
            dtype=np.complex64,
        ),
        "axisymmetric_projection_residual": _axisymmetric_projection_residual(
            axisymmetric_profile
        ),
        "experimental_surface_distance_um": _experimental_surface_distance_um(),
    }


def analyze() -> Path:
    pupil_report = json.loads(PUPIL_REPORT.read_text(encoding="utf-8"))
    pupil_summary = pupil_report["per_wavelength"][str(WAVELENGTH_NM)]
    pupil_na = float(pupil_summary["source_median_effective_na"])
    model = _model_profiles(pupil_na=pupil_na)
    crossed_h0_order_partition = _crossed_h0_order_partition(
        model["axisymmetric_radius_um"],
        model["axisymmetric_abcd"],
    )
    repeatability = _experimental_repeatability(
        surface_distance_um=float(model["experimental_surface_distance_um"])
    )
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    experimental_source_paths = _experimental_source_paths()
    input_paths = (
        EXPERIMENT_PATH,
        EXPERIMENT_MANIFEST,
        *experimental_source_paths,
        *(Path(spec[3]) for spec in MODEL_SPECS if spec[0] != "experiment"),
        PUPIL_REPORT,
        AXISYMMETRIC_EVOLUTION_REPORT,
        COVARIANT_RESIDUAL_REPORT,
    )
    metadata = {
        "schema_version": "main-symmetry-520-v6",
        "inputs": [
            {
                "path": _relative(path),
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "wavelength_nm": WAVELENGTH_NM,
        "wavelength_um": WAVELENGTH_UM,
        "field_calculation_wavelength_um": WAVELENGTH_UM,
        "experimental_jones_propagation_wavelength_um": (
            EXPERIMENTAL_JONES_PROPAGATION_WAVELENGTH_UM
        ),
        "experimental_pixel_size_um": float(model["pixel_sizes"][0]),
        "experimental_center_px": np.asarray(model["centers"])[0].tolist(),
        "experimental_surface_distance_um": float(
            model["experimental_surface_distance_um"]
        ),
        "experimental_plane": (
            "liquid-crystal exit surface inferred by annular-focus matching"
        ),
        "center_method": (
            "registration anchor stored in the nine-reconstruction complex average"
        ),
        "axisymmetric_projection": {
            "method": (
                "cubic sampling followed by exact circular-basis spin-order "
                "locking: J++ and J-- keep m=0, J-+ keeps m=+2, and J+- "
                "keeps m=-2"
            ),
            "integrated_relative_jones_residual": float(
                model["axisymmetric_projection_residual"]
            ),
            "background_interval_um": list(AXISYMMETRIC_BACKGROUND_UM),
            "background_estimator": PUBLICATION_BACKGROUND.estimator,
        },
        "crossed_H0_order_partition": crossed_h0_order_partition,
        "harmonic_method": (
            "Gaussian-annulus Fourier coefficients of |J_yx|^2 with sigma=0.75 um"
        ),
        "covariant_residual_decomposition": json.loads(
            COVARIANT_RESIDUAL_REPORT.read_text(encoding="utf-8")
        )["average"],
        "pupil": {
            "effective_na": pupil_na,
            "robust_sd": float(pupil_summary["source_robust_sd_effective_na"]),
            "calibration": _relative(PUPIL_REPORT),
            "method": (
                "sharpest common radial spectral edge of the four complex Jones "
                "channels, estimated independently in nine 520-nm reconstructions"
            ),
            "application": "applied componentwise only to simulated Jones fields",
        },
        "repeatability": repeatability,
    }
    np.savez_compressed(
        ANALYSIS_ARCHIVE,
        schema_version=np.asarray("main-symmetry-520-v6"),
        wavelength_nm=np.asarray(WAVELENGTH_NM),
        radial_centers_um=RADIAL_CENTERS_UM.astype(np.float32),
        modes=np.asarray((0, 2, 4), dtype=np.int16),
        model_keys=np.asarray([item[0] for item in ANALYSIS_SPECS]),
        model_labels=np.asarray([item[1] for item in ANALYSIS_SPECS]),
        model_colors=np.asarray([item[2] for item in ANALYSIS_SPECS]),
        model_paths=np.asarray(model["paths"]),
        model_centers_px=np.asarray(model["centers"]),
        model_pixel_size_um=np.asarray(model["pixel_sizes"]),
        model_coefficients=np.asarray(model["coefficients"]),
        model_visibility=np.asarray(model["visibility"]),
        model_integrated_visibility=np.asarray(model["integrated"]),
        model_regional_visibility=np.asarray(model["regional"]),
        axisymmetric_radius_um=np.asarray(model["axisymmetric_radius_um"]),
        axisymmetric_abcd=np.asarray(model["axisymmetric_abcd"]),
        axisymmetric_projection_residual=np.asarray(
            model["axisymmetric_projection_residual"], dtype=np.float64
        ),
        experimental_surface_distance_um=np.asarray(
            model["experimental_surface_distance_um"], dtype=np.float64
        ),
        metadata_json=np.asarray(json.dumps(metadata, indent=2)),
        repeatability_json=np.asarray(json.dumps(repeatability, indent=2)),
    )
    report = {
        **metadata,
        "regional_visibility": {
            key: {
                "inner_1_10_um": {
                    "V2": float(model["regional"][index, 0, 0]),
                    "V4": float(model["regional"][index, 0, 1]),
                },
                "outer_10_28_um": {
                    "V2": float(model["regional"][index, 1, 0]),
                    "V4": float(model["regional"][index, 1, 1]),
                },
            }
            for index, (key, *_rest) in enumerate(ANALYSIS_SPECS)
        },
        "model_integrated_visibility": {
            key: {"V2": float(model["integrated"][index, 0]), "V4": float(model["integrated"][index, 1])}
            for index, (key, *_rest) in enumerate(ANALYSIS_SPECS)
        },
        "archive": _relative(ANALYSIS_ARCHIVE),
    }
    ANALYSIS_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return ANALYSIS_ARCHIVE


def _load_analysis() -> dict[str, np.ndarray]:
    if not ANALYSIS_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing {ANALYSIS_ARCHIVE}; run this script with --analyze first"
        )
    with np.load(ANALYSIS_ARCHIVE, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _centered_crop(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    width_um: float = FIELD_WIDTH_UM,
) -> np.ndarray:
    side = min(int(round(float(width_um) / float(pixel_size_um))), *jones.shape[:2])
    side = max(side, 2)
    x0 = int(round(float(center_px[0]) - 0.5 * (side - 1)))
    y0 = int(round(float(center_px[1]) - 0.5 * (side - 1)))
    x0 = int(np.clip(x0, 0, jones.shape[0] - side))
    y0 = int(np.clip(y0, 0, jones.shape[1] - side))
    return np.ascontiguousarray(jones[x0 : x0 + side, y0 : y0 + side])


def _render_crossed_panel(
    jones: np.ndarray,
    *,
    wavelength_um: float,
    pixel_size_um: float,
    show_axes: bool,
    show_scale_bar: bool,
    show_polarization: bool,
    pixels: int = 1500,
    frame_scale: float = 1.0,
) -> np.ndarray:
    # Enlarge the frame, keeping the optical image and annotations at their
    # original printed size. The extra viewport is presentation background.
    frame_pixels = int(round(pixels * frame_scale))
    frame_scale = frame_pixels / float(pixels)
    container = tvmgl.render_precomputed_optics(
        {"jones_matrices": np.ascontiguousarray(jones)},
        size=(frame_pixels, frame_pixels),
        scene_state={
            "phase_reference_mode": "Solver native",
            "wavelength_um": float(wavelength_um),
            "pixel_size_um": float(pixel_size_um),
        },
    )
    scene = container.scene
    scene.set_overlay_outline_mode("Hide")
    scene.set_polarization_preset("Linear X")
    scene.set_analyzer_preset("Linear 90")
    scene.set_exposure_ev(0.0)
    scene.set_tone_curve("sRGB")
    scene.set_color_bar_show("Hide")
    scene.set_panel_label_show("Hide")
    scene.set_panel_outside_background("Black")
    scene.show_axes = bool(show_axes)
    scene.axes_render.color_mode = "White"
    scene.axes_render.location = (-1.0 + 0.22 / frame_scale,) * 2
    scene.axes_render.length = 0.18 / frame_scale
    # TopoVec caps the rendered annotation sizes at 10%.
    scene.axes_render.font_size_percent = 10.0 / frame_scale
    scene.axes_render.label_gap_percent = 1.5 / frame_scale
    scene.set_scale_bar_show("Show" if show_scale_bar else "Hide")
    scene.set_scale_bar_length_mode("Manual")
    scene.set_scale_bar_length(20.0)
    scale_bar_location = (1.0 - 0.12 / frame_scale, -1.0 + 0.20 / frame_scale)
    scene.set_scale_bar_location_x(scale_bar_location[0])
    scene.set_scale_bar_location_y(scale_bar_location[1])
    scene.set_scale_bar_color("White")
    scene.scale_bar_font_size_percent = 10.0 / frame_scale
    scene.set_scale_bar_line_width_percent(0.9 / frame_scale)
    overlay_mode = "Auto" if show_polarization else "Hide"
    scene.set_polarization_incident_show(overlay_mode)
    scene.set_polarization_analyzer_show(overlay_mode)
    scene.set_polarization_direction_show(overlay_mode)
    scene.set_polarization_location_x(1.0 - 0.32 / frame_scale)
    scene.set_polarization_location_y(1.0 - 0.30 / frame_scale)
    scene.set_polarization_size(0.10 / frame_scale)
    scene.set_polarization_line_width(0.0030 / frame_scale)
    scene.set_polarization_color("White")
    # The image geometry is initialized on the first draw, before zoom works.
    container.render_array_rgba()
    scene.zoom_fixed_view(frame_scale)
    image = np.asarray(container.save(alpha=True), dtype=np.uint8)
    if show_scale_bar and getattr(scene, "_fixed_view_scale_bar_overlay_failed", False):
        raise RuntimeError("TopoVec failed to render the optical scale bar")
    scene.show_axes = False
    scene.set_scale_bar_show("Hide")
    scene.set_polarization_incident_show("Hide")
    scene.set_polarization_analyzer_show("Hide")
    scene.set_polarization_direction_show("Hide")
    background = np.asarray(container.render_array_rgba(), dtype=np.uint8)
    image = recolor_white_annotations(image, background)
    release_render_container(container)
    del scene, container
    gc.collect()
    return image


def _panel_label(ax: plt.Axes, label: str, *, image: bool = False, fontsize: float = 10.0) -> None:
    if image:
        x, y = 0.018, 0.982
        horizontal_alignment, vertical_alignment, color = "left", "top", OPTICAL_ANNOTATION_COLOR
    else:
        x, y = -0.095, -0.16
        horizontal_alignment, vertical_alignment, color = "right", "top", "black"
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        color=color,
        fontsize=fontsize,
        clip_on=False,
    )


def _style_plot_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10, length=2.5)


def _top_panel_label(ax: plt.Axes, label: str, *, fontsize: float = 10.0) -> None:
    ax.text(
        0.01,
        1.015,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="black",
        fontsize=fontsize,
        clip_on=False,
    )


def _lower_left_figure_label(
    fig: plt.Figure,
    *,
    left: float,
    bottom: float,
    label: str,
) -> None:
    fig.text(
        left,
        bottom,
        f"({label})",
        ha="left",
        va="bottom",
        color="black",
        fontsize=10.0,
        zorder=20,
    )


def _axis_label_intersection_panel_label(
    fig: plt.Figure,
    ax: plt.Axes,
    label: str,
) -> None:
    """Place a panel label at the crossing of the two axis-label baselines."""

    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    x_label_box = ax.xaxis.label.get_window_extent(renderer).transformed(to_figure)
    y_label_box = ax.yaxis.label.get_window_extent(renderer).transformed(to_figure)
    fig.text(
        0.5 * (y_label_box.x0 + y_label_box.x1),
        0.5 * (x_label_box.y0 + x_label_box.y1),
        f"({label})",
        ha="center",
        va="center",
        color="black",
        fontsize=10.0,
        zorder=20,
    )


def _horizontal_axis_panel_label(
    fig: plt.Figure,
    ax: plt.Axes,
    label: str,
) -> None:
    """Place a panel label at the left edge of the horizontal-axis label line."""

    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    x_label_box = ax.xaxis.label.get_window_extent(renderer).transformed(to_figure)
    fig.text(
        ax.get_position().x0,
        0.5 * (x_label_box.y0 + x_label_box.y1),
        f"({label})",
        ha="left",
        va="center",
        color="black",
        fontsize=10.0,
        zorder=20,
    )


def _plot_director_color_sphere(ax: plt.Axes) -> None:
    """Draw the director-color legend viewed along the y axis."""

    tvmpl.plot_direction_colormap(
        axis=1,
        hemisphere=-1,
        size=320,
        ax=ax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(-1.24, 1.62)
    ax.set_ylim(-1.24, 1.24)
    arrow_kw = {
        "arrowstyle": "-|>",
        "mutation_scale": 7.5,
        "linewidth": 1.0,
        "color": "black",
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate("", xy=(1.0, 0.0), xytext=(0.0, 0.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.0, 1.0), xytext=(0.0, 0.0), arrowprops=arrow_kw)
    ax.text(1.46, 0.0, r"$n_x$", ha="center", va="center", fontsize=10.0)
    ax.text(0.0, 1.13, r"$n_z$", ha="center", va="center", fontsize=10.0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _load_display_field(
    key: str,
    analysis: dict[str, np.ndarray],
    *,
    pupil_na: float,
) -> tuple[np.ndarray, float]:
    if key in {"experiment", "axisymmetric_experiment"}:
        jones, center, pixel_size_um = _experimental_surface_field()
        if key == "axisymmetric_experiment":
            jones, _profile = _axisymmetric_surface_projection(
                jones,
                center_px=center,
                pixel_size_um=pixel_size_um,
            )
        cropped = _centered_crop(
            jones,
            center_px=center,
            pixel_size_um=pixel_size_um,
        )
        return cropped, pixel_size_um

    spec = next(item for item in MODEL_SPECS if item[0] == key)
    _key, _label, _color, path, stored_center = spec
    archive = load_jones_archive(path)
    jones = np.asarray(archive.jones)
    pixel_size_um = float(archive.pixel_size_um)
    center = stored_center
    jones = apply_coherent_pupil(
        jones,
        pixel_size_um=pixel_size_um,
        wavelength_um=WAVELENGTH_UM,
        numerical_aperture=float(pupil_na),
    )
    cropped = _centered_crop(
        jones,
        center_px=center,
        pixel_size_um=pixel_size_um,
    )
    del archive, jones
    gc.collect()
    return cropped, pixel_size_um


def plot_figure_1() -> Path:
    font_size = 9.05
    analysis = _load_analysis()
    metadata = json.loads(str(analysis["metadata_json"]))
    pupil_na = float(metadata["pupil"]["effective_na"])
    fig = plt.figure(figsize=(7.15, 2.05))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(3.153, 2.0),
        left=0.005,
        right=0.995,
        bottom=0.22,
        top=0.79,
        wspace=0.122,
    )
    image_grid = outer[0, 0].subgridspec(1, 3, wspace=0.012)
    axes = [fig.add_subplot(image_grid[0, column]) for column in range(3)]
    slot = axes[0].get_position(original=True)
    old_side_inches = slot.height * fig.get_figheight()
    group = outer[0, 0].get_position(fig)
    # Fix the outer margin to preserve the physical scale at text width.
    old_slot_width = group.width / (3.0 + 2.0 * 0.08)
    image_left = group.x0 + 0.5 * (old_slot_width - old_side_inches / fig.get_figwidth())
    panel_width = (group.x1 - image_left) / (3.0 + 2.0 * 0.012)
    new_side_inches = panel_width * fig.get_figwidth()
    frame_scale = new_side_inches / old_side_inches
    # Grow upward from the same baseline as the radial graph's lower spine.
    for index, axis in enumerate(axes):
        axis.set_position(
            (image_left + index * panel_width * 1.012, group.y0,
             panel_width, new_side_inches / fig.get_figheight())
        )
    panels = []
    image_keys = (
        "axisymmetric_experiment",
        "marching",
        "local_jones",
    )
    for index, key in enumerate(image_keys):
        jones, pixel_size_um = _load_display_field(key, analysis, pupil_na=pupil_na)
        panel_wavelength_um = (
            EXPERIMENTAL_JONES_PROPAGATION_WAVELENGTH_UM
            if key == "axisymmetric_experiment"
            else WAVELENGTH_UM
        )
        panels.append(
            _render_crossed_panel(
                jones,
                wavelength_um=panel_wavelength_um,
                pixel_size_um=pixel_size_um,
                show_axes=index == 0,
                show_scale_bar=True,
                show_polarization=index == 0,
                frame_scale=frame_scale,
            )
        )
        del jones
        gc.collect()

    image_titles = (
        "Measured field,\ncovariant projection",
        "Marching-wave\nmodel",
        "Local-Jones\nmodel",
    )
    for index, panel in enumerate(panels):
        axes[index].imshow(panel, origin="upper", interpolation="nearest")
        axes[index].set_anchor("C")
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        for spine in axes[index].spines.values():
            spine.set_visible(False)
        _panel_label(axes[index], chr(ord("a") + index), image=True, fontsize=font_size)
        axes[index].set_xlabel(image_titles[index], fontsize=font_size, labelpad=3.0, linespacing=0.95)

    radial_axis = fig.add_subplot(outer[0, 1])
    radius = np.asarray(analysis["radial_centers_um"], dtype=np.float64)
    visibility = np.asarray(analysis["model_visibility"], dtype=np.float64)
    labels = _analysis_display_labels(analysis["model_keys"])
    colors = tuple(str(value) for value in analysis["model_colors"])
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        radial_axis.plot(
            radius,
            visibility[index, 1],
            color=color,
            linewidth=1.3,
            label=label,
        )
        radial_axis.plot(
            radius,
            visibility[index, 2],
            color=color,
            linewidth=1.05,
            linestyle=(0, (3.0, 2.0)),
        )
    radial_axis.set_xlim(1.0, 30.0)
    radial_axis.set_ylim(bottom=1.0e-2)
    radial_axis.set_xlabel(r"$\rho\;(\mathregular{\mu}\mathrm{m})$", fontsize=font_size)
    radial_axis.set_ylabel(r"$2|H_m|/H_0$", fontsize=font_size, labelpad=1.0)
    handles, _ = radial_axis.get_legend_handles_labels()
    legend_entries = sorted(
        zip(handles, labels, strict=True), key=lambda item: len(item[1]), reverse=True
    )
    dataset_legend = radial_axis.legend(
        [handle for handle, _ in legend_entries],
        [label for _, label in legend_entries],
        frameon=False,
        fontsize=font_size,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        markerfirst=False,
        handlelength=1.2,
        borderaxespad=0.2,
        labelspacing=0.12,
    )
    dataset_legend.set_clip_on(False)
    radial_axis.add_artist(dataset_legend)
    radial_axis.legend(
        handles=[
            plt.Line2D([0], [0], color="black", linewidth=1.2, label=r"$H_2$"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.0,
                linestyle=(0, (3.0, 2.0)),
                label=r"$H_4$",
            ),
        ],
        frameon=False,
        fontsize=font_size,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.63),
        ncols=2,
        columnspacing=0.8,
        markerfirst=False,
        handlelength=1.2,
        borderaxespad=0.2,
        labelspacing=0.12,
    )
    _style_plot_axis(radial_axis)
    radial_axis.tick_params(labelsize=font_size)
    _top_panel_label(radial_axis, "d", fontsize=font_size)
    MAIN_OPTICAL_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        MAIN_OPTICAL_FIGURE,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)
    return MAIN_OPTICAL_FIGURE


def _wall_level_geometry(
    radius_um: np.ndarray,
    n_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius_um = np.asarray(radius_um, dtype=np.float64)
    n_z = np.asarray(n_z, dtype=np.float64)
    if n_z.ndim != 2 or n_z.shape[1] != radius_um.size:
        raise ValueError("n_z must have shape (n_z_layers, n_radius)")
    level_radii = np.full((3, n_z.shape[0]), np.nan, dtype=np.float64)
    levels = (-0.5, 0.0, 0.5)
    for z_index, profile in enumerate(n_z):
        finite = np.isfinite(profile[:-1]) & np.isfinite(profile[1:])
        for level_index, level in enumerate(levels):
            crossing = np.flatnonzero(
                finite
                & (profile[:-1] <= level)
                & (profile[1:] >= level)
            )
            if crossing.size == 0:
                continue
            index = crossing[-1]
            left = profile[index]
            right = profile[index + 1]
            denominator = right - left
            fraction = (
                0.5 if abs(denominator) < 1.0e-12 else (level - left) / denominator
            )
            level_radii[level_index, z_index] = radius_um[index] + fraction * (
                radius_um[index + 1] - radius_um[index]
            )
    ordered = (
        (level_radii[0] < level_radii[1])
        & (level_radii[1] < level_radii[2])
    )
    level_radii[:, ~ordered] = np.nan
    return level_radii[0], level_radii[1], level_radii[2]


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...],
) -> np.ndarray:
    """Return linearly interpolated weighted quantiles of a finite sample."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.full(probabilities_array.shape, np.nan, dtype=np.float64)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(
        probabilities_array,
        cumulative,
        values,
        left=values[0],
        right=values[-1],
    )


def _lift_nematic_angles(
    nematic_axes: np.ndarray,
    reference_rad: float,
) -> np.ndarray:
    """Lift angles modulo pi onto the branch centered on ``reference_rad``."""

    nematic_axes = np.asarray(nematic_axes, dtype=np.complex128)
    reference_rad = float(reference_rad)
    return reference_rad + 0.5 * np.angle(
        nematic_axes * np.exp(-2.0j * reference_rad)
    )


def _director_geometry(
    mean_tau_rad: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    archive = load_director_archive(DIRECTOR_SOURCE)
    field = np.asarray(archive.director, dtype=np.float64)
    thickness_um = float(archive.thickness_um)
    with np.load(HELICITY_ARCHIVE, allow_pickle=False) as helicity:
        center = tuple(np.asarray(helicity["center_px"], dtype=np.float64))
    mean_tau_rad = np.asarray(mean_tau_rad, dtype=np.float64)
    if mean_tau_rad.shape != (field.shape[2],):
        raise ValueError(
            "The layer helicity must contain one value per director layer"
        )
    spacing_um = thickness_um / float(field.shape[2] - 1)
    radius_um = np.arange(0.0, 40.0001, 0.10, dtype=np.float64)
    coordinates = np.stack(
        (
            center[0] + radius_um / spacing_um,
            np.full(radius_um.size, center[1]),
        )
    )
    n_z_section = np.empty((field.shape[2], radius_um.size), dtype=np.float64)
    helicity_quantiles = np.full(
        (len(HELICITY_QUANTILE_PROBABILITIES), field.shape[2]),
        np.nan,
        dtype=np.float64,
    )
    parameters = optical_parameters(WAVELENGTH_NM)
    ordinary_index = float(parameters.ordinary_index)
    extraordinary_index = float(parameters.extraordinary_index)
    wave_number = 2.0 * np.pi / float(parameters.wavelength_um)
    x_index, y_index = np.indices(field.shape[:2], dtype=np.float64)
    dx = x_index - center[0]
    dy = y_index - center[1]
    spatial_phase = np.exp(-2.0j * np.arctan2(dy, dx))
    away_from_axis = dx * dx + dy * dy >= 0.25

    for z_index in range(field.shape[2]):
        nz = map_coordinates(
            field[..., z_index, 2],
            coordinates,
            order=1,
            mode="nearest",
        )
        n_z_section[z_index] = nz

        if not np.isfinite(mean_tau_rad[z_index]):
            continue
        director_xy = (
            field[..., z_index, 0] + 1.0j * field[..., z_index, 1]
        )
        local_nematic = director_xy**2 * spatial_phase
        transverse_power = np.abs(director_xy) ** 2
        denominator = np.sqrt(
            field[..., z_index, 2] ** 2
            + (ordinary_index / extraordinary_index) ** 2 * transverse_power
        )
        effective_index = ordinary_index / np.maximum(
            denominator, np.finfo(float).tiny
        )
        optical_weight = wave_number * np.maximum(
            effective_index - ordinary_index, 0.0
        )
        active = away_from_axis & (optical_weight > np.finfo(float).tiny)
        helicity_samples = _lift_nematic_angles(
            local_nematic[active], mean_tau_rad[z_index]
        )
        helicity_quantiles[:, z_index] = _weighted_quantiles(
            helicity_samples,
            optical_weight[active],
            HELICITY_QUANTILE_PROBABILITIES,
        )
    wall_inner, wall_radius, wall_outer = _wall_level_geometry(
        radius_um, n_z_section
    )
    wall_center_tau = np.full(field.shape[2], np.nan, dtype=np.float64)
    for z_index, zero_radius in enumerate(wall_radius):
        if (
            not np.isfinite(mean_tau_rad[z_index])
            or not np.isfinite(zero_radius)
        ):
            continue
        zero_coordinates = np.asarray(
            [[center[0] + zero_radius / spacing_um], [center[1]]],
            dtype=np.float64,
        )
        nx = map_coordinates(
            field[..., z_index, 0],
            zero_coordinates,
            order=1,
            mode="nearest",
        )[0]
        ny = map_coordinates(
            field[..., z_index, 1],
            zero_coordinates,
            order=1,
            mode="nearest",
        )[0]
        nematic_axis = (nx + 1.0j * ny) ** 2
        wall_center_tau[z_index] = _lift_nematic_angles(
            nematic_axis, mean_tau_rad[z_index]
        )
    z_um = np.linspace(0.0, thickness_um, field.shape[2])
    del archive, field
    gc.collect()
    return (
        z_um,
        wall_inner,
        wall_radius,
        wall_outer,
        wall_center_tau,
        helicity_quantiles,
    )


def _render_main_director_top_view() -> np.ndarray:
    """Render the detached central xy slice used as panel 2(a)."""

    archive = load_director_archive(DIRECTOR_SOURCE)
    field = np.asarray(archive.director, dtype=np.float32)
    if field.shape[:3] != (200, 200, 51):
        raise ValueError(
            "The publication camera is calibrated for a 200x200x51 director; "
            f"got {field.shape[:3]}"
        )
    system = System.rectangular(tuple(int(value) for value in field.shape[:3]))
    container = tvmgl.render_prepare(
        "Slice glyphs",
        system,
        imgsize=1800,
    )
    container.upload(np.ascontiguousarray(field[..., np.newaxis, :]))
    scene = container.scene
    scene["Style"]["LC surfaces"] = True
    scene["Style"]["Show view marker"] = False
    scene["Axes"]["X"] = -0.76
    scene["Axes"]["Y"] = -0.78
    scene["Axes"]["Length"] = 0.1
    scene["Axes"]["Center size"] = 0.35
    scene["Axes"]["Font size"] = 10.0
    scene["Scale bar"]["X"] = 0.76
    scene["Scale bar"]["Y"] = -0.88
    scene.scale_bar_font_size_percent = 10.0
    scene["Scale bar"]["Line width"] = 0.4
    scene.set_camera(
        tvmgl.Camera.from_angles(
            theta=3.14159,
            phi=0.0,
            alpha=0.0,
            scale=121.326,
            perspective=True,
            look_at=[89.4539, 90.2267, 25.0],
        )
    )
    scene.set_overlay_outline_mode("Hide")
    scene.set_measurement_thickness_um(27.4)
    scene.set_scale_bar_show("Show")
    scene.set_scale_bar_length_mode("Manual")
    scene.set_scale_bar_length(10.0)
    image = np.asarray(container.save(alpha=True), dtype=np.uint8)
    MAIN_DIRECTOR_TOP_VIEW.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGBA").save(MAIN_DIRECTOR_TOP_VIEW)
    release_render_container(container)
    del scene, container, field, archive
    gc.collect()
    return image


def plot_figure_2() -> Path:
    if not HELICITY_ARCHIVE.exists():
        raise FileNotFoundError(HELICITY_ARCHIVE)
    radius_um, map_z_um, rgb, inner_edge, map_wall_radius, outer_edge = (
        _director_orientation_section()
    )
    with np.load(HELICITY_ARCHIVE, allow_pickle=False) as archive:
        helicity_z = np.asarray(archive["z_um"], dtype=np.float64)
        weighting_names = tuple(str(value) for value in archive["weighting_names"])
        optical_index = weighting_names.index("optical")
        tau_rad = np.asarray(
            archive["layer_tau_rad"][optical_index], dtype=np.float64
        )
        tau = np.rad2deg(tau_rad)
        tau_fit = np.rad2deg(
            np.asarray(
                archive["fit_linear_tau_rad"][optical_index], dtype=np.float64
            )
        )
    (
        z_um,
        _wall_inner,
        _profile_wall_radius,
        _wall_outer,
        wall_tau_rad,
        tau_quantiles_rad,
    ) = _director_geometry(tau_rad)
    wall_tau = np.rad2deg(wall_tau_rad)
    tau_quantiles = np.rad2deg(tau_quantiles_rad)
    visible = np.isfinite(tau) & np.isfinite(wall_tau)
    fig = plt.figure(figsize=(7.15, 2.80), layout="compressed")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        wspace=0.06,
    )
    top_view = _render_main_director_top_view()
    reserve_ax = fig.add_subplot(grid[0, 0])
    map_ax = fig.add_subplot(grid[0, 1])
    helicity_ax = fig.add_subplot(grid[0, 2])

    reserve_ax.imshow(top_view, origin="upper", interpolation="none")
    reserve_ax.set_axis_off()

    map_ax.imshow(
        rgb,
        origin="lower",
        extent=(radius_um[0], radius_um[-1], map_z_um[0], map_z_um[-1]),
        interpolation="bilinear",
        aspect="equal",
    )
    map_ax.plot(inner_edge, map_z_um, color="white", linewidth=0.65, zorder=6)
    map_ax.plot(outer_edge, map_z_um, color="#202020", linewidth=0.60, zorder=6)
    map_ax.plot(
        map_wall_radius,
        map_z_um,
        color="#202020",
        linewidth=1.0,
        label=r"$n_z=0$",
        zorder=8,
    )
    map_ax.plot(
        [],
        [],
        color="#202020",
        linewidth=0.60,
        label=r"$n_z=\pm1/2$",
    )
    map_ax.set_xlabel(r"$\rho\;(\mathregular{\mu}\mathrm{m})$", labelpad=1.5)
    map_ax.set_ylabel(r"$z\;(\mathregular{\mu}\mathrm{m})$", labelpad=1.5)
    square_radius_max = radius_um[0] + (map_z_um[-1] - map_z_um[0])
    map_ax.set_xlim(radius_um[0], square_radius_max)
    map_ax.set_ylim(map_z_um[0], map_z_um[-1])
    map_ax.set_box_aspect(1.0)
    map_ax.tick_params(labelsize=10, direction="out", length=2.5, pad=1.5)
    map_ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10.0,
        handlelength=1.6,
        borderaxespad=0.2,
    )
    sphere_ax = map_ax.inset_axes((0.675, 0.68, 0.30, 0.29))
    _plot_director_color_sphere(sphere_ax)

    quantile_bands = []
    for lower, upper, alpha, label in (
        (0, -1, 0.05, "1-99%"),
        (1, -2, 0.08, "5-95%"),
        (2, -3, 0.12, "10-90%"),
        (3, -4, 0.20, "25-75%"),
    ):
        quantile_bands.append(
            helicity_ax.fill_between(
                helicity_z,
                tau_quantiles[lower],
                tau_quantiles[upper],
                color=HELICITY_COLOR,
                alpha=alpha,
                linewidth=0.0,
                label=label,
            )
        )
    helicity_ax.plot(
        z_um,
        np.where(visible, wall_tau, np.nan),
        color="#2f6f92",
        linewidth=2.0,
        label="Wall center",
    )
    helicity_ax.plot(
        helicity_z,
        np.where(visible, tau, np.nan),
        color=HELICITY_COLOR,
        linewidth=1.15,
        label="Optical mean",
    )
    helicity_ax.plot(
        helicity_z,
        np.where(visible, tau_fit, np.nan),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Linear fit",
    )
    helicity_ax.set_xlabel(r"$z\;(\mathregular{\mu}\mathrm{m})$", labelpad=1.5)
    helicity_ax.set_ylabel(r"helicity $\tau$ (deg)", labelpad=1.5)
    helicity_ax.set_box_aspect(1.0)
    line_legend = helicity_ax.legend(
        handles=helicity_ax.lines,
        frameon=False,
        fontsize=10.0,
        loc="lower left",
        handlelength=1.5,
        borderaxespad=0.15,
        labelspacing=0.25,
    )
    helicity_ax.add_artist(line_legend)
    helicity_ax.legend(
        handles=quantile_bands,
        frameon=False,
        fontsize=10.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.25,
        borderaxespad=0.15,
        labelspacing=0.18,
        ncol=2,
        columnspacing=0.7,
        title="Weighted quantiles",
        title_fontsize=10.0,
    )

    _style_plot_axis(helicity_ax)
    helicity_ax.spines["top"].set_visible(True)
    helicity_ax.spines["right"].set_visible(True)
    helicity_ax.tick_params(pad=1.5)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    director_position = reserve_ax.get_position()
    director_scale = 1.10
    director_width = director_scale * director_position.width
    director_height = director_scale * director_position.height
    reserve_ax.set_position(
        (
            director_position.x1 - director_width,
            director_position.y1 - director_height,
            director_width,
            director_height,
        )
    )
    axes = (reserve_ax, map_ax, helicity_ax)
    positions = tuple(ax.get_position() for ax in axes)
    panel_left_edges = (
        0.0,
        0.5 * (positions[0].x1 + positions[1].x0),
        0.5 * (positions[1].x1 + positions[2].x0),
    )
    for left, label in zip(panel_left_edges, "abc", strict=True):
        _lower_left_figure_label(
            fig,
            left=left + 0.006,
            bottom=0.018,
            label=label,
        )
    MAIN_DIRECTOR_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        MAIN_DIRECTOR_FIGURE,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.012,
        dpi=FINAL_DPI,
    )
    plt.close(fig)
    return MAIN_DIRECTOR_FIGURE


def plot_supplement_director_views() -> Path:
    _render_supplement_director_base_panels()
    _render_supplement_director_isolines()
    for path in (*DIRECTOR_PANEL_PATHS, DIRECTOR_SPHERE_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    images = [
        np.asarray(Image.open(path).convert("RGBA"))
        for path in DIRECTOR_PANEL_PATHS
    ]
    sphere = np.asarray(Image.open(DIRECTOR_SPHERE_PATH).convert("RGBA"))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    fig.subplots_adjust(
        left=0.012,
        right=0.995,
        bottom=0.015,
        top=0.985,
        wspace=0.008,
    )
    for index, (ax, image) in enumerate(zip(axes, images, strict=True)):
        ax.imshow(image, origin="upper", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _top_panel_label(ax, "abc"[index])
    inset = axes[2].inset_axes((0.705, 0.035, 0.25, 0.25))
    inset.imshow(sphere, origin="upper", interpolation="nearest")
    inset.text(0.88, 0.53, r"$n_x$", transform=inset.transAxes, fontsize=10, ha="center", va="center")
    inset.text(0.53, 0.88, r"$n_y$", transform=inset.transAxes, fontsize=10, ha="center", va="center")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    SUPPLEMENT_DIRECTOR_VIEWS_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        SUPPLEMENT_DIRECTOR_VIEWS_FIGURE,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.012,
    )
    plt.close(fig)
    return SUPPLEMENT_DIRECTOR_VIEWS_FIGURE


def _render_supplement_director_base_panels() -> None:
    """Render S16(a,b) and its orientation sphere from the source director.

    The camera and rendering parameters are fixed for the saved director grid.
    """

    archive = load_director_archive(DIRECTOR_SOURCE)
    field = np.asarray(archive.director, dtype=np.float32)
    if field.shape[:3] != (200, 200, 51):
        raise ValueError(
            "The publication cameras are calibrated for a 200x200x51 "
            f"director; got {field.shape[:3]}"
        )
    system = System.rectangular(tuple(int(value) for value in field.shape[:3]))
    uploaded = np.ascontiguousarray(field[..., np.newaxis, :])

    def camera(*, theta: float, phi: float, perspective: bool):
        return tvmgl.Camera.from_angles(
            theta=theta,
            phi=phi,
            alpha=0.00379143 if perspective else 0.0,
            scale=54.6244,
            perspective=perspective,
            look_at=[99.5, 99.5, 25.0],
        )

    def style(container) -> None:
        scene = container.scene
        scene["Style"]["Show bounds"] = False
        scene["Overlays"]["Frame annotation"] = "Hide"
        scene.set_overlay_outline_mode("Hide")
        scene.axes_render.color_mode = "Black"
        scene.axes_render.location = (-0.72, -0.72)
        scene.axes_render.length = 0.15
        scene.axes_render.font_size_percent = 8.0
        scene.axes_render.label_gap_percent = 1.0

    slice_container = tvmgl.render_prepare("Slice glyphs", system, imgsize=1800)
    slice_container.upload(uploaded)
    style(slice_container)
    slice_scene = slice_container.scene
    slice_scene.show_view_direction_marker = False
    slice_scene.set_measurement_thickness_um(float(archive.thickness_um))
    slice_scene.set_scale_bar_show("Show")
    slice_scene.set_scale_bar_length_mode("Manual")
    slice_scene.set_scale_bar_length(10.0)
    slice_scene.set_scale_bar_location_x(0.80)
    slice_scene.set_scale_bar_location_y(-0.72)
    slice_scene.set_scale_bar_color("Black")
    slice_scene.set_scale_bar_font_size_percent(8.0)
    slice_scene.set_scale_bar_line_width_percent(0.45)
    slice_scene.glyph_render.grid = "Hex"
    slice_scene.glyph_render.spacing_percent = 4.0
    slice_scene.glyph_render.mesh_shape = "Cylinder"
    slice_scene.glyph_render.length_spacing = 0.75
    slice_scene.glyph_render.width_spacing = 0.40
    slice_scene.glyph_render.detalization = 3
    slice_scene.set_camera(
        camera(theta=0.5 * np.pi, phi=0.5 * np.pi, perspective=False)
    )
    DIRECTOR_PANEL_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    slice_container.save(alpha=True).save(DIRECTOR_PANEL_PATHS[0])
    release_render_container(slice_container)

    surface_container = tvmgl.render_prepare(
        "Level surface",
        system,
        imgsize=1800,
    )
    surface_container.upload(uploaded)
    style(surface_container)
    surface_scene = surface_container.scene
    surface_scene.surface_render.angle_name = "polar angle"
    surface_scene.surface_render.level = 0.0
    surface_scene.surface_render.levels_number = 5
    surface_scene.surface_render.cutx = 99.5
    surface_scene.surface_render.cuty = 99.5
    surface_scene.surface_render.cutz = 25.0
    surface_scene.surface_render.cutphi = 0.5
    surface_scene.surface_render.cuttheta = 0.0
    surface_scene.set_camera(
        camera(theta=1.89864, phi=2.06804, perspective=True)
    )
    surface_container.save(alpha=True).save(DIRECTOR_PANEL_PATHS[1])
    release_render_container(surface_container)

    sphere_figure, sphere_axis = tvmpl.plot_bloch_sphere(
        axis=2,
        size=512,
        imagewidth_inch=1.4,
    )
    sphere_axis.set_xticks([])
    sphere_axis.set_yticks([])
    sphere_axis.set_xlabel("")
    sphere_axis.set_ylabel("")
    sphere_axis.set_xlim(-1.28, 1.12)
    sphere_axis.set_ylim(-1.28, 1.12)
    arrow = {
        "arrowstyle": "-|>",
        "mutation_scale": 10,
        "linewidth": 1.4,
        "color": "black",
        "shrinkA": 0,
        "shrinkB": 0,
    }
    sphere_axis.annotate("", xy=(0.58, 0.0), xytext=(0.0, 0.0), arrowprops=arrow)
    sphere_axis.annotate("", xy=(0.0, 0.58), xytext=(0.0, 0.0), arrowprops=arrow)
    for spine in sphere_axis.spines.values():
        spine.set_visible(False)
    sphere_figure.patch.set_alpha(0.0)
    sphere_axis.set_facecolor("none")
    sphere_figure.savefig(
        DIRECTOR_SPHERE_PATH,
        dpi=220,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(sphere_figure)

    del (
        surface_scene,
        surface_container,
        slice_scene,
        slice_container,
        uploaded,
        field,
        archive,
    )
    gc.collect()


def _render_supplement_director_isolines() -> Path:
    """Render the complete orientation isolines used in supplement panel S16(c)."""

    with np.load(HELICITY_ARCHIVE, allow_pickle=False) as helicity:
        weighting_names = tuple(str(value) for value in helicity["weighting_names"])
        optical_index = weighting_names.index("optical")
        tau_rad = np.asarray(
            helicity["layer_tau_rad"][optical_index], dtype=np.float64
        )
        center = np.asarray(helicity["center_px"], dtype=np.float64)
    z_um, _, wall_radius_um, _, wall_tau_rad, _ = _director_geometry(tau_rad)

    archive = load_director_archive(DIRECTOR_SOURCE)
    field = np.asarray(archive.director, dtype=np.float32)
    if field.shape[:3] != (200, 200, 51):
        raise ValueError(
            "The publication camera is calibrated for a 200x200x51 director; "
            f"got {field.shape[:3]}"
        )
    system = System.rectangular(tuple(int(value) for value in field.shape[:3]))
    container = tvmgl.render_prepare("Isolines", system, imgsize=1800)
    overlay_field = np.zeros_like(field)
    overlay_field[..., 2] = 1.0
    container.upload(np.ascontiguousarray(overlay_field[..., np.newaxis, :]))
    scene = container.scene
    scene["Style"]["Show bounds"] = False
    scene["Overlays"]["Frame annotation"] = "Hide"
    scene.set_overlay_outline_mode("Hide")
    scene.axes_render.color_mode = "Black"
    scene.axes_render.location = (-0.72, -0.72)
    scene.axes_render.length = 0.15
    scene.axes_render.font_size_percent = 8.0
    scene.axes_render.label_gap_percent = 1.0
    camera = tvmgl.Camera.from_angles(
        theta=1.89864,
        phi=2.06804,
        alpha=0.00379143,
        scale=54.6244,
        perspective=True,
        look_at=[99.5, 99.5, 25.0],
    )
    scene.set_camera(camera)

    # TopoVec's tetrahedral extractor follows an oriented vector field.  A
    # liquid-crystal director is instead headless, and the vector extractor
    # leaves gaps where an isoline crosses its sign branch.  For this
    # axisymmetric director the same curves have an exact branch-independent
    # parametrization on the n_z=0 wall: phi(z) = alpha - tau_wall(z).
    valid = np.isfinite(wall_radius_um) & np.isfinite(wall_tau_rad)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2 or not np.array_equal(
        valid_indices,
        np.arange(valid_indices[0], valid_indices[-1] + 1),
    ):
        raise ValueError("The n_z=0 director wall is not one contiguous surface")
    spacing_um = float(z_um[-1]) / float(field.shape[2] - 1)
    wall_z = np.asarray(z_um[valid] / spacing_um, dtype=np.float64)
    wall_radius = np.asarray(wall_radius_um[valid] / spacing_um, dtype=np.float64)
    wall_tau = np.asarray(wall_tau_rad[valid], dtype=np.float64)
    wall_z = np.concatenate(([wall_z[0] - 0.5], wall_z, [wall_z[-1] + 0.5]))
    wall_radius = np.concatenate(([0.0], wall_radius, [0.0]))
    wall_tau = np.concatenate(([wall_tau[0]], wall_tau, [wall_tau[-1]]))
    curve_z = np.linspace(wall_z[0], wall_z[-1], 500)
    curve_radius = PchipInterpolator(wall_z, wall_radius)(curve_z)
    curve_tau = PchipInterpolator(wall_z, wall_tau)(curve_z)

    image = Image.fromarray(
        np.asarray(container.save(alpha=True), dtype=np.uint8),
        mode="RGBA",
    )
    width, height = image.size
    modelview = camera.get_view_matrix()
    projection = camera.get_projection_matrix(width / height)
    angles = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    axes = np.stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1
    )
    colors = np.rint(255.0 * np.clip(vector_to_rgb(axes), 0.0, 1.0)).astype(
        np.uint8
    )
    curves = []
    for angle, color in zip(angles, colors, strict=True):
        phi = angle - curve_tau
        positions = np.stack(
            (
                center[0] + curve_radius * np.cos(phi),
                center[1] + curve_radius * np.sin(phi),
                curve_z,
                np.ones_like(curve_z),
            ),
            axis=-1,
        )
        projected = positions @ modelview.T @ projection.T
        pixels = np.stack(
            (
                0.5 * (projected[:, 0] + 1.0) * (width - 1),
                0.5 * (1.0 - projected[:, 1]) * (height - 1),
            ),
            axis=-1,
        )
        curves.append(
            (
                float(np.mean(projected[:, 2])),
                [tuple(point) for point in pixels],
                tuple(int(value) for value in color) + (255,),
            )
        )
    curve_dpi = 300
    curve_figure = plt.figure(
        figsize=(width / curve_dpi, height / curve_dpi),
        dpi=curve_dpi,
    )
    curve_axis = curve_figure.add_axes((0.0, 0.0, 1.0, 1.0))
    curve_axis.set_xlim(0.0, float(width))
    curve_axis.set_ylim(float(height), 0.0)
    curve_axis.set_axis_off()
    curve_figure.patch.set_alpha(0.0)
    curve_axis.patch.set_alpha(0.0)
    line_width_points = 0.01 * min(width, height) * 72.0 / curve_dpi
    for _, pixels, color in sorted(curves, reverse=True):
        pixel_array = np.asarray(pixels, dtype=np.float64)
        curve_axis.plot(
            pixel_array[:, 0],
            pixel_array[:, 1],
            color=np.asarray(color[:3], dtype=np.float64) / 255.0,
            linewidth=line_width_points,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
    curve_figure.canvas.draw()
    line_layer = Image.fromarray(
        np.asarray(curve_figure.canvas.buffer_rgba(), dtype=np.uint8),
        mode="RGBA",
    )
    plt.close(curve_figure)
    image = Image.alpha_composite(image, line_layer)

    output = DIRECTOR_PANEL_PATHS[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    release_render_container(container)
    del scene, container, overlay_field, field, archive
    gc.collect()
    return output


def _director_orientation_section(
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    archive = load_director_archive(DIRECTOR_SOURCE)
    field = np.asarray(archive.director, dtype=np.float64)
    thickness_um = float(archive.thickness_um)
    with np.load(HELICITY_ARCHIVE, allow_pickle=False) as helicity:
        center = np.asarray(helicity["center_px"], dtype=np.float64)
    spacing_um = thickness_um / float(field.shape[2] - 1)
    radius_um = np.arange(
        0.0,
        DIRECTOR_MAP_MAX_RADIUS_UM + 0.5 * DIRECTOR_MAP_RADIAL_STEP_UM,
        DIRECTOR_MAP_RADIAL_STEP_UM,
        dtype=np.float64,
    )
    z_um = np.linspace(0.0, thickness_um, field.shape[2], dtype=np.float64)
    x_coordinate = np.broadcast_to(
        center[0] + radius_um[None, :] / spacing_um,
        (field.shape[2], radius_um.size),
    )
    y_coordinate = np.full_like(x_coordinate, center[1])
    z_coordinate = np.broadcast_to(
        np.arange(field.shape[2], dtype=np.float64)[:, None],
        x_coordinate.shape,
    )
    section = np.empty(x_coordinate.shape + (3,), dtype=np.float64)
    coordinates = (x_coordinate, y_coordinate, z_coordinate)
    for component in range(3):
        section[..., component] = map_coordinates(
            field[..., component],
            coordinates,
            order=1,
            mode="nearest",
        )
    norm = np.linalg.norm(section, axis=-1)
    section = np.divide(
        section,
        norm[..., None],
        out=np.zeros_like(section),
        where=norm[..., None] > 1.0e-12,
    )
    rgb = vector_to_rgb(section.astype(np.float32, copy=False))
    helicity = np.angle(section[..., 0] + 1.0j * section[..., 1])
    polar_tilt = np.arccos(np.clip(section[..., 2], -1.0, 1.0))
    wall_inner, wall_radius, wall_outer = _wall_level_geometry(
        radius_um, section[..., 2]
    )
    del archive, field, section
    gc.collect()
    return radius_um, z_um, rgb, wall_inner, wall_radius, wall_outer


def plot_director_orientation_map() -> Path:
    radius_um, z_um, rgb, inner_edge, wall_radius, outer_edge = (
        _director_orientation_section()
    )
    with plt.rc_context({"font.size": 9.5}):
        fig, ax = plt.subplots(figsize=(4.75, 4.25))
        ax.imshow(
            rgb,
            origin="lower",
            extent=(radius_um[0], radius_um[-1], z_um[0], z_um[-1]),
            interpolation="bilinear",
            aspect="equal",
        )
        ax.plot(inner_edge, z_um, color="white", linewidth=0.70, zorder=6)
        ax.plot(outer_edge, z_um, color="#202020", linewidth=0.55, zorder=6)
        ax.plot(
            wall_radius,
            z_um,
            color="#202020",
            linewidth=1.10,
            label=r"$n_z=0$",
            zorder=8,
        )
        ax.plot(
            [],
            [],
            color="#202020",
            linewidth=0.55,
            label=r"$n_z=\pm1/2$",
        )
        ax.set_xlabel(r"$\rho\;(\mathregular{\mu}\mathrm{m})$")
        ax.set_ylabel(r"$z\;(\mathregular{\mu}\mathrm{m})$")
        ax.set_xlim(radius_um[0], radius_um[-1])
        ax.set_ylim(z_um[0], z_um[-1])
        ax.tick_params(direction="out")
        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=10.0,
            handlelength=2.1,
        )
        inset = ax.inset_axes((0.70, 0.69, 0.275, 0.275))
        _plot_director_color_sphere(inset)
        fig.subplots_adjust(left=0.145, right=0.985, bottom=0.14, top=0.985)
    DIRECTOR_ORIENTATION_MAP_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        DIRECTOR_ORIENTATION_MAP_FIGURE,
        transparent=True,
    )
    plt.close(fig)
    return DIRECTOR_ORIENTATION_MAP_FIGURE


def _select_factorial_controls() -> tuple[list[str], list[dict[str, object]]]:
    if not CONTROL_REPORT.exists():
        raise FileNotFoundError(CONTROL_REPORT)
    report = json.loads(CONTROL_REPORT.read_text(encoding="utf-8"))
    selected = {}
    for record in report["results"]:
        if not np.isclose(
            float(record.get("wavelength_um", np.nan)), CONTROL_WAVELENGTH_UM
        ):
            continue
        profile = record.get("helicity_profile")
        model = record.get("model")
        phase_step = record.get("phase_step")
        if (
            profile == "constant"
            and model == "spectral-jones"
            and np.isclose(float(record.get("tau_mid_deg", np.nan)), -90.0)
        ):
            selected["0"] = record
        elif profile == "measured-layer" and model == "spectral-jones":
            selected["H"] = record
        elif (
            profile == "constant"
            and model == "marching-wave"
            and np.isclose(float(record.get("tau_mid_deg", np.nan)), -90.0)
            and np.isclose(float(phase_step), CONTROL_PHASE_STEP)
        ):
            selected["D"] = record
        elif (
            profile == "measured-layer"
            and model == "marching-wave"
            and np.isclose(float(phase_step), CONTROL_PHASE_STEP)
        ):
            selected["HD"] = record
    missing = [key for key in ("0", "H", "D", "HD") if key not in selected]
    if missing:
        raise ValueError(
            f"Missing saved {CONTROL_WAVELENGTH_NM}-nm factorial controls: "
            f"{', '.join(missing)}"
        )
    keys = ["0", "H", "D", "HD"]
    return keys, [selected[key] for key in keys]


def _factorial_control_profiles(
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    labels, records = _select_factorial_controls()
    channel_power = []
    phase_unlocking = []
    integrated_twofold_modulation = []
    for label, record in zip(labels, records, strict=True):
        archive = load_jones_archive(
            CONTROL_REPORT.parent / Path(str(record["destination"])).name
        )
        jones = np.asarray(archive.jones, dtype=np.complex128)
        synthetic_side = int(record["synthetic_side"])
        synthetic_center = 0.5 * float(synthetic_side)
        center = np.asarray(
            map_center_between_uniform_grids(
                (synthetic_center, synthetic_center),
                source_shape=(synthetic_side, synthetic_side),
                target_shape=tuple(int(value) for value in jones.shape[:2]),
                target_sampling=(
                    "source-nodes"
                    if record["model"] == "spectral-jones"
                    else "cell-centers"
                ),
            ),
            dtype=np.float64,
        )
        pixel_size_um = float(record["pixel_size_um"])
        pixel_area_um2 = pixel_size_um**2
        x, y = np.indices(jones.shape[:2], dtype=np.float64)
        dx = x - center[0]
        dy = y - center[1]
        radius_um = np.hypot(dx, dy) * pixel_size_um
        phi = np.arctan2(dy, dx)
        radial_order = np.argsort(radius_um, axis=None)
        sorted_radius = radius_um.ravel()[radial_order]
        radial_indices = np.searchsorted(
            sorted_radius, CONTROL_RADIAL_CENTERS_UM, side="right"
        ) - 1

        def cumulative_disk_sum(values: np.ndarray) -> np.ndarray:
            cumulative = np.cumsum(
                np.asarray(values, dtype=np.float64).ravel()[radial_order]
            )
            result = np.zeros(CONTROL_RADIAL_CENTERS_UM.shape, dtype=np.float64)
            populated = radial_indices >= 0
            result[populated] = cumulative[radial_indices[populated]]
            return result

        background = radius_um >= float(record["background_inner_um"])
        background_jones = np.mean(jones[background], axis=0)
        perturbation = jones - background_jones
        perturbation_b = 0.5 * (
            perturbation[..., 1, 0] - perturbation[..., 0, 1]
        )
        channel_power.append(
            pixel_area_um2
            * cumulative_disk_sum(2.0 * np.abs(perturbation_b) ** 2)
        )

        cos2 = np.cos(2.0 * phi)
        sin2 = np.sin(2.0 * phi)
        b = 0.5 * (jones[..., 1, 0] - jones[..., 0, 1])
        radial = 0.5 * (jones[..., 0, 0] - jones[..., 1, 1])
        symmetric = 0.5 * (jones[..., 0, 1] + jones[..., 1, 0])
        c = cos2 * radial + sin2 * symmetric
        d = -sin2 * radial + cos2 * symmetric
        unlocking_numerator_cumulative = cumulative_disk_sum(
            np.real(np.conj(b) * c) ** 2
            + np.real(np.conj(b) * d) ** 2
        )
        unlocking_denominator_cumulative = cumulative_disk_sum(
            np.abs(b) ** 2 * (np.abs(c) ** 2 + np.abs(d) ** 2)
        )
        phase_unlocking.append(
            np.sqrt(
                np.divide(
                    unlocking_numerator_cumulative,
                    unlocking_denominator_cumulative,
                    out=np.full(
                        CONTROL_RADIAL_CENTERS_UM.shape,
                        np.nan,
                        dtype=np.float64,
                    ),
                    where=unlocking_denominator_cumulative > 1.0e-12,
                )
            )
        )

        # Use the same canonical soft-annulus extraction of H_0 and H_2 as
        # every other crossed-intensity harmonic result in the article.
        # The cumulative observable is Eq. (integrated-visibility) with the
        # lower limit fixed at the toron axis, not a separately named C_2
        # statistic computed directly from Cartesian pixels.
        harmonic_profiles = crossed_intensity_harmonics(
            jones,
            center_px=tuple(map(float, center)),
            pixel_size_um=pixel_size_um,
            radial_centers_um=CONTROL_RADIAL_CENTERS_UM,
            radial_sigma_um=RADIAL_SIGMA_UM,
            modes=(0, 2),
        )
        cumulative_modulation = np.full(
            CONTROL_RADIAL_CENTERS_UM.shape,
            np.nan,
            dtype=np.float64,
        )
        for radius_index in range(1, CONTROL_RADIAL_CENTERS_UM.size):
            cumulative_modulation[radius_index] = integrated_harmonic_visibility(
                harmonic_profiles,
                mode=2,
                min_radius_um=0.0,
                max_radius_um=float(CONTROL_RADIAL_CENTERS_UM[radius_index]),
            )
        integrated_twofold_modulation.append(cumulative_modulation)
        del harmonic_profiles
        del archive, jones, perturbation
        gc.collect()
    return (
        CONTROL_RADIAL_CENTERS_UM.copy(),
        labels,
        np.asarray(channel_power),
        np.asarray(phase_unlocking),
        np.asarray(integrated_twofold_modulation),
    )


def plot_figure_3() -> Path:
    font_size = 9.25
    (
        radius,
        control_labels,
        channel_power,
        phase_unlocking,
        integrated_twofold_modulation,
    ) = _factorial_control_profiles()
    control_colors = ("#8f8f8f", "#7b3294", "#008837", "#222222")
    control_styles = (":", "-", "--", "-")

    fig = plt.figure(figsize=(7.15, 1.88))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        left=0.060,
        right=0.995,
        bottom=0.220,
        top=0.850,
        wspace=0.31,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    quantities = (channel_power, phase_unlocking, integrated_twofold_modulation)
    ylabels = (
        r"$P_b\;(\mathregular{\mu}\mathrm{m}^2)$",
        r"$\eta_2$",
        r"$\mathcal{V}_2$",
    )
    for axis, values, ylabel in zip(axes, quantities, ylabels, strict=True):
        for label, color, linestyle, profile in zip(
            control_labels,
            control_colors,
            control_styles,
            values,
            strict=True,
        ):
            if label == "0":
                continue
            axis.plot(
                radius,
                profile,
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                label=label,
            )
        axis.set_xlim(INTEGRATION_INTERVAL_UM)
        axis.set_xticks((5, 10, 15, 20, 25))
        axis.set_xlabel(r"$R\;(\mathregular{\mu}\mathrm{m})$", fontsize=font_size, labelpad=2.0)
        axis.set_ylabel(ylabel, fontsize=font_size)
        axis.set_box_aspect(1.0 / GOLDEN_RATIO)
    displayed = radius <= INTEGRATION_INTERVAL_UM[1]
    axes[0].set_ylim(0.0, 1.05 * float(np.nanmax(channel_power[:, displayed])))
    axes[1].set_ylim(-0.03, 1.0)
    modulation_upper = max(1.0, 1.05 * np.nanmax(integrated_twofold_modulation))
    axes[2].set_ylim(-0.03 * modulation_upper, modulation_upper)
    endpoint_index = int(np.argmin(np.abs(radius - 28.0)))
    endpoint = {
        label: {
            "radius_um": float(radius[endpoint_index]),
            "P_b_um2": float(channel_power[index, endpoint_index]),
            "eta_2": (
                None
                if not np.isfinite(phase_unlocking[index, endpoint_index])
                else float(phase_unlocking[index, endpoint_index])
            ),
            "V2bar_0_R": float(
                integrated_twofold_modulation[index, endpoint_index]
            ),
        }
        for index, label in enumerate(control_labels)
    }
    MECHANISM_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MECHANISM_REPORT.write_text(
        json.dumps(
            {
                "schema_version": "mechanism-metrics-v4",
                "wavelength_nm": CONTROL_WAVELENGTH_NM,
                "source": _relative(CONTROL_REPORT),
                "method": (
                    "P_b and eta_2 use pointwise co-rotating a,b,c,d products "
                    "followed by cumulative physical-area disk integrals. "
                    "The observable m=2 modulation uses the article-wide "
                    "soft-annulus H_0/H_2 extraction and the integrated-harmonic "
                    "definition over [0,R]. P_b includes the physical pixel area, "
                    "and eta_2 is reported for every finite-population numerical "
                    "control"
                ),
                "controls": {
                    "0": "constant helicity and local Jones propagation",
                    "H": "varying helicity and local Jones propagation",
                    "D": "constant helicity and marching-wave propagation",
                    "HD": "varying helicity and marching-wave propagation",
                },
                "endpoint": endpoint,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    control_descriptions = {
        "H": "H: helicity variation",
        "D": "D: transverse coupling",
        "HD": "HD: both",
    }
    fig.legend(
        handles,
        [control_descriptions[label] for label in labels],
        frameon=False,
        fontsize=font_size,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.88),
        handlelength=1.5,
        borderaxespad=0.0,
        ncols=3,
        columnspacing=1.2,
    )

    for index, ax in enumerate(axes):
        _style_plot_axis(ax)
        ax.tick_params(labelsize=font_size, pad=2.0)
        _panel_label(ax, chr(ord("a") + index), fontsize=font_size)
    MAIN_MECHANISM_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        MAIN_MECHANISM_FIGURE,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)
    return MAIN_MECHANISM_FIGURE


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("At least two one-dimensional centers are required")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _label_vertical_reference(
    axis: plt.Axes,
    coordinate_um: float,
    label: str,
) -> None:
    axis.annotate(
        label,
        xy=(float(coordinate_um), 0.965),
        xycoords=axis.get_xaxis_transform(),
        xytext=(-2.0, 0.0),
        textcoords="offset points",
        rotation=90,
        rotation_mode="anchor",
        ha="right",
        va="bottom",
        fontsize=10.0,
        color="0.18",
        clip_on=True,
        zorder=5,
    )


def plot_figure_5() -> Path:
    """Render the continuous focus dependence from the saved propagation."""

    with np.load(AXISYMMETRIC_SYMMETRY_ARCHIVE, allow_pickle=False) as archive:
        dataset_keys = tuple(str(value) for value in archive["dataset_keys"])
        dataset_labels = tuple(str(value) for value in archive["dataset_labels"])
        fields = {
            key: {
                "z_um": np.asarray(archive[f"{key}_z_um"], dtype=np.float64),
                "radius_um": np.asarray(
                    archive[f"{key}_radius_um"], dtype=np.float64
                ),
                "c4_breaking": np.asarray(
                    archive[f"{key}_c4_breaking"], dtype=np.float64
                ),
            }
            for key in dataset_keys
        }
    report = json.loads(AXISYMMETRIC_EVOLUTION_REPORT.read_text(encoding="utf-8"))
    surface_offset_um = float(
        report["annular_focus_alignment"][str(WAVELENGTH_NM)][
            "inferred_experimental_surface_z_um"
        ]
    )
    focus_report_keys = {
        "experiment": f"{WAVELENGTH_NM}_experiment",
        "marching_wave_pupil_matched": (
            f"{WAVELENGTH_NM}_marching_wave_pupil_matched"
        ),
        "local_jones_pupil_matched": (
            f"{WAVELENGTH_NM}_local_jones_pupil_matched"
        ),
    }
    focus_labels = {
        "experiment": "annular focus",
        "marching_wave_pupil_matched": "annular focus",
        "local_jones_pupil_matched": "annular focus",
    }

    fig = plt.figure(figsize=(7.15, 2.15))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.035),
        left=0.055,
        right=0.975,
        bottom=0.245,
        top=0.865,
        wspace=0.14,
    )
    axes: list[plt.Axes] = []
    image = None
    for index, (key, label) in enumerate(
        zip(dataset_keys, dataset_labels, strict=True)
    ):
        field = fields[key]
        z_edges = _centers_to_edges(field["z_um"])
        radius_edges = _centers_to_edges(field["radius_um"])
        axis = fig.add_subplot(grid[0, index])
        image = axis.imshow(
            field["c4_breaking"].T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(z_edges[0], z_edges[-1], radius_edges[0], radius_edges[-1]),
            cmap=MOUNTAIN_HAZE_CMAP,
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        focus_z_um = float(
            report["annular_focus"][focus_report_keys[key]]["z_um"]
        )
        if key == "experiment":
            focus_z_um -= surface_offset_um
        axis.axvline(0.0, color="0.28", linewidth=0.75)
        surface_label = "exit surface"
        _label_vertical_reference(axis, 0.0, surface_label)
        axis.axvline(
            focus_z_um,
            color="0.28",
            linewidth=0.75,
            linestyle="-.",
        )
        _label_vertical_reference(axis, focus_z_um, focus_labels[key])
        if key == "experiment":
            axis.axvline(
                -surface_offset_um,
                color="0.28",
                linewidth=0.7,
                linestyle="--",
            )
            _label_vertical_reference(
                axis,
                -surface_offset_um,
                "holographic plane",
            )
        axis.set_title(label, pad=2.0)
        axis.set_xlabel(r"$z\;(\mathregular{\mu}\mathrm{m})$", labelpad=1.5)
        axis.set_xlim(-40.0, 40.0)
        axis.set_ylim(0.0, 40.0)
        axis.tick_params(labelsize=10.0, length=2.5, pad=1.5)
        if index == 0:
            axis.set_ylabel(r"$\rho\;(\mathregular{\mu}\mathrm{m})$", labelpad=2.0)
        else:
            axis.tick_params(labelleft=False)
        axes.append(axis)

    if image is None:
        raise RuntimeError("No focus-symmetry fields were loaded")
    color_axis = fig.add_subplot(grid[0, 3])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_ticks((0.0, 0.5, 1.0))
    colorbar.set_label(r"$B_{4\to2}$", labelpad=1.5)
    colorbar.ax.tick_params(labelsize=10.0, length=2.5, pad=1.5)

    fig.canvas.draw()
    for index, axis in enumerate(axes):
        _horizontal_axis_panel_label(fig, axis, chr(ord("a") + index))

    MAIN_FOCUS_SYMMETRY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        MAIN_FOCUS_SYMMETRY_FIGURE,
        dpi=FINAL_DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)
    return MAIN_FOCUS_SYMMETRY_FIGURE


def plot_supplement_crossover() -> Path:
    analysis = _load_analysis()
    radius = np.asarray(analysis["radial_centers_um"], dtype=np.float64)
    visibility = np.asarray(analysis["model_visibility"], dtype=np.float64)
    labels = _analysis_display_labels(analysis["model_keys"])
    colors = tuple(str(value) for value in analysis["model_colors"])
    fig, ax = plt.subplots(figsize=(4.6, 2.65))
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        ax.plot(
            radius,
            visibility[index, 1],
            color=color,
            linewidth=1.3,
            label=label.replace("Measured field, covariant projection", "Measured, covariant projection"),
        )
        ax.plot(
            radius,
            visibility[index, 2],
            color=color,
            linewidth=1.05,
            linestyle=(0, (3.0, 2.0)),
        )
    ax.set_xlim(1.0, 30.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$\rho\;(\mathregular{\mu}\mathrm{m})$")
    ax.set_ylabel(r"$v_m(\rho)$")
    dataset_legend = ax.legend(
        frameon=False,
        fontsize=9.0,
        loc="upper right",
        handlelength=1.5,
    )
    ax.add_artist(dataset_legend)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="black", linewidth=1.2, label=r"$m=2$"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.0,
                linestyle=(0, (3.0, 2.0)),
                label=r"$m=4$",
            ),
        ],
        frameon=False,
        fontsize=9.0,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.64),
        ncols=2,
        columnspacing=0.8,
        handlelength=1.5,
    )
    _style_plot_axis(ax)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.19, top=0.985)
    SUPPLEMENT_CROSSOVER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        SUPPLEMENT_CROSSOVER_FIGURE,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)
    return SUPPLEMENT_CROSSOVER_FIGURE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="recompute the compact 520-nm harmonic archive",
    )
    parser.add_argument(
        "--figure",
        choices=(
            "1",
            "2",
            "3",
            "5",
            "director-map",
            "director-views",
            "supplement",
            "all",
        ),
        default="all",
        help="figure to render after loading the saved analysis",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="stop after --analyze without rendering figures",
    )
    add_force_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    requested = (
        ("1", "2", "3", "5", "director-map", "director-views", "supplement")
        if args.figure == "all"
        else (args.figure,)
    )
    figure_outputs = {
        "1": (MAIN_OPTICAL_FIGURE,),
        "2": (MAIN_DIRECTOR_FIGURE,),
        "3": (MAIN_MECHANISM_FIGURE, MECHANISM_REPORT),
        "5": (MAIN_FOCUS_SYMMETRY_FIGURE,),
        "director-map": (DIRECTOR_ORIENTATION_MAP_FIGURE,),
        "director-views": (SUPPLEMENT_DIRECTOR_VIEWS_FIGURE,),
        "supplement": (SUPPLEMENT_CROSSOVER_FIGURE,),
    }
    outputs = []
    if args.analyze:
        outputs.extend((ANALYSIS_ARCHIVE, ANALYSIS_REPORT))
    if not args.analysis_only:
        outputs.extend(
            path
            for key in requested
            for path in figure_outputs[key]
        )
    refuse_existing_outputs(
        outputs,
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    if args.analyze:
        print(f"[letter-figures] analysis: {analyze()}", flush=True)
    if args.analysis_only:
        return 0
    functions = {
        "1": plot_figure_1,
        "2": plot_figure_2,
        "3": plot_figure_3,
        "5": plot_figure_5,
        "director-map": plot_director_orientation_map,
        "director-views": plot_supplement_director_views,
        "supplement": plot_supplement_crossover,
    }
    for key in requested:
        print(f"[letter-figures] figure {key}: {functions[key]()}", flush=True)
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
