#!/usr/bin/env python3
"""Map visible C4-to-C2 reduction versus radius and coherent aperture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from toronoptics.article_data import (
    ARTICLE_VARIANT_SUFFIX,
    effective_coherent_na,
    optical_parameters,
    simulated_array_center,
    simulated_jones_path,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.axisymmetric_propagation import (
    circular_from_abcd,
    derived_radial_observables,
    extract_axisymmetric_profile,
    propagate_axisymmetric_circular_hankel,
    reflection_axis_observables,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.figure_output import figure_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = figure_output_dir(PROJECT_ROOT)
WAVELENGTH_NM = 520
DATASET_KEYS = ("experiment", "marching_wave", "local_jones")
DATASET_LABELS = {
    "experiment": "Experiment",
    "marching_wave": "Marching waves",
    "local_jones": "Local Jones",
}
BACKGROUND_INTERVAL_UM = (41.0, 45.0)
ANGULAR_SAMPLES = 720
EXTENSION_RADIUS_UM = 96.0
DISPLAY_MAX_RADIUS_UM = 40.0
MINIMUM_NA = 0.05
NA_STEP = 0.005
MEDIUM_INDEX = 1.0
SPECTRAL_OVERSAMPLING = 2.5
FINAL_DPI = 600

DEFAULT_EVOLUTION_DIR = (
    PROJECT_ROOT / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}"
)
DEFAULT_ARCHIVE = DEFAULT_EVOLUTION_DIR / "axisymmetric_c4_aperture_520.npz"
DEFAULT_REPORT = DEFAULT_EVOLUTION_DIR / "c4_aperture_report.json"
DEFAULT_FIGURE = (
    FIGURE_DIR
    / f"supplement_c4_breaking_aperture_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Coordinate centers must contain at least two values")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _load_archived_sources(
    coefficient_archive: Path,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    with np.load(coefficient_archive, allow_pickle=False) as archive:
        wavelengths = np.asarray(archive["wavelength_nm"], dtype=np.int64)
        dataset_keys = tuple(str(value) for value in archive["dataset_keys"])
        wavelength_matches = np.flatnonzero(wavelengths == WAVELENGTH_NM)
        if wavelength_matches.size != 1:
            raise ValueError(f"Expected one {WAVELENGTH_NM}-nm coefficient block")
        wavelength_index = int(wavelength_matches[0])
        radius_um = np.asarray(archive["radius_um"], dtype=np.float64)
        sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for output_key, archive_key in (
            ("experiment", "experiment"),
            ("marching_wave", "marching_wave"),
        ):
            dataset_index = dataset_keys.index(archive_key)
            sources[output_key] = (
                np.asarray(
                    archive["circular"][wavelength_index, dataset_index],
                    dtype=np.complex128,
                ),
                np.asarray(
                    archive["background_abcd"][wavelength_index, dataset_index],
                    dtype=np.complex128,
                ),
            )
    return radius_um, sources


def _extract_local_source(
    radius_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Path]:
    source_path = simulated_jones_path(
        PROJECT_ROOT, WAVELENGTH_NM, "spectral-jones"
    )
    loaded = load_jones_archive(source_path, dtype=np.complex64)
    profile = extract_axisymmetric_profile(
        loaded.jones,
        center_px=simulated_array_center(source_path),
        pixel_size_um=float(loaded.pixel_size_um),
        radius_um=radius_um,
        angular_samples=ANGULAR_SAMPLES,
        background_interval_um=BACKGROUND_INTERVAL_UM,
    )
    return profile.circular, profile.background_abcd, source_path


def _aperture_grid(maximum_na: float) -> np.ndarray:
    values = np.arange(MINIMUM_NA, maximum_na, NA_STEP, dtype=np.float64)
    values = values[values < maximum_na - 0.25 * NA_STEP]
    return np.concatenate((values, np.asarray((maximum_na,), dtype=np.float64)))


def _compute(
    coefficient_archive: Path,
    focus_report: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    radius_um, sources = _load_archived_sources(coefficient_archive)
    local_circular, local_background, local_path = _extract_local_source(radius_um)
    sources["local_jones"] = (local_circular, local_background)

    focus = json.loads(focus_report.read_text(encoding="utf-8"))
    experimental_surface_distance_um = float(
        focus["annular_focus_alignment"][str(WAVELENGTH_NM)][
            "inferred_experimental_surface_z_um"
        ]
    )
    maximum_na = float(effective_coherent_na(PROJECT_ROOT, WAVELENGTH_NM))
    numerical_aperture = _aperture_grid(maximum_na)
    wavelength_um = float(optical_parameters(WAVELENGTH_NM).wavelength_um)

    h2_by_dataset: dict[str, np.ndarray] = {}
    h4_by_dataset: dict[str, np.ndarray] = {}
    mean_by_dataset: dict[str, np.ndarray] = {}
    propagated_radius_um: np.ndarray | None = None
    for key in DATASET_KEYS:
        circular, background_abcd = sources[key]
        background_circular = circular_from_abcd(background_abcd)
        distance_um = (
            experimental_surface_distance_um if key == "experiment" else 0.0
        )
        h2_values = []
        h4_values = []
        mean_values = []
        for aperture in numerical_aperture:
            propagated = propagate_axisymmetric_circular_hankel(
                circular,
                radius_um,
                np.asarray((distance_um,), dtype=np.float64),
                wavelength_um=wavelength_um,
                medium_index=MEDIUM_INDEX,
                numerical_aperture=float(aperture),
                background_circular=background_circular,
                extension_radius_um=EXTENSION_RADIUS_UM,
                spectral_oversampling=SPECTRAL_OVERSAMPLING,
            )
            if propagated_radius_um is None:
                propagated_radius_um = propagated.radius_um
            elif not np.array_equal(propagated_radius_um, propagated.radius_um):
                raise RuntimeError("Aperture sweep returned inconsistent radial grids")
            observables = derived_radial_observables(
                propagated.abcd[0],
                background_abcd=background_abcd,
            )
            h2_values.append(observables["crossed_H2"])
            h4_values.append(observables["crossed_H4"])
            mean_values.append(observables["crossed_mean_intensity"])
        h2_by_dataset[key] = np.asarray(h2_values, dtype=np.complex128)
        h4_by_dataset[key] = np.asarray(h4_values, dtype=np.complex128)
        mean_by_dataset[key] = np.asarray(mean_values, dtype=np.float64)

    if propagated_radius_um is None:
        raise RuntimeError("No aperture fields were produced")
    selected = propagated_radius_um <= DISPLAY_MAX_RADIUS_UM
    display_radius_um = propagated_radius_um[selected]
    all_mean = np.concatenate(
        [mean_by_dataset[key][:, selected].ravel() for key in DATASET_KEYS]
    )
    intensity_floor = 0.01 * float(np.percentile(all_mean, 99.0))

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2, dtype=np.int16),
        "wavelength_nm": np.asarray(WAVELENGTH_NM, dtype=np.int16),
        "wavelength_um": np.asarray(wavelength_um, dtype=np.float64),
        "dataset_keys": np.asarray(DATASET_KEYS),
        "dataset_labels": np.asarray([DATASET_LABELS[key] for key in DATASET_KEYS]),
        "numerical_aperture": numerical_aperture,
        "maximum_effective_na": np.asarray(maximum_na, dtype=np.float64),
        "radius_um": display_radius_um,
        "intensity_floor": np.asarray(intensity_floor, dtype=np.float64),
        "experimental_surface_distance_um": np.asarray(
            experimental_surface_distance_um, dtype=np.float64
        ),
    }
    for key in DATASET_KEYS:
        h2 = h2_by_dataset[key][:, selected]
        h4 = h4_by_dataset[key][:, selected]
        mean_intensity = mean_by_dataset[key][:, selected]
        reflection = reflection_axis_observables(h2, h4)
        visibility = np.clip(
            np.sqrt(2.0 * reflection["angular_power"])
            / (mean_intensity + intensity_floor),
            0.0,
            1.0,
        )
        c4_breaking = np.clip(
            visibility * (1.0 - reflection["fourfold_quality"]),
            0.0,
            1.0,
        )
        arrays[f"{key}_H2"] = h2.astype(np.complex64)
        arrays[f"{key}_H4"] = h4.astype(np.complex64)
        arrays[f"{key}_mean_intensity"] = mean_intensity.astype(np.float32)
        arrays[f"{key}_angular_visibility"] = visibility.astype(np.float32)
        arrays[f"{key}_fourfold_quality"] = reflection[
            "fourfold_quality"
        ].astype(np.float32)
        arrays[f"{key}_c4_breaking"] = c4_breaking.astype(np.float32)

    comparison_mask = (
        (display_radius_um >= 1.0) & (display_radius_um <= 28.0)
    )
    reference_index = int(np.argmin(np.abs(numerical_aperture - 0.40)))
    aperture_convergence = {}
    for key in DATASET_KEYS:
        values = np.asarray(arrays[f"{key}_c4_breaking"], dtype=np.float64)
        aperture_convergence[key] = {
            "reference_na": float(numerical_aperture[reference_index]),
            "rms_to_maximum_na": float(
                np.sqrt(
                    np.mean(
                        (
                            values[reference_index, comparison_mask]
                            - values[-1, comparison_mask]
                        )
                        ** 2
                    )
                )
            ),
        }

    report = {
        "schema_version": "c4-aperture-sweep-v2",
        "wavelength_nm": WAVELENGTH_NM,
        "wavelength_um": wavelength_um,
        "aperture_interval": [float(numerical_aperture[0]), maximum_na],
        "aperture_samples": int(numerical_aperture.size),
        "maximum_effective_na": maximum_na,
        "effective_na_source": (
            "sharp wavelength-scaled edge of the transverse power spectrum "
            "of the nine holographic Jones reconstructions; not a separately "
            "measured diaphragm setting"
        ),
        "experimental_surface_distance_um": experimental_surface_distance_um,
        "fixed_comparison_plane": "MW-referenced liquid-crystal exit surface z=0",
        "pupil": "centered polarization-independent hard circular coherent pupil",
        "experimental_limitation": (
            "the reconstructed field already contains the acquisition and "
            "reconstruction transfer cutoff; the sweep applies only nested "
            "pupils with NA <= maximum_effective_na"
        ),
        "visibility": (
            "min(1,sqrt(2*(abs(H2)^2+abs(H4)^2))/(I0_crossed+epsilon))"
        ),
        "visible_c4_breaking": (
            "visibility*abs(H2)^2/(abs(H2)^2+abs(H4)^2)"
        ),
        "aperture_convergence_over_1_to_28_um": aperture_convergence,
        "intensity_floor": intensity_floor,
        "coefficient_archive": str(coefficient_archive.relative_to(PROJECT_ROOT)),
        "focus_report": str(focus_report.relative_to(PROJECT_ROOT)),
        "local_jones_source": str(local_path.relative_to(PROJECT_ROOT)),
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return arrays, report


def _plot(archive_path: Path, figure_path: Path) -> None:
    with np.load(archive_path, allow_pickle=False) as archive:
        keys = tuple(str(value) for value in archive["dataset_keys"])
        labels = tuple(str(value) for value in archive["dataset_labels"])
        numerical_aperture = np.asarray(
            archive["numerical_aperture"], dtype=np.float64
        )
        radius_um = np.asarray(archive["radius_um"], dtype=np.float64)
        c4_breaking = {
            key: np.asarray(archive[f"{key}_c4_breaking"], dtype=np.float64)
            for key in keys
        }

    figure = plt.figure(figsize=(7.9, 2.5), layout="compressed")
    grid = figure.add_gridspec(
        1,
        len(keys) + 1,
        width_ratios=(*([1.0] * len(keys)), 0.045),
        wspace=0.04,
    )
    na_edges = _centers_to_edges(numerical_aperture)
    radius_edges = _centers_to_edges(radius_um)
    for column, (key, label) in enumerate(zip(keys, labels, strict=True)):
        axis = figure.add_subplot(grid[0, column])
        axis.pcolormesh(
            na_edges,
            radius_edges,
            c4_breaking[key].T,
            shading="flat",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        axis.set_title(label)
        axis.set_xlabel(r"numerical aperture, $\mathrm{NA}$")
        axis.set_xlim(na_edges[0], na_edges[-1])
        axis.set_ylim(radius_edges[0], radius_edges[-1])
        if column == 0:
            axis.set_ylabel(r"$\rho$ ($\mu$m)")
        else:
            axis.set_yticklabels([])

    color_axis = figure.add_subplot(grid[0, -1])
    colorbar = figure.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=1.0),
            cmap=plt.get_cmap("Greys"),
        ),
        cax=color_axis,
    )
    colorbar.set_ticks((0.0, 1.0))
    colorbar.set_label(r"$B_{4\to2}$")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=FINAL_DPI)
    figure.savefig(figure_path.with_suffix(".png"), dpi=FINAL_DPI)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-archive",
        type=Path,
        default=DEFAULT_EVOLUTION_DIR / "axisymmetric_jones_coefficients.npz",
    )
    parser.add_argument(
        "--focus-report",
        type=Path,
        default=DEFAULT_EVOLUTION_DIR / "report.json",
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--mode",
        choices=("compute", "plot", "all"),
        default="all",
    )
    add_force_argument(parser)
    args = parser.parse_args()

    outputs = []
    if args.mode in {"compute", "all"}:
        outputs.extend((args.archive, args.report))
    if args.mode in {"plot", "all"}:
        outputs.extend((args.figure, args.figure.with_suffix(".png")))
    refuse_existing_outputs(
        outputs,
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    if args.mode in {"compute", "all"}:
        arrays, report = _compute(args.coefficient_archive, args.focus_report)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.archive, **arrays)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.mode in {"plot", "all"}:
        _plot(args.archive, args.figure)

    print(args.archive)
    print(args.report)
    print(args.figure)


if __name__ == "__main__":
    main()
