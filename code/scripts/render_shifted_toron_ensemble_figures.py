#!/usr/bin/env python3
"""Compose the three repeatability figures from persisted TopoVec panel caches."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import topovec.mgl as tvmgl

from toronoptics.experimental_jones import load_jones_archive
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.figure_output import figure_output_dir, release_render_container
from toronoptics.jones_ensemble_average import extract_centered_jones_physical


ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = ROOT / "data/analysis/results_toron25um_50X_averaging/topovec_panels"
FIGS = figure_output_dir(ROOT)
AVERAGE_DIR = ROOT / "data/experimental/results_toron25um_50X_averaged"
VIEW_LABELS = (r"$I_{x\rightarrow y}$", r"$S_0$", r"$\arg E_x$", r"$I_{+\rightarrow-}$")
VIEW_SPECS = (
    ("Linear X", "Linear 90"),
    ("Linear X", "Stokes S0"),
    ("Linear X", "Field phase Ex"),
    ("Circular +chi", "Circular -chi"),
)


def _panel(wavelength_nm: int, row: str, view: int) -> np.ndarray:
    path = PANEL_DIR / str(wavelength_nm) / f"{row}_{view}.png"
    return np.asarray(Image.open(path).convert("RGBA"))


def _render_panel(
    field: np.ndarray,
    *,
    pixel_size_um: float,
    wavelength_nm: int,
    incident: str,
    analyzer: str,
    overlays: bool,
    show_scale_bar: bool,
) -> np.ndarray:
    container = tvmgl.render_precomputed_optics(
        {"jones_matrices": np.ascontiguousarray(field)},
        size=(440, 440),
        scene_state={
            "phase_reference_mode": "Solver native",
            "wavelength_um": wavelength_nm / 1000.0,
            "pixel_size_um": pixel_size_um,
        },
    )
    scene = container.scene
    scene.set_polarization_preset(incident)
    scene.set_analyzer_preset(analyzer)
    scene.set_overlay_outline_mode("Hide")
    scene.set_panel_label_show("Hide")
    scene.set_panel_gap_percent(0.0)
    scene.set_panel_outside_background("Transparent")
    scene.set_color_bar_show("Hide")
    scene.set_tone_curve("sRGB")
    scene.set_exposure_ev(0.0)
    scene.show_axes = overlays
    scene.axes_render.color_mode = "Contrast"
    scene.axes_render.location = (-0.76, -0.75)
    scene.axes_render.length = 0.13
    scene.axes_render.font_size_percent = 20.0
    scene.axes_render.label_gap_percent = 1.0
    scene.set_scale_bar_show("Show" if show_scale_bar else "Hide")
    scene.set_scale_bar_length_mode("Manual")
    scene.set_scale_bar_length(10.0)
    scene.set_scale_bar_location_x(0.67)
    scene.set_scale_bar_location_y(-0.79)
    scene.set_scale_bar_color("Contrast")
    scene.scale_bar_font_size_percent = 20.0
    scene.set_scale_bar_line_width_percent(0.9)
    scene.set_polarization_incident_show("Auto" if overlays else "Hide")
    scene.set_polarization_analyzer_show("Auto" if overlays else "Hide")
    scene.set_polarization_direction_show("Hide")
    scene.set_polarization_location_x(0.70)
    scene.set_polarization_location_y(0.70)
    scene.set_polarization_size(0.13)
    scene.set_polarization_line_width(0.004)
    scene.set_polarization_color("Contrast")
    result = np.asarray(container.render_array_rgba(), dtype=np.uint8)
    release_render_container(container)
    del scene, container
    gc.collect()
    return result


def _render_row(field: np.ndarray, pixel_size_um: float, wavelength_nm: int, stem: str, overlays: bool) -> None:
    directory = PANEL_DIR / str(wavelength_nm)
    directory.mkdir(parents=True, exist_ok=True)
    for view, (incident, analyzer) in enumerate(VIEW_SPECS, start=1):
        Image.fromarray(
            _render_panel(
                field,
                pixel_size_um=pixel_size_um,
                wavelength_nm=wavelength_nm,
                incident=incident,
                analyzer=analyzer,
                overlays=overlays and view == 1,
                show_scale_bar=view == 1,
            )
        ).save(directory / f"{stem}_{view}.png")


def _rerender_panels(wavelength_nm: int) -> None:
    path = AVERAGE_DIR / f"{wavelength_nm}nm_average.npz"
    with np.load(path, allow_pickle=False) as data:
        source_paths = tuple(Path(str(value)) for value in data["source_paths"])
        source_centers = np.asarray(data["source_centers_px"], dtype=np.float64)
    if len(source_paths) != 9 or source_centers.shape != (9, 2):
        raise ValueError(
            f"Expected nine repeat acquisitions in {path}, got "
            f"{len(source_paths)} paths and centers {source_centers.shape}"
        )
    for index, (source_path, center) in enumerate(
        zip(source_paths, source_centers, strict=True),
        start=1,
    ):
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        archive = load_jones_archive(source_path, dtype=np.complex64)
        pixel_size_um = float(archive.pixel_size_um)
        side = round(90.0 / pixel_size_um)
        field, valid = extract_centered_jones_physical(
            archive.jones,
            center=tuple(float(value) for value in center),
            source_pixel_size_um=pixel_size_um,
            output_pixel_size_um=pixel_size_um,
            crop_size_px=side,
        )
        if not np.all(valid):
            raise ValueError(
                f"The repeatability crop extends beyond source field {source_path}"
            )
        _render_row(
            field,
            pixel_size_um,
            wavelength_nm,
            f"source_{index}",
            index == 1,
        )

    archive = load_jones_archive(path, dtype=np.complex64)
    with np.load(path, allow_pickle=False) as data:
        center = tuple(float(value) for value in data["array_center_px"])
    side = round(90.0 / float(archive.pixel_size_um))
    field, valid = extract_centered_jones_physical(
        archive.jones,
        center=center,
        source_pixel_size_um=float(archive.pixel_size_um),
        output_pixel_size_um=float(archive.pixel_size_um),
        crop_size_px=side,
    )
    if not np.all(valid):
        raise ValueError("The average repeatability crop is incomplete")
    _render_row(field, float(archive.pixel_size_um), wavelength_nm, "avg", False)


def _panel_cache_complete(wavelength_nm: int) -> bool:
    directory = PANEL_DIR / str(wavelength_nm)
    stems = tuple(f"source_{index}" for index in range(1, 10)) + ("avg",)
    return all(
        (directory / f"{stem}_{view}.png").is_file()
        for stem in stems
        for view in range(1, 5)
    )


def render(wavelength_nm: int, *, rerender_panels: bool = False) -> Path:
    if rerender_panels or not _panel_cache_complete(wavelength_nm):
        _rerender_panels(wavelength_nm)
    rows = [(str(index), f"source_{index}") for index in range(1, 10)] + [("avg", "avg")]
    layout = ((0, 5), (1, 6), (2, 7), (3, 8), (4, 9))
    fig, axes = plt.subplots(5, 8, figsize=(7.15, 4.72))
    fig.subplots_adjust(left=0.028, right=0.998, bottom=0.018, top=0.935, wspace=0.018, hspace=0.018)
    for grid_row, pair in enumerate(layout):
        for block, row_index in enumerate(pair):
            label, stem = rows[row_index]
            for view in range(1, 5):
                axis = axes[grid_row, block * 4 + view - 1]
                axis.imshow(_panel(wavelength_nm, stem, view), interpolation="nearest")
                axis.set_xticks(())
                axis.set_yticks(())
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if view == 1:
                    axis.text(0.015, 0.97, label, transform=axis.transAxes, ha="left", va="top", fontsize=10.0, color="white")
    for block in range(2):
        for view, label in enumerate(VIEW_LABELS):
            axes[0, block * 4 + view].set_title(label, fontsize=10.0, pad=2.0)
    FIGS.mkdir(parents=True, exist_ok=True)
    output = FIGS / f"shifted_toron_ensemble_{wavelength_nm}.pdf"
    fig.savefig(output, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerender-panels", action="store_true")
    add_force_argument(parser)
    args = parser.parse_args(argv)
    outputs = tuple(
        FIGS / f"shifted_toron_ensemble_{wavelength_nm}.pdf"
        for wavelength_nm in (450, 520, 660)
    )
    refuse_existing_outputs(
        outputs,
        force=args.force,
        project_root=ROOT,
    )
    for wavelength_nm in (450, 520, 660):
        print(
            render(
                wavelength_nm,
                rerender_panels=args.force or args.rerender_panels,
            )
        )


if __name__ == "__main__":
    main()
