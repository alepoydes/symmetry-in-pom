#!/usr/bin/env python3
"""Build the publication optical-comparison figures from saved Jones fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from topovec.core import (
    DEFAULT_DIVERGING_COLORMAP,
    DEFAULT_PHASE_COLORMAP,
    System,
    apply_diverging_colormap,
    apply_phase_colormap,
)
from topovec.mgl.image import ImageContainer
from topovec.mgl.precomputed_optics import PrecomputedOpticsScene
import topovec.mgl as tvmgl
from topovec.optics import apply_coherent_pupil

from toronoptics.article_data import (
    ARTICLE_COMPARISON_FIELD_WIDTH_UM,
    TILTED_CONTROL_WAVELENGTH_NM,
    effective_coherent_na,
    experimental_array_center,
    experimental_average_path,
    optical_parameters,
    simulated_array_center,
    simulated_jones_path,
    tilted_simulated_jones_path,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.jones_ensemble_average import extract_centered_jones_physical
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.figure_output import figure_output_dir, release_render_container


ROOT = Path(__file__).resolve().parents[2]
FIGS = figure_output_dir(ROOT)
FIGURE_DPI = 400
SOURCE_PANEL_PIXELS = 1200
PANEL_PIXELS = 1000
SCALE_BAR_UM = 20.0
GRID_LAYOUT_SCALE = 0.5
S0_LAYOUT_SCALE = 0.75
EXPERIMENTAL_MODE_ANALYSIS = (
    ROOT
    / "data/analysis/experimental_angular_modes/experimental_angular_modes.npz"
)
EXPERIMENTAL_MODE_FIGURE = "supplement_experimental_angular_modes.pdf"
TILTED_MODE_ANALYSIS = (
    ROOT
    / "data/analysis/tilted_angular_modes_520_soft/tilted_angular_modes.npz"
)
TILTED_LINEAR_FIGURE = "supplement_tilted_linear_phase_520_soft.pdf"
TILTED_MODE_FIGURE = "supplement_tilted_angular_modes_520_soft.pdf"

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "savefig.dpi": FIGURE_DPI,
    }
)


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    jones: np.ndarray
    pixel_size_um: float
    wavelength_um: float
    center_px: tuple[float, float] | None
    display_width_um: float | None


def _load_dataset(wavelength_nm: int, key: str, label: str) -> Dataset:
    if key == "exp":
        path = experimental_average_path(ROOT, wavelength_nm)
        center = experimental_array_center(path)
    else:
        model = "marching-wave" if key == "marching" else "spectral-jones"
        path = simulated_jones_path(ROOT, wavelength_nm, model)
        center = simulated_array_center(path)
    archive = load_jones_archive(path, dtype=np.complex64)
    jones = archive.jones
    wavelength_um = optical_parameters(wavelength_nm).wavelength_um
    if key != "exp":
        jones = apply_coherent_pupil(
            jones,
            pixel_size_um=float(archive.pixel_size_um),
            wavelength_um=wavelength_um,
            numerical_aperture=effective_coherent_na(ROOT, wavelength_nm),
        )
    return Dataset(
        key,
        label,
        np.ascontiguousarray(jones),
        float(archive.pixel_size_um),
        wavelength_um,
        center,
        ARTICLE_COMPARISON_FIELD_WIDTH_UM,
    )


def _load_triplet(wavelength_nm: int) -> tuple[Dataset, ...]:
    return (
        _load_dataset(wavelength_nm, "exp", "Experiment"),
        _load_dataset(wavelength_nm, "marching", "Marching waves"),
        _load_dataset(wavelength_nm, "jones", "Local Jones"),
    )


def _display_jones(dataset: Dataset) -> np.ndarray:
    if dataset.display_width_um is None:
        return dataset.jones
    if dataset.center_px is None:
        raise ValueError(f"No center for {dataset.label}")
    crop_size_px = max(
        2,
        int(
            np.floor(
                float(dataset.display_width_um) / float(dataset.pixel_size_um)
                + 0.5
            )
        ),
    )
    cropped, valid = extract_centered_jones_physical(
        dataset.jones,
        center=dataset.center_px,
        source_pixel_size_um=dataset.pixel_size_um,
        output_pixel_size_um=dataset.pixel_size_um,
        crop_size_px=crop_size_px,
    )
    if not np.all(valid) or not np.all(np.isfinite(cropped)):
        raise ValueError(
            f"The requested {dataset.display_width_um:g}-um display crop "
            f"around {dataset.center_px!r} does not fit inside "
            f"{dataset.label} with shape {dataset.jones.shape[:2]}"
        )
    return np.ascontiguousarray(cropped, dtype=dataset.jones.dtype)


def _optics_panels(
    jones: np.ndarray,
    *,
    analyzer: str,
    input_component: int | None,
    wavelength_um: float,
    pixel_size_um: float,
) -> list[np.ndarray]:
    counts = {
        "Linear 0/90 panels": 2,
        "Field phase": 3,
        "Stokes parameters": 4,
        "Jones circular intensity panels": 4,
        "Jones circular phase panels": 4,
    }
    panel_count = counts[analyzer]
    container = tvmgl.render_precomputed_optics(
        {"jones_matrices": np.ascontiguousarray(jones)},
        size=(SOURCE_PANEL_PIXELS * panel_count, SOURCE_PANEL_PIXELS),
        scene_state={
            "phase_reference_mode": "Solver native",
            "wavelength_um": float(wavelength_um),
            "pixel_size_um": float(pixel_size_um),
        },
    )
    scene = container.scene
    scene.set_overlay_outline_mode("Hide")
    if input_component is not None:
        scene.set_polarization_preset(("Linear X", "Linear Y")[input_component])
    scene.set_analyzer_preset(analyzer)
    scene.set_color_bar_show("Hide")
    scene.set_panel_rows(1)
    scene.set_panel_gap_percent(0.0)
    scene.set_panel_label_show("Hide")
    scene.set_panel_outside_background("Transparent")
    scene.show_axes = False
    scene.set_scale_bar_show("Hide")
    rendered = np.asarray(container.render_array(), dtype=np.float32) / 255.0
    release_render_container(container)
    del scene, container
    gc.collect()
    return [np.ascontiguousarray(panel) for panel in np.split(rendered, panel_count, axis=1)]


def _decorate_panel(
    panel: np.ndarray,
    *,
    physical_width_um: float,
    show_axes: bool,
    show_scale_bar: bool,
) -> np.ndarray:
    height, width = panel.shape[:2]
    system = System.rectangular((width, height))
    container = ImageContainer(
        size=(PANEL_PIXELS, PANEL_PIXELS),
        scene=lambda ctx: PrecomputedOpticsScene(
            ctx,
            system=system,
            state={"phase_reference_mode": "Solver native"},
        ),
    )
    scene = container.scene
    container.upload_inputs(
        {"optics": {"rgb_panels": panel[None, ...], "labels": ("",)}}
    )
    scene.set_pixel_size_um(float(physical_width_um) / width)
    scene.set_overlay_outline_mode("Hide")
    scene.set_panel_rows(1)
    scene.set_panel_gap_percent(0.8)
    scene.set_panel_outside_background("Transparent")
    scene.set_panel_label_size_percent(3.0)
    scene.set_panel_label_contrast("Auto")
    scene.show_axes = bool(show_axes)
    scene.axes_render.color_mode = "Black"
    scene.axes_render.location = (-0.68, -0.70)
    scene.axes_render.length = 0.17
    scene.axes_render.font_size_percent = 13.0
    scene.axes_render.label_gap_percent = 2.0
    scene.set_scale_bar_show("Show" if show_scale_bar else "Hide")
    scene.set_scale_bar_length_mode("Manual")
    scene.set_scale_bar_length(SCALE_BAR_UM)
    scene.set_scale_bar_location_x(0.74)
    scene.set_scale_bar_location_y(-0.82)
    scene.set_scale_bar_color("White")
    scene.scale_bar_font_size_percent = 13.0
    scene.set_scale_bar_line_width_percent(1.0)
    result = np.asarray(container.save(alpha=True), dtype=np.float32) / 255.0
    release_render_container(container)
    del scene, container
    gc.collect()
    return result


def _colorbar_image(kind: str, width: int = 900, height: int = 30) -> np.ndarray:
    if kind == "phase":
        values = np.tile(np.linspace(-np.pi, np.pi, width, dtype=np.float32), (height, 1))
        return apply_phase_colormap(values, DEFAULT_PHASE_COLORMAP)
    values = np.tile(np.linspace(-1.0, 1.0, width, dtype=np.float32), (height, 1))
    return apply_diverging_colormap(
        values, DEFAULT_DIVERGING_COLORMAP, vmin=-1.0, vmax=1.0
    )


def _add_colorbar(fig, axes, kind: str, *, layout_scale: float = 1.0) -> None:
    boxes = [axis.get_position() for axis in np.ravel(axes)]
    left, right = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    bottom = min(b.y0 for b in boxes)
    cax = fig.add_axes([left, max(0.018, bottom - 0.056), right - left, 0.018])
    cax.imshow(_colorbar_image(kind), aspect="auto", extent=(0, 1, 0, 1))
    cax.set_yticks(())
    cax.set_xticks((0, 0.5, 1))
    if kind == "phase":
        cax.set_xticklabels(
            (r"$-\pi$", "0", r"$\pi$"), fontsize=max(9.0, 13 * layout_scale)
        )
        cax.set_xlabel(
            "phase (rad)", fontsize=max(9.0, 13 * layout_scale), labelpad=layout_scale
        )
    else:
        cax.set_xticklabels(("-1", "0", "1"), fontsize=max(9.0, 13 * layout_scale))
        cax.set_xlabel(
            "normalized Stokes parameter",
            fontsize=max(9.0, 13 * layout_scale),
            labelpad=layout_scale,
        )
    cax.tick_params(length=0, pad=layout_scale)
    for spine in cax.spines.values():
        spine.set_visible(False)


def _compose_grid(
    datasets: tuple[Dataset, ...],
    *,
    kind: str,
    output: Path,
    row_label_size: float = 20.0,
) -> None:
    if kind == "linear":
        labels = (r"$I_0$", r"$I_{90}$", r"$\arg E_x$", r"$\arg E_y$", r"$\arg(E_y/E_x)$")
        if len(datasets) == 2:
            figsize = (11.0, 5.15)
            adjust = dict(left=0.086, right=0.943, top=0.860, bottom=0.135)
            heading_y = 0.985
        else:
            figsize = (11.0, 7.7)
            adjust = dict(left=0.086, right=0.943, top=0.900, bottom=0.135)
            heading_y = 0.962
    elif kind == "stokes":
        labels = (r"$S_0$", r"$s_1$", r"$s_2$", r"$s_3$")
        figsize, adjust = (9.1, 7.7), dict(left=0.100, right=0.937, top=0.900, bottom=0.135)
    elif kind in {"circular-intensity", "circular-phase"}:
        suffix = ("++", "+-", "-+", "--")
        labels = tuple(
            rf"$|J_{{{item}}}|^2$" if kind.endswith("intensity") else rf"$\arg J_{{{item}}}$"
            for item in suffix
        )
        figsize = (9.1, 7.7)
        adjust = dict(left=0.100, right=0.937, top=0.900, bottom=0.080 if kind.endswith("intensity") else 0.135)
    else:
        raise ValueError(kind)

    figsize = tuple(GRID_LAYOUT_SCALE * value for value in figsize)
    fig, axes = plt.subplots(
        len(datasets), len(labels), figsize=figsize,
        gridspec_kw={"wspace": 0.010, "hspace": 0.010}, squeeze=False,
    )
    for col, label in enumerate(labels):
        axes[0, col].set_title(
            label, pad=5 * GRID_LAYOUT_SCALE, fontsize=max(9.0, 15 * GRID_LAYOUT_SCALE)
        )
    for row, dataset in enumerate(datasets):
        jones = _display_jones(dataset)
        physical_width_um = jones.shape[0] * dataset.pixel_size_um
        kwargs = dict(
            jones=jones,
            input_component=None if kind.startswith("circular") else 0,
            wavelength_um=dataset.wavelength_um,
            pixel_size_um=dataset.pixel_size_um,
        )
        if kind == "linear":
            panels = _optics_panels(analyzer="Linear 0/90 panels", **kwargs)
            panels += _optics_panels(analyzer="Field phase", **kwargs)
        elif kind == "stokes":
            panels = _optics_panels(analyzer="Stokes parameters", **kwargs)
        elif kind == "circular-intensity":
            panels = _optics_panels(analyzer="Jones circular intensity panels", **kwargs)
        else:
            panels = _optics_panels(analyzer="Jones circular phase panels", **kwargs)
        for col, panel in enumerate(panels):
            image = _decorate_panel(
                panel,
                physical_width_um=physical_width_um,
                show_axes=row == len(datasets) - 1 and col == 0,
                show_scale_bar=col == len(panels) - 1,
            )
            axes[row, col].imshow(image, interpolation="nearest")
            axes[row, col].set_xticks(())
            axes[row, col].set_yticks(())
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        axes[row, 0].text(
            -0.060, 0.5, dataset.label, transform=axes[row, 0].transAxes,
            ha="right", va="center", rotation=90,
            fontsize=row_label_size * GRID_LAYOUT_SCALE,
        )
        del jones
        gc.collect()
    fig.subplots_adjust(**adjust)
    if kind == "linear":
        fig.text(
            0.255, heading_y, "Linear analyzer intensity",
            ha="center", va="top", fontsize=19 * GRID_LAYOUT_SCALE,
        )
        fig.text(
            0.686, heading_y, "Output field phase",
            ha="center", va="top", fontsize=19 * GRID_LAYOUT_SCALE,
        )
        _add_colorbar(
            fig, axes[:, 2:5], "phase", layout_scale=GRID_LAYOUT_SCALE
        )
    elif kind == "stokes":
        fig.text(
            0.5, 0.962, "Stokes parameters",
            ha="center", va="top", fontsize=19 * GRID_LAYOUT_SCALE,
        )
        _add_colorbar(
            fig, axes[:, 1:4], "diverging", layout_scale=GRID_LAYOUT_SCALE
        )
    else:
        title = "Circular-basis Jones intensity" if kind.endswith("intensity") else "Circular-basis Jones phase"
        fig.text(
            0.5, 0.962, title,
            ha="center", va="top", fontsize=19 * GRID_LAYOUT_SCALE,
        )
        if kind.endswith("phase"):
            _add_colorbar(
                fig, axes, "phase", layout_scale=GRID_LAYOUT_SCALE
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=FIGURE_DPI)
    plt.close(fig)


def _tilted_datasets() -> tuple[Dataset, ...]:
    result = []
    wavelength_um = optical_parameters(TILTED_CONTROL_WAVELENGTH_NM).wavelength_um
    for key, label, model in (
        ("tilted-marching", "Marching waves", "marching-wave"),
        ("tilted-jones", "Local Jones", "spectral-jones"),
    ):
        archive = load_jones_archive(
            tilted_simulated_jones_path(
                ROOT,
                TILTED_CONTROL_WAVELENGTH_NM,
                model,
            ),
            dtype=np.complex64,
        )
        result.append(
            Dataset(
                key,
                label,
                archive.jones,
                float(archive.pixel_size_um),
                wavelength_um,
                None,
                None,
            )
        )
    return tuple(result)


def _build_experimental_mode_amplitudes() -> None:
    with np.load(EXPERIMENTAL_MODE_ANALYSIS, allow_pickle=False) as stored:
        modes = np.asarray(stored["modes"], dtype=np.int64)
        wavelength_nm = np.asarray(stored["wavelength_nm"], dtype=np.int64)
        amplitudes = np.asarray(
            stored["mode_amplitude_fraction"], dtype=np.float64
        )
    expected_wavelengths = np.asarray((450, 520, 660), dtype=np.int64)
    if not np.array_equal(wavelength_nm, expected_wavelengths):
        raise ValueError(
            f"Unexpected wavelength order in {EXPERIMENTAL_MODE_ANALYSIS}: "
            f"{wavelength_nm.tolist()}"
        )
    if amplitudes.shape != (wavelength_nm.size, modes.size):
        raise ValueError(
            f"Unexpected experimental-mode shape: {amplitudes.shape!r}"
        )

    fig, axes = plt.subplots(
        1,
        wavelength_nm.size,
        figsize=(7.15, 2.35),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.21,
        top=0.86,
        wspace=0.075,
    )
    for panel_index, (axis, wavelength, values) in enumerate(
        zip(axes, wavelength_nm, amplitudes, strict=True)
    ):
        axis.bar(
            modes,
            values,
            width=0.72,
            facecolor="#3b79a6",
            edgecolor="#1f4f70",
            linewidth=0.7,
            zorder=3,
        )
        axis.set_title(rf"${int(wavelength)}\,\mathrm{{nm}}$", fontsize=10.0, pad=5)
        axis.set_xlim(float(modes[0]) - 0.62, float(modes[-1]) + 0.62)
        axis.set_ylim(0.0, 1.02)
        axis.set_xticks(modes)
        axis.set_xlabel(r"angular order $m$")
        axis.grid(axis="y", color="#d5d5d5", linewidth=0.6)
        axis.tick_params(labelsize=10, length=2.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.13,
            1.04,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
        )
    axes[0].set_ylabel(r"Jones-mode amplitude $q_m$")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGS / EXPERIMENTAL_MODE_FIGURE,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _build_tilted_mode_amplitudes() -> None:
    with np.load(TILTED_MODE_ANALYSIS, allow_pickle=False) as stored:
        modes = np.asarray(stored["modes"], dtype=np.int64)
        plane_z_um = np.asarray(stored["plane_z_um"], dtype=np.float64)
        keys = tuple(str(value) for value in stored["dataset_keys"])
        amplitudes = np.asarray(
            stored["mode_amplitude_fraction"], dtype=np.float64
        )
    expected = (
        "symmetric_local",
        "inclined_local",
        "symmetric_marching",
        "inclined_marching",
    )
    if keys != expected:
        raise ValueError(
            f"Unexpected tilted-mode dataset order in {TILTED_MODE_ANALYSIS}: "
            f"{keys!r}"
        )
    if (
        plane_z_um.shape != (1,)
        or not np.isclose(plane_z_um[0], 0.0)
        or amplitudes.shape != (1, len(expected), modes.size)
    ):
        raise ValueError(
            "Expected one exit-surface tilted-mode plane; got "
            f"z={plane_z_um.tolist()}, amplitudes={amplitudes.shape!r}"
        )

    colors = {
        "local": "#d55e00",
        "marching": "#0072b2",
    }
    styles = (
        ("Local Jones", colors["local"]),
        ("Marching waves", colors["marching"]),
    )
    panels = (
        ("Symmetric toron", (0, 2)),
        ("Inclined toron", (1, 3)),
    )
    width = 0.34
    offsets = np.asarray((-0.18, 0.18))
    exit_values = amplitudes[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.20,
        top=0.81,
        wspace=0.08,
    )
    for axis, (title, dataset_indices) in zip(axes, panels, strict=True):
        for offset, dataset_index, (label, color) in zip(
            offsets,
            dataset_indices,
            styles,
            strict=True,
        ):
            axis.bar(
                modes + offset,
                exit_values[dataset_index],
                width=width,
                label=label,
                facecolor=color,
                edgecolor=color,
                linewidth=0.85,
                alpha=0.82,
                zorder=3,
            )
        axis.set_title(title, fontsize=10.0, pad=5)
        axis.set_xlim(float(modes[0]) - 0.62, float(modes[-1]) + 0.62)
        axis.set_ylim(0.0, 1.02)
        axis.set_xticks(modes)
        axis.set_xlabel(r"angular order $m$")
        axis.grid(axis="y", color="#d5d5d5", linewidth=0.6)
        axis.tick_params(labelsize=10, length=2.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Jones-mode amplitude $q_m$")
    axes[0].text(
        0.02, 0.96, "(a)", transform=axes[0].transAxes,
        ha="left", va="top", fontsize=10,
    )
    axes[1].text(
        0.02, 0.96, "(b)", transform=axes[1].transAxes,
        ha="left", va="top", fontsize=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.3,
    )
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGS / TILTED_MODE_FIGURE,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _build_s0_lensing() -> None:
    incident = (
        np.asarray((1, 0), complex), np.asarray((0, 1), complex),
        np.asarray((1, 1j), complex) / np.sqrt(2),
        np.asarray((1, -1j), complex) / np.sqrt(2),
    )
    rows: list[tuple[int, Dataset, list[np.ndarray]]] = []
    for wavelength_nm in (660, 520, 450):
        for dataset in _load_triplet(wavelength_nm)[:2]:
            jones = dataset.jones
            maps = []
            for state in incident:
                field = np.einsum("...ij,j->...i", jones, state, optimize=True)
                maps.append(np.sum(np.abs(field) ** 2, axis=-1) - 1.0)
            rows.append((wavelength_nm, dataset, maps))
    finite = np.concatenate([value[np.isfinite(value)].ravel() for _, _, maps in rows for value in maps])
    scale = max(abs(float(np.percentile(finite, 1))), abs(float(np.percentile(finite, 99))), 0.05)
    fig, axes = plt.subplots(
        len(rows),
        4,
        figsize=(7.9 * S0_LAYOUT_SCALE, 10.25 * S0_LAYOUT_SCALE),
    )
    fig.subplots_adjust(left=0.095, right=0.91, bottom=0.025, top=0.965, wspace=0.035, hspace=0.055)
    for col, label in enumerate((r"$\mathbf{e}_x$", r"$\mathbf{e}_y$", r"$\mathbf{e}_+$", r"$\mathbf{e}_-$")):
        axes[0, col].set_title(
            label, fontsize=11 * S0_LAYOUT_SCALE, pad=3 * S0_LAYOUT_SCALE
        )
    image = None
    for row, (wavelength_nm, dataset, maps) in enumerate(rows):
        nx, ny = dataset.jones.shape[:2]
        extent = (
            -dataset.center_px[0] * dataset.pixel_size_um if dataset.center_px else -0.5 * nx * dataset.pixel_size_um,
            (nx - 1 - dataset.center_px[0]) * dataset.pixel_size_um if dataset.center_px else 0.5 * nx * dataset.pixel_size_um,
            -dataset.center_px[1] * dataset.pixel_size_um if dataset.center_px else -0.5 * ny * dataset.pixel_size_um,
            (ny - 1 - dataset.center_px[1]) * dataset.pixel_size_um if dataset.center_px else 0.5 * ny * dataset.pixel_size_um,
        )
        for col, values in enumerate(maps):
            image = axes[row, col].imshow(values.T, cmap="RdBu_r", vmin=-scale, vmax=scale, extent=extent, origin="lower", interpolation="nearest")
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xticks(())
            axes[row, col].set_yticks(())
        axes[row, 0].text(
            -0.11, 0.5, f"{dataset.label}\n{wavelength_nm} nm",
            transform=axes[row, 0].transAxes, rotation=90,
            ha="center", va="center", fontsize=9 * S0_LAYOUT_SCALE,
        )
    orientation = axes[-1, 0]
    arrow = dict(
        transform=orientation.transAxes,
        length_includes_head=True,
        head_width=0.025,
        head_length=0.035,
        linewidth=1.0 * S0_LAYOUT_SCALE,
        facecolor="black",
        edgecolor="black",
        clip_on=False,
    )
    orientation.arrow(0.12, 0.14, 0.18, 0.0, **arrow)
    orientation.arrow(0.12, 0.14, 0.0, 0.18, **arrow)
    orientation.text(
        0.335, 0.14, r"$x$", transform=orientation.transAxes,
        fontsize=11 * S0_LAYOUT_SCALE, ha="center", va="center",
    )
    orientation.text(
        0.12, 0.365, r"$y$", transform=orientation.transAxes,
        fontsize=11 * S0_LAYOUT_SCALE, ha="center", va="center",
    )
    scale_axis = axes[-1, -1]
    x0, x1 = scale_axis.get_xlim()
    y0, y1 = scale_axis.get_ylim()
    x_span, y_span = x1 - x0, y1 - y0
    bar = min(10.0, 0.35 * abs(x_span))
    x_end = x1 - 0.09 * x_span
    y_bar = y0 + 0.12 * y_span
    scale_axis.plot(
        (x_end - bar, x_end), (y_bar, y_bar),
        color="black", lw=1.6 * S0_LAYOUT_SCALE,
        solid_capstyle="butt", clip_on=False,
    )
    scale_axis.text(
        x_end - 0.5 * bar,
        y_bar + 0.035 * y_span,
        rf"${bar:g}\,\mathregular{{\mu}}\mathrm{{m}}$",
        ha="center",
        va="bottom",
        fontsize=11 * S0_LAYOUT_SCALE,
        color="black",
        clip_on=False,
    )
    fig.canvas.draw()
    positions = [axis.get_position() for axis in axes.ravel()]
    cax = fig.add_axes([
        max(position.x1 for position in positions) + 0.012,
        min(position.y0 for position in positions),
        0.014,
        max(position.y1 for position in positions) - min(position.y0 for position in positions),
    ])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(
        r"$\Delta S_0=S_0-1$", fontsize=12 * S0_LAYOUT_SCALE
    )
    cbar.ax.tick_params(labelsize=9 * S0_LAYOUT_SCALE)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "supplement_s0_lensing_wavelengths_soft.pdf", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength", type=int, choices=(450, 520, 660))
    parser.add_argument("--experimental-modes", action="store_true")
    parser.add_argument("--tilted", action="store_true")
    parser.add_argument("--s0", action="store_true")
    add_force_argument(parser)
    args = parser.parse_args()
    if (
        args.wavelength is None
        and not args.experimental_modes
        and not args.tilted
        and not args.s0
    ):
        parser.error(
            "select --wavelength, --experimental-modes, --tilted, or --s0"
        )
    FIGS.mkdir(parents=True, exist_ok=True)
    names = {
        450: ("supplement_linear_phase_450_soft.pdf", "supplement_stokes_450_soft.pdf"),
        520: ("supplement_linear_phase_520_soft.pdf", "supplement_stokes_520_soft.pdf"),
        660: ("figure_2_linear_phase_660_soft.pdf", "figure_3_stokes_660_soft.pdf"),
    }
    outputs = []
    if args.wavelength is not None:
        outputs.extend(FIGS / name for name in names[int(args.wavelength)])
        if args.wavelength == 520:
            outputs.extend(
                (
                    FIGS / "supplement_circular_jones_intensity_520_soft.pdf",
                    FIGS / "supplement_circular_jones_phase_520_soft.pdf",
                )
            )
    if args.experimental_modes:
        outputs.append(FIGS / EXPERIMENTAL_MODE_FIGURE)
    if args.tilted:
        outputs.extend(
            (
                FIGS / TILTED_LINEAR_FIGURE,
                FIGS / TILTED_MODE_FIGURE,
            )
        )
    if args.s0:
        outputs.append(FIGS / "supplement_s0_lensing_wavelengths_soft.pdf")
    refuse_existing_outputs(
        outputs,
        force=args.force,
        project_root=ROOT,
    )
    if args.wavelength is not None:
        wavelength_nm = int(args.wavelength)
        linear_name, stokes_name = names[wavelength_nm]
        datasets = _load_triplet(wavelength_nm)
        _compose_grid(datasets, kind="linear", output=FIGS / linear_name)
        _compose_grid(datasets, kind="stokes", output=FIGS / stokes_name)
        if wavelength_nm == 520:
            _compose_grid(datasets, kind="circular-intensity", output=FIGS / "supplement_circular_jones_intensity_520_soft.pdf")
            _compose_grid(datasets, kind="circular-phase", output=FIGS / "supplement_circular_jones_phase_520_soft.pdf")
        del datasets
        gc.collect()
    if args.experimental_modes:
        _build_experimental_mode_amplitudes()
    if args.tilted:
        _compose_grid(
            _tilted_datasets(), kind="linear",
            output=FIGS / TILTED_LINEAR_FIGURE,
            row_label_size=15.0,
        )
        _build_tilted_mode_amplitudes()
    if args.s0:
        _build_s0_lensing()


if __name__ == "__main__":
    main()
