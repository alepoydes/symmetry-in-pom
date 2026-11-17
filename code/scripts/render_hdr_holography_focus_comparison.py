#!/usr/bin/env python3
"""Render HDR controls and the holographic through-focus I90 sequence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toronoptics.figure_output import (
    FOCUS_PANEL_FRAME_SCALE,
    OPTICAL_ANNOTATION_COLOR,
    expand_focus_panels,
    recolor_white_annotations,
    release_render_container,
)
from scipy.ndimage import map_coordinates
import topovec.mgl as tvmgl

from toronoptics.article_data import (
    ARTICLE_GREEN_WAVELENGTH_NM,
    ARTICLE_VARIANT_SUFFIX,
    decimate_centered_jones,
    experimental_jones_propagation_wavelength_um,
    publication_focus_grid,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.axisymmetric_jones import intensity_harmonics_on_circles
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.focus_stack import (
    AngularSpectrumJonesPropagator,
    estimate_radial_center,
)
from toronoptics.focus_stack_hdr import fill_small_display_holes


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JONES = (
    PROJECT_ROOT
    / "data/experimental/results_toron25um_50X_averaged/520nm_average.npz"
)
DEFAULT_HDR = (
    PROJECT_ROOT / "data/experimental/results_toron25um_50X_focus_hdr"
)
DEFAULT_FOCUS = (
    PROJECT_ROOT
    / "data/analysis/results_toron25um_50X_focus_recovery/focus_fit.npz"
)
DEFAULT_FOCUS_REPORT = (
    PROJECT_ROOT
    / "data/analysis/results_toron25um_50X_focus_recovery/report.json"
)
DEFAULT_EVOLUTION_REPORT = (
    PROJECT_ROOT
    / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}/report.json"
)
DEFAULT_EVOLUTION_ARCHIVE = (
    PROJECT_ROOT
    / f"data/analysis/axisymmetric_jones_evolution{ARTICLE_VARIANT_SUFFIX}"
    / "axisymmetric_jones_evolution.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / f"data/analysis/results_toron25um_50X_focus_recovery{ARTICLE_VARIANT_SUFFIX}"
)
DEFAULT_FIGURE = (
    DEFAULT_OUTPUT
    / f"direct_polarimetry_focus_i90_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)
DEFAULT_HOLOGRAPHY_FIGURE = (
    DEFAULT_OUTPUT
    / f"holography_focus_i90_520{ARTICLE_VARIANT_SUFFIX}.pdf"
)

FIELD_WIDTH_UM = 80.0
PANEL_PIXELS = 900
FINAL_DPI = 600
HOLOGRAPHY_DISPLAY_PERCENTILE = 99.5
HARMONIC_MAX_RADIUS_UM = 30.0
HARMONIC_LOG_DECADES = 2.0
HARMONIC_COLORS = ("#242424", "#704C8F", "#B47A0B")
GOLDEN_RATIO = 0.5 * (1.0 + np.sqrt(5.0))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _publication_jones_wavelength_um(
    evolution_report: dict[str, Any],
) -> float:
    """Read and validate the wavelength of the full-Jones publication samples."""

    try:
        value = float(
            evolution_report["exact_sample_sets"]["figure1_through_focus"][
                "propagation_wavelength_um"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "The evolution report lacks the propagation wavelength of the "
            "exact Figure-1 samples; rerun the radial Jones analysis."
        ) from exc
    expected = experimental_jones_propagation_wavelength_um(
        ARTICLE_GREEN_WAVELENGTH_NM
    )
    if not np.isclose(value, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "The exact Figure-1 samples use a stale full-Jones propagation "
            f"wavelength ({value:g} um instead of {expected:g} um); rerun "
            "analyze_axisymmetric_jones_evolution.py --mode compute."
        )
    return value


def _portable_path(path: Path) -> str:
    """Store repository-local provenance without checkout-specific prefixes."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jones(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float], float]:
    archive = load_jones_archive(path, dtype=np.complex64)
    with np.load(path, allow_pickle=False) as data:
        if "array_center_px" in data.files:
            center = tuple(
                float(value)
                for value in np.asarray(data["array_center_px"]).reshape(2)
            )
        else:
            center = (
                0.5 * (archive.jones.shape[0] - 1),
                0.5 * (archive.jones.shape[1] - 1),
            )
    reduced, reduced_center = decimate_centered_jones(
        archive.jones,
        center,
        factor=2,
    )
    return (
        np.asarray(reduced, dtype=np.complex64),
        reduced_center,
        2.0 * float(archive.pixel_size_um),
    )


def _nearest_focus_indices(
    focus_um: np.ndarray,
    *,
    count: int,
    zero_tolerance_um: float = 1.0e-9,
) -> np.ndarray:
    values = np.asarray(focus_um, dtype=np.float64)
    candidates = np.flatnonzero(np.abs(values) > float(zero_tolerance_um))
    if candidates.size < int(count):
        raise ValueError(
            f"Only {candidates.size} nonzero focus planes are available"
        )
    nearest = candidates[
        np.argsort(np.abs(values[candidates]), kind="stable")[: int(count)]
    ]
    return nearest[np.argsort(values[nearest], kind="stable")]


def _extract_centered(
    image_yx: np.ndarray,
    *,
    center_xy: tuple[float, float],
    side: int,
) -> np.ndarray:
    coordinate = np.arange(int(side), dtype=np.float64) - 0.5 * (side - 1)
    x = float(center_xy[0]) + coordinate[None, :]
    y = float(center_xy[1]) + coordinate[:, None]
    return map_coordinates(
        np.asarray(image_yx, dtype=np.float64),
        (
            np.broadcast_to(y, (side, side)),
            np.broadcast_to(x, (side, side)),
        ),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def _direct_i90(
    hdr_root: Path,
    *,
    frame_number: int,
    center_xy: tuple[float, float],
    side: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    with np.load(
        hdr_root / "frames" / f"{int(frame_number):04d}.npz",
        allow_pickle=False,
    ) as data:
        channels = tuple(str(value) for value in data["channel_names"])
        intensity = np.asarray(data["intensity_yxc"], dtype=np.float64)
        confidence = (
            np.asarray(data["confidence_yxc"], dtype=np.float64) / 255.0
        )
    index = channels.index("I90")
    intensity_crop = _extract_centered(
        intensity[..., index],
        center_xy=center_xy,
        side=side,
    )
    confidence_crop = _extract_centered(
        confidence[..., index],
        center_xy=center_xy,
        side=side,
    )
    return fill_small_display_holes(
        intensity_crop,
        confidence_crop,
        maximum_distance_px=4.0,
    )


def _model_i90(
    propagated_jones: np.ndarray,
    *,
    input_angle_deg: float,
    initial_center_xy: tuple[float, float],
    side: int,
    beamsplitter_output_ratio: complex,
) -> tuple[np.ndarray, tuple[float, float], float]:
    alpha = np.radians(float(input_angle_deg))
    incident = np.asarray(
        (np.cos(alpha), np.sin(alpha)),
        dtype=np.complex64,
    )
    field = propagated_jones @ incident
    field = np.asarray(field).copy()
    field[..., 1] *= complex(beamsplitter_output_ratio)
    intensity_yx = np.asarray(np.abs(field[..., 1].T) ** 2, dtype=np.float64)
    center = estimate_radial_center(
        intensity_yx,
        initial_center_xy=initial_center_xy,
        search_half_width_px=24.0,
        minimum_radius_px=18.0,
        maximum_radius_px=min(320.0, 0.46 * min(intensity_yx.shape)),
    )
    crop = _extract_centered(
        intensity_yx,
        center_xy=center.center_xy,
        side=side,
    )
    return (
        crop,
        center.center_xy,
        center.rms_distance_px,
    )


def _render_topovec_intensity(
    intensity_yx: np.ndarray,
    *,
    wavelength_um: float,
    pixel_size_um: float,
    exposure_ev: float,
    show_axes: bool,
    show_scale_bar: bool,
    annotation_font_size_percent: float = 11.5,
    annotation_color: str | None = None,
    frame_scale: float = 1.0,
) -> np.ndarray:
    values = np.nan_to_num(
        np.clip(np.asarray(intensity_yx, dtype=np.float32), 0.0, None),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    pseudo_jones = np.zeros(
        (*values.T.shape, 2, 2),
        dtype=np.complex64,
    )
    pseudo_jones[..., 1, 0] = np.sqrt(values.T)
    frame_pixels = int(round(PANEL_PIXELS * frame_scale))
    frame_scale = frame_pixels / float(PANEL_PIXELS)
    container = tvmgl.render_precomputed_optics(
        {"jones_matrices": np.ascontiguousarray(pseudo_jones)},
        size=(frame_pixels, frame_pixels),
        scene_state={
            "phase_reference_mode": "Solver native",
            "wavelength_um": float(wavelength_um),
            "pixel_size_um": float(pixel_size_um),
        },
    )
    scene = container.scene
    scene.set_polarization_preset("Linear X")
    scene.set_analyzer_preset("Linear 90")
    scene.set_exposure_ev(float(exposure_ev))
    scene.set_tone_curve("sRGB")
    scene.set_overlay_outline_mode("Hide")
    scene.set_panel_label_show("Hide")
    scene.set_panel_gap_percent(0.0)
    scene.set_panel_outside_background("Black" if frame_scale > 1.0 else "Transparent")
    scene.set_color_bar_show("Hide")
    scene.show_axes = bool(show_axes)
    scene.axes_render.color_mode = "White" if annotation_color else "Contrast"
    scene.axes_render.location = (-0.76, -0.75)
    scene.axes_render.length = 0.13
    scene.axes_render.font_size_percent = float(annotation_font_size_percent)
    scene.axes_render.label_gap_percent = 1.0
    scene.set_scale_bar_show("Show" if show_scale_bar else "Hide")
    scene.set_scale_bar_length_mode("Manual")
    scene.set_scale_bar_length(10.0)
    scene.set_scale_bar_location_x(0.52 if annotation_font_size_percent > 10.0 else 0.68)
    scene.set_scale_bar_location_y(-0.79)
    scene.set_scale_bar_color("White" if annotation_color else "Contrast")
    # Frame compensation below accounts for TopoVec's effective 10% cap.
    scene.scale_bar_font_size_percent = float(annotation_font_size_percent)
    scene.set_scale_bar_line_width_percent(0.9)
    scene.set_polarization_incident_show("Hide")
    scene.set_polarization_analyzer_show("Hide")
    scene.set_polarization_direction_show("Hide")
    if frame_scale > 1.0:
        scene.axes_render.location = (-1.0 + 0.24 / frame_scale, -1.0 + 0.25 / frame_scale)
        scene.axes_render.length = 0.13 / frame_scale
        scene.axes_render.font_size_percent = min(annotation_font_size_percent, 10.0) / frame_scale
        scene.axes_render.label_gap_percent = 1.0 / frame_scale
        scene.set_scale_bar_location_x(1.0 - 0.24 / frame_scale)
        scene.set_scale_bar_location_y(-1.0 + 0.21 / frame_scale)
        scene.scale_bar_font_size_percent = min(annotation_font_size_percent, 10.0) / frame_scale
        scene.set_scale_bar_line_width_percent(0.9 / frame_scale)
        # TopoVec initializes the view geometry on the first draw.
        container.render_array_rgba()
        scene.zoom_fixed_view(frame_scale)
    rendered = np.asarray(container.render_array_rgba(), dtype=np.uint8)
    if show_scale_bar and getattr(scene, "_fixed_view_scale_bar_overlay_failed", False):
        raise RuntimeError("TopoVec failed to render the optical scale bar")
    if annotation_color and (show_axes or show_scale_bar):
        scene.show_axes = False
        scene.set_scale_bar_show("Hide")
        background = np.asarray(container.render_array_rgba(), dtype=np.uint8)
        rendered = recolor_white_annotations(rendered, background, color=annotation_color)
    release_render_container(container)
    del scene, container, pseudo_jones
    gc.collect()
    return rendered


def _holography_display_exposure(
    holography_maps: list[np.ndarray],
    *,
    percentile: float = HOLOGRAPHY_DISPLAY_PERCENTILE,
) -> tuple[float, float]:
    finite = np.concatenate(
        [
            values[np.isfinite(values) & (values >= 0.0)].ravel()
            for values in holography_maps
        ]
    )
    if finite.size == 0:
        raise ValueError("Propagated holography contains no finite intensities")
    reference = float(np.percentile(finite, float(percentile)))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError(
            f"Invalid holographic display reference {reference!r}"
        )
    return reference, -float(np.log2(reference))


def _compose(
    direct_panels: list[np.ndarray],
    model_panels: list[np.ndarray],
    *,
    focus_um: np.ndarray,
    output_pdf: Path,
    output_png: Path,
) -> None:
    fig, axes = plt.subplots(2, len(focus_um), figsize=(7.15, 3.02))
    fig.subplots_adjust(
        left=0.070,
        right=0.997,
        bottom=0.018,
        top=0.895,
        wspace=0.012,
        hspace=0.018,
    )
    for column, value in enumerate(focus_um):
        axes[0, column].set_title(
            rf"$z={value:.2f}\,\mathregular{{\mu}}\mathrm{{m}}$",
            fontsize=10.0,
            pad=2.5,
        )
        for row, panel in enumerate(
            (direct_panels[column], model_panels[column])
        ):
            ax = axes[row, column]
            ax.imshow(panel, origin="upper", interpolation="none")
            ax.set_xticks(())
            ax.set_yticks(())
            for spine in ax.spines.values():
                spine.set_visible(False)
    for row, label in enumerate(("direct polarimetry", "propagated holography")):
        axes[row, 0].text(
            -0.065,
            0.50,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10.0,
            clip_on=False,
        )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
        dpi=FINAL_DPI,
    )
    fig.savefig(
        output_png,
        dpi=FINAL_DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)


def _compose_holography(
    panels: list[np.ndarray],
    *,
    focus_um: np.ndarray,
    surface_um: float,
    annular_focus_um: float,
    radius_um: np.ndarray,
    harmonics: np.ndarray,
    output_pdf: Path,
    output_png: Path,
) -> None:
    if not (
        np.isclose(focus_um[1], surface_um)
        and np.isclose(focus_um[3], annular_focus_um)
    ):
        raise ValueError(
            "Publication focus grid does not place the requested landmarks "
            "in panels 2 and 4"
        )
    if harmonics.shape != (len(focus_um), 3, radius_um.size):
        raise ValueError("Expected harmonics with shape (plane,3,radius)")
    font_size = 9.05
    fig = plt.figure(figsize=(7.15, 2.85))
    grid = fig.add_gridspec(
        2,
        len(focus_um),
        height_ratios=(GOLDEN_RATIO, 1.0),
        left=0.070,
        right=0.990,
        bottom=0.210,
        top=0.860,
        wspace=0.16,
        hspace=0.03,
    )
    image_axes = [fig.add_subplot(grid[0, column]) for column in range(len(focus_um))]
    profile_axes: list[plt.Axes] = []
    for column in range(len(focus_um)):
        shared = profile_axes[0] if profile_axes else None
        profile_axes.append(
            fig.add_subplot(grid[1, column], sharex=shared, sharey=shared)
        )
    landmarks = {
        1: "LC surface",
        3: "annular focus",
    }
    surface_referenced_um = np.asarray(focus_um) - float(surface_um)
    for column, (value, panel) in enumerate(
        zip(surface_referenced_um, panels, strict=True)
    ):
        title = rf"$z={value:.2f}\,\mathregular{{\mu}}\mathrm{{m}}$"
        if column in landmarks:
            title = landmarks[column] + "\n" + title
        else:
            title = "\n" + title
        image_axes[column].set_title(title, fontsize=font_size, pad=2.0, linespacing=0.95)
        image_axes[column].imshow(panel, origin="upper", interpolation="none")
        image_axes[column].text(0.04, 0.95, f"({chr(97 + column)})", transform=image_axes[column].transAxes, ha="left", va="top", fontsize=font_size, color=OPTICAL_ANNOTATION_COLOR)
        image_axes[column].set_xticks(())
        image_axes[column].set_yticks(())
        for spine in image_axes[column].spines.values():
            spine.set_visible(False)
    displayed = np.asarray(radius_um) <= HARMONIC_MAX_RADIUS_UM
    displayed_values = harmonics[:, :, displayed]
    if not np.any(np.isfinite(displayed_values) & (displayed_values > 0.0)):
        raise ValueError("No positive harmonic amplitudes are available")
    upper_limit = 1.08 * float(np.nanmax(displayed_values))
    lower_limit = upper_limit / 10.0**HARMONIC_LOG_DECADES
    labels = (r"$H_0$", r"$|H_2|$", r"$|H_4|$")
    for column, axis in enumerate(profile_axes):
        for mode_index, (label, color) in enumerate(
            zip(labels, HARMONIC_COLORS, strict=True)
        ):
            profile = harmonics[column, mode_index, displayed]
            axis.plot(
                radius_um[displayed],
                profile,
                color=color,
                lw=0.85,
                label=label,
                zorder=3,
            )
            if mode_index > 0:
                axis.fill_between(
                    radius_um[displayed],
                    lower_limit,
                    np.maximum(profile, lower_limit),
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    zorder=1,
                )
        axis.text(0.96, 0.94, f"({chr(102 + column)})", transform=axis.transAxes, ha="right", va="top", fontsize=font_size)
        axis.set_box_aspect(1.0 / GOLDEN_RATIO)
        axis.set_yscale("log")
        axis.set_xlim(0.0, HARMONIC_MAX_RADIUS_UM)
        axis.set_ylim(lower_limit, upper_limit)
        axis.set_xticks((0.0, 15.0, 30.0))
        axis.tick_params(labelsize=font_size, length=2.0, pad=2.0, labelleft=column == 0)
        axis.grid(axis="y", which="major", color="0.85", lw=0.6)
        if column == 0:
            axis.set_ylabel(r"$H_m$", fontsize=font_size, labelpad=2.0)
    handles, legend_labels = profile_axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, legend_labels, loc="lower right", bbox_to_anchor=(0.990, 0.090),
        ncols=3, fontsize=font_size, frameon=False, handlelength=1.5,
        columnspacing=1.5, borderaxespad=0, borderpad=0,
    )
    fig.canvas.draw()
    legend_text = legend.get_texts()[0]
    baseline = fig.transFigure.inverted().transform(
        legend_text.get_transform().transform(legend_text.get_position())
    )[1]
    fig.supxlabel(
        r"$\rho$ ($\mathregular{\mu}$m)", fontsize=font_size,
        x=0.53, y=baseline, va="baseline",
    )
    bounds = expand_focus_panels(fig, image_axes, profile_axes)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        transparent=True,
        bbox_inches=bounds,
        dpi=FINAL_DPI,
    )
    fig.savefig(
        output_png,
        dpi=FINAL_DPI,
        transparent=True,
        bbox_inches=bounds,
    )
    plt.close(fig)


def _load_publication_harmonics(
    archive_path: Path,
    *,
    expected_z_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    required = (
        "exact_sample_set_keys",
        "exact_sample_z_um",
        "exact_sample_radius_um",
        "exact_sample_physical_crossed_H0",
        "exact_sample_physical_crossed_H2",
        "exact_sample_physical_crossed_H4",
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(
                f"{archive_path} lacks exact publication samples {missing}; "
                "run analyze_axisymmetric_jones_evolution.py --mode publication"
            )
        set_keys = tuple(str(value) for value in archive["exact_sample_set_keys"])
        z_um = np.asarray(archive["exact_sample_z_um"], dtype=np.float64)
        radius_um = np.asarray(archive["exact_sample_radius_um"], dtype=np.float64)
        h0 = np.asarray(
            archive["exact_sample_physical_crossed_H0"], dtype=np.float64
        )
        h2 = np.asarray(
            archive["exact_sample_physical_crossed_H2"], dtype=np.complex128
        )
        h4 = np.asarray(
            archive["exact_sample_physical_crossed_H4"], dtype=np.complex128
        )
    if set_keys != ("figure1_through_focus",):
        raise ValueError(f"Unexpected exact sample set: {set_keys!r}")
    if z_um.shape != expected_z_um.shape or not np.allclose(
        z_um, expected_z_um, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("Archived publication planes do not match the figure grid")
    plotted = np.stack((h0, np.abs(h2), np.abs(h4)), axis=1)
    complex_coefficients = np.stack((h0.astype(np.complex128), h2, h4), axis=1)
    metadata = {
        "archive": _portable_path(archive_path),
        "sample_set": set_keys[0],
        "z_um": z_um.tolist(),
        "coefficient_convention": "H_m=(2 pi)^-1 integral I exp(-i m phi) dphi",
        "radial_smoothing": "none in the independent 2-D audit",
    }
    return radius_um, plotted, {
        **metadata,
        "complex_coefficients": complex_coefficients,
    }


def _weighted_complex_error(
    radius_um: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    weight = np.asarray(radius_um, dtype=np.float64)
    difference = np.asarray(candidate) - np.asarray(reference)
    numerator = np.trapezoid(np.abs(difference) ** 2 * weight, radius_um)
    denominator = np.trapezoid(np.abs(reference) ** 2 * weight, radius_um)
    return {
        "relative_rms": float(np.sqrt(numerator / max(denominator, 1.0e-300))),
        "absolute_rms": float(
            np.sqrt(
                np.trapezoid(np.abs(difference) ** 2 * weight, radius_um)
                / max(np.trapezoid(weight, radius_um), 1.0e-300)
            )
        ),
        "maximum_absolute": float(np.max(np.abs(difference))),
    }


def _write_harmonic_audit(
    output: Path,
    publication_maps: list[np.ndarray],
    *,
    inputs: list[dict[str, str]],
    pixel_size_um: float,
    radius_um: np.ndarray,
    axisymmetric_coefficients: np.ndarray,
    surface_referenced_z_um: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(radius_um) <= HARMONIC_MAX_RADIUS_UM
    audit_radius = np.asarray(radius_um)[selected]
    direct = []
    for values in publication_maps:
        side_y, side_x = values.shape
        profiles = intensity_harmonics_on_circles(
            values,
            center_xy_px=(0.5 * (side_x - 1), 0.5 * (side_y - 1)),
            pixel_size_um=pixel_size_um,
            radius_um=audit_radius,
            modes=(0, 2, 4),
            angular_samples=720,
        )
        direct.append(profiles.coefficients)
    direct_coefficients = np.stack(direct, axis=0)
    radial_coefficients = axisymmetric_coefficients[:, :, selected]
    mode_records: dict[str, Any] = {}
    for plane in range(direct_coefficients.shape[0]):
        mode_records[str(plane + 1)] = {
            str(mode): _weighted_complex_error(
                audit_radius,
                direct_coefficients[plane, index],
                radial_coefficients[plane, index],
            )
            for index, mode in enumerate((0, 2, 4))
        }

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "figure1_harmonic_audit.npz"
    figure_path = output / "figure1_harmonic_audit.pdf"
    report_path = output / "figure1_harmonic_audit.json"
    report = {
        "schema_version": "figure1-harmonic-audit-v2",
        "inputs": inputs,
        "definition": "direct circles from the exact displayed 2-D intensities versus H_m derived after axisymmetric Jones projection",
        "radial_interval_um": [float(audit_radius[0]), float(audit_radius[-1])],
        "radial_weight": "rho d rho",
        "radial_smoothing": "none",
        "errors_by_plane_and_mode": mode_records,
        "archive": _portable_path(archive_path),
        "figure": _portable_path(figure_path),
        "report": _portable_path(report_path),
    }
    np.savez_compressed(
        archive_path,
        schema_version=np.asarray(report["schema_version"]),
        metadata_json=np.asarray(
            json.dumps(report, sort_keys=True, ensure_ascii=True)
        ),
        radius_um=audit_radius,
        modes=np.asarray((0, 2, 4), dtype=np.int16),
        surface_referenced_z_um=np.asarray(surface_referenced_z_um),
        direct_2d_coefficients=direct_coefficients.astype(np.complex64),
        axisymmetric_coefficients=radial_coefficients.astype(np.complex64),
    )
    figure, axes = plt.subplots(
        1, direct_coefficients.shape[0], figsize=(7.15, 1.75), sharex=True, sharey=True
    )
    labels = (r"$H_0$", r"$|H_2|$", r"$|H_4|$")
    for plane, axis in enumerate(np.atleast_1d(axes)):
        for index, (label, color) in enumerate(
            zip(labels, HARMONIC_COLORS, strict=True)
        ):
            axis.plot(
                audit_radius,
                np.abs(radial_coefficients[plane, index]),
                color=color,
                lw=0.9,
                label=label,
            )
            axis.plot(
                audit_radius,
                np.abs(direct_coefficients[plane, index]),
                color=color,
                lw=0.7,
                ls=":",
            )
        axis.set_title(rf"$z={surface_referenced_z_um[plane]:.2f}\,\mathregular{{\mu}}$m", fontsize=7)
        axis.set_xlim(0.0, HARMONIC_MAX_RADIUS_UM)
        axis.tick_params(labelsize=6.5)
        axis.set_xlabel(r"$\rho$ ($\mathregular{\mu}$m)", fontsize=7)
        if plane == 0:
            axis.set_ylabel(r"$|H_m|$", fontsize=7)
            axis.legend(fontsize=5.7, frameon=False)
    figure.tight_layout(pad=0.4, w_pad=0.25)
    figure.savefig(figure_path, dpi=FINAL_DPI)
    plt.close(figure)
    _write_json(report_path, report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    jones_path = args.jones.expanduser().resolve()
    hdr_root = args.hdr_root.expanduser().resolve()
    focus_path = args.focus_fit.expanduser().resolve()
    focus_report_path = args.focus_report.expanduser().resolve()
    evolution_report_path = args.evolution_report.expanduser().resolve()
    evolution_archive_path = args.evolution_archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    figure = args.figure.expanduser().resolve()
    holography_figure = args.holography_figure.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    focus_report = json.loads(
        focus_report_path.read_text(encoding="utf-8")
    )
    evolution_report = json.loads(
        evolution_report_path.read_text(encoding="utf-8")
    )
    direct_pom_wavelength_um = float(
        focus_report["optical_model"]["wavelength_um"]
    )
    publication_jones_wavelength_um = _publication_jones_wavelength_um(
        evolution_report
    )
    numerical_aperture = float(
        focus_report["optical_model"]["numerical_aperture"]
    )
    padding_factor = float(
        focus_report["optical_model"]["padding_factor"]
    )
    with np.load(focus_path, allow_pickle=False) as data:
        frame_numbers = np.asarray(data["frame_numbers"], dtype=np.int64)
        focus_um_all = np.asarray(data["recovered_focus_um"], dtype=np.float64)
        centers_xy = np.asarray(data["centers_xy_px"], dtype=np.float64)
        input_angle_deg = float(data["input_angle_deg"])
    ratio_record = focus_report["optical_model"]["beamsplitter_output_field_ratio"]
    beamsplitter_output_ratio = complex(
        float(ratio_record["real"]),
        float(ratio_record["imag"]),
    )
    selected = _nearest_focus_indices(
        focus_um_all,
        count=int(args.count),
    )
    selected = np.asarray(selected, dtype=np.int64)
    selected_frame_paths = tuple(
        hdr_root / "frames" / f"{int(frame_numbers[index]):04d}.npz"
        for index in selected
    )
    input_paths = (
        jones_path,
        focus_path,
        focus_report_path,
        evolution_report_path,
        evolution_archive_path,
        *selected_frame_paths,
    )
    input_records = [
        {
            "path": _portable_path(path),
            "sha256": _sha256(path),
        }
        for path in input_paths
    ]
    harmonic_input_records = input_records[:5]

    jones, jones_center, pixel_size_um = _load_jones(jones_path)
    direct_pixel_size_um = float(
        focus_report["optical_model"]["direct_pixel_size_um"]
    )
    if not np.isclose(
        pixel_size_um,
        direct_pixel_size_um,
        atol=5.0e-5,
    ):
        raise ValueError(
            f"Jones and direct grids differ: {pixel_size_um} versus "
            f"{direct_pixel_size_um} um"
        )
    side = int(round(float(args.field_width_um) / pixel_size_um))
    side += 1 - side % 2

    propagator = AngularSpectrumJonesPropagator(
        jones,
        pixel_size_um=pixel_size_um,
        wavelength_um=direct_pom_wavelength_um,
        numerical_aperture=numerical_aperture,
        padding_factor=padding_factor,
        workers=int(args.workers),
    )
    direct_maps: list[np.ndarray] = []
    direct_display_repairs: list[dict[str, float | int]] = []
    model_maps: list[np.ndarray] = []
    model_centers: list[tuple[float, float]] = []
    center_residuals: list[float] = []
    for index in selected:
        direct_map, display_repair = _direct_i90(
            hdr_root,
            frame_number=int(frame_numbers[index]),
            center_xy=tuple(centers_xy[index]),
            side=side,
        )
        direct_maps.append(direct_map)
        direct_display_repairs.append(display_repair)
        model, center, residual = _model_i90(
            propagator.propagate_jones(float(focus_um_all[index])),
            input_angle_deg=input_angle_deg,
            initial_center_xy=jones_center,
            side=side,
            beamsplitter_output_ratio=beamsplitter_output_ratio,
        )
        model_maps.append(model)
        model_centers.append(center)
        center_residuals.append(residual)

    direct_panels: list[np.ndarray] = []
    model_panels: list[np.ndarray] = []
    display_reference, comparison_exposure_ev = _holography_display_exposure(
        model_maps
    )
    panel_dir = output / "topovec_i90_focus_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for column, index in enumerate(selected):
        direct_panel = _render_topovec_intensity(
            direct_maps[column],
            wavelength_um=direct_pom_wavelength_um,
            pixel_size_um=pixel_size_um,
            exposure_ev=comparison_exposure_ev,
            show_axes=(column == 0),
            show_scale_bar=(column == 0),
        )
        model_panel = _render_topovec_intensity(
            model_maps[column],
            wavelength_um=direct_pom_wavelength_um,
            pixel_size_um=pixel_size_um,
            exposure_ev=comparison_exposure_ev,
            show_axes=(column == 0),
            show_scale_bar=(column == 0),
        )
        direct_panels.append(direct_panel)
        model_panels.append(model_panel)
        from PIL import Image

        frame_number = int(frame_numbers[index])
        Image.fromarray(direct_panel).save(
            panel_dir / f"frame_{frame_number:04d}_direct.png"
        )
        Image.fromarray(model_panel).save(
            panel_dir / f"frame_{frame_number:04d}_propagated.png"
        )

    publication_focus_um, surface_um, annular_focus_um = (
        publication_focus_grid(evolution_report)
    )

    png_path = output / "direct_polarimetry_focus_i90_520.png"
    _compose(
        direct_panels,
        model_panels,
        focus_um=focus_um_all[selected] - surface_um,
        output_pdf=figure,
        output_png=png_path,
    )

    publication_propagator = AngularSpectrumJonesPropagator(
        jones,
        pixel_size_um=pixel_size_um,
        wavelength_um=publication_jones_wavelength_um,
        numerical_aperture=None,
        padding_factor=padding_factor,
        workers=int(args.workers),
    )
    publication_maps: list[np.ndarray] = []
    publication_panels: list[np.ndarray] = []
    publication_centers: list[tuple[float, float]] = []
    publication_center_residuals: list[float] = []
    publication_panel_dir = output / "topovec_i90_holography_propagation"
    publication_panel_dir.mkdir(parents=True, exist_ok=True)
    for column, focus_um in enumerate(publication_focus_um):
        model, center, residual = _model_i90(
            publication_propagator.propagate_jones(float(focus_um)),
            input_angle_deg=input_angle_deg,
            initial_center_xy=jones_center,
            side=side,
            beamsplitter_output_ratio=1.0,
        )
        panel = _render_topovec_intensity(
            model,
            wavelength_um=publication_jones_wavelength_um,
            pixel_size_um=pixel_size_um,
            exposure_ev=0.0,
            show_axes=(column == 0),
            show_scale_bar=(column == 0),
            annotation_font_size_percent=11.5,
            annotation_color=OPTICAL_ANNOTATION_COLOR,
            frame_scale=FOCUS_PANEL_FRAME_SCALE,
        )
        publication_maps.append(model)
        publication_panels.append(panel)
        publication_centers.append(center)
        publication_center_residuals.append(residual)
        from PIL import Image

        Image.fromarray(panel).save(
            publication_panel_dir / f"plane_{column + 1}_propagated.png"
        )
    radius_um, plotted_harmonics, harmonic_metadata = _load_publication_harmonics(
        evolution_archive_path,
        expected_z_um=publication_focus_um,
    )
    harmonic_audit = _write_harmonic_audit(
        output,
        publication_maps,
        inputs=harmonic_input_records,
        pixel_size_um=pixel_size_um,
        radius_um=radius_um,
        axisymmetric_coefficients=harmonic_metadata.pop("complex_coefficients"),
        surface_referenced_z_um=publication_focus_um - surface_um,
    )
    holography_png = output / "holography_focus_i90_520.png"
    _compose_holography(
        publication_panels,
        focus_um=publication_focus_um,
        surface_um=surface_um,
        annular_focus_um=annular_focus_um,
        radius_um=radius_um,
        harmonics=plotted_harmonics,
        output_pdf=holography_figure,
        output_png=holography_png,
    )
    report = {
        "schema_version": "hdr-holography-focus-i90-comparison-v6",
        "inputs": input_records,
        "sources": {
            "jones": _portable_path(jones_path),
            "hdr_stack": _portable_path(hdr_root),
            "focus_fit": _portable_path(focus_path),
            "focus_report": _portable_path(focus_report_path),
            "evolution_report": _portable_path(evolution_report_path),
            "evolution_archive": _portable_path(evolution_archive_path),
        },
        "selection": {
            "rule": (
                "five acquired planes with smallest nonzero absolute "
                "distance from the holographic reconstruction plane, "
                "displayed in increasing z"
            ),
            "frame_numbers": frame_numbers[selected].tolist(),
            "focus_um": focus_um_all[selected].tolist(),
            "surface_referenced_z_um": (
                focus_um_all[selected] - surface_um
            ).tolist(),
        },
        "holography_propagation_selection": {
            "rule": (
                "five equally spaced planes, with plane 2 at the "
                "focus-transferred LC surface and plane 4 at the "
                "basis-invariant annular focus"
            ),
            "focus_um": publication_focus_um.tolist(),
            "surface_referenced_z_um": (
                publication_focus_um - surface_um
            ).tolist(),
            "step_um": float(publication_focus_um[1] - publication_focus_um[0]),
            "surface_um": surface_um,
            "annular_focus_um": annular_focus_um,
            "wavelength_um": publication_jones_wavelength_um,
        },
        "holography_harmonics": {
            **harmonic_metadata,
            "displayed": ["H0", "abs(H2)", "abs(H4)"],
            "normalization": "none; physical coefficients of the crossed intensity",
            "maximum_radius_um": HARMONIC_MAX_RADIUS_UM,
            "ordinate": {
                "scale": "logarithmic",
                "range_decades": HARMONIC_LOG_DECADES,
                "minimum": "joint displayed maximum divided by 100",
                "maximum": "1.08 times the joint displayed maximum",
            },
            "two_dimensional_audit": harmonic_audit,
        },
        "centering": {
            "direct_centers_xy_px": centers_xy[selected].tolist(),
            "propagated_centers_xy_px": [
                list(center) for center in model_centers
            ],
            "propagated_center_rms_px": center_residuals,
            "holography_propagation_centers_xy_px": [
                list(center) for center in publication_centers
            ],
            "holography_propagation_center_rms_px": (
                publication_center_residuals
            ),
            "method": (
                "subpixel centered interpolation; direct centers are from "
                "the HDR registration and propagated centers are refitted "
                "from radial image gradients"
            ),
        },
        "radiometry": {
            "channel": "I90",
            "direct_normalization": (
                "none in this renderer; canonical HDR was divided once by "
                "a common frame incident level during assembly"
            ),
            "model_calibration": (
                "none; propagated holographic field followed by the coherent "
                "BS2 response diag(1,sqrt(2))"
            ),
            "fitted_intensity_coefficients": "none",
            "per_panel_normalization": False,
            "invalid_pixel_rendering": {
                "analysis_policy": (
                    "pixels invalid in both exposures retain zero confidence "
                    "and are excluded from focus fitting"
                ),
                "display_policy": (
                    "only gaps no farther than 4 px from valid data are "
                    "continued from nearest valid values and Gaussian-smoothed "
                    "inside the gap; canonical HDR arrays are unchanged"
                ),
                "selected_frames": {
                    str(int(frame_numbers[index])): record
                    for index, record in zip(
                        selected,
                        direct_display_repairs,
                        strict=True,
                    )
                },
            },
            "topovec_exposure_mode": (
                "one shared display exposure fixed exclusively by the "
                "propagated holographic row"
            ),
            "holography_display_reference_percentile": (
                HOLOGRAPHY_DISPLAY_PERCENTILE
            ),
            "holography_display_reference_intensity": display_reference,
            "shared_topovec_exposure_ev": comparison_exposure_ev,
            "topovec_tone_curve_mode": "sRGB",
            "holography_figure_detector_response": (
                "none; physical |J_yx|^2 from the normalized holographic "
                "Jones field"
            ),
            "holography_continuation_pupil": (
                "none; the measured field already contains the acquisition pupil"
            ),
            "holography_figure_intensity_percentiles": {
                str(percentile): [
                    float(np.nanpercentile(values, percentile))
                    for values in publication_maps
                ]
                for percentile in (50.0, 95.0, 99.0, 99.9, 100.0)
            },
        },
        "geometry": {
            "direct_pom_wavelength_um": direct_pom_wavelength_um,
            "publication_jones_wavelength_um": (
                publication_jones_wavelength_um
            ),
            "nominal_wavelength_nm_on_figures": 520,
            "pixel_size_um": pixel_size_um,
            "field_width_um_requested": float(args.field_width_um),
            "field_width_um_realized": float(side * pixel_size_um),
            "scale_bar_um": 10.0,
            "direct_spatial_transform": "mirror project x after centering",
            "source_panel_pixels": PANEL_PIXELS,
            "assembly_dpi": FINAL_DPI,
        },
        "outputs": {
            "figure_pdf": _portable_path(figure),
            "figure_png": _portable_path(png_path),
            "topovec_panels": _portable_path(panel_dir),
            "holography_figure_pdf": _portable_path(holography_figure),
            "holography_figure_png": _portable_path(holography_png),
            "holography_topovec_panels": _portable_path(publication_panel_dir),
            "harmonic_audit": {
                "archive": harmonic_audit["archive"],
                "figure": harmonic_audit["figure"],
                "report": harmonic_audit["report"],
            },
        },
    }
    _write_json(output / "focus_i90_comparison_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jones", type=Path, default=DEFAULT_JONES)
    parser.add_argument("--hdr-root", type=Path, default=DEFAULT_HDR)
    parser.add_argument("--focus-fit", type=Path, default=DEFAULT_FOCUS)
    parser.add_argument(
        "--focus-report",
        type=Path,
        default=DEFAULT_FOCUS_REPORT,
    )
    parser.add_argument(
        "--evolution-report",
        type=Path,
        default=DEFAULT_EVOLUTION_REPORT,
    )
    parser.add_argument(
        "--evolution-archive",
        type=Path,
        default=DEFAULT_EVOLUTION_ARCHIVE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--holography-figure",
        type=Path,
        default=DEFAULT_HOLOGRAPHY_FIGURE,
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--field-width-um",
        type=float,
        default=FIELD_WIDTH_UM,
    )
    parser.add_argument("--workers", type=int, default=-1)
    add_force_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    populated_outputs = (
        (output,)
        if output.exists() and any(output.iterdir())
        else ()
    )
    refuse_existing_outputs(
        (*populated_outputs, args.figure, args.holography_figure),
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    report = run(args)
    print(
        "Rendered matched I90 focus planes "
        f"{report['selection']['frame_numbers']} at "
        f"{report['selection']['focus_um']} um."
    )
    print(f"Figure: {report['outputs']['figure_pdf']}")
    print(
        "Holography propagation figure: "
        f"{report['outputs']['holography_figure_pdf']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
