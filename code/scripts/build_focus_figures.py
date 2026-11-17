#!/usr/bin/env python3
"""Recompose through-focus publication figures from persisted compact caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from toronoptics.figure_output import (
    OPTICAL_ANNOTATION_COLOR,
    expand_focus_panels,
    figure_output_dir,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs


ROOT = Path(__file__).resolve().parents[2]
FIGS = figure_output_dir(ROOT)
ANALYSIS = ROOT / "data/analysis/results_toron25um_50X_focus_recovery"
CACHE = ROOT / "data/analysis/results_toron25um_50X_focus_recovery_soft"
EVOLUTION = ROOT / "data/analysis/axisymmetric_jones_evolution_soft"
FINAL_DPI = 400
GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5.0))
COLORS = ("#242424", "#704C8F", "#B47A0B")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def _rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def _focus_diagnostics() -> None:
    with np.load(ANALYSIS / "focus_fit.npz", allow_pickle=False) as data:
        frame = np.asarray(data["frame_numbers"])
        focus_grid = np.asarray(data["focus_grid_um"])
        shape_cost = np.asarray(data["focus_cost"])
        focus = np.asarray(data["recovered_focus_um"])
        local = np.asarray(data["local_focus_minimum_um"])
        radial_controls = np.asarray(data["radial_control_focus_um"])
        channel_controls = np.asarray(data["channel_control_focus_um"])

    relative = shape_cost - np.min(shape_cost, axis=1, keepdims=True)
    relative /= np.maximum(
        np.percentile(relative, 35.0, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    extent = (
        focus_grid[0],
        focus_grid[-1],
        frame[0] - 0.5,
        frame[-1] + 0.5,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.55),
        gridspec_kw={"width_ratios": (1.25, 1.0)},
    )
    image = axes[0].imshow(
        np.clip(relative, 0.0, 4.0), origin="lower", aspect="auto",
        extent=extent, cmap="magma_r", interpolation="nearest", rasterized=True,
    )
    axes[0].plot(
        focus_grid[np.argmin(shape_cost, axis=1)],
        frame,
        "o",
        ms=3.0,
        color="#2b8cbe",
        label="profile minimum",
    )
    axes[0].plot(
        focus,
        frame,
        "-",
        lw=1.1,
        color="#d7301f",
        label=r"$10\,\mathregular{\mu}$m spacing",
    )
    axes[0].axvline(0.0, color="white", lw=0.8, ls=":")
    axes[0].set_xlabel(r"Reconstruction coordinate $\zeta$ ($\mathregular{\mu}$m)")
    axes[0].set_ylabel("Source frame")
    axes[0].set_yticks(frame)
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.85,
        fontsize=10,
    )
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.047, pad=0.025)
    colorbar.set_label("Row-normalized mismatch", fontsize=10)
    colorbar.ax.tick_params(labelsize=10)

    axes[1].plot(
        frame,
        local,
        "o",
        ms=4.0,
        color="#242424",
        label=r"joint $I_0,I_{90}$",
        zorder=4,
    )
    for values, label, color in zip(
        radial_controls,
        (
            r"$\rho=2$--$25\,\mathregular{\mu}$m",
            r"$8$--$32\,\mathregular{\mu}$m",
            r"$15$--$39\,\mathregular{\mu}$m",
        ),
        ("#8c6bb1", "#2b8cbe", "#31a354"),
        strict=True,
    ):
        axes[1].plot(frame, values, ".", ms=4.2, color=color, label=label)
    for values, label, color in zip(
        channel_controls,
        (r"$I_0$", r"$I_{90}$"),
        ("#e6550d", "#756bb1"),
        strict=True,
    ):
        axes[1].plot(
            frame,
            values,
            "x",
            ms=4.0,
            mew=0.8,
            color=color,
            label=label,
        )
    axes[1].plot(frame, focus, "-", lw=0.9, color="0.45", zorder=1)
    axes[1].set_xlabel("Source frame")
    axes[1].set_ylabel(r"Recovered $\zeta$ ($\mathregular{\mu}$m)")
    axes[1].set_xticks(frame)
    axes[1].legend(
        frameon=False,
        fontsize=10,
        ncols=2,
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.35,
        columnspacing=0.7,
    )
    for label, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.text(
            -0.10,
            -0.19,
            label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            clip_on=False,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=10)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.19,
        top=0.72,
        wspace=0.58,
    )
    fig.savefig(FIGS / "focus_cost_map.pdf", dpi=FINAL_DPI)
    plt.close(fig)


def _direct_comparison(report: dict) -> None:
    frames = report["selection"]["frame_numbers"]
    z = np.asarray(report["selection"]["surface_referenced_z_um"])
    panels = CACHE / "topovec_i90_focus_panels"
    fig, axes = plt.subplots(2, 5, figsize=(7.15, 3.02))
    fig.subplots_adjust(left=0.070, right=0.997, bottom=0.018, top=0.895, wspace=0.012, hspace=0.018)
    for col, (frame, coordinate) in enumerate(zip(frames, z, strict=True)):
        axes[0, col].set_title(rf"$z={coordinate:.2f}\,\mathregular{{\mu}}\mathrm{{m}}$", fontsize=10.0, pad=2.5)
        for row, suffix in enumerate(("direct", "propagated")):
            axes[row, col].imshow(
                _rgba(panels / f"frame_{frame:04d}_{suffix}.png"),
                interpolation="nearest",
            )
            axes[row, col].set_xticks(())
            axes[row, col].set_yticks(())
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
    for row, label in enumerate(("direct polarimetry", "propagated holography")):
        axes[row, 0].text(-0.065, 0.5, label, transform=axes[row, 0].transAxes, rotation=90, ha="right", va="center", fontsize=10.0, clip_on=False)
    fig.savefig(FIGS / "direct_polarimetry_focus_i90_520_soft.pdf", transparent=True, bbox_inches="tight", pad_inches=0.015, dpi=FINAL_DPI)
    plt.close(fig)


def _main_focus(report: dict) -> None:
    z = np.asarray(report["holography_propagation_selection"]["surface_referenced_z_um"])
    panels = [_rgba(CACHE / "topovec_i90_holography_propagation" / f"plane_{index}_propagated.png") for index in range(1, 6)]
    with np.load(EVOLUTION / "axisymmetric_jones_evolution.npz", allow_pickle=False) as data:
        radius = np.asarray(data["exact_sample_radius_um"], dtype=float)
        h0 = np.asarray(data["exact_sample_physical_crossed_H0"], dtype=float)
        h2 = np.abs(np.asarray(data["exact_sample_physical_crossed_H2"]))
        h4 = np.abs(np.asarray(data["exact_sample_physical_crossed_H4"]))
    harmonics = np.stack((h0, h2, h4), axis=1)
    font_size = 9.05
    fig = plt.figure(figsize=(7.15, 2.85))
    grid = fig.add_gridspec(2, 5, height_ratios=(GOLDEN_RATIO, 1.0), left=0.070, right=0.990, bottom=0.210, top=0.860, wspace=0.16, hspace=0.03)
    image_axes = [fig.add_subplot(grid[0, col]) for col in range(5)]
    profile_axes = []
    for col in range(5):
        shared = profile_axes[0] if profile_axes else None
        profile_axes.append(fig.add_subplot(grid[1, col], sharex=shared, sharey=shared))
    for col, (coordinate, panel) in enumerate(zip(z, panels, strict=True)):
        landmark = {1: "LC surface", 3: "annular focus"}.get(col, "")
        image_axes[col].set_title((landmark + "\n" if landmark else "\n") + rf"$z={coordinate:.2f}\,\mathregular{{\mu}}\mathrm{{m}}$", fontsize=font_size, pad=2.0, linespacing=0.95)
        image_axes[col].imshow(panel, interpolation="nearest")
        image_axes[col].text(0.04, 0.95, f"({chr(97 + col)})", transform=image_axes[col].transAxes, ha="left", va="top", fontsize=font_size, color=OPTICAL_ANNOTATION_COLOR)
        image_axes[col].set_xticks(())
        image_axes[col].set_yticks(())
        for spine in image_axes[col].spines.values():
            spine.set_visible(False)
    shown = radius <= 30.0
    upper = 1.08 * float(np.nanmax(harmonics[:, :, shown]))
    lower = upper / 100.0
    labels = (r"$H_0$", r"$|H_2|$", r"$|H_4|$")
    for col, axis in enumerate(profile_axes):
        for mode, (label, color) in enumerate(zip(labels, COLORS, strict=True)):
            values = harmonics[col, mode, shown]
            axis.plot(radius[shown], values, color=color, lw=0.85, label=label, zorder=3)
            if mode:
                axis.fill_between(radius[shown], lower, np.maximum(values, lower), color=color, alpha=0.16, linewidth=0, zorder=1)
        axis.text(0.96, 0.94, f"({chr(102 + col)})", transform=axis.transAxes, ha="right", va="top", fontsize=font_size)
        axis.set_box_aspect(1.0 / GOLDEN_RATIO)
        axis.set_yscale("log")
        axis.set_xlim(0, 30)
        axis.set_ylim(lower, upper)
        axis.set_xticks((0, 15, 30))
        axis.tick_params(labelsize=font_size, length=2, pad=2, labelleft=col == 0)
        axis.grid(axis="y", which="major", color="0.85", lw=0.6)
        if col == 0:
            axis.set_ylabel(r"$H_m$", fontsize=font_size, labelpad=2)
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
    fig.savefig(FIGS / "holography_focus_i90_520_soft.pdf", transparent=True, bbox_inches=bounds, dpi=FINAL_DPI)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    args = parser.parse_args(argv)
    outputs = (
        FIGS / "focus_cost_map.pdf",
        FIGS / "direct_polarimetry_focus_i90_520_soft.pdf",
        FIGS / "holography_focus_i90_520_soft.pdf",
    )
    refuse_existing_outputs(
        outputs,
        force=args.force,
        project_root=ROOT,
    )
    FIGS.mkdir(parents=True, exist_ok=True)
    report = json.loads((CACHE / "focus_i90_comparison_report.json").read_text(encoding="utf-8"))
    _focus_diagnostics()
    _direct_comparison(report)
    _main_focus(report)


if __name__ == "__main__":
    main()
