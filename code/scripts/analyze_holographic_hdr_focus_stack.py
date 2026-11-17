#!/usr/bin/env python3
"""Fit the measured HDR focus stack to a propagated holographic Jones field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from toronoptics.article_data import decimate_centered_jones
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.focus_recovery_hdr import (
    CHANNELS,
    HDRRadialStack,
    fit_angle_and_trajectory,
    fit_trajectory_control,
    load_hdr_radial_stack,
    local_branch_metrics,
)
from toronoptics.focus_stack import (
    ANALYZER_ANGLES_DEG,
    AngularSpectrumJonesPropagator,
    radial_jones_gram,
)
from toronoptics.source_config import (
    DEFAULT_EXPERIMENT_MANIFEST,
    load_direct_pom_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JONES = PROJECT_ROOT / "data/experimental/results_toron25um_50X_averaged/520nm_average.npz"
DEFAULT_HDR = PROJECT_ROOT / "data/experimental/results_toron25um_50X_focus_hdr"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/analysis/results_toron25um_50X_focus_recovery"
DEFAULT_PUPIL_REPORT = (
    PROJECT_ROOT / "data/analysis/averaged_pupil_spectrum/report.json"
)
DIRECT_POM_CONFIG = load_direct_pom_config()
DIRECT_PIXEL_SIZE_UM = DIRECT_POM_CONFIG.pixel_size_um
WAVELENGTH_UM = DIRECT_POM_CONFIG.wavelength_um
RADII_UM = np.arange(0.5, 44.01, 0.25)
FIT_INTERVAL_UM = (2.0, 39.0)
FOCUS_GRID_UM = np.arange(-100.0, 100.01, 2.0)
FIT_CHANNELS = ("I0", "I90")
BS2_OUTPUT_RATIO = DIRECT_POM_CONFIG.beamsplitter_amplitude_ratio
INPUT_ANGLE_DEG = 0.0
RADIAL_CONTROLS = {
    "inner": (2.0, 25.0),
    "middle": (8.0, 32.0),
    "outer": (15.0, 39.0),
}


def _measured_effective_na(
    report_path: Path = DEFAULT_PUPIL_REPORT,
    *,
    wavelength_nm: int = 520,
) -> float:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        value = float(
            report["per_wavelength"][str(int(wavelength_nm))][
                "source_median_effective_na"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{report_path} does not contain the measured {wavelength_nm}-nm pupil"
        ) from exc
    if not 0.0 < value <= 1.0:
        raise ValueError(f"invalid measured coherent NA {value!r} in {report_path}")
    return value


def _relative(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_decimated_jones(path: Path) -> tuple[np.ndarray, tuple[float, float], float]:
    archive = load_jones_archive(path, dtype=np.complex64)
    with np.load(path, allow_pickle=False) as data:
        center = (
            tuple(float(value) for value in np.asarray(data["array_center_px"]).reshape(2))
            if "array_center_px" in data.files
            else (0.5 * (archive.jones.shape[0] - 1), 0.5 * (archive.jones.shape[1] - 1))
        )
    reduced, reduced_center = decimate_centered_jones(archive.jones, center, factor=2)
    pixel_size = 2.0 * archive.pixel_size_um
    if not np.isclose(pixel_size, DIRECT_PIXEL_SIZE_UM, atol=5.0e-5):
        raise ValueError(f"Jones grid {pixel_size} um does not match direct grid {DIRECT_PIXEL_SIZE_UM} um")
    return reduced, reduced_center, pixel_size


def _cache_metadata(args: argparse.Namespace, source: Path, pixel_size_um: float) -> dict[str, Any]:
    stat = source.stat()
    return {
        "schema": "focus-radial-gram-v1",
        "source": _relative(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "pixel_size_um": pixel_size_um,
        "wavelength_um": args.wavelength_um,
        "numerical_aperture": args.numerical_aperture,
        "padding_factor": args.padding_factor,
        "focus_grid_um": [args.focus_min_um, args.focus_max_um, args.focus_step_um],
        "radii_um": [float(RADII_UM[0]), float(RADII_UM[-1]), float(np.median(np.diff(RADII_UM)))],
    }


def _radial_gram_bank(
    propagator: AngularSpectrumJonesPropagator,
    *,
    center_xy: tuple[float, float],
    focus_grid_um: np.ndarray,
    cache_path: Path,
    metadata: dict[str, Any],
    force: bool,
) -> np.ndarray:
    encoded = json.dumps(metadata, sort_keys=True)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            if (
                str(np.asarray(data["metadata_json"]).item()) == encoded
                and np.array_equal(np.asarray(data["focus_grid_um"]), focus_grid_um)
                and np.array_equal(np.asarray(data["radii_um"]), RADII_UM)
            ):
                print(f"[propagation] reuse {cache_path}")
                return np.asarray(data["radial_gram"])
    grams = np.empty((focus_grid_um.size, RADII_UM.size, 4, 4), dtype=np.complex64)
    start = time.monotonic()
    for index, focus in enumerate(focus_grid_um):
        grams[index] = radial_jones_gram(
            propagator.propagate_jones(float(focus)),
            center_xy=center_xy,
            radii_px=RADII_UM / propagator.pixel_size_um,
            angle_count=240,
        )
        if index % 25 == 0 or index + 1 == focus_grid_um.size:
            print(f"[propagation] {index + 1}/{focus_grid_um.size}, {time.monotonic() - start:.1f} s")
    np.savez_compressed(
        cache_path,
        radial_gram=grams,
        focus_grid_um=focus_grid_um,
        radii_um=RADII_UM,
        metadata_json=np.asarray(encoded),
    )
    return grams


def _robust_line(frames: np.ndarray, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(values)
    initial = np.polyfit(frames[selected], values[selected], 1)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return parameters[0] * frames[selected] + parameters[1] - values[selected]

    fit = least_squares(residual, initial, loss="soft_l1", f_scale=3.0).x
    return fit, values - np.polyval(fit, frames)


def _center_jump(centers_xy_px: np.ndarray) -> tuple[int, float, np.ndarray]:
    steps = np.linalg.norm(np.diff(centers_xy_px, axis=0), axis=1)
    threshold = max(8.0, 8.0 * float(np.median(steps)))
    candidates = np.flatnonzero(steps > threshold)
    onset = int(candidates[0] + 1) if candidates.size else centers_xy_px.shape[0]
    return onset, threshold, steps


def _interpolate_profile_bank(
    predicted: np.ndarray,
    focus_grid_um: np.ndarray,
    focus_um: np.ndarray,
) -> np.ndarray:
    result = np.empty((focus_um.size, predicted.shape[1], predicted.shape[2]))
    for frame_index, focus in enumerate(focus_um):
        for channel_index in range(predicted.shape[1]):
            for radius_index in range(predicted.shape[2]):
                result[frame_index, channel_index, radius_index] = np.interp(
                    focus,
                    focus_grid_um,
                    predicted[:, channel_index, radius_index],
                )
    return result


def _profile_agreement(
    observed: HDRRadialStack,
    predicted: np.ndarray,
    *,
    radial_interval_um: tuple[float, float],
) -> dict[str, np.ndarray]:
    radial_mask = (
        (observed.radii_um >= radial_interval_um[0])
        & (observed.radii_um <= radial_interval_um[1])
    )
    shape = observed.profiles.shape[:2]
    relative_rms = np.empty(shape, dtype=np.float64)
    correlation = np.empty(shape, dtype=np.float64)
    for frame_index in range(shape[0]):
        for channel_index in range(shape[1]):
            measured = observed.profiles[frame_index, channel_index, radial_mask]
            model = predicted[frame_index, channel_index, radial_mask]
            weight = (
                observed.radii_um[radial_mask]
                * observed.coverage[frame_index, channel_index, radial_mask] ** 2
            )
            valid = np.isfinite(measured) & np.isfinite(model) & (weight > 0.0)
            y = measured[valid]
            x = model[valid]
            w = weight[valid]
            relative_rms[frame_index, channel_index] = np.sqrt(
                np.sum(w * (y - x) ** 2)
                / max(np.sum(w * y**2), np.finfo(float).eps)
            )
            y_centered = y - np.sum(w * y) / np.sum(w)
            x_centered = x - np.sum(w * x) / np.sum(w)
            correlation[frame_index, channel_index] = np.sum(w * x_centered * y_centered) / max(
                np.sqrt(np.sum(w * x_centered**2) * np.sum(w * y_centered**2)),
                np.finfo(float).eps,
            )
    return {"relative_rms": relative_rms, "correlation": correlation}


def _extract_centered(image_yx: np.ndarray, center_xy: tuple[float, float], side: int) -> np.ndarray:
    coordinate = np.arange(side, dtype=np.float64) - 0.5 * (side - 1)
    x = float(center_xy[0]) + coordinate[None, :]
    y = float(center_xy[1]) + coordinate[:, None]
    return map_coordinates(
        np.asarray(image_yx, dtype=np.float64),
        (np.broadcast_to(y, (side, side)), np.broadcast_to(x, (side, side))),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, -0.15, label, transform=ax.transAxes, ha="right", va="top", fontsize=9, clip_on=False)


def _plot_cost_maps(
    output: Path,
    observed: HDRRadialStack,
    focus_grid: np.ndarray,
    shape_cost: np.ndarray,
    absolute_cost: np.ndarray,
    focus_um: np.ndarray,
) -> None:
    relative = shape_cost - np.min(shape_cost, axis=1, keepdims=True)
    row_scale = np.percentile(relative, 35.0, axis=1, keepdims=True)
    relative /= np.maximum(row_scale, np.finfo(float).eps)
    extent = (focus_grid[0], focus_grid[-1], observed.frame_numbers[0], observed.frame_numbers[-1])
    for name, values, label, limits in (
        ("focus_cost_map", relative, "Row-normalized absolute mismatch", (0.0, 4.0)),
        ("focus_cost_map_absolute", absolute_cost, "Absolute relative squared error", (0.0, float(np.nanpercentile(absolute_cost, 92.0)))),
    ):
        fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
        image = ax.imshow(
            np.clip(values, limits[0], limits[1]),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma_r",
            interpolation="nearest",
            rasterized=True,
        )
        local = focus_grid[np.argmin(shape_cost if name == "focus_cost_map" else absolute_cost, axis=1)]
        ax.plot(local, observed.frame_numbers, ".", ms=2.5, color="#2b8cbe", label="framewise minimum")
        ax.plot(
            focus_um,
            observed.frame_numbers,
            "-",
            lw=1.3,
            color="#d7301f",
            label="equal-step fit",
        )
        ax.axvline(0.0, color="white", lw=0.8, ls=":")
        ax.set_xlabel(r"Reconstruction-plane coordinate $\zeta$ ($\mu$m)")
        ax.set_ylabel("Frame index")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=7)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(label)
        fig.savefig(output / f"{name}.pdf")
        fig.savefig(output / f"{name}.png", dpi=300)
        plt.close(fig)


def _plot_trajectory(
    output: Path,
    observed: HDRRadialStack,
    focus_um: np.ndarray,
    local_minimum_um: np.ndarray,
    reliable: np.ndarray,
    control_trajectories: dict[str, np.ndarray],
) -> None:
    frames = observed.frame_numbers
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    axes[0, 0].plot(frames, focus_um, "-", lw=1.2, color="#2166ac", label="HDR fit")
    axes[0, 0].plot(frames, local_minimum_um, ".", ms=3.0, color="#d7301f", label="local data minimum")
    axes[0, 0].plot(frames[~reliable], focus_um[~reliable], "o", ms=3, mfc="white", mec="#2166ac")
    axes[0, 0].set_xlabel("Frame index")
    axes[0, 0].set_ylabel(r"Recovered $\zeta$ ($\mu$m)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].plot(frames, local_minimum_um - focus_um, "o-", ms=2.3, lw=0.9, color="#1b7837")
    axes[0, 1].axhline(0.0, color="black", lw=0.8, ls="--")
    axes[0, 1].set_xlabel("Frame index")
    axes[0, 1].set_ylabel(r"Local minimum minus fit ($\mu$m)")
    axes[1, 0].plot(frames, observed.centers_xy_px[:, 0], lw=1.0, label="x", color="#b2182b")
    axes[1, 0].plot(frames, observed.centers_xy_px[:, 1], lw=1.0, label="y", color="#4d4d4d")
    axes[1, 0].set_xlabel("Frame index")
    axes[1, 0].set_ylabel("Symmetry center (px)")
    axes[1, 0].legend(frameon=False)
    for name, values in control_trajectories.items():
        axes[1, 1].plot(frames, values - focus_um, lw=0.9, label=name)
    axes[1, 1].axhline(0.0, color="black", lw=0.7)
    axes[1, 1].set_xlabel("Frame index")
    axes[1, 1].set_ylabel(r"Control minus production $\zeta$ ($\mu$m)")
    axes[1, 1].legend(frameon=False, fontsize=6, ncol=2)
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel(), strict=True):
        _panel_label(ax, label)
    fig.savefig(output / "focus_trajectory.pdf")
    fig.savefig(output / "focus_trajectory.png", dpi=300)
    plt.close(fig)


def _plot_profiles(
    output: Path,
    observed: HDRRadialStack,
    model: np.ndarray,
    focus_um: np.ndarray,
) -> None:
    # Frames after 61 contain the documented accidental lateral displacement
    # and are retained in the numerical archive but not used as representative
    # visual matches.
    selected = np.unique(np.linspace(0, min(observed.frame_numbers.size - 1, 61), 6).round().astype(int))
    fig, axes = plt.subplots(2, selected.size, figsize=(10.0, 4.0), sharex=True, constrained_layout=True)
    for column, frame_index in enumerate(selected):
        for row, name in enumerate(FIT_CHANNELS):
            channel = CHANNELS.index(name)
            axes[row, column].plot(observed.radii_um, observed.profiles[frame_index, channel], lw=0.9, color="#333333", label="HDR" if row == column == 0 else None)
            axes[row, column].plot(observed.radii_um, model[frame_index, channel], lw=0.9, color="#d7301f", label="propagated" if row == column == 0 else None)
            axes[row, column].axvspan(FIT_INTERVAL_UM[0], FIT_INTERVAL_UM[1], color="#eeeeee", zorder=-3)
            axes[row, column].set_title(f"{int(observed.frame_numbers[frame_index])}, z={focus_um[frame_index]:.1f}", fontsize=8)
            if column == 0:
                axes[row, column].set_ylabel(name)
            if row == 1:
                axes[row, column].set_xlabel(r"$\rho$ ($\mu$m)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(output / "radial_profile_matches.pdf")
    fig.savefig(output / "radial_profile_matches.png", dpi=300)
    plt.close(fig)


def _plot_closest_channels(
    output: Path,
    hdr_root: Path,
    observed: HDRRadialStack,
    propagated_jones: np.ndarray,
    jones_center: tuple[float, float],
    frame_index: int,
    angle_deg: float,
    beamsplitter_output_ratio: complex,
    side: int = 441,
) -> None:
    frame_number = int(observed.frame_numbers[frame_index])
    with np.load(hdr_root / "frames" / f"{frame_number:04d}.npz", allow_pickle=False) as data:
        intensity = np.asarray(data["intensity_yxc"], dtype=np.float64)
        source_channels = tuple(str(value) for value in data["channel_names"])
    alpha = np.radians(angle_deg)
    field = propagated_jones @ np.asarray((np.cos(alpha), np.sin(alpha)), dtype=np.complex64)
    field = np.asarray(field).copy()
    field[..., 1] *= complex(beamsplitter_output_ratio)
    yy, xx = np.indices((side, side), dtype=np.float64)
    mask = np.hypot(xx - 0.5 * (side - 1), yy - 0.5 * (side - 1)) <= FIT_INTERVAL_UM[1] / DIRECT_PIXEL_SIZE_UM
    residuals = []
    panels = []
    for name in CHANNELS:
        source_index = source_channels.index(name)
        direct_full = intensity[..., source_index]
        direct = _extract_centered(direct_full, tuple(observed.centers_xy_px[frame_index]), side)
        beta = np.radians(ANALYZER_ANGLES_DEG[name])
        analyzer = np.asarray((np.cos(beta), np.sin(beta)))
        predicted_xy = np.abs(field @ analyzer) ** 2
        model = _extract_centered(predicted_xy.T, jones_center, side)
        residual = direct - model
        residuals.append(residual)
        panels.append((direct, model))
    residual_limit = float(np.nanpercentile(np.abs(np.stack(residuals)[:, mask]), 99.0))
    fig, axes = plt.subplots(3, len(CHANNELS), figsize=(8.0, 6.0), constrained_layout=True)
    for column, name in enumerate(CHANNELS):
        direct, model = panels[column]
        valid = mask & np.isfinite(direct) & np.isfinite(model)
        limits = np.nanpercentile(np.concatenate((direct[valid], model[valid])), (1.0, 99.0))
        axes[0, column].imshow(direct, cmap="gray", vmin=limits[0], vmax=limits[1], origin="lower")
        axes[1, column].imshow(model, cmap="gray", vmin=limits[0], vmax=limits[1], origin="lower")
        axes[2, column].imshow(residuals[column], cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit, origin="lower")
        axes[0, column].set_title(name)
        for row in range(3):
            axes[row, column].set_axis_off()
    for row, label in enumerate(("HDR", "propagated", "difference")):
        axes[row, 0].text(-0.05, 0.5, label, transform=axes[row, 0].transAxes, rotation=90, ha="right", va="center", fontsize=8)
    fig.savefig(output / "closest_focus_channels.pdf")
    fig.savefig(output / "closest_focus_channels.png", dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    jones_path = args.jones.expanduser().resolve()
    hdr_root = args.hdr_root.expanduser().resolve()
    focus_grid = np.arange(args.focus_min_um, args.focus_max_um + 0.5 * args.focus_step_um, args.focus_step_um)
    jones, jones_center, pixel_size = _load_decimated_jones(jones_path)
    observed = load_hdr_radial_stack(
        hdr_root,
        radii_um=RADII_UM,
        pixel_size_um=DIRECT_PIXEL_SIZE_UM,
    )
    propagator = AngularSpectrumJonesPropagator(
        jones,
        pixel_size_um=pixel_size,
        wavelength_um=args.wavelength_um,
        numerical_aperture=args.numerical_aperture,
        padding_factor=args.padding_factor,
        workers=args.workers,
    )
    grams = _radial_gram_bank(
        propagator,
        center_xy=jones_center,
        focus_grid_um=focus_grid,
        cache_path=output / "propagation_cache.npz",
        metadata=_cache_metadata(args, jones_path, pixel_size),
        force=args.force,
    )
    production = fit_angle_and_trajectory(
        observed,
        grams,
        focus_grid,
        nominal_step_um=args.nominal_stage_step_um,
        radial_interval_um=FIT_INTERVAL_UM,
        channels=FIT_CHANNELS,
        beamsplitter_output_ratio=BS2_OUTPUT_RATIO,
        fixed_input_angle_deg=INPUT_ANGLE_DEG,
        fit_affine=False,
    )
    print(f"[fit] input angle {production.input_angle_deg:.4f} deg")

    controls: dict[str, np.ndarray] = {}
    control_surfaces: dict[str, np.ndarray] = {}
    for name, interval in RADIAL_CONTROLS.items():
        surface, trajectory = fit_trajectory_control(
            observed,
            production.predicted_profiles,
            focus_grid,
            nominal_step_um=args.nominal_stage_step_um,
            radial_interval_um=interval,
            channels=FIT_CHANNELS,
        )
        label = f"rho {interval[0]:g}-{interval[1]:g}"
        controls[label] = trajectory.focus_um
        control_surfaces[label] = surface
    for name in FIT_CHANNELS:
        surface, trajectory = fit_trajectory_control(
            observed,
            production.predicted_profiles,
            focus_grid,
            nominal_step_um=args.nominal_stage_step_um,
            radial_interval_um=FIT_INTERVAL_UM,
            channels=(name,),
        )
        controls[name] = trajectory.focus_um
        control_surfaces[name] = surface
    branch = local_branch_metrics(production.cost, focus_grid, production.trajectory.focus_um)
    jump_onset, jump_threshold, center_steps = _center_jump(observed.centers_xy_px)
    preliminary_control_minima = {
        name: local_branch_metrics(surface, focus_grid, production.trajectory.focus_um)["local_minimum_um"]
        for name, surface in control_surfaces.items()
    }
    radial_minima = np.stack(
        [values for name, values in preliminary_control_minima.items() if name.startswith("rho ")]
    )
    channel_minima = np.stack((preliminary_control_minima["I0"], preliminary_control_minima["I90"]))
    radial_span = np.ptp(radial_minima, axis=0)
    channel_disagreement = np.abs(channel_minima[0] - channel_minima[1])
    reliable = (
        (branch["prior_pull_um"] <= 6.0)
        & (radial_span <= 12.0)
        & (channel_disagreement <= 12.0)
        & (branch["alias_gap"] > 0.002)
        & (np.arange(branch["prior_pull_um"].size) < jump_onset)
    )
    line_mask = reliable
    line, line_residual = _robust_line(observed.frame_numbers, branch["local_minimum_um"], line_mask)
    focus_um = np.polyval(line, observed.frame_numbers)

    # Re-evaluate support around the physically appropriate equal-step line.
    branch = local_branch_metrics(production.cost, focus_grid, focus_um)
    control_minima = {
        name: local_branch_metrics(surface, focus_grid, focus_um)["local_minimum_um"]
        for name, surface in control_surfaces.items()
    }
    radial_minima = np.stack(
        [values for name, values in control_minima.items() if name.startswith("rho ")]
    )
    channel_minima = np.stack((control_minima["I0"], control_minima["I90"]))
    radial_span = np.ptp(radial_minima, axis=0)
    channel_disagreement = np.abs(channel_minima[0] - channel_minima[1])
    reliable = (
        (branch["prior_pull_um"] <= 6.0)
        & (radial_span <= 12.0)
        & (channel_disagreement <= 12.0)
        & (branch["alias_gap"] > 0.002)
        & (np.arange(branch["prior_pull_um"].size) < jump_onset)
    )
    line_mask = reliable
    line, line_residual = _robust_line(observed.frame_numbers, branch["local_minimum_um"], line_mask)
    focus_um = np.polyval(line, observed.frame_numbers)

    control_reports: dict[str, Any] = {}
    control_lines: dict[str, np.ndarray] = {}
    for name, minima in control_minima.items():
        control_line, _control_residual = _robust_line(observed.frame_numbers, minima, line_mask)
        values = np.polyval(control_line, observed.frame_numbers)
        control_lines[name] = values
        difference = values - focus_um
        report_name = (
            f"channel_{name}"
            if name in FIT_CHANNELS
            else "radial_" + name.split()[1].replace("-", "_")
        )
        control_reports[report_name] = {
            "label": name,
            "robust_line_slope_um_per_frame": float(control_line[0]),
            "robust_line_intercept_um": float(control_line[1]),
            "median_line_difference_um": float(np.median(difference)),
            "maximum_line_difference_um": float(np.max(np.abs(difference))),
            "local_minimum_um": minima.tolist(),
        }

    interpolated_model = _interpolate_profile_bank(
        production.predicted_profiles,
        focus_grid,
        focus_um,
    )
    agreement = _profile_agreement(
        observed,
        interpolated_model,
        radial_interval_um=FIT_INTERVAL_UM,
    )
    absolute_cost = production.cost
    control_step_sd = float(
        np.std(
            [
                values["robust_line_slope_um_per_frame"]
                for values in control_reports.values()
            ],
            ddof=1,
        )
    )

    closest_index = int(np.argmin(np.abs(focus_um)))
    closest_jones = propagator.propagate_jones(float(focus_um[closest_index]))
    input_state = np.asarray(
        (np.cos(np.radians(production.input_angle_deg)), np.sin(np.radians(production.input_angle_deg))),
        dtype=np.complex64,
    )
    spectral_slope = np.asarray(propagator.transverse_momentum(input_state))
    used_center_count = jump_onset if jump_onset >= 3 else observed.frame_numbers.size
    center_design = np.column_stack((focus_um[:used_center_count], np.ones(used_center_count)))
    center_coefficients = np.column_stack(
        [np.linalg.lstsq(center_design, observed.centers_xy_px[:used_center_count, axis], rcond=None)[0] for axis in range(2)]
    )
    center_prediction = center_design @ center_coefficients
    center_residual = observed.centers_xy_px[:used_center_count] - center_prediction
    center_slope_dimensionless = center_coefficients[0] * DIRECT_PIXEL_SIZE_UM

    _plot_cost_maps(output, observed, focus_grid, production.cost, absolute_cost, focus_um)
    _plot_trajectory(output, observed, focus_um, branch["local_minimum_um"], reliable, control_lines)
    _plot_profiles(output, observed, interpolated_model, focus_um)
    _plot_closest_channels(
        output,
        hdr_root,
        observed,
        closest_jones,
        jones_center,
        closest_index,
        production.input_angle_deg,
        BS2_OUTPUT_RATIO,
    )

    np.savez_compressed(
        output / "focus_fit.npz",
        frame_numbers=observed.frame_numbers,
        recovered_focus_um=focus_um,
        preliminary_smooth_focus_um=production.trajectory.focus_um,
        focus_grid_um=focus_grid,
        focus_cost=production.cost.astype(np.float32),
        absolute_cost=absolute_cost.astype(np.float32),
        centers_xy_px=observed.centers_xy_px,
        radial_profiles=observed.profiles.astype(np.float32),
        radial_coverage=observed.coverage.astype(np.float32),
        radii_um=observed.radii_um,
        incident_levels=observed.incident_levels,
        input_angle_deg=np.asarray(production.input_angle_deg),
        input_angle_scan_deg=production.angle_scan_deg,
        input_angle_scan_loss=production.angle_scan_loss,
        local_focus_minimum_um=branch["local_minimum_um"],
        reliable_focus_mask=reliable,
        radial_control_focus_um=np.stack(
            [values for name, values in control_lines.items() if name.startswith("rho ")]
        ),
        channel_control_focus_um=np.stack((control_lines["I0"], control_lines["I90"])),
        profile_relative_rms=agreement["relative_rms"],
        profile_correlation=agreement["correlation"],
    )

    report = {
        "schema_version": "holographic-hdr-focus-stack-v4",
        "sources": {
            "experiment_manifest": _relative(DEFAULT_EXPERIMENT_MANIFEST),
            "pupil_report": _relative(args.pupil_report),
            "jones": _relative(jones_path),
            "hdr_stack": _relative(hdr_root),
            "hdr_manifest": _relative(hdr_root / "manifest.json"),
            "frame_count": int(observed.frame_numbers.size),
        },
        "optical_model": {
            "method": "band-limited angular-spectrum propagation of the localized Jones perturbation with analytic plane-wave background",
            "wavelength_um": args.wavelength_um,
            "external_refractive_index": 1.0,
            "propagation_coordinate": "effective scalar image-space defocus relative to the holographic reconstruction plane",
            "direct_pixel_size_um": DIRECT_PIXEL_SIZE_UM,
            "jones_pixel_size_after_decimation_um": pixel_size,
            "numerical_aperture": args.numerical_aperture,
            "padding_factor": args.padding_factor,
            "beamsplitter_output_field_ratio": {
                "real": BS2_OUTPUT_RATIO,
                "imag": 0.0,
                "model": "diag(1,q) before the four detector analyzers",
            },
            "focus_grid_um": [float(focus_grid[0]), float(focus_grid[-1]), float(np.median(np.diff(focus_grid)))],
            "radial_fit_interval_um": list(FIT_INTERVAL_UM),
        },
        "radiometry": {
            "input": "canonical dimensionless 12/42-ms HDR frames in the project basis",
            "normalization": "performed once during HDR assembly by one common robust I0 incident level per frame; no Ground division, channel gain, scale, or offset is applied here",
            "confidence": "per-pixel HDR confidence enters twice: the radial mean and the squared profile weight",
            "median_incident_level": float(np.median(observed.incident_levels)),
            "incident_level_range": [float(np.min(observed.incident_levels)), float(np.max(observed.incident_levels))],
            "median_confidence_by_channel": {name: float(np.median(observed.mean_confidence[:, index])) for index, name in enumerate(CHANNELS)},
            "median_short_exposure_fraction_by_channel": {name: float(np.median(observed.low_exposure_fraction[:, index])) for index, name in enumerate(CHANNELS)},
            "fitted_intensity_coefficients": "none",
        },
        "input_polarization": {
            "model": "linear",
            "angle_deg_in_holographic_xy_basis": production.input_angle_deg,
            "constraint": "fixed to the independently specified horizontal experimental input",
        },
        "focus_recovery": {
            "frame_numbers": observed.frame_numbers.tolist(),
            "focus_um_relative_to_holographic_plane": focus_um.tolist(),
            "preliminary_smooth_focus_um": production.trajectory.focus_um.tolist(),
            "local_profile_minima_um": branch["local_minimum_um"].tolist(),
            "median_step_um": float(line[0]),
            "control_step_sd_um": control_step_sd,
            "linear_slope_um_per_frame": float(line[0]),
            "linear_intercept_um": float(line[1]),
            "closest_frame_to_holographic_plane": int(observed.frame_numbers[closest_index]),
            "closest_distance_um": float(focus_um[closest_index]),
            "trajectory_objective": production.trajectory.objective,
            "data_supported": {
                "criterion": "local branch pull <=6 um, three-band span <=12 um, I0/I90 disagreement <=12 um, and nonlocal alias gap >0.002",
                "mask": reliable.tolist(),
                "frame_numbers": observed.frame_numbers[reliable].tolist(),
                "count": int(np.count_nonzero(reliable)),
                "prior_pull_um": branch["prior_pull_um"].tolist(),
                "radial_band_span_um": radial_span.tolist(),
                "channel_disagreement_um": channel_disagreement.tolist(),
                "alias_gap": branch["alias_gap"].tolist(),
            },
            "robust_data_only_line": {
                "slope_um_per_frame": float(line[0]),
                "intercept_um": float(line[1]),
                "zero_crossing_frame": float(-line[1] / line[0]),
                "rms_um": float(np.sqrt(np.mean(line_residual[line_mask] ** 2))),
            },
            "controls": control_reports,
        },
        "profile_agreement": {
            "calibration": "none",
            "median_correlation_by_channel": {name: float(np.median(agreement["correlation"][:, index])) for index, name in enumerate(CHANNELS)},
            "median_relative_rms_by_channel": {name: float(np.median(agreement["relative_rms"][:, index])) for index, name in enumerate(CHANNELS)},
        },
        "registration": {
            "centers_xy_px": observed.centers_xy_px.tolist(),
            "accidental_shift_onset_frame": int(observed.frame_numbers[jump_onset]) if jump_onset < observed.frame_numbers.size else None,
            "center_jump_threshold_px": jump_threshold,
            "center_steps_px": center_steps.tolist(),
            "center_motion_used_frame_count": used_center_count,
            "center_slope_px_per_um": center_coefficients[0].tolist(),
            "center_slope_dimensionless": center_slope_dimensionless.tolist(),
            "center_motion_rms_residual_px": float(np.sqrt(np.mean(np.sum(center_residual * center_residual, axis=1)))),
            "holographic_spectral_slope_dimensionless": spectral_slope.tolist(),
        },
        "limitations": [
            "Recovered z is an effective image-space propagation coordinate relative to the holographic reconstruction plane, not an absolute distance from a glass or liquid-crystal interface.",
            "The scalar n=1 propagation kernel does not model the stratified glass/liquid-crystal path; comparison with an exit-surface simulation requires an additional axial offset and layer transfer.",
            "Only I0 and I90 infer focus; I45 and I135 validate the coherent BS2 model after the measured path-parity exchange between camera and project-basis labels.",
            "The full and low-frequency Ground fields are not removed because they contain optical illumination structure that is not a local multiplicative transmission under propagation.",
        ],
        "outputs": {
            "archive": _relative(output / "focus_fit.npz"),
            "propagation_cache": _relative(output / "propagation_cache.npz"),
            "focus_cost_map": _relative(output / "focus_cost_map.pdf"),
            "absolute_cost_map": _relative(output / "focus_cost_map_absolute.pdf"),
            "focus_trajectory": _relative(output / "focus_trajectory.pdf"),
            "radial_profile_matches": _relative(
                output / "radial_profile_matches.pdf"
            ),
            "closest_focus_channels": _relative(
                output / "closest_focus_channels.pdf"
            ),
        },
    }
    _write_json(output / "report.json", report)
    command = (
        "uv run python "
        "code/scripts/analyze_holographic_hdr_focus_stack.py"
    )
    (output / "README.md").write_text(
        "# HDR focus recovery\n\n"
        "This directory matches the registered dark-corrected 12/42-ms HDR focus stack "
        "to angular-spectrum propagation of the averaged holographic Jones field. "
        "The primary cost uses confidence-weighted I0 and I90 radial profiles from the "
        "canonical dimensionless HDR data without further radiometric normalization. "
        "The horizontal input is fixed, and the measured "
        "BS2 response diag(1,sqrt(2)) is included coherently in the forward model. No "
        "intensity scale or offset is fitted. Ground is not divided from the data; "
        "radial-band and single-channel controls are recorded in report.json.\n\n"
        "Reproduce from the repository root:\n\n"
        f"```bash\n{command}\n```\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jones", type=Path, default=DEFAULT_JONES)
    parser.add_argument("--hdr-root", type=Path, default=DEFAULT_HDR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pupil-report",
        type=Path,
        default=DEFAULT_PUPIL_REPORT,
        help="pupil-analysis report used when --numerical-aperture is omitted",
    )
    parser.add_argument("--wavelength-um", type=float, default=WAVELENGTH_UM)
    parser.add_argument(
        "--numerical-aperture",
        type=float,
        default=None,
        help="coherent NA; default: 520-nm source median from --pupil-report",
    )
    parser.add_argument("--padding-factor", type=float, default=2.0)
    parser.add_argument("--focus-min-um", type=float, default=float(FOCUS_GRID_UM[0]))
    parser.add_argument("--focus-max-um", type=float, default=float(FOCUS_GRID_UM[-1]))
    parser.add_argument("--focus-step-um", type=float, default=float(np.median(np.diff(FOCUS_GRID_UM))))
    parser.add_argument(
        "--nominal-stage-step-um",
        type=float,
        default=DIRECT_POM_CONFIG.translation_reading_per_frame_um,
    )
    parser.add_argument("--workers", type=int, default=-1)
    add_force_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.numerical_aperture is None:
        args.numerical_aperture = _measured_effective_na(
            args.pupil_report.expanduser().resolve()
        )
    output = args.output.expanduser().resolve()
    populated_outputs = (
        (output,)
        if output.exists() and any(output.iterdir())
        else ()
    )
    refuse_existing_outputs(
        populated_outputs,
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    report = run(args)
    focus = report["focus_recovery"]
    print(
        "Recovered {count} HDR frames: dz/frame={step:.3f} um; closest frame={frame} "
        "at z={distance:.3f} um; input angle={angle:.3f} deg.".format(
            count=report["sources"]["frame_count"],
            step=focus["median_step_um"],
            frame=focus["closest_frame_to_holographic_plane"],
            distance=focus["closest_distance_um"],
            angle=report["input_polarization"]["angle_deg_in_holographic_xy_basis"],
        )
    )
    print(f"Report: {Path(args.output).expanduser().resolve() / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
