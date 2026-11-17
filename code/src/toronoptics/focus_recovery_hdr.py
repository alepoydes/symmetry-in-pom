"""HDR-aware recovery of a through-focus trajectory from a Jones field.

The acquisition radiometry is resolved before this module is called: the two
exposures have already been dark-corrected, registered, merged, normalized,
and transformed to the project basis.  This module reduces those HDR images to
confidence-weighted radial profiles and
matches them to a bank of propagated Jones Gram matrices.  The implementation
keeps the data-only focus information separate from the mechanical-step prior
and reports channel and radial-band sensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from toronoptics.focus_stack import (
    ANALYZER_ANGLES_DEG,
    FocusTrajectoryFit,
    affine_profile_costs,
    fit_smooth_focus_trajectory,
    sample_radial_profile,
)
from toronoptics.beamsplitter_calibration import (
    post_beamsplitter_intensities,
    radial_output_coherency,
)
from toronoptics.focus_stack_hdr import FOCUS_HDR_SCHEMA_VERSION


CHANNELS = tuple(ANALYZER_ANGLES_DEG)


@dataclass(frozen=True)
class HDRRadialStack:
    """Confidence-weighted radial representation of the merged focus stack."""

    frame_numbers: np.ndarray
    centers_xy_px: np.ndarray
    incident_levels: np.ndarray
    profiles: np.ndarray
    coverage: np.ndarray
    mean_confidence: np.ndarray
    low_exposure_fraction: np.ndarray
    radii_um: np.ndarray
    pixel_size_um: float


@dataclass(frozen=True)
class FocusFitResult:
    """Joint input-angle and focus-trajectory fit."""

    input_angle_deg: float
    cost: np.ndarray
    trajectory: FocusTrajectoryFit
    predicted_profiles: np.ndarray
    angle_scan_deg: np.ndarray
    angle_scan_loss: np.ndarray


def load_hdr_radial_stack(
    hdr_root: Path,
    *,
    radii_um: np.ndarray,
    pixel_size_um: float,
    angle_count: int = 360,
) -> HDRRadialStack:
    """Load canonical normalized frames and form their radial profiles.

    This function performs no radiometric correction, coordinate reflection,
    analyzer relabeling, or background division.  Those operations are fixed by
    the versioned HDR data contract before the files reach this module.
    """

    root = Path(hdr_root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema != FOCUS_HDR_SCHEMA_VERSION:
        raise ValueError(
            f"Expected HDR schema {FOCUS_HDR_SCHEMA_VERSION!r}, got {schema!r} in {root}"
        )
    frame_numbers = np.asarray(manifest["sources"]["frame_numbers"], dtype=np.int64)

    radii = np.asarray(radii_um, dtype=np.float64)
    radii_px = radii / float(pixel_size_um)
    shape = (frame_numbers.size, len(CHANNELS), radii.size)
    profiles = np.empty(shape, dtype=np.float64)
    coverage = np.empty(shape, dtype=np.float64)
    centers = np.empty((frame_numbers.size, 2), dtype=np.float64)
    incident = np.empty(frame_numbers.size, dtype=np.float64)
    mean_confidence = np.empty((frame_numbers.size, len(CHANNELS)), dtype=np.float64)
    low_fraction = np.empty_like(mean_confidence)

    for frame_index, frame_number in enumerate(frame_numbers):
        path = root / "frames" / f"{int(frame_number):04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            source_channels = tuple(str(value) for value in data["channel_names"])
            intensity = np.asarray(data["intensity_yxc"], dtype=np.float64)
            confidence = np.asarray(data["confidence_yxc"], dtype=np.float64) / 255.0
            low_weight = (
                np.asarray(data["low_exposure_fraction_yxc"], dtype=np.float64)
                / 255.0
            )
            centers[frame_index] = np.asarray(data["center_xy_px"], dtype=np.float64)
            incident[frame_index] = float(data["incident_level_raw_42ms_equivalent"])
            frame_schema = str(data["schema_version"])
            units = str(data["units"])
        if frame_schema != FOCUS_HDR_SCHEMA_VERSION:
            raise ValueError(
                f"Expected frame schema {FOCUS_HDR_SCHEMA_VERSION!r}, got {frame_schema!r} in {path}"
            )
        if source_channels != CHANNELS:
            raise ValueError(
                f"Expected project channel order {CHANNELS}, got {source_channels} in {path}"
            )
        if units != "dimensionless detector intensity relative to frame I0 background":
            raise ValueError(f"Unexpected intensity units {units!r} in {path}")
        if not np.isfinite(incident[frame_index]) or incident[frame_index] <= 0.0:
            raise ValueError(f"Invalid incident level in {path}: {incident[frame_index]}")

        for output_index, name in enumerate(CHANNELS):
            source_index = source_channels.index(name)
            confidence_map = np.clip(confidence[..., source_index], 0.0, 1.0)
            profile, profile_coverage = sample_radial_profile(
                intensity[..., source_index],
                center_xy=tuple(centers[frame_index]),
                radii_px=radii_px,
                weights_yx=confidence_map,
                angle_count=angle_count,
            )
            profiles[frame_index, output_index] = profile
            coverage[frame_index, output_index] = profile_coverage
            valid = np.isfinite(intensity[..., source_index])
            mean_confidence[frame_index, output_index] = float(
                np.mean(confidence_map[valid])
            )
            low_fraction[frame_index, output_index] = float(
                np.mean(low_weight[..., source_index][valid])
            )

    return HDRRadialStack(
        frame_numbers=frame_numbers,
        centers_xy_px=centers,
        incident_levels=incident,
        profiles=profiles,
        coverage=coverage,
        mean_confidence=mean_confidence,
        low_exposure_fraction=low_fraction,
        radii_um=radii,
        pixel_size_um=float(pixel_size_um),
    )


def predicted_profiles(
    radial_gram: np.ndarray,
    input_angle_deg: float,
    *,
    beamsplitter_output_ratio: complex = 1.0,
) -> np.ndarray:
    """Evaluate detector-plane profiles including the coherent BS2 response."""

    coherency = radial_output_coherency(radial_gram, input_angle_deg)
    incident_i0 = max(np.cos(np.radians(float(input_angle_deg))) ** 2, 1.0e-12)
    stokes_order = post_beamsplitter_intensities(
        coherency,
        beamsplitter_output_ratio,
        normalization=incident_i0,
    )
    stokes_names = ("I0", "I90", "I45", "I135")
    return np.stack(
        [stokes_order[..., stokes_names.index(name)] for name in CHANNELS],
        axis=-2,
    )


def profile_cost_surface(
    observed: HDRRadialStack,
    predicted: np.ndarray,
    *,
    channels: Iterable[str] = ("I0", "I90"),
    radial_interval_um: tuple[float, float] = (2.0, 39.0),
    fit_affine: bool = False,
) -> np.ndarray:
    """Return radial mismatch by frame and focus.

    The default compares calibrated detector intensities directly.  An affine
    shape metric is available only when explicitly requested as a sensitivity
    control; it is not used by the production focus recovery.
    """

    selected_channels = tuple(channels)
    unknown = set(selected_channels) - set(CHANNELS)
    if unknown:
        raise ValueError(f"Unknown analyzer channels: {sorted(unknown)}")
    radii = observed.radii_um
    radial_mask = (radii >= radial_interval_um[0]) & (radii <= radial_interval_um[1])
    if np.count_nonzero(radial_mask) < 8:
        raise ValueError(f"Radial interval is too narrow: {radial_interval_um}")
    surface = np.zeros((observed.frame_numbers.size, predicted.shape[0]), dtype=np.float64)
    total_channel_weight = np.zeros(observed.frame_numbers.size, dtype=np.float64)
    radius_weight = np.maximum(radii[radial_mask], 0.5)

    for frame_index in range(observed.frame_numbers.size):
        for name in selected_channels:
            channel_index = CHANNELS.index(name)
            profile_coverage = observed.coverage[frame_index, channel_index, radial_mask]
            if float(np.mean(profile_coverage)) < 0.70:
                continue
            weight = radius_weight * profile_coverage * profile_coverage
            measured = observed.profiles[frame_index, channel_index, radial_mask]
            model = predicted[:, channel_index, radial_mask]
            if fit_affine:
                costs = affine_profile_costs(measured, model, weights=weight)
            else:
                valid = np.isfinite(measured) & np.isfinite(weight) & (weight > 0.0)
                residual = measured[valid][None, :] - model[:, valid]
                costs = np.sum(weight[valid][None, :] * residual**2, axis=1) / max(
                    float(np.sum(weight[valid] * measured[valid] ** 2)),
                    np.finfo(float).eps,
                )
            # Equal analyzer weighting prevents the bright co-polarized channel
            # from suppressing the crossed channel solely through throughput.
            surface[frame_index] += costs
            total_channel_weight[frame_index] += 1.0
    if np.any(total_channel_weight == 0.0):
        missing = observed.frame_numbers[total_channel_weight == 0.0]
        raise ValueError(f"No valid radial channels in frames {missing.tolist()}")
    return surface / total_channel_weight[:, None]


def trajectory_loss(
    surface: np.ndarray,
    focus_grid_um: np.ndarray,
    focus_um: np.ndarray,
) -> float:
    """Mean profile mismatch sampled along a trajectory."""

    return float(
        np.mean(
            [
                np.interp(focus, focus_grid_um, surface[index])
                for index, focus in enumerate(focus_um)
            ]
        )
    )


def fit_angle_and_trajectory(
    observed: HDRRadialStack,
    radial_gram: np.ndarray,
    focus_grid_um: np.ndarray,
    *,
    nominal_step_um: float = 10.0,
    radial_interval_um: tuple[float, float] = (2.0, 39.0),
    channels: Iterable[str] = ("I0", "I90"),
    beamsplitter_output_ratio: complex = 1.0,
    fixed_input_angle_deg: float | None = None,
    fit_affine: bool = False,
) -> FocusFitResult:
    """Alternate a global input-angle fit with the monotone focus trajectory."""

    angle = 0.0 if fixed_input_angle_deg is None else float(fixed_input_angle_deg)
    scan = np.linspace(-15.0, 15.0, 121)
    scan_loss = np.empty_like(scan)
    trajectory: FocusTrajectoryFit | None = None
    surface = np.empty((observed.frame_numbers.size, focus_grid_um.size))
    predicted = predicted_profiles(
        radial_gram,
        angle,
        beamsplitter_output_ratio=beamsplitter_output_ratio,
    )

    if fixed_input_angle_deg is not None:
        surface = profile_cost_surface(
            observed,
            predicted,
            channels=channels,
            radial_interval_um=radial_interval_um,
            fit_affine=fit_affine,
        )
        trajectory = fit_smooth_focus_trajectory(
            surface,
            focus_grid_um,
            nominal_step_um=nominal_step_um,
            step_tolerance_um=4.0,
            curvature_tolerance_um=1.5,
        )
        return FocusFitResult(
            input_angle_deg=angle,
            cost=surface,
            trajectory=trajectory,
            predicted_profiles=predicted,
            angle_scan_deg=np.asarray([angle]),
            angle_scan_loss=np.asarray([trajectory_loss(surface, focus_grid_um, trajectory.focus_um)]),
        )

    for iteration in range(3):
        predicted = predicted_profiles(
            radial_gram,
            angle,
            beamsplitter_output_ratio=beamsplitter_output_ratio,
        )
        surface = profile_cost_surface(
            observed,
            predicted,
            channels=channels,
            radial_interval_um=radial_interval_um,
            fit_affine=fit_affine,
        )
        trajectory = fit_smooth_focus_trajectory(
            surface,
            focus_grid_um,
            nominal_step_um=nominal_step_um,
            step_tolerance_um=4.0,
            curvature_tolerance_um=1.5,
        )
        for index, candidate in enumerate(scan):
            candidate_surface = profile_cost_surface(
                observed,
                predicted_profiles(
                    radial_gram,
                    float(candidate),
                    beamsplitter_output_ratio=beamsplitter_output_ratio,
                ),
                channels=channels,
                radial_interval_um=radial_interval_um,
                fit_affine=fit_affine,
            )
            scan_loss[index] = trajectory_loss(
                candidate_surface,
                focus_grid_um,
                trajectory.focus_um,
            )
        best = int(np.argmin(scan_loss))
        lower = scan[max(best - 1, 0)]
        upper = scan[min(best + 1, scan.size - 1)]

        def residual(parameter: np.ndarray) -> np.ndarray:
            candidate_surface = profile_cost_surface(
                observed,
                predicted_profiles(
                    radial_gram,
                    float(parameter[0]),
                    beamsplitter_output_ratio=beamsplitter_output_ratio,
                ),
                channels=channels,
                radial_interval_um=radial_interval_um,
                fit_affine=fit_affine,
            )
            return np.sqrt(
                np.maximum(
                    [
                        np.interp(value, focus_grid_um, candidate_surface[row])
                        for row, value in enumerate(trajectory.focus_um)
                    ],
                    0.0,
                )
            )

        optimized = least_squares(
            residual,
            np.asarray([scan[best]]),
            bounds=(np.asarray([lower]), np.asarray([upper])),
            xtol=1.0e-6,
            ftol=1.0e-6,
            gtol=1.0e-6,
        )
        updated = float(optimized.x[0])
        if iteration > 0 and abs(updated - angle) < 0.002:
            angle = updated
            break
        angle = updated

    predicted = predicted_profiles(
        radial_gram,
        angle,
        beamsplitter_output_ratio=beamsplitter_output_ratio,
    )
    surface = profile_cost_surface(
        observed,
        predicted,
        channels=channels,
        radial_interval_um=radial_interval_um,
        fit_affine=fit_affine,
    )
    trajectory = fit_smooth_focus_trajectory(
        surface,
        focus_grid_um,
        nominal_step_um=nominal_step_um,
        step_tolerance_um=4.0,
        curvature_tolerance_um=1.5,
    )
    for index, candidate in enumerate(scan):
        candidate_surface = profile_cost_surface(
            observed,
            predicted_profiles(
                radial_gram,
                float(candidate),
                beamsplitter_output_ratio=beamsplitter_output_ratio,
            ),
            channels=channels,
            radial_interval_um=radial_interval_um,
            fit_affine=fit_affine,
        )
        scan_loss[index] = trajectory_loss(candidate_surface, focus_grid_um, trajectory.focus_um)
    return FocusFitResult(
        input_angle_deg=angle,
        cost=surface,
        trajectory=trajectory,
        predicted_profiles=predicted,
        angle_scan_deg=scan,
        angle_scan_loss=scan_loss,
    )


def fit_trajectory_control(
    observed: HDRRadialStack,
    predicted: np.ndarray,
    focus_grid_um: np.ndarray,
    *,
    nominal_step_um: float,
    radial_interval_um: tuple[float, float],
    channels: Iterable[str],
    fit_affine: bool = False,
) -> tuple[np.ndarray, FocusTrajectoryFit]:
    """Fit one fixed-angle control trajectory."""

    surface = profile_cost_surface(
        observed,
        predicted,
        channels=channels,
        radial_interval_um=radial_interval_um,
        fit_affine=fit_affine,
    )
    trajectory = fit_smooth_focus_trajectory(
        surface,
        focus_grid_um,
        nominal_step_um=nominal_step_um,
        step_tolerance_um=4.0,
        curvature_tolerance_um=1.5,
    )
    return surface, trajectory


def local_branch_metrics(
    surface: np.ndarray,
    focus_grid_um: np.ndarray,
    trajectory_um: np.ndarray,
    *,
    local_half_width_um: float = 12.0,
    alias_exclusion_um: float = 8.0,
) -> dict[str, np.ndarray]:
    """Measure local data support without jumping to another ring alias."""

    local_minimum = np.empty_like(trajectory_um, dtype=np.float64)
    local_cost = np.empty_like(trajectory_um, dtype=np.float64)
    alias_gap = np.empty_like(trajectory_um, dtype=np.float64)
    global_minimum = focus_grid_um[np.argmin(surface, axis=1)]
    for index, center in enumerate(trajectory_um):
        local = np.abs(focus_grid_um - center) <= local_half_width_um
        local_indices = np.flatnonzero(local)
        selected = local_indices[int(np.argmin(surface[index, local]))]
        local_minimum[index] = focus_grid_um[selected]
        local_cost[index] = surface[index, selected]
        alternative = np.abs(focus_grid_um - local_minimum[index]) > alias_exclusion_um
        alias_gap[index] = (
            float(np.min(surface[index, alternative])) - local_cost[index]
            if np.any(alternative)
            else np.nan
        )
    return {
        "global_minimum_um": global_minimum,
        "local_minimum_um": local_minimum,
        "local_cost": local_cost,
        "prior_pull_um": np.abs(local_minimum - trajectory_um),
        "alias_gap": alias_gap,
    }
