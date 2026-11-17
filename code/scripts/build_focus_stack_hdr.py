#!/usr/bin/env python3
"""Build a registered 12/42-ms HDR focus stack and reference background."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, shift

from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.focus_stack import MACRO_QUADRANT_SLICES, estimate_radial_center
from toronoptics.focus_stack_hdr import (
    DarkFrameCalibration,
    FOCUS_HDR_CHANNELS,
    FOCUS_HDR_SCHEMA_VERSION,
    RAW_TO_PROJECT_SOURCE_CHANNEL,
    RankOneReference,
    calibrate_dark_frame,
    fit_linear_scale,
    merge_hdr_exposures,
    raw_camera_center_to_project,
    raw_camera_to_project,
    robust_rank_one_reference,
    robust_sigma,
)
from toronoptics.source_config import (
    DEFAULT_EXPERIMENT_MANIFEST,
    load_direct_pom_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data_src/direct_pom/520nm"
DEFAULT_DATA_OUTPUT = PROJECT_ROOT / "data/experimental/results_toron25um_50X_focus_hdr"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "data/analysis/results_toron25um_50X_focus_hdr"
DIRECT_POM_CONFIG = load_direct_pom_config()
EXPOSURES = DIRECT_POM_CONFIG.exposure_times_ms
CHANNELS = ("I0", "I135", "I45", "I90")
BRIGHT_CHANNELS = ("I0", "I135", "I45")
if CHANNELS != FOCUS_HDR_CHANNELS:
    raise RuntimeError("HDR channel constants disagree")
GROUND_RE = re.compile(r"^Ground \((?P<index>\d+)\)$")
EXPECTED_FREE_FIELD = {"I0": 1.0, "I135": 0.5, "I45": 0.5, "I90": 0.0}
BS2_OUTPUT_AMPLITUDE_RATIO = DIRECT_POM_CONFIG.beamsplitter_amplitude_ratio


def _relative(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_macro(path: Path) -> tuple[dict[str, np.ndarray], float]:
    with Image.open(path) as image:
        encoded = np.asarray(image)
    if encoded.shape != DIRECT_POM_CONFIG.encoded_shape_yx or not np.issubdtype(
        encoded.dtype, np.integer
    ):
        raise ValueError(
            f"Expected a {DIRECT_POM_CONFIG.encoded_shape_yx} integer macro image, "
            f"got {encoded.shape}: {path}"
        )
    full_scale = float(np.iinfo(encoded.dtype).max)
    normalized = encoded.astype(np.float32) / full_scale
    return {
        name: np.ascontiguousarray(normalized[slices])
        for name, slices in MACRO_QUADRANT_SLICES.items()
    }, full_scale


def _numeric_paths(directory: Path) -> dict[int, Path]:
    result = {
        int(path.stem): path
        for path in Path(directory).glob("*.png")
        if path.stem.isdigit()
    }
    if not result:
        raise FileNotFoundError(f"No numbered PNG files in {directory}")
    return result


def _ground_paths(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in Path(directory).glob("Ground*.png"):
        match = GROUND_RE.fullmatch(path.stem)
        if match is not None:
            result[int(match.group("index"))] = path
    if not result:
        raise FileNotFoundError(f"No indexed Ground images in {directory}")
    return result


def _estimate_center(
    image_yx: np.ndarray,
    initial_xy: tuple[float, float] | None,
) -> tuple[float, float]:
    estimate = estimate_radial_center(
        np.maximum(image_yx[::2, ::2], 0.0),
        initial_center_xy=(
            None
            if initial_xy is None
            else (0.5 * initial_xy[0], 0.5 * initial_xy[1])
        ),
        search_half_width_px=195.0,
        gradient_sigma_px=0.9,
        highpass_sigma_px=22.5,
        minimum_radius_px=9.0,
        maximum_radius_px=165.0,
        gradient_percentile=72.0,
    )
    return 2.0 * estimate.center_xy[0], 2.0 * estimate.center_xy[1]


def _shift_image(
    image_yx: np.ndarray,
    shift_xy: tuple[float, float],
    *,
    preserve_edges: bool,
) -> np.ndarray:
    dx, dy = shift_xy
    return shift(
        np.asarray(image_yx, dtype=np.float32),
        shift=(float(dy), float(dx)),
        order=1,
        mode="nearest" if preserve_edges else "constant",
        cval=0.0 if preserve_edges else np.nan,
        prefilter=False,
    )


def _load_dark_calibration(
    input_root: Path,
) -> tuple[dict[int, dict[str, DarkFrameCalibration]], float]:
    result: dict[int, dict[str, DarkFrameCalibration]] = {}
    full_scales = []
    for exposure in EXPOSURES:
        channels, full_scale = _load_macro(
            input_root / f"Exposure_{exposure}ms" / "Environment.png"
        )
        full_scales.append(full_scale)
        result[exposure] = {
            name: calibrate_dark_frame(image, full_scale=full_scale)
            for name, image in channels.items()
        }
    if len(set(full_scales)) != 1:
        raise ValueError(f"Exposure images have different encoded full scales: {full_scales}")
    return result, full_scales[0]


def _corrected_channels(
    path: Path,
    dark: dict[str, DarkFrameCalibration],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw, _full_scale = _load_macro(path)
    corrected = {
        name: np.asarray(raw[name] - dark[name].dark_yx, dtype=np.float32)
        for name in CHANNELS
    }
    return raw, corrected


def _estimate_channel_offsets(
    paths: dict[int, dict[int, Path]],
    dark: dict[int, dict[str, DarkFrameCalibration]],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    frame_numbers = sorted(set(paths[12]) & set(paths[42]))
    selection_count = min(9, len(frame_numbers))
    selected = [
        frame_numbers[index]
        for index in np.linspace(
            0, len(frame_numbers) - 1, selection_count
        ).round().astype(int)
    ]
    values: dict[str, list[tuple[float, float]]] = {name: [] for name in CHANNELS}
    for exposure in EXPOSURES:
        for frame_number in selected:
            _raw, corrected = _corrected_channels(paths[exposure][frame_number], dark[exposure])
            centers = {name: _estimate_center(corrected[name], None) for name in CHANNELS}
            reference = centers["I0"]
            for name in CHANNELS:
                values[name].append(
                    (centers[name][0] - reference[0], centers[name][1] - reference[1])
                )
    offsets = {
        name: (
            float(np.median([value[0] for value in channel_values])),
            float(np.median([value[1] for value in channel_values])),
        )
        for name, channel_values in values.items()
    }
    spreads = {
        name: (
            robust_sigma(np.asarray([value[0] for value in channel_values])),
            robust_sigma(np.asarray([value[1] for value in channel_values])),
        )
        for name, channel_values in values.items()
    }
    return offsets, spreads


def _estimate_frame_centers(
    paths: dict[int, dict[int, Path]],
    dark: dict[int, dict[str, DarkFrameCalibration]],
) -> dict[int, dict[int, tuple[float, float]]]:
    result: dict[int, dict[int, tuple[float, float]]] = {}
    for exposure in EXPOSURES:
        result[exposure] = {}
        previous = None
        frame_numbers = sorted(paths[exposure])
        for position, frame_number in enumerate(frame_numbers):
            _raw, corrected = _corrected_channels(paths[exposure][frame_number], dark[exposure])
            center = _estimate_center(corrected["I0"], previous)
            result[exposure][frame_number] = center
            previous = center
            if position % 10 == 0 or position + 1 == len(frame_numbers):
                print(f"[centers] {exposure} ms: {position + 1}/{len(frame_numbers)}")
    return result


def _align_channels(
    raw: dict[str, np.ndarray],
    corrected: dict[str, np.ndarray],
    offsets_xy: dict[str, tuple[float, float]],
    *,
    exposure_shift_xy: tuple[float, float] = (0.0, 0.0),
    preserve_edges: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    aligned_raw = {}
    aligned_corrected = {}
    for name in CHANNELS:
        offset = offsets_xy[name]
        total_shift = (
            float(exposure_shift_xy[0] - offset[0]),
            float(exposure_shift_xy[1] - offset[1]),
        )
        aligned_raw[name] = _shift_image(
            raw[name],
            total_shift,
            preserve_edges=preserve_edges,
        )
        aligned_corrected[name] = _shift_image(
            corrected[name],
            total_shift,
            preserve_edges=preserve_edges,
        )
    return aligned_raw, aligned_corrected


def _build_ground_reference(
    input_root: Path,
    dark: dict[int, dict[str, DarkFrameCalibration]],
    offsets_xy: dict[str, tuple[float, float]],
) -> tuple[
    RankOneReference,
    np.ndarray,
    list[tuple[int, int, str]],
    dict[str, Any],
]:
    paths = {
        exposure: _ground_paths(input_root / f"Exposure_{exposure}ms")
        for exposure in EXPOSURES
    }
    common_indices = sorted(set(paths[12]) & set(paths[42]))
    if not common_indices:
        raise ValueError("The two exposures have no common indexed Ground images")

    images: list[np.ndarray] = []
    labels: list[tuple[int, int, str]] = []
    i90_images: dict[tuple[int, int], np.ndarray] = {}
    for exposure in EXPOSURES:
        for index in common_indices:
            raw, corrected = _corrected_channels(paths[exposure][index], dark[exposure])
            aligned_raw, aligned = _align_channels(
                raw,
                corrected,
                offsets_xy,
                preserve_edges=True,
            )
            del aligned_raw
            for name in BRIGHT_CHANNELS:
                images.append(aligned[name])
                labels.append((exposure, index, name))
            i90_images[(exposure, index)] = aligned["I90"]

    reference = robust_rank_one_reference(images)
    amplitudes = {
        label: float(amplitude)
        for label, amplitude in zip(labels, reference.amplitudes, strict=True)
    }
    exposure_ratios = []
    exposure_ratios_by_channel: dict[str, list[float]] = {name: [] for name in BRIGHT_CHANNELS}
    for index in common_indices:
        for name in BRIGHT_CHANNELS:
            ratio = amplitudes[(42, index, name)] / amplitudes[(12, index, name)]
            exposure_ratios.append(ratio)
            exposure_ratios_by_channel[name].append(ratio)

    channel_ratios: dict[str, list[float]] = {name: [] for name in BRIGHT_CHANNELS}
    for exposure in EXPOSURES:
        for index in common_indices:
            i0 = amplitudes[(exposure, index, "I0")]
            for name in BRIGHT_CHANNELS:
                channel_ratios[name].append(amplitudes[(exposure, index, name)] / i0)

    mode = np.asarray(reference.mode_yx, dtype=np.float64)
    mode_flat = mode[::4, ::4].ravel()
    design = np.column_stack((mode_flat, np.ones(mode_flat.size)))
    i90_leakage = []
    i90_offsets = []
    for exposure in EXPOSURES:
        for index in common_indices:
            values = i90_images[(exposure, index)][::4, ::4].ravel()
            finite = np.isfinite(values)
            coefficient, offset = np.linalg.lstsq(design[finite], values[finite], rcond=None)[0]
            i0_amplitude = amplitudes[(exposure, index, "I0")]
            i90_leakage.append(float(coefficient / i0_amplitude))
            i90_offsets.append(float(offset))

    smooth = gaussian_filter(reference.mode_yx, sigma=12.0, mode="reflect")
    smooth /= np.mean(smooth)
    full = reference.mode_yx / np.mean(reference.mode_yx)
    channel_mode_agreement = {}
    for name in BRIGHT_CHANNELS:
        selected = [
            image / reference.amplitudes[position]
            for position, (image, label) in enumerate(zip(images, labels, strict=True))
            if label[2] == name
        ]
        channel_mode = np.mean(selected, axis=0, dtype=np.float64)
        channel_mode = gaussian_filter(channel_mode, sigma=12.0, mode="reflect")
        channel_mode /= np.mean(channel_mode)
        margin = 60
        reference_crop = smooth[margin:-margin, margin:-margin]
        channel_crop = channel_mode[margin:-margin, margin:-margin]
        channel_mode_agreement[name] = {
            "correlation": float(
                np.corrcoef(reference_crop.ravel(), channel_crop.ravel())[0, 1]
            ),
            "relative_rms": float(
                np.sqrt(np.mean((channel_crop - reference_crop) ** 2))
                / np.sqrt(np.mean(reference_crop * reference_crop))
            ),
        }

    normalized_images = [
        image / amplitude
        for image, amplitude in zip(images, reference.amplitudes, strict=True)
    ]
    leave_one_out = []
    for held_index in common_indices:
        training = np.mean(
            [
                image
                for image, label in zip(normalized_images, labels, strict=True)
                if label[1] != held_index
            ],
            axis=0,
            dtype=np.float64,
        )
        held = np.mean(
            [
                image
                for image, label in zip(normalized_images, labels, strict=True)
                if label[1] == held_index
            ],
            axis=0,
            dtype=np.float64,
        )
        training = gaussian_filter(training, sigma=12.0, mode="reflect")
        held = gaussian_filter(held, sigma=12.0, mode="reflect")
        training /= np.mean(training)
        held /= np.mean(held)
        training_crop = training[60:-60, 60:-60]
        held_crop = held[60:-60, 60:-60]
        leave_one_out.append(
            {
                "ground_index": held_index,
                "correlation": float(
                    np.corrcoef(training_crop.ravel(), held_crop.ravel())[0, 1]
                ),
                "relative_rms": float(
                    np.sqrt(np.mean((held_crop - training_crop) ** 2))
                    / np.sqrt(np.mean(training_crop * training_crop))
                ),
            }
        )
    metrics = {
        "ground_indices": common_indices,
        "exposure_ratio": float(np.median(exposure_ratios)),
        "exposure_ratio_mad": robust_sigma(np.asarray(exposure_ratios)),
        "exposure_ratio_by_channel": {
            name: {
                "median": float(np.median(values)),
                "mad": robust_sigma(np.asarray(values)),
            }
            for name, values in exposure_ratios_by_channel.items()
        },
        "channel_background_ratio_to_i0": {
            name: float(np.median(values))
            for name, values in channel_ratios.items()
        },
        "channel_background_ratio_mad": {
            name: robust_sigma(np.asarray(values))
            for name, values in channel_ratios.items()
        },
        "i90_leakage_relative_to_i0": float(np.median(i90_leakage)),
        "i90_leakage_mad": robust_sigma(np.asarray(i90_leakage)),
        "i90_leakage_upper_95": float(
            max(0.0, np.median(i90_leakage) + 2.0 * robust_sigma(np.asarray(i90_leakage)))
        ),
        "i90_additive_offset_median": float(np.median(i90_offsets)),
        "rank_one_iterations": int(reference.iterations),
        "rank_one_first_energy_fraction": float(reference.singular_energy_fraction[0]),
        "rank_one_residual_median": float(np.median(reference.relative_residuals)),
        "rank_one_residual_maximum": float(np.max(reference.relative_residuals)),
        "lowpass_channel_mode_agreement": channel_mode_agreement,
        "leave_one_ground_out": leave_one_out,
        "leave_one_ground_out_correlation_minimum": float(
            min(value["correlation"] for value in leave_one_out)
        ),
        "leave_one_ground_out_relative_rms_maximum": float(
            max(value["relative_rms"] for value in leave_one_out)
        ),
    }
    return reference, np.stack((full, smooth)).astype(np.float32), labels, metrics


def _joint_frame_scale(
    low_raw: dict[str, np.ndarray],
    low: dict[str, np.ndarray],
    high_raw: dict[str, np.ndarray],
    high: dict[str, np.ndarray],
) -> tuple[float, dict[str, Any]]:
    fits = {}
    for name in BRIGHT_CHANNELS:
        valid = (
            np.isfinite(low[name])
            & np.isfinite(high[name])
            & np.isfinite(low_raw[name])
            & np.isfinite(high_raw[name])
            & (low_raw[name] < 0.28)
            & (high_raw[name] < 0.74)
        )
        fit = fit_linear_scale(
            low[name],
            high[name],
            valid_yx=valid,
            minimum_low=0.008,
            maximum_high=0.74,
            sample_step=4,
        )
        fits[name] = fit
    scales = np.asarray([fits[name].scale for name in BRIGHT_CHANNELS])
    common = float(np.median(scales))
    return common, {
        name: {
            "scale": float(fits[name].scale),
            "relative_residual": float(fits[name].relative_residual),
            "correlation": float(fits[name].correlation),
            "used_pixels": int(fits[name].used_pixels),
        }
        for name in BRIGHT_CHANNELS
    }


def _estimate_saturation_onset(
    high_values: np.ndarray,
    ratios: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, 51)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(centers.shape, np.nan)
    counts = np.zeros(centers.shape, dtype=np.int64)
    for index in range(centers.size):
        selected = (high_values >= edges[index]) & (high_values < edges[index + 1])
        counts[index] = int(np.count_nonzero(selected))
        if counts[index] >= 100:
            medians[index] = float(np.median(ratios[selected]))
    baseline_values = medians[(centers >= 0.15) & (centers <= 0.55) & np.isfinite(medians)]
    baseline = float(np.median(baseline_values)) if baseline_values.size else 1.0
    normalized = medians / max(baseline, np.finfo(float).tiny)
    candidates = np.flatnonzero(
        (centers >= 0.60)
        & np.isfinite(normalized)
        & (normalized < 0.98)
    )
    onset = float(centers[candidates[0]] - 0.01) if candidates.size else 0.94
    onset = float(np.clip(onset, 0.75, 0.94))
    return onset, {
        "bin_centers": centers.tolist(),
        "median_high_over_scaled_low": [
            float(value) if np.isfinite(value) else None
            for value in medians
        ],
        "counts": counts.tolist(),
        "baseline": baseline,
        "soft_threshold": onset,
    }


def _first_pass_scales(
    paths: dict[int, dict[int, Path]],
    dark: dict[int, dict[str, DarkFrameCalibration]],
    offsets_xy: dict[str, tuple[float, float]],
    centers: dict[int, dict[int, tuple[float, float]]],
) -> tuple[dict[int, float], dict[int, dict[str, Any]], dict[str, Any]]:
    frame_scales: dict[int, float] = {}
    frame_fits: dict[int, dict[str, Any]] = {}
    high_samples = []
    ratio_samples = []
    frame_numbers = sorted(set(paths[12]) & set(paths[42]))
    for position, frame_number in enumerate(frame_numbers):
        low_raw_native, low_native = _corrected_channels(paths[12][frame_number], dark[12])
        high_raw_native, high_native = _corrected_channels(paths[42][frame_number], dark[42])
        target = centers[42][frame_number]
        source = centers[12][frame_number]
        exposure_shift = (target[0] - source[0], target[1] - source[1])
        low_raw, low = _align_channels(
            low_raw_native,
            low_native,
            offsets_xy,
            exposure_shift_xy=exposure_shift,
        )
        high_raw, high = _align_channels(high_raw_native, high_native, offsets_xy)
        scale, fits = _joint_frame_scale(low_raw, low, high_raw, high)
        frame_scales[frame_number] = scale
        frame_fits[frame_number] = fits

        for name in CHANNELS:
            low_sample = low[name][::8, ::8]
            high_sample = high[name][::8, ::8]
            high_raw_sample = high_raw[name][::8, ::8]
            valid = (
                np.isfinite(low_sample)
                & np.isfinite(high_sample)
                & np.isfinite(high_raw_sample)
                & (low_sample > 0.008)
                & (high_raw_sample > 0.04)
                & (high_raw_sample < 0.999)
            )
            if np.any(valid):
                high_samples.append(high_raw_sample[valid])
                ratio_samples.append(high_sample[valid] / (scale * low_sample[valid]))
        if position % 10 == 0 or position + 1 == len(frame_numbers):
            print(f"[radiometry] {position + 1}/{len(frame_numbers)}")

    onset, saturation_curve = _estimate_saturation_onset(
        np.concatenate(high_samples),
        np.concatenate(ratio_samples),
    )
    saturation_curve["frame_scale_median"] = float(np.median(list(frame_scales.values())))
    saturation_curve["frame_scale_mad"] = robust_sigma(np.asarray(list(frame_scales.values())))
    return frame_scales, frame_fits, saturation_curve


def _incident_level(
    intensity_i0_yx: np.ndarray,
    reference_yx: np.ndarray,
    center_xy: tuple[float, float],
    confidence_yx: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    height, width = intensity_i0_yx.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    radius = np.hypot(xx - float(center_xy[0]), yy - float(center_xy[1]))
    border = 24
    mask = (
        np.isfinite(intensity_i0_yx)
        & np.isfinite(confidence_yx)
        & (confidence_yx >= 0.8)
        & (radius >= 360.0)
        & (yy >= border)
        & (yy < height - border)
        & (xx >= border)
        & (xx < width - border)
    )
    ratio = intensity_i0_yx / np.maximum(reference_yx, 1.0e-8)
    values = ratio[mask]
    if values.size < 1000:
        mask = (
            np.isfinite(ratio)
            & np.isfinite(confidence_yx)
            & (confidence_yx >= 0.5)
            & (radius >= 280.0)
        )
        values = ratio[mask]
    lower, upper = np.quantile(values, (0.15, 0.85))
    selected = values[(values >= lower) & (values <= upper)]
    level = float(np.median(selected))
    reference_values = reference_yx[mask].astype(np.float64)
    image_values = intensity_i0_yx[mask].astype(np.float64)
    confidence_values = np.clip(confidence_yx[mask].astype(np.float64), 0.0, 1.0)
    for _iteration in range(8):
        residual = image_values - level * reference_values
        sigma = max(robust_sigma(residual), 1.0e-8)
        limit = 3.5 * sigma
        huber = np.minimum(1.0, limit / np.maximum(np.abs(residual), limit))
        weight = confidence_values * huber
        updated = float(
            np.sum(weight * reference_values * image_values, dtype=np.float64)
            / max(
                np.sum(weight * reference_values * reference_values, dtype=np.float64),
                np.finfo(float).tiny,
            )
        )
        if abs(updated - level) <= 1.0e-7 * max(abs(level), 1.0):
            level = updated
            break
        level = updated
    residual = image_values - level * reference_values
    relative_residual = float(
        robust_sigma(residual)
        / max(
            abs(level) * float(np.median(reference_values)),
            np.finfo(float).tiny,
        )
    )
    return level, mask, relative_residual


def _save_reference(
    output_path: Path,
    dark: dict[int, dict[str, DarkFrameCalibration]],
    reference_maps: np.ndarray,
    reference: RankOneReference,
    labels: list[tuple[int, int, str]],
    offsets_xy: dict[str, tuple[float, float]],
    offset_spreads_xy: dict[str, tuple[float, float]],
    ground_metrics: dict[str, Any],
    full_scale: float,
) -> None:
    dark_stack = np.stack(
        [dark[exposure][name].dark_yx for exposure in EXPOSURES for name in CHANNELS]
    ).reshape(len(EXPOSURES), len(CHANNELS), *reference_maps.shape[1:])
    hot_stack = np.stack(
        [dark[exposure][name].hot_mask_yx for exposure in EXPOSURES for name in CHANNELS]
    ).reshape(len(EXPOSURES), len(CHANNELS), *reference_maps.shape[1:])
    np.savez_compressed(
        output_path,
        reference_full_yx=np.flip(reference_maps[0], axis=1),
        reference_smooth_yx=np.flip(reference_maps[1], axis=1),
        raw_dark_yx=dark_stack.astype(np.float32),
        raw_hot_pixel_mask_yx=hot_stack,
        exposure_ms=np.asarray(EXPOSURES, dtype=np.int16),
        channel_names=np.asarray(CHANNELS),
        raw_channel_names=np.asarray(CHANNELS),
        ground_labels=np.asarray([f"{e}ms:{index}:{name}" for e, index, name in labels]),
        ground_amplitudes=reference.amplitudes,
        ground_relative_residuals=reference.relative_residuals,
        singular_energy_fraction=reference.singular_energy_fraction,
        channel_offsets_xy_px=np.asarray([offsets_xy[name] for name in CHANNELS]),
        channel_offset_spreads_xy_px=np.asarray([offset_spreads_xy[name] for name in CHANNELS]),
        encoded_full_scale=np.asarray(full_scale),
        ground_metrics_json=np.asarray(json.dumps(ground_metrics, ensure_ascii=True)),
        coordinate_convention=np.asarray(
            "project image[y,x]; mirror_x applied relative to raw camera coordinates"
        ),
        raw_calibration_coordinate_convention=np.asarray(
            "raw image[y,x]; dark maps retain native macro-quadrant coordinates"
        ),
    )


def _assemble_frames(
    paths: dict[int, dict[int, Path]],
    dark: dict[int, dict[str, DarkFrameCalibration]],
    offsets_xy: dict[str, tuple[float, float]],
    centers: dict[int, dict[int, tuple[float, float]]],
    frame_scales: dict[int, float],
    reference_smooth_yx: np.ndarray,
    saturation_soft: float,
    frames_output: Path,
) -> dict[str, Any]:
    frame_numbers = sorted(set(paths[12]) & set(paths[42]))
    saturation_fraction = np.empty((len(frame_numbers), len(EXPOSURES), len(CHANNELS)))
    invalid_fraction = np.empty((len(frame_numbers), len(CHANNELS)))
    low_fraction = np.empty((len(frame_numbers), len(CHANNELS)))
    forward_residual = np.empty((len(frame_numbers), len(EXPOSURES), len(CHANNELS)))
    incident_levels = np.empty(len(frame_numbers))
    incident_fit_residuals = np.empty(len(frame_numbers))
    exposure_shifts = np.empty((len(frame_numbers), 2))
    project_centers = np.empty((len(frame_numbers), 2))

    for position, frame_number in enumerate(frame_numbers):
        low_raw_native, low_native = _corrected_channels(paths[12][frame_number], dark[12])
        high_raw_native, high_native = _corrected_channels(paths[42][frame_number], dark[42])
        target = centers[42][frame_number]
        source = centers[12][frame_number]
        exposure_shift = (target[0] - source[0], target[1] - source[1])
        exposure_shifts[position] = exposure_shift
        low_raw, low = _align_channels(
            low_raw_native,
            low_native,
            offsets_xy,
            exposure_shift_xy=exposure_shift,
        )
        high_raw, high = _align_channels(high_raw_native, high_native, offsets_xy)
        scale = frame_scales[frame_number]

        intensity = np.empty((*reference_smooth_yx.shape, len(CHANNELS)), dtype=np.float32)
        confidence = np.empty_like(intensity, dtype=np.uint8)
        low_weight = np.empty_like(intensity, dtype=np.uint8)
        for channel_index, name in enumerate(CHANNELS):
            merged = merge_hdr_exposures(
                low[name],
                high[name],
                low_raw_yx=low_raw[name],
                high_raw_yx=high_raw[name],
                low_to_high_scale=scale,
                high_soft_threshold=saturation_soft,
            )
            intensity[..., channel_index] = merged.intensity_yx
            confidence[..., channel_index] = np.rint(255.0 * merged.confidence_yx).astype(np.uint8)
            low_weight[..., channel_index] = np.rint(
                255.0 * merged.low_exposure_fraction_yx
            ).astype(np.uint8)
            invalid_fraction[position, channel_index] = 1.0 - float(np.mean(merged.valid_yx))
            low_fraction[position, channel_index] = float(
                np.nanmean(merged.low_exposure_fraction_yx[merged.valid_yx])
            )
            saturation_fraction[position, 0, channel_index] = float(
                np.nanmean(low_raw[name] >= 0.995)
            )
            saturation_fraction[position, 1, channel_index] = float(
                np.nanmean(high_raw[name] >= 0.995)
            )

            valid_high = (
                np.isfinite(high[name])
                & (high_raw[name] < saturation_soft)
                & merged.valid_yx
            )
            valid_low = (
                np.isfinite(low[name])
                & (low_raw[name] < 0.94)
                & merged.valid_yx
            )
            for exposure_index, (observed, predicted, valid) in enumerate(
                (
                    (low[name], merged.intensity_yx / scale, valid_low),
                    (high[name], merged.intensity_yx, valid_high),
                )
            ):
                residual = observed[valid] - predicted[valid]
                denominator = np.sqrt(np.mean(observed[valid] ** 2))
                forward_residual[position, exposure_index, channel_index] = float(
                    np.sqrt(np.mean(residual * residual)) / max(denominator, 1.0e-12)
                )

        incident_level, incident_mask, incident_fit_residual = _incident_level(
            intensity[..., CHANNELS.index("I0")],
            reference_smooth_yx,
            target,
            confidence[..., CHANNELS.index("I0")].astype(np.float32) / 255.0,
        )
        incident_levels[position] = incident_level
        incident_fit_residuals[position] = incident_fit_residual
        normalized = raw_camera_to_project(
            np.asarray(intensity / incident_level, dtype=np.float32)
        )
        project_confidence = raw_camera_to_project(confidence)
        project_low_weight = raw_camera_to_project(low_weight)
        project_incident_mask = np.flip(incident_mask, axis=1)
        project_center = raw_camera_center_to_project(
            target,
            width=normalized.shape[1],
        )
        project_centers[position] = project_center
        np.savez_compressed(
            frames_output / f"{frame_number:04d}.npz",
            intensity_yxc=normalized,
            confidence_yxc=project_confidence,
            low_exposure_fraction_yxc=project_low_weight,
            incident_reference_mask_yx=project_incident_mask,
            channel_names=np.asarray(CHANNELS),
            frame_number=np.asarray(frame_number, dtype=np.int16),
            center_xy_px=project_center,
            raw_center_xy_px=np.asarray(target, dtype=np.float64),
            low_to_high_scale=np.asarray(scale, dtype=np.float64),
            incident_level_raw_42ms_equivalent=np.asarray(incident_level, dtype=np.float64),
            incident_fit_relative_residual=np.asarray(
                incident_fit_residual,
                dtype=np.float64,
            ),
            schema_version=np.asarray(FOCUS_HDR_SCHEMA_VERSION),
            units=np.asarray("dimensionless detector intensity relative to frame I0 background"),
            coordinate_convention=np.asarray(
                "project image[y,x], mirror_x applied, analyzer labels transformed covariantly"
            ),
            normalization=np.asarray(
                "dark-corrected HDR divided once by a common robust frame incident level; no Ground division"
            ),
        )
        if position % 8 == 0 or position + 1 == len(frame_numbers):
            print(f"[HDR] {position + 1}/{len(frame_numbers)}")

    return {
        "frame_numbers": frame_numbers,
        "incident_levels": incident_levels,
        "incident_fit_relative_residual": incident_fit_residuals,
        "exposure_shifts_xy_px": exposure_shifts,
        "project_centers_xy_px": project_centers,
        "saturation_fraction": saturation_fraction,
        "invalid_fraction": invalid_fraction,
        "low_exposure_fraction": low_fraction,
        "forward_residual": forward_residual,
    }


def _plot_image(ax: plt.Axes, image: np.ndarray, title: str, *, cmap: str = "gray") -> None:
    finite = image[np.isfinite(image)]
    if finite.size:
        vmin, vmax = np.quantile(finite, (0.01, 0.99))
    else:
        vmin, vmax = 0.0, 1.0
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()


def _write_pdf_report(
    path: Path,
    *,
    input_root: Path,
    output_data: Path,
    dark: dict[int, dict[str, DarkFrameCalibration]],
    reference: RankOneReference,
    reference_maps: np.ndarray,
    ground_metrics: dict[str, Any],
    offsets_xy: dict[str, tuple[float, float]],
    offset_spreads_xy: dict[str, tuple[float, float]],
    centers: dict[int, dict[int, tuple[float, float]]],
    frame_scales: dict[int, float],
    frame_fits: dict[int, dict[str, Any]],
    saturation_curve: dict[str, Any],
    assembly: dict[str, Any],
) -> None:
    frame_numbers = np.asarray(assembly["frame_numbers"])
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.text(0.08, 0.94, "HDR focus-stack calibration report", fontsize=16, weight="bold")
        summary = (
            f"Input: {_relative(input_root)}\n"
            f"Output: {_relative(output_data)}\n\n"
            f"Frames: {len(frame_numbers)}; channels: {', '.join(CHANNELS)}\n"
            f"Ground exposure ratio 42/12: {ground_metrics['exposure_ratio']:.5f} "
            f"+/- {ground_metrics['exposure_ratio_mad']:.5f}\n"
            f"Frame-wise ratio median: {saturation_curve['frame_scale_median']:.5f} "
            f"+/- {saturation_curve['frame_scale_mad']:.5f}\n"
            f"High-exposure soft saturation threshold: {saturation_curve['soft_threshold']:.3f}\n"
            f"Ground first-mode energy: {ground_metrics['rank_one_first_energy_fraction']:.6f}\n"
            f"Ground rank-1 median residual: {ground_metrics['rank_one_residual_median']:.4f}\n"
            f"I90 leakage/I0: {ground_metrics['i90_leakage_relative_to_i0']:.5f}; "
            f"95% upper estimate {ground_metrics['i90_leakage_upper_95']:.5f}\n\n"
            "The stored HDR is dark-corrected, registered, transformed to the project basis,\n"
            "and divided by one robust I0 incident level per frame. The full and low-pass\n"
            "Ground maps are saved only as diagnostics and are not divided from the data."
        )
        fig.text(0.08, 0.88, summary, va="top", fontsize=10, linespacing=1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6), constrained_layout=True)
        for row, exposure in enumerate(EXPOSURES):
            for column, name in enumerate(CHANNELS):
                _plot_image(
                    axes[row, column],
                    dark[exposure][name].dark_yx * 255.0,
                    f"{exposure} ms, {name}",
                    cmap="magma",
                )
        fig.suptitle("Denoised dark response (8-bit counts)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.1), constrained_layout=True)
        _plot_image(axes[0], reference_maps[0], "Robust common Ground")
        _plot_image(axes[1], reference_maps[1], "Low-pass reference")
        difference = reference_maps[0] - reference_maps[1]
        limit = float(np.quantile(np.abs(difference), 0.99))
        axes[2].imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper")
        axes[2].set_title("Full minus low-pass")
        axes[2].set_axis_off()
        axes[3].semilogy(
            np.arange(1, min(12, reference.singular_energy_fraction.size) + 1),
            reference.singular_energy_fraction[:12],
            "o-",
        )
        axes[3].set_xlabel("Uncentered singular component")
        axes[3].set_ylabel("Fraction of squared singular values")
        axes[3].grid(alpha=0.25)
        fig.suptitle("Reference background from all bright Ground channels")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
        ratios = ground_metrics["channel_background_ratio_to_i0"]
        ratio_mad = ground_metrics["channel_background_ratio_mad"]
        axes[0].bar(
            BRIGHT_CHANNELS,
            [ratios[name] for name in BRIGHT_CHANNELS],
            yerr=[ratio_mad[name] for name in BRIGHT_CHANNELS],
            capsize=3,
        )
        axes[0].axhline(0.5, color="0.4", linestyle=":", linewidth=1.0)
        axes[0].axhline(1.0, color="0.4", linestyle=":", linewidth=1.0)
        axes[0].set_ylabel("Ground amplitude / I0 amplitude")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].hist(reference.relative_residuals, bins=16, color="0.35")
        axes[1].set_xlabel("Relative rank-1 residual")
        axes[1].set_ylabel("Ground-channel images")
        axes[1].grid(axis="y", alpha=0.25)
        agreement = ground_metrics["lowpass_channel_mode_agreement"]
        axes[2].bar(
            BRIGHT_CHANNELS,
            [agreement[name]["correlation"] for name in BRIGHT_CHANNELS],
        )
        axes[2].set_ylim(0.98, 1.0005)
        axes[2].set_ylabel("Low-pass correlation with common mode")
        axes[2].grid(axis="y", alpha=0.25)
        fig.suptitle("Ground channel factors and rank-1 residuals")
        fig.text(
            0.5,
            -0.01,
            "Leave-one-Ground-out: minimum correlation "
            f"{ground_metrics['leave_one_ground_out_correlation_minimum']:.4f}; "
            "maximum relative RMS "
            f"{ground_metrics['leave_one_ground_out_relative_rms_maximum']:.4f}",
            ha="center",
            fontsize=9,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
        centers_bin = np.asarray(saturation_curve["bin_centers"])
        ratio_bin = np.asarray(saturation_curve["median_high_over_scaled_low"], dtype=float)
        baseline = float(saturation_curve["baseline"])
        axes[0].plot(centers_bin, ratio_bin / baseline, "o-", markersize=3)
        axes[0].axhline(1.0, color="0.4", linestyle=":")
        axes[0].axvline(saturation_curve["soft_threshold"], color="tab:red", linestyle="--")
        axes[0].set(xlabel="42-ms encoded intensity", ylabel="42 ms / scaled 12 ms")
        axes[0].set_ylim(0.75, 1.08)
        axes[0].grid(alpha=0.25)
        axes[1].plot(frame_numbers, [frame_scales[int(number)] for number in frame_numbers])
        axes[1].axhline(ground_metrics["exposure_ratio"], color="0.4", linestyle=":")
        axes[1].set(xlabel="Frame", ylabel="Fitted 12-to-42-ms scale")
        axes[1].grid(alpha=0.25)
        fig.suptitle("Exposure linearity and frame-wise radiometric scale")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
        for exposure in EXPOSURES:
            values = np.asarray([centers[exposure][int(number)] for number in frame_numbers])
            axes[0].plot(frame_numbers, values[:, 0], label=f"x, {exposure} ms")
            axes[0].plot(frame_numbers, values[:, 1], label=f"y, {exposure} ms")
        axes[0].set(xlabel="Frame", ylabel="Center coordinate (px)")
        axes[0].legend(ncol=2, fontsize=8)
        axes[0].grid(alpha=0.25)
        shifts = np.asarray(assembly["exposure_shifts_xy_px"])
        axes[1].plot(frame_numbers, shifts[:, 0], label="x")
        axes[1].plot(frame_numbers, shifts[:, 1], label="y")
        axes[1].set(xlabel="Frame", ylabel="12-to-42-ms registration shift (px)")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.suptitle("Spatial registration")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
        saturation = np.asarray(assembly["saturation_fraction"])
        low_fraction = np.asarray(assembly["low_exposure_fraction"])
        invalid = np.asarray(assembly["invalid_fraction"])
        for channel_index, name in enumerate(CHANNELS):
            axes[0].plot(frame_numbers, saturation[:, 1, channel_index], label=name)
            axes[1].plot(frame_numbers, low_fraction[:, channel_index], label=name)
            axes[2].plot(frame_numbers, invalid[:, channel_index], label=name)
        axes[0].set(xlabel="Frame", ylabel="42-ms hard-saturated fraction")
        axes[1].set(xlabel="Frame", ylabel="Mean 12-ms HDR fraction")
        axes[2].set(xlabel="Frame", ylabel="Invalid HDR fraction")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle("HDR coverage and exposure contribution")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        selected_positions = np.linspace(0, len(frame_numbers) - 1, 5).round().astype(int)
        selected_frames = frame_numbers[selected_positions]
        fig, axes = plt.subplots(3, len(selected_frames), figsize=(11.0, 6.7), constrained_layout=True)
        for column, frame_number in enumerate(selected_frames):
            with np.load(output_data / "frames" / f"{int(frame_number):04d}.npz") as data:
                intensity = np.asarray(data["intensity_yxc"])
                confidence = np.asarray(data["confidence_yxc"], dtype=np.float32) / 255.0
                low_weight = np.asarray(data["low_exposure_fraction_yxc"], dtype=np.float32) / 255.0
            _plot_image(axes[0, column], intensity[..., 0], f"Frame {frame_number}")
            _plot_image(axes[1, column], intensity[..., 3], "")
            axes[2, column].imshow(low_weight[..., 0], vmin=0.0, vmax=1.0, cmap="viridis")
            axes[2, column].contour(confidence[..., 0] < 0.25, levels=(0.5,), colors="white", linewidths=0.4)
            axes[2, column].set_axis_off()
        axes[0, 0].text(
            -0.08,
            0.5,
            "I0",
            transform=axes[0, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
        )
        axes[1, 0].text(
            -0.08,
            0.5,
            "I90",
            transform=axes[1, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
        )
        axes[2, 0].text(
            -0.08,
            0.5,
            "12-ms fraction",
            transform=axes[2, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
        )
        fig.suptitle(
            "HDR stack in raw camera orientation; white contours mark confidence below 0.25"
        )
        pdf.savefig(fig, bbox_inches="tight", dpi=180)
        plt.close(fig)

        forward = np.asarray(assembly["forward_residual"])
        fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
        for channel_index, name in enumerate(CHANNELS):
            ax.plot(
                frame_numbers,
                forward[:, 0, channel_index],
                label=name,
            )
        ax.set(
            xlabel="Frame",
            ylabel="Relative RMS disagreement",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.suptitle(
            "Registered 12-ms disagreement with the unsaturated 42-ms primary signal"
        )
        fig.text(
            0.5,
            -0.01,
            "Below the soft-saturation threshold the HDR preserves the 42-ms measurement exactly.",
            ha="center",
            fontsize=9,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-data", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT_OUTPUT)
    add_force_argument(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    input_root = args.input.expanduser().resolve()
    output_data = args.output_data.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    populated_outputs = tuple(
        path
        for path in (output_data, output_report)
        if path.exists() and any(path.iterdir())
    )
    refuse_existing_outputs(
        populated_outputs,
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    output_data.mkdir(parents=True, exist_ok=True)
    output_report.mkdir(parents=True, exist_ok=True)
    frames_output = output_data / "frames"
    frames_output.mkdir(parents=True, exist_ok=True)

    paths = {
        exposure: _numeric_paths(input_root / f"Exposure_{exposure}ms")
        for exposure in EXPOSURES
    }
    common_frames = sorted(set(paths[12]) & set(paths[42]))
    if sorted(paths[12]) != common_frames or sorted(paths[42]) != common_frames:
        raise ValueError("The two exposure directories must contain identical frame indices")
    if tuple(common_frames) != DIRECT_POM_CONFIG.selected_frame_numbers:
        raise ValueError(
            "Direct-POM frames disagree with data_src/experiment.toml: "
            f"{common_frames!r} != {DIRECT_POM_CONFIG.selected_frame_numbers!r}"
        )

    dark, full_scale = _load_dark_calibration(input_root)
    offsets_xy, offset_spreads_xy = _estimate_channel_offsets(paths, dark)
    centers = _estimate_frame_centers(paths, dark)
    reference, reference_maps, labels, ground_metrics = _build_ground_reference(
        input_root,
        dark,
        offsets_xy,
    )
    _save_reference(
        output_data / "reference_background.npz",
        dark,
        reference_maps,
        reference,
        labels,
        offsets_xy,
        offset_spreads_xy,
        ground_metrics,
        full_scale,
    )

    frame_scales, frame_fits, saturation_curve = _first_pass_scales(
        paths,
        dark,
        offsets_xy,
        centers,
    )
    assembly = _assemble_frames(
        paths,
        dark,
        offsets_xy,
        centers,
        frame_scales,
        reference_maps[1],
        float(saturation_curve["soft_threshold"]),
        frames_output,
    )

    report = {
        "schema_version": FOCUS_HDR_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "experiment_manifest": _relative(DEFAULT_EXPERIMENT_MANIFEST),
            "input_root": _relative(input_root),
            "environment": {
                str(exposure): _relative(
                    input_root / f"Exposure_{exposure}ms" / "Environment.png"
                )
                for exposure in EXPOSURES
            },
            "frame_count": len(common_frames),
            "frame_numbers": common_frames,
            "source_policy": "read-only acquisition-native archive",
        },
        "outputs": {
            "data_root": _relative(output_data),
            "frames": _relative(frames_output),
            "reference_background": _relative(
                output_data / "reference_background.npz"
            ),
            "pdf_report": _relative(output_report / "hdr_calibration_report.pdf"),
        },
        "dark": {
            str(exposure): {
                name: {
                    "residual_sigma": dark[exposure][name].residual_sigma,
                    "hot_pixel_count": int(np.count_nonzero(dark[exposure][name].hot_mask_yx)),
                }
                for name in CHANNELS
            }
            for exposure in EXPOSURES
        },
        "registration": {
            "channel_offsets_xy_px": {name: list(offsets_xy[name]) for name in CHANNELS},
            "channel_offset_spreads_xy_px": {
                name: list(offset_spreads_xy[name]) for name in CHANNELS
            },
            "raw_centers_xy_px": {
                str(exposure): [list(centers[exposure][number]) for number in common_frames]
                for exposure in EXPOSURES
            },
            "project_centers_xy_px": np.asarray(
                assembly["project_centers_xy_px"]
            ).tolist(),
            "low_to_high_shifts_xy_px": np.asarray(
                assembly["exposure_shifts_xy_px"]
            ).tolist(),
        },
        "reference_background": ground_metrics,
        "radiometry": {
            "encoded_full_scale": full_scale,
            "frame_low_to_high_scale": {
                str(number): frame_scales[number] for number in common_frames
            },
            "frame_channel_scale_fits": {
                str(number): frame_fits[number] for number in common_frames
            },
            "saturation_curve": saturation_curve,
            "ground_reference_expectation": EXPECTED_FREE_FIELD,
            "detector_gain_policy": (
                "No scalar analyzer correction exists in the data contract. Stokes closure improves only "
                "marginally under fitted 1-3% factors, below the systematic closure residual."
            ),
            "beamsplitter_field_response": {
                "model": "E_detector = diag(1,q) E_sample up to a common amplitude",
                "q_real": BS2_OUTPUT_AMPLITUDE_RATIO,
                "q_imag": 0.0,
                "source": (
                    "acquisition metadata gives a sqrt(2) amplitude ratio; independently "
                    "consistent with direct I90 versus corrected holography"
                ),
                "application": "not applied to stored HDR; included coherently in forward optical models",
            },
        },
        "hdr_quality": {
            "incident_level_raw_42ms_equivalent": np.asarray(
                assembly["incident_levels"]
            ).tolist(),
            "incident_fit_relative_residual": np.asarray(
                assembly["incident_fit_relative_residual"]
            ).tolist(),
            "saturation_fraction": np.asarray(assembly["saturation_fraction"]).tolist(),
            "invalid_fraction": np.asarray(assembly["invalid_fraction"]).tolist(),
            "mean_low_exposure_fraction": np.asarray(
                assembly["low_exposure_fraction"]
            ).tolist(),
            "forward_relative_residual": np.asarray(assembly["forward_residual"]).tolist(),
        },
        "interpretation": {
            "primary_units": "dimensionless detector intensity relative to the robust frame I0 background",
            "normalization": (
                "Each registered HDR frame is divided once by one common robust I0 incident level. "
                "No channel-specific gain, affine correction, or pointwise Ground division is applied."
            ),
            "reference_application": (
                "reference_smooth_yx is used only as a robust template for estimating the scalar incident level; "
                "reference_full_yx and reference_smooth_yx remain diagnostics and are neither subtracted nor divided"
            ),
            "negative_values": "retained after dark correction",
            "stored_coordinate_basis": "project basis with mirror_x already applied",
            "raw_to_project_source_channel": RAW_TO_PROJECT_SOURCE_CHANNEL,
            "invalid_pixels": "confidence_yxc == 0",
            "confidence_encoding": "uint8 maps linearly to [0,1]",
            "low_exposure_fraction_encoding": "uint8 maps linearly to [0,1]",
        },
    }
    _write_json(output_report / "report.json", report)
    _write_json(output_data / "manifest.json", report)
    (output_data / "README.md").write_text(
        """# Registered 12/42-ms HDR focus stack

This derived data set is generated by
`code/scripts/build_focus_stack_hdr.py` from the minimal source images under
`data_src/direct_pom/520nm/`.

`frames/NNNN.npz` stores `intensity_yxc` as dimensionless detector intensity,
already divided by one robust `I0` incident level common to all four channels.
`confidence_yxc` and
`low_exposure_fraction_yxc` are uint8 encodings of values on `[0,1]`.
Channels are ordered by `channel_names`; all are registered to the 42-ms `I0`
image and transformed once to the project coordinate and polarization basis.
The stored `center_xy_px` is expressed in that same basis.

`reference_background.npz` stores the denoised dark maps, hot-pixel masks,
channel offsets, the robust common Ground mode, and its low-pass version.
The smooth reference is used only to estimate the single frame incident level.
Neither the full nor the smooth Ground map is divided from the stored HDR:
doing so would remove real coherent illumination structure that changes under
propagation.  The raw fitted scalar is retained as
`incident_level_raw_42ms_equivalent` for provenance, not for downstream use.

No scalar analyzer gains are applied: the fitted 1--3 percent Stokes-closure
corrections improve the residual only marginally and are retained only in the
calibration sensitivity report.  The polarization-dependent response of BS2 is
a coherent field transformation, not an `I90` detector gain.  Forward models
use `diag(1, sqrt(2))` up to a common amplitude.  The raw `I45` and `I135`
channels are exchanged exactly once while transforming to the mirrored project
basis.
Complete provenance, numerical factors, uncertainties, and interpretation
limits are recorded in `manifest.json` and the paired report under
`data/analysis/results_toron25um_50X_focus_hdr/`.
""",
        encoding="utf-8",
    )
    _write_pdf_report(
        output_report / "hdr_calibration_report.pdf",
        input_root=input_root,
        output_data=output_data,
        dark=dark,
        reference=reference,
        reference_maps=reference_maps,
        ground_metrics=ground_metrics,
        offsets_xy=offsets_xy,
        offset_spreads_xy=offset_spreads_xy,
        centers=centers,
        frame_scales=frame_scales,
        frame_fits=frame_fits,
        saturation_curve=saturation_curve,
        assembly=assembly,
    )
    print(f"Wrote HDR data to {output_data}")
    print(f"Wrote calibration report to {output_report}")


if __name__ == "__main__":
    main()
