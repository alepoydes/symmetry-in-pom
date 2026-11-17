#!/usr/bin/env python3
"""Measure layer helicity and its longitudinal gradient in the supplied director."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

from toronoptics.article_data import (
    ARTICLE_VARIANT_SUFFIX,
    model_director_path,
)
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.simulate_jones import load_director_archive

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIRECTOR_SOURCE = model_director_path(PROJECT_ROOT)
ANALYSIS_DIR = (
    PROJECT_ROOT / f"data/analysis/director_helicity{ARTICLE_VARIANT_SUFFIX}"
)
ANALYSIS_ARCHIVE = ANALYSIS_DIR / "director_helicity_analysis.npz"
ANALYSIS_REPORT = ANALYSIS_DIR / "director_helicity_analysis.json"
SCHEMA_VERSION = f"director-layer-helicity-v1{ARTICLE_VARIANT_SUFFIX}"

WAVELENGTH_UM = 0.662
ORDINARY_INDEX = 1.52993
EXTRAORDINARY_INDEX = 1.70178
INITIAL_CENTER_PX = (99.5, 99.5)
CENTER_SCAN_RADIUS_PX = 2.0
CENTER_SCAN_STEP_PX = 0.5
VISIBILITY_WEIGHT_FLOOR = 0.01


def _relative(path):
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _director_field(path):
    archive = load_director_archive(path)
    field = np.asarray(archive.director, dtype=np.float64)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Expected director[x,y,z,3], got {field.shape!r}")
    manifest = archive.source_manifest or {}
    thickness_um = float(manifest["problem"]["geometry"]["thickness_um"])
    return field, manifest, thickness_um

def _constant_chirality(manifest):
    for block in manifest.get("problem", {}).get("blocks", []):
        if block.get("kind") == "constant_chirality":
            return float(block["q_um_inv"])
    raise ValueError("Director manifest does not define constant chirality")

def _spatial_phase(shape_xy, center_px):
    x, y = np.indices(shape_xy, dtype=np.float64)
    dx = x - float(center_px[0])
    dy = y - float(center_px[1])
    phase = np.exp(-2.0j * np.arctan2(dy, dx))
    phase[dx * dx + dy * dy < 0.25] = 0.0
    return phase

def _center_score(nematic_xy, projection_power, center_px):
    phase = _spatial_phase(nematic_xy.shape[:2], center_px)
    moment = phase.ravel() @ nematic_xy.reshape(-1, nematic_xy.shape[2])
    total_weight = float(np.sum(projection_power))
    return float(np.sum(np.abs(moment)) / max(total_weight, np.finfo(float).tiny))

def _fit_center(
    field,
    initial_center_px,
    *,
    scan_radius_px,
    scan_step_px,
):
    nematic_xy = (field[..., 0] + 1.0j * field[..., 1]) ** 2
    projection_power = field[..., 0] ** 2 + field[..., 1] ** 2
    offsets = np.arange(
        -float(scan_radius_px),
        float(scan_radius_px) + 0.5 * float(scan_step_px),
        float(scan_step_px),
        dtype=np.float64,
    )
    score_map = np.empty((offsets.size, offsets.size), dtype=np.float64)
    initial = np.asarray(initial_center_px, dtype=np.float64)
    for ix, dx in enumerate(offsets):
        for iy, dy in enumerate(offsets):
            score_map[ix, iy] = _center_score(
                nematic_xy,
                projection_power,
                initial + np.asarray((dx, dy)),
            )
    best_grid_index = np.unravel_index(np.argmax(score_map), score_map.shape)
    best_grid = initial + np.asarray(
        (offsets[best_grid_index[0]], offsets[best_grid_index[1]])
    )
    fit = minimize(
        lambda center: -_center_score(nematic_xy, projection_power, center),
        best_grid,
        method="Nelder-Mead",
        options={"xatol": 1.0e-5, "fatol": 1.0e-11, "maxiter": 160},
    )
    return {
        "center_px": np.asarray(fit.x, dtype=np.float64),
        "coherence": float(-fit.fun),
        "converged": bool(fit.success),
        "offsets_px": offsets,
        "score_map": score_map,
    }

def _effective_birefringence(field, wavelength_um, no, ne):
    transverse_power = field[..., 0] ** 2 + field[..., 1] ** 2
    denominator = np.sqrt(
        field[..., 2] ** 2 + (float(no) / float(ne)) ** 2 * transverse_power
    )
    neff = float(no) / np.maximum(denominator, np.finfo(float).tiny)
    gamma = (2.0 * np.pi / float(wavelength_um)) * np.maximum(neff - float(no), 0.0)
    return transverse_power, gamma

def _layer_moments(field, center_px, wavelength_um, no, ne):
    phase = _spatial_phase(field.shape[:2], center_px)
    director_xy = field[..., 0] + 1.0j * field[..., 1]
    nematic_xy = director_xy**2 * phase[..., None]
    transverse_power, gamma = _effective_birefringence(
        field,
        wavelength_um,
        no,
        ne,
    )

    geometric_weight = transverse_power
    geometric_moment = np.sum(nematic_xy, axis=(0, 1))

    unit_nematic = np.zeros_like(nematic_xy)
    active = transverse_power > 64.0 * np.finfo(float).eps
    unit_nematic[active] = nematic_xy[active] / transverse_power[active]
    optical_weight = gamma
    optical_moment = np.sum(optical_weight * unit_nematic, axis=(0, 1))

    return {
        "geometric": {
            "weight": np.sum(geometric_weight, axis=(0, 1)),
            "moment": geometric_moment,
        },
        "optical": {
            "weight": np.sum(optical_weight, axis=(0, 1)),
            "moment": optical_moment,
        },
    }

def _layer_summary(moment, weight):
    moment = np.asarray(moment, dtype=np.complex128)
    weight = np.asarray(weight, dtype=np.float64)
    coherence = np.divide(
        np.abs(moment),
        weight,
        out=np.zeros_like(weight),
        where=weight > np.finfo(float).tiny,
    )
    coherence = np.clip(coherence, 0.0, 1.0)
    phase = np.unwrap(np.angle(moment))
    tau = 0.5 * phase
    tau[weight <= np.finfo(float).tiny] = np.nan
    spread = 0.5 * np.sqrt(-2.0 * np.log(np.clip(coherence, 1.0e-12, 1.0)))
    spread[weight <= np.finfo(float).tiny] = np.nan
    return tau, coherence, spread

def _layer_helicity_extrema(field, center_px, mean_tau, weight_floor):
    phase = _spatial_phase(field.shape[:2], center_px)
    director_xy = field[..., 0] + 1.0j * field[..., 1]
    transverse_power = np.abs(director_xy) ** 2
    local_nematic = director_xy**2 * phase[..., None]
    minimum = np.full(field.shape[2], np.nan, dtype=np.float64)
    maximum = np.full(field.shape[2], np.nan, dtype=np.float64)
    for index in range(field.shape[2]):
        weight = transverse_power[..., index]
        threshold = float(weight_floor) * float(np.max(weight))
        active = (weight >= threshold) & (weight > 64.0 * np.finfo(float).eps)
        if not np.any(active) or not np.isfinite(mean_tau[index]):
            continue
        deviation = 0.5 * np.angle(
            local_nematic[..., index][active]
            * np.exp(-2.0j * float(mean_tau[index]))
        )
        minimum[index] = float(mean_tau[index] + np.min(deviation))
        maximum[index] = float(mean_tau[index] + np.max(deviation))
    return minimum, maximum

def _fit_linear_helicity(moment, z_fraction):
    moment = np.asarray(moment, dtype=np.complex128)
    magnitude = np.abs(moment)
    valid = magnitude > 1.0e-12 * np.nanmax(magnitude)
    unit = np.zeros_like(moment)
    unit[valid] = moment[valid] / magnitude[valid]
    phase_tau = 0.5 * np.unwrap(np.angle(moment[valid]))
    centered = np.asarray(z_fraction, dtype=np.float64) - 0.5
    full_design = np.column_stack((np.ones_like(centered), centered))
    fit_weight = magnitude[valid]
    sqrt_weight = np.sqrt(fit_weight / np.sum(fit_weight))

    def fit_design(design):
        fit_weight = magnitude[valid]
        initial = np.linalg.lstsq(
            design * sqrt_weight[:, None],
            phase_tau * sqrt_weight,
            rcond=None,
        )[0]

        def residual(parameters):
            model = design @ parameters
            difference = np.exp(2.0j * model) - unit[valid]
            return np.concatenate(
                (sqrt_weight * difference.real, sqrt_weight * difference.imag)
            )

        fit = least_squares(
            residual,
            initial,
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=2000,
        )
        prediction_valid = design @ fit.x
        difference = np.exp(2.0j * prediction_valid) - unit[valid]
        objective = float(
            np.sum(fit_weight * np.abs(difference) ** 2) / np.sum(fit_weight)
        )
        return fit, prediction_valid, objective

    constant_design = np.ones((np.count_nonzero(valid), 1), dtype=np.float64)
    constant_fit, _, baseline_objective = fit_design(constant_design)
    linear_fit, _, objective = fit_design(full_design[valid])
    prediction = full_design @ linear_fit.x
    residual = np.full_like(centered, np.nan)
    residual[valid] = 0.5 * np.angle(
        unit[valid] * np.exp(-2.0j * prediction[valid])
    )
    rms = float(
        np.sqrt(
            np.sum(fit_weight * residual[valid] ** 2)
            / np.sum(fit_weight)
        )
    )
    return {
        "constant": float(constant_fit.x[0]),
        "coefficients": np.asarray(linear_fit.x, dtype=np.float64),
        "prediction": prediction,
        "residual": residual,
        "objective": objective,
        "explained_fraction": float(
            1.0
            - objective / max(baseline_objective, np.finfo(float).tiny)
        ),
        "rms": rms,
        "converged": bool(linear_fit.success),
    }

def compute_helicity_analysis(
    source,
    *,
    initial_center_px,
    center_scan_radius_px,
    center_scan_step_px,
    wavelength_um,
    no,
    ne,
    visibility_weight_floor,
):
    field, manifest, thickness_um = _director_field(source)
    q0_um_inv = _constant_chirality(manifest)
    spacing_um = thickness_um / float(field.shape[2] - 1)
    center_result = _fit_center(
        field,
        initial_center_px,
        scan_radius_px=center_scan_radius_px,
        scan_step_px=center_scan_step_px,
    )
    center_px = center_result["center_px"]
    moments = _layer_moments(field, center_px, wavelength_um, no, ne)
    z_um = np.linspace(0.0, thickness_um, field.shape[2], dtype=np.float64)
    z_fraction = z_um / thickness_um

    weighting_names = ("geometric", "optical")
    layer_weight = []
    layer_moment = []
    layer_tau = []
    layer_coherence = []
    layer_spread = []
    fit_objective = []
    fit_explained = []
    fit_coefficients = []
    fit_prediction = []
    fit_residual = []
    fit_rms = []
    for name in weighting_names:
        weight = moments[name]["weight"]
        moment = moments[name]["moment"]
        tau, coherence, spread = _layer_summary(moment, weight)
        fit = _fit_linear_helicity(moment, z_fraction)
        layer_weight.append(weight)
        layer_moment.append(moment)
        layer_tau.append(tau)
        layer_coherence.append(coherence)
        layer_spread.append(spread)
        fit_objective.append(fit["objective"])
        fit_explained.append(fit["explained_fraction"])
        fit_coefficients.append(fit["coefficients"])
        fit_prediction.append(fit["prediction"])
        fit_residual.append(fit["residual"])
        fit_rms.append(fit["rms"])

    # Keep the same branch for the geometric and optical layer means.
    branch_weight = np.abs(np.asarray(layer_moment[0])) + np.abs(
        np.asarray(layer_moment[1])
    )
    branch_delta = np.asarray(layer_tau[0]) - np.asarray(layer_tau[1])
    branch_valid = np.isfinite(branch_delta) & (branch_weight > 0.0)
    branch_shift = np.pi * np.round(
        np.sum(branch_weight[branch_valid] * branch_delta[branch_valid])
        / (np.pi * np.sum(branch_weight[branch_valid]))
    )
    layer_tau[1] = np.asarray(layer_tau[1]) + branch_shift
    fit_prediction[1] = np.asarray(fit_prediction[1]) + branch_shift
    fit_coefficients[1][0] += branch_shift

    layer_minimum, layer_maximum = _layer_helicity_extrema(
        field,
        center_px,
        np.asarray(layer_tau[0]),
        visibility_weight_floor,
    )

    input_paths = (Path(source),)
    inputs = [
        {
            "path": str(_relative(path)),
            "sha256": _sha256(path),
        }
        for path in input_paths
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "inputs": inputs,
        "source_path": _relative(source),
        "director_shape": list(field.shape),
        "thickness_um": thickness_um,
        "spacing_um": spacing_um,
        "wavelength_um": float(wavelength_um),
        "ordinary_index": float(no),
        "extraordinary_index": float(ne),
        "q0_um_inv": q0_um_inv,
        "center_definition": "maximum summed layer nematic coherence",
        "layer_angle_definition": "0.5*arg(sum w*exp(2i*(alpha-phi)))",
        "geometric_weight": "nx^2+ny^2",
        "optical_weight": "2*pi/lambda*(n_eff-no)",
        "longitudinal_model": "tau(z)=tau_mid+g_z*(z-L/2)",
        "source_manifest": manifest,
    }
    data = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "weighting_names": np.asarray(weighting_names),
        "z_um": z_um,
        "q0_um_inv": np.asarray(q0_um_inv),
        "center_px": center_px,
        "center_scan_offsets_px": center_result["offsets_px"],
        "center_coherence_map": center_result["score_map"],
        "layer_weight": np.asarray(layer_weight),
        "layer_moment": np.asarray(layer_moment),
        "layer_tau_rad": np.asarray(layer_tau),
        "layer_coherence": np.asarray(layer_coherence),
        "layer_spread_rad": np.asarray(layer_spread),
        "layer_tau_min_rad": layer_minimum,
        "layer_tau_max_rad": layer_maximum,
        "fit_linear_objective": np.asarray(fit_objective),
        "fit_linear_explained_fraction": np.asarray(fit_explained),
        "fit_linear_coefficients_rad": np.asarray(fit_coefficients),
        "fit_linear_tau_rad": np.asarray(fit_prediction),
        "fit_linear_residual_rad": np.asarray(fit_residual),
        "fit_linear_rms_rad": np.asarray(fit_rms),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2)),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": inputs,
        "source_path": _relative(source),
        "center_px": center_px.tolist(),
        "center_shift_from_initial_px": (
            center_px - np.asarray(initial_center_px, dtype=np.float64)
        ).tolist(),
        "center_coherence": center_result["coherence"],
        "center_fit_converged": center_result["converged"],
        "q0_um_inv": q0_um_inv,
        "weightings": {},
    }
    for index, name in enumerate(weighting_names):
        normalized_weight = data["layer_weight"][index] / np.max(
            data["layer_weight"][index]
        )
        visible = normalized_weight >= 0.01
        report["weightings"][name] = {
            "linear_total_change_deg": float(
                np.rad2deg(data["fit_linear_coefficients_rad"][index, 1])
            ),
            "linear_slope_deg_per_um": float(
                np.rad2deg(data["fit_linear_coefficients_rad"][index, 1])
                / thickness_um
            ),
            "linear_slope_rad_per_um": float(
                data["fit_linear_coefficients_rad"][index, 1]
                / thickness_um
            ),
            "linear_slope_over_q0": float(
                data["fit_linear_coefficients_rad"][index, 1]
                / (thickness_um * q0_um_inv)
            ),
            "linear_explained_fraction": float(
                data["fit_linear_explained_fraction"][index]
            ),
            "linear_rms_deg": float(
                np.rad2deg(data["fit_linear_rms_rad"][index])
            ),
            "coherence_weighted_mean": float(
                np.sum(
                    data["layer_weight"][index]
                    * data["layer_coherence"][index]
                )
                / np.sum(data["layer_weight"][index])
            ),
            "visible_z_range_um": [
                float(np.min(z_um[visible])),
                float(np.max(z_um[visible])),
            ],
        }

    return data, report

def save_analysis(archive_path, report_path, data, report):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    try:
        with temporary_archive.open("wb") as stream:
            np.savez(stream, **data)
        temporary_report.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_archive.replace(archive_path)
        temporary_report.replace(report_path)
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure geometric and optically weighted layer helicity in the director field."
        )
    )
    add_force_argument(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    refuse_existing_outputs(
        (ANALYSIS_ARCHIVE, ANALYSIS_REPORT),
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    helicity_data, helicity_report = compute_helicity_analysis(
        DIRECTOR_SOURCE,
        initial_center_px=INITIAL_CENTER_PX,
        center_scan_radius_px=CENTER_SCAN_RADIUS_PX,
        center_scan_step_px=CENTER_SCAN_STEP_PX,
        wavelength_um=WAVELENGTH_UM,
        no=ORDINARY_INDEX,
        ne=EXTRAORDINARY_INDEX,
        visibility_weight_floor=VISIBILITY_WEIGHT_FLOOR,
    )
    save_analysis(
        ANALYSIS_ARCHIVE,
        ANALYSIS_REPORT,
        helicity_data,
        helicity_report,
    )
    print(f"Analysis archive: {ANALYSIS_ARCHIVE}")
    print(f"Report: {ANALYSIS_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
