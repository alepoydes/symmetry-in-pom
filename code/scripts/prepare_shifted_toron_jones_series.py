#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import shlex
import sys
import warnings
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from toronoptics.axisymmetric_jones import refine_axisymmetric_center
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.jones_ensemble_average import (
    JonesSample,
    average_jones_samples,
    write_jones_average_npz,
)
from toronoptics.jones_registration import (
    JonesRegistrationFeature,
    make_registration_feature,
    register_jones_features,
)
from toronoptics.jones_registration_quality import (
    render_registration_quality_pdf,
    write_registration_quality_archive,
)
from toronoptics.toron_boxes import detect_toron_bounding_boxes
from toronoptics.toron_center_quality import (
    render_center_quality_pdf,
    write_center_quality_archive,
)
from toronoptics.toron_harmonic_centers import analyze_centers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data_src" / "holography"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "experimental" / "results_toron25um_50X_averaged"
)
DEFAULT_ANALYSIS = (
    PROJECT_ROOT / "data" / "analysis" / "results_toron25um_50X_averaging"
)


def measure_combined_center(
    jones: np.ndarray,
    *,
    pixel_size_um: float,
    source_pixel_size_um: float,
) -> dict[str, object]:
    """Measure the average's center without changing its registration anchor."""

    anchor = np.asarray(
        (0.5 * (jones.shape[0] - 1), 0.5 * (jones.shape[1] - 1)),
        dtype=np.float64,
    )
    result = refine_axisymmetric_center(
        jones,
        initial_center_px=(float(anchor[0]), float(anchor[1])),
        pixel_size_um=float(pixel_size_um),
        min_radius_um=4.0,
        max_radius_um=30.0,
        radial_step_um=0.75,
        angular_samples=360,
        broad_radius_px=24.0,
        broad_step_px=2.0,
        coarse_radius_px=2.0,
        coarse_step_px=0.25,
        fine_radius_px=0.75,
        fine_step_px=0.05,
    )
    fitted = np.asarray(result.center_px, dtype=np.float64)
    offset = fitted - anchor
    source_pixel_ratio = float(pixel_size_um) / float(source_pixel_size_um)
    return {
        "method": "minimum normalized corotating m=+/-1 Jones power",
        "registration_anchor_px": anchor.tolist(),
        "fitted_center_px": fitted.tolist(),
        "offset_output_px": offset.tolist(),
        "offset_norm_output_px": float(np.linalg.norm(offset)),
        "offset_source_px": (offset * source_pixel_ratio).tolist(),
        "offset_norm_source_px": float(np.linalg.norm(offset) * source_pixel_ratio),
        "offset_um": (offset * float(pixel_size_um)).tolist(),
        "offset_norm_um": float(np.linalg.norm(offset) * float(pixel_size_um)),
        "objective": float(result.objective),
        "resolution_output_px": 0.05,
        "applied_to_average": False,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _indexed_npz(directory: Path) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            directory.glob("*.npz"),
            key=lambda path: (int(path.stem) if path.stem.isdigit() else 10**9, path.name),
        )
    )
    if not paths:
        raise FileNotFoundError(f"No Jones NPZ files found in {directory}")
    return paths


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _protected_wavelength_outputs(
    wavelength_nm: int,
    *,
    output_dir: Path,
    analysis_dir: Path,
    reuse_centers: bool,
) -> tuple[Path, ...]:
    output_path = output_dir / f"{wavelength_nm}nm_average.npz"
    centers_path = analysis_dir / f"{wavelength_nm}nm_centers.csv"
    report_path = analysis_dir / f"{wavelength_nm}nm_report.json"
    center_quality_path = analysis_dir / f"{wavelength_nm}nm_center_quality.npz"
    center_quality_figure = analysis_dir / f"{wavelength_nm}nm_center_quality.pdf"
    registration_path = analysis_dir / f"{wavelength_nm}nm_registration.npz"
    registration_figure = analysis_dir / f"{wavelength_nm}nm_registration.pdf"
    if reuse_centers:
        return output_path, report_path, registration_path, registration_figure
    return (
        output_path,
        centers_path,
        report_path,
        center_quality_path,
        center_quality_figure,
        registration_path,
        registration_figure,
    )


def process_wavelength(
    wavelength_nm: int,
    paths: tuple[Path, ...],
    *,
    output_dir: Path,
    analysis_dir: Path,
    size_um: float,
    superresolution_factor: float,
    max_radius_cv: float,
    reuse_centers: bool,
    force: bool,
) -> dict[str, object]:
    output_path = output_dir / f"{wavelength_nm}nm_average.npz"
    centers_path = analysis_dir / f"{wavelength_nm}nm_centers.csv"
    report_path = analysis_dir / f"{wavelength_nm}nm_report.json"
    center_quality_path = analysis_dir / f"{wavelength_nm}nm_center_quality.npz"
    center_quality_figure = analysis_dir / f"{wavelength_nm}nm_center_quality.pdf"
    registration_path = analysis_dir / f"{wavelength_nm}nm_registration.npz"
    registration_figure = analysis_dir / f"{wavelength_nm}nm_registration.pdf"
    protected_outputs = _protected_wavelength_outputs(
        wavelength_nm,
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        reuse_centers=reuse_centers,
    )
    conflicts = [path for path in protected_outputs if path.exists()]
    if conflicts and not force:
        raise FileExistsError(
            f"Output already exists: {conflicts[0]}; pass force=True to overwrite"
        )

    centers = []
    rows: list[dict[str, object]] = []
    pixel_sizes = []
    center_analyses = []
    registration_features: list[JonesRegistrationFeature] = []
    if reuse_centers:
        if not centers_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing center table: {centers_path}")
        with centers_path.open(newline="", encoding="utf-8") as stream:
            saved_rows = list(csv.DictReader(stream))
        if len(saved_rows) != len(paths):
            raise ValueError(
                f"{centers_path} has {len(saved_rows)} rows for {len(paths)} source files"
            )
        rows = [dict(row) for row in saved_rows]
        for source, row in zip(paths, saved_rows, strict=True):
            if Path(row["source"]).name != source.name:
                raise ValueError(f"Center row {row['source']} does not match {source}")
            archive = load_jones_archive(source, dtype=np.complex64)
            jones = archive.jones
            pixel_size_um = float(archive.pixel_size_um)
            del archive
            pixel_sizes.append(pixel_size_um)
            registration_features.append(
                make_registration_feature(
                    jones,
                    rough_center_px=(float(row["initial_cx"]), float(row["initial_cy"])),
                    crop_size_px=int(round(float(size_um) / pixel_size_um)),
                )
            )
            del jones
            centers.append(
                SimpleNamespace(
                    cx=float(row["cx"]),
                    cy=float(row["cy"]),
                    initial_cx=float(row["initial_cx"]),
                    initial_cy=float(row["initial_cy"]),
                    equivalent_radius_um=float(row["equivalent_radius_um"]),
                )
            )
        print(f"[{wavelength_nm} nm] reusing {len(centers)} saved centers", flush=True)
    for index, source in enumerate(() if reuse_centers else paths, start=1):
        print(f"[{wavelength_nm} nm] locating {index}/{len(paths)}: {source.name}", flush=True)
        archive = load_jones_archive(source, dtype=np.complex64)
        jones = archive.jones
        pixel_size_um = float(archive.pixel_size_um)
        del archive
        pixel_sizes.append(pixel_size_um)
        _evidence, threshold, boxes = detect_toron_bounding_boxes(
            jones,
            pixel_size_um=pixel_size_um,
            max_boxes=1,
            include_partial=False,
        )
        del _evidence
        if len(boxes) != 1:
            raise RuntimeError(
                f"{source}: expected one fully contained central toron, found {len(boxes)}"
            )
        analysis = analyze_centers(
            jones,
            pixel_size_um=pixel_size_um,
            boxes=boxes,
            progress=False,
        )[0]
        center = analysis.center
        center_analyses.append(analysis)
        registration_features.append(
            make_registration_feature(
                jones,
                rough_center_px=(center.initial_cx, center.initial_cy),
                crop_size_px=int(round(float(size_um) / pixel_size_um)),
            )
        )
        del jones
        centers.append(center)
        rows.append(
            {
                "source": _relative(source),
                **asdict(center),
                "detection_threshold": threshold,
            }
        )
        print(
            f"  center=({center.cx:.4f}, {center.cy:.4f}) px, "
            f"R_eq={center.equivalent_radius_um:.4f} um, "
            f"R_95={center.outer_radius_um:.4f} um",
            flush=True,
        )

    pixel_sizes_array = np.asarray(pixel_sizes, dtype=np.float64)
    if not np.allclose(pixel_sizes_array, pixel_sizes_array[0], rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{wavelength_nm} nm files have inconsistent pixel sizes")
    radii = np.asarray([item.equivalent_radius_um for item in centers], dtype=np.float64)
    radius_mean = float(np.mean(radii))
    radius_std = float(np.std(radii, ddof=1)) if radii.size > 1 else 0.0
    radius_cv = radius_std / radius_mean
    radius_max_relative_deviation = float(np.max(np.abs(radii - radius_mean)) / radius_mean)
    if radius_cv > float(max_radius_cv):
        warnings.warn(
            f"{wavelength_nm} nm equivalent-radius CV {radius_cv:.3%} exceeds "
            f"the requested {max_radius_cv:.3%}",
            RuntimeWarning,
            stacklevel=2,
        )

    labels = tuple(path.stem for path in paths)
    harmonic_centers = np.asarray([(item.cx, item.cy) for item in centers], dtype=np.float64)
    registration = register_jones_features(
        tuple(registration_features),
        anchor_centers_px=harmonic_centers,
        upsample_factor=20,
    )
    del registration_features
    for row, registered, correction in zip(
        rows, registration.centers_px, registration.corrections_px, strict=True
    ):
        row["registration_cx"] = float(registered[0])
        row["registration_cy"] = float(registered[1])
        row["registration_correction_x_px"] = float(correction[0])
        row["registration_correction_y_px"] = float(correction[1])
        row["averaging_center_method"] = "joint-complex-Jones-image-registration"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if center_analyses:
        analyses_tuple = tuple(center_analyses)
        write_center_quality_archive(
            center_quality_path, analyses_tuple, labels=labels, force=force
        )
        render_center_quality_pdf(
            center_quality_figure, analyses_tuple, labels=labels, force=force
        )
    write_registration_quality_archive(
        registration_path,
        registration,
        labels=labels,
        anchor_centers_px=harmonic_centers,
        force=force,
    )
    render_registration_quality_pdf(
        registration_figure,
        registration,
        labels=labels,
        anchor_centers_px=harmonic_centers,
        force=force,
    )
    samples = tuple(
        JonesSample(source=source, center=(float(center[0]), float(center[1])))
        for source, center in zip(paths, registration.centers_px, strict=True)
    )
    output_pixel_size_um = float(pixel_sizes_array[0]) / float(superresolution_factor)
    print(
        f"[{wavelength_nm} nm] averaging {len(samples)} fields at "
        f"{output_pixel_size_um:.7f} um/px",
        flush=True,
    )
    average = average_jones_samples(
        samples,
        size_um=size_um,
        output_pixel_size_um=output_pixel_size_um,
        resampling="drizzle",
        progress=True,
    )
    combined_center = measure_combined_center(
        average.jones,
        pixel_size_um=average.pixel_size_um,
        source_pixel_size_um=float(pixel_sizes_array[0]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jones_average_npz(
        output_path,
        average,
        source_path_base=PROJECT_ROOT,
        force=force,
    )
    _write_csv(centers_path, rows)

    report = {
        "schema_version": "shifted-toron-jones-average-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "wavelength_nm": wavelength_nm,
        "sources": [_relative(path) for path in paths],
        "centers_csv": _relative(centers_path),
        "center_quality_archive": _relative(center_quality_path),
        "center_quality_figure": _relative(center_quality_figure),
        "registration_archive": _relative(registration_path),
        "registration_figure": _relative(registration_figure),
        "average_archive": _relative(output_path),
        "center_method": "minimum normalized corotating m=+/-1 Jones power",
        "averaging_registration_method": (
            "joint pairwise complex-Jones image registration with a robust "
            "overdetermined shift graph"
        ),
        "averaging_registration_gauge": (
            "mean registered-center correction relative to the harmonic centers is zero"
        ),
        "harmonic_centers_px": harmonic_centers.tolist(),
        "averaging_centers_px": registration.centers_px.tolist(),
        "registration_corrections_px": registration.corrections_px.tolist(),
        "registration_pairwise_residual_median_px": float(
            np.nanmedian(registration.pairwise_residual_px)
        ),
        "registration_pairwise_residual_max_px": float(
            np.nanmax(registration.pairwise_residual_px)
        ),
        "registration_triangle_closure_median_px": float(
            np.median(registration.triangle_closure_px)
        ),
        "registration_triangle_closure_95_px": float(
            np.quantile(registration.triangle_closure_px, 0.95)
        ),
        "registration_triangle_closure_max_px": float(
            np.max(registration.triangle_closure_px)
        ),
        "registration_peak_ratio_min": float(
            np.nanmin(
                registration.pairwise_peak_ratio[
                    ~np.eye(len(paths), dtype=bool)
                ]
            )
        ),
        "radius_definition": "sqrt(A/pi) for the filled connected projective-Jones evidence footprint",
        "outer_radius_definition": "95th percentile of footprint-pixel distance from the fitted center",
        "equivalent_radius_um": radii.tolist(),
        "equivalent_radius_mean_um": radius_mean,
        "equivalent_radius_std_um": radius_std,
        "equivalent_radius_cv": radius_cv,
        "equivalent_radius_max_relative_deviation": radius_max_relative_deviation,
        "radius_consistency_limit_cv": float(max_radius_cv),
        "radius_consistency_passed": bool(radius_cv <= float(max_radius_cv)),
        "source_pixel_size_um": float(pixel_sizes_array[0]),
        "output_pixel_size_um": average.pixel_size_um,
        "superresolution_factor": float(superresolution_factor),
        "crop_size_um": average.size_um,
        "output_shape": list(average.jones.shape[:2]),
        "resampling": average.resampling,
        "global_phase_correction": "none",
        "background_phase_rad": average.background_phase_rad.tolist(),
        "background_phase_coherence": average.background_phase_coherence.tolist(),
        "valid_fraction": average.valid_fraction.tolist(),
        "weight_coverage_fraction": float(np.mean(average.weight_sum > 0.0)),
        "weight_min_positive": float(np.min(average.weight_sum[average.weight_sum > 0.0])),
        "weight_max": float(np.max(average.weight_sum)),
        "combined_center_diagnostic": combined_center,
    }
    _write_json(report_path, report)
    print(
        f"[{wavelength_nm} nm] R_eq={radius_mean:.4f}+/-{radius_std:.4f} um "
        f"(CV={radius_cv:.3%}); background phase range "
        f"[{np.min(average.background_phase_rad):+.4f}, "
        f"{np.max(average.background_phase_rad):+.4f}] rad",
        flush=True,
    )
    print(
        f"[{wavelength_nm} nm] combined-center residual "
        f"{combined_center['offset_norm_output_px']:.3f} output px "
        f"({combined_center['offset_norm_source_px']:.3f} source px, "
        f"{combined_center['offset_norm_um']:.4f} um); not applied",
        flush=True,
    )
    return report


def diagnose_existing_average(
    wavelength_nm: int,
    *,
    output_dir: Path,
    analysis_dir: Path,
) -> dict[str, object]:
    output_path = output_dir / f"{wavelength_nm}nm_average.npz"
    report_path = analysis_dir / f"{wavelength_nm}nm_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing existing report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    archive = load_jones_archive(output_path, dtype=np.complex64)
    jones = archive.jones
    pixel_size_um = float(archive.pixel_size_um)
    del archive
    gc.collect()
    source_pixel_size_um = float(report["source_pixel_size_um"])
    diagnostic = measure_combined_center(
        jones,
        pixel_size_um=pixel_size_um,
        source_pixel_size_um=source_pixel_size_um,
    )
    del jones
    gc.collect()
    report["combined_center_diagnostic"] = diagnostic
    _write_json(report_path, report)
    print(
        f"[{wavelength_nm} nm] combined-center residual "
        f"{diagnostic['offset_norm_output_px']:.3f} output px "
        f"({diagnostic['offset_norm_source_px']:.3f} source px, "
        f"{diagnostic['offset_norm_um']:.4f} um); not applied",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locate, subpixel-center, size-check, and drizzle-average the shifted "
            "single-toron Jones series without global-phase alignment."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--wavelength", action="append", type=int, default=[])
    parser.add_argument("--size-um", type=float, default=96.0)
    parser.add_argument("--superresolution-factor", type=float, default=2.0)
    parser.add_argument("--max-radius-cv", type=float, default=0.03)
    parser.add_argument(
        "--reuse-centers",
        action="store_true",
        help="Reuse existing center/radius CSV files and rerun only averaging checks.",
    )
    parser.add_argument(
        "--diagnose-existing-average-centers",
        action="store_true",
        help=(
            "Measure and report centers of existing averages without rebuilding or "
            "shifting the averaged Jones matrices."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-wavelength generated outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argument_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argument_list)
    source_root = args.source.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    analysis_dir = args.analysis.expanduser().resolve()
    wavelengths = tuple(args.wavelength) if args.wavelength else (450, 520, 660)
    reports = []
    if args.diagnose_existing_average_centers:
        for wavelength_nm in wavelengths:
            reports.append(
                diagnose_existing_average(
                    wavelength_nm,
                    output_dir=output_dir,
                    analysis_dir=analysis_dir,
                )
            )
    else:
        if not args.force:
            conflicts = {
                wavelength_nm: existing
                for wavelength_nm in wavelengths
                if (
                    existing := tuple(
                        path
                        for path in _protected_wavelength_outputs(
                            wavelength_nm,
                            output_dir=output_dir,
                            analysis_dir=analysis_dir,
                            reuse_centers=args.reuse_centers,
                        )
                        if path.exists()
                    )
                )
            }
            if conflicts:
                print(
                    "Existing generated outputs would be overwritten; "
                    "no files were changed:",
                    file=sys.stderr,
                )
                for wavelength_nm, paths in conflicts.items():
                    print(f"  {wavelength_nm} nm:", file=sys.stderr)
                    for path in paths:
                        print(f"    {_relative(path)}", file=sys.stderr)
                force_command = shlex.join(
                    (
                        "uv",
                        "run",
                        "python",
                        _relative(Path(__file__)),
                        *argument_list,
                        "--force",
                    )
                )
                print(
                    "\nRerun with --force to replace these outputs:",
                    file=sys.stderr,
                )
                print(f"  {force_command}", file=sys.stderr)
                return 2
        for wavelength_nm in wavelengths:
            reports.append(
                process_wavelength(
                    wavelength_nm,
                    _indexed_npz(source_root / str(wavelength_nm)),
                    output_dir=output_dir,
                    analysis_dir=analysis_dir,
                    size_um=args.size_um,
                    superresolution_factor=args.superresolution_factor,
                    max_radius_cv=args.max_radius_cv,
                    reuse_centers=args.reuse_centers,
                    force=args.force,
                )
            )
    report_paths = sorted(
        analysis_dir.glob("*nm_report.json"),
        key=lambda path: int(path.name.split("nm_", maxsplit=1)[0]),
    )
    summary_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    summary_wavelengths = [int(report["wavelength_nm"]) for report in summary_reports]
    summary = {
        "schema_version": "shifted-toron-jones-series-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": _relative(source_root),
        "output_directory": _relative(output_dir),
        "analysis_directory": _relative(analysis_dir),
        "wavelengths_nm": summary_wavelengths,
        "reports": summary_reports,
    }
    summary_path = analysis_dir / "report.json"
    _write_json(summary_path, summary)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
