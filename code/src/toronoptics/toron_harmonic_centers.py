from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from toronoptics.axisymmetric_jones import (
    CenterHarmonicDiagnostics,
    SubpixelCenterResult,
    center_harmonic_diagnostics,
    refine_axisymmetric_center,
)
from toronoptics.cli import refuse_existing_outputs
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.toron_boxes import (
    ApproximateBox,
    measure_toron_radius,
    projective_background_evidence,
    read_boxes_csv,
    robust_evidence_threshold,
)
from toronoptics.toron_center_quality import (
    render_center_quality_pdf,
    write_center_quality_archive,
)


@dataclass(frozen=True)
class HarmonicCenter:
    toron_id: int
    box_id: str
    cx: float
    cy: float
    initial_cx: float
    initial_cy: float
    x0: int
    x1: int
    y0: int
    y1: int
    objective: float
    all_forbidden_objective: float
    all_forbidden_local_cx: float
    all_forbidden_local_cy: float
    all_forbidden_center_disagreement_px: float
    resolution_px: float
    broad_converged: bool
    coarse_converged: bool
    pixel_size_um: float
    equivalent_radius_px: float
    equivalent_radius_um: float
    outer_radius_px: float
    outer_radius_um: float
    footprint_area_px: int
    evidence_threshold: float
    method: str = "minimum-corotating-m-plus-minus-1-power"


@dataclass(frozen=True)
class HarmonicCenterAnalysis:
    center: HarmonicCenter
    search: SubpixelCenterResult
    harmonics: CenterHarmonicDiagnostics


def refine_centers_from_archive(
    source: Path,
    boxes: tuple[ApproximateBox, ...],
    *,
    min_radius_um: float = 4.0,
    max_radius_um: float = 30.0,
    radial_step_um: float = 0.75,
    angular_samples: int = 360,
    broad_radius_px: float = 16.0,
    broad_step_px: float = 2.0,
    coarse_radius_px: float = 2.0,
    coarse_step_px: float = 0.25,
    fine_radius_px: float = 0.5,
    fine_step_px: float = 0.05,
    progress: bool = False,
) -> tuple[HarmonicCenter, ...]:
    return tuple(
        analysis.center
        for analysis in analyze_centers_from_archive(
            source,
            boxes,
            min_radius_um=min_radius_um,
            max_radius_um=max_radius_um,
            radial_step_um=radial_step_um,
            angular_samples=angular_samples,
            broad_radius_px=broad_radius_px,
            broad_step_px=broad_step_px,
            coarse_radius_px=coarse_radius_px,
            coarse_step_px=coarse_step_px,
            fine_radius_px=fine_radius_px,
            fine_step_px=fine_step_px,
            progress=progress,
        )
    )


def analyze_centers_from_archive(
    source: Path,
    boxes: tuple[ApproximateBox, ...],
    *,
    min_radius_um: float = 4.0,
    max_radius_um: float = 30.0,
    radial_step_um: float = 0.75,
    angular_samples: int = 360,
    broad_radius_px: float = 16.0,
    broad_step_px: float = 2.0,
    coarse_radius_px: float = 2.0,
    coarse_step_px: float = 0.25,
    fine_radius_px: float = 0.5,
    fine_step_px: float = 0.05,
    progress: bool = False,
) -> tuple[HarmonicCenterAnalysis, ...]:
    source = source.expanduser().resolve()
    archive = load_jones_archive(source, dtype=np.complex64)
    return analyze_centers(
        archive.jones,
        pixel_size_um=archive.pixel_size_um,
        boxes=boxes,
        min_radius_um=min_radius_um,
        max_radius_um=max_radius_um,
        radial_step_um=radial_step_um,
        angular_samples=angular_samples,
        broad_radius_px=broad_radius_px,
        broad_step_px=broad_step_px,
        coarse_radius_px=coarse_radius_px,
        coarse_step_px=coarse_step_px,
        fine_radius_px=fine_radius_px,
        fine_step_px=fine_step_px,
        progress=progress,
    )


def analyze_centers(
    jones: np.ndarray,
    *,
    pixel_size_um: float,
    boxes: tuple[ApproximateBox, ...],
    min_radius_um: float = 4.0,
    max_radius_um: float = 30.0,
    radial_step_um: float = 0.75,
    angular_samples: int = 360,
    broad_radius_px: float = 16.0,
    broad_step_px: float = 2.0,
    coarse_radius_px: float = 2.0,
    coarse_step_px: float = 0.25,
    fine_radius_px: float = 0.5,
    fine_step_px: float = 0.05,
    progress: bool = False,
) -> tuple[HarmonicCenterAnalysis, ...]:
    matrix = np.asarray(jones)
    evidence = projective_background_evidence(matrix)
    evidence_threshold = robust_evidence_threshold(evidence)
    analyses: list[HarmonicCenterAnalysis] = []
    for index, box in enumerate(boxes, start=1):
        if progress:
            print(f"[centers] {index}/{len(boxes)} {box.box_id}", flush=True)
        _validate_box_for_shape(box, matrix.shape[:2])
        _validate_sampling_domain(
            box,
            shape=matrix.shape[:2],
            max_radius_px=float(max_radius_um) / float(pixel_size_um),
            search_radius_px=float(broad_radius_px),
        )
        result = refine_axisymmetric_center(
            matrix,
            initial_center_px=(box.initial_cx, box.initial_cy),
            pixel_size_um=pixel_size_um,
            min_radius_um=min_radius_um,
            max_radius_um=max_radius_um,
            radial_step_um=radial_step_um,
            angular_samples=angular_samples,
            broad_radius_px=broad_radius_px,
            broad_step_px=broad_step_px,
            coarse_radius_px=coarse_radius_px,
            coarse_step_px=coarse_step_px,
            fine_radius_px=fine_radius_px,
            fine_step_px=fine_step_px,
        )
        radius = measure_toron_radius(
            evidence,
            center=result.center_px,
            pixel_size_um=pixel_size_um,
            box=box,
            threshold=evidence_threshold,
        )
        fine_index = np.unravel_index(
            int(np.nanargmin(result.fine_objective)), result.fine_objective.shape
        )
        all_index = np.unravel_index(
            int(np.nanargmin(result.coarse_all_objective)),
            result.coarse_all_objective.shape,
        )
        all_center = np.asarray(
            (result.coarse_x_px[all_index[0]], result.coarse_y_px[all_index[1]]),
            dtype=np.float64,
        )
        fitted_center = np.asarray(result.center_px, dtype=np.float64)
        center = HarmonicCenter(
            toron_id=index,
            box_id=box.box_id,
            cx=float(result.center_px[0]),
            cy=float(result.center_px[1]),
            initial_cx=float(box.initial_cx),
            initial_cy=float(box.initial_cy),
            x0=box.x0,
            x1=box.x1,
            y0=box.y0,
            y1=box.y1,
            objective=float(result.objective),
            all_forbidden_objective=float(result.fine_all_objective[fine_index]),
            all_forbidden_local_cx=float(all_center[0]),
            all_forbidden_local_cy=float(all_center[1]),
            all_forbidden_center_disagreement_px=float(
                np.linalg.norm(all_center - fitted_center)
            ),
            resolution_px=float(fine_step_px),
            broad_converged=bool(result.broad_converged),
            coarse_converged=bool(result.coarse_converged),
            pixel_size_um=float(pixel_size_um),
            equivalent_radius_px=radius.equivalent_radius_px,
            equivalent_radius_um=radius.equivalent_radius_um,
            outer_radius_px=radius.outer_radius_px,
            outer_radius_um=radius.outer_radius_um,
            footprint_area_px=radius.area_px,
            evidence_threshold=radius.threshold,
        )
        harmonics = center_harmonic_diagnostics(
            matrix,
            center_px=result.center_px,
            pixel_size_um=pixel_size_um,
            radii_um=np.arange(
                float(min_radius_um),
                float(max_radius_um) + 0.5 * float(radial_step_um),
                float(radial_step_um),
            ),
            angular_samples=angular_samples,
        )
        analyses.append(
            HarmonicCenterAnalysis(center=center, search=result, harmonics=harmonics)
        )
    return tuple(analyses)


def write_centers_csv(path: Path, centers: tuple[HarmonicCenter, ...], *, force: bool = False) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(HarmonicCenter.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(item) for item in centers)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_bbox(value: str, *, index: int) -> ApproximateBox:
    parts = [part.strip() for part in value.replace(":", ",").split(",")]
    if len(parts) == 4:
        box_id = f"toron_{index}"
        bounds = parts
    elif len(parts) == 5:
        box_id = parts[0]
        bounds = parts[1:]
    else:
        raise argparse.ArgumentTypeError(
            "bbox must be x0,x1,y0,y1 or box_id,x0,x1,y0,y1"
        )
    try:
        x0, x1, y0, y1 = (int(item) for item in bounds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid bbox {value!r}") from exc
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError(f"invalid bbox bounds {value!r}")
    return ApproximateBox(
        box_id=box_id,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        initial_cx=0.5 * (x0 + x1 - 1),
        initial_cy=0.5 * (y0 + y1 - 1),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine toron centers to subpixel precision by minimizing the forbidden "
            "m=+/-1 harmonics of the corotating Jones field."
        )
    )
    parser.add_argument("source", type=Path, help="Jones NPZ archive")
    parser.add_argument("output", type=Path, help="Output centers CSV")
    parser.add_argument("--boxes-file", type=Path, default=None)
    parser.add_argument(
        "--bbox",
        action="append",
        default=[],
        help="x0,x1,y0,y1 or box_id,x0,x1,y0,y1; repeatable",
    )
    parser.add_argument("--min-radius-um", type=float, default=4.0)
    parser.add_argument("--max-radius-um", type=float, default=30.0)
    parser.add_argument("--radial-step-um", type=float, default=0.75)
    parser.add_argument("--angular-samples", type=int, default=360)
    parser.add_argument("--broad-radius-px", type=float, default=16.0)
    parser.add_argument("--broad-step-px", type=float, default=2.0)
    parser.add_argument("--coarse-radius-px", type=float, default=2.0)
    parser.add_argument("--coarse-step-px", type=float, default=0.25)
    parser.add_argument("--fine-radius-px", type=float, default=0.5)
    parser.add_argument("--fine-step-px", type=float, default=0.05)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--quality-archive", type=Path, default=None)
    parser.add_argument("--quality-figure", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_path = args.output.expanduser().resolve()
    manifest_path = (
        output_path.with_suffix(".json") if args.manifest is None else args.manifest.expanduser().resolve()
    )
    quality_archive_path = (
        output_path.with_name(f"{output_path.stem}_quality.npz")
        if args.quality_archive is None
        else args.quality_archive.expanduser().resolve()
    )
    quality_figure_path = (
        output_path.with_name(f"{output_path.stem}_quality.pdf")
        if args.quality_figure is None
        else args.quality_figure.expanduser().resolve()
    )
    refuse_existing_outputs(
        (output_path, manifest_path, quality_archive_path, quality_figure_path),
        force=args.force,
        project_root=Path(__file__).resolve().parents[3],
    )
    boxes: list[ApproximateBox] = []
    if args.boxes_file is not None:
        boxes.extend(read_boxes_csv(args.boxes_file))
    boxes.extend(parse_bbox(value, index=len(boxes) + index) for index, value in enumerate(args.bbox, start=1))
    if not boxes:
        raise SystemExit("At least one --boxes-file or --bbox is required")
    analyses = analyze_centers_from_archive(
        args.source,
        tuple(boxes),
        min_radius_um=args.min_radius_um,
        max_radius_um=args.max_radius_um,
        radial_step_um=args.radial_step_um,
        angular_samples=args.angular_samples,
        broad_radius_px=args.broad_radius_px,
        broad_step_px=args.broad_step_px,
        coarse_radius_px=args.coarse_radius_px,
        coarse_step_px=args.coarse_step_px,
        fine_radius_px=args.fine_radius_px,
        fine_step_px=args.fine_step_px,
        progress=True,
    )
    centers = tuple(item.center for item in analyses)
    write_centers_csv(output_path, centers, force=args.force)
    labels = tuple(item.box_id for item in centers)
    write_center_quality_archive(
        quality_archive_path, analyses, labels=labels, force=args.force
    )
    render_center_quality_pdf(
        quality_figure_path, analyses, labels=labels, force=args.force
    )
    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "minimum normalized m=+/-1 power of the corotating complex Jones matrix",
        "source": str(args.source.expanduser().resolve()),
        "boxes_file": None if args.boxes_file is None else str(args.boxes_file.expanduser().resolve()),
        "parameters": {
            "min_radius_um": args.min_radius_um,
            "max_radius_um": args.max_radius_um,
            "radial_step_um": args.radial_step_um,
            "angular_samples": args.angular_samples,
            "broad_radius_px": args.broad_radius_px,
            "broad_step_px": args.broad_step_px,
            "coarse_radius_px": args.coarse_radius_px,
            "coarse_step_px": args.coarse_step_px,
            "fine_radius_px": args.fine_radius_px,
            "fine_step_px": args.fine_step_px,
        },
        "centers": [asdict(item) for item in centers],
        "quality_archive": str(quality_archive_path),
        "quality_figure": str(quality_figure_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Refined {len(centers)} centers")
    for item in centers:
        print(
            f"  {item.box_id}: ({item.cx:.4f}, {item.cy:.4f}) px, "
            f"R_eq={item.equivalent_radius_um:.4f} um, "
            f"R_95={item.outer_radius_um:.4f} um, "
            f"m=+/-1 objective={item.objective:.6g}"
        )
    print(f"Centers: {args.output.expanduser().resolve()}")
    print(f"Center quality: {quality_figure_path}")
    return 0


def _validate_box_for_shape(box: ApproximateBox, shape: tuple[int, int]) -> None:
    if box.x0 < 0 or box.y0 < 0 or box.x1 > shape[0] or box.y1 > shape[1]:
        raise ValueError(f"Box {box.box_id!r} lies outside Jones shape {shape!r}")


def _validate_sampling_domain(
    box: ApproximateBox,
    *,
    shape: tuple[int, int],
    max_radius_px: float,
    search_radius_px: float,
) -> None:
    margin = float(max_radius_px) + float(search_radius_px) + 1.0
    if (
        box.initial_cx - margin < 0.0
        or box.initial_cx + margin > shape[0] - 1
        or box.initial_cy - margin < 0.0
        or box.initial_cy + margin > shape[1] - 1
    ):
        raise ValueError(
            f"Polar center-search domain for {box.box_id!r} extends outside the Jones field; "
            "reduce --max-radius-um/--broad-radius-px or omit a partial toron"
        )


if __name__ == "__main__":
    raise SystemExit(main())
