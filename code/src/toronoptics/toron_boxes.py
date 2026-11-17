from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from toronoptics.cli import refuse_existing_outputs
from toronoptics.experimental_jones import load_jones_archive


@dataclass(frozen=True)
class ApproximateBox:
    box_id: str
    x0: int
    x1: int
    y0: int
    y1: int
    initial_cx: float
    initial_cy: float
    score: float = float("nan")
    area_px: int = 0
    fully_contained: bool = True


@dataclass(frozen=True)
class ToronRadiusMeasurement:
    equivalent_radius_px: float
    equivalent_radius_um: float
    outer_radius_px: float
    outer_radius_um: float
    area_px: int
    threshold: float


def projective_background_evidence(
    jones: np.ndarray,
    *,
    border_fraction: float = 0.08,
    smooth_sigma_px: float = 2.0,
) -> np.ndarray:
    """Measure polarization-changing departure from the Jones background.

    At every pixel the best complex scalar multiple of the robust border Jones
    matrix is removed.  The result is insensitive to a scalar illumination or
    reference-wave factor but responds to the polarization structure of a
    toron.
    """

    matrix = _as_jones(jones)
    if not 0.0 < float(border_fraction) < 0.5:
        raise ValueError("border_fraction must lie between 0 and 0.5")
    width = max(4, int(round(float(border_fraction) * min(matrix.shape[:2]))))
    border = np.zeros(matrix.shape[:2], dtype=bool)
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True
    samples = matrix[border]
    trace = 0.5 * (samples[:, 0, 0] + samples[:, 1, 1])
    finite = np.isfinite(samples).all(axis=(1, 2)) & np.isfinite(trace)
    if np.count_nonzero(finite) < 16:
        raise ValueError("Too few finite border pixels to estimate the Jones background")
    samples = samples[finite]
    trace = trace[finite]
    amplitude_cutoff = float(np.percentile(np.abs(trace), 20.0))
    usable = np.abs(trace) >= max(amplitude_cutoff, 1.0e-12)
    if np.count_nonzero(usable) < 16:
        raise ValueError("The scalar border response is too small to estimate the Jones background")
    normalized = samples[usable] / trace[usable, None, None]
    background = _complex_component_median(normalized)
    background_norm = float(np.sum(np.abs(background) ** 2))
    if not np.isfinite(background_norm) or background_norm <= 1.0e-20:
        raise ValueError("Estimated Jones background has zero norm")

    scalar = np.einsum("ij,xyij->xy", np.conj(background), matrix, optimize=True) / background_norm
    residual = matrix - scalar[..., None, None] * background
    numerator = np.sum(np.abs(residual) ** 2, axis=(-2, -1))
    denominator = np.sum(np.abs(matrix) ** 2, axis=(-2, -1))
    evidence = np.sqrt(numerator / np.maximum(denominator, 1.0e-20))
    evidence[~np.isfinite(evidence)] = 0.0
    if smooth_sigma_px > 0.0:
        evidence = ndi.gaussian_filter(evidence, sigma=float(smooth_sigma_px))
    return np.ascontiguousarray(evidence, dtype=np.float32)


def detect_toron_bounding_boxes(
    jones: np.ndarray,
    *,
    pixel_size_um: float,
    mad_threshold: float = 6.0,
    min_area_um2: float = 50.0,
    padding_um: float = 5.0,
    morphology_um: float = 0.75,
    border_fraction: float = 0.08,
    smooth_sigma_px: float = 2.0,
    include_partial: bool = False,
    max_boxes: int | None = None,
) -> tuple[np.ndarray, float, tuple[ApproximateBox, ...]]:
    matrix = _as_jones(jones)
    _positive("pixel_size_um", pixel_size_um)
    _positive("mad_threshold", mad_threshold)
    _nonnegative("min_area_um2", min_area_um2)
    _nonnegative("padding_um", padding_um)
    _nonnegative("morphology_um", morphology_um)
    evidence = projective_background_evidence(
        matrix,
        border_fraction=border_fraction,
        smooth_sigma_px=smooth_sigma_px,
    )
    threshold = robust_evidence_threshold(evidence, mad_threshold=mad_threshold)
    mask = evidence >= threshold
    iterations = max(1, int(round(float(morphology_um) / float(pixel_size_um))))
    structure = ndi.generate_binary_structure(2, 1)
    mask = ndi.binary_closing(mask, structure=structure, iterations=iterations)
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask, structure=structure)
    min_area_px = max(1, int(round(float(min_area_um2) / float(pixel_size_um) ** 2)))
    padding_px = int(np.ceil(float(padding_um) / float(pixel_size_um)))
    records: list[ApproximateBox] = []
    nx, ny = matrix.shape[:2]
    for label in range(1, int(count) + 1):
        component = labels == label
        area = int(np.count_nonzero(component))
        if area < min_area_px:
            continue
        coordinates = np.argwhere(component)
        raw_x0, raw_y0 = coordinates.min(axis=0)
        raw_x1, raw_y1 = coordinates.max(axis=0) + 1
        unclipped = (
            int(raw_x0) - padding_px,
            int(raw_x1) + padding_px,
            int(raw_y0) - padding_px,
            int(raw_y1) + padding_px,
        )
        fully_contained = (
            unclipped[0] >= 0
            and unclipped[1] <= nx
            and unclipped[2] >= 0
            and unclipped[3] <= ny
        )
        if not include_partial and not fully_contained:
            continue
        x0 = max(0, unclipped[0])
        x1 = min(nx, unclipped[1])
        y0 = max(0, unclipped[2])
        y1 = min(ny, unclipped[3])
        weights = np.maximum(evidence[component].astype(np.float64) - threshold, 0.0)
        if not np.any(weights > 0.0):
            weights = np.ones(coordinates.shape[0], dtype=np.float64)
        center = np.average(coordinates.astype(np.float64), axis=0, weights=weights)
        records.append(
            ApproximateBox(
                box_id="",
                x0=int(x0),
                x1=int(x1),
                y0=int(y0),
                y1=int(y1),
                initial_cx=float(center[0]),
                initial_cy=float(center[1]),
                score=float(np.sum(evidence[component] - threshold)),
                area_px=area,
                fully_contained=bool(fully_contained),
            )
        )
    records.sort(key=lambda item: (item.initial_cy, item.initial_cx))
    if max_boxes is not None:
        if int(max_boxes) < 1:
            raise ValueError("max_boxes must be positive")
        records = sorted(records, key=lambda item: item.score, reverse=True)[: int(max_boxes)]
        records.sort(key=lambda item: (item.initial_cy, item.initial_cx))
    records = [replace(item, box_id=f"toron_{index}") for index, item in enumerate(records, start=1)]
    return evidence, float(threshold), tuple(records)


def robust_evidence_threshold(
    evidence: np.ndarray,
    *,
    mad_threshold: float = 6.0,
) -> float:
    _positive("mad_threshold", mad_threshold)
    finite = np.asarray(evidence)[np.isfinite(evidence)]
    if finite.size < 16:
        raise ValueError("Too few finite evidence pixels to estimate a threshold")
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    threshold = median + float(mad_threshold) * mad
    if not np.isfinite(threshold) or mad <= 1.0e-12:
        threshold = float(np.percentile(finite, 99.0))
    return float(threshold)


def measure_toron_radius(
    evidence: np.ndarray,
    *,
    center: tuple[float, float],
    pixel_size_um: float,
    box: ApproximateBox | None = None,
    threshold: float | None = None,
    mad_threshold: float = 6.0,
    morphology_um: float = 0.75,
    outer_percentile: float = 95.0,
) -> ToronRadiusMeasurement:
    """Measure a connected toron footprint around an independently fitted center.

    The equivalent radius is ``sqrt(area/pi)`` for the filled connected
    polarization-changing region.  The outer radius is a percentile of the
    component-pixel distances from the fitted center and is retained as a
    shape-sensitive control rather than as the primary toron radius.
    """

    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected two-dimensional evidence, got {values.shape!r}")
    _positive("pixel_size_um", pixel_size_um)
    _nonnegative("morphology_um", morphology_um)
    if not 50.0 <= float(outer_percentile) < 100.0:
        raise ValueError("outer_percentile must lie between 50 and 100")
    resolved_threshold = (
        robust_evidence_threshold(values, mad_threshold=mad_threshold)
        if threshold is None
        else float(threshold)
    )
    x0, x1, y0, y1 = (0, values.shape[0], 0, values.shape[1])
    if box is not None:
        x0, x1, y0, y1 = box.x0, box.x1, box.y0, box.y1
    local = values[x0:x1, y0:y1] >= resolved_threshold
    iterations = max(1, int(round(float(morphology_um) / float(pixel_size_um))))
    structure = ndi.generate_binary_structure(2, 1)
    local = ndi.binary_closing(local, structure=structure, iterations=iterations)
    local = ndi.binary_fill_holes(local)
    labels, count = ndi.label(local, structure=structure)
    if count == 0:
        raise ValueError("No connected toron footprint was found at the fitted center")

    local_center = (float(center[0]) - x0, float(center[1]) - y0)
    center_index = (
        int(np.clip(round(local_center[0]), 0, local.shape[0] - 1)),
        int(np.clip(round(local_center[1]), 0, local.shape[1] - 1)),
    )
    center_label = int(labels[center_index])
    if center_label == 0:
        label_ids, counts = np.unique(labels[labels > 0], return_counts=True)
        center_label = int(label_ids[int(np.argmax(counts))])
    component = labels == center_label
    coordinates = np.argwhere(component).astype(np.float64)
    area_px = int(coordinates.shape[0])
    if area_px == 0:
        raise ValueError("The selected toron footprint is empty")
    radius_px = float(np.sqrt(area_px / np.pi))
    distances = np.hypot(
        coordinates[:, 0] - local_center[0],
        coordinates[:, 1] - local_center[1],
    )
    outer_radius_px = float(np.percentile(distances, float(outer_percentile)))
    return ToronRadiusMeasurement(
        equivalent_radius_px=radius_px,
        equivalent_radius_um=radius_px * float(pixel_size_um),
        outer_radius_px=outer_radius_px,
        outer_radius_um=outer_radius_px * float(pixel_size_um),
        area_px=area_px,
        threshold=resolved_threshold,
    )


def read_boxes_csv(path: Path) -> tuple[ApproximateBox, ...]:
    path = path.expanduser().resolve()
    result: list[ApproximateBox] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"box_id", "x0", "x1", "y0", "y1"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row_index, row in enumerate(reader, start=2):
            box_id = str(row["box_id"]).strip()
            if not box_id:
                raise ValueError(f"{path}:{row_index}: empty box_id")
            x0, x1 = int(row["x0"]), int(row["x1"])
            y0, y1 = int(row["y0"]), int(row["y1"])
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"{path}:{row_index}: invalid bounding box")
            initial_cx = _optional_float(row.get("initial_cx"), 0.5 * (x0 + x1 - 1))
            initial_cy = _optional_float(row.get("initial_cy"), 0.5 * (y0 + y1 - 1))
            result.append(
                ApproximateBox(
                    box_id=box_id,
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    initial_cx=initial_cx,
                    initial_cy=initial_cy,
                    score=_optional_float(row.get("score")),
                    area_px=int(float(row.get("area_px") or 0)),
                    fully_contained=str(row.get("fully_contained", "true")).lower()
                    not in {"0", "false", "no"},
                )
            )
    if not result:
        raise ValueError(f"{path} contains no bounding boxes")
    return tuple(result)


def write_boxes_csv(path: Path, boxes: tuple[ApproximateBox, ...], *, force: bool = False) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields = tuple(ApproximateBox.__dataclass_fields__)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(item.__dict__ for item in boxes)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find approximate toron bounding boxes from departure of a Jones field from its background."
    )
    parser.add_argument("source", type=Path, help="Jones NPZ archive")
    parser.add_argument("output", type=Path, help="Output bounding-box CSV")
    parser.add_argument("--mad-threshold", type=float, default=6.0)
    parser.add_argument("--min-area-um2", type=float, default=50.0)
    parser.add_argument("--padding-um", type=float, default=5.0)
    parser.add_argument("--morphology-um", type=float, default=0.75)
    parser.add_argument("--border-fraction", type=float, default=0.08)
    parser.add_argument("--smooth-sigma-px", type=float, default=2.0)
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    refuse_existing_outputs(
        (args.output,),
        force=args.force,
        project_root=Path(__file__).resolve().parents[3],
    )
    archive = load_jones_archive(args.source)
    _evidence, threshold, boxes = detect_toron_bounding_boxes(
        archive.jones,
        pixel_size_um=archive.pixel_size_um,
        mad_threshold=args.mad_threshold,
        min_area_um2=args.min_area_um2,
        padding_um=args.padding_um,
        morphology_um=args.morphology_um,
        border_fraction=args.border_fraction,
        smooth_sigma_px=args.smooth_sigma_px,
        include_partial=args.include_partial,
        max_boxes=args.max_boxes,
    )
    write_boxes_csv(args.output, boxes, force=args.force)
    print(f"Found {len(boxes)} toron boxes at evidence threshold {threshold:.6g}")
    for item in boxes:
        print(
            f"  {item.box_id}: x=[{item.x0},{item.x1}), y=[{item.y0},{item.y1}), "
            f"initial center=({item.initial_cx:.2f},{item.initial_cy:.2f})"
        )
    print(f"Bounding boxes: {args.output.expanduser().resolve()}")
    return 0


def _complex_component_median(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return np.median(array.real, axis=0) + 1j * np.median(array.imag, axis=0)


def _as_jones(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 4 or array.shape[-2:] != (2, 2):
        raise ValueError(f"Expected Jones field with shape (x,y,2,2), got {array.shape!r}")
    return np.asarray(array, dtype=np.complex128)


def _optional_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _positive(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _nonnegative(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")


if __name__ == "__main__":
    raise SystemExit(main())
