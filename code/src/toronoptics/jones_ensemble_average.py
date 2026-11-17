from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage as ndi

from toronoptics.cli import refuse_existing_outputs
from toronoptics.experimental_jones import PIXEL_SIZE_KEY, load_jones_archive


Resampling = Literal["interpolate", "drizzle"]
RESAMPLING_METHODS: tuple[Resampling, ...] = ("interpolate", "drizzle")


@dataclass(frozen=True)
class JonesSample:
    source: Path
    center: tuple[float, float]


@dataclass(frozen=True)
class JonesAverageResult:
    jones: np.ndarray
    weight_sum: np.ndarray
    pixel_size_um: float
    size_um: float
    center_px: tuple[float, float]
    valid_fraction: np.ndarray
    background_phase_rad: np.ndarray
    background_phase_coherence: np.ndarray
    samples: tuple[JonesSample, ...]
    resampling: str


def average_jones_samples(
    samples: tuple[JonesSample, ...],
    *,
    size_um: float,
    output_pixel_size_um: float | None = None,
    resampling: Resampling = "drizzle",
    background_inner_fraction: float = 0.94,
    max_background_phase_rad: float = 0.10,
    min_background_phase_coherence: float = 0.90,
    min_valid_fraction: float = 0.90,
    progress: bool = False,
) -> JonesAverageResult:
    if not samples:
        raise ValueError("At least one Jones sample is required")
    _positive("size_um", size_um)
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(f"Unsupported resampling method {resampling!r}")
    if not 0.0 < float(background_inner_fraction) < 1.0:
        raise ValueError("background_inner_fraction must lie between 0 and 1")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must lie between 0 and 1")
    _nonnegative("max_background_phase_rad", max_background_phase_rad)
    if not 0.0 <= float(min_background_phase_coherence) <= 1.0:
        raise ValueError("min_background_phase_coherence must lie between 0 and 1")

    first = load_jones_archive(samples[0].source, dtype=np.complex64)
    target_pixel = (
        float(first.pixel_size_um)
        if output_pixel_size_um is None
        else float(output_pixel_size_um)
    )
    _positive("output_pixel_size_um", target_pixel)
    crop_size = max(2, int(round(float(size_um) / target_pixel)))
    output_center = (0.5 * (crop_size - 1), 0.5 * (crop_size - 1))
    total = np.zeros((crop_size, crop_size, 2, 2), dtype=np.complex128)
    weight_sum = np.zeros((crop_size, crop_size), dtype=np.float64)
    valid_fraction = np.empty(len(samples), dtype=np.float64)
    background_phase_rad = np.empty(len(samples), dtype=np.float64)
    background_phase_coherence = np.empty(len(samples), dtype=np.float64)
    for index, sample in enumerate(samples, start=1):
        if progress:
            print(f"[average] {index}/{len(samples)} {sample.source}", flush=True)
        archive = first if index == 1 else load_jones_archive(sample.source, dtype=np.complex64)
        coverage = physical_crop_coverage_fraction(
            archive.jones.shape[:2],
            center=sample.center,
            source_pixel_size_um=archive.pixel_size_um,
            output_pixel_size_um=target_pixel,
            crop_size_px=crop_size,
        )
        if coverage < float(min_valid_fraction):
            raise ValueError(
                f"Sample {sample.source} has only {coverage:.3f} valid coverage; "
                f"required {min_valid_fraction:.3f}"
            )
        diagnostic_size = max(2, int(round(float(size_um) / archive.pixel_size_um)))
        diagnostic, diagnostic_valid = extract_centered_jones_physical(
            archive.jones,
            center=sample.center,
            source_pixel_size_um=archive.pixel_size_um,
            output_pixel_size_um=archive.pixel_size_um,
            crop_size_px=diagnostic_size,
        )
        phase, coherence = background_phase_statistics(
            diagnostic,
            diagnostic_valid,
            inner_fraction=background_inner_fraction,
        )
        background_phase_rad[index - 1] = phase
        background_phase_coherence[index - 1] = coherence
        _warn_background_phase(
            sample.source,
            phase_rad=phase,
            coherence=coherence,
            max_phase_rad=max_background_phase_rad,
            min_coherence=min_background_phase_coherence,
        )

        if resampling == "interpolate":
            field, valid = extract_centered_jones_physical(
                archive.jones,
                center=sample.center,
                source_pixel_size_um=archive.pixel_size_um,
                output_pixel_size_um=target_pixel,
                crop_size_px=crop_size,
            )
            finite = valid & np.isfinite(field).all(axis=(-2, -1))
            total[finite] += field[finite]
            weight_sum[finite] += 1.0
            valid_fraction[index - 1] = float(np.mean(finite))
        else:
            sample_total, sample_weight = drizzle_centered_jones(
                archive.jones,
                center=sample.center,
                source_pixel_size_um=archive.pixel_size_um,
                output_pixel_size_um=target_pixel,
                crop_size_px=crop_size,
            )
            total += sample_total
            weight_sum += sample_weight
            valid_fraction[index - 1] = coverage
        del diagnostic, diagnostic_valid, archive
    del first
    averaged = np.zeros_like(total)
    usable = weight_sum > 0.0
    averaged[usable] = total[usable] / weight_sum[usable, None, None]
    return JonesAverageResult(
        jones=np.ascontiguousarray(averaged),
        weight_sum=np.ascontiguousarray(weight_sum),
        pixel_size_um=target_pixel,
        size_um=float(size_um),
        center_px=output_center,
        valid_fraction=valid_fraction,
        background_phase_rad=background_phase_rad,
        background_phase_coherence=background_phase_coherence,
        samples=samples,
        resampling=resampling,
    )


def extract_centered_jones_physical(
    jones: np.ndarray,
    *,
    center: tuple[float, float],
    source_pixel_size_um: float,
    output_pixel_size_um: float,
    crop_size_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected Jones field with shape (x,y,2,2), got {matrix.shape!r}")
    _positive("source_pixel_size_um", source_pixel_size_um)
    _positive("output_pixel_size_um", output_pixel_size_um)
    if int(crop_size_px) < 2:
        raise ValueError("crop_size_px must be at least 2")
    origin = 0.5 * (int(crop_size_px) - 1)
    scale = float(output_pixel_size_um) / float(source_pixel_size_um)
    x = float(center[0]) + (np.arange(int(crop_size_px), dtype=np.float64) - origin) * scale
    y = float(center[1]) + (np.arange(int(crop_size_px), dtype=np.float64) - origin) * scale
    xx, yy = np.meshgrid(x, y, indexing="ij")
    valid = (
        (xx >= 0.0)
        & (xx <= matrix.shape[0] - 1)
        & (yy >= 0.0)
        & (yy <= matrix.shape[1] - 1)
    )
    coordinates = np.vstack([xx.ravel(), yy.ravel()])
    result = np.empty((int(crop_size_px), int(crop_size_px), 2, 2), dtype=np.complex128)
    for output_index in range(2):
        for input_index in range(2):
            values = np.asarray(matrix[..., output_index, input_index])
            real = ndi.map_coordinates(values.real, coordinates, order=1, mode="constant", cval=np.nan)
            imag = ndi.map_coordinates(values.imag, coordinates, order=1, mode="constant", cval=np.nan)
            result[..., output_index, input_index] = (real + 1j * imag).reshape(
                int(crop_size_px), int(crop_size_px)
            )
    result[~valid] = np.nan + 1j * np.nan
    return np.ascontiguousarray(result), np.ascontiguousarray(valid)


def physical_crop_coverage_fraction(
    shape: tuple[int, int],
    *,
    center: tuple[float, float],
    source_pixel_size_um: float,
    output_pixel_size_um: float,
    crop_size_px: int,
) -> float:
    origin = 0.5 * (int(crop_size_px) - 1)
    scale = float(output_pixel_size_um) / float(source_pixel_size_um)
    x = float(center[0]) + (np.arange(int(crop_size_px), dtype=np.float64) - origin) * scale
    y = float(center[1]) + (np.arange(int(crop_size_px), dtype=np.float64) - origin) * scale
    valid_x = (x >= 0.0) & (x <= int(shape[0]) - 1)
    valid_y = (y >= 0.0) & (y <= int(shape[1]) - 1)
    return float(np.mean(valid_x) * np.mean(valid_y))


def drizzle_centered_jones(
    jones: np.ndarray,
    *,
    center: tuple[float, float],
    source_pixel_size_um: float,
    output_pixel_size_um: float,
    crop_size_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deposit native complex samples onto a centered finer physical grid.

    Each source pixel is deposited according to its square detector footprint
    (unit ``pixfrac``).  Fractional center shifts therefore interlace
    independent detector samples without first interpolating every exposure
    onto a common image.
    """

    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected Jones field with shape (x,y,2,2), got {matrix.shape!r}")
    _positive("source_pixel_size_um", source_pixel_size_um)
    _positive("output_pixel_size_um", output_pixel_size_um)
    size = int(crop_size_px)
    if size < 2:
        raise ValueError("crop_size_px must be at least 2")

    output_origin = 0.5 * (size - 1)
    half_width_um = output_origin * float(output_pixel_size_um)
    source_margin = half_width_um / float(source_pixel_size_um) + 1.0
    x0 = max(0, int(np.floor(float(center[0]) - source_margin)))
    x1 = min(matrix.shape[0], int(np.ceil(float(center[0]) + source_margin)) + 1)
    y0 = max(0, int(np.floor(float(center[1]) - source_margin)))
    y1 = min(matrix.shape[1], int(np.ceil(float(center[1]) + source_margin)) + 1)
    source_x = np.arange(x0, x1, dtype=np.float64)
    source_y = np.arange(y0, y1, dtype=np.float64)
    output_x = output_origin + (source_x - float(center[0])) * (
        float(source_pixel_size_um) / float(output_pixel_size_um)
    )
    output_y = output_origin + (source_y - float(center[1])) * (
        float(source_pixel_size_um) / float(output_pixel_size_um)
    )
    xx, yy = np.meshgrid(output_x, output_y, indexing="ij")
    values = np.asarray(matrix[x0:x1, y0:y1], dtype=np.complex128).reshape(-1, 4)
    coordinates_x = xx.ravel()
    coordinates_y = yy.ravel()
    finite = (
        (coordinates_x >= 0.0)
        & (coordinates_x <= size - 1)
        & (coordinates_y >= 0.0)
        & (coordinates_y <= size - 1)
        & np.isfinite(values).all(axis=1)
    )
    coordinates_x = coordinates_x[finite]
    coordinates_y = coordinates_y[finite]
    values = values[finite]
    total = np.zeros((size * size, 4), dtype=np.complex128)
    weight_sum = np.zeros(size * size, dtype=np.float64)
    footprint_half_width = 0.5 * float(source_pixel_size_um) / float(output_pixel_size_um)
    max_offset = int(np.ceil(footprint_half_width + 0.5))
    base_x = np.floor(coordinates_x).astype(np.int64)
    base_y = np.floor(coordinates_y).astype(np.int64)
    for offset_x in range(-max_offset, max_offset + 1):
        grid_x = base_x + offset_x
        overlap_x = np.maximum(
            0.0,
            np.minimum(coordinates_x + footprint_half_width, grid_x + 0.5)
            - np.maximum(coordinates_x - footprint_half_width, grid_x - 0.5),
        )
        for offset_y in range(-max_offset, max_offset + 1):
            grid_y = base_y + offset_y
            overlap_y = np.maximum(
                0.0,
                np.minimum(coordinates_y + footprint_half_width, grid_y + 0.5)
                - np.maximum(coordinates_y - footprint_half_width, grid_y - 0.5),
            )
            weight = overlap_x * overlap_y
            usable = (
                (weight > 0.0)
                & (grid_x >= 0)
                & (grid_x < size)
                & (grid_y >= 0)
                & (grid_y < size)
            )
            if not np.any(usable):
                continue
            flat_index = grid_x[usable] * size + grid_y[usable]
            selected_weight = weight[usable]
            np.add.at(weight_sum, flat_index, selected_weight)
            for component in range(4):
                np.add.at(
                    total[:, component],
                    flat_index,
                    selected_weight * values[usable, component],
                )
    return total.reshape(size, size, 2, 2), weight_sum.reshape(size, size)


def background_phase_statistics(
    field: np.ndarray,
    valid: np.ndarray,
    *,
    inner_fraction: float = 0.94,
) -> tuple[float, float]:
    size_x, size_y = field.shape[:2]
    x = np.arange(size_x, dtype=np.float64) - 0.5 * (size_x - 1)
    y = np.arange(size_y, dtype=np.float64) - 0.5 * (size_y - 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    radius = np.hypot(xx, yy)
    half_width = 0.5 * float(min(size_x, size_y) - 1)
    background = np.asarray(valid, dtype=bool) & (radius >= float(inner_fraction) * half_width)
    trace = 0.5 * (field[..., 0, 0] + field[..., 1, 1])
    selected = trace[background & np.isfinite(trace)]
    selected = selected[np.abs(selected) > 1.0e-12]
    if selected.size < 8:
        raise ValueError("Too few valid background pixels for the phase check")
    unit = selected / np.abs(selected)
    mean = np.mean(unit)
    return float(np.angle(mean)), float(np.abs(mean))


def _warn_background_phase(
    source: Path,
    *,
    phase_rad: float,
    coherence: float,
    max_phase_rad: float,
    min_coherence: float,
) -> None:
    problems: list[str] = []
    if abs(float(phase_rad)) > float(max_phase_rad):
        problems.append(
            f"background phase {phase_rad:+.4f} rad exceeds {max_phase_rad:.4f} rad"
        )
    if float(coherence) < float(min_coherence):
        problems.append(
            f"background phase coherence {coherence:.4f} is below {min_coherence:.4f}"
        )
    if problems:
        warnings.warn(
            f"{source}: {'; '.join(problems)}. No global-phase correction was applied.",
            RuntimeWarning,
            stacklevel=2,
        )


def write_jones_average_npz(
    path: Path,
    result: JonesAverageResult,
    *,
    precision: Literal["single", "double"] = "single",
    source_path_base: Path | None = None,
    force: bool = False,
) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")
    dtype = np.complex64 if precision == "single" else np.complex128
    if precision not in {"single", "double"}:
        raise ValueError("precision must be single or double")
    matrix = result.jones.astype(dtype, copy=False)
    stored_base = (
        None
        if source_path_base is None
        else Path(source_path_base).expanduser().resolve()
    )

    def stored_source_path(source: Path) -> str:
        resolved = source.expanduser().resolve()
        if stored_base is None:
            return str(resolved)
        try:
            return str(resolved.relative_to(stored_base))
        except ValueError:
            return str(resolved)

    manifest = {
        "version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": (
            "complex Jones average after subpixel physical-grid registration; "
            "the source global phases are preserved without correction"
        ),
        "resampling": result.resampling,
        "global_phase_correction": "none",
        "pixel_size_um": result.pixel_size_um,
        "size_um": result.size_um,
        "array_center_px": list(result.center_px),
        "sources": [
            {**asdict(sample), "source": stored_source_path(sample.source)}
            for sample in result.samples
        ],
        "valid_fraction": result.valid_fraction.tolist(),
        "background_phase_rad": result.background_phase_rad.tolist(),
        "background_phase_coherence": result.background_phase_coherence.tolist(),
    }
    arrays = {
        "Jxx": matrix[..., 0, 0],
        "Jxy": matrix[..., 0, 1],
        "Jyx": matrix[..., 1, 0],
        "Jyy": matrix[..., 1, 1],
        PIXEL_SIZE_KEY: np.asarray(result.pixel_size_um, dtype=np.float64),
        "array_center_px": np.asarray(result.center_px, dtype=np.float64),
        "weight_sum": result.weight_sum.astype(np.float32, copy=False),
        "source_paths": np.asarray(
            [stored_source_path(item.source) for item in result.samples]
        ),
        "source_centers_px": np.asarray([item.center for item in result.samples], dtype=np.float64),
        "background_phase_rad": result.background_phase_rad.astype(np.float64, copy=False),
        "background_phase_coherence": result.background_phase_coherence.astype(
            np.float64, copy=False
        ),
        "resampling": np.asarray(result.resampling),
        "global_phase_correction": np.asarray("none"),
        "average_manifest": np.asarray(json.dumps(manifest, ensure_ascii=False)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_center(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.replace(":", ",").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("center must be CX,CY")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid center {value!r}") from exc


def read_centers_csv(path: Path) -> tuple[tuple[float, float], ...]:
    path = path.expanduser().resolve()
    result: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"cx", "cy"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            result.append((float(row["cx"]), float(row["cy"])))
    if not result:
        raise ValueError(f"{path} contains no centers")
    return tuple(result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subpixel-align and coherently average Jones fields from repeated --input and "
            "--center arguments. Arguments are paired in their command-line order."
        )
    )
    parser.add_argument("output", type=Path, help="Output averaged Jones NPZ")
    parser.add_argument("--input", action="append", type=Path, required=True, help="Jones NPZ; repeatable")
    centers = parser.add_mutually_exclusive_group(required=True)
    centers.add_argument("--center", action="append", type=parse_center, help="CX,CY; repeatable")
    centers.add_argument(
        "--centers-file",
        type=Path,
        help=(
            "CSV with cx,cy. With one --input all centers are taken from that field; "
            "otherwise its row count must match the number of inputs."
        ),
    )
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--size-um", type=float)
    size.add_argument("--radius-um", type=float)
    parser.add_argument("--output-pixel-size-um", type=float, default=None)
    parser.add_argument("--resampling", choices=RESAMPLING_METHODS, default="drizzle")
    parser.add_argument("--background-inner-fraction", type=float, default=0.94)
    parser.add_argument("--max-background-phase-rad", type=float, default=0.10)
    parser.add_argument("--min-background-phase-coherence", type=float, default=0.90)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    refuse_existing_outputs(
        (args.output,),
        force=args.force,
        project_root=Path(__file__).resolve().parents[3],
    )
    centers = (
        tuple(args.center)
        if args.center is not None
        else read_centers_csv(args.centers_file)
    )
    inputs = tuple(args.input)
    if len(inputs) == 1 and len(centers) > 1:
        inputs = inputs * len(centers)
    if len(inputs) != len(centers):
        raise SystemExit(
            f"The number of inputs ({len(inputs)}) and centers ({len(centers)}) must match"
        )
    samples = tuple(
        JonesSample(source=source.expanduser().resolve(), center=center)
        for source, center in zip(inputs, centers, strict=True)
    )
    size_um = float(args.size_um if args.size_um is not None else 2.0 * args.radius_um)
    result = average_jones_samples(
        samples,
        size_um=size_um,
        output_pixel_size_um=args.output_pixel_size_um,
        resampling=args.resampling,
        background_inner_fraction=args.background_inner_fraction,
        max_background_phase_rad=args.max_background_phase_rad,
        min_background_phase_coherence=args.min_background_phase_coherence,
        min_valid_fraction=args.min_valid_fraction,
        progress=True,
    )
    write_jones_average_npz(args.output, result, precision=args.precision, force=args.force)
    print(
        f"Averaged {len(samples)} Jones fields to {result.jones.shape[0]} x {result.jones.shape[1]} "
        f"pixels at {result.pixel_size_um:.9g} um/px"
    )
    print(f"Array center: ({result.center_px[0]:.6g}, {result.center_px[1]:.6g}) px")
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


def _positive(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _nonnegative(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")


if __name__ == "__main__":
    raise SystemExit(main())
