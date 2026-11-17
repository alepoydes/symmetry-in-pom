from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from toronoptics.article_data import map_center_between_uniform_grids
from toronoptics.cli import refuse_existing_outputs
from toronoptics.experimental_jones import PIXEL_SIZE_KEY


DIRECTOR_KEY = "PATH"
SNAPSHOT_MANIFEST_KEY = "snapshot_manifest"
SIMULATION_MANIFEST_KEY = "simulation_manifest"
PIXEL_SIZE_NAME = "Pixel_size.txt"
ModelName = Literal[
    "spectral-jones",
    "local-maxwell",
    "marching-wave",
    "bidirectional-marching-wave",
]
MODEL_NAMES: tuple[ModelName, ...] = (
    "spectral-jones",
    "local-maxwell",
    "marching-wave",
    "bidirectional-marching-wave",
)
DEFAULT_PROPAGATION_PHASE_STEP = 0.025
BidirectionalOutput = Literal["transmitted", "reflected"]
BIDIRECTIONAL_OUTPUTS: tuple[BidirectionalOutput, ...] = ("transmitted", "reflected")
Precision = Literal["single", "double"]
PRECISIONS: tuple[Precision, ...] = ("single", "double")
PhaseReferenceMode = Literal["reference", "absolute", "solver-native"]
PHASE_REFERENCE_MODES: tuple[PhaseReferenceMode, ...] = (
    "reference",
    "absolute",
    "solver-native",
)
DEFAULT_EXTERNAL_N = 1.0
DEFAULT_SUBSTRATE_N = 1.52
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DirectorArchive:
    path: Path
    director: np.ndarray
    source_manifest: dict[str, Any] | None
    thickness_um: float | None


@dataclass(frozen=True)
class SimulatedJonesResult:
    source: Path
    destination: Path
    model: ModelName
    shape: tuple[int, int, int, int]
    dtype: str
    pixel_size_um: float
    manifest: dict[str, Any]


def load_director_archive(path: Path) -> DirectorArchive:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        if DIRECTOR_KEY not in data.files:
            raise ValueError(f"{path} does not contain required director array {DIRECTOR_KEY!r}")
        raw = np.asarray(data[DIRECTOR_KEY])
        source_manifest = (
            _decode_json_scalar(data[SNAPSHOT_MANIFEST_KEY])
            if SNAPSHOT_MANIFEST_KEY in data.files
            else None
        )

    director = _coerce_lcsim_path_director(raw, source=path)
    thickness_um = _manifest_thickness_um(source_manifest)
    return DirectorArchive(
        path=path,
        director=director,
        source_manifest=source_manifest,
        thickness_um=thickness_um,
    )


def simulate_jones_matrices(
    director: np.ndarray,
    *,
    model: ModelName,
    wavelength_um: float,
    no: float,
    ne: float,
    thickness_um: float,
    subdiv: int = 1,
    phase_step: float | None = None,
    lambda_tile: int | None = None,
    taichi_arch: str | None = None,
    camera_aperture_enabled: bool = False,
    camera_na: float = 1.0,
    bidirectional_output: BidirectionalOutput = "transmitted",
    bidirectional_iterations: int = 2,
    bidirectional_chunks: int | None = None,
    substrate_thickness_um: float = 0.0,
    substrate_n: float | None = DEFAULT_SUBSTRATE_N,
    external_n: float | None = DEFAULT_EXTERNAL_N,
    phase_reference_mode: PhaseReferenceMode = "reference",
    axisymmetric_second_input: bool = False,
    progress: bool = False,
) -> np.ndarray:
    director = np.ascontiguousarray(np.asarray(director, dtype=np.float32))
    if director.ndim != 4 or director.shape[-1] != 3:
        raise ValueError(f"Director field must have shape (x, y, z, 3), got {director.shape!r}")
    if model not in MODEL_NAMES:
        options = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unsupported model {model!r}; expected one of: {options}")
    _validate_positive("wavelength_um", wavelength_um)
    _validate_positive("no", no)
    _validate_positive("ne", ne)
    _validate_positive("thickness_um", thickness_um)
    if int(subdiv) < 1:
        raise ValueError(f"subdiv must be at least 1, got {subdiv!r}")
    if float(camera_na) < 0.0 or not np.isfinite(float(camera_na)):
        raise ValueError(f"camera_na must be finite and nonnegative, got {camera_na!r}")
    _validate_nonnegative("substrate_thickness_um", substrate_thickness_um)
    if substrate_n is not None:
        _validate_positive("substrate_n", substrate_n)
    if external_n is not None:
        _validate_positive("external_n", external_n)
    if bidirectional_output not in BIDIRECTIONAL_OUTPUTS:
        options = ", ".join(BIDIRECTIONAL_OUTPUTS)
        raise ValueError(
            f"Unsupported bidirectional output {bidirectional_output!r}; "
            f"expected one of: {options}"
        )
    if int(bidirectional_iterations) < 1:
        raise ValueError(
            f"bidirectional_iterations must be at least 1, got {bidirectional_iterations!r}"
        )
    resolved_bidirectional_chunks = _resolve_bidirectional_chunks(
        model=model,
        director_shape=tuple(int(v) for v in director.shape),
        subdiv=int(subdiv),
        thickness_um=float(thickness_um),
        phase_step=phase_step,
        wavelength_um=float(wavelength_um),
        no=float(no),
        ne=float(ne),
        bidirectional_output=bidirectional_output,
        bidirectional_iterations=int(bidirectional_iterations),
        bidirectional_chunks=bidirectional_chunks,
    )

    renderer = _build_optics_renderer(
        model=model,
        director_shape=tuple(int(v) for v in director.shape),
        subdiv=int(subdiv),
        thickness_um=float(thickness_um),
        phase_step=phase_step,
        lambda_tile=lambda_tile,
        taichi_arch=taichi_arch,
        camera_aperture_enabled=bool(camera_aperture_enabled),
        camera_na=float(camera_na),
        bidirectional_iterations=int(bidirectional_iterations),
        bidirectional_chunks=resolved_bidirectional_chunks,
    )
    renderer.upload(director)

    field_kwargs = _monochrome_field_kwargs(
        model=model,
        bidirectional_output=bidirectional_output,
        bidirectional_iterations=int(bidirectional_iterations),
        substrate_thickness_um=float(substrate_thickness_um),
        substrate_n=substrate_n,
        external_n=external_n,
    )
    field_x_args = (
        float(wavelength_um),
        float(no),
        float(ne),
        np.asarray([1.0, 0.0], dtype=np.complex64),
    )
    if progress and model in {"marching-wave", "bidirectional-marching-wave"}:
        field_x = renderer.render_monochrome_field(
            *field_x_args,
            progress=_progress_reporter(f"{model} x input"),
            **field_kwargs,
        )
    else:
        field_x = renderer.render_monochrome_field(*field_x_args, **field_kwargs)
    field_x = _apply_mono_field_phase_reference(
        field_x,
        model=model,
        phase_reference_mode=phase_reference_mode,
        wavelength_um=float(wavelength_um),
        no=float(no),
        thickness_um=float(thickness_um),
        bidirectional_output=bidirectional_output,
        substrate_thickness_um=float(substrate_thickness_um),
        substrate_n=substrate_n,
        external_n=external_n,
    )
    if axisymmetric_second_input:
        if field_x.shape[0] != field_x.shape[1] or field_x.shape[-1] != 2:
            raise ValueError(
                "Axisymmetric input reconstruction requires a square transverse "
                f"field with two components, got {field_x.shape!r}"
            )
        # Rotational covariance gives J(r) e_y = R J(R^-1 r) e_x.
        rotated = np.rot90(field_x, k=1, axes=(0, 1))
        field_y = np.stack((-rotated[..., 1], rotated[..., 0]), axis=-1)
    else:
        field_y_args = (
            float(wavelength_um),
            float(no),
            float(ne),
            np.asarray([0.0, 1.0], dtype=np.complex64),
        )
        if progress and model in {"marching-wave", "bidirectional-marching-wave"}:
            field_y = renderer.render_monochrome_field(
                *field_y_args,
                progress=_progress_reporter(f"{model} y input"),
                **field_kwargs,
            )
        else:
            field_y = renderer.render_monochrome_field(*field_y_args, **field_kwargs)
        field_y = _apply_mono_field_phase_reference(
            field_y,
            model=model,
            phase_reference_mode=phase_reference_mode,
            wavelength_um=float(wavelength_um),
            no=float(no),
            thickness_um=float(thickness_um),
            bidirectional_output=bidirectional_output,
            substrate_thickness_um=float(substrate_thickness_um),
            substrate_n=substrate_n,
            external_n=external_n,
        )
    return jones_from_basis_fields(field_x, field_y)


def _build_optics_renderer(
    *,
    model: ModelName,
    director_shape: tuple[int, ...],
    subdiv: int,
    thickness_um: float,
    phase_step: float | None,
    lambda_tile: int | None,
    taichi_arch: str | None,
    camera_aperture_enabled: bool,
    camera_na: float,
    bidirectional_iterations: int,
    bidirectional_chunks: int | None,
):
    renderer_kwargs: dict[str, Any] = {
        "nn_shape": director_shape,
        "subdiv": int(subdiv),
        "thickness": float(thickness_um),
    }
    if lambda_tile is not None:
        renderer_kwargs["lambda_tile"] = int(lambda_tile)

    if model in {"spectral-jones", "local-maxwell"}:
        from topovec.ti import LocalMaxwell, SpectralJones, configure
        from topovec.ti.runtime import ensure_initialized

        configure(arch=taichi_arch)
        ensure_initialized()
        if model == "local-maxwell" and phase_step is not None:
            renderer_kwargs["phase_step"] = float(phase_step)
        return (
            SpectralJones(**renderer_kwargs)
            if model == "spectral-jones"
            else LocalMaxwell(**renderer_kwargs)
        )

    renderer_kwargs.update(
        {
            "camera_aperture_enabled": bool(camera_aperture_enabled),
            "camera_na": float(camera_na),
        }
    )
    if phase_step is not None:
        renderer_kwargs["phase_step"] = float(phase_step)

    try:
        from topovec.cuda import BidirectionalMarchingWave, MarchingWave
    except ImportError as exc:
        raise ImportError(
            f"Model {model!r} requires topovec CUDA support. "
            "Install this project's declared CUDA dependencies or install "
            "topovec with its `cuda` extra."
        ) from exc

    if model == "marching-wave":
        return MarchingWave(**renderer_kwargs)
    if model == "bidirectional-marching-wave":
        bidirectional_kwargs: dict[str, Any] = {"iterations": int(bidirectional_iterations)}
        if bidirectional_chunks is not None:
            bidirectional_kwargs["chunks"] = int(bidirectional_chunks)
        return BidirectionalMarchingWave(**renderer_kwargs, **bidirectional_kwargs)
    options = ", ".join(MODEL_NAMES)
    raise ValueError(f"Unsupported model {model!r}; expected one of: {options}")


def _monochrome_field_kwargs(
    *,
    model: ModelName,
    bidirectional_output: BidirectionalOutput,
    bidirectional_iterations: int,
    substrate_thickness_um: float,
    substrate_n: float | None,
    external_n: float | None,
) -> dict[str, Any]:
    if model == "marching-wave":
        return {} if substrate_n is None else {"substrate_n": float(substrate_n)}
    if model != "bidirectional-marching-wave":
        return {}
    kwargs: dict[str, Any] = {
        "output": bidirectional_output,
        "iterations": int(bidirectional_iterations),
        "substrate_thickness": float(substrate_thickness_um),
    }
    if substrate_n is not None:
        kwargs["substrate_n"] = float(substrate_n)
    if external_n is not None:
        kwargs["external_n"] = float(external_n)
    return kwargs


def _resolve_bidirectional_chunks(
    *,
    model: ModelName,
    director_shape: tuple[int, ...],
    subdiv: int,
    thickness_um: float,
    phase_step: float | None,
    wavelength_um: float,
    no: float,
    ne: float,
    bidirectional_output: BidirectionalOutput,
    bidirectional_iterations: int,
    bidirectional_chunks: int | None,
) -> int | None:
    if model != "bidirectional-marching-wave":
        return None
    if bidirectional_chunks is not None:
        chunks = int(bidirectional_chunks)
        if chunks < 1:
            raise ValueError(f"bidirectional_chunks must be at least 1, got {bidirectional_chunks!r}")
        return chunks
    if phase_step is None:
        return None
    try:
        from topovec.mgl.bidirectional_marching import recommend_bidirectional_marching_wave_chunks
    except ImportError as exc:
        raise RuntimeError(
            "Automatic bidirectional chunk selection requires "
            "topovec.mgl.bidirectional_marching; pass --bidirectional-chunks explicitly."
        ) from exc
    return int(
        recommend_bidirectional_marching_wave_chunks(
            director_shape=tuple(int(v) for v in director_shape),
            subdiv=int(subdiv),
            thickness=float(thickness_um),
            phase_step=float(phase_step),
            wavelengths=np.asarray([float(wavelength_um)], dtype=np.float32),
            no_values=np.asarray([float(no)], dtype=np.float32),
            ne_values=np.asarray([float(ne)], dtype=np.float32),
            weights=np.asarray([[1.0]], dtype=np.float32),
            projections=np.asarray([[1.0, 0.0]], dtype=np.complex64),
            iterations=int(bidirectional_iterations),
            output=bidirectional_output,
        )
    )


def jones_from_basis_fields(field_x: np.ndarray, field_y: np.ndarray) -> np.ndarray:
    x_input = np.asarray(field_x)
    y_input = np.asarray(field_y)
    if x_input.shape != y_input.shape or x_input.ndim != 3 or x_input.shape[-1] != 2:
        raise ValueError(
            "Basis fields must both have shape (x, y, 2), "
            f"got {x_input.shape!r} and {y_input.shape!r}"
        )
    jones = np.empty(x_input.shape[:2] + (2, 2), dtype=np.result_type(x_input, y_input, np.complex64))
    jones[..., 0, 0] = x_input[..., 0]
    jones[..., 1, 0] = x_input[..., 1]
    jones[..., 0, 1] = y_input[..., 0]
    jones[..., 1, 1] = y_input[..., 1]
    return np.ascontiguousarray(jones)


def _apply_mono_field_phase_reference(
    field: np.ndarray,
    *,
    model: ModelName,
    phase_reference_mode: PhaseReferenceMode,
    wavelength_um: float,
    no: float,
    thickness_um: float,
    bidirectional_output: BidirectionalOutput,
    substrate_thickness_um: float,
    substrate_n: float | None,
    external_n: float | None,
) -> np.ndarray:
    if phase_reference_mode not in PHASE_REFERENCE_MODES:
        options = ", ".join(PHASE_REFERENCE_MODES)
        raise ValueError(
            f"Unsupported phase reference mode {phase_reference_mode!r}; "
            f"expected one of: {options}"
        )
    if model == "spectral-jones":
        convention = "reference"
        solver_native = "reference"
        phasor = _ordinary_isotropic_reference_phasor(
            wavelength_um=wavelength_um,
            no=no,
            thickness_um=thickness_um,
        )
        reference_kind = "ordinary_isotropic_ne_eq_no"
    elif model in {"local-maxwell", "marching-wave"}:
        convention = "absolute"
        solver_native = "absolute"
        phasor = _ordinary_isotropic_reference_phasor(
            wavelength_um=wavelength_um,
            no=no,
            thickness_um=thickness_um,
        )
        reference_kind = "ordinary_isotropic_ne_eq_no"
    elif model == "bidirectional-marching-wave":
        convention = "absolute"
        solver_native = "absolute"
        phasor = _flat_isotropic_stack_reference_phasor(
            wavelength_um=wavelength_um,
            no=no,
            thickness_um=thickness_um,
            substrate_thickness_um=substrate_thickness_um,
            substrate_n=substrate_n,
            external_n=external_n,
            output=bidirectional_output,
        )
        reference_kind = "flat_isotropic_stack_ne_eq_no"
    else:
        options = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unsupported model {model!r}; expected one of: {options}")

    cache: dict[str, object] = {
        "mono_field": np.asarray(field, dtype=np.complex64),
        "mono_field_phase_convention": convention,
        "mono_field_solver_native_mode": solver_native,
    }
    if phasor is not None:
        cache["phase_reference_phasor"] = phasor
        cache["phase_reference_kind"] = reference_kind
    elif phase_reference_mode != "solver-native":
        raise ValueError(
            f"Phase reference mode {phase_reference_mode!r} is unavailable for {model!r}."
        )

    from topovec.optics import prepare_phase_referenced_optics_cache

    updated = prepare_phase_referenced_optics_cache(
        cache,
        requested_mode=phase_reference_mode,
    )
    warning = updated.get("phase_reference_warning")
    if warning is not None:
        raise ValueError(str(warning))
    return np.ascontiguousarray(updated["mono_field"], dtype=np.complex64)


def _ordinary_isotropic_reference_phasor(
    *,
    wavelength_um: float,
    no: float,
    thickness_um: float,
) -> np.ndarray | None:
    if (
        not np.isfinite(float(wavelength_um))
        or float(wavelength_um) <= 0.0
        or not np.isfinite(float(no))
        or not np.isfinite(float(thickness_um))
    ):
        return None
    phase = np.float32(
        2.0 * np.pi * float(no) * float(thickness_um) / float(wavelength_um)
    )
    return np.asarray([[[np.exp(1j * phase)]]], dtype=np.complex64)


def _flat_isotropic_stack_reference_phasor(
    *,
    wavelength_um: float,
    no: float,
    thickness_um: float,
    substrate_thickness_um: float,
    substrate_n: float | None,
    external_n: float | None,
    output: BidirectionalOutput,
) -> np.ndarray | None:
    try:
        from topovec.optics import flat_isotropic_stack_jones_matrix

        jones = flat_isotropic_stack_jones_matrix(
            wavelength_um=float(wavelength_um),
            layer_n=float(no),
            layer_thickness_um=float(thickness_um),
            substrate_n=substrate_n,
            substrate_thickness_um=float(substrate_thickness_um),
            external_n=external_n,
            output=output,
        )
    except (ImportError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    matrix = np.asarray(jones, dtype=np.complex64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        return None
    diagonal = np.asarray((matrix[0, 0], matrix[1, 1]), dtype=np.complex64)
    scale = np.complex64(np.mean(diagonal))
    magnitude = float(abs(scale))
    if magnitude <= 1e-7:
        return None
    residual = matrix - np.eye(2, dtype=np.complex64) * scale
    if float(np.max(np.abs(residual))) > max(1e-5, 1e-3 * magnitude):
        return None
    return np.asarray([[[scale / np.float32(magnitude)]]], dtype=np.complex64)


def write_simulated_jones(
    destination: Path,
    jones: np.ndarray,
    manifest: dict[str, Any],
    *,
    pixel_size_um: float,
    array_center_px: tuple[float, float],
    precision: Precision = "single",
    force: bool = False,
) -> None:
    if precision not in PRECISIONS:
        options = ", ".join(PRECISIONS)
        raise ValueError(f"Unsupported precision {precision!r}; expected one of: {options}")
    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Jones matrices must have shape (x, y, 2, 2), got {matrix.shape!r}")
    _validate_positive("pixel_size_um", pixel_size_um)
    dtype = np.complex64 if precision == "single" else np.complex128
    matrix = np.ascontiguousarray(matrix, dtype=dtype)
    center = np.asarray(array_center_px, dtype=np.float64).reshape(-1)
    if (
        center.size != 2
        or not np.all(np.isfinite(center))
        or np.any(center < 0.0)
        or np.any(center > np.asarray(matrix.shape[:2]) - 1.0)
    ):
        raise ValueError(
            f"array_center_px must lie inside {matrix.shape[:2]}, got "
            f"{array_center_px!r}"
        )
    arrays = {
        "Jxx": matrix[..., 0, 0],
        "Jxy": matrix[..., 0, 1],
        "Jyx": matrix[..., 1, 0],
        "Jyy": matrix[..., 1, 1],
        PIXEL_SIZE_KEY: np.array(float(pixel_size_um), dtype=np.float64),
        "array_center_px": center,
        SIMULATION_MANIFEST_KEY: np.array(json.dumps(manifest, ensure_ascii=False)),
    }
    _write_npz(destination, arrays, force=force)


def simulate_director_archive(
    source: Path,
    destination: Path | None,
    *,
    model: ModelName,
    wavelength_um: float,
    no: float,
    ne: float,
    thickness_um: float | None = None,
    subdiv: int = 1,
    phase_step: float | None = None,
    lambda_tile: int | None = None,
    precision: Precision = "single",
    taichi_arch: str | None = None,
    camera_aperture_enabled: bool = False,
    camera_na: float = 1.0,
    bidirectional_output: BidirectionalOutput = "transmitted",
    bidirectional_iterations: int = 2,
    bidirectional_chunks: int | None = None,
    substrate_thickness_um: float = 0.0,
    substrate_n: float | None = DEFAULT_SUBSTRATE_N,
    external_n: float | None = DEFAULT_EXTERNAL_N,
    phase_reference_mode: PhaseReferenceMode = "reference",
    source_center_px: tuple[float, float] | None = None,
    center_method: str | None = None,
    force: bool = False,
    write_pixel_size: bool = True,
    progress: bool = False,
) -> SimulatedJonesResult:
    _report(progress, f"[simulate] source: {source.expanduser().resolve()}")
    _report(progress, "[simulate] load director archive")
    archive = load_director_archive(source)
    source_sha256 = _sha256(archive.path)
    resolved_thickness = float(thickness_um if thickness_um is not None else archive.thickness_um or 0.0)
    _validate_positive("thickness_um", resolved_thickness)
    destination = (
        _default_destination(archive.path, model=model, wavelength_um=wavelength_um)
        if destination is None
        else destination.expanduser().resolve()
    )
    _report(progress, f"[simulate] destination: {destination}")
    _report(
        progress,
        f"[simulate] model={model}, wavelength={wavelength_um:.6g} um, "
        f"no={no:.6g}, ne={ne:.6g}, thickness={resolved_thickness:.6g} um",
    )
    _report(progress, f"[simulate] phase_reference_mode={phase_reference_mode}")
    source_shape = tuple(int(value) for value in archive.director.shape[:2])
    if source_center_px is None:
        resolved_source_center = tuple(
            0.5 * (float(value) - 1.0) for value in source_shape
        )
        resolved_center_method = "geometric_source_array_midpoint"
    else:
        resolved_source_center = tuple(
            float(value)
            for value in np.asarray(source_center_px, dtype=np.float64).reshape(-1)
        )
        if len(resolved_source_center) != 2 or not np.all(
            np.isfinite(resolved_source_center)
        ):
            raise ValueError(
                f"source_center_px must contain two finite coordinates, got "
                f"{source_center_px!r}"
            )
        if np.any(np.asarray(resolved_source_center) < 0.0) or np.any(
            np.asarray(resolved_source_center) > np.asarray(source_shape) - 1.0
        ):
            raise ValueError(
                f"source_center_px lies outside {source_shape}: "
                f"{resolved_source_center!r}"
            )
        resolved_center_method = center_method or "explicit_source_center_px"
    resolved_bidirectional_chunks = _resolve_bidirectional_chunks(
        model=model,
        director_shape=tuple(int(v) for v in archive.director.shape),
        subdiv=int(subdiv),
        thickness_um=resolved_thickness,
        phase_step=phase_step,
        wavelength_um=float(wavelength_um),
        no=float(no),
        ne=float(ne),
        bidirectional_output=bidirectional_output,
        bidirectional_iterations=int(bidirectional_iterations),
        bidirectional_chunks=bidirectional_chunks,
    )
    if model == "bidirectional-marching-wave":
        chunks_note = "topovec default" if resolved_bidirectional_chunks is None else str(resolved_bidirectional_chunks)
        source_note = "auto" if bidirectional_chunks is None and resolved_bidirectional_chunks is not None else "explicit"
        if resolved_bidirectional_chunks is None and bidirectional_chunks is None:
            source_note = "implicit"
        _report(progress, f"[simulate] bidirectional chunks={chunks_note} ({source_note})")

    _report(progress, "[simulate] compute Jones matrices")
    jones = simulate_jones_matrices(
        archive.director,
        model=model,
        wavelength_um=wavelength_um,
        no=no,
        ne=ne,
        thickness_um=resolved_thickness,
        subdiv=subdiv,
        phase_step=phase_step,
        lambda_tile=lambda_tile,
        taichi_arch=taichi_arch,
        camera_aperture_enabled=camera_aperture_enabled,
        camera_na=camera_na,
        bidirectional_output=bidirectional_output,
        bidirectional_iterations=bidirectional_iterations,
        bidirectional_chunks=resolved_bidirectional_chunks,
        substrate_thickness_um=substrate_thickness_um,
        substrate_n=substrate_n,
        external_n=external_n,
        phase_reference_mode=phase_reference_mode,
        progress=progress,
    )
    pixel_size_um = resolved_thickness / (archive.director.shape[2] - 1) / int(subdiv)
    target_sampling = (
        "source-nodes" if model == "spectral-jones" else "cell-centers"
    )
    array_center_px = map_center_between_uniform_grids(
        resolved_source_center,
        source_shape=source_shape,
        target_shape=tuple(int(value) for value in jones.shape[:2]),
        target_sampling=target_sampling,
    )
    _report(progress, f"[simulate] computed Jones shape={jones.shape}, pixel_size={pixel_size_um:.6g} um")
    manifest = _simulation_manifest(
        source=archive.path,
        source_sha256=source_sha256,
        destination=destination,
        source_manifest=archive.source_manifest,
        model=model,
        wavelength_um=wavelength_um,
        no=no,
        ne=ne,
        thickness_um=resolved_thickness,
        subdiv=subdiv,
        phase_step=phase_step,
        lambda_tile=lambda_tile,
        taichi_arch=taichi_arch,
        camera_aperture_enabled=camera_aperture_enabled,
        camera_na=camera_na,
        bidirectional_output=bidirectional_output,
        bidirectional_iterations=bidirectional_iterations,
        bidirectional_chunks=resolved_bidirectional_chunks,
        bidirectional_chunks_auto=bidirectional_chunks is None,
        substrate_thickness_um=substrate_thickness_um,
        substrate_n=substrate_n,
        external_n=external_n,
        phase_reference_mode=phase_reference_mode,
        precision=precision,
        director_shape=archive.director.shape,
        jones_shape=jones.shape,
        pixel_size_um=pixel_size_um,
        source_center_px=resolved_source_center,
        array_center_px=array_center_px,
        center_method=resolved_center_method,
    )
    _report(progress, f"[simulate] write Jones archive: {destination}")
    write_simulated_jones(
        destination,
        jones,
        manifest,
        pixel_size_um=pixel_size_um,
        array_center_px=array_center_px,
        precision=precision,
        force=force,
    )
    if write_pixel_size:
        _report(progress, f"[simulate] write pixel size file in {destination.parent}")
        _write_pixel_size(destination.parent, pixel_size_um)
    return SimulatedJonesResult(
        source=archive.path,
        destination=destination,
        model=model,
        shape=tuple(int(v) for v in jones.shape),
        dtype=str(np.dtype(np.complex64 if precision == "single" else np.complex128)),
        pixel_size_um=float(pixel_size_um),
        manifest=manifest,
    )


def _coerce_lcsim_path_director(raw: np.ndarray, *, source: Path) -> np.ndarray:
    array = np.asarray(raw)
    if array.ndim == 6:
        if array.shape[0] < 1 or array.shape[-2:] != (1, 3):
            raise ValueError(f"{source} has unsupported {DIRECTOR_KEY!r} shape {array.shape!r}")
        director = array[0, :, :, :, 0, :]
    elif array.ndim == 5:
        if array.shape[-2:] != (1, 3):
            raise ValueError(f"{source} has unsupported {DIRECTOR_KEY!r} shape {array.shape!r}")
        director = array[:, :, :, 0, :]
    elif array.ndim == 4 and array.shape[-1] == 3:
        director = array
    else:
        raise ValueError(f"{source} has unsupported {DIRECTOR_KEY!r} shape {array.shape!r}")
    return np.ascontiguousarray(director, dtype=np.float32)


def _decode_json_scalar(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"JSON metadata must be a scalar string, got shape {array.shape!r}")
    raw = array.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise TypeError(f"JSON metadata must be a string, got {type(raw).__name__}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON metadata must decode to an object")
    return data


def _manifest_thickness_um(manifest: dict[str, Any] | None) -> float | None:
    if manifest is None:
        return None
    try:
        return float(manifest["problem"]["geometry"]["thickness_um"])
    except (KeyError, TypeError, ValueError):
        return None


def _simulation_manifest(
    *,
    source: Path,
    source_sha256: str,
    destination: Path,
    source_manifest: dict[str, Any] | None,
    model: ModelName,
    wavelength_um: float,
    no: float,
    ne: float,
    thickness_um: float,
    subdiv: int,
    phase_step: float | None,
    lambda_tile: int | None,
    taichi_arch: str | None,
    camera_aperture_enabled: bool,
    camera_na: float,
    bidirectional_output: BidirectionalOutput,
    bidirectional_iterations: int,
    bidirectional_chunks: int | None,
    bidirectional_chunks_auto: bool,
    substrate_thickness_um: float,
    substrate_n: float | None,
    external_n: float | None,
    phase_reference_mode: PhaseReferenceMode,
    precision: Precision,
    director_shape: tuple[int, ...],
    jones_shape: tuple[int, ...],
    pixel_size_um: float,
    source_center_px: tuple[float, float],
    array_center_px: tuple[float, float],
    center_method: str,
) -> dict[str, Any]:
    cuda_model = model in {"marching-wave", "bidirectional-marching-wave"}
    bidirectional_model = model == "bidirectional-marching-wave"
    return {
        "version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": _portable_project_path(source),
        "source_sha256": source_sha256,
        "destination": _portable_project_path(destination),
        "software": {
            "topovec": version("topovec"),
            "numpy": np.__version__,
        },
        "source_kind": "lcsim director-field npz",
        "axis_order": {
            "director": "director[x, y, z, component]",
            "jones_matrix": "J[x, y, output_component, input_component]",
        },
        "source_director_shape": list(director_shape),
        "jones_shape": list(jones_shape),
        "pixel_size_um": float(pixel_size_um),
        "source_center_px": [float(value) for value in source_center_px],
        "array_center_px": [float(value) for value in array_center_px],
        "center_method": {
            "name": str(center_method),
            "target_sampling": (
                "source-nodes" if model == "spectral-jones" else "cell-centers"
            ),
            "mapping": (
                "c_out = s*c_in"
                if model == "spectral-jones"
                else "c_out = s*c_in - 1/2"
            ),
            "transverse_scale": (
                float(jones_shape[0]) / float(director_shape[0])
            ),
        },
        "optics": {
            "model": model,
            "wavelength_um": float(wavelength_um),
            "no": float(no),
            "ne": float(ne),
            "thickness_um": float(thickness_um),
            "subdiv": int(subdiv),
            "phase_step": None if phase_step is None else float(phase_step),
            "lambda_tile": None if lambda_tile is None else int(lambda_tile),
            "taichi_arch": taichi_arch or "auto",
            "camera_aperture_enabled": (
                bool(camera_aperture_enabled) if cuda_model else None
            ),
            "camera_na": float(camera_na) if cuda_model else None,
            "bidirectional_output": bidirectional_output if bidirectional_model else None,
            "bidirectional_iterations": (
                int(bidirectional_iterations) if bidirectional_model else None
            ),
            "bidirectional_chunks": (
                None
                if not bidirectional_model or bidirectional_chunks is None
                else int(bidirectional_chunks)
            ),
            "bidirectional_chunks_auto": (
                bool(bidirectional_chunks_auto) if bidirectional_model else None
            ),
            "substrate_thickness_um": (
                float(substrate_thickness_um) if bidirectional_model else None
            ),
            "substrate_n": (
                None
                if not cuda_model or substrate_n is None
                else float(substrate_n)
            ),
            "external_n": (
                None
                if not bidirectional_model or external_n is None
                else float(external_n)
            ),
            "phase_reference_mode": phase_reference_mode,
            "precision": precision,
        },
        "source_snapshot_manifest": source_manifest,
    }


def _default_destination(source: Path, *, model: ModelName, wavelength_um: float) -> Path:
    wavelength_nm = int(round(float(wavelength_um) * 1000.0))
    return PROJECT_ROOT / "data" / "simulated" / f"{source.stem}_{wavelength_nm}nm_{model}.npz"


def _portable_project_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return resolved.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz(destination: Path, arrays: dict[str, np.ndarray], *, force: bool) -> None:
    destination = destination.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"Output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_pixel_size(directory: Path, pixel_size_um: float) -> Path:
    """Atomically synchronize the pixel-size sidecar."""

    value = float(pixel_size_um)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"pixel_size_um must be finite and positive, got {pixel_size_um!r}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PIXEL_SIZE_NAME
    text = f"{value:.12g} um\n"
    existing = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else None
    )
    if existing == text:
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if existing is not None:
        print(
            f"[simulate] updated stale pixel-size sidecar: {path} -> "
            f"{text.strip()}",
            flush=True,
        )
    return path


def refresh_pixel_size_sidecar_from_archive(archive_path: Path) -> Path:
    """Refresh ``Pixel_size.txt`` from an existing Jones archive only.

    This maintenance path reads the scalar output sampling already stored in
    the archive.  It never imports or invokes an optical solver.
    """

    resolved = archive_path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as archive:
        if PIXEL_SIZE_KEY not in archive.files:
            raise ValueError(
                f"{resolved} does not contain required scalar "
                f"{PIXEL_SIZE_KEY!r}"
            )
        pixel_size = np.asarray(
            archive[PIXEL_SIZE_KEY],
            dtype=np.float64,
        )
    if pixel_size.shape != ():
        raise ValueError(
            f"{resolved} has non-scalar {PIXEL_SIZE_KEY!r}: "
            f"shape={pixel_size.shape!r}"
        )
    return _write_pixel_size(resolved.parent, float(pixel_size))


def _report(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _progress_reporter(label: str) -> Callable[[float, float], None]:
    """Return a sparse terminal reporter for TopoVec propagation progress."""

    last_bucket = -10

    def report(completed: float, total: float) -> None:
        nonlocal last_bucket
        denominator = max(float(total), np.finfo(float).tiny)
        percent = int(np.clip(100.0 * float(completed) / denominator, 0.0, 100.0))
        bucket = min(100, (percent // 10) * 10)
        if bucket <= last_bucket:
            return
        last_bucket = bucket
        print(f"[simulate] {label}: {bucket}%", flush=True)

    return report


def _validate_positive(name: str, value: float) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def _validate_nonnegative(name: str, value: float) -> None:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute topovec Jones matrices from an lcsim director-field NPZ."
    )
    parser.add_argument("source", type=Path, help="Input director-field .npz")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        help="Output Jones .npz; defaults to ../data/simulated/<source>_<wavelength>_<model>.npz",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="spectral-jones")
    parser.add_argument("--wavelength-um", type=float, required=True)
    parser.add_argument("--no", type=float, required=True, help="Ordinary refractive index")
    parser.add_argument("--ne", type=float, required=True, help="Extraordinary refractive index")
    parser.add_argument(
        "--thickness-um",
        type=float,
        default=None,
        help="LC thickness in micrometers; defaults to snapshot_manifest geometry thickness",
    )
    parser.add_argument("--subdiv", type=int, default=1, help="Topovec transverse/z subdivision")
    parser.add_argument(
        "--phase-step",
        type=float,
        default=None,
        help=(
            "Local Maxwell and CUDA marching phase step; default: 0.025 for "
            "propagation models; ignored by spectral-jones"
        ),
    )
    parser.add_argument("--lambda-tile", type=int, default=None)
    parser.add_argument(
        "--camera-aperture",
        action="store_true",
        help="Enable coherent camera aperture for CUDA marching models",
    )
    parser.add_argument(
        "--camera-na",
        type=float,
        default=1.0,
        help="Camera numerical aperture for CUDA marching models",
    )
    parser.add_argument(
        "--bidirectional-output",
        choices=BIDIRECTIONAL_OUTPUTS,
        default="transmitted",
        help="Output channel for bidirectional-marching-wave",
    )
    parser.add_argument(
        "--bidirectional-iterations",
        type=int,
        default=2,
        help="Visible pass count for bidirectional-marching-wave",
    )
    parser.add_argument(
        "--bidirectional-chunks",
        type=_parse_bidirectional_chunks,
        default=None,
        help="Z chunks for bidirectional-marching-wave; default: auto",
    )
    parser.add_argument(
        "--substrate-thickness-um",
        type=float,
        default=0.0,
        help="Substrate thickness for bidirectional port matching",
    )
    parser.add_argument(
        "--substrate-n",
        type=float,
        default=DEFAULT_SUBSTRATE_N,
        help="Substrate refractive index for CUDA marching models",
    )
    parser.add_argument(
        "--external-n",
        type=float,
        default=DEFAULT_EXTERNAL_N,
        help="External medium refractive index for bidirectional port matching",
    )
    parser.add_argument(
        "--phase-reference-mode",
        choices=PHASE_REFERENCE_MODES,
        default="reference",
        help=(
            "Mono-field phase convention saved to Jones archives; "
            "default: reference wave."
        ),
    )
    parser.add_argument(
        "--source-center-px",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help=(
            "Physical-axis center in source-director pixels; mapped with the "
            "selected solver's transverse sampling convention and stored as "
            "array_center_px"
        ),
    )
    parser.add_argument(
        "--center-method",
        default=None,
        help="Short provenance label stored with --source-center-px",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="single",
        help="Output complex precision; default: single.",
    )
    parser.add_argument(
        "--taichi-arch",
        choices=("auto", "cpu", "cuda", "vulkan", "metal"),
        default="auto",
        help="Taichi backend for spectral-jones/local-maxwell. Ignored by CUDA models.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument(
        "--no-pixel-size-file",
        action="store_true",
        help=f"Do not create {PIXEL_SIZE_NAME} next to the output",
    )
    return parser


def _parse_bidirectional_chunks(value: str) -> int | None:
    text = str(value).strip().lower()
    if text == "auto":
        return None
    try:
        chunks = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'auto'") from exc
    if chunks < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'auto'")
    return chunks


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    destination = (
        _default_destination(
            args.source.expanduser().resolve(),
            model=args.model,
            wavelength_um=args.wavelength_um,
        )
        if args.destination is None
        else args.destination.expanduser().resolve()
    )
    refuse_existing_outputs(
        (destination,),
        force=args.force,
        project_root=Path(__file__).resolve().parents[3],
    )
    phase_step = args.phase_step
    if phase_step is None and args.model != "spectral-jones":
        phase_step = DEFAULT_PROPAGATION_PHASE_STEP
    result = simulate_director_archive(
        args.source,
        destination,
        model=args.model,
        wavelength_um=args.wavelength_um,
        no=args.no,
        ne=args.ne,
        thickness_um=args.thickness_um,
        subdiv=args.subdiv,
        phase_step=phase_step,
        lambda_tile=args.lambda_tile,
        precision=args.precision,
        taichi_arch=None if args.taichi_arch == "auto" else args.taichi_arch,
        camera_aperture_enabled=args.camera_aperture,
        camera_na=args.camera_na,
        bidirectional_output=args.bidirectional_output,
        bidirectional_iterations=args.bidirectional_iterations,
        bidirectional_chunks=args.bidirectional_chunks,
        substrate_thickness_um=args.substrate_thickness_um,
        substrate_n=args.substrate_n,
        external_n=args.external_n,
        phase_reference_mode=args.phase_reference_mode,
        source_center_px=(
            None
            if args.source_center_px is None
            else tuple(map(float, args.source_center_px))
        ),
        center_method=args.center_method,
        force=args.force,
        write_pixel_size=not args.no_pixel_size_file,
        progress=True,
    )
    print(
        "Simulated Jones matrices: "
        f"{result.source} -> {result.destination}; "
        f"model={result.model}, shape={result.shape}, dtype={result.dtype}, "
        f"pixel_size={result.pixel_size_um:.6g} um."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
