from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import convolve1d

ARTICLE_WAVELENGTHS_NM: tuple[int, ...] = (450, 520, 660)
ARTICLE_GREEN_WAVELENGTH_NM = 520
TILTED_CONTROL_WAVELENGTH_NM = ARTICLE_GREEN_WAVELENGTH_NM
ARTICLE_COMPARISON_FIELD_WIDTH_UM = 96.0
MODEL_DIRECTOR_FILE_NAME = "toron_27um_5cb_soft.npz"
MODEL_DIRECTORY_NAME = Path(MODEL_DIRECTOR_FILE_NAME).stem
TILTED_DIRECTOR_FILE_NAME = "toron_27um_5cb_soft_inclined_0.03.npz"
TILTED_MODEL_DIRECTORY_NAME = Path(TILTED_DIRECTOR_FILE_NAME).stem
# Fixed reference axis of the primary director.  Simulated archives store the
# corresponding model-specific output-grid center explicitly; this source-grid
# value supplies the center when only grid dimensions are recorded.
ARTICLE_DIRECTOR_AXIS_CENTER_PX = (100.987577711, 100.987570909)
# The inclined state has no exact rotational axis.  Its independent structural
# origin is the volume centroid of the transverse director weight
# n_x^2+n_y^2, evaluated on the supplied source grid.  Using the primary
# toron's center here would turn a rigid offset into spurious odd-mode power.
TILTED_DIRECTOR_STRUCTURAL_CENTER_PX = (
    100.96645890033328,
    98.85857979393059,
)
ARTICLE_VARIANT_SUFFIX = "_soft"
EXPERIMENT_DIRECTORY_NAME = "results_toron25um_50X_averaged"
PUPIL_REPORT_RELATIVE_PATH = Path("data/analysis/averaged_pupil_spectrum/report.json")


@dataclass(frozen=True)
class OpticalParameters:
    wavelength_um: float
    ordinary_index: float
    extraordinary_index: float


# The three entries reproduce the article wavelengths.  The 520-nm values are
# evaluated from the same three-term Cauchy interpolation that supplied the
# blue and red optical constants.
OPTICAL_PARAMETERS: dict[int, OpticalParameters] = {
    450: OpticalParameters(0.453, 1.55824, 1.76726),
    520: OpticalParameters(0.520, 1.5440722923, 1.7335786888),
    660: OpticalParameters(0.662, 1.52993, 1.70178),
}


def experimental_average_path(project_root: Path, wavelength_nm: int) -> Path:
    _require_article_wavelength(wavelength_nm)
    return (
        Path(project_root)
        / "data"
        / "experimental"
        / EXPERIMENT_DIRECTORY_NAME
        / f"{int(wavelength_nm)}nm_average.npz"
    )


def model_director_path(project_root: Path) -> Path:
    return Path(project_root) / "data_src" / "directors" / MODEL_DIRECTOR_FILE_NAME


def tilted_director_path(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "data_src"
        / "directors"
        / TILTED_DIRECTOR_FILE_NAME
    )


def tilted_simulated_jones_path(
    project_root: Path,
    wavelength_nm: int,
    model: str,
) -> Path:
    _require_article_wavelength(wavelength_nm)
    if model not in {"marching-wave", "spectral-jones"}:
        raise ValueError(f"Unsupported article model: {model!r}")
    return (
        Path(project_root)
        / "data"
        / "simulated"
        / TILTED_MODEL_DIRECTORY_NAME
        / f"{int(wavelength_nm)}nm_{model}.npz"
    )


def simulated_jones_path(
    project_root: Path,
    wavelength_nm: int,
    model: str,
) -> Path:
    _require_article_wavelength(wavelength_nm)
    if model not in {"marching-wave", "spectral-jones"}:
        raise ValueError(f"Unsupported article model: {model!r}")
    return (
        Path(project_root)
        / "data"
        / "simulated"
        / MODEL_DIRECTORY_NAME
        / f"{int(wavelength_nm)}nm_{model}.npz"
    )


def experimental_array_center(path: Path) -> tuple[float, float]:
    with np.load(path, allow_pickle=False) as data:
        if "array_center_px" not in data.files:
            raise ValueError(f"{path} does not contain array_center_px")
        center = np.asarray(data["array_center_px"], dtype=np.float64).reshape(-1)
    if center.size != 2 or not np.all(np.isfinite(center)):
        raise ValueError(f"Invalid array_center_px in {path}: {center!r}")
    return float(center[0]), float(center[1])


def map_center_between_uniform_grids(
    center_px: tuple[float, float],
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    target_sampling: Literal["source-nodes", "cell-centers"],
) -> tuple[float, float]:
    """Map a source-lattice center to a uniformly refined optical grid.

    TopoVec's independent-column ``SpectralJones`` samples the source director
    at ``x_out / s`` and therefore uses ``c_out = s*c_in``.  Its Maxwell and
    marching-wave grids sample transverse cell centers at
    ``(x_out + 1/2) / s`` and therefore use
    ``c_out = s*c_in - 1/2``.  Keeping the convention explicit prevents a
    half-output-pixel centering error when the two models are compared.
    """

    center = np.asarray(center_px, dtype=np.float64).reshape(-1)
    source = np.asarray(source_shape, dtype=np.float64).reshape(-1)
    target = np.asarray(target_shape, dtype=np.float64).reshape(-1)
    if center.size != 2 or not np.all(np.isfinite(center)):
        raise ValueError(f"Invalid source-grid center: {center_px!r}")
    if (
        source.size != 2
        or target.size != 2
        or np.any(source <= 0.0)
        or np.any(target <= 0.0)
    ):
        raise ValueError(
            f"Expected positive two-dimensional shapes, got "
            f"{source_shape!r} -> {target_shape!r}"
        )
    scale = target / source
    if not np.allclose(scale, scale[0], rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"Center mapping requires uniform transverse scaling, got {scale!r}"
        )
    if target_sampling == "source-nodes":
        offset = 0.0
    elif target_sampling == "cell-centers":
        offset = 0.5
    else:
        raise ValueError(
            "target_sampling must be 'source-nodes' or 'cell-centers'"
        )
    mapped = scale * center - offset
    return float(mapped[0]), float(mapped[1])


def director_reference_center_for_source(
    source: str | Path | None,
) -> tuple[float, float]:
    """Return the declared source-grid structural origin for an article input."""

    if source is not None and Path(source).name == TILTED_DIRECTOR_FILE_NAME:
        return TILTED_DIRECTOR_STRUCTURAL_CENTER_PX
    return ARTICLE_DIRECTOR_AXIS_CENTER_PX


def simulated_array_center(path: Path) -> tuple[float, float]:
    """Return the physical-axis center stored by a simulated Jones archive.

    Read ``array_center_px`` directly when present. Otherwise map the declared
    director-axis center using the source and output grid dimensions in
    ``simulation_manifest``. Use the array midpoint only when neither form of
    metadata is available.
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "Jxx" not in data.files:
            raise ValueError(f"{path} does not contain a simulated Jones field")
        target_shape = tuple(int(value) for value in data["Jxx"].shape[:2])
        if "array_center_px" in data.files:
            center = np.asarray(
                data["array_center_px"], dtype=np.float64
            ).reshape(-1)
            if (
                center.size != 2
                or not np.all(np.isfinite(center))
                or np.any(center < 0.0)
                or np.any(center > np.asarray(target_shape) - 1.0)
            ):
                raise ValueError(
                    f"Invalid array_center_px in {path}: {center!r}"
                )
            return float(center[0]), float(center[1])
        raw_manifest = (
            np.asarray(data["simulation_manifest"]).item()
            if "simulation_manifest" in data.files
            else None
        )

    if isinstance(raw_manifest, bytes):
        raw_manifest = raw_manifest.decode("utf-8")
    if isinstance(raw_manifest, str):
        try:
            manifest = json.loads(raw_manifest)
            source_shape = tuple(
                int(value) for value in manifest["source_director_shape"][:2]
            )
            model = str(manifest["optics"]["model"])
            source = manifest.get("source")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        else:
            target_sampling = (
                "source-nodes"
                if model == "spectral-jones"
                else "cell-centers"
            )
            return map_center_between_uniform_grids(
                director_reference_center_for_source(source),
                source_shape=source_shape,
                target_shape=target_shape,
                target_sampling=target_sampling,
            )

    midpoint = 0.5 * (np.asarray(target_shape, dtype=np.float64) - 1.0)
    return float(midpoint[0]), float(midpoint[1])


def optical_parameters(wavelength_nm: int) -> OpticalParameters:
    try:
        return OPTICAL_PARAMETERS[int(wavelength_nm)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported optical wavelength: {wavelength_nm} nm"
        ) from exc


def experimental_jones_propagation_wavelength_um(
    wavelength_nm: int,
) -> float:
    """Return the common wavelength assigned to a full experimental Jones field.

    A reconstructed Jones matrix is treated as one monochromatic linear
    operator, so both input columns are propagated at the calculation
    wavelength associated with the dataset label.  Source-specific measured
    wavelengths belong to controls and to single-input direct-POM comparisons,
    not to the primary full-matrix propagation.
    """

    _require_article_wavelength(wavelength_nm)
    return float(optical_parameters(wavelength_nm).wavelength_um)


def publication_focus_grid(
    evolution_report: dict,
    *,
    wavelength_nm: int = ARTICLE_GREEN_WAVELENGTH_NM,
) -> tuple[np.ndarray, float, float]:
    """Return the five exact through-focus planes used in the main figure."""

    key = str(int(wavelength_nm))
    surface_um = float(
        evolution_report["annular_focus_alignment"][key][
            "inferred_experimental_surface_z_um"
        ]
    )
    annular_focus_um = float(
        evolution_report["annular_focus"][f"{key}_experiment"]["z_um"]
    )
    step_um = 0.5 * (annular_focus_um - surface_um)
    focus_um = surface_um + step_um * np.arange(-1, 4, dtype=np.float64)
    return focus_um, surface_um, annular_focus_um


def effective_coherent_na(project_root: Path, wavelength_nm: int) -> float:
    """Return the robust source-ensemble pupil estimate used for model matching."""
    _require_article_wavelength(wavelength_nm)
    report_path = Path(project_root) / PUPIL_REPORT_RELATIVE_PATH
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        value = report["per_wavelength"][str(int(wavelength_nm))][
            "source_median_effective_na"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Missing {wavelength_nm}-nm source-median pupil estimate in {report_path}"
        ) from exc
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"Invalid effective NA in {report_path}: {value!r}")
    return value


def decimate_centered_jones(
    jones: np.ndarray,
    center_px: tuple[float, float],
    *,
    factor: int = 2,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Low-pass and decimate an odd centered Jones grid by an integer factor."""
    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected Jones shape (x, y, 2, 2), got {matrix.shape!r}")
    if int(factor) != 2:
        raise ValueError("The centered drizzle reduction currently supports factor=2")
    if any(size % 2 != 1 for size in matrix.shape[:2]):
        raise ValueError(f"Centered decimation requires odd spatial dimensions, got {matrix.shape[:2]}")

    kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float64)
    filtered = convolve1d(matrix, kernel, axis=0, mode="reflect")
    filtered = convolve1d(filtered, kernel, axis=1, mode="reflect")
    reduced = np.ascontiguousarray(filtered[::2, ::2])
    reduced_center = tuple(float(value) / 2.0 for value in center_px)
    return reduced, reduced_center


def _require_article_wavelength(wavelength_nm: int) -> None:
    if int(wavelength_nm) not in ARTICLE_WAVELENGTHS_NM:
        raise ValueError(
            f"Expected an article wavelength in {ARTICLE_WAVELENGTHS_NM}, "
            f"got {wavelength_nm!r}"
        )
