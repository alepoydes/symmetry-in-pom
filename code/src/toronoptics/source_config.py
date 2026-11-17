"""Typed access to the checked-in source-data manifests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_MANIFEST = PROJECT_ROOT / "data_src" / "experiment.toml"


@dataclass(frozen=True)
class DirectPOMConfig:
    exposure_times_ms: tuple[int, int]
    selected_frame_numbers: tuple[int, ...]
    display_frame_numbers: tuple[int, ...]
    pixel_size_um: float
    wavelength_um: float
    translation_reading_per_frame_um: float
    beamsplitter_amplitude_ratio: float
    encoded_shape_yx: tuple[int, int]
    analyzer_quadrant_shape_yx: tuple[int, int]


@dataclass(frozen=True)
class HolographyColumnWavelengthConfig:
    """Fixed control inputs and separately recorded source measurements."""

    primary_wavelength_um: float
    horizontal_wavelength_um: float
    vertical_wavelength_um: float
    measured_horizontal_wavelength_um: float
    measured_horizontal_uncertainty_um: float
    measured_vertical_wavelength_um: float
    measured_vertical_uncertainty_um: float
    measurement_instrument: str


def _positive_float(table: dict[str, Any], key: str) -> float:
    value = float(table[key])
    if not _is_positive_finite(value):
        raise ValueError(f"{key} must be positive and finite, got {value!r}")
    return value


def load_holography_column_wavelength_config(
    path: Path = DEFAULT_EXPERIMENT_MANIFEST,
) -> HolographyColumnWavelengthConfig:
    """Read fixed propagation-control inputs independently of measured centers."""

    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != "symmetry-in-pom.source-data.v1":
        raise ValueError(f"unsupported experiment manifest schema: {path}")
    measured = document["holography"]["primary_full_jones"]
    control = document["holography"]["column_wavelength_control"]
    instrument = str(measured["wavelength_measurement_instrument"]).strip()
    if not instrument:
        raise ValueError("wavelength_measurement_instrument must be nonempty")
    return HolographyColumnWavelengthConfig(
        primary_wavelength_um=0.001 * _positive_float(control, "common_wavelength_nm"),
        horizontal_wavelength_um=0.001 * _positive_float(control, "horizontal_wavelength_nm"),
        vertical_wavelength_um=0.001 * _positive_float(control, "vertical_wavelength_nm"),
        measured_horizontal_wavelength_um=0.001 * _positive_float(measured, "horizontal_input_wavelength_nm"),
        measured_horizontal_uncertainty_um=0.001 * _positive_float(measured, "horizontal_input_wavelength_uncertainty_nm"),
        measured_vertical_wavelength_um=0.001 * _positive_float(measured, "vertical_input_wavelength_nm"),
        measured_vertical_uncertainty_um=0.001 * _positive_float(measured, "vertical_input_wavelength_uncertainty_nm"),
        measurement_instrument=instrument,
    )


def _is_positive_finite(value: float) -> bool:
    """Avoid importing NumPy solely to validate scalar TOML values."""

    return math.isfinite(value) and value > 0.0


def load_direct_pom_config(
    path: Path = DEFAULT_EXPERIMENT_MANIFEST,
) -> DirectPOMConfig:
    """Read and validate the direct-POM parameters used by the reductions."""

    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != "symmetry-in-pom.source-data.v1":
        raise ValueError(f"unsupported experiment manifest schema: {path}")
    table = document["direct_pom"]
    expected_quadrants = (
        "top-left:I0",
        "top-right:I135",
        "bottom-left:I45",
        "bottom-right:I90",
    )
    if tuple(str(value) for value in table["analyzer_quadrants"]) != expected_quadrants:
        raise ValueError("the direct-POM analyzer-quadrant map is unsupported")
    if table["environment_file"] != "Environment.png":
        raise ValueError("the direct-POM dark-frame filename is unsupported")
    if table["ground_file_glob"] != "Ground (*).png":
        raise ValueError("the direct-POM Ground filename pattern is unsupported")
    if str(table["input_polarization"]).lower() != "horizontal":
        raise ValueError("the focus comparison currently requires horizontal input")
    exposures = tuple(int(value) for value in table["exposure_times_ms"])
    if exposures != (12, 42):
        raise ValueError(
            "the current HDR algorithm requires the ordered 12/42-ms pair; "
            f"manifest contains {exposures!r}"
        )
    frames = tuple(int(value) for value in table["selected_frame_numbers"])
    display_frames = tuple(int(value) for value in table["display_frame_numbers"])
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("selected_frame_numbers must be nonempty and unique")
    if set(display_frames) != set(frames):
        raise ValueError("display_frame_numbers must be a permutation of selected frames")
    encoded_shape = tuple(int(value) for value in table["encoded_shape_yx"])
    quadrant_shape = tuple(int(value) for value in table["analyzer_quadrant_shape_yx"])
    if encoded_shape != (2 * quadrant_shape[0], 2 * quadrant_shape[1]):
        raise ValueError("encoded_shape_yx must contain a 2x2 analyzer-quadrant array")
    return DirectPOMConfig(
        exposure_times_ms=(exposures[0], exposures[1]),
        selected_frame_numbers=frames,
        display_frame_numbers=display_frames,
        pixel_size_um=_positive_float(table, "component_pixel_size_um"),
        wavelength_um=0.001 * _positive_float(table, "wavelength_center_nm"),
        translation_reading_per_frame_um=_positive_float(
            table, "translation_reading_per_frame_um"
        ),
        beamsplitter_amplitude_ratio=_positive_float(
            table, "beamsplitter_vertical_to_horizontal_amplitude_ratio"
        ),
        encoded_shape_yx=(encoded_shape[0], encoded_shape[1]),
        analyzer_quadrant_shape_yx=(quadrant_shape[0], quadrant_shape[1]),
    )
