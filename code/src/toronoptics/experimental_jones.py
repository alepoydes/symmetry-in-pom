from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


JONES_COMPONENTS = ("Jxx", "Jxy", "Jyx", "Jyy")
PIXEL_SIZE_KEY = "pixel_size_um"


@dataclass(frozen=True)
class JonesArchive:
    path: Path
    components: dict[str, np.ndarray]
    jones: np.ndarray
    pixel_size_um: float


def load_jones_archive(
    path: Path,
    *,
    dtype: np.dtype | type | None = np.complex64,
    pixel_size_um: float | None = None,
) -> JonesArchive:
    """Load imported Jones components in the project convention.

    Imported archives store each component as ``component[x, y]``. The returned
    matrix field has shape ``(x, y, 2, 2)`` with the last axes ordered as
    ``(output_component, input_component)``. The pixel size must be stored in
    the archive as scalar ``pixel_size_um`` or supplied explicitly.
    """
    path = path.expanduser().resolve()
    with np.load(path) as data:
        missing = [name for name in JONES_COMPONENTS if name not in data.files]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"{path} does not contain required Jones components: {names}")
        components = {name: np.asarray(data[name]) for name in JONES_COMPONENTS}
        archive_pixel_size_um = (
            _read_pixel_size_scalar(data[PIXEL_SIZE_KEY], source=path)
            if PIXEL_SIZE_KEY in data.files
            else None
        )

    shapes = {array.shape for array in components.values()}
    if len(shapes) != 1:
        raise ValueError(f"{path} contains Jones components with inconsistent shapes: {sorted(shapes)!r}")

    resolved_pixel_size_um = _resolve_pixel_size_um(
        path,
        archive_pixel_size_um=archive_pixel_size_um,
        override_pixel_size_um=pixel_size_um,
    )
    output_dtype = None if dtype is None else np.dtype(dtype)
    jones = np.empty((*next(iter(shapes)), 2, 2), dtype=output_dtype or np.result_type(*components.values()))
    jones[..., 0, 0] = components["Jxx"]
    jones[..., 0, 1] = components["Jxy"]
    jones[..., 1, 0] = components["Jyx"]
    jones[..., 1, 1] = components["Jyy"]
    return JonesArchive(
        path=path,
        components=components,
        jones=np.ascontiguousarray(jones),
        pixel_size_um=resolved_pixel_size_um,
    )


def infer_wavelength_um(path: Path, default: float = 0.633) -> float:
    match = re.search(r"(?P<wavelength_nm>\d+(?:\.\d+)?)nm", path.stem)
    if match is None:
        return float(default)
    return float(match.group("wavelength_nm")) / 1000.0


def read_pixel_size_um(directory: Path, default: float | None = 1.0) -> float | None:
    path = directory.expanduser().resolve() / "Pixel_size.txt"
    if not path.exists():
        return None if default is None else float(default)
    text = path.read_text(encoding="utf-8")
    number = r"(?P<pixel_size>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    match = re.search(
        rf"pixel\s*size\s*(?:=|:)\s*{number}(?:\(\d+\))?\s*(?:u|µ|μ)?m?\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.fullmatch(
            rf"\s*{number}(?:\(\d+\))?\s*(?:(?:u|µ|μ)m)?\s*",
            text,
            flags=re.IGNORECASE,
        )
    if match is None:
        return None if default is None else float(default)
    return float(match.group("pixel_size"))


def _read_pixel_size_scalar(value: np.ndarray, *, source: Path) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{source} contains non-scalar {PIXEL_SIZE_KEY!r}: shape {array.shape!r}")
    pixel_size_um = float(array.item())
    _validate_pixel_size_um(pixel_size_um, source=source)
    return pixel_size_um


def _resolve_pixel_size_um(
    path: Path,
    *,
    archive_pixel_size_um: float | None,
    override_pixel_size_um: float | None,
) -> float:
    pixel_size_um = override_pixel_size_um if override_pixel_size_um is not None else archive_pixel_size_um
    if pixel_size_um is None:
        raise ValueError(
            f"{path} does not contain required scalar {PIXEL_SIZE_KEY!r}; "
            "pass pixel_size_um explicitly when loading archives without sampling metadata."
        )
    pixel_size_um = float(pixel_size_um)
    _validate_pixel_size_um(pixel_size_um, source=path)
    return pixel_size_um


def _validate_pixel_size_um(value: float, *, source: Path) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{source} has invalid pixel size {value!r}; expected a positive finite value")


def jones_characteristics(jones: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(jones)
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected Jones matrices with shape (x, y, 2, 2), got {matrix.shape!r}")

    jxx = matrix[..., 0, 0]
    jxy = matrix[..., 0, 1]
    jyx = matrix[..., 1, 0]
    jyy = matrix[..., 1, 1]
    ix = np.abs(jxx) ** 2 + np.abs(jyx) ** 2
    iy = np.abs(jxy) ** 2 + np.abs(jyy) ** 2
    det = jxx * jyy - jxy * jyx
    frobenius2 = ix + iy
    det2 = np.abs(det) ** 2
    discriminant = np.sqrt(np.maximum(frobenius2 * frobenius2 - 4.0 * det2, 0.0))
    smax2 = 0.5 * (frobenius2 + discriminant)
    smin2 = 0.5 * (frobenius2 - discriminant)
    eps = np.finfo(np.float32).eps
    condition = np.sqrt(smax2 / np.maximum(smin2, eps))

    return {
        "|Jxx|": np.abs(jxx),
        "|Jxy|": np.abs(jxy),
        "|Jyx|": np.abs(jyx),
        "|Jyy|": np.abs(jyy),
        "I(x input)": ix,
        "I(y input)": iy,
        "cross x->y": np.abs(jyx) ** 2 / np.maximum(ix, eps),
        "cross y->x": np.abs(jxy) ** 2 / np.maximum(iy, eps),
        "|det J|": np.abs(det),
        "arg det J": np.angle(det),
        "condition": condition,
        "global amplitude": np.sqrt(0.5 * frobenius2),
    }
