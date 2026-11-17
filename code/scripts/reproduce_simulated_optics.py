#!/usr/bin/env python3
"""Compute every publication Jones field from the supplied directors.

The primary toron is evaluated at all three experimental wavelength labels
with both the local spectral-Jones approximation and the nonlocal
marching-wave solver.  The inclined-anchoring positive control is evaluated at
the primary 520-nm wavelength.  This driver also owns the 520-nm phase-step
check and the four synthetic fields that separate layer helicity from
transverse diffraction.  Optical constants and propagation tolerances are
fixed here so a reproduction run cannot silently drift from the values used in
the article.

Director relaxation is not repeated: both inputs are supplied final states,
and their embedded manifests provide the grid, cell, material, and boundary
conditions used by the optical solvers.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np

from toronoptics.article_data import (
    ARTICLE_DIRECTOR_AXIS_CENTER_PX,
    ARTICLE_WAVELENGTHS_NM,
    TILTED_CONTROL_WAVELENGTH_NM,
    TILTED_DIRECTOR_STRUCTURAL_CENTER_PX,
    map_center_between_uniform_grids,
    model_director_path,
    optical_parameters,
    simulated_jones_path,
    tilted_director_path,
    tilted_simulated_jones_path,
)
from toronoptics.cli import add_force_argument
from toronoptics.reproduction import ReproductionStep, run_steps
from toronoptics.simulate_jones import (
    DEFAULT_SUBSTRATE_N,
    load_director_archive,
    refresh_pixel_size_sidecar_from_archive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = ("uv", "run", "--extra", "simulation", "python")
PYTHON_MODULE = (*PYTHON, "-m", "toronoptics.simulate_jones")
HELICITY_CONTROL_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "constant_helicity_diffraction_control_520_soft"
)
PRODUCTION_SUBDIV = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _director_spec(
    resolved_source: str,
    source_sha256: str,
) -> tuple[tuple[int, ...], float, dict[str, object] | None]:
    """Read immutable source properties, caching them by content digest."""

    del source_sha256  # The digest is deliberately part of the cache key.
    archive = load_director_archive(Path(resolved_source))
    if archive.thickness_um is None:
        raise ValueError(
            f"Director archive has no physical thickness: {resolved_source}"
        )
    return (
        tuple(int(value) for value in archive.director.shape),
        float(archive.thickness_um),
        archive.source_manifest,
    )


def _manifest_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _manifest_value_matches(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, float):
        if isinstance(actual, bool):
            return False
        try:
            return bool(
                np.isclose(
                    float(actual),
                    expected,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _simulation_output_is_current(
    path: Path,
    *,
    source: Path,
    wavelength_nm: int,
    model: str,
    phase_step: float | None,
    source_center_px: tuple[float, float] | None = None,
    center_method: str = "fixed_article_director_reference_axis",
) -> bool:
    """Validate complete version-2 provenance without invoking a solver."""

    optical = optical_parameters(wavelength_nm)
    try:
        source_digest = _sha256(source)
        with np.load(path, allow_pickle=False) as archive:
            raw = np.asarray(archive["simulation_manifest"]).item()
            component_shapes = {
                tuple(int(value) for value in archive[key].shape)
                for key in ("Jxx", "Jxy", "Jyx", "Jyy")
            }
            component_dtypes = {
                np.asarray(archive[key]).dtype
                for key in ("Jxx", "Jxy", "Jyx", "Jyy")
            }
            archive_pixel_size = np.asarray(
                archive["pixel_size_um"], dtype=np.float64
            )
            archive_center = np.asarray(
                archive["array_center_px"], dtype=np.float64
            )
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        manifest = json.loads(raw)
        if (
            manifest.get("version") != 2
            or manifest.get("source_sha256") != source_digest
        ):
            return False

        source_shape, thickness_um, source_snapshot_manifest = _director_spec(
            str(source.resolve()),
            source_digest,
        )
        if len(source_shape) != 4 or source_shape[-1] != 3:
            return False
        source_transverse_shape = source_shape[:2]
        expected_jones_plane = tuple(
            PRODUCTION_SUBDIV * value for value in source_transverse_shape
        )
        expected_jones_shape = (*expected_jones_plane, 2, 2)
        expected_pixel_size = (
            thickness_um / (source_shape[2] - 1) / PRODUCTION_SUBDIV
        )
        resolved_source_center = (
            ARTICLE_DIRECTOR_AXIS_CENTER_PX
            if source_center_px is None
            else source_center_px
        )
        target_sampling = (
            "source-nodes" if model == "spectral-jones" else "cell-centers"
        )
        expected_center = map_center_between_uniform_grids(
            resolved_source_center,
            source_shape=source_transverse_shape,
            target_shape=expected_jones_plane,
            target_sampling=target_sampling,
        )

        parameters = manifest["optics"]
        cuda_model = model == "marching-wave"
        expected_parameters = {
            "model": model,
            "wavelength_um": optical.wavelength_um,
            "no": optical.ordinary_index,
            "ne": optical.extraordinary_index,
            "thickness_um": thickness_um,
            "subdiv": PRODUCTION_SUBDIV,
            "phase_step": phase_step,
            "lambda_tile": None,
            "taichi_arch": "auto",
            "camera_aperture_enabled": False if cuda_model else None,
            "camera_na": 1.0 if cuda_model else None,
            "bidirectional_output": None,
            "bidirectional_iterations": None,
            "bidirectional_chunks": None,
            "bidirectional_chunks_auto": None,
            "substrate_thickness_um": None,
            "substrate_n": DEFAULT_SUBSTRATE_N if cuda_model else None,
            "external_n": None,
            "phase_reference_mode": "reference",
            "precision": "single",
        }
        parameters_match = (
            set(parameters) == set(expected_parameters)
            and all(
                _manifest_value_matches(
                    parameters[key],
                    expected,
                )
                for key, expected in expected_parameters.items()
            )
        )
        center_metadata = manifest["center_method"]
        manifest_source_center = np.asarray(
            manifest["source_center_px"], dtype=np.float64
        )
        manifest_array_center = np.asarray(
            manifest["array_center_px"], dtype=np.float64
        )
        expected_mapping = (
            "c_out = s*c_in"
            if model == "spectral-jones"
            else "c_out = s*c_in - 1/2"
        )
        software = manifest["software"]
        return bool(
            _manifest_path(manifest["source"]).resolve() == source.resolve()
            and _manifest_path(manifest["destination"]).resolve()
            == path.resolve()
            and manifest.get("source_kind") == "lcsim director-field npz"
            and manifest.get("axis_order")
            == {
                "director": "director[x, y, z, component]",
                "jones_matrix": (
                    "J[x, y, output_component, input_component]"
                ),
            }
            and manifest.get("source_snapshot_manifest")
            == source_snapshot_manifest
            and tuple(int(value) for value in manifest["source_director_shape"])
            == source_shape
            and tuple(int(value) for value in manifest["jones_shape"])
            == expected_jones_shape
            and component_shapes == {expected_jones_plane}
            and component_dtypes == {np.dtype(np.complex64)}
            and archive_pixel_size.shape == ()
            and np.isclose(
                float(archive_pixel_size),
                expected_pixel_size,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.isclose(
                float(manifest["pixel_size_um"]),
                expected_pixel_size,
                rtol=0.0,
                atol=1.0e-12,
            )
            and archive_center.shape == (2,)
            and np.allclose(
                archive_center,
                expected_center,
                rtol=0.0,
                atol=1.0e-9,
            )
            and manifest_source_center.shape == (2,)
            and np.allclose(
                manifest_source_center,
                resolved_source_center,
                rtol=0.0,
                atol=1.0e-9,
            )
            and manifest_array_center.shape == (2,)
            and np.allclose(
                manifest_array_center,
                expected_center,
                rtol=0.0,
                atol=1.0e-9,
            )
            and center_metadata.get("name") == center_method
            and center_metadata.get("target_sampling") == target_sampling
            and center_metadata.get("mapping") == expected_mapping
            and np.isclose(
                float(center_metadata["transverse_scale"]),
                float(PRODUCTION_SUBDIV),
                rtol=0.0,
                atol=1.0e-12,
            )
            and software.get("topovec") == version("topovec")
            and software.get("numpy") == np.__version__
            and parameters_match
        )
    except (
        AttributeError,
        OSError,
        KeyError,
        PackageNotFoundError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ):
        return False


def _simulation_step(
    *,
    title: str,
    explanation: str,
    source: Path,
    destination: Path,
    wavelength_nm: int,
    model: str,
    marching_phase_step: float | None = None,
    source_center_px: tuple[float, float] | None = None,
    center_method: str = "fixed_article_director_reference_axis",
) -> ReproductionStep:
    optical = optical_parameters(wavelength_nm)
    resolved_source_center = (
        ARTICLE_DIRECTOR_AXIS_CENTER_PX
        if source_center_px is None
        else source_center_px
    )
    command = (
        *PYTHON_MODULE,
        str(source.relative_to(PROJECT_ROOT)),
        str(destination.relative_to(PROJECT_ROOT)),
        "--model",
        model,
        "--wavelength-um",
        str(optical.wavelength_um),
        "--no",
        str(optical.ordinary_index),
        "--ne",
        str(optical.extraordinary_index),
        "--subdiv",
        str(PRODUCTION_SUBDIV),
        "--source-center-px",
        *(str(value) for value in resolved_source_center),
        "--center-method",
        center_method,
    )
    phase_step = None
    if model == "marching-wave":
        # The red calculation needs the smaller phase step used in the
        # convergence study; blue and green are converged at 0.025 rad.
        phase_step = (
            float(marching_phase_step)
            if marching_phase_step is not None
            else (0.0125 if wavelength_nm == 660 else 0.025)
        )
        command = (*command, "--phase-step", str(phase_step))
    return ReproductionStep(
        title,
        explanation,
        command,
        outputs=(destination,),
        output_is_current=lambda path: _simulation_output_is_current(
            path,
            source=source,
            wavelength_nm=wavelength_nm,
            model=model,
            phase_step=phase_step,
            source_center_px=resolved_source_center,
            center_method=center_method,
        ),
    )


def _helicity_control_outputs() -> tuple[Path, ...]:
    """All fields and the summary produced by the 520-nm factorial control."""
    names = (
        "tau_m090p0_subdiv2_phase_none_radial-v3_spectral-jones.npz",
        "tau_m090p0_subdiv2_phase_0p025_radial-v3_marching-wave.npz",
        "tau_measured_subdiv2_phase_none_radial-v3_spectral-jones.npz",
        "tau_measured_subdiv2_phase_0p025_radial-v3_marching-wave.npz",
        "constant_helicity_diffraction_control.json",
    )
    return tuple(HELICITY_CONTROL_DIRECTORY / name for name in names)


def _helicity_control_output_is_current(path: Path) -> bool:
    source = model_director_path(PROJECT_ROOT)
    try:
        source_digest = _sha256(source)
        if path.suffix == ".json":
            report = json.loads(path.read_text(encoding="utf-8"))
            return bool(
                report.get("schema_version")
                == "constant-helicity-diffraction-control-v7"
                and report.get("source_sha256") == source_digest
            )
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(
                str(np.asarray(archive["metadata_json"]).item())
            )
        expected_model = (
            "marching-wave"
            if path.name.endswith("_marching-wave.npz")
            else "spectral-jones"
        )
        expected_profile = (
            "measured-layer" if "tau_measured" in path.name else "constant"
        )
        expected_phase_step = (
            0.025 if expected_model == "marching-wave" else None
        )
        return bool(
            metadata.get("source_sha256") == source_digest
            and metadata.get("model") == expected_model
            and int(metadata["subdiv"]) == PRODUCTION_SUBDIV
            and metadata.get("helicity_profile") == expected_profile
            and metadata.get("phase_step") == expected_phase_step
            and int(metadata["synthetic_side"]) == 128
            and np.isclose(
                float(metadata["radial_bin_width_px"]),
                0.5,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.isclose(
                float(metadata["wavelength_um"]),
                0.520,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.isclose(
                float(metadata["no"]),
                1.5440722923,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.isclose(
                float(metadata["ne"]),
                1.7335786888,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _steps() -> tuple[ReproductionStep, ...]:
    source = model_director_path(PROJECT_ROOT)
    steps: list[ReproductionStep] = []
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        steps.append(
            _simulation_step(
                title=f"Local Jones field at {wavelength_nm} nm",
                explanation=(
                    "Integrate the pointwise transverse Jones response through "
                    "the relaxed finite-anchoring toron."
                ),
                source=source,
                destination=simulated_jones_path(
                    PROJECT_ROOT, wavelength_nm, "spectral-jones"
                ),
                wavelength_nm=wavelength_nm,
                model="spectral-jones",
            )
        )
        steps.append(
            _simulation_step(
                title=f"Marching-wave field at {wavelength_nm} nm",
                explanation=(
                    "Propagate the vector field through the same director with "
                    "transverse diffraction retained and a converged phase step."
                ),
                source=source,
                destination=simulated_jones_path(
                    PROJECT_ROOT, wavelength_nm, "marching-wave"
                ),
                wavelength_nm=wavelength_nm,
                model="marching-wave",
            )
        )

    steps.append(
        _simulation_step(
            title="520-nm marching-wave phase-step check",
            explanation=(
                "Repeat the primary field at half the production phase "
                "increment for the numerical-sensitivity comparison."
            ),
            source=source,
            destination=(
                PROJECT_ROOT
                / "data"
                / "simulated"
                / "toron_27um_5cb_soft"
                / "520nm_marching-wave_phase-step-0p0125.npz"
            ),
            wavelength_nm=520,
            model="marching-wave",
            marching_phase_step=0.0125,
        )
    )

    tilted_source = tilted_director_path(PROJECT_ROOT)
    for model in ("spectral-jones", "marching-wave"):
        steps.append(
            _simulation_step(
                title=(
                    f"Inclined-control {model} field at "
                    f"{TILTED_CONTROL_WAVELENGTH_NM} nm"
                ),
                explanation=(
                    "Repeat the primary-wavelength calculation for the independently "
                    "supplied director with both SoftAxis directions inclined by 0.03."
                ),
                source=tilted_source,
                destination=tilted_simulated_jones_path(
                    PROJECT_ROOT,
                    TILTED_CONTROL_WAVELENGTH_NM,
                    model,
                ),
                wavelength_nm=TILTED_CONTROL_WAVELENGTH_NM,
                model=model,
                source_center_px=TILTED_DIRECTOR_STRUCTURAL_CENTER_PX,
                center_method=(
                    "volume_centroid_of_transverse_director_weight"
                ),
            )
        )

    steps.append(
        ReproductionStep(
            "Synthetic 520-nm helicity/diffraction controls",
            "Generate the constant-helicity and measured-layer-helicity "
            "Jones fields with local and marching-wave optics. Existing "
            "compatible fields are reused and only missing fields are solved.",
            (
                *PYTHON,
                "code/scripts/constant_helicity_diffraction_control.py",
                str(source.relative_to(PROJECT_ROOT)),
                str(HELICITY_CONTROL_DIRECTORY.relative_to(PROJECT_ROOT)),
                "--side",
                "128",
                "--radial-bin-width-px",
                "0.5",
                "--subdiv",
                "2",
                "--model",
                "spectral-jones",
                "marching-wave",
                "--analysis-radius-um",
                "28",
                "--background-inner-um",
                "31",
                "--source-center-x",
                "100.987577711",
                "--source-center-y",
                "100.987570909",
                "--tau-deg",
                "-90",
                "--include-measured-helicity",
                "--wavelength-um",
                "0.520",
                "--no",
                "1.5440722923",
                "--ne",
                "1.7335786888",
                "--phase-step",
                "0.025",
            ),
            force_on_partial=False,
            outputs=_helicity_control_outputs(),
            output_is_current=_helicity_control_output_is_current,
        )
    )
    return tuple(steps)


def _refresh_declared_pixel_size_sidecars() -> tuple[Path, ...]:
    """Synchronize sidecars for current production Jones outputs only."""

    refreshed = []
    for step in _steps():
        if step.command[: len(PYTHON_MODULE)] != PYTHON_MODULE:
            continue
        archive = step.outputs[0]
        if archive.exists():
            refreshed.append(
                refresh_pixel_size_sidecar_from_archive(archive)
            )
    return tuple(dict.fromkeys(refreshed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    parser.add_argument("--dry-run", action="store_true", help="Check existing outputs without running calculations")
    parser.add_argument(
        "--refresh-pixel-size-sidecars-only",
        action="store_true",
        help=(
            "Refresh Pixel_size.txt from existing declared Jones archives "
            "without invoking any optical solver."
        ),
    )
    args = parser.parse_args(argv)
    if args.force and args.refresh_pixel_size_sidecars_only:
        parser.error(
            "--force and --refresh-pixel-size-sidecars-only are mutually "
            "exclusive"
        )
    if args.refresh_pixel_size_sidecars_only:
        refreshed = _refresh_declared_pixel_size_sidecars()
        print(
            f"Refreshed {len(refreshed)} production pixel-size sidecar(s)."
        )
        return 0
    run_steps(_steps(), project_root=PROJECT_ROOT, force=args.force, dry_run=args.dry_run)
    if not args.dry_run:
        _refresh_declared_pixel_size_sidecars()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
