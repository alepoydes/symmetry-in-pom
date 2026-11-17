#!/usr/bin/env python3
"""Validate source-data schemas and their agreement with the TOML manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
import zipfile

import numpy as np
from PIL import Image

from toronoptics.source_config import load_direct_pom_config


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data_src"


def _npy_header(archive: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(archive) as container:
        member = f"{key}.npy"
        if member not in container.namelist():
            raise ValueError(f"{archive} does not contain {key!r}")
        with container.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"{archive}:{key} uses unsupported NPY {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _validate_jones_archives(experiment: dict) -> None:
    table = experiment["holography"]
    shape = tuple(int(value) for value in table["component_shape_xy"])
    dtype = np.dtype(table["jones_component_dtype"])
    keys = tuple(str(value) for value in table["jones_component_names"])
    pixel_size = float(table["component_pixel_size_um"])
    count = int(table["acquisitions_per_wavelength"])
    wavelengths = tuple(int(value) for value in table["wavelengths"]["directory_labels_nm"])
    for wavelength in wavelengths:
        for acquisition in range(1, count + 1):
            path = SOURCE_ROOT / "holography" / str(wavelength) / f"{acquisition}.npz"
            for key in keys:
                actual_shape, actual_dtype = _npy_header(path, key)
                if actual_shape != shape or actual_dtype != dtype:
                    raise ValueError(
                        f"{path}:{key} is {actual_shape}/{actual_dtype}, "
                        f"expected {shape}/{dtype}"
                    )
            with np.load(path, allow_pickle=False) as archive:
                stored_pixel_size = float(np.asarray(archive["pixel_size_um"]).item())
            if not np.isclose(stored_pixel_size, pixel_size, rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    f"{path} pixel size {stored_pixel_size} != {pixel_size}"
                )


def _validate_directors(document: dict) -> None:
    if document.get("schema_version") != "symmetry-in-pom.director-source.v1":
        raise ValueError("unsupported directors.toml schema")
    common = document["common"]
    shape_xyz = tuple(int(value) for value in common["grid_shape_xyz"])
    expected_path_shape = (1, *shape_xyz, 1, 3)
    expected_node_shape = [shape_xyz[2], shape_xyz[0], shape_xyz[1]]
    cases = (
        ("primary", document["primary"]),
        ("inclined_positive_control", document["inclined_positive_control"]),
    )
    for case_name, case in cases:
        path = SOURCE_ROOT / str(case["file"])
        path_shape, path_dtype = _npy_header(path, "PATH")
        if path_shape != expected_path_shape or path_dtype != np.dtype("float32"):
            raise ValueError(
                f"{path}:PATH is {path_shape}/{path_dtype}, "
                f"expected {expected_path_shape}/float32"
            )
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(np.asarray(archive["snapshot_manifest"]).item()))
        problem = manifest["problem"]
        geometry = problem["geometry"]
        if geometry["node_shape"] != expected_node_shape:
            raise ValueError(f"{path} node_shape disagrees with directors.toml")
        if not np.isclose(
            float(geometry["thickness_um"]),
            float(common["cell_thickness_um"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"{path} thickness disagrees with directors.toml")
        if manifest["run"]["integrator"] != common["integrator"]:
            raise ValueError(f"{path} integrator disagrees with directors.toml")
        boundaries = {
            item["id"]: item["constraints"][0] for item in geometry["boundaries"]
        }
        for boundary_id in ("x_min", "x_max", "y_min", "y_max"):
            constraint = boundaries[boundary_id]
            if (
                constraint["kind"] != common["lateral_boundary"]
                or constraint["value"] != common["lateral_direction"]
            ):
                raise ValueError(f"{path} {boundary_id} disagrees with directors.toml")
        for boundary_id, direction_key in (
            ("bottom_substrate", "bottom_substrate_direction"),
            ("top_substrate", "top_substrate_direction"),
        ):
            constraint = boundaries[boundary_id]
            if constraint["bc"] != common["substrate_boundary"]:
                raise ValueError(f"{path} {boundary_id} type disagrees with directors.toml")
            if not np.isclose(
                float(constraint["strength"]),
                float(common["substrate_strength"]),
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError(f"{path} {boundary_id} strength disagrees with directors.toml")
            if constraint["direction"] != case[direction_key]:
                raise ValueError(f"{path} {boundary_id} direction disagrees with directors.toml")
        blocks = {item["kind"]: item for item in problem["blocks"]}
        expected_scalars = {
            ("orientation_dynamics", "gamma_mPa_s"): common[
                "rotational_viscosity_mPa_s"
            ],
            ("elastic", "K1_pN"): common["K1_pN"],
            ("elastic", "K2_pN"): common["K2_pN"],
            ("elastic", "K3_pN"): common["K3_pN"],
            ("constant_chirality", "q_um_inv"): common["q0_um_inv"],
        }
        for (block_name, key), expected in expected_scalars.items():
            if not np.isclose(
                float(blocks[block_name][key]),
                float(expected),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"{path} {block_name}.{key} disagrees with directors.toml"
                )
        print(f"[source-data] {case_name} director manifest: OK")


def _validate_direct_pom(experiment: dict) -> None:
    config = load_direct_pom_config(SOURCE_ROOT / "experiment.toml")
    table = experiment["direct_pom"]
    expected_mode = str(table["encoded_mode"])
    if expected_mode != "8-bit grayscale PNG":
        raise ValueError(f"unsupported direct-POM encoding: {expected_mode}")
    for exposure in config.exposure_times_ms:
        directory = SOURCE_ROOT / "direct_pom" / "520nm" / f"Exposure_{exposure}ms"
        names = [
            *(f"{frame:04d}.png" for frame in config.selected_frame_numbers),
            str(table["environment_file"]),
            *(f"Ground ({index}).png" for index in range(1, 10)),
        ]
        for name in names:
            path = directory / name
            with Image.open(path) as image:
                shape = (image.height, image.width)
                if shape != config.encoded_shape_yx or image.mode != "L":
                    raise ValueError(
                        f"{path} is {shape}/{image.mode}, expected "
                        f"{config.encoded_shape_yx}/L"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--experimental", action="store_true")
    group.add_argument("--directors", action="store_true")
    args = parser.parse_args()
    if not args.directors:
        with (SOURCE_ROOT / "experiment.toml").open("rb") as stream:
            experiment = tomllib.load(stream)
        load_direct_pom_config(SOURCE_ROOT / "experiment.toml")
        _validate_jones_archives(experiment)
        _validate_direct_pom(experiment)
    if not args.experimental:
        with (SOURCE_ROOT / "directors.toml").open("rb") as stream:
            directors = tomllib.load(stream)
        _validate_directors(directors)
    print("[source-data] TOML, NPZ, and PNG schemas agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
