#!/usr/bin/env python3
"""Compare local-Jones and marching-wave optics at fixed helicity."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np

from toronoptics.article_data import map_center_between_uniform_grids
from toronoptics.director_profiles import (
    layer_helicity as _layer_helicity,
    radial_polar_profile as _radial_polar_profile,
    scaled_helicity_profile as _scaled_helicity_profile,
)
from toronoptics.simulate_jones import load_director_archive, simulate_jones_matrices


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
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


def _write_field(path: Path, jones: np.ndarray, metadata: dict[str, object]) -> None:
    """Atomically write a reusable Jones-field archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=".npz",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        np.savez_compressed(
            temporary_name,
            Jxx=np.asarray(jones[..., 0, 0], dtype=np.complex64),
            Jxy=np.asarray(jones[..., 0, 1], dtype=np.complex64),
            Jyx=np.asarray(jones[..., 1, 0], dtype=np.complex64),
            Jyy=np.asarray(jones[..., 1, 1], dtype=np.complex64),
            pixel_size_um=np.asarray(metadata["pixel_size_um"], dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        Path(temporary_name).replace(path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _fixed_helicity_director(
    radius_px: np.ndarray,
    theta_profile: np.ndarray,
    *,
    side: int,
    tau_rad: float | np.ndarray,
) -> np.ndarray:
    center = 0.5 * float(side)
    x, y = np.indices((side, side), dtype=np.float64)
    radius = np.hypot(x - center, y - center)
    phi = np.arctan2(y - center, x - center)
    theta = np.empty((side, side, theta_profile.shape[1]), dtype=np.float64)
    for z_index in range(theta_profile.shape[1]):
        theta[..., z_index] = np.interp(
            radius,
            radius_px,
            theta_profile[:, z_index],
            left=theta_profile[0, z_index],
            right=theta_profile[-1, z_index],
        )

    tau = np.asarray(tau_rad, dtype=np.float64)
    if tau.ndim == 0:
        tau = np.full(theta_profile.shape[1], float(tau), dtype=np.float64)
    if tau.shape != (theta_profile.shape[1],):
        raise ValueError(
            "tau_rad must be scalar or contain one value per director layer"
        )

    sin_theta = np.sin(theta)
    director = np.empty(theta.shape + (3,), dtype=np.float32)
    director[..., 0] = sin_theta * np.cos(phi[..., None] + tau[None, None, :])
    director[..., 1] = sin_theta * np.sin(phi[..., None] + tau[None, None, :])
    director[..., 2] = np.cos(theta)
    director /= np.linalg.norm(director, axis=-1, keepdims=True)
    return np.ascontiguousarray(director, dtype=np.float32)


def _straight_tube_profile(theta_profile: np.ndarray) -> np.ndarray:
    """Continue the midplane wall profile uniformly through the cell."""
    theta = np.asarray(theta_profile, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] < 1:
        raise ValueError("theta_profile must have shape (radius, z)")
    midplane = theta[:, theta.shape[1] // 2]
    return np.repeat(midplane[:, None], theta.shape[1], axis=1)


def _complex_background(jones: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.mean(jones[mask], axis=0, dtype=np.complex128)


def _crossed_c2_visibility(
    crossed: np.ndarray,
    radius_px: np.ndarray,
    phi: np.ndarray,
    mask: np.ndarray,
) -> float:
    intensity = np.abs(crossed) ** 2
    radial_bin = np.floor(radius_px).astype(np.int32)
    numerator = 0.0
    denominator = 0.0
    for index in range(1, int(radial_bin[mask].max()) + 1):
        ring = mask & (radial_bin == index)
        count = int(np.count_nonzero(ring))
        if count < 8:
            continue
        i0 = float(np.mean(intensity[ring]))
        h2 = np.mean(intensity[ring] * np.exp(-2j * phi[ring]))
        numerator += count * 4.0 * float(np.abs(h2) ** 2)
        denominator += count * i0 * i0
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def _metrics(
    jones: np.ndarray,
    *,
    center_px: tuple[float, float],
    pixel_size_um: float,
    analysis_radius_um: float,
    background_inner_um: float,
) -> dict[str, float]:
    nx, ny = jones.shape[:2]
    center = np.asarray(center_px, dtype=np.float64)
    x, y = np.indices((nx, ny), dtype=np.float64)
    dx = x - center[0]
    dy = y - center[1]
    radius_px = np.hypot(dx, dy)
    radius_um = radius_px * float(pixel_size_um)
    phi = np.arctan2(dy, dx)
    analysis = radius_um <= float(analysis_radius_um)
    background = radius_um >= float(background_inner_um)
    jones_bg = _complex_background(jones, background)
    perturbation = jones - jones_bg

    jxy = perturbation[..., 0, 1]
    jyx = perturbation[..., 1, 0]
    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)
    b = 0.5 * (jyx - jxy)
    total_power = float(np.sum(np.abs(perturbation[analysis]) ** 2))
    b_power = float(2.0 * np.sum(np.abs(b[analysis]) ** 2))
    full_jxx = jones[..., 0, 0]
    full_jxy = jones[..., 0, 1]
    full_jyx = jones[..., 1, 0]
    full_jyy = jones[..., 1, 1]
    full_b = 0.5 * (full_jyx - full_jxy)
    full_c = 0.5 * (
        (full_jxx - full_jyy) * cos2 + (full_jxy + full_jyx) * sin2
    )
    full_d = 0.5 * (
        -(full_jxx - full_jyy) * sin2 + (full_jxy + full_jyx) * cos2
    )
    phase_unlocking_numerator = float(
        np.sum(
            np.real(np.conj(full_b[analysis]) * full_c[analysis]) ** 2
            + np.real(np.conj(full_b[analysis]) * full_d[analysis]) ** 2
        )
    )
    phase_unlocking_denominator = float(
        np.sum(
            np.abs(full_b[analysis]) ** 2
            * (np.abs(full_c[analysis]) ** 2 + np.abs(full_d[analysis]) ** 2)
        )
    )
    pixel_area_um2 = float(pixel_size_um) ** 2

    reflected_y = np.rint(2.0 * center[1] - np.arange(ny)).astype(np.int64) % ny
    reflected = np.take(jones, reflected_y, axis=1)
    mirror_expected = jones.copy()
    mirror_expected[..., 0, 1] *= -1.0
    mirror_expected[..., 1, 0] *= -1.0
    mirror_error = float(np.sum(np.abs((reflected - mirror_expected)[analysis]) ** 2))

    return {
        "b_power_fraction": b_power / max(total_power, np.finfo(float).tiny),
        "b_relative_l2_amplitude": np.sqrt(
            b_power / max(total_power, np.finfo(float).tiny)
        ),
        "phase_unlocking": (
            None
            if b_power / max(total_power, np.finfo(float).tiny) < 1.0e-10
            else np.sqrt(
                phase_unlocking_numerator
                / max(phase_unlocking_denominator, np.finfo(float).tiny)
            )
        ),
        "mirror_relative_l2_residual": np.sqrt(
            mirror_error / max(total_power, np.finfo(float).tiny)
        ),
        "crossed_c2_visibility": _crossed_c2_visibility(
            jones[..., 1, 0], radius_px, phi, analysis
        ),
        "total_perturbation_power": total_power,
        "total_perturbation_integral_um2": total_power * pixel_area_um2,
        "b_integral_um2": b_power * pixel_area_um2,
    }


def _slug(tau_deg: float) -> str:
    sign = "p" if tau_deg >= 0.0 else "m"
    return f"tau_{sign}{abs(float(tau_deg)):05.1f}".replace(".", "p")


def _phase_slug(phase_step: float) -> str:
    return f"phase_{float(phase_step):0.3f}".replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source toron director NPZ")
    parser.add_argument("output_dir", type=Path, help="Control output directory")
    parser.add_argument("--source-center-x", type=float, default=100.99717149433567)
    parser.add_argument("--source-center-y", type=float, default=100.99714732612402)
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--radial-bin-width-px", type=float, default=0.5)
    parser.add_argument("--tau-deg", type=float, nargs="+", default=(0.0, -45.0, -90.0))
    parser.add_argument("--subdiv", type=int, nargs="+", default=(1, 2))
    parser.add_argument(
        "--model",
        choices=("spectral-jones", "marching-wave"),
        nargs="+",
        default=("spectral-jones", "marching-wave"),
    )
    parser.add_argument("--wavelength-um", type=float, default=0.662)
    parser.add_argument("--no", type=float, default=1.52993)
    parser.add_argument("--ne", type=float, default=1.70178)
    parser.add_argument("--phase-step", type=float, nargs="+", default=(0.1,))
    parser.add_argument("--analysis-radius-um", type=float, default=28.0)
    parser.add_argument("--background-inner-um", type=float, default=31.0)
    parser.add_argument(
        "--include-measured-helicity",
        action="store_true",
        help=(
            "also use the layer helicity extracted from the source director, "
            "on the same radially averaged polar profile"
        ),
    )
    parser.add_argument(
        "--helicity-scale",
        type=float,
        nargs="+",
        default=(),
        metavar="S",
        help=(
            "add tau_s(z)=tau_mid+S[tau_measured(z)-tau_mid] controls, "
            "with 0 <= S <= 1"
        ),
    )
    parser.add_argument(
        "--include-straight-tube",
        action="store_true",
        help=(
            "add an ideal straight tube by continuing the midplane radial "
            "wall profile uniformly along z"
        ),
    )
    parser.add_argument(
        "--axisymmetric-second-input",
        action="store_true",
        help=(
            "propagate x input only and reconstruct the y-input column from "
            "the exact rotational covariance of the synthetic director"
        ),
    )
    parser.add_argument(
        "--report-name",
        default="constant_helicity_diffraction_control.json",
        help="JSON report filename inside output_dir",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite generated outputs that already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = load_director_archive(args.source)
    source_file = args.source.expanduser().resolve()
    source_path = _portable_path(source_file)
    source_sha256 = _sha256(source_file)
    if source.thickness_um is None:
        raise ValueError("Source manifest does not define cell thickness")
    output_dir = args.output_dir.expanduser().resolve()
    report_path = output_dir / str(args.report_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_destination: dict[str, dict[str, object]] = {}
    if report_path.exists() and not args.force:
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        compatible_schema = previous_report.get("schema_version") in {
            "constant-helicity-diffraction-control-v3",
            "constant-helicity-diffraction-control-v4",
            "constant-helicity-diffraction-control-v5",
            "constant-helicity-diffraction-control-v6",
            "constant-helicity-diffraction-control-v7",
        }
        previous_source_hash = previous_report.get("source_sha256")
        compatible_source = (
            previous_source_hash == source_sha256
            if previous_source_hash is not None
            else Path(str(previous_report.get("source", ""))).name
            == source_file.name
        )
        if not compatible_schema or not compatible_source:
            raise RuntimeError(
                f"Existing report is incompatible with this run: {report_path}. "
                "Delete it and the fields that need replacement, or use --force "
                "for a complete rebuild."
            )

    radius_px, theta_profile, coherence = _radial_polar_profile(
        np.asarray(source.director, dtype=np.float64),
        (float(args.source_center_x), float(args.source_center_y)),
        bin_width_px=float(args.radial_bin_width_px),
    )
    measured_tau = _layer_helicity(
        np.asarray(source.director, dtype=np.float64),
        (float(args.source_center_x), float(args.source_center_y)),
    )
    helicity_cases = [
        {
            "slug": _slug(float(tau_deg)),
            "profile": "constant",
            "tau_deg": float(tau_deg),
            "tau_rad": float(np.deg2rad(tau_deg)),
        }
        for tau_deg in args.tau_deg
    ]
    if args.include_measured_helicity:
        helicity_cases.append(
            {
                "slug": "tau_measured",
                "profile": "measured-layer",
                "tau_deg": None,
                "tau_rad": measured_tau,
                "helicity_scale": 1.0,
            }
        )
    requested_scales = set(float(value) for value in args.helicity_scale)
    if args.include_measured_helicity:
        requested_scales.discard(1.0)
    for scale in sorted(requested_scales):
        helicity_cases.append(
            {
                "slug": f"tau_scale_{scale:0.3f}".replace(".", "p"),
                "profile": "scaled-measured-layer",
                "tau_deg": None,
                "tau_rad": _scaled_helicity_profile(measured_tau, scale),
                "helicity_scale": scale,
            }
        )
    if args.include_straight_tube:
        helicity_cases.append(
            {
                "slug": "straight_tube_tau_mid",
                "profile": "constant",
                "tau_deg": float(np.rad2deg(measured_tau[len(measured_tau) // 2])),
                "tau_rad": float(measured_tau[len(measured_tau) // 2]),
                "director_profile": "straight-tube",
                "geometry_slug": "straight-tube-v1",
                "theta_profile": _straight_tube_profile(theta_profile),
            }
        )

    calculations_per_helicity = sum(
        len(args.phase_step) if model == "marching-wave" else 1
        for model in args.model
    ) * len(args.subdiv)
    calculation_count = len(helicity_cases) * calculations_per_helicity
    calculation_index = 0
    print(
        f"[constant-helicity] planned optical calculations: {calculation_count}",
        flush=True,
    )

    spacing_um = float(source.thickness_um) / (source.director.shape[2] - 1)
    for helicity_case in helicity_cases:
        tau_rad = np.asarray(helicity_case["tau_rad"], dtype=np.float64)
        tau_mid_rad = (
            float(tau_rad.item())
            if tau_rad.ndim == 0
            else float(tau_rad[len(tau_rad) // 2])
        )
        tau_span_rad = 0.0 if tau_rad.ndim == 0 else float(np.ptp(tau_rad))
        director = _fixed_helicity_director(
            radius_px,
            np.asarray(helicity_case.get("theta_profile", theta_profile)),
            side=int(args.side),
            tau_rad=tau_rad,
        )
        for subdiv in args.subdiv:
            for model in args.model:
                phase_steps: tuple[float | None, ...]
                if model == "marching-wave":
                    phase_steps = tuple(float(value) for value in args.phase_step)
                else:
                    phase_steps = (None,)
                for phase_step in phase_steps:
                    calculation_index += 1
                    calculation_label = (
                        f"[constant-helicity {calculation_index}/"
                        f"{calculation_count}]"
                    )
                    phase_slug = (
                        "phase_none" if phase_step is None else _phase_slug(phase_step)
                    )
                    destination = output_dir / (
                        f"{helicity_case['slug']}_subdiv{subdiv}_{phase_slug}_"
                        f"{helicity_case.get('geometry_slug', 'radial-v3')}_{model}.npz"
                    )
                    pixel_size_um = spacing_um / int(subdiv)
                    output_side = int(args.side) * int(subdiv)
                    synthetic_center = 0.5 * float(args.side)
                    analysis_center = map_center_between_uniform_grids(
                        (synthetic_center, synthetic_center),
                        source_shape=(int(args.side), int(args.side)),
                        target_shape=(output_side, output_side),
                        target_sampling=(
                            "source-nodes"
                            if model == "spectral-jones"
                            else "cell-centers"
                        ),
                    )
                    metadata = {
                        "source": source_path,
                        "source_sha256": source_sha256,
                        "source_center_px": [
                            float(args.source_center_x),
                            float(args.source_center_y),
                        ],
                        "synthetic_side": int(args.side),
                        "radial_bin_width_px": float(args.radial_bin_width_px),
                        "topovec_version": version("topovec"),
                        "construction": (
                            f"{helicity_case.get('director_profile', 'toron-wall')} "
                            "radial polar tilt with "
                            + (
                                "constant helicity"
                                if helicity_case["profile"] == "constant"
                                else "measured or scaled layer helicity"
                            )
                        ),
                        "director_profile": helicity_case.get(
                            "director_profile", "toron-wall"
                        ),
                        "helicity_profile": helicity_case["profile"],
                        "helicity_scale": helicity_case.get("helicity_scale"),
                        "tau_deg": helicity_case["tau_deg"],
                        "tau_mid_deg": float(np.rad2deg(tau_mid_rad)),
                        "tau_span_deg": float(np.rad2deg(tau_span_rad)),
                        "model": model,
                        "subdiv": int(subdiv),
                        "wavelength_um": float(args.wavelength_um),
                        "no": float(args.no),
                        "ne": float(args.ne),
                        "thickness_um": float(source.thickness_um),
                        "pixel_size_um": pixel_size_um,
                        "analysis_center_px": [
                            float(analysis_center[0]),
                            float(analysis_center[1]),
                        ],
                        "analysis_center_mapping": (
                            "c_out=s*c_in"
                            if model == "spectral-jones"
                            else "c_out=s*c_in-1/2"
                        ),
                        "phase_step": phase_step,
                        "analysis_radius_um": float(args.analysis_radius_um),
                        "background_inner_um": float(args.background_inner_um),
                        "axisymmetric_second_input": bool(
                            args.axisymmetric_second_input
                        ),
                    }
                    result_metadata = metadata
                    if destination.exists() and not args.force:
                        print(
                            f"{calculation_label} reuse: {destination}",
                            flush=True,
                        )
                        with np.load(destination, allow_pickle=False) as stored:
                            stored_metadata = json.loads(
                                str(np.asarray(stored["metadata_json"]).item())
                            )
                            required_metadata = {
                                "source_sha256": metadata["source_sha256"],
                                "source_center_px": metadata["source_center_px"],
                                "synthetic_side": metadata["synthetic_side"],
                                "radial_bin_width_px": metadata[
                                    "radial_bin_width_px"
                                ],
                                "director_profile": metadata["director_profile"],
                                "helicity_profile": metadata["helicity_profile"],
                                "helicity_scale": metadata["helicity_scale"],
                                "tau_deg": metadata["tau_deg"],
                                "tau_mid_deg": metadata["tau_mid_deg"],
                                "tau_span_deg": metadata["tau_span_deg"],
                                "model": metadata["model"],
                                "subdiv": metadata["subdiv"],
                                "wavelength_um": metadata["wavelength_um"],
                                "no": metadata["no"],
                                "ne": metadata["ne"],
                                "thickness_um": metadata["thickness_um"],
                                "pixel_size_um": metadata["pixel_size_um"],
                                "phase_step": metadata["phase_step"],
                                "axisymmetric_second_input": metadata[
                                    "axisymmetric_second_input"
                                ],
                            }
                            legacy_archive = "source_sha256" not in stored_metadata
                            legacy_only = {
                                "source_sha256",
                                "source_center_px",
                                "synthetic_side",
                                "radial_bin_width_px",
                            }
                            mismatched = {
                                key: (stored_metadata.get(key), expected)
                                for key, expected in required_metadata.items()
                                if not (legacy_archive and key in legacy_only)
                                and stored_metadata.get(key) != expected
                            }
                            if legacy_archive and Path(
                                str(stored_metadata.get("source", ""))
                            ).name != source_file.name:
                                mismatched["source"] = (
                                    stored_metadata.get("source"),
                                    source_path,
                                )
                            if mismatched:
                                raise RuntimeError(
                                    f"Stored field metadata does not match {destination}: "
                                    f"{mismatched}. Recompute with --force."
                                )
                            jones = np.empty(
                                stored["Jxx"].shape + (2, 2), dtype=np.complex64
                            )
                            jones[..., 0, 0] = stored["Jxx"]
                            jones[..., 0, 1] = stored["Jxy"]
                            jones[..., 1, 0] = stored["Jyx"]
                            jones[..., 1, 1] = stored["Jyy"]
                        result_metadata = metadata
                        if stored_metadata != metadata:
                            print(
                                f"{calculation_label} migrate cache metadata",
                                flush=True,
                            )
                            _write_field(destination, jones, metadata)
                    else:
                        print(
                            f"{calculation_label} "
                            f"director={metadata['director_profile']}, "
                            f"helicity={helicity_case['profile']}, "
                            f"tau_mid={np.rad2deg(tau_mid_rad):g} deg, "
                            f"model={model}, subdiv={subdiv}, "
                            f"phase_step={phase_step}, "
                            f"shape={director.shape}",
                            flush=True,
                        )
                        jones = simulate_jones_matrices(
                            director,
                            model=model,
                            wavelength_um=float(args.wavelength_um),
                            no=float(args.no),
                            ne=float(args.ne),
                            thickness_um=float(source.thickness_um),
                            subdiv=int(subdiv),
                            phase_step=phase_step,
                            camera_aperture_enabled=False,
                            phase_reference_mode="reference",
                            axisymmetric_second_input=bool(
                                args.axisymmetric_second_input
                            ),
                            progress=True,
                        )
                        _write_field(destination, jones, metadata)
                    metrics = _metrics(
                        jones,
                        center_px=analysis_center,
                        pixel_size_um=pixel_size_um,
                        analysis_radius_um=float(args.analysis_radius_um),
                        background_inner_um=float(args.background_inner_um),
                    )
                    result = {
                        **result_metadata,
                        **metrics,
                        "destination": destination.name,
                    }
                    results_by_destination[destination.name] = result
                    print(
                        f"{calculation_label} "
                        f"b fraction={metrics['b_power_fraction']:.6g}, "
                        f"crossed C2={metrics['crossed_c2_visibility']:.6g}, "
                        f"mirror residual={metrics['mirror_relative_l2_residual']:.6g}",
                        flush=True,
                    )

    report = {
        "schema_version": "constant-helicity-diffraction-control-v7",
        "source": source_path,
        "source_sha256": source_sha256,
        "source_center_px": [float(args.source_center_x), float(args.source_center_y)],
        "synthetic_side": int(args.side),
        "radial_bin_width_px": float(args.radial_bin_width_px),
        "minimum_radial_nematic_coherence": float(np.min(coherence)),
        "measured_layer_helicity_deg": np.rad2deg(measured_tau).tolist(),
        "results": sorted(
            results_by_destination.values(),
            key=lambda item: (
                str(item.get("director_profile", "toron-wall")),
                str(item.get("helicity_profile", "constant")),
                -1.0
                if item.get("helicity_scale") is None
                else float(item["helicity_scale"]),
                float(item.get("tau_mid_deg", item.get("tau_deg", 0.0))),
                str(item["model"]),
                int(item["subdiv"]),
                -1.0 if item["phase_step"] is None else float(item["phase_step"]),
            ),
        ),
    }
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    print(f"[constant-helicity] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
