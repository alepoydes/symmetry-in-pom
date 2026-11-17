#!/usr/bin/env python3
"""Estimate the coherent reconstruction pupil from the new Jones ensembles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "build/.matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toronoptics.article_data import (
    ARTICLE_WAVELENGTHS_NM,
    experimental_average_path,
    optical_parameters,
)
from toronoptics.experimental_jones import load_jones_archive
from toronoptics.cli import add_force_argument, refuse_existing_outputs
from toronoptics.pupil_spectrum import (
    estimate_pupil_cutoff,
    radial_jones_pupil_spectrum,
)


OUTPUT_DIR = PROJECT_ROOT / "data/analysis/averaged_pupil_spectrum"
ARCHIVE_PATH = OUTPUT_DIR / "averaged_pupil_spectrum.npz"
REPORT_PATH = OUTPUT_DIR / "report.json"
FIGURE_PATH = OUTPUT_DIR / "averaged_jones_pupil_spectrum.pdf"
COLORS = {450: "#2676b8", 520: "#299447", 660: "#c83c3c"}
PUPIL_NA_INTERVAL = (0.40, 0.56)
OBJECTIVE_NOMINAL_NA = 0.55


def _relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _analyze_path(path: Path, wavelength_nm: int):
    archive = load_jones_archive(path)
    parameters = optical_parameters(wavelength_nm)
    profile = radial_jones_pupil_spectrum(
        archive.jones,
        pixel_size_um=float(archive.pixel_size_um),
        wavelength_um=parameters.wavelength_um,
    )
    estimate = estimate_pupil_cutoff(
        profile,
        na_interval=PUPIL_NA_INTERVAL,
    )
    return profile, estimate


def analyze() -> dict[str, object]:
    records: list[dict[str, object]] = []
    average_profiles = {}
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        average_path = experimental_average_path(PROJECT_ROOT, wavelength_nm)
        average_profile, average_estimate = _analyze_path(average_path, wavelength_nm)
        average_profiles[wavelength_nm] = average_profile
        records.append(
            _record(wavelength_nm, "average", average_path, average_estimate)
        )
        with np.load(average_path, allow_pickle=False) as data:
            source_paths = [Path(str(value)) for value in np.asarray(data["source_paths"])]
        for index, source_path in enumerate(source_paths, start=1):
            if not source_path.is_absolute():
                source_path = PROJECT_ROOT / source_path
            _profile, estimate = _analyze_path(source_path, wavelength_nm)
            records.append(
                _record(wavelength_nm, f"source_{index}", source_path, estimate)
            )

    summary = {}
    source_na_all = []
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        selected = np.asarray(
            [
                record["effective_na"]
                for record in records
                if record["wavelength_nm"] == wavelength_nm
                and str(record["kind"]).startswith("source_")
            ],
            dtype=np.float64,
        )
        source_na_all.extend(selected.tolist())
        median = float(np.median(selected))
        mad = float(1.4826 * np.median(np.abs(selected - median)))
        average = next(
            record
            for record in records
            if record["wavelength_nm"] == wavelength_nm and record["kind"] == "average"
        )
        summary[str(wavelength_nm)] = {
            "source_median_effective_na": median,
            "source_robust_sd_effective_na": mad,
            "source_min_effective_na": float(np.min(selected)),
            "source_max_effective_na": float(np.max(selected)),
            "average_effective_na": float(average["effective_na"]),
        }
    pooled = np.asarray(source_na_all, dtype=np.float64)
    common_na = float(np.median(pooled))
    common_mad = float(1.4826 * np.median(np.abs(pooled - common_na)))
    result = {
        "schema_version": "averaged-pupil-spectrum-v2",
        "method": (
            "sharpest wavelength-scaled downward edge of the Tukey-apodized "
            "radial power spectrum of J minus its outer-annulus background; "
            "all four complex Jones channels are summed before edge detection"
        ),
        "effective_na_definition": "NA_eff = wavelength_um * cutoff_frequency_um_inv",
        "search_interval_effective_na": list(PUPIL_NA_INTERVAL),
        "objective_nominal_na": OBJECTIVE_NOMINAL_NA,
        "edge_selection_note": (
            "The selected edge has a wavelength-independent NA near 0.5. "
            "The lower-frequency toron ring and the higher, nearly "
            "wavelength-independent digital reconstruction cutoff are excluded."
        ),
        "records": records,
        "per_wavelength": summary,
        "common_effective_na": common_na,
        "common_effective_na_robust_sd": common_mad,
    }
    _save(result, average_profiles)
    _plot(result, average_profiles)
    return result


def _record(wavelength_nm: int, kind: str, path: Path, estimate) -> dict[str, object]:
    return {
        "wavelength_nm": int(wavelength_nm),
        "kind": kind,
        "path": _relative(path),
        "cutoff_frequency_um_inv": float(estimate.frequency_um_inv),
        "effective_na": float(estimate.effective_na),
        "edge_slope_decades_per_um_inv": float(estimate.slope_decades_per_um_inv),
        "edge_half_width_na": float(estimate.half_width_na),
    }


def _save(result: dict[str, object], profiles: dict[int, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = result["records"]
    wavelength_order = np.asarray(ARTICLE_WAVELENGTHS_NM, dtype=np.int32)
    frequency = np.stack(
        [profiles[int(wavelength)].frequency_um_inv for wavelength in wavelength_order]
    )
    power = np.stack([profiles[int(wavelength)].power for wavelength in wavelength_order])
    log_power = np.stack(
        [profiles[int(wavelength)].log_power for wavelength in wavelength_order]
    )
    log_slope = np.stack(
        [profiles[int(wavelength)].log_slope for wavelength in wavelength_order]
    )
    np.savez_compressed(
        ARCHIVE_PATH,
        schema_version=np.asarray(result["schema_version"]),
        wavelength_nm=wavelength_order,
        frequency_um_inv=frequency.astype(np.float32),
        power=power.astype(np.float32),
        log_power=log_power.astype(np.float32),
        log_slope=log_slope.astype(np.float32),
        record_wavelength_nm=np.asarray([record["wavelength_nm"] for record in records]),
        record_kind=np.asarray([record["kind"] for record in records]),
        record_path=np.asarray([record["path"] for record in records]),
        cutoff_frequency_um_inv=np.asarray(
            [record["cutoff_frequency_um_inv"] for record in records], dtype=np.float32
        ),
        effective_na=np.asarray(
            [record["effective_na"] for record in records], dtype=np.float32
        ),
        common_effective_na=np.asarray(result["common_effective_na"]),
        common_effective_na_robust_sd=np.asarray(
            result["common_effective_na_robust_sd"]
        ),
        metadata_json=np.asarray(json.dumps(result, indent=2)),
    )
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _plot(result: dict[str, object], profiles: dict[int, object]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        profile = profiles[wavelength_nm]
        normalized = profile.log_power - np.nanmax(profile.log_power)
        axes[0].plot(
            profile.numerical_aperture,
            normalized,
            color=COLORS[wavelength_nm],
            linewidth=1.2,
            label=f"{wavelength_nm} nm",
        )
        average = next(
            record
            for record in result["records"]
            if record["wavelength_nm"] == wavelength_nm and record["kind"] == "average"
        )
        axes[0].axvline(
            average["effective_na"], color=COLORS[wavelength_nm], linewidth=0.65
        )
        source_values = [
            record["effective_na"]
            for record in result["records"]
            if record["wavelength_nm"] == wavelength_nm
            and str(record["kind"]).startswith("source_")
        ]
        x = np.full(len(source_values), wavelength_nm, dtype=np.float64)
        axes[1].scatter(x, source_values, s=12, color=COLORS[wavelength_nm], alpha=0.65)
        summary = result["per_wavelength"][str(wavelength_nm)]
        axes[1].errorbar(
            wavelength_nm,
            summary["source_median_effective_na"],
            yerr=summary["source_robust_sd_effective_na"],
            fmt="o",
            markersize=4.2,
            color=COLORS[wavelength_nm],
            capsize=2.0,
        )
    axes[0].set_xlim(0.04, 0.62)
    axes[0].set_ylim(-8.2, 0.2)
    axes[0].set_xlabel(r"spatial-frequency coordinate $\lambda f_\perp$")
    axes[0].set_ylabel("radial spectral power (decades)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(
        result["common_effective_na"], color="0.25", linewidth=0.8, linestyle="--"
    )
    axes[1].axhline(
        result["objective_nominal_na"], color="0.60", linewidth=0.65, linestyle=":"
    )
    axes[1].set_xticks(ARTICLE_WAVELENGTHS_NM)
    axes[1].set_xlabel(r"wavelength (nm)")
    axes[1].set_ylabel(r"effective coherent NA")
    axes[1].set_ylim(0.46, 0.565)
    for panel, axis in zip("ab", axes, strict=True):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.10,
            -0.17,
            f"({panel})",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            clip_on=False,
        )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.98, wspace=0.28)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    args = parser.parse_args(argv)
    refuse_existing_outputs(
        (ARCHIVE_PATH, REPORT_PATH, FIGURE_PATH),
        force=args.force,
        project_root=PROJECT_ROOT,
    )
    result = analyze()
    print(
        f"common effective NA = {result['common_effective_na']:.4f} "
        f"+/- {result['common_effective_na_robust_sd']:.4f} (robust SD)"
    )
    for wavelength_nm in ARTICLE_WAVELENGTHS_NM:
        summary = result["per_wavelength"][str(wavelength_nm)]
        print(
            f"{wavelength_nm} nm: {summary['source_median_effective_na']:.4f} "
            f"+/- {summary['source_robust_sd_effective_na']:.4f}; "
            f"average {summary['average_effective_na']:.4f}"
        )
    print(f"Report: {REPORT_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
