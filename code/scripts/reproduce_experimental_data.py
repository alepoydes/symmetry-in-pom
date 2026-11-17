#!/usr/bin/env python3
"""Rebuild registered Jones averages and direct-POM reductions from data_src/."""

from __future__ import annotations

import argparse
from pathlib import Path

from toronoptics.cli import add_force_argument
from toronoptics.reproduction import ReproductionStep, run_steps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = ("uv", "run", "python")


STEPS = (
    # Fail before expensive work if a Jones acquisition, selected
    # direct-POM frame, dark frame, or Ground reference is absent.
    ReproductionStep(
        "Validate the minimal source-data set",
        "Check 27 holographic Jones fields and the selected "
        "dual-exposure direct-POM frames and references.",
        ("./scripts/check_data_src",),
        accepts_force=False,
    ),
    # Each wavelength contains nine shifted single-toron acquisitions.  Locate
    # the toron independently, refine its center from forbidden m=+/-1 power,
    # register all four complex Jones channels jointly, and drizzle-average.
    ReproductionStep(
        "Register and average the holographic Jones fields",
        "Infer centers and pairwise shifts for all nine acquisitions at each "
        "wavelength, then form one centered complex Jones field.",
        (*PYTHON, "code/scripts/prepare_shifted_toron_jones_series.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/results_toron25um_50X_averaging/report.json",
            *tuple(
                PROJECT_ROOT
                / f"data/experimental/results_toron25um_50X_averaged/{wavelength}nm_average.npz"
                for wavelength in (450, 520, 660)
            ),
            *tuple(
                PROJECT_ROOT
                / f"data/analysis/results_toron25um_50X_averaging/{wavelength}nm_centers.csv"
                for wavelength in (450, 520, 660)
            ),
        ),
    ),
    # The acquisition pupil is a measured property, not a fitted parameter in
    # later comparisons.  Estimate its coherent cutoff from every acquisition
    # and pool the source-field estimates wavelength by wavelength.
    ReproductionStep(
        "Estimate the coherent holographic pupil",
        "Measure the effective coherent NA from the registered source ensemble "
        "and its averages.",
        (*PYTHON, "code/scripts/estimate_averaged_pupil.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/averaged_pupil_spectrum/averaged_pupil_spectrum.npz",
            PROJECT_ROOT / "data/analysis/averaged_pupil_spectrum/report.json",
        ),
    ),
    # Decode the polarization-camera quadrants once.  Dark-correct both
    # exposures, register them, and use the short exposure only where the
    # 42-ms signal saturates.  Doubly saturated pixels retain zero confidence.
    ReproductionStep(
        "Build the direct-POM HDR subset",
        "Combine the selected 12-ms and 42-ms frames after dark correction, "
        "quadrant registration, exposure-ratio estimation, and confidence masking.",
        (*PYTHON, "code/scripts/build_focus_stack_hdr.py"),
        outputs=(
            PROJECT_ROOT
            / "data/experimental/results_toron25um_50X_focus_hdr/manifest.json",
            PROJECT_ROOT
            / "data/experimental/results_toron25um_50X_focus_hdr/reference_background.npz",
            *tuple(
                PROJECT_ROOT
                / f"data/experimental/results_toron25um_50X_focus_hdr/frames/{index:04d}.npz"
                for index in range(45, 50)
            ),
        ),
    ),
    # Propagate the measured 520-nm Jones field and compare its radial I0/I90
    # profiles directly with the five HDR states.  This infers only the five
    # effective axial coordinates used in the publication comparison.
    ReproductionStep(
        "Recover the five direct-POM axial coordinates",
        "Match confidence-weighted radial profiles to propagated holographic "
        "fields without fitting an intensity scale or offset.",
        (*PYTHON, "code/scripts/analyze_holographic_hdr_focus_stack.py"),
        outputs=(
            PROJECT_ROOT
            / "data/analysis/results_toron25um_50X_focus_recovery/report.json",
            PROJECT_ROOT
            / "data/analysis/results_toron25um_50X_focus_recovery/focus_fit.npz",
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_force_argument(parser)
    parser.add_argument("--dry-run", action="store_true", help="Check existing outputs without running calculations")
    args = parser.parse_args(argv)
    run_steps(STEPS, project_root=PROJECT_ROOT, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
