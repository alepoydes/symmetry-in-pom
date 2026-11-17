from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from toronoptics.toron_harmonic_centers import HarmonicCenterAnalysis


def write_center_quality_archive(
    path: Path,
    analyses: tuple[HarmonicCenterAnalysis, ...],
    *,
    labels: tuple[str, ...],
    force: bool = False,
) -> None:
    _validate_inputs(analyses, labels)
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    arrays = {
        "labels": np.asarray(labels),
        "center_px": np.asarray(
            [(item.center.cx, item.center.cy) for item in analyses], dtype=np.float64
        ),
        "initial_center_px": np.asarray(
            [(item.center.initial_cx, item.center.initial_cy) for item in analyses],
            dtype=np.float64,
        ),
        "broad_x_px": np.stack([item.search.broad_x_px for item in analyses]),
        "broad_y_px": np.stack([item.search.broad_y_px for item in analyses]),
        "broad_first_objective": np.stack(
            [item.search.broad_objective for item in analyses]
        ),
        "broad_all_objective": np.stack(
            [item.search.broad_all_objective for item in analyses]
        ),
        "fine_x_px": np.stack([item.search.fine_x_px for item in analyses]),
        "fine_y_px": np.stack([item.search.fine_y_px for item in analyses]),
        "fine_first_objective": np.stack(
            [item.search.fine_objective for item in analyses]
        ),
        "fine_all_objective": np.stack(
            [item.search.fine_all_objective for item in analyses]
        ),
        "broad_converged": np.asarray(
            [item.search.broad_converged for item in analyses], dtype=bool
        ),
        "coarse_converged": np.asarray(
            [item.search.coarse_converged for item in analyses], dtype=bool
        ),
        "radius_um": np.stack([item.harmonics.radius_um for item in analyses]),
        "first_harmonic_power": np.stack(
            [item.harmonics.first_harmonic_power for item in analyses]
        ),
        "all_forbidden_power": np.stack(
            [item.harmonics.all_forbidden_power for item in analyses]
        ),
        "radial_signal_power": np.stack(
            [item.harmonics.radial_signal_power for item in analyses]
        ),
        "modes": np.stack([item.harmonics.modes for item in analyses]),
        "integrated_mode_power": np.stack(
            [item.harmonics.integrated_mode_power for item in analyses]
        ),
    }
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def render_center_quality_pdf(
    path: Path,
    analyses: tuple[HarmonicCenterAnalysis, ...],
    *,
    labels: tuple[str, ...],
    force: bool = False,
) -> None:
    _validate_inputs(analyses, labels)
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with PdfPages(temporary) as pdf:
        for analysis, label in zip(analyses, labels, strict=True):
            figure, axes = plt.subplots(2, 3, figsize=(8.2, 5.6), constrained_layout=True)
            _objective_map(
                axes[0, 0],
                analysis.search.broad_x_px,
                analysis.search.broad_y_px,
                analysis.search.broad_objective,
                analysis,
                title=r"broad $m=\pm1$ score",
            )
            _objective_map(
                axes[0, 1],
                analysis.search.fine_x_px,
                analysis.search.fine_y_px,
                analysis.search.fine_objective,
                analysis,
                title=r"fine $m=\pm1$ score",
            )
            _objective_map(
                axes[0, 2],
                analysis.search.broad_x_px,
                analysis.search.broad_y_px,
                analysis.search.broad_all_objective,
                analysis,
                title="all forbidden harmonics",
            )
            _radial_power(axes[1, 0], analysis)
            _mode_power(axes[1, 1], analysis)
            axes[1, 2].axis("off")
            center = analysis.center
            axes[1, 2].text(
                0.0,
                1.0,
                "\n".join(
                    (
                        label,
                        f"initial: ({center.initial_cx:.3f}, {center.initial_cy:.3f}) px",
                        f"center: ({center.cx:.3f}, {center.cy:.3f}) px",
                        f"Q1: {center.objective:.5g}",
                        f"Qall: {center.all_forbidden_objective:.5g}",
                        (
                            "local Qall-center difference: "
                            f"{center.all_forbidden_center_disagreement_px:.3f} px"
                        ),
                        f"broad minimum interior: {center.broad_converged}",
                        f"local minimum interior: {center.coarse_converged}",
                    )
                ),
                va="top",
                ha="left",
                fontsize=8,
                linespacing=1.45,
            )
            pdf.savefig(figure, dpi=180)
            plt.close(figure)
    temporary.replace(path)


def _objective_map(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    objective: np.ndarray,
    analysis: HarmonicCenterAnalysis,
    *,
    title: str,
) -> None:
    initial = np.asarray(
        (analysis.center.initial_cx, analysis.center.initial_cy), dtype=np.float64
    )
    values = np.asarray(objective, dtype=np.float64)
    minimum = max(float(np.nanmin(values)), np.finfo(float).tiny)
    image = axis.imshow(
        np.log10(np.maximum(values / minimum, 1.0)).T,
        origin="lower",
        extent=(x[0] - initial[0], x[-1] - initial[0], y[0] - initial[1], y[-1] - initial[1]),
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )
    center = np.asarray((analysis.center.cx, analysis.center.cy)) - initial
    axis.plot(0.0, 0.0, marker="+", color="white", ms=7, mew=1.1)
    axis.plot(center[0], center[1], marker="x", color="red", ms=6, mew=1.2)
    axis.set_xlabel(r"$c_x-c_{x,0}$ (px)")
    axis.set_ylabel(r"$c_y-c_{y,0}$ (px)")
    axis.set_title(title, fontsize=9)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label(r"$\log_{10}(Q/Q_{\min})$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def _radial_power(axis: plt.Axes, analysis: HarmonicCenterAnalysis) -> None:
    harmonics = analysis.harmonics
    scale = np.maximum(harmonics.radial_signal_power, np.finfo(float).tiny)
    axis.semilogy(
        harmonics.radius_um,
        np.maximum(harmonics.first_harmonic_power / scale, np.finfo(float).tiny),
        label=r"$m=\pm1$",
    )
    axis.semilogy(
        harmonics.radius_um,
        np.maximum(harmonics.all_forbidden_power / scale, np.finfo(float).tiny),
        label=r"all $m\ne0$",
    )
    axis.set_xlabel(r"$\rho$ ($\mu$m)")
    axis.set_ylabel("power / radial signal")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)


def _mode_power(axis: plt.Axes, analysis: HarmonicCenterAnalysis) -> None:
    harmonics = analysis.harmonics
    power = harmonics.integrated_mode_power
    power = power / max(float(np.sum(power)), np.finfo(float).tiny)
    axis.bar(harmonics.modes, np.maximum(power, np.finfo(float).tiny), width=0.7)
    axis.set_yscale("log")
    axis.set_xlabel(r"corotating mode $m$")
    axis.set_ylabel("fraction of sampled power")
    axis.grid(axis="y", alpha=0.2)


def _validate_inputs(
    analyses: tuple[HarmonicCenterAnalysis, ...], labels: tuple[str, ...]
) -> None:
    if not analyses:
        raise ValueError("At least one center analysis is required")
    if len(analyses) != len(labels):
        raise ValueError("labels and analyses must have the same length")
