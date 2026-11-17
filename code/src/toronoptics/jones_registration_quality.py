from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toronoptics.jones_registration import JonesRegistrationResult


def write_registration_quality_archive(
    path: Path,
    result: JonesRegistrationResult,
    *,
    labels: tuple[str, ...],
    anchor_centers_px: np.ndarray,
    force: bool = False,
) -> None:
    anchors = _validate(result, labels, anchor_centers_px)
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        labels=np.asarray(labels),
        anchor_centers_px=anchors,
        registered_centers_px=result.centers_px,
        corrections_px=result.corrections_px,
        pairwise_center_difference_px=result.pairwise_center_difference_px,
        pairwise_shift_px=result.pairwise_shift_px,
        pairwise_peak=result.pairwise_peak,
        pairwise_peak_ratio=result.pairwise_peak_ratio,
        pairwise_residual_px=result.pairwise_residual_px,
        triangle_closure_px=result.triangle_closure_px,
    )
    temporary.replace(path)


def render_registration_quality_pdf(
    path: Path,
    result: JonesRegistrationResult,
    *,
    labels: tuple[str, ...],
    anchor_centers_px: np.ndarray,
    force: bool = False,
) -> None:
    anchors = _validate(result, labels, anchor_centers_px)
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), constrained_layout=True)

    correction = result.centers_px - anchors
    axes[0, 0].axhline(0.0, color="0.75", lw=0.7)
    axes[0, 0].axvline(0.0, color="0.75", lw=0.7)
    axes[0, 0].scatter(correction[:, 0], correction[:, 1], s=24)
    for index, label in enumerate(labels):
        axes[0, 0].annotate(
            label,
            correction[index],
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_xlabel(r"$c_x^{\rm reg}-c_x^{m=1}$ (px)")
    axes[0, 0].set_ylabel(r"$c_y^{\rm reg}-c_y^{m=1}$ (px)")
    axes[0, 0].set_title("image-registration corrections", fontsize=9)

    residual_image = axes[0, 1].imshow(
        result.pairwise_residual_px,
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    axes[0, 1].set_title("pairwise graph residual (px)", fontsize=9)
    axes[0, 1].set_xlabel("record")
    axes[0, 1].set_ylabel("record")
    figure.colorbar(residual_image, ax=axes[0, 1], fraction=0.046, pad=0.03)

    ratio = np.asarray(result.pairwise_peak_ratio).copy()
    np.fill_diagonal(ratio, np.nan)
    ratio_image = axes[1, 0].imshow(
        ratio,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1, 0].set_title("correlation peak / competing peak", fontsize=9)
    axes[1, 0].set_xlabel("record")
    axes[1, 0].set_ylabel("record")
    figure.colorbar(ratio_image, ax=axes[1, 0], fraction=0.046, pad=0.03)

    closure = result.triangle_closure_px
    axes[1, 1].hist(closure, bins=max(8, int(np.sqrt(closure.size))), color="0.25")
    axes[1, 1].set_xlabel("triangle closure error (px)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("redundant-registration consistency", fontsize=9)
    axes[1, 1].text(
        0.98,
        0.96,
        (
            f"median: {np.median(closure):.3f} px\n"
            f"95%: {np.quantile(closure, 0.95):.3f} px\n"
            f"max: {np.max(closure):.3f} px"
        ),
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="pdf")
    plt.close(figure)
    temporary.replace(path)


def _validate(
    result: JonesRegistrationResult,
    labels: tuple[str, ...],
    anchor_centers_px: np.ndarray,
) -> np.ndarray:
    anchors = np.asarray(anchor_centers_px, dtype=np.float64)
    if anchors.shape != result.centers_px.shape:
        raise ValueError("anchor center shape does not match registration result")
    if len(labels) != result.centers_px.shape[0]:
        raise ValueError("labels and registration result must have the same length")
    return anchors
