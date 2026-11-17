"""Shared output paths and overlay styling for publication figures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


FIGURE_OUTPUT_ENV = "ARTICLE_FIGURE_DIR"
OPTICAL_ANNOTATION_COLOR = "#FFFFFF"
FOCUS_PANEL_FRAME_SCALE = 1.12
FOCUS_FIGURE_WIDTH_PT = 511.77


def expand_focus_panels(fig, image_axes, profile_axes, *, padding: float = 0.015):
    """Expand paired columns into their gaps without scaling the PDF or fonts.

    The optical renders use FOCUS_PANEL_FRAME_SCALE to add surrounding space
    at the original image scale. Keep the profiles' height and the two rows'
    baseline positions; grow the square image frames upward.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    horizontal_bounds = fig.get_tightbbox(fig.canvas.get_renderer()).frozen()
    image_boxes = [axis.get_position().frozen() for axis in image_axes]
    profile_boxes = [axis.get_position().frozen() for axis in profile_axes]
    left, right = image_boxes[0].x0, image_boxes[-1].x1
    width = image_boxes[0].width * FOCUS_PANEL_FRAME_SCALE
    step = (right - left - width) / (len(image_axes) - 1)
    if step <= width:
        raise ValueError("Expanded focus panels would overlap")
    for index, (image_axis, profile_axis, image_box, profile_box) in enumerate(
        zip(image_axes, profile_axes, image_boxes, profile_boxes, strict=True)
    ):
        x0 = left + index * step
        image_axis.set_position(
            (x0, image_box.y0, width, width * fig.get_figwidth() / fig.get_figheight())
        )
        profile_axis.set_box_aspect(None)
        profile_axis.set_position((x0, profile_box.y0, width, profile_box.height))
        labels = profile_axis.get_xticklabels()
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    # Keep the horizontal crop fixed so insertion at text width preserves scale.
    crop = Bbox.from_extents(
        horizontal_bounds.x0, bounds.y0, horizontal_bounds.x1, bounds.y1
    ).padded(padding)
    # Agg and PDF font metrics differ slightly; retain the publication width
    # explicitly so the journal's text-width insertion preserves image scale.
    return crop.expanded(FOCUS_FIGURE_WIDTH_PT / (72.0 * crop.width), 1.0)


def recolor_white_annotations(
    annotated: np.ndarray,
    background: np.ndarray,
    *,
    color: str = OPTICAL_ANNOTATION_COLOR,
) -> np.ndarray:
    """Tint white overlays using paired RGBA renders of an opaque panel.

    Recover antialiasing coverage from the white-on-background composite,
    then composite the requested color with that coverage. Unannotated
    pixels, including any white pixels in the optical image, stay unchanged.
    Both renders must have outlines disabled and use the same optical state.
    Retain TopoVec's original alpha channel, including translucent edges.
    """
    if (
        annotated.shape != background.shape
        or annotated.ndim != 3
        or annotated.shape[-1] != 4
        or annotated.dtype != np.uint8
        or background.dtype != np.uint8
    ):
        raise ValueError("Expected matching uint8 RGBA images")
    if np.any(background[..., 3] != 255):
        raise ValueError("Annotation recoloring requires an opaque background")
    rgb = np.asarray([int(color[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0
    changed = np.any(annotated[..., :3] != background[..., :3], axis=-1)
    base = background[changed, :3].astype(np.float64) / 255.0
    white = annotated[changed, :3].astype(np.float64) / 255.0
    headroom = 1.0 - base
    coverage = np.clip(
        np.sum((white - base) * headroom, axis=-1)
        / np.maximum(np.sum(headroom**2, axis=-1), np.finfo(float).eps),
        0.0,
        1.0,
    )[:, None]
    if np.max(np.abs(white - (base + coverage * headroom)), initial=0.0) > 2.0 / 255.0:
        raise ValueError("The render pair differs by more than white annotations")
    result = annotated.copy()
    result[changed, :3] = np.rint(255.0 * (base + coverage * (rgb - base))).astype(np.uint8)
    return result


def figure_output_dir(project_root: Path) -> Path:
    """Return the requested figure directory, or ``figs`` for direct use."""
    configured = os.environ.get(FIGURE_OUTPUT_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()
    return project_root / "figs"


def release_render_container(container) -> None:
    """Release an offscreen TopoVec renderer after copying its output image.

    The pinned TopoVec ImageContainer has no close method. Its GL resources
    need explicit release; deleting the Python wrapper alone can retain them
    across the many panels in a supplementary figure.
    """
    try:
        container.scene.release()
    finally:
        try:
            container._target.release()
        finally:
            container._ctx.release()
