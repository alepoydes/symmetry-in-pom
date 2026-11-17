"""Explicit, serializable specifications for publication analyses.

The numerical results in the article depend on choices that used to be spread
across plotting and analysis scripts: the physical center, the outer
background region, and the radial kernel used for angular harmonics.  The
small immutable objects in this module make those choices part of the data
flow and provide a single metadata representation for generated archives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BackgroundEstimator = Literal["mean", "huber"]


@dataclass(frozen=True)
class CenterSpec:
    """A physical analysis center and its provenance."""

    x_px: float
    y_px: float
    method: str
    fitted: bool = False

    @property
    def center_px(self) -> tuple[float, float]:
        return float(self.x_px), float(self.y_px)

    def metadata(self) -> dict[str, object]:
        return {
            "center_px": [float(self.x_px), float(self.y_px)],
            "method": str(self.method),
            "fitted_to_optical_modes": bool(self.fitted),
        }


@dataclass(frozen=True)
class BackgroundSpec:
    """Definition of a constant complex Jones background."""

    min_radius_um: float
    max_radius_um: float
    estimator: BackgroundEstimator = "huber"
    huber_threshold: float = 1.5
    maximum_iterations: int = 25
    relative_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_radius_um) or self.min_radius_um < 0.0:
            raise ValueError("min_radius_um must be finite and nonnegative")
        if (
            not np.isfinite(self.max_radius_um)
            or self.max_radius_um <= self.min_radius_um
        ):
            raise ValueError("max_radius_um must exceed min_radius_um")
        if self.estimator not in {"mean", "huber"}:
            raise ValueError(f"Unsupported background estimator: {self.estimator!r}")
        if not np.isfinite(self.huber_threshold) or self.huber_threshold <= 0.0:
            raise ValueError("huber_threshold must be positive")
        if int(self.maximum_iterations) < 1:
            raise ValueError("maximum_iterations must be positive")
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")

    def metadata(self) -> dict[str, object]:
        return {
            "region": "physical annulus",
            "radius_um": [float(self.min_radius_um), float(self.max_radius_um)],
            "estimator": str(self.estimator),
            "huber_threshold": (
                float(self.huber_threshold) if self.estimator == "huber" else None
            ),
        }


@dataclass(frozen=True)
class HarmonicSpec:
    """Radial sampling and integration convention for intensity harmonics."""

    radius_start_um: float = 1.0
    radius_stop_um: float = 32.0
    radius_step_um: float = 0.25
    radial_sigma_um: float = 0.75
    integration_min_um: float = 4.0
    integration_max_um: float = 28.0
    modes: tuple[int, ...] = (0, 2, 4)

    def __post_init__(self) -> None:
        values = (
            self.radius_start_um,
            self.radius_stop_um,
            self.radius_step_um,
            self.radial_sigma_um,
            self.integration_min_um,
            self.integration_max_um,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Harmonic specification values must be finite")
        if self.radius_start_um < 0.0 or self.radius_stop_um <= self.radius_start_um:
            raise ValueError("Invalid radial sampling interval")
        if self.radius_step_um <= 0.0 or self.radial_sigma_um <= 0.0:
            raise ValueError("Radial step and Gaussian width must be positive")
        if not (
            self.radius_start_um
            <= self.integration_min_um
            < self.integration_max_um
            <= self.radius_stop_um
        ):
            raise ValueError("Integration interval must lie inside the radial grid")
        if len(self.modes) == 0 or len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must be nonempty and unique")
        if 0 not in self.modes:
            raise ValueError("modes must contain zero")

    @property
    def radial_centers_um(self) -> np.ndarray:
        return np.arange(
            float(self.radius_start_um),
            float(self.radius_stop_um) + 0.5 * float(self.radius_step_um),
            float(self.radius_step_um),
            dtype=np.float64,
        )

    @property
    def integration_interval_um(self) -> tuple[float, float]:
        return float(self.integration_min_um), float(self.integration_max_um)

    def metadata(self) -> dict[str, object]:
        return {
            "radial_grid_um": {
                "start": float(self.radius_start_um),
                "stop": float(self.radius_stop_um),
                "step": float(self.radius_step_um),
            },
            "radial_kernel": {
                "kind": "Gaussian annulus",
                "sigma_um": float(self.radial_sigma_um),
            },
            "integration_interval_um": [
                float(self.integration_min_um),
                float(self.integration_max_um),
            ],
            "radial_measure": "rho d rho",
            "fourier_convention": "mean_phi[f(phi) exp(-i m phi)]",
            "modes": [int(mode) for mode in self.modes],
        }


PUBLICATION_BACKGROUND = BackgroundSpec(41.0, 45.0, estimator="huber")
PUBLICATION_HARMONICS = HarmonicSpec()
