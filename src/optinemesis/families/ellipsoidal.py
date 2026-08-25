"""Conditioned, rotatable, shiftable ellipsoid (quadratic bowl).

Parameters:
    conditioning      kappa in [10, 1000]   eigenvalue spread lambda_i = kappa^(i/(d-1))
    rotation_strength gamma in [0, 1]       seeded Givens blend from identity to dense rotation
    shift_radius      rho   in (0, 1)       optimum placed in a shrunk box

Known optimum: f(x*) = 0 at the seeded shifted center x*, analytically exact.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from optinemesis.core.bounds import Bounds
from optinemesis.core.contract import ParamRange
from optinemesis.families.base import FamilyBuild, ProblemFamily
from optinemesis.families.registry import register_family
from optinemesis.families.transforms import (
    givens_rotation,
    shifted_center,
)


@register_family
class EllipsoidalFamily(ProblemFamily):
    name: ClassVar[str] = "Ellipsoidal"
    version: ClassVar[str] = "1"
    parameter_box: ClassVar[tuple[ParamRange, ...]] = (
        ParamRange("conditioning", 10.0, 1000.0),
        ParamRange("rotation_strength", 0.0, 1.0),
        ParamRange("shift_radius", 0.1, 0.9),
    )
    transform_tags: ClassVar[tuple[str, ...]] = ("conditioning", "rotation", "shift")

    def _build(
        self,
        theta: tuple[float, ...],
        dimension: int,
        bounds: Bounds,
        generator: np.random.Generator,
    ) -> FamilyBuild:
        kappa, strength, radius = theta
        lo, hi = bounds.as_arrays()
        center = shifted_center(lo, hi, radius, generator)
        rotation = givens_rotation(dimension, strength, generator)

        if dimension == 1:
            eigenvalues = np.array([1.0])
        else:
            exponents = np.arange(dimension, dtype=float) / (dimension - 1)
            eigenvalues = np.power(kappa, exponents)

        def evaluate(x: np.ndarray) -> float:
            u = rotation.T @ (x - center)
            return float(np.dot(eigenvalues * u, u))

        reach = np.maximum(np.abs(lo - center), np.abs(hi - center))
        regret_scale = float(kappa * np.dot(reach, reach))

        return FamilyBuild(
            evaluator=evaluate,
            optimum_value=0.0,
            regret_scale=regret_scale,
            tags={
                "conditioning": kappa,
                "rotation_strength": strength,
                "shift_radius": radius,
                "modality": 1.0,
                "separability": 1.0 - strength,
            },
            spec={"lambda_max": float(eigenvalues[-1])},
            optimum_x=center.copy(),
        )
