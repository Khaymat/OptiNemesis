"""Multimodal Rastrigin blend with controllable modality and separability.

Parameters:
    modality          m  in [1, 3]   local-basin frequency (amplitude compensated)
    separability_blend s in [0, 1]   0 = separable in native coordinates,
                                    1 = fully rotated (non-separable) form
    shift_radius      rho in (0, 1)   optimum placement margin

Known optimum: f(x*) = 0 at the seeded shifted center, analytically exact:

    h_m(t) = t^2 - (A/m^2) cos(2 pi m t) + A/m^2  >= 0,  h_m(0) = 0
    f(x)   = (1-s) sum_i h_m(v_i) + s sum_i h_m((Q^T v)_i),  v = x - x*
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np

from optinemesis.core.bounds import Bounds
from optinemesis.core.contract import ParamRange
from optinemesis.families.base import FamilyBuild, ProblemFamily
from optinemesis.families.registry import register_family
from optinemesis.families.transforms import (
    corner_upper_bound,
    givens_rotation,
    shifted_center,
)

_RASTRIGIN_A = 10.0


def _basin_profile(modality: float) -> Callable[[np.ndarray], np.ndarray]:
    a_scaled = _RASTRIGIN_A / (modality * modality)

    def h(t: np.ndarray) -> np.ndarray:
        return t * t - a_scaled * np.cos(2.0 * np.pi * modality * t) + a_scaled

    return h


@register_family
class MultimodalBlendFamily(ProblemFamily):
    name: ClassVar[str] = "MultimodalBlend"
    version: ClassVar[str] = "1"
    parameter_box: ClassVar[tuple[ParamRange, ...]] = (
        ParamRange("modality", 1.0, 3.0),
        ParamRange("separability_blend", 0.0, 1.0),
        ParamRange("shift_radius", 0.1, 0.9),
    )
    transform_tags: ClassVar[tuple[str, ...]] = ("modality", "separability", "shift")

    def _build(
        self,
        theta: tuple[float, ...],
        dimension: int,
        bounds: Bounds,
        generator: np.random.Generator,
    ) -> FamilyBuild:
        modality, separability_blend, radius = theta
        lo, hi = bounds.as_arrays()
        center = shifted_center(lo, hi, radius, generator)
        rotation = givens_rotation(dimension, 1.0, generator)
        h = _basin_profile(modality)

        def evaluate(x: np.ndarray) -> float:
            v = x - center
            sep = float(np.sum(h(v)))
            nonsep = float(np.sum(h(rotation.T @ v)))
            return (1.0 - separability_blend) * sep + separability_blend * nonsep

        bound = corner_upper_bound(center, lo, hi)
        regret_scale = bound + 2.0 * dimension * _RASTRIGIN_A / (modality * modality)

        return FamilyBuild(
            evaluator=evaluate,
            optimum_value=0.0,
            regret_scale=float(regret_scale),
            tags={
                "modality": modality,
                "separability": 1.0 - separability_blend,
                "shift_radius": radius,
                "conditioning": 1.0,
            },
            spec={"rastrigin_a": _RASTRIGIN_A},
            optimum_x=center.copy(),
        )
