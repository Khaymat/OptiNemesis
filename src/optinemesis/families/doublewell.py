"""Deceptive double-well: a wide shallow basin versus a narrow deep global basin.

Parameters:
    depth_ratio          r in [2, 20]   deep-well depth / wide-well depth
    width_ratio          w in [2, 50]   wide-basin width / deep-basin width
    separation_fraction  s in (0, 1)    distance between well centers relative
                                        to the smallest box span

Construction: f(x) = min(q_wide(x), q_deep(x)) with quadratic wells

    q_wide(x) = a_base * ||x - c_w||^2 - delta_w
    q_deep(x) = a_base * w^2 * ||x - c_d||^2 - r * delta_w

Known optimum: f(c_d) = -r * delta_w exactly; the wide well can never go below
-delta_w > -r*delta_w, so the narrow well is the unique global minimum.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from optinemesis.core.bounds import Bounds
from optinemesis.core.contract import ParamRange
from optinemesis.families.base import FamilyBuild, ProblemFamily
from optinemesis.families.registry import register_family
from optinemesis.families.transforms import random_unit_vector


@register_family
class DeceptiveDoubleWellFamily(ProblemFamily):
    name: ClassVar[str] = "DeceptiveDoubleWell"
    version: ClassVar[str] = "1"
    parameter_box: ClassVar[tuple[ParamRange, ...]] = (
        ParamRange("depth_ratio", 2.0, 20.0),
        ParamRange("width_ratio", 2.0, 50.0),
        ParamRange("separation_fraction", 0.25, 0.75),
    )
    transform_tags: ClassVar[tuple[str, ...]] = ("deception", "modality")

    def _build(
        self,
        theta: tuple[float, ...],
        dimension: int,
        bounds: Bounds,
        generator: np.random.Generator,
    ) -> FamilyBuild:
        depth_ratio, width_ratio, separation = theta
        lo, hi = bounds.as_arrays()
        box_center = (lo + hi) / 2.0
        direction = random_unit_vector(dimension, generator)
        span_min = float(np.min(hi - lo))
        half_distance = 0.5 * separation * span_min
        c_wide = box_center - half_distance * direction
        c_deep = box_center + half_distance * direction

        a_base = 1.0 / (span_min * span_min)
        delta_w = a_base * half_distance * half_distance
        delta_deep = depth_ratio * delta_w

        def evaluate(x: np.ndarray) -> float:
            dw = x - c_wide
            dd = x - c_deep
            q_wide = a_base * float(np.dot(dw, dw)) - delta_w
            q_deep = a_base * width_ratio * width_ratio * float(np.dot(dd, dd)) - delta_deep
            return min(q_wide, q_deep)

        optimum_value = -delta_deep
        regret_scale = delta_deep + float(a_base) * float(np.dot(lo - hi, lo - hi))

        return FamilyBuild(
            evaluator=evaluate,
            optimum_value=float(optimum_value),
            regret_scale=float(regret_scale),
            tags={
                "deception_depth_ratio": depth_ratio,
                "basin_width_ratio": width_ratio,
                "separation_fraction": separation,
                "modality": 2.0,
                "conditioning": 1.0,
                "separability": 1.0,
            },
            spec={
                "c_wide": c_wide.tolist(),
                "c_deep": c_deep.tolist(),
                "delta_wide": float(delta_w),
                "a_base": float(a_base),
            },
            optimum_x=c_deep.copy(),
        )
