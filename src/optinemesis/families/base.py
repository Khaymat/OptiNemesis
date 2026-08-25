"""ProblemFamily base class and sampling protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from optinemesis.core.bounds import Bounds
from optinemesis.core.contract import FamilyRef, ParamRange
from optinemesis.core.objective import ProblemInstance


@dataclass(frozen=True)
class FamilyBuild:
    """Everything a family needs to emit a ProblemInstance."""

    evaluator: Callable[[np.ndarray], float]
    optimum_value: float
    regret_scale: float
    tags: dict[str, float]
    spec: dict[str, object]
    optimum_x: np.ndarray | None = None


class ProblemFamily(ABC):
    """A parameterized generator of reproducible problem instances.

    Subclasses fix ``name``, ``version``, an interpretable ``parameter_box``
    and implement :meth:`_build`. All stochastic structure (rotations,
    shifts) must derive exclusively from the provided generator so that
    ``(family, version, theta, dimension, instance_seed, bounds)`` reproduces
    a bit-identical instance.
    """

    name: ClassVar[str]
    version: ClassVar[str] = "1"
    parameter_box: ClassVar[tuple[ParamRange, ...]]
    transform_tags: ClassVar[tuple[str, ...]] = ()

    def registry_key(self) -> str:
        return f"{self.name}@{self.version}"

    def family_ref(self) -> FamilyRef:
        return FamilyRef(
            name=self.name,
            version=self.version,
            parameter_box=self.parameter_box,
            transform_tags=self.transform_tags,
        )

    @abstractmethod
    def _build(
        self,
        theta: tuple[float, ...],
        dimension: int,
        bounds: Bounds,
        generator: np.random.Generator,
    ) -> FamilyBuild: ...

    def sample(
        self,
        theta: tuple[float, ...],
        dimension: int,
        instance_seed: int,
        bounds: Bounds,
    ) -> ProblemInstance:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if bounds.dimension != dimension:
            raise ValueError(
                f"bounds have dimension {bounds.dimension}, expected {dimension}"
            )
        checked = self.family_ref().check_theta(tuple(theta))
        generator = np.random.default_rng(instance_seed)
        build = self._build(checked, dimension, bounds, generator)
        if not np.isfinite(build.optimum_value):
            raise ValueError("family produced a non-finite claimed optimum")
        if build.regret_scale <= 0.0 or not np.isfinite(build.regret_scale):
            raise ValueError("family produced an invalid regret scale")
        spec: dict[str, object] = {
            "kind": self.name,
            "family_version": self.version,
            "theta": list(checked),
            "dimension": dimension,
            "instance_seed": int(instance_seed),
            "bounds": bounds.to_spec(),
            "regret_scale": build.regret_scale,
            **build.spec,
        }
        tags = {"regret_scale": build.regret_scale, **build.tags}
        return ProblemInstance(
            family_name=self.name,
            family_version=self.version,
            theta=checked,
            dimension=dimension,
            bounds=bounds,
            optimum_value=float(build.optimum_value),
            landscape_tags=tags,
            spec=spec,
            _evaluator=build.evaluator,
            optimum_x=(
                None if build.optimum_x is None else np.asarray(build.optimum_x, dtype=float)
            ),
        )
