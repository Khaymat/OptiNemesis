"""Pure problem instances and the harness-owned counting objective."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optinemesis.core.bounds import Bounds
from optinemesis.core.errors import BudgetExhausted


@dataclass(frozen=True, eq=False)
class ProblemInstance:
    """An immutable, reproducible optimization problem.

    ``evaluate`` must be a pure function of the input vector. Instances are
    reconstructed from ``spec`` plus the family registry, never from pickles.
    """

    family_name: str
    family_version: str
    theta: tuple[float, ...]
    dimension: int
    bounds: Bounds
    optimum_value: float
    _evaluator: Callable[[np.ndarray], float] = field(repr=False, compare=False)
    landscape_tags: Mapping[str, float] = field(default_factory=dict)
    spec: Mapping[str, Any] = field(default_factory=dict)
    optimum_x: np.ndarray | None = None

    def evaluate(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != self.dimension:
            raise ValueError(
                f"input has {arr.size} variables, problem dimension is {self.dimension}"
            )
        return float(self._evaluator(arr))

    def to_spec(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "family_version": self.family_version,
            "theta": list(self.theta),
            "dimension": self.dimension,
            "bounds": self.bounds.to_spec(),
            "optimum_value": self.optimum_value,
            "landscape_tags": dict(self.landscape_tags),
            "spec": dict(self.spec),
        }


class CountingObjective:
    """Harness-owned wrapper enforcing the exact evaluation budget.

    Every adapter receives this object as *the* objective. It raises
    :class:`BudgetExhausted` when the budget would be exceeded and records
    non-finite evaluations. Non-finite returns are passed to the optimizer as
    ``+inf`` (never optimal under minimization); they are counted and flagged,
    never silently replaced by fabricated finite values.
    """

    def __init__(self, instance: ProblemInstance, budget: int) -> None:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        self._instance = instance
        self._budget = int(budget)
        self.consumed = 0
        self.non_finite = 0

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def remaining(self) -> int:
        return self._budget - self.consumed

    @property
    def instance(self) -> ProblemInstance:
        return self._instance

    def __call__(self, x: np.ndarray) -> float:
        if self.consumed >= self._budget:
            raise BudgetExhausted(self._budget, self.consumed)
        value = self._instance.evaluate(x)
        self.consumed += 1
        if not np.isfinite(value):
            self.non_finite += 1
            return float(np.inf)
        return value
