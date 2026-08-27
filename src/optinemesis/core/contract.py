"""The experimental fairness contract and its building blocks."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from optinemesis.core.bounds import Bounds
from optinemesis.core.errors import ContractError
from optinemesis.core.fingerprint import canonical_json, fingerprint

METRIC_NAMES = ("rank_biserial", "median_gap", "probability_superiority")
# v0.1: primary metric frozen to rank_biserial; paired initialization not honestly
# implementable across heterogeneous adapters (DE vs single-point methods). Only
# independent is supported. Kept as tuple for schema clarity.
INITIALIZATION_POLICIES = ("independent",)
META_SEARCH_METHODS = ("random", "lhs")

RHO_MAX = 1.0e6


@dataclass(frozen=True)
class ParamRange:
    """One interpretable family parameter and its fixed box."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("parameter name must be non-empty")
        if not (np.isfinite(self.lower) and np.isfinite(self.upper)):
            raise ContractError(f"parameter {self.name!r}: range must be finite")
        if not self.lower < self.upper:
            raise ContractError(f"parameter {self.name!r}: lower must be below upper")

    def to_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class FamilyRef:
    """Reference to a registered problem family and its parameter box."""

    name: str
    version: str
    parameter_box: tuple[ParamRange, ...]
    transform_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ContractError("family name and version must be non-empty")
        if len(self.parameter_box) == 0:
            raise ContractError("family parameter box must not be empty")
        names = [p.name for p in self.parameter_box]
        if len(set(names)) != len(names):
            raise ContractError("duplicate parameter names in family parameter box")

    def check_theta(self, theta: tuple[float, ...]) -> tuple[float, ...]:
        if len(theta) != len(self.parameter_box):
            raise ContractError(
                f"theta length {len(theta)} does not match family parameter box "
                f"length {len(self.parameter_box)}"
            )
        checked: list[float] = []
        for value, rng in zip(theta, self.parameter_box, strict=True):
            v = float(value)
            if not np.isfinite(v) or not (rng.lower <= v <= rng.upper):
                raise ContractError(
                    f"theta[{rng.name!r}]={v} outside declared range "
                    f"[{rng.lower}, {rng.upper}]"
                )
            checked.append(v)
        return tuple(checked)

    def center_theta(self) -> tuple[float, ...]:
        return tuple((p.lower + p.upper) / 2.0 for p in self.parameter_box)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parameter_box": [p.to_dict() for p in self.parameter_box],
            "transform_tags": list(self.transform_tags),
        }


@dataclass(frozen=True)
class SeedPlan:
    root_entropy: str
    n_theta_candidates: int = 64
    n_search_seeds: int = 20
    n_validation_seeds: int = 100
    meta_search_method: str = "lhs"

    def __post_init__(self) -> None:
        int(self.root_entropy)
        for attr in ("n_theta_candidates", "n_search_seeds", "n_validation_seeds"):
            value = getattr(self, attr)
            minimum = 2 if attr != "n_theta_candidates" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ContractError(
                    f"seeds.{attr} must be an integer >= {minimum}, got {value!r}"
                )
        if self.meta_search_method not in META_SEARCH_METHODS:
            raise ContractError(f"unknown meta-search method {self.meta_search_method!r}")


@dataclass(frozen=True)
class MetricPlan:
    primary: str = "rank_biserial"
    exploratory: tuple[str, ...] = ("median_gap", "probability_superiority")

    def __post_init__(self) -> None:
        if self.primary not in METRIC_NAMES:
            raise ContractError(f"primary metric {self.primary!r} is not a known metric")
        # v0.1: primary frozen to rank_biserial per DESIGN.md §2. Other
        # values are schema-valid but would require recalibrating epsilon_min
        # (which is defined on rank-biserial scale) and changing the
        # nomination/validation statistics. Reject here rather than silently
        # ignoring the plan (audit MINOR).
        if self.primary != "rank_biserial":
            raise ContractError(
                f"primary metric {self.primary!r} not supported in v0.1; "
                "only 'rank_biserial' is frozen for this version"
            )
        unknown = [m for m in self.exploratory if m not in METRIC_NAMES]
        if unknown:
            raise ContractError(f"unknown exploratory metrics: {unknown}")
        if self.primary in self.exploratory:
            raise ContractError(
                "primary metric must not be repeated among exploratory metrics"
            )


@dataclass(frozen=True)
class Thresholds:
    epsilon_min: float = 0.33
    epsilon_zero: float = 0.0
    alpha_search: float = 0.05
    alpha_validation: float = 0.05
    bootstrap_resamples: int = 10_000

    def __post_init__(self) -> None:
        for name in ("epsilon_min", "alpha_search", "alpha_validation"):
            value = getattr(self, name)
            if not 0.0 < float(value) <= 1.0:
                raise ContractError(f"thresholds.{name} must lie in (0, 1], got {value}")
        if float(self.epsilon_zero) < 0.0:
            raise ContractError("thresholds.epsilon_zero must be >= 0")
        if self.bootstrap_resamples < 100:
            raise ContractError("thresholds.bootstrap_resamples must be >= 100")


@dataclass(frozen=True)
class OptimizerSpec:
    """A frozen optimizer configuration under comparison."""

    name: str
    implementation: str
    config: dict[str, Any] = field(default_factory=dict)
    seeded: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.implementation:
            raise ContractError("optimizer name and implementation must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "config": self.config,
            "seeded": self.seeded,
        }


@dataclass(frozen=True)
class EnvironmentInfo:
    python: str
    numpy_version: str
    scipy_version: str
    optinemesis_version: str
    platform: str

    @classmethod
    def capture(cls) -> EnvironmentInfo:
        import scipy

        from optinemesis import __version__

        return cls(
            python=platform.python_version(),
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            optinemesis_version=__version__,
            platform=platform.platform(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "python": self.python,
            "numpy": self.numpy_version,
            "scipy": self.scipy_version,
            "optinemesis": self.optinemesis_version,
            "platform": self.platform,
        }


@dataclass(frozen=True)
class StudySpec:
    """Everything needed to define one counterexample-discovery study.

    A :class:`StudySpec` has no concrete parameter point; ``bind(theta)``
    produces a :class:`Contract` at a specific family parameter vector.
    """

    family: FamilyRef
    dimension: int
    bounds: Bounds
    budget: int
    optimizers: tuple[OptimizerSpec, OptimizerSpec]
    seeds: SeedPlan = field(default_factory=lambda: SeedPlan(root_entropy="0"))
    metrics: MetricPlan = field(default_factory=MetricPlan)
    thresholds: Thresholds = field(default_factory=Thresholds)
    initialization_policy: str = "independent"
    environment: EnvironmentInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool):
            raise ContractError("dimension must be an integer")
        if self.dimension < 1:
            raise ContractError("dimension must be >= 1")
        if not isinstance(self.budget, int) or isinstance(self.budget, bool) or self.budget < 1:
            raise ContractError("budget must be an integer >= 1")
        if self.initialization_policy not in INITIALIZATION_POLICIES:
            raise ContractError(
                f"initialization_policy {self.initialization_policy!r} must be one of "
                f"{INITIALIZATION_POLICIES}"
            )
        names = [o.name for o in self.optimizers]
        if len(names) != 2:
            raise ContractError("exactly two optimizer configurations are required")
        if names[0] == names[1]:
            raise ContractError("the two optimizer configurations must have distinct names")
        if self.bounds.dimension != self.dimension:
            raise ContractError(
                f"bounds have dimension {self.bounds.dimension}, expected {self.dimension}"
            )
        if self.seeds.n_search_seeds < 2 or self.seeds.n_validation_seeds < 2:
            raise ContractError("at least two seeds per stage are required")

    def bind(self, theta: tuple[float, ...]) -> Contract:
        checked = self.family.check_theta(tuple(theta))
        return Contract(study=self, theta=checked)


@dataclass(frozen=True)
class Contract:
    """A :class:`StudySpec` bound to a concrete family parameter vector."""

    study: StudySpec
    theta: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "theta", self.study.family.check_theta(tuple(self.theta))
        )

    @property
    def family(self) -> FamilyRef:
        return self.study.family

    @property
    def dimension(self) -> int:
        return self.study.dimension

    @property
    def bounds(self) -> Bounds:
        return self.study.bounds

    @property
    def budget(self) -> int:
        return self.study.budget

    @property
    def initialization_policy(self) -> str:
        return self.study.initialization_policy

    @property
    def seeds(self) -> SeedPlan:
        return self.study.seeds

    @property
    def metrics(self) -> MetricPlan:
        return self.study.metrics

    @property
    def thresholds(self) -> Thresholds:
        return self.study.thresholds

    @property
    def optimizers(self) -> tuple[OptimizerSpec, OptimizerSpec]:
        return self.study.optimizers

    @property
    def environment(self) -> EnvironmentInfo | None:
        return self.study.environment

    def contract_id(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        # optimizer_order preserves A/B label order through JSON sorted-key
        # serialization (canonical_json sorts dict keys, so raw dict order is lost).
        d: dict[str, Any] = {
            "schema_version": "1",
            "contract_id": "",
            "family": {
                **self.study.family.to_dict(),
                "theta": list(self.theta),
                "dimension": self.study.dimension,
            },
            "bounds": self.study.bounds.to_spec(),
            "budget": self.study.budget,
            "initialization_policy": self.study.initialization_policy,
            "optimizer_order": [o.name for o in self.study.optimizers],
            "seeds": {
                "root_entropy": self.study.seeds.root_entropy,
                "n_theta_candidates": self.study.seeds.n_theta_candidates,
                "n_search_seeds": self.study.seeds.n_search_seeds,
                "n_validation_seeds": self.study.seeds.n_validation_seeds,
                "meta_search_method": self.study.seeds.meta_search_method,
            },
            "metrics": {
                "primary": self.study.metrics.primary,
                "exploratory": list(self.study.metrics.exploratory),
            },
            "thresholds": {
                "epsilon_min": self.study.thresholds.epsilon_min,
                "epsilon_zero": self.study.thresholds.epsilon_zero,
                "alpha_search": self.study.thresholds.alpha_search,
                "alpha_validation": self.study.thresholds.alpha_validation,
                "bootstrap_resamples": self.study.thresholds.bootstrap_resamples,
            },
            "optimizers": {o.name: o.to_dict() for o in self.study.optimizers},
        }
        if self.study.environment is not None:
            d["environment"] = self.study.environment.to_dict()
        d["contract_id"] = fingerprint(canonical_json(d))
        return d


def validate_study(spec: StudySpec) -> StudySpec:
    """Re-validate a deserialized study specification."""
    return replace(spec)
