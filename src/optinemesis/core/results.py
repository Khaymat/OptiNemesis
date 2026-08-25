"""Result model: runs, search-stage evaluations, validation outcomes.

``SearchResult`` and ``ValidationResult`` are deliberately distinct types.
A ``SearchResult`` is exploratory and can never be rendered as a confirmed
finding; only a ``ValidationResult`` carries confirmatory statistics
(see docs/DESIGN.md §6 and docs/CLAIMS.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SEARCH_STAGE_LABEL = "Search stage (exploratory - not confirmatory evidence)"

CONFIRMATION_CRITERIA = (
    "holm_significant",
    "effect_threshold_met",
    "ci_excludes_margin",
    "reference_direction_opposed",
)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one optimizer configuration on one problem instance."""

    optimizer_name: str
    implementation: str
    best_x: np.ndarray | None
    best_f: float | None
    regret_normalized: float | None = None
    n_evals_consumed: int = 0
    overshoot_events: int = 0
    non_finite_evals: int = 0
    runtime_s: float = 0.0
    termination_reason: str = "unknown"
    compliance_flags: tuple[str, ...] = ()
    history: tuple[tuple[int, float], ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_evals_consumed < 0:
            raise ValueError("n_evals_consumed must be >= 0")
        if self.overshoot_events < 0:
            raise ValueError("overshoot_events must be >= 0")

    def with_flags(self, *flags: str) -> RunResult:
        merged = tuple(dict.fromkeys((*self.compliance_flags, *flags)))
        return RunResult(
            optimizer_name=self.optimizer_name,
            implementation=self.implementation,
            best_x=self.best_x,
            best_f=self.best_f,
            regret_normalized=self.regret_normalized,
            n_evals_consumed=self.n_evals_consumed,
            overshoot_events=self.overshoot_events,
            non_finite_evals=self.non_finite_evals,
            runtime_s=self.runtime_s,
            termination_reason=self.termination_reason,
            compliance_flags=merged,
            history=self.history,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    """One search-stage (exploratory) record for one parameter vector."""

    theta: tuple[float, ...]
    effect: float
    lcb: float
    se: float
    median_regret_a: float
    median_regret_b: float
    probability_superiority: float
    nominated: bool


@dataclass(frozen=True)
class ReferenceDirection:
    """Aggregate gap direction over the neutral reference sample of theta."""

    sample_size: int
    median_gap: float
    sign: int

    def __post_init__(self) -> None:
        if self.sign not in (-1, 0, 1):
            raise ValueError("sign must be -1, 0 or 1")


@dataclass(frozen=True)
class SearchResult:
    """Exploratory output of the meta-search stage.

    This object must never be presented as evidence; it only nominates
    candidates for mandatory fresh-seed validation.
    """

    evaluations: tuple[CandidateEvaluation, ...]
    reference_direction: ReferenceDirection

    @property
    def label(self) -> str:
        return SEARCH_STAGE_LABEL

    @property
    def nominated(self) -> tuple[CandidateEvaluation, ...]:
        return tuple(e for e in self.evaluations if e.nominated)


@dataclass(frozen=True)
class CriteriaFlags:
    holm_significant: bool = False
    effect_threshold_met: bool = False
    ci_excludes_margin: bool = False
    reference_direction_opposed: bool = False

    def all_met(self) -> bool:
        return all(getattr(self, name) for name in CONFIRMATION_CRITERIA)


@dataclass(frozen=True)
class CandidateOutcome:
    """Validation-stage outcome for one nominated candidate."""

    theta: tuple[float, ...]
    dimension: int
    effect: float
    effect_ci_low: float
    effect_ci_high: float
    p_value_raw: float
    p_value_holm: float
    test_name: str
    n_validation_seeds: int
    median_regret_a: float
    median_regret_b: float
    median_gap: float
    median_gap_ci_low: float
    median_gap_ci_high: float
    probability_superiority: float
    criteria: CriteriaFlags = field(default_factory=CriteriaFlags)

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.criteria.holm_significant:
            reasons.append(
                f"Holm-corrected validation test not significant "
                f"(p={self.p_value_holm:.4g})"
            )
        if not self.criteria.effect_threshold_met:
            reasons.append(
                f"validation effect size {self.effect:.3g} below declared epsilon_min"
            )
        if not self.criteria.ci_excludes_margin:
            reasons.append(
                f"median-gap CI [{self.median_gap_ci_low:.3g}, {self.median_gap_ci_high:.3g}] "
                f"does not exclude the practical-equivalence margin"
            )
        if not self.criteria.reference_direction_opposed:
            reasons.append("gap direction does not oppose the neutral reference direction")
        return tuple(reasons)

    @property
    def confirmed(self) -> bool:
        return len(self.failure_reasons) == 0


@dataclass(frozen=True)
class ValidationResult:
    """Confirmatory outcome over all validated candidates.

    Failed validations are first-class members of this object and are rendered
    with equal prominence by reporting.
    """

    confirmed: tuple[CandidateOutcome, ...]
    failed: tuple[CandidateOutcome, ...]
    holm_family_alpha: float

    @property
    def candidates(self) -> tuple[CandidateOutcome, ...]:
        return (*self.confirmed, *self.failed)
