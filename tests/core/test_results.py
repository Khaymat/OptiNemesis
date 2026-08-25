import numpy as np
import pytest

from optinemesis.core import (
    SEARCH_STAGE_LABEL,
    CandidateEvaluation,
    CandidateOutcome,
    CriteriaFlags,
    ReferenceDirection,
    RunResult,
    SearchResult,
    ValidationResult,
)


def make_run(**overrides: object) -> RunResult:
    fields = {
        "optimizer_name": "a",
        "implementation": "builtin.random_search",
        "best_x": np.array([0.1, 0.2]),
        "best_f": 0.05,
        "regret_normalized": 0.01,
        "n_evals_consumed": 100,
        "overshoot_events": 0,
        "non_finite_evals": 0,
        "runtime_s": 0.25,
        "termination_reason": "budget_exhausted",
        "compliance_flags": (),
        "history": ((1, 3.0), (10, 1.5), (50, 0.4)),
        "metadata": {"seed": 7},
    }
    fields.update(overrides)
    return RunResult(**fields)  # type: ignore[arg-type]


def test_runresult_with_flags_deduplicates_and_preserves() -> None:
    r = make_run()
    r2 = r.with_flags("init_deviation", "init_deviation")
    assert r2.compliance_flags == ("init_deviation",)
    assert r.compliance_flags == ()
    r3 = r.with_flags("x", "y").with_flags("y", "z")
    assert r3.compliance_flags == ("x", "y", "z")


def test_runresult_negative_counts_rejected() -> None:
    with pytest.raises(ValueError):
        make_run(n_evals_consumed=-1)
    with pytest.raises(ValueError):
        make_run(overshoot_events=-2)


def test_search_result_is_exploratory_only() -> None:
    evals = (
        CandidateEvaluation(
            theta=(0.5, 0.5),
            effect=0.6,
            lcb=0.4,
            se=0.1,
            median_regret_a=0.9,
            median_regret_b=0.3,
            probability_superiority=0.8,
            nominated=True,
        ),
        CandidateEvaluation(
            theta=(0.1, 0.9),
            effect=0.1,
            lcb=-0.05,
            se=0.08,
            median_regret_a=0.5,
            median_regret_b=0.45,
            probability_superiority=0.55,
            nominated=False,
        ),
    )
    sr = SearchResult(
        evaluations=evals,
        reference_direction=ReferenceDirection(sample_size=20, median_gap=-0.2, sign=-1),
    )
    assert sr.label == SEARCH_STAGE_LABEL
    assert len(sr.nominated) == 1
    assert not hasattr(sr, "p_value_holm")


def test_reference_direction_sign_constrained() -> None:
    with pytest.raises(ValueError):
        ReferenceDirection(sample_size=10, median_gap=0.5, sign=2)


def make_outcome(criteria: CriteriaFlags, **overrides: object) -> CandidateOutcome:
    fields = {
        "theta": (0.5,),
        "dimension": 10,
        "effect": 0.55,
        "effect_ci_low": 0.35,
        "effect_ci_high": 0.7,
        "p_value_raw": 0.001,
        "p_value_holm": 0.002,
        "test_name": "wilcoxon_signed_rank",
        "n_validation_seeds": 100,
        "median_regret_a": 0.9,
        "median_regret_b": 0.35,
        "median_gap": 0.55,
        "median_gap_ci_low": 0.3,
        "median_gap_ci_high": 0.8,
        "probability_superiority": 0.77,
        "criteria": criteria,
    }
    fields.update(overrides)
    return CandidateOutcome(**fields)  # type: ignore[arg-type]


def test_confirmed_outcome_requires_all_criteria() -> None:
    full = CriteriaFlags(True, True, True, True)
    outcome = make_outcome(full)
    assert outcome.confirmed
    assert outcome.failure_reasons == ()

    partial = CriteriaFlags(True, True, False, True)
    bad = make_outcome(partial)
    assert not bad.confirmed
    assert any("practical-equivalence" in reason for reason in bad.failure_reasons)


def test_failed_validation_is_first_class() -> None:
    good = make_outcome(CriteriaFlags(True, True, True, True))
    bad = make_outcome(CriteriaFlags(False, True, True, True), theta=(0.2,))
    vr = ValidationResult(confirmed=(good,), failed=(bad,), holm_family_alpha=0.05)
    assert vr.candidates == (good, bad)
    assert not bad.confirmed
    assert any("Holm-corrected" in r for r in bad.failure_reasons)
