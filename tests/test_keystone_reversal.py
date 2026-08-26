"""Phase 5 keystone: deterministic ranking-reversal pipeline test.

This is the single most important test in the repository. It exercises:

    problem generation -> optimizer execution -> budget accounting ->
    meta-search -> candidate nomination -> fresh-seed validation ->
    artifact serialization -> conservative report rendering

with NO dependence on stochastic luck: the reversal is structural
(see tests/_keystone.py KeystoneFlip family).

- flip < 0.3  (smooth Ellipsoidal): L-BFGS-B solves to ~0, random_search
  stalls at its sampling floor -> random worse (positive effect for
  configuration A=random_search).
- flip >= 0.3 (rugged Rastrigin): random_search best-of-N beats L-BFGS-B
  trapped in a ripple basin -> random better (negative effect).

The threshold at 0.3 makes the rugged regime dominate the neutral median
(sign -1), so the smooth-regime nominated candidates oppose the reference
direction and survive fresh-seed validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optinemesis.core import Bounds, OptimizerSpec, SeedPlan, StudySpec, Thresholds
from optinemesis.families import get_family
from optinemesis.reporting import (
    load_artifact,
    load_study_from_artifact,
    render_markdown,
    save_artifact,
)
from optinemesis.search import run_search
from optinemesis.validate import validate_candidates


def make_keystone_study(root_entropy: int = 1953) -> StudySpec:
    family = get_family("KeystoneFlip", "1")
    return StudySpec(
        family=family.family_ref(),
        dimension=2,
        bounds=Bounds(lower=(-5.0, -5.0), upper=(5.0, 5.0)),
        budget=300,
        optimizers=(
            OptimizerSpec(name="rs", implementation="builtin.random_search"),
            OptimizerSpec(name="lbfgs", implementation="scipy.l_bfgs_b"),
        ),
        seeds=SeedPlan(
            root_entropy=str(root_entropy),
            n_theta_candidates=20,
            n_search_seeds=4,
            n_validation_seeds=16,
            meta_search_method="lhs",
        ),
        thresholds=Thresholds(
            epsilon_min=0.33,
            epsilon_zero=0.0,
            alpha_validation=0.05,
            bootstrap_resamples=300,
        ),
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    study = make_keystone_study()
    search = run_search(study)
    validation = validate_candidates(search, study)
    path = tmp_path_factory.mktemp("keystone") / "keystone.nemesis.json"
    save_artifact(study, search, validation, path)
    return study, search, validation, Path(path)


def test_raw_reversal_facts_present(pipeline) -> None:  # type: ignore[no-untyped-def]
    _, search, _, _ = pipeline
    smooth = [e for e in search.evaluations if e.theta[0] < 0.3]
    rugged = [e for e in search.evaluations if e.theta[0] >= 0.3]

    assert smooth, "LHS sample must include smooth-regime candidates (flip < 0.3)"
    assert rugged, "LHS sample must include rugged-regime candidates"

    best_smooth = max(e.effect for e in smooth)
    worst_rugged = min(e.effect for e in rugged)
    assert best_smooth > 0.5, (
        f"random should be worse than L-BFGS on smooth instances, got {best_smooth}"
    )
    assert worst_rugged < -0.3, (
        f"random should be better than trapped L-BFGS on rugged instances, got {worst_rugged}"
    )
    assert best_smooth - worst_rugged > 0.8, "reversal margin must be wide"


def test_reference_direction_is_negative(pipeline) -> None:  # type: ignore[no-untyped-def]
    _, search, _, _ = pipeline
    assert search.reference_direction.sign == -1, (
        "rugged regime must dominate the neutral-box median (sign -1); "
        "otherwise the nominated smooth-regime candidates cannot oppose it"
    )


def test_nomination_targets_the_reversal_region(pipeline) -> None:  # type: ignore[no-untyped-def]
    _, search, _, _ = pipeline
    nominated = search.nominated
    assert nominated, "keystone must nominate at least one smooth-regime candidate"
    for candidate in nominated:
        assert candidate.lcb >= make_keystone_study().thresholds.epsilon_min
        assert candidate.theta[0] < 0.3, (
            "nominated candidates must lie in the smooth regime (flip < 0.3)"
        )


def test_validation_confirms_reversal_on_fresh_seeds(
    pipeline,
) -> None:  # type: ignore[no-untyped-def]
    _, _search, validation, _ = pipeline
    assert validation.confirmed, "keystone reversal must survive fresh-seed validation"
    for outcome in validation.confirmed:
        assert outcome.criteria.all_met()
        assert outcome.p_value_holm < make_keystone_study().thresholds.alpha_validation
        assert outcome.n_validation_seeds == 16
        assert outcome.median_regret_a > outcome.median_regret_b


def test_budget_accounting_held_throughout(pipeline) -> None:  # type: ignore[no-untyped-def]
    from optinemesis.runners.execution import execute_run

    study, _, _, _ = pipeline
    contract = study.bind(study.family.center_theta())
    for spec in contract.optimizers:
        run = execute_run(contract, spec, optimizer_seed=7, instance_seed=11)
        assert run.n_evals_consumed <= study.budget
        assert run.regret_normalized is not None
        assert 0.0 <= run.regret_normalized <= 1.0e6


def test_artifact_roundtrip_and_reconstruction(
    pipeline, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    study, _, validation, path = pipeline
    data = load_artifact(path)
    rebuilt_study = load_study_from_artifact(data)
    assert rebuilt_study.seeds.root_entropy == study.seeds.root_entropy
    confirmed_thetas = [list(c.theta) for c in validation.confirmed]
    stored_thetas = [list(c["theta"]) for c in data["validation_stage"]["confirmed"]]
    assert confirmed_thetas == stored_thetas


def test_report_renders_conservative_findings(pipeline) -> None:  # type: ignore[no-untyped-def]
    study, search, validation, path = pipeline
    load_artifact(path)
    report = render_markdown(study, search, validation)
    assert "underperformed configuration" in report
    assert "not confirmatory evidence" in report
    assert "fresh" in report and "disjoint" in report
    assert "better than optimizer" not in report


def test_full_pipeline_deterministic() -> None:
    s1 = run_search(make_keystone_study())
    s2 = run_search(make_keystone_study())
    assert s1.evaluations == s2.evaluations
