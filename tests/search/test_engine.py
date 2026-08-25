import dataclasses

from optinemesis.adapters.protocol import build_optimizer
from optinemesis.core import Bounds, OptimizerSpec, SeedPlan, StudySpec, Thresholds
from optinemesis.families import get_family
from optinemesis.search import run_search


def make_study(root_entropy: int = 2025, **overrides: object) -> StudySpec:
    family = get_family("Ellipsoidal", "1")
    fields = {
        "family": family.family_ref(),
        "dimension": 3,
        "bounds": Bounds(lower=(-5.0, -5.0, -5.0), upper=(5.0, 5.0, 5.0)),
        "budget": 150,
        "optimizers": (
            OptimizerSpec(name="rand", implementation="builtin.random_search"),
            OptimizerSpec(
                name="de",
                implementation="scipy.differential_evolution",
                config={"popsize": 4},
            ),
        ),
        "seeds": SeedPlan(
            root_entropy=str(root_entropy),
            n_theta_candidates=4,
            n_search_seeds=3,
            n_validation_seeds=5,
            meta_search_method="random",
        ),
        "thresholds": Thresholds(bootstrap_resamples=200),
    }
    fields.update(overrides)
    return StudySpec(**fields)  # type: ignore[arg-type]


def test_run_search_returns_structured_result() -> None:
    study = make_study()
    result = run_search(study)
    assert len(result.evaluations) == study.seeds.n_theta_candidates
    assert result.reference_direction.sample_size == len(result.evaluations)
    assert result.reference_direction.sign in (-1, 0, 1)
    nominations = [e.nominated for e in result.evaluations]
    assert sum(nominations) == len(result.nominated)


def test_run_search_is_deterministic() -> None:
    r1 = run_search(make_study())
    r2 = run_search(make_study())
    assert r1.evaluations == r2.evaluations
    assert r1.reference_direction == r2.reference_direction


def test_different_roots_change_exploration() -> None:
    r1 = run_search(make_study(root_entropy=2025))
    r2 = run_search(make_study(root_entropy=777))
    assert r1.evaluations != r2.evaluations


def test_nomination_respects_lcb_threshold() -> None:
    study = make_study()
    result = run_search(study)
    epsilon_min = study.thresholds.epsilon_min
    nominated = result.nominated
    for candidate in nominated:
        assert candidate.lcb >= epsilon_min
    if nominated:
        best_lcb = max(e.lcb for e in result.evaluations)
        assert nominated[0].lcb == best_lcb or len(nominated) > 1


def test_lhs_method_changes_samples() -> None:
    study_random = make_study()
    study_lhs = make_study(
        seeds=dataclasses.replace(study_random.seeds, meta_search_method="lhs")
    )
    r_rand = run_search(study_random)
    r_lhs = run_search(study_lhs)
    assert r_rand.evaluations != r_lhs.evaluations


def test_de_adapter_usable_through_engine() -> None:
    optimizer = build_optimizer("scipy.differential_evolution", {"popsize": 4})
    assert optimizer is not None
