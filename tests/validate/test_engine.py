

from optinemesis.core import Bounds, OptimizerSpec, SeedPlan, StudySpec
from optinemesis.families import get_family
from optinemesis.search import run_search
from optinemesis.validate import validate_candidates


def make_study(root_entropy: int = 31415) -> StudySpec:
    family = get_family("Ellipsoidal", "1")
    return StudySpec(
        family=family.family_ref(),
        dimension=3,
        bounds=Bounds(lower=(-5.0, -5.0, -5.0), upper=(5.0, 5.0, 5.0)),
        budget=150,
        optimizers=(
            OptimizerSpec(name="rand", implementation="builtin.random_search"),
            OptimizerSpec(
                name="de",
                implementation="scipy.differential_evolution",
                config={"popsize": 4},
            ),
        ),
        seeds=SeedPlan(
            root_entropy=str(root_entropy),
            n_theta_candidates=6,
            n_search_seeds=3,
            n_validation_seeds=8,
            meta_search_method="random",
        ),
    )


def test_empty_nomination_yields_empty_validation() -> None:
    study = make_study()
    search = run_search(study)
    validation = validate_candidates(search, study)
    assert isinstance(validation.confirmed, tuple)
    total = len(validation.confirmed) + len(validation.failed)
    assert total == len(search.nominated)


def test_validation_uses_only_fresh_seeds() -> None:
    from optinemesis.core.seeds import SeedTree

    study = make_study()
    tree = SeedTree.create(int(study.seeds.root_entropy))
    n = study.seeds.n_search_seeds
    m = study.seeds.n_validation_seeds
    search_seeds = set(tree.problem_search.seeds(n))
    val_subtrees = tree.validation_subtrees()
    val_seeds = set(val_subtrees["problem_val"].seeds(m))
    assert not (search_seeds & val_seeds)
    assert not (set(tree.opt_a.seeds(n)) & set(val_subtrees["opt_a_val"].seeds(m)))
    assert not (set(tree.opt_b.seeds(n)) & set(val_subtrees["opt_b_val"].seeds(m)))


def test_failed_validation_is_first_class() -> None:
    study = make_study(root_entropy=271828)
    search = run_search(study)
    validation = validate_candidates(search, study)
    for outcome in validation.failed:
        assert outcome.failure_reasons
        assert not outcome.confirmed
    for outcome in validation.confirmed:
        assert outcome.criteria.all_met()
        assert outcome.failure_reasons == ()
        assert outcome.p_value_holm >= outcome.p_value_raw


def test_deterministic_validation() -> None:
    study = make_study()
    v1 = validate_candidates(run_search(study), study)
    v2 = validate_candidates(run_search(study), study)
    assert v1.confirmed == v2.confirmed
    assert v1.failed == v2.failed


def test_candidate_cap_constant_documented() -> None:
    from optinemesis.validate.engine import MAX_VALIDATED_CANDIDATES

    assert MAX_VALIDATED_CANDIDATES == 5
