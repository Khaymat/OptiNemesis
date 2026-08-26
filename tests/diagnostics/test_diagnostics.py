
from optinemesis.core import Bounds, OptimizerSpec, SeedPlan, StudySpec, Thresholds
from optinemesis.diagnostics import ablate_candidate, sensitivity_marginals
from optinemesis.families import get_family
from optinemesis.search import run_search


def make_study() -> StudySpec:
    return StudySpec(
        family=get_family("Ellipsoidal").family_ref(),
        dimension=3,
        bounds=Bounds(lower=(-5.0, -5.0, -5.0), upper=(5.0, 5.0, 5.0)),
        budget=120,
        optimizers=(
            OptimizerSpec(name="rs", implementation="builtin.random_search"),
            OptimizerSpec(
                name="de",
                implementation="scipy.differential_evolution",
                config={"popsize": 4},
            ),
        ),
        seeds=SeedPlan(
            root_entropy="77",
            n_theta_candidates=6,
            n_search_seeds=3,
            n_validation_seeds=5,
        ),
        thresholds=Thresholds(bootstrap_resamples=200),
    )


def test_sensitivity_marginals_shape() -> None:
    study = make_study()
    search = run_search(study)
    marginals = sensitivity_marginals(search, study)
    assert set(marginals) == {"conditioning", "rotation_strength", "shift_radius"}
    for values in marginals.values():
        assert len(values) == len(search.evaluations)


def test_ablation_returns_baseline_and_deltas() -> None:
    study = make_study()
    search = run_search(study)
    candidate = max(search.evaluations, key=lambda e: abs(e.effect)).theta
    ablation = ablate_candidate(study, candidate)
    assert "baseline" in ablation
    assert ablation["baseline"]["delta"] == 0.0
    for param in ("conditioning", "rotation_strength", "shift_radius"):
        assert param in ablation
        assert isinstance(ablation[param]["effect"], float)


def test_ablation_is_deterministic() -> None:
    study = make_study()
    theta = study.family.center_theta()
    first = ablate_candidate(study, theta)
    second = ablate_candidate(study, theta)
    assert first == second
