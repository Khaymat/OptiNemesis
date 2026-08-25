import dataclasses

import pytest

from optinemesis.core import (
    Bounds,
    Contract,
    ContractError,
    EnvironmentInfo,
    FamilyRef,
    MetricPlan,
    OptimizerSpec,
    ParamRange,
    SeedPlan,
    StudySpec,
    Thresholds,
)


def make_study(**overrides: object) -> StudySpec:
    fields = {
        "family": FamilyRef(
            name="Ellipsoidal",
            version="1",
            parameter_box=(
                ParamRange("conditioning", 10.0, 1000.0),
                ParamRange("rotation_strength", 0.0, 1.0),
                ParamRange("shift_radius", 0.1, 0.9),
            ),
            transform_tags=("conditioning", "rotation", "shift"),
        ),
        "dimension": 5,
        "bounds": Bounds(lower=tuple(-5.0 for _ in range(5)), upper=tuple(5.0 for _ in range(5))),
        "budget": 500,
        "optimizers": (
            OptimizerSpec(name="a", implementation="builtin.random_search"),
            OptimizerSpec(name="b", implementation="scipy.differential_evolution"),
        ),
        "seeds": SeedPlan(root_entropy="12345"),
        "metrics": MetricPlan(),
        "thresholds": Thresholds(),
        "initialization_policy": "independent",
        "environment": EnvironmentInfo.capture(),
    }
    fields.update(overrides)  # type: ignore[arg-type]
    return StudySpec(**fields)  # type: ignore[arg-type]


def test_bind_produces_contract_with_checked_theta() -> None:
    study = make_study()
    contract = study.bind((100.0, 0.5, 0.5))
    assert isinstance(contract, Contract)
    assert contract.theta == (100.0, 0.5, 0.5)
    assert contract.budget == 500


def test_bind_rejects_out_of_box_theta() -> None:
    with pytest.raises(ContractError):
        make_study().bind((5.0, 0.5, 0.5))
    with pytest.raises(ContractError):
        make_study().bind((100.0, 0.5))


def test_contract_id_is_stable_and_sensitive() -> None:
    study = make_study()
    c1 = study.bind((100.0, 0.5, 0.5))
    c2 = make_study().bind((100.0, 0.5, 0.5))
    assert c1.contract_id() == c2.contract_id()
    c3 = study.bind((200.0, 0.5, 0.5))
    assert c1.contract_id() != c3.contract_id()
    c4 = dataclasses.replace(study, budget=501).bind((100.0, 0.5, 0.5))
    assert c1.contract_id() != c4.contract_id()


def test_to_dict_is_json_clean() -> None:
    d = make_study().bind((100.0, 0.5, 0.5)).to_dict()
    assert d["schema_version"] == "1"
    assert d["family"]["name"] == "Ellipsoidal"
    assert len(d["optimizers"]) == 2
    assert d["contract_id"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"dimension": 0},
        {"dimension": True},
        {"budget": 0},
        {"budget": -5},
        {"initialization_policy": "chaotic"},
    ],
)
def test_malformed_studies_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        make_study(**overrides)


def test_optimizer_rules_enforced() -> None:
    same = (
        OptimizerSpec(name="x", implementation="builtin.random_search"),
        OptimizerSpec(name="x", implementation="scipy.differential_evolution"),
    )
    with pytest.raises(ContractError):
        make_study(optimizers=same)
    single = (
        OptimizerSpec(name="x", implementation="builtin.random_search"),
        OptimizerSpec(name="y", implementation="builtin.random_search"),
        OptimizerSpec(name="z", implementation="builtin.random_search"),
    )
    with pytest.raises(ContractError):
        make_study(optimizers=single)  # type: ignore[arg-type]


def test_seed_plan_minimums() -> None:
    with pytest.raises(ContractError):
        SeedPlan(root_entropy="1", n_search_seeds=1)
    with pytest.raises(ContractError):
        SeedPlan(root_entropy="1", n_validation_seeds=1)
    with pytest.raises(ContractError):
        SeedPlan(root_entropy="1", meta_search_method="cmaes")


def test_metric_plan_rules() -> None:
    with pytest.raises(ContractError):
        MetricPlan(primary="mean_gap")
    with pytest.raises(ContractError):
        MetricPlan(primary="rank_biserial", exploratory=("rank_biserial",))
    with pytest.raises(ContractError):
        MetricPlan(exploratory=("nope",))


def test_thresholds_rules() -> None:
    with pytest.raises(ContractError):
        Thresholds(epsilon_min=0.0)
    with pytest.raises(ContractError):
        Thresholds(alpha_validation=1.5)
    with pytest.raises(ContractError):
        Thresholds(bootstrap_resamples=10)
