import numpy as np
import pytest

from optinemesis.core import (
    Bounds,
    BudgetExhausted,
    CountingObjective,
    ProblemInstance,
)


def make_instance() -> ProblemInstance:
    def sphere(x: np.ndarray) -> float:
        return float(np.dot(x, x)) - 3.0

    return ProblemInstance(
        family_name="Test",
        family_version="1",
        theta=(0.0,),
        dimension=3,
        bounds=Bounds(lower=(-10.0,) * 3, upper=(10.0,) * 3),
        optimum_value=-3.0,
        landscape_tags={"modality": 1.0},
        spec={"kind": "test-sphere"},
        _evaluator=sphere,
    )


def test_exact_budget_enforcement() -> None:
    obj = CountingObjective(make_instance(), budget=5)
    for i in range(5):
        obj(np.array([float(i), 0.0, 0.0]))
        assert obj.consumed == i + 1
    with pytest.raises(BudgetExhausted) as excinfo:
        obj(np.zeros(3))
    assert excinfo.value.budget == 5
    assert excinfo.value.consumed == 5
    assert obj.remaining == 0


def test_non_finite_counted_and_reported_as_inf() -> None:
    def flaky(x: np.ndarray) -> float:
        return float("nan") if x[0] > 5 else float(np.dot(x, x))

    inst = ProblemInstance(
        family_name="Test",
        family_version="1",
        theta=(0.0,),
        dimension=1,
        bounds=Bounds(lower=(-10.0,), upper=(10.0,)),
        optimum_value=0.0,
        spec={},
        _evaluator=flaky,
    )
    obj = CountingObjective(inst, budget=10)
    v_finite = obj(np.array([1.0]))
    v_nan = obj(np.array([9.0]))
    assert np.isfinite(v_finite)
    assert v_nan == float(np.inf)
    assert obj.non_finite == 1
    assert obj.consumed == 2


def test_dimension_mismatch_rejected() -> None:
    obj = CountingObjective(make_instance(), budget=3)
    with pytest.raises(ValueError):
        obj(np.zeros(2))


def test_zero_or_negative_budget_rejected() -> None:
    with pytest.raises(ValueError):
        CountingObjective(make_instance(), budget=0)


def test_spec_roundtrip_fields() -> None:
    inst = make_instance()
    spec = inst.to_spec()
    assert spec["family_name"] == "Test"
    assert Bounds.from_spec(spec["bounds"]) == inst.bounds
    assert spec["spec"]["kind"] == "test-sphere"


def test_evaluator_purity_assumption_documented() -> None:
    calls: list[np.ndarray] = []

    def recorder(x: np.ndarray) -> float:
        calls.append(np.asarray(x).copy())
        return 1.0

    inst = ProblemInstance(
        family_name="T",
        family_version="1",
        theta=(),
        dimension=1,
        bounds=Bounds(lower=(-1.0,), upper=(1.0,)),
        optimum_value=0.0,
        spec={},
        _evaluator=recorder,
    )
    inst.evaluate(np.array([0.5]))
    inst.evaluate(np.array([0.5]))
    assert len(calls) == 2
