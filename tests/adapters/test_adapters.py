import numpy as np
import pytest

from optinemesis.adapters import build_optimizer, list_implementations
from optinemesis.adapters.builtin import SeededRandomSearch
from optinemesis.core import Bounds, BudgetExhausted, CountingObjective, ProblemInstance


def make_instance(evaluator=None) -> ProblemInstance:
    if evaluator is None:
        def evaluator(x: np.ndarray) -> float:
            return float(np.sum((x - 0.3) ** 2))

    return ProblemInstance(
        family_name="Test",
        family_version="1",
        theta=(0.0,),
        dimension=4,
        bounds=Bounds(lower=(-5.0,) * 4, upper=(5.0,) * 4),
        optimum_value=0.0,
        spec={},
        _evaluator=evaluator,
    )


ALL_IMPLEMENTATIONS = [
    ("builtin.random_search", {}),
    ("scipy.differential_evolution", {"popsize": 5}),
    ("scipy.l_bfgs_b", {}),
    ("scipy.nelder_mead", {}),
    ("scipy.powell", {}),
]


@pytest.mark.parametrize(("implementation", "config"), ALL_IMPLEMENTATIONS)
class TestAdapterContract:
    def test_exact_budget_or_recorded_early_convergence(
        self, implementation: str, config: dict
    ) -> None:
        optimizer = build_optimizer(implementation, config)
        objective = CountingObjective(make_instance(), budget=97)
        result = optimizer.run(objective, seed=11)
        assert 1 <= result.n_evals_consumed <= 97
        if result.n_evals_consumed < 97:
            assert result.termination_reason.startswith("backend_converged"), (
                f"{implementation} stopped early without recording convergence"
            )
        else:
            assert result.termination_reason == "budget_exhausted"

    def test_budget_one(self, implementation: str, config: dict) -> None:
        optimizer = build_optimizer(implementation, config)
        objective = CountingObjective(make_instance(), budget=1)
        result = optimizer.run(objective, seed=3)
        assert result.n_evals_consumed == 1
        assert result.n_evals_consumed <= 1

    def test_best_point_within_bounds(self, implementation: str, config: dict) -> None:
        optimizer = build_optimizer(implementation, config)
        objective = CountingObjective(make_instance(), budget=120)
        result = optimizer.run(objective, seed=5)
        assert result.best_x is not None
        assert objective.instance.bounds.contains(np.asarray(result.best_x))

    def test_history_is_anytime_monotone(self, implementation: str, config: dict) -> None:
        optimizer = build_optimizer(implementation, config)
        objective = CountingObjective(make_instance(), budget=80)
        result = optimizer.run(objective, seed=9)
        assert result.history is not None and len(result.history) >= 1
        evals = [h[0] for h in result.history]
        values = [h[1] for h in result.history]
        assert evals == sorted(evals)
        assert all(values[i] > values[i + 1] for i in range(len(values) - 1))
        assert min(values) == pytest.approx(result.best_f)

    def test_termination_and_metadata_shape(
        self, implementation: str, config: dict
    ) -> None:
        optimizer = build_optimizer(implementation, config)
        objective = CountingObjective(make_instance(), budget=60)
        result = optimizer.run(objective, seed=17)
        assert result.termination_reason
        assert result.metadata["seed"] == 17
        assert isinstance(result.runtime_s, float)


class TestBudgetAccounting:
    def test_random_search_never_attempts_overshoot(self) -> None:
        optimizer = SeededRandomSearch()
        objective = CountingObjective(make_instance(), budget=33)
        result = optimizer.run(objective, seed=0)
        assert result.overshoot_events == 0
        assert result.n_evals_consumed == 33
        assert result.termination_reason == "budget_exhausted"

    def test_objective_refuses_past_budget(self) -> None:
        objective = CountingObjective(make_instance(), budget=2)
        objective(np.zeros(4))
        objective(np.zeros(4))
        with pytest.raises(BudgetExhausted):
            objective(np.zeros(4))

    def test_scipy_backend_attempt_records_overshoot_without_consuming(self) -> None:
        optimizer = build_optimizer("scipy.l_bfgs_b", {})
        objective = CountingObjective(make_instance(), budget=7)
        result = optimizer.run(objective, seed=1)
        assert result.n_evals_consumed == 7
        assert (
            result.overshoot_events >= 1
        ), "L-BFGS-B should have attempted at least one evaluation past the cap"
        assert "budget_overshoot_attempted" in result.compliance_flags


class TestDeterminism:
    def test_random_search_same_seed_identical(self) -> None:
        a = SeededRandomSearch().run(CountingObjective(make_instance(), budget=50), seed=42)
        b = SeededRandomSearch().run(CountingObjective(make_instance(), budget=50), seed=42)
        assert a.best_f == b.best_f
        np.testing.assert_array_equal(a.best_x, b.best_x)
        assert a.history == b.history

    def test_de_same_seed_identical(self) -> None:
        def run() -> object:
            opt = build_optimizer("scipy.differential_evolution", {"popsize": 4})
            return opt.run(CountingObjective(make_instance(), budget=150), seed=99)

        a, b = run(), run()
        assert a.best_f == b.best_f
        np.testing.assert_array_equal(a.best_x, b.best_x)

    def test_different_seeds_differ_on_average(self) -> None:
        values = [
            SeededRandomSearch({"batch_size": 1})
            .run(CountingObjective(make_instance(), budget=10), seed=s)
            .best_f
            for s in range(6)
        ]
        assert len(set(values)) > 1


class TestPathologicalObjectives:
    def test_all_nan_objective(self) -> None:
        def nan_obj(x: np.ndarray) -> float:
            return float("nan")

        for implementation, config in ALL_IMPLEMENTATIONS:
            optimizer = build_optimizer(implementation, config)
            objective = CountingObjective(make_instance(nan_obj), budget=40)
            result = optimizer.run(objective, seed=21)
            assert result.n_evals_consumed == 40
            assert result.non_finite_evals == 40
            assert "non_finite_evals" in result.compliance_flags
            if result.best_f is not None:
                assert not np.isfinite(result.best_f)

    def test_raising_objective_propagates(self) -> None:
        def boom(x: np.ndarray) -> float:
            raise RuntimeError("objective exploded")

        optimizer = SeededRandomSearch()
        objective = CountingObjective(make_instance(boom), budget=10)
        with pytest.raises(RuntimeError):
            optimizer.run(objective, seed=0)


def test_unknown_implementation_rejected() -> None:
    from optinemesis.core import AdapterError

    with pytest.raises(AdapterError):
        build_optimizer("no.such.optimizer")


def test_builtin_listed() -> None:
    impls = list_implementations()
    assert "builtin.random_search" in impls
    assert "scipy.differential_evolution" in impls


def test_duplicate_factory_registration_rejected() -> None:
    from optinemesis.adapters.protocol import register_optimizer_factory
    from optinemesis.core import AdapterError

    with pytest.raises(AdapterError):
        @register_optimizer_factory("builtin.random_search")
        def _dup(config: dict) -> SeededRandomSearch:  # pragma: no cover
            return SeededRandomSearch(config)
