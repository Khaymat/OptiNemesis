"""Built-in baseline optimizer: seeded uniform random search."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from optinemesis.adapters.protocol import Optimizer, register_optimizer_factory
from optinemesis.adapters.recorder import RecordingObjective
from optinemesis.core.objective import CountingObjective
from optinemesis.core.results import RunResult


class SeededRandomSearch:
    """Uniform random search over the box; consumes the budget exactly.

    This baseline never attempts an evaluation past the budget (it checks the
    remaining count first), so it produces zero overshoot events by
    construction. It serves both as a sanity baseline and as the deterministic
    keystone test fixture.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.batch_size = int(config.get("batch_size", 1))
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def run(self, objective: CountingObjective, seed: int) -> RunResult:
        started = time.perf_counter()
        recorder = RecordingObjective(objective)
        instance = objective.instance
        lo, hi = instance.bounds.as_arrays()
        rng = np.random.default_rng(seed)

        while objective.remaining > 0:
            take = min(self.batch_size, objective.remaining)
            for _ in range(take):
                x = rng.uniform(lo, hi)
                recorder(x)

        runtime = time.perf_counter() - started
        return RunResult(
            optimizer_name="",
            implementation="builtin.random_search",
            best_x=recorder.best_x,
            best_f=recorder.best_f,
            n_evals_consumed=objective.consumed,
            overshoot_events=recorder.overshoot_events,
            non_finite_evals=objective.non_finite,
            runtime_s=runtime,
            termination_reason=(
                "budget_exhausted"
                if recorder.terminated_by_budget
                else "internal_error_stopped_early"
            ),
            compliance_flags=("non_finite_evals",) if objective.non_finite else (),
            history=tuple(recorder.history),
            metadata={"seed": seed, "seeded": True},
        )


@register_optimizer_factory("builtin.random_search")
def _factory(config: dict[str, Any]) -> Optimizer:
    return SeededRandomSearch(config)
