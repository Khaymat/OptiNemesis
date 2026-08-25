"""Execution of contract-bound runs and paired evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optinemesis.adapters.protocol import build_optimizer
from optinemesis.core.contract import RHO_MAX, Contract, OptimizerSpec
from optinemesis.core.objective import CountingObjective
from optinemesis.core.results import RunResult
from optinemesis.families.registry import get_family


@dataclass(frozen=True)
class PairedOutcome:
    """Regret samples for both configurations on one instance."""

    instance_seed: int
    regrets_a: tuple[float, ...]
    regrets_b: tuple[float, ...]


def normalized_regret(
    best_f: float | None,
    optimum_value: float,
    regret_scale: float,
) -> float:
    """Clip-normalized regret; a non-finite final value maps to RHO_MAX."""
    if best_f is None or not np.isfinite(best_f):
        return RHO_MAX
    raw = (float(best_f) - float(optimum_value)) / float(regret_scale)
    if not np.isfinite(raw):
        return RHO_MAX
    return float(np.clip(raw, 0.0, RHO_MAX))


def execute_run(
    contract: Contract,
    optimizer_spec: OptimizerSpec,
    optimizer_seed: int,
    instance_seed: int,
) -> RunResult:
    """Run one configuration once on one seeded instance under the contract budget."""
    family = get_family(contract.family.name, contract.family.version)
    instance = family.sample(
        theta=contract.theta,
        dimension=contract.dimension,
        instance_seed=instance_seed,
        bounds=contract.bounds,
    )
    objective = CountingObjective(instance, budget=contract.budget)
    optimizer = build_optimizer(optimizer_spec.implementation, optimizer_spec.config)
    result = optimizer.run(objective, seed=optimizer_seed)

    scale = float(instance.landscape_tags.get("regret_scale", 1.0))
    regret = normalized_regret(result.best_f, instance.optimum_value, scale)
    flags = list(result.compliance_flags)
    if result.best_f is None or not np.isfinite(result.best_f):
        flags.append("no_finite_solution")
    if result.n_evals_consumed != objective.consumed:
        raise RuntimeError("adapter returned inconsistent evaluation count")
    if result.n_evals_consumed > contract.budget:
        raise RuntimeError("budget exceeded: this is a harness bug")

    return RunResult(
        optimizer_name=optimizer_spec.name,
        implementation=result.implementation,
        best_x=result.best_x,
        best_f=result.best_f,
        regret_normalized=regret,
        n_evals_consumed=result.n_evals_consumed,
        overshoot_events=result.overshoot_events,
        non_finite_evals=result.non_finite_evals,
        runtime_s=result.runtime_s,
        termination_reason=result.termination_reason,
        compliance_flags=tuple(dict.fromkeys(flags)),
        history=result.history,
        metadata=result.metadata,
    )


def execute_paired_instance(
    contract: Contract,
    instance_seed: int,
    seed_a: int,
    seed_b: int,
) -> PairedOutcome:
    """Both configurations on the identical problem instance."""
    spec_a, spec_b = contract.optimizers
    run_a = execute_run(contract, spec_a, seed_a, instance_seed)
    run_b = execute_run(contract, spec_b, seed_b, instance_seed)
    assert run_a.regret_normalized is not None
    assert run_b.regret_normalized is not None
    return PairedOutcome(
        instance_seed=instance_seed,
        regrets_a=(run_a.regret_normalized,),
        regrets_b=(run_b.regret_normalized,),
    )
