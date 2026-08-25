"""SciPy optimizer adapters.

All adapters route every evaluation through the harness-owned counting
objective, record best-so-far state in a shared recorder, and therefore
respect the evaluation budget exactly even when the backend attempts an
evaluation past the cap (recorded as ``overshoot_events``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

from optinemesis.adapters.protocol import Optimizer, register_optimizer_factory
from optinemesis.adapters.recorder import RecordingObjective
from optinemesis.core.errors import AdapterError, BudgetExhausted
from optinemesis.core.objective import CountingObjective
from optinemesis.core.results import RunResult


def _flags(recorder: RecordingObjective, objective: CountingObjective) -> tuple[str, ...]:
    flags: list[str] = []
    if recorder.overshoot_events > 0:
        flags.append("budget_overshoot_attempted")
    if objective.non_finite > 0:
        flags.append("non_finite_evals")
    return tuple(flags)


def _finish(
    implementation: str,
    recorder: RecordingObjective,
    objective: CountingObjective,
    started: float,
    seed: int,
    backend_message: str | None,
) -> RunResult:
    runtime = time.perf_counter() - started
    if recorder.terminated_by_budget:
        reason = "budget_exhausted"
    elif backend_message:
        reason = f"backend_converged: {backend_message}"
    else:
        reason = "backend_stopped"
    return RunResult(
        optimizer_name="",
        implementation=implementation,
        best_x=recorder.best_x,
        best_f=recorder.best_f,
        n_evals_consumed=objective.consumed,
        overshoot_events=recorder.overshoot_events,
        non_finite_evals=objective.non_finite,
        runtime_s=runtime,
        termination_reason=reason,
        compliance_flags=_flags(recorder, objective),
        history=tuple(recorder.history),
        metadata={"seed": seed, "seeded": True},
    )


def _seeded_x0(instance: Any, seed: int) -> np.ndarray:
    lo, hi = instance.bounds.as_arrays()
    return np.random.default_rng(seed).uniform(lo, hi)


class _ScipyAdapter(Optimizer):
    method: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def _minimize(
        self,
        recorder: RecordingObjective,
        x0: np.ndarray,
        bounds_pairs: Any,
        seed: int,
    ):
        raise NotImplementedError

    def run(self, objective: CountingObjective, seed: int) -> RunResult:
        if not self.method:
            raise AdapterError("_ScipyAdapter subclass must define method")
        started = time.perf_counter()
        recorder = RecordingObjective(objective)
        instance = objective.instance
        lo, hi = instance.bounds.as_arrays()
        bounds_pairs = list(zip(lo.tolist(), hi.tolist(), strict=True))
        x0 = _seeded_x0(instance, seed)
        message: str | None = None
        try:
            result = self._minimize(recorder, x0, bounds_pairs, seed)
            message = getattr(result, "message", None)
            if isinstance(message, bytes):
                message = message.decode()
        except BudgetExhausted:
            pass
        return _finish(
            self.method,
            recorder,
            objective,
            started,
            seed,
            str(message) if message else None,
        )


@register_optimizer_factory("scipy.differential_evolution")
class ScipyDifferentialEvolution(_ScipyAdapter):
    method = "scipy.differential_evolution"

    def _minimize(
        self,
        recorder: RecordingObjective,
        x0: np.ndarray,
        bounds_pairs: Any,
        seed: int,
    ):
        dimension = recorder.inner.instance.dimension
        budget = recorder.inner.budget
        popsize = int(self.config.get("popsize", 15))
        cfg = {
            "tol": self.config.get("tol", 0.0),
            "atol": self.config.get("atol", 0.0),
            "maxiter": int(
                self.config.get("maxiter", budget // max(1, popsize * dimension)) + 10
            ),
            "popsize": popsize,
            "mutation": self.config.get("mutation", (0.5, 1.0)),
            "recombination": self.config.get("recombination", 0.7),
            "init": self.config.get("init", "latinhypercube"),
            "polish": bool(self.config.get("polish", False)),
            "updating": self.config.get("updating", "immediate"),
            "workers": 1,
        }
        extra = self.config.get("options")
        if isinstance(extra, dict):
            cfg.update(extra)
        return differential_evolution(recorder, bounds_pairs, seed=seed, **cfg)


@register_optimizer_factory("scipy.l_bfgs_b")
class ScipyLBFGSB(_ScipyAdapter):
    method = "scipy.l_bfgs_b"

    def _minimize(
        self,
        recorder: RecordingObjective,
        x0: np.ndarray,
        bounds_pairs: Any,
        seed: int,
    ):
        cfg = {
            "maxiter": int(self.config.get("maxiter", recorder.inner.budget * 10)),
            "maxfun": int(self.config.get("maxfun", recorder.inner.budget)),
            "ftol": float(self.config.get("ftol", 0.0)),
            "gtol": float(self.config.get("gtol", 0.0)),
        }
        extra = self.config.get("options")
        if isinstance(extra, dict):
            cfg.update(extra)
        return minimize(
            recorder,
            x0,
            method="L-BFGS-B",
            bounds=bounds_pairs,
            options=cfg,
        )


@register_optimizer_factory("scipy.nelder_mead")
class ScipyNelderMead(_ScipyAdapter):
    method = "scipy.nelder_mead"

    def _minimize(
        self,
        recorder: RecordingObjective,
        x0: np.ndarray,
        bounds_pairs: Any,
        seed: int,
    ):
        cfg = {
            "maxiter": int(self.config.get("maxiter", recorder.inner.budget * 10)),
            "maxfev": int(self.config.get("maxfev", recorder.inner.budget)),
            "xatol": float(self.config.get("xatol", 0.0)),
            "fatol": float(self.config.get("fatol", 0.0)),
        }
        extra = self.config.get("options")
        if isinstance(extra, dict):
            cfg.update(extra)
        if "initial_simplex" in self.config:
            cfg["initial_simplex"] = self.config["initial_simplex"]
        return minimize(
            recorder,
            x0,
            method="Nelder-Mead",
            bounds=bounds_pairs,
            options=cfg,
        )


@register_optimizer_factory("scipy.powell")
class ScipyPowell(_ScipyAdapter):
    method = "scipy.powell"

    def _minimize(
        self,
        recorder: RecordingObjective,
        x0: np.ndarray,
        bounds_pairs: Any,
        seed: int,
    ):
        cfg = {
            "maxiter": int(self.config.get("maxiter", recorder.inner.budget * 10)),
            "maxfev": int(self.config.get("maxfev", recorder.inner.budget)),
            "xtol": float(self.config.get("xtol", 0.0)),
            "ftol": float(self.config.get("ftol", 0.0)),
        }
        extra = self.config.get("options")
        if isinstance(extra, dict):
            cfg.update(extra)
        return minimize(
            recorder,
            x0,
            method="Powell",
            bounds=bounds_pairs,
            options=cfg,
        )
