"""Deterministic keystone fixtures.

This module provides *test-only* problem families and optimizers whose
ranking reversals are structural, not stochastic, exercising the full
pipeline deterministically. They are registered on import by
tests/conftest.py and are not part of the public API.

Families
--------
- ``KeystoneBasins``: deep-narrow well vs shallow decoys (exploration vs
  exploitation trade-off; kept for unit tests).
- ``KeystoneFlip``: synthetic two-regime flip family whose parameter
  ``flip`` in [0, 1] selects between a smooth convex regime (Ellipsoidal,
  L-BFGS wins) and a rugged multimodal regime (Rastrigin-like, random
  search wins). By placing the flip threshold at 0.3, the rugged regime
  dominates the neutral-box median, so a validated reversal is guaranteed
  for the pair (random_search vs L-BFGS-B).

Optimizers
----------
- ``keystone.local_descent`` / ``keystone.scan_then_descend``: compass-based
  baselines for basin tests.
- ``keystone.coordinate_descent``: exact 3-point line-search cyclic
  descent, generic over any box-bounded function.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from optinemesis.adapters.protocol import register_optimizer_factory
from optinemesis.adapters.recorder import RecordingObjective
from optinemesis.core.bounds import Bounds
from optinemesis.core.contract import ParamRange
from optinemesis.core.objective import CountingObjective
from optinemesis.core.results import RunResult
from optinemesis.families.base import FamilyBuild, ProblemFamily
from optinemesis.families.registry import register_family
from optinemesis.families.transforms import random_unit_vector

_DECOY_COUNT = 3


@register_family
class KeystoneFlipFamily(ProblemFamily):
    """Synthetic flip family: smooth vs rugged regimes.

    - flip < 0.3  => Ellipsoidal (conditioning 200, rotation 0) — L-BFGS wins.
    - flip >= 0.3 => MultimodalBlend-like Rastrigin (modality 1.2) — random wins.

    The threshold is deliberately off-center so the rugged regime dominates the
    neutral median, guaranteeing the reference direction opposes the smooth-
    regime nominated candidates.
    """

    name: ClassVar[str] = "KeystoneFlip"
    version: ClassVar[str] = "1"
    parameter_box: ClassVar[tuple[ParamRange, ...]] = (
        ParamRange("flip", 0.0, 1.0),
    )
    transform_tags: ClassVar[tuple[str, ...]] = ("ruggedness",)

    def _build(
        self,
        theta: tuple[float, ...],
        dimension: int,
        bounds: Bounds,
        generator: np.random.Generator,
    ) -> FamilyBuild:
        (flip,) = theta
        lo, hi = bounds.as_arrays()
        if float(flip) < 0.3:
            center = (lo + hi) / 2.0
            eigenvalues = np.geomspace(1.0, 200.0, dimension)

            def _eval_smooth(x: np.ndarray) -> float:
                u = x - center
                return float(np.dot(eigenvalues * u, u))

            scale = float(200.0 * float(np.dot(hi - lo, hi - lo)))
            return FamilyBuild(
                evaluator=_eval_smooth,
                optimum_value=0.0,
                regret_scale=scale,
                tags={"flip": float(flip), "regime": 0.0},
                spec={"regime": "smooth", "flip": float(flip)},
                optimum_x=center.copy(),
            )

        center = (lo + hi) / 2.0
        amplitude = 10.0
        modality = 1.2

        def _eval_rugged(x: np.ndarray) -> float:
            v = x - center
            return float(
                np.sum(v * v - amplitude * np.cos(2.0 * np.pi * modality * v) + amplitude)
            )

        scale = float(np.dot(hi - lo, hi - lo) + 2.0 * dimension * amplitude)
        return FamilyBuild(
            evaluator=_eval_rugged,
            optimum_value=0.0,
            regret_scale=scale,
            tags={"flip": float(flip), "regime": 1.0},
            spec={"regime": "rugged", "flip": float(flip)},
            optimum_x=center.copy(),
        )


@register_family
class KeystoneBasinsFamily(ProblemFamily):
    """Deep-narrow global well + shallow-wide decoys + gentle tilt."""

    name: ClassVar[str] = "KeystoneBasins"
    version: ClassVar[str] = "1"
    parameter_box: ClassVar[tuple[ParamRange, ...]] = (
        ParamRange("trap_depth_ratio", 5.0, 40.0),
        ParamRange("deep_radius_fraction", 0.30, 0.95),
        ParamRange("curvature_multiplier", 1.0, 20.0),
    )
    transform_tags: ClassVar[tuple[str, ...]] = ("modality", "deception", "shift")

    def _build(
        self,
        theta: tuple[float, ...],
        dimension: int,
        bounds: Bounds,
        generator: np.random.Generator,
    ) -> FamilyBuild:
        depth_ratio, deep_radius_fraction, curvature_multiplier = theta
        lo, hi = bounds.as_arrays()
        span_min = float(np.min(hi - lo))
        center_global = (lo + hi) / 2.0

        deep_center = center_global + 0.25 * span_min * random_unit_vector(
            dimension, generator
        )
        deep_center = np.clip(deep_center, lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
        deep_radius_sq = (
            0.25 * deep_radius_fraction**2 * span_min**2 / curvature_multiplier
        )

        decoy_centers = []
        for _ in range(_DECOY_COUNT):
            direction = random_unit_vector(dimension, generator)
            c = deep_center + 0.5 * span_min * direction
            decoy_centers.append(np.clip(c, lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
        decoy_radius_sq = 4.0 * deep_radius_sq

        def evaluate(x: np.ndarray) -> float:
            value = 0.0
            d_deep = x - deep_center
            value += -depth_ratio * max(0.0, 1.0 - float(np.dot(d_deep, d_deep)) / deep_radius_sq)
            for c in decoy_centers:
                d = x - c
                value += -1.0 * max(0.0, 1.0 - float(np.dot(d, d)) / decoy_radius_sq)
            return value

        optimum_value = -float(depth_ratio)
        regret_scale = float(depth_ratio + 1.0)

        return FamilyBuild(
            evaluator=evaluate,
            optimum_value=optimum_value,
            regret_scale=regret_scale,
            tags={
                "trap_depth_ratio": depth_ratio,
                "deep_volume_share": float(deep_radius_fraction**dimension),
                "curvature_multiplier": curvature_multiplier,
                "modality": float(_DECOY_COUNT + 1),
                "conditioning": 1.0,
                "separability": 1.0,
            },
            spec={
                "deep_center": deep_center.tolist(),
                "deep_radius_sq": float(deep_radius_sq),
                "decoy_centers": [c.tolist() for c in decoy_centers],
                "decoy_radius_sq": float(decoy_radius_sq),
            },
            optimum_x=deep_center.copy(),
        )


def _seeded_start(instance: Any, seed: int) -> tuple[np.ndarray, np.random.Generator]:
    lo, hi = instance.bounds.as_arrays()
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi), rng


def _finish(
    implementation: str,
    recorder: RecordingObjective,
    started: float,
    seed: int,
    extra_metadata: dict[str, Any] | None = None,
) -> RunResult:
    metadata: dict[str, Any] = {"seed": seed, "seeded": True}
    if extra_metadata:
        metadata.update(extra_metadata)
    return RunResult(
        optimizer_name="",
        implementation=implementation,
        best_x=recorder.best_x,
        best_f=recorder.best_f,
        n_evals_consumed=recorder.inner.consumed,
        overshoot_events=recorder.overshoot_events,
        non_finite_evals=recorder.inner.non_finite,
        runtime_s=time.perf_counter() - started,
        termination_reason=(
            "budget_exhausted" if recorder.inner.remaining <= 0 else "stalled"
        ),
        compliance_flags=("non_finite_evals",) if recorder.inner.non_finite else (),
        history=tuple(recorder.history),
        metadata=metadata,
    )


def _vertex_offset(f_plus: float, f_minus: float, f_zero: float, step: float) -> float:
    denominator = f_plus + f_minus - 2.0 * f_zero
    if denominator <= 0.0:
        return 0.0
    return step * (f_minus - f_plus) / (2.0 * denominator)


def _compass_descend(
    recorder: RecordingObjective,
    x: np.ndarray,
    f_current: float,
    step: float,
    lo: np.ndarray,
    hi: np.ndarray,
    min_step: float,
) -> tuple[np.ndarray, float, float]:
    """One compass-search probe cycle; returns updated (x, f, step)."""
    dimension = x.size
    improved = False
    for i in range(dimension):
        if recorder.inner.remaining < 2:
            break
        for sign in (1.0, -1.0):
            candidate = x.copy()
            candidate[i] = float(np.clip(candidate[i] + sign * step, lo[i], hi[i]))
            if candidate[i] == x[i]:
                continue
            f_new = recorder(candidate)
            if f_new < f_current:
                x, f_current, improved = candidate, f_new, True
                break
    if not improved and recorder.inner.remaining >= 1:
        step *= 0.5
        if step < min_step:
            step = 0.0
    return x, f_current, step


@register_optimizer_factory("keystone.local_descent")
class KeystoneLocalDescent:
    """Single-start compass descent: exploitation without exploration."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        del config

    def run(self, objective: CountingObjective, seed: int) -> RunResult:
        started = time.perf_counter()
        recorder = RecordingObjective(objective)
        instance = objective.instance
        lo, hi = instance.bounds.as_arrays()
        x, _rng = _seeded_start(instance, seed)
        f_current = recorder(x)
        min_step = 1e-5 * float(np.min(hi - lo))
        step = 0.125 * float(np.min(hi - lo))

        while objective.remaining >= 2 and step > 0.0:
            x, f_current, step = _compass_descend(
                recorder, x, f_current, step, lo, hi, min_step
            )
        return _finish("keystone.local_descent", recorder, started, seed)


@register_optimizer_factory("keystone.scan_then_descend")
class KeystoneScanThenDescend:
    """Uniform scan for SCAN_SHARE of budget, then descend from the best
    three distinct scan points with the remaining budget split across them."""

    SCAN_SHARE = 0.6
    DESCENT_STARTS = 3

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        del config

    def run(self, objective: CountingObjective, seed: int) -> RunResult:
        started = time.perf_counter()
        recorder = RecordingObjective(objective)
        instance = objective.instance
        lo, hi = instance.bounds.as_arrays()
        rng = np.random.default_rng(seed)
        budget = objective.budget
        scan_evals = int(self.SCAN_SHARE * budget)

        scanned: list[tuple[float, np.ndarray]] = []
        while objective.consumed < scan_evals and objective.remaining >= 1:
            point = rng.uniform(lo, hi)
            value = recorder(point)
            scanned.append((value, point))

        starts: list[np.ndarray] = [
            point for _, point in sorted(scanned, key=lambda pair: pair[0])
        ][: self.DESCENT_STARTS]
        if not starts:
            return _finish(
                "keystone.scan_then_descend",
                recorder,
                started,
                seed,
                extra_metadata={"scan_evals": scan_evals},
            )
        per_start = max(objective.remaining // len(starts), 0)
        step = 0.125 * float(np.min(hi - lo))
        min_step = 1e-5 * float(np.min(hi - lo))
        best_x = starts[0].copy()
        assert recorder.best_f is not None
        best_f = float(recorder.best_f)
        for index, start in enumerate(starts):
            allowance_end = objective.consumed + per_start + (
                1 if index < objective.remaining % len(starts) else 0
            )
            x = start.copy()
            f_current = float("inf")
            probe = recorder(x)
            if probe < f_current:
                x, f_current = start.copy(), probe
            while (
                objective.remaining > 0
                and objective.consumed < allowance_end
                and step > 0.0
            ):
                x, f_current, step = _compass_descend(
                    recorder, x, f_current, step, lo, hi, min_step
                )
            if f_current < best_f:
                best_x, best_f = x, f_current
            if objective.remaining <= 0:
                break
        del best_x, best_f
        return _finish(
            "keystone.scan_then_descend",
            recorder,
            started,
            seed,
            extra_metadata={"scan_evals": scan_evals, "descents": len(starts)},
        )


def register_keystone_fixtures() -> None:
    """Idempotent registration entry point used by conftest."""
    KeystoneLocalDescent({})
    KeystoneScanThenDescend({})
    KeystoneCoordinateDescent({})


@register_optimizer_factory("keystone.coordinate_descent")
class KeystoneCoordinateDescent:
    """Cyclic coordinate descent with exact 3-point quadratic line searches.

    On smooth, near-unimodal landscapes it polishes to the optimization
    floor; on coarsely rugged ones it terminates in the nearest ripple
    basin. Its refinement depth scales with its evaluation budget.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.rel_step = float(config.get("rel_step", 1e-2))
        if not 0.0 < self.rel_step < 1.0:
            raise ValueError("rel_step must lie in (0, 1)")

    def run(self, objective: CountingObjective, seed: int) -> RunResult:
        started = time.perf_counter()
        recorder = RecordingObjective(objective)
        instance = objective.instance
        dimension = instance.dimension
        lo, hi = instance.bounds.as_arrays()
        spans = hi - lo
        x, _rng = _seeded_start(instance, seed)
        f_current = recorder(x)
        min_step = 1e-6 * float(np.min(spans))

        while objective.remaining >= 3:
            moved = False
            for i in range(dimension):
                if objective.remaining < 3:
                    break
                step_i = self.rel_step * spans[i]
                e_i = np.zeros(dimension)
                e_i[i] = step_i
                f_plus = recorder(x + e_i)
                f_minus = recorder(x - e_i)
                offset = _vertex_offset(f_plus, f_minus, f_current, step_i)
                if abs(offset) < 1e-18:
                    continue
                candidate = x.copy()
                candidate[i] += offset
                f_new = recorder(candidate)
                if f_new < f_current:
                    x, f_current, moved = candidate, f_new, True
            if not moved:
                scale_down = 0.5 * self.rel_step * float(np.min(spans))
                if scale_down < min_step:
                    break
                self.rel_step *= 0.5
        return _finish("keystone.coordinate_descent", recorder, started, seed)
