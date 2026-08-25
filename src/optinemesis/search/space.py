"""Deterministic theta-space samplers: uniform random and scrambled LHS."""

from __future__ import annotations

import numpy as np

from optinemesis.core.contract import FamilyRef


def _scale(unit_column: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return lower + unit_column * (upper - lower)


def sample_thetas_random(
    family_ref: FamilyRef,
    n_candidates: int,
    *,
    root_seed: int,
) -> list[tuple[float, ...]]:
    """Uniform i.i.d. sampling from the family parameter box."""
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    rng = np.random.default_rng(root_seed)
    box = family_ref.parameter_box
    draws = rng.uniform(size=(n_candidates, len(box)))
    thetas: list[tuple[float, ...]] = []
    for row in draws:
        theta = tuple(
            float(_scale(row[j], p.lower, p.upper)) for j, p in enumerate(box)
        )
        thetas.append(family_ref.check_theta(theta))
    return thetas


def sample_thetas_lhs(
    family_ref: FamilyRef,
    n_candidates: int,
    *,
    root_seed: int,
) -> list[tuple[float, ...]]:
    """Scrambled Latin Hypercube sampling from the family parameter box.

    Each dimension is independently stratified into ``n_candidates`` equal
    intervals, permuted once per dimension, and jittered uniformly within its
    interval. One-dimensional marginals are therefore well spread while points
    remain randomized.
    """
    if n_candidates < 2:
        raise ValueError("lhs requires n_candidates >= 2")
    rng = np.random.default_rng(root_seed)
    box = family_ref.parameter_box
    k = len(box)
    unit = np.empty((n_candidates, k), dtype=float)
    for j in range(k):
        perm = rng.permutation(n_candidates)
        jitter = rng.uniform(size=n_candidates)
        unit[:, j] = (perm + jitter) / n_candidates
    thetas = [
        tuple(
            float(_scale(unit[i, j], p.lower, p.upper)) for j, p in enumerate(box)
        )
        for i in range(n_candidates)
    ]
    return [family_ref.check_theta(t) for t in thetas]
