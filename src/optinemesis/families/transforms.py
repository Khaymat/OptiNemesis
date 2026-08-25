"""Shared deterministic transformation helpers for problem families."""

from __future__ import annotations

import numpy as np


def givens_rotation(
    dimension: int,
    strength: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Orthogonal matrix blending identity toward a dense rotation.

    Applies ``k = round(strength * d*(d-1)/2)`` seeded random Givens rotations.
    ``strength == 0`` yields exactly the identity; ``strength == 1`` yields a
    dense product of plane rotations. ``strength`` is interpreted as the
    fraction of principal coordinate planes subjected to rotation; Frobenius
    distance from the identity grows in expectation but is not monotone in
    ``strength`` for a fixed seed.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    q = np.eye(dimension)
    total_pairs = dimension * (dimension - 1) // 2
    k = round(strength * total_pairs)
    if k == 0:
        return q
    pairs = [(i, j) for i in range(dimension) for j in range(i + 1, dimension)]
    indices = generator.permutation(len(pairs))[:k]
    angles = generator.uniform(-np.pi, np.pi, size=k)
    for idx, angle in zip(indices, angles, strict=True):
        i, j = pairs[int(idx)]
        c, s = np.cos(angle), np.sin(angle)
        qi = q[:, i].copy()
        qj = q[:, j].copy()
        q[:, i] = c * qi - s * qj
        q[:, j] = s * qi + c * qj
    return q


def random_unit_vector(dimension: int, generator: np.random.Generator) -> np.ndarray:
    """Uniformly distributed unit vector (Marsaglia's method)."""
    while True:
        v = generator.standard_normal(dimension)
        norm = float(np.linalg.norm(v))
        if norm > 1e-12:
            return v / norm


def shifted_center(
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    radius_fraction: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Center drawn uniformly from a box shrunk by ``radius_fraction`` per side.

    ``radius_fraction`` in [0, 1): each coordinate is confined to
    ``[lo + f*(hi-lo)/2, hi - f*(hi-lo)/2]``, i.e. the fraction refers to the
    half-span, mitigating boundary-bias exploitation while remaining nonempty
    for every valid fraction.
    """
    if not 0.0 <= radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie in [0, 1)")
    lo = np.asarray(bounds_lower, dtype=float)
    hi = np.asarray(bounds_upper, dtype=float)
    margin = 0.5 * radius_fraction * (hi - lo)
    return generator.uniform(lo + margin, hi - margin)


def corner_upper_bound(
    center: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
) -> float:
    """Exact upper bound of ``||x - center||^2`` over the axis-aligned box."""
    lo = np.asarray(bounds_lower, dtype=float)
    hi = np.asarray(bounds_upper, dtype=float)
    reach = np.maximum(np.abs(lo - center), np.abs(hi - center))
    return float(np.dot(reach, reach))
