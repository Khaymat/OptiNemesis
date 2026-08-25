"""Effect sizes and robust location statistics.

Sign conventions follow docs/DESIGN.md §2: effects are computed on regret
samples ``a`` (configuration A) and ``b`` (configuration B); positive values
mean configuration A shows larger regret than B.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def median(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("median requires non-empty sample")
    return float(np.median(arr))


def iqr(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("iqr requires non-empty sample")
    q25, q75 = np.percentile(arr, [25, 75])
    return float(q75 - q25)


def paired_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size in [-1, 1].

    Computed as the sign-weighted mean rank of the absolute paired
    differences; ties contribute zero. ``delta > 0`` means ``a`` tends to
    larger regret than ``b``.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("paired inputs must be non-empty 1-D arrays of equal length")
    diff = x - y
    nonzero = diff != 0.0
    if not np.any(nonzero):
        return 0.0
    magnitudes = np.abs(diff[nonzero])
    ranks = rankdata(magnitudes)
    signs = np.sign(diff[nonzero])
    return float(np.sum(signs * ranks) / np.sum(ranks))


def probability_superiority(a: np.ndarray, b: np.ndarray, paired: bool = True) -> float:
    """P(regret_A > regret_B) with ties counted as one half."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if paired:
        if x.shape != y.shape or x.size == 0:
            raise ValueError("paired inputs must be equal-length non-empty arrays")
        greater = float(np.mean(x > y))
        ties = float(np.mean(x == y))
        return greater + 0.5 * ties
    if x.size == 0 or y.size == 0:
        raise ValueError("inputs must be non-empty")
    total = 0.0
    count = 0
    for xi in x:
        total += float(np.sum(xi > y)) + 0.5 * float(np.sum(xi == y))
        count += y.size
    return total / count


def cliffs_delta_from_samples(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta for independent samples: 2 * P(A > B) - 1."""
    return 2.0 * probability_superiority(a, b, paired=False) - 1.0


def median_gap(a: np.ndarray, b: np.ndarray) -> float:
    """median(a) - median(b); positive means A has larger median regret."""
    return median(a) - median(b)


def summarize_regrets(values: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("empty regret sample")
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "median": float(q50),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
    }
